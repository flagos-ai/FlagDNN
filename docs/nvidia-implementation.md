# NVIDIA 实现与开发导航

本文描述 NVIDIA 生产代码的职责边界；公共 API 与跨平台约束见
[架构边界](architecture.md)，验证入口见[功能与性能测试](testing.md)，算子类型与输入语义见[算子支持契约](operator-support.md)。

## 编译器

外部协议只要求 `compiler.py` 的 `compiler_identity()` 和
`compile_request()`，入口保留在 NVIDIA 根目录，负责请求协议与整体编排。
`dispatch/` 按 operation、shape、dtype、stride、对齐和设备架构进行选核、调优候选
及执行计划构造；`codegen/` 消费计划，完成 Triton 编译、产物生成和缓存指纹。
`codegen/` 可以依赖 `dispatch/` 的计划及公共契约，`dispatch/` 不依赖
`codegen/` 或入口模块，内部模块也不反向导入 `compiler.py`。

| 职责 | NVIDIA 文件 |
| --- | --- |
| 请求校验、阶段依赖、最终 manifest | `compiler.py` |
| tensor 表、运行时参数 ABI | `dispatch/tensor.py` |
| 公共校验、`ExecutionGroup`、`KernelPlan` | `dispatch/common.py` |
| 图融合、流水线分发、活跃 workspace | `dispatch/graph.py` |
| 卷积算法选择与阶段展开 | `dispatch/pipeline_fprop.py`、`dispatch/pipeline_dgrad.py`、`dispatch/pipeline_wgrad.py` |
| 大尺寸 FP32 MatMul 的布局规划与打包/TMA 契约 | `dispatch/pipeline_matmul.py`、`dispatch/matmul_tf32.py` |
| 单 kernel 的合法输入和 launch 配置 | `dispatch/conv_*.py`、`dispatch/matmul.py`、`dispatch/layout.py`、`dispatch/normalization.py` |
| Attention 配置与阶段展开 | `dispatch/attention_*.py` |
| operation 到 kernel 配置的分发 | `dispatch/selection.py` |
| 调优表校验、候选变体和 grid | `dispatch/tuning.py` |
| Triton 编译、JIT 描述和 artifact 生成 | `codegen/emit.py`、`codegen/io.py` |
| 每个变体的 host TensorMap ABI、artifact 内编译缓存 | `codegen/tensor_map.py`、`codegen/jit_cache.py` |
| 每次 launch 的 scratch 与顺序执行阶段的资源规划 | `codegen/resources.py` |
| 编译器、kernel、registry、tuning 等依赖指纹 | `codegen/identity.py` |

数据顺序为：校验请求 → 图融合 → 展开多 kernel 流水线 → 分配活跃
tensor workspace → 选择并生成各阶段 kernel → 追加编译器 scratch → 输出执行程序。算子配置的
`KernelPlan` 包含函数名、signature、constexpr、grid 和参数布局。
`ExecutionGroup` 明确记录 source nodes、输入输出 UID 和 tensor 元数据。

卷积的专用 shape 条件按优先级留在对应的 pipeline 模块。调整顺序可能改变算法，
不能当作纯重构。新增阶段必须同时写出输入/输出 UID、tensor dtype 和参数；
workspace 在最终阶段展开后计算，只跳过完全未引用的虚拟 tensor，
不复用仍然活跃的 tensor 内存。

Triton 的 `global_scratch_size` 是每 CTA 的字节数，必须乘完整三维 launch grid，
不能使用固定 4096 字节代替。`codegen/resources.py` 将它对齐到 256 字节，
顺序执行的 stages/variants 共享一个与所有 graph tensor 分离的 workspace 后缀。
JIT 在 prepare/autotune 之前按相同 AST、指针对齐提示和编译参数读取实际资源，
之后 libtriton_jit 复用同一份 Triton 缓存。没有 TMA scratch 的 JIT kernel 暂保留
原有 4096 字节最小值；执行与 autotune 都传入各自 launch 的 scratch 指针。
manifest 校验拒绝越界、重叠或不支持的 profile scratch。

`compiler.py` 以及 `dispatch/`、`codegen/` 下的全部 Python 模块递归参与安装
及编译器 identity，包括包的 `__init__.py`；`__pycache__` 不参与。模块指纹标签
使用相对 NVIDIA 根目录的路径，避免同名模块冲突，并使源码与安装布局的指纹一致。
新增模块无需保留 `compiler_` 前缀。移动模块、kernel 或调优配置后应验证 identity
和安装后的执行链。图 tensor workspace 由 `dispatch/graph.py` 规划；
依赖 Triton 编译结果的 warp 数量和 scratch 由 `codegen/resources.py` 处理。

## 执行与构建

NVIDIA 只支持 `libtriton_jit`。CMake、编译器入口和 manifest 解析均显式拒绝
其他执行引擎；其他后端的引擎选项不受影响。`ExecutionEngine` 使用私有实现封装
JIT 准备、调优和已准备好的 CUDA launch，不再维护外部 cubin 执行分支。
执行与 autotune 共享 `cuda_launch.hpp` 的参数和临时分配工具：

- `KernelArguments` 使用每次调用独占的栈空间装配参数，只初始化实际使用的槽位。
  tensor binding 按 UID 查找，不依赖调用者的排列顺序。
- CUDA module 由 libtriton_jit 管理；后端保存准备好的函数与 launch 元数据，
  稳态执行不调用 Python，也不重新编译或构造 JIT signature。
- engine 持有 context 的时间必须覆盖 module 的生命周期。默认流、
  legacy/per-thread 默认流执行时临时恢复 executable 的 context；
  显式 stream 保持原来的快速路径。

公共 C/C++ API 不变。私有插件可选导出 `flagdnnBackendGetBuildApiV1`：

1. Core 在独占进程环境锁下调用环境准备函数，并传入执行引擎、context 和实际请求；
   失败可以重试。
2. 准备成功后，`is_environment_prepared()` 对该执行引擎必须线程安全且一直返回真。
3. 后续 executable 构建持共享环境锁，不得修改进程环境。
4. 未提供该扩展的旧插件继续在独占锁下构建。

当前 libtriton_jit 没有独立的环境初始化 API；仅配置 `PYTHONPATH` 不够，因为它还
会通过 Python 写入 `TRITON_JIT_BACKEND` 并延迟初始化 CUDA。因此 NVIDIA 在独占
阶段预备首个请求中的真实 variant，完成这些副作用，再进入其余 JIT/autotune 工作。
缓存路径同样必须完成 Python 初始化，不能仅加载 CUDA 缓存就宣称环境准备完成。
当前依赖的静态签名提取仅接受普通 TL 函数，因此编译器为缓存模块附加一个普通 TL
签名入口，供公开 JIT 构造接口完成初始化；该入口不编译、不发射。真正的 CUDA
预热仍使用实际 variant，没有额外的 GPU 启动 kernel。非法引擎请求在准备之前拒绝。
JIT 自身仍保留必要的内部互斥，这不意味着依赖库内部编译已经可以任意并行。
应用线程也不能绕过协议并发调用 `setenv/unsetenv`。

启动 JIT 时，由 NVIDIA 后端自己保持插件与依赖级缓存到进程结束，core 不需要
硬编码 NVIDIA 的生命周期策略；旧 core 走构造函数准备路径时也有同样保障。
此后不能在同一进程内热替换该插件。
context/executable 仍按 RAII 释放自己持有的资源引用。若由后端首次创建 Python
解释器，会释放初始 GIL，允许后续线程构建，并在退出析构 Python 依赖对象前取得 GIL。
后端不释放 Python 调用者原先持有的 GIL，也不主动 finalize 外部解释器。

## 精度与缓存

FP32 卷积内部展开、打包和 GEMM 的操作数保持 FP32，不能为了 Tensor Core
吞吐量悄悄增加 FP16 中间 tensor。原先窄化会使大值溢出、小值下溢，并破坏相消。
修复这种错误可能增加 workspace 或执行时间，应与相同精度要求的实现比较。

正确性和性能均以实际 cuDNN 路径为基准，不把所有 FP32 算子一律切换为 TF32。
当前已验证的 rank ≤ 3 MatMul 在 K/N 四元素对齐时使用 TF32 最近偶数舍入和
FP32 累加；不对齐或更高 rank 的通用路径保留 `tf32x3`。直接空间卷积仍保留
`tf32x3`，大型 im2col/P5 GEMM 使用 MatMul 的显式精度策略。

Hopper 的普通 Triton `input_precision="tf32"` 本身不能替代操作数舍入，kernel
显式执行 TF32 RNE。strided GEMM 在 SM90+ 使用 `cvt.rn.tf32.f32`，SM80 保留
整数舍入回退。NVIDIA 功能和性能只与实际 cuDNN 输出比较，不运行 CPU oracle。

IEEE 和 TF32 比较分进程运行。IEEE 进程启动前设置 `NVIDIA_TF32_OVERRIDE=0`，
因为仅筛选 cuDNN numeric notes 不足以禁止隐式 TF32。TF32 组选择 Tensor Core
计划，并核对舍入方式。cuDNN 9.24 的部分 Tensor Core 引擎对中点采用远离零
舍入，wgrad engine 68 则截断；numeric notes 不足以表达这些差别。当前验证栈
使用 matmul 的最近偶数引擎、fprop 67、dgrad 71/76/78 和 wgrad 70，并对每个
配置实际执行验证。这些筛选属于 validation adapter，不进入生产 dispatcher。
范围、相消和尾块检查均使用对应精度的实际 cuDNN 对照。升级 cuDNN 或设备后，
需要重新验证执行计划和精度策略。

### FP32 前向卷积的性能策略

布局/算法选择由 `dispatch/pipeline_fprop.py` 负责；kernel 不判断模型名称。
当前等精度优化在 H100 上验证，不承诺其他 GPU 上相同的收益。

| 路径 | 实现和工作区 |
| --- | --- |
| 三通道、大空间的 3×3 stride2 FP32 stem | 27 项归约放入一个 dot tile，复用 `conv2d_spatial_nchw_kernel`，不分配 im2col 工作区 |
| P5 及大归约 FP32 im2col | 直接生成 K 连续的 patches，逻辑维度仍是 `[batch, K, output_hw]`，显式步长为 `[K*output_hw, 1, K]`；不添加独立转置阶段，也不增加存储量 |
| FP16/BF16、较短归约与非对称路径 | 保持原算法和布局；小型 1x1/3D 扩展 tile 候选，不改变精度策略 |

stem 的规划层 workspace 为零；libtriton_jit 执行仍保留原有 4096 字节全局
scratch，因此 JIT 原生基准报告的总 workspace 为 4096 字节，而不是零。

P5 仍采用四路 split-K 和 FP32 归约；GEMM 显式接收 B 的两个矩阵步长。
新布局复用通用 im2col 参数校验，新的二维 materialization kernel 只做搬运，
GEMM 仍使用 FP32 操作数，乘法精度按上述 cuDNN 对齐策略选择。相关配置分别放在
`conv_fprop_im2col_transposed`、`conv_fprop_fp32_mm` 和
`conv_fprop_small_reduction_fp32` 三个调优表中；独立 MatMul 的布局规划见下节。
修改布局必须同步验证 workspace 元数据、grid、尾块、数值范围和安装后的编译链。

### MatMul、DGrad 与布局优化

SM90+ 上，批量数至少为 4、M/N/K 均为 512、输入/输出连续且至少 16 字节对齐的
FP32 MatMul 优先运行 `matmul_tf32_tma_direct_kernel`。它以 `Cᵀ = Bᵀ × Aᵀ`
计算，让原始 K 连续的 A 成为 MMA 右操作数，省去全局 B 打包阶段；两输入在 tile
内显式执行 TF32 RNE，结果仍按原始 FP32 布局写回。规划只需真实 JIT scratch，
不再分配 B 工作区。`matmul_tf32_tma_direct` 调优表独立描述网格，设备能力和布局
条件由 pipeline 与 kernel 选择共同调用同一 helper，SM80 和不满足条件者不启用。

SM90 上更大的整 tile FP32 MatMul 可直接使用两个 host TensorMap，省去 B 打包。
条件集中在 `_tf32_tensor_map_eligible()`：batch≥4、M/N≥512、M 整除 256、
N 整除 128、K 整除 32，输入连续、非虚拟且至少 16 字节对齐；还要求 K≥1024，
或者 K=512 且 M/N≥1024。SM80、SM100、虚拟输入与尾块均保留原路径。

- K≥1024 使用普通 TL `matmul_tf32_tensor_map_kernel`，调优表为
  `matmul_tf32_tensor_map`。
- K=512 使用独立文件 `kernels/matmul_tf32_short.py` 中的 Gluon 内核。一个 TMA
  生产者交错供给两个计算组，相邻 N tile 共享 A；三/四个输入槽分别等两个消费者
  释放后才能复用。每轮等待 WGMMA 完成后才释放寄存器和输入槽，输出直接写回。
  调优表 `matmul_tf32_short` 的 4 warps 是生产者数量，manifest 的 12 warps
  是编译后真实 CTA 数量；两者不能混用。

两种内核都由既有 `libtriton_jit` 缓存 API 加载和预备，稳态仍使用同一准备好的
CUDA launch。TensorMap 的 TF32-RNE 数据类型只控制读取舍入，不改变 FP32 存储；
描述符在每次调用时从真实绑定地址重新编码，不能在 executable 内缓存调用者地址。
元数据按变体独立解析，检查 tile、逻辑 shape/stride、对齐、容量、编译布局和 scratch。
缓存 JSON/cubin 属于 artifact，检查相对路径、非符号链接、文件大小和 SHA-256。
这些缓存不是第二个执行引擎，也不允许任意 cubin 绕过 JIT 执行边界。

其余批量数至少为 4、M/N/K 均至少为 512 的连续 FP32 MatMul 使用 K 连续的 B 工作区。
K/N 四元素对齐时，按 batch 转置并完成 TF32 RNE，逻辑维度仍为 `[batch,K,N]`，
strides 为 `[K*N,1,K]`。SM90+ 且 M/K 满足候选 tile 边界、指针对齐时运行
`matmul_tf32_tma_kernel`：B 无需在归约循环中重复舍入，A 在加载后执行 RNE。
不满足 TMA 约束时回退到 strided GEMM。打包和计算的契约集中在
`dispatch/matmul_tf32.py`，调优表分别为 `matmul_tf32_pack` / `matmul_tf32_tma`。

K/N 不对齐的高精度路径保留整批转置 `[batch*K,N]`、strides `[K,1,batch*K]`
和 `matmul_fp32_k_contiguous` 调优表；其打包只做 FP32 复制。两种布局的 tensor
workspace 都只增加 B 的 FP32 存储量，TMA scratch 另外按真实编译资源规划。
性能计时包含每次执行的全部打包工作。小矩阵、广播、原有非连续布局和低精度
MatMul 不切换到这些打包路径。

Hopper 上，满足 tile 边界和对齐约束、batch 不超过 32 的低精度 512³ MatMul 使用
`matmul_batched_tma_short_kernel`：两个输入 TMA descriptor、短 K 循环展开、普通
masked epilogue 和 streaming output store。更大形状保留原 persistent kernel；
batch 小于 8 时默认使用 64×128 的小 tile，并将它加入短 K autotune 候选，
避免少量大 tile 无法占满设备；超过 32 的 batch 保留原 persistent 调度。
不满足 M/K tile 边界、N stride 或指针对齐条件时走带掩码的指针路径，避免跨 batch
读取或写入。短 K 调优独立使用 `matmul_short_persistent`，累加仍为 FP32。

P5X 的 `[1,768,40,40]` stride-2 DGrad 将 loss 和 weight 的归约通道维置为
物理连续布局，复用通用二维 tiled transpose，再运行四个原有 parity 计算阶段。
它额外使用一份 `[400,768]` loss 缓冲区，保持 FP32/FP16/BF16 dtype 和原有 dot
精度不变。其他 DGrad 形状和三维卷积路径不切换该布局。

通用二维 transpose 的大尺寸连续矩阵路径使用 `matrix_transpose_kernel`；
grid 随 tile 大小重新计算，两个轴均有尾块掩码；索引乘法使用 64 位，超过二维
grid 上限的形状保留原有一维 materialization 路径。LayerNorm 的静态行数提示仅在
行组整除时消除行掩码，尾部行组保留检查；原有调用者不传提示时行为不变。
LayerNorm 的可选 paired reduction 同时归约 sum/square，并提前读取 affine 参数；
仍以 FP32 计算原有 E[x²]−E[x]² 公式。流式多轮归约与 RMSNorm 不受影响。

低精度 `[8,64,28,28]`、`[128,64,1,1]` WGrad 的短归约路径使用普通指针加载，
每 batch 分四段，编译时固定循环次数并掩码最后一个 K tile。它避免每 CTA 构造 TMA
descriptor，部分结果仍用原有低精度 dtype、最终归约仍用 FP32；FP32 WGrad 与其他
卷积形状保持原算法。候选位于 `conv_wgrad_short_split`，阶段规划仍由 WGrad 模块负责。

### Autotune 缓存

autotune 的唯一可运行候选也会写入 selection cache，但无需运行计时回调。
多个候选仍走原有计时策略。缓存命中仍要求身份校验和 selected candidate
在当前候选集合中；不强制候选集合完全相等，允许恢复到完整候选空间后复用合法选择。

## 如何修改一个算子

1. 在对应配置模块描述合法 dtype、shape、strides 和单 kernel ABI。
2. 多 kernel 算法在对应 pipeline 模块展开，显式说明中间 tensor 精度与依赖。
3. 在 `dispatch/selection.py` 接入配置；按需更新 kernel registry 与 tuning 表。
4. 在 `validation/compiler_contract.py` 增加主机端计划回归，在 C++ functional
   case 或 execution contract 中验证真实设备结果。
5. 通过统一 runner 的功能、性能、preflight 以及安装后的 consumer 验证；
   execution contract 还覆盖两个工作线程首次构建、线程退出后由主线程销毁
   最后一个 handle、由新线程重新构建、回到主线程执行及进程退出清理。
   不把 reference 不支持当作 FlagDNN 执行通过。

## 性能记录

NVIDIA benchmark 输出 schema v3；通用 schema 继续接受其他后端的 v1/v2。

| 字段 | 含义 |
| --- | --- |
| 顶层 `median/p90/samples` | CUDA Graph 批量重放的 GPU event 耗时，按执行次数归一化，单位微秒 |
| `host_submit_us` | 直接调用 executable 的主机提交耗时，不是 CUDA Graph 重放；包含 binding/参数装配/驱动提交 |
| `build_us` | 当前 case 第一次 provider.build 的主机墙钟耗时，单位微秒 |
| `warm_build_us` | 同一进程中立即再次 provider.build 的耗时，不含该临时 executable 的销毁 |
| `workspace_bytes` | 首次构建 executable 要求的 workspace 字节数 |
| `build_cache` | `fresh_artifact_cache`、`reuse_allowed` 或 reference 的 `provider_managed` |

host 样本开始前与结束后同步，显式等待不计入提交区间；驱动队列反压仍可能体现在
提交耗时中。因此它不能直接解释为纯 CPU 指令耗时，也不能代替端到端 latency。

`FLAGDNN_BENCHMARK_FRESH_CACHE=1` 为每个 benchmark 进程创建新的 FlagDNN
artifact/selection cache 子目录，结束后仅删除该临时子目录。它**不清空**
Triton 缓存，不重启进程，也不代表整个系统完全冷启动。同一进程后续 case
可以复用先前加载的依赖，因此 `build_us` 不是每个 case 的完整进程启动成本。

全量复现入口：

```bash
FLAGDNN_BENCHMARK_FRESH_CACHE=1 python3 tools/run_tests.py \
  --build-dir /path/to/release-build --platform nvidia --ops all \
  --suites functional,benchmark --preflight --timeout 3600 \
  --output /path/to/nvidia-results.json
```

NVIDIA runner 默认预算和原生 benchmark 的单算子预算均为 3600 秒。
该预算包含编译、完整候选调优、正确性检查及采样，并非单次 kernel 耗时。
`--timeout` 约束外层任务，不能放宽 CTest 测试自身更短的超时属性。

添加 `--min-speedup 0.9` 可按每个可比较用例进行性能门禁。`performance` 中的
`gate_scope=comparable_cases_only` 表示只对同时有 FlagDNN/cuDNN GPU 时间的用例
做比值比较；`comparable_coverage` 保留无参考原因，不能把 N/A 当作达标。
该比值不包含 host submission、构建时间或 workspace，不允许用平均值抵消慢用例。

当前 H100 原生 Nsight Systems trace 确认大型 FP32 MatMul 使用
`sm90_xmma_gemm_f32f32_tf32f32_f32_nn_...`，大型 stride-2 FProp 使用
`sm90_xmma_fprop_implicit_gemm_f32f32_tf32f32_f32_...`。精度策略已按用户要求
对齐参考，不再把默认 `tf32x3` 作为所有 FP32 用例的统一验收前提。
性能优化仍不得通过修改 reference、放宽精度容差或重新引入 FP16 窄化来过门禁。
实验原型和逐用例结果保存在测试产物目录，不随生产源码安装。

对比性能需使用同一设备、软件版本、输入、精度和采样配置，保持 GPU 测试串行，
分别查看 GPU 稳态、主机提交、构建成本和 workspace。单次冷/热 build 是诊断数据，
不是经过多轮统计的性能承诺。
