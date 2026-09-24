# FlagDNN Hygon 全量适配与后续维护指南

## 2026-09 架构更新

Hygon 对照 NVIDIA 的 provider 分层，将原 `compiler.py`、`compiler_nn.py`、
`compiler_tensor.py`、`compiler_identity.py` 拆分为以下结构。旧章节中的
`compiler_nn` / `compiler_tensor` 指下述 dispatch 模块。

```text
backends/hygon/
├── compiler.py                 # request/identity 协议、图验证和 program 装配
├── dispatch/                   # 纯 Python 算子规划，不导入 Triton/Torch
│   ├── common.py               # schema、目标、类型与基本检查
│   ├── graph.py                # 算子族分发、图和内部 workspace 布局
│   ├── tensor_metadata.py      # Graph tensor 解析、storage/stride 检查
│   ├── pointwise.py            # pointwise schema、布局和参数选择
│   ├── tensor.py               # layout/reduction/matmul 规划
│   ├── nn.py / nn_common.py    # DNN 分发、stage/plan 数据结构
│   ├── convolution.py          # Hygon 卷积与多阶段方案
│   ├── normalization.py        # normalization 规划
│   ├── attention.py            # attention 规划
│   ├── extended.py / extended_schema.py  # 扩展算子分发和端口约束
│   ├── index.py / statistics.py / random.py
│   ├── position_embedding.py / resample.py / causal_convolution.py
│   ├── normalization_extended.py / moe_matmul.py / fp8_matmul.py
│   └── tuning.py               # 调优配置与 HCU LDS 候选过滤
└── codegen/
    ├── emit.py                 # registry source 到可执行 stage
    ├── abi.py                  # 参数、signature、wave64 launch ABI
    ├── pointwise.py            # Hygon FP32 compute source 特化
    ├── io.py                   # 原子文件写入和生成模块载入
    └── identity.py             # 编译资源、环境与 JIT 身份
```

依赖方向是 `compiler → codegen → dispatch`；dispatch 可以独立用于规划测试。
全部 codegen/dispatch Python 文件参与 identity 哈希和 dependency snapshot，
包目录变动也使缓存失效。CMake 同步安装两套包，安装态 consumer 使用安装树
内的 provider、kernel、tuning 和私有 JIT 资源。Hygon 保留现有 artifact schema 5、
wave64、严格 IEEE dot、私有卷积规划、workspace 和 JIT provenance 契约。

公共执行接口现在要求 workspace 基址至少 256 字节对齐。Hygon 图测试仍验证
更大的虚拟张量对齐，但传入地址也必须满足公共接口的 256 字节要求。

当前算子支持范围以 [operator-support.md](operator-support.md) 的 Hygon 章节为准。
下文 2026-08 的数量和 hipDNN 能力矩阵属于历史记录，不代表当前功能覆盖。

新增算子族的 schema 和 launch 规划分别放在 `dispatch/extended_schema.py`、
`dispatch/extended.py` 及对应 family 模块中；仍由 `codegen/emit.py` 生成
现有 stage/artifact ABI。FP8 普通 matmul 保留 Graph operation 名称，只切换
后端 kernel 选择；concatenate 拆为依赖有序的逐输入 copy stages。
RoPE backward 注册 Hygon 私有 kernel：逻辑坐标保持 INT32，仅在存储地址
计算时扩展为 INT64，规避当前 DTK 对部分旋转宽度下成对频率读取的错误编译。

功能参考先按输入 dtype 量化数据。Hygon 默认 matmul 明确使用 IEEE 参考，
避免继承公共 CPU oracle 的 NVIDIA 默认 TF32 策略。扩展算子采用 NV paired
runner 的容差公式和重复执行检查；MoE 直接比较高精度 CPU 累加结果，避免
先将参考输出降为 FP16/BF16 后产生额外舍入误差。


> 适用环境：Hygon/DTK/HIP，当前目标架构为 `gfx936`
>
> 当前代码目录：`backends/hygon`
>
> 2026-08 历史注册面：61 个 functional、57 个 benchmark、146 项 CTest。
> 当前支持范围见上方链接的 operator-support.md。

## 1. 结论与不可破坏的边界

Hygon 适配沿用 FlagDNN 已有的多平台架构：FlagDNN Frontend Graph API 是用户入口和被测接口，Hygon backend 独立完成 Graph IR 到 HCU Triton kernel 的编译和执行。validation 以 hipDNN classic primitive API 作为 vendor 数值 reference；缺少等价 primitive 的功能测试使用独立 CPU 数学参考，实际执行 Hygon kernel 并比较结果。CPU 参考仅用于功能测试，缺少等价 hipDNN 对照的性能测试仍 SKIP。

必须同时遵守以下约束：

1. `backends/hygon/**` 独立拥有 Hygon 的 artifact、backend、context、error、engine、compiler、tuning 和 validation 实现。
2. 生产代码与常规功能/性能测试不 include、import、编译或链接 `backends/nvidia/**`，也不修改 NVIDIA 源码。唯一例外是独立的 `integration.hygon.dtype_catalog` 元数据检查：直接编译 NVIDIA 原始用例选择器，以只记录 tensor 元数据的执行器替换实际运行，核对两端 shape/stride/dtype/offset；不调用 NVIDIA kernel，不依赖 CUDA/cuDNN。与 NVIDIA “架构一致”仍指生产职责和 contract 对齐。
3. 不在公共 Graph、Graph IR、`src/graph/lowering`、`tests/common` 或 `benchmark/common` 中增加 Hygon 算子分支。
4. production backend 不直接查找、链接或调用 hipDNN、MIOpen、hipBLAS、rocBLAS；hipDNN 只能出现在 `backends/hygon/validation/**`。
5. hipDNN 是 validation 唯一允许的 vendor reference SDK；不得直接调用 MIOpen、BLAS 或自定义 device oracle，也不得用 FlagDNN 自身 kernel 充当 reference。功能参考优先复用 `reference/cpu`；公共库缺少的逐元素公式、FP8 注意力公式及原始字节拷贝校验由 Hygon validation 自身维护。
6. 没有 hipDNN 对照不代表算子无法适配。功能测试可走独立 CPU 参考；确实未实现的 dtype/precision 必须明确 SKIP。CPU oracle 不得用于 benchmark，也不能产出 latency 或 speedup。
7. Hygon production 当前只支持 `libtriton_jit` execution engine；稳态执行必须使用调用者的 `hipStream_t`。

主验证关系固定为：

```text
同一个 platform-neutral case
        |
        +--> FlagDNN Frontend Graph --> Hygon compiler/runtime --> output A
        |
        +--> hipDNN primitive/sequence -------------------------> output B
        |                                                         |
        +--> independent CPU semantic oracle (functional only) -> output C
                                                                  |
                                                        stride-aware compare
```

hipDNN 没有与 cuDNN Frontend Graph 对等的 Graph builder 不构成阻塞。一个 FlagDNN Graph 节点可以映射为一个 hipDNN primitive，也可以映射为严格等价的 primitive sequence；functional 也可以使用独立 CPU semantic oracle；benchmark 缺少等价 hipDNN 对照时 SKIP。

## 2. 目录与职责对齐

下图记录 2026-08 拆分前的目录，用于理解后续历史章节；现行目录见文首：

```text
backends/hygon/
├── CMakeLists.txt
├── artifact.cpp / artifact.hpp       # artifact schema、ABI、stage DAG 校验
├── backend.cpp                       # backend ABI v2 唯一导出入口
├── context.cpp / context.hpp         # HCU device/context/target identity
├── error.cpp / error.hpp             # HIP 错误映射和 last error
├── compiler.py                       # provider 入口、pointwise、统一编排
├── compiler_tensor.py                # layout/reduction/matmul plan
├── compiler_nn.py                    # convolution/normalization/attention plan
├── compiler_identity.py              # compiler/resource identity
├── python_environment_identity.py    # Python/Torch/Triton 精确环境 identity
├── cmake/ResolveTritonJIT.cmake      # HCU JIT 单一 provenance 解析
├── kernels/
│   ├── registry.json                 # Hygon 私有 kernel override
│   └── binary_minmax.py              # DTK MIN/MAX 特殊值语义
├── engines/
│   ├── engine.cpp / engine.hpp
│   └── libtriton_jit.cpp             # HCU JIT、autotune、稳态 HIP launch
├── tuning/
│   └── common.yaml                   # Hygon-owned wave64 tuning space
└── validation/
    ├── CMakeLists.txt
    ├── hip_driver.hpp
    ├── tensor_io.cpp / tensor_io.hpp
    ├── hipdnn_reference.cpp / hipdnn_reference.hpp       # hipDNN primitive 基础
    ├── pointwise_reference.cpp / pointwise_reference.hpp
    ├── tensor_reference.cpp / tensor_reference.hpp
    ├── convolution_reference.cpp / convolution_reference.hpp
    ├── normalization_reference.cpp / normalization_reference.hpp
    ├── attention_reference.cpp / attention_reference.hpp
    ├── functional/
    │   ├── *_runner.cpp                                  # 9 个 family runner
    │   ├── hipdnn_*.cpp                                  # 9 个 reference hook wrapper
    │   └── test_jit.cpp / test_pointwise_smoke.cpp /
    │       test_runtime.cpp / test_graph.cpp
    └── benchmark/
        ├── runner.cpp
        ├── hipdnn_provider.cpp / hipdnn_provider.hpp
        ├── hip_graph.hpp / ops.hpp
        └── pointwise.cpp / tensor.cpp / convolution.cpp /
            normalization.cpp
```

functional 的 9 个 family runner 与 NVIDIA 的 family 边界一一对应：add、attention、composite、convolution、layout、matmul、normalization、pointwise、reduction。9 个 `hipdnn_*.cpp` 只实现公共 test contract 所需的 reference hook；真正的 primitive、capability 与执行 plan 位于独立的 `flagdnn_validation_hygon_reference` 静态库中，供 functional 和 benchmark 共同复用。

平台无关 CPU oracle 位于独立根目录，避免把 host 语义实现塞进 Hygon adapter：

```text
reference/
├── cpu/
│   ├── CMakeLists.txt
│   └── pointwise.cpp / pointwise.hpp
└── tests/
    └── cpu_pointwise_reference.cpp   # div/pow/mod/cmp_eq 的 cuDNN 独立验证
```

`reference/cpu` 不 include/link CUDA、cuDNN、HIP、hipDNN、MIOpen 或 BLAS；最初实现按右对齐规则广播的标量语义，当前还包含矩阵、卷积、归一化、注意力和其他扩展算子的数学参考。`reference/tests` 在 NVIDIA 构建中使用 cuDNN Graph，分别对 `div`、`pow`、`mod`、`cmp_eq` 的相同形状、标量和多轴广播语义做独立验证。

允许复用的代码必须是真正的 platform-neutral contract 或 utility，例如：

- `backends/backend_api.h`；
- `backends/autotune_policy.*`；
- runtime JSON/SHA；
- common kernel registry 和 common Triton kernel；
- `tests/common` 与 `benchmark/common` 的 case/runner contract。

如果 Hygon 与 NVIDIA 后续出现可证明的公共逻辑，应先在独立变更中抽到中立目录，并为两个 backend 建立 no-regression 测试。除上述独立元数据检查外，Hygon 不反向依赖 NVIDIA 文件，也不为 Hygon 修改 NVIDIA 私有实现。

## 3. 2026-08 算子目录（历史记录）

当时公共 manifest 定义 61 个 functional 和 57 个 benchmark。Hygon CMake 必须一一注册，不能通过漏注册隐藏 unsupported reference。

| Family | Functional | Benchmark | 说明 |
|---|---:|---:|---|
| Pointwise/composite | 44 | 44 | unary、binary、ternary、scale、sigmoid backward |
| Tensor | 5 | 5 | `reshape`、`transpose`、`slice`、`reduction`、`matmul` |
| Convolution | 4 | 4 | `conv_fprop`、`conv_dgrad`、`conv_wgrad`、`conv_bias_relu` |
| Normalization | 4 | 4 | `layernorm`、`rmsnorm`、`batchnorm`、`batchnorm_inference` |
| Attention | 4 | 0 | `sdpa`、`sdpa_backward`、`sdpa_fp8`、`sdpa_fp8_backward` |
| 合计 | **61** | **57** | attention 无 benchmark 是公共 manifest 的明确设计 |

44 个 pointwise/composite 算子为：

```text
add sub mul div pow max min mod add_square cmp_eq
abs ceil cos elu erf exp floor gelu gelu_approx_tanh identity
leaky_relu log logical_not neg reciprocal relu rsqrt sigmoid sin
softplus sqrt swish tan tanh
binary_select cmp_ge cmp_gt cmp_le cmp_lt cmp_neq logical_and
logical_or scale sigmoid_backward
```

“已注册 production lowering”与“hipDNN 能做数值比较”是两个独立维度。本文第 6 节的 SUPPORTED/SKIP 只表示 reference 能力，不表示 Hygon compiler 是否生成 kernel。

## 4. Production 编译与执行架构

### 4.1 Graph 到 provider

生产链路保持公共 Graph contract 不变：

```text
FlagDNN Frontend Graph
  -> platform-neutral validation/lowering
  -> versioned Graph IR（backend="hygon"）
  -> backends/hygon/compiler.py
  -> artifact manifest + execution-program stage DAG
  -> libflagdnn_backend_hygon.so.2
  -> HCU libtriton_jit
  -> Triton HCU kernel / HIP module launch
```

provider 必须校验：

- request schema、artifact schema 和 execution-program version；
- `backend == "hygon"`；
- `target` 满足安全的 `gfx...` 编码；
- engine 为 `libtriton_jit`；
- tensor UID、dtype、dimensions、strides、alignment、binding offset；
- node arity、attribute、workspace 和 argument ABI；
- tuning candidate 与 stage 消费的 META contract。

任何 schema、dtype、layout、target、ABI 或候选错误都必须显式失败，不得切换 NVIDIA provider 或隐式回退到另一套平台实现。

### 4.2 Compiler 模块分工

`compiler.py` 是唯一 provider 入口，负责：

- 解析 request、tensor table 和 pointwise node；
- 统一调度 `compiler_tensor` 与 `compiler_nn`；
- 分配 graph virtual workspace、provider-local workspace 和 global scratch；
- 加载 Hygon tuning，准备默认/自动调优 variants；
- 调用 `standalone_compile` 生成 HCU code object；
- 生成 artifact、stage dependency、launch 和 argument ABI。

`compiler_tensor.py` 负责：

- `reshape`、`transpose`、`slice`；
- reduction ADD/AVG/MUL 等公共 reduction mode；
- strided/batched `matmul`；
- tensor operation 的 shape/stride 校验、grid 和 workspace plan。

`compiler_nn.py` 负责：

- convolution fprop/dgrad/wgrad；
- layernorm、rmsnorm、batchnorm、batchnorm inference；
- SDPA forward/backward 与 FP8 forward/backward；
- 一个 Graph node 展开成一个或多个 `KernelStagePlan`；
- internal stage dependency、runtime scalar 和 provider-local workspace。

这种拆分只是 Hygon provider 内部的可维护性边界。对外仍遵守与 NVIDIA 相同的 Graph IR、compiler loader、artifact schema 和 backend ABI。

### 4.3 多阶段 plan、workspace 与 runtime scalar ABI

NN node 使用以下 plan 模型：

```text
NodePlan
├── stages: KernelStagePlan[]
├── internal dependencies
└── workspace tensors（可选）
```

典型多阶段场景包括：

- SDPA backward 的局部 `delta` workspace；
- backward 中 `dq`、`dk`、`dv` 的依赖 stage；
- FP8 attention 的清零/统计 stage；
- fused/composite operation 的顺序 stage。

workspace 必须按以下顺序统一打包：

1. 公共 Graph virtual tensor workspace；
2. `compiler_nn` 申请的 provider-local workspace，使用合成的正 UID 和 256-byte 对齐；
3. engine 使用的 4096-byte global scratch。

不得让 provider-local workspace 与 Graph virtual tensor、外部 binding 或 global scratch 重叠。artifact validator 必须校验每个 workspace tensor 的 UID、offset、size、alignment 和范围。

stage ABI 支持：

| ABI kind | 含义 |
|---|---|
| `tensor` | Graph node 的第 N 个输入/输出 tensor |
| `workspace_tensor` | provider-local 的命名 workspace tensor |
| `scalar_i32` | 运行时 32 位整数/flag |
| `scalar_f32` | 运行时 FP32 scalar，例如 attention scale |
| hidden scratch | engine 附加的两个 scratch pointer |

只有真正的 `tl.constexpr` 才能进入 stage `constants`。attention scale、动态 flag 等非 constexpr 数值必须进入 `runtime_values` 和 runtime signature，不能因为 Python 中是数值就被错误特化。full signature、runtime signature、argument layout 和实际 launch 参数必须一一对应。

外部 Graph dependency 连接到展开 node 的首个可执行 stage；该 node 的 Graph output 在最后一个生产该 output 的 stage 后才可见。internal dependency 必须写入 artifact，autotune 和稳态执行都按同一 DAG 回放。

### 4.4 Kernel 与 autotune

Hygon 优先复用已证明 platform-neutral 的 common Triton kernel，并由 `backends/hygon/tuning/common.yaml` 拥有 HCU/wave64 候选。当前 tuning table 包括：

```text
binary / relu
reduction / matmul
conv2d_spatial
layer_norm / rms_norm / batch_norm
sdpa / sdpa_backward_dq
sdpa_fp8 / sdpa_fp8_backward_dq
```

基本规则：

- wavefront 固定为 64，不能继承 CUDA warp32 假设；
- Hygon matmul 使用严格 IEEE input precision，不继承 CUDA TF32 policy；
- stage 只接收自己声明的 META，provider 从 tuning superset 投影；
- LayerNorm/RMSNorm 的大 shape 必须保留 tuning YAML 中 10 个显式候选，不能因 normalized row 大小退化为单候选；
- BatchNorm 在 `N > 512` 时使用 generic kernel，并保留 4 个合法候选；
- attention 的 D/V tile 必须至少为 16，D 或 V 大于 256 在 compiler validation 阶段明确拒绝；普通 SDPA backward 要求 K/V head 数一致，但 forward 仍允许独立的 grouped-query K/V head 数；
- 普通 SDPA backward 的所有 query tile 都保持 `FULL_ATTENTION=false`，包括非 banded case，确保 `SQ` 非 `BLOCK_M` 整倍数时仍执行尾块 mask；
- FP8 SDPA backward 的 DQ stage 固定 `FULL_BLOCKS=false`，避免错误选择 full-block 假设；
- 可调 stage 必须保留合法候选，内部固定清零 stage 可以使用固定配置；
- candidate compile、warmup、依赖回放、HIP event 测量和 winner 固化发生在 executable build 阶段；
- steady-state `execute` 不得重新编译或重新调优。

autotune 的 candidate 过滤优先使用 FlagDNN 自身的类型化 `HygonError`。
只有以下 HIP 状态允许丢弃 candidate 后继续：`hipErrorLaunchOutOfResources`、
`hipErrorInvalidConfiguration`、`hipErrorInvalidDeviceFunction`、
`hipErrorInvalidImage`、`hipErrorNoBinaryForGpu`、
`hipErrorInvalidKernelFile`。上游 HCU 仅以普通 `std::runtime_error` 暴露部分
HIP 错误，因此 Hygon 私有适配器还允许精确匹配这些状态的完整
`hipGetErrorString`，或带固定 `HCU kernel launch failed: ` 前缀的完整消息。
不得使用子串或宽泛的错误关键词匹配，也不得依赖外部 `HcuError` 扩展。
派生异常、未知消息、没有 HIP status 的 `HygonError`、compiler/JIT protocol、
source/identity/ABI、timeout 及其他 HIP 错误均使 executable build 失败。
缓存 winner 的重新准备使用相同分类。

首次 launch 前，FlagDNN 使用已校验的 standalone helper 和相同编译参数获取
缓存目录，通过上游公开 metadata API 检查架构与共享内存。架构不匹配直接失败；
共享内存超限由 FlagDNN 抛出类型化资源错误，在上游分配模块前拒绝候选。
后续 JIT launch 复用编译缓存。此逻辑仅位于 Hygon 后端，不修改依赖源码或其他平台。

只有 common kernel 无法承载且有可复现收益或语义差异时才新增 Hygon 私有 registry override。一旦登记 Hygon override，编译或加载失败必须直接失败，不允许捕获异常后回退 common。当前 `min/max` 使用 Hygon 私有 `binary_minmax.py`，原因是已验证 DTK hipDNN OpTensor 对 NaN 与 signed-zero 的行为不能用原 common kernel 无条件表达；该 override 不修改 NVIDIA 或 common kernel。

### 4.5 Backend plugin 与 libtriton_jit engine

C++ plugin 与 NVIDIA 保持职责对齐但源码完全独立：

- `backend.cpp` 提供 ABI v2 入口；
- `context.cpp` 管理 HCU device、primary context、target fingerprint 和 device identity；
- `artifact.cpp` 校验 manifest、file hash、stage DAG、variant 和 ABI；
- `engines/engine.cpp` 选择 Hygon engine；
- `engines/libtriton_jit.cpp` 准备 HCU JIT function、autotune 并执行 HIP launch。

`create_executable` 阶段允许：

- 解析 artifact；
- 编译/加载所有候选；
- 创建私有 HIP stream/event/buffer；
- 回放依赖 stage 并测量候选；
- 过滤明确的资源不足候选；
- 固化 winner、workspace 和 `hipFunction_t`。

`execute` 阶段只允许：

- 校验 binding UID、workspace 和 caller stream；
- 组装已固化的 tensor/workspace/scalar 参数；
- 按 stage DAG 在调用者 `hipStream_t` 上异步 `hipModuleLaunchKernel`。

稳态路径不得 `hipMalloc`、首次 compile、创建 descriptor、做全局/stream synchronize，或永久改变调用线程的 current device。

外部 binding 的 alias 语义只服从公共 Graph/operator contract。不同 tensor UID
即使绑定的 device pointer 范围相同或部分重叠，Hygon engine 也不得增加 NVIDIA
和公共 runtime 都没有的 blanket pairwise-overlap 拒绝；这并不把公共 contract
未定义的 input/output alias 宣称为合法，只是保证平台间行为一致。Graph tensor
UID、producer/dependency、null/alignment、地址范围溢出以及内部 workspace/global
scratch 不重叠等既有校验仍然保留。

### 4.6 Identity、cache 与安装资源

`compiler_identity.py` 的 provider identity 覆盖：

- `compiler.py`、`compiler_tensor.py`、`compiler_nn.py` 和 identity 自身；
- schema/version、backend/engine、target 和 wave64 policy；
- 实际选中的 common/Hygon kernel source；
- Hygon tuning YAML；
- CMake 选择的 JIT library、三个 runtime helper 及 Python 环境摘要。

`python_environment_identity.py` 使用稳定 JSON schema 记录实际 Python executable，以及 `torch`、`torch._C`、`yaml`、Triton 顶层模块和关键 compiler/runtime/backend 模块的真实文件哈希；同时哈希完整 Triton package tree，并把 distribution metadata 绑定到实际 module root。已加载的 `triton.*` 模块只要来自所选 package root 之外就直接失败。

Hygon 的 CMake 发现变量全部带平台命名空间：

```text
FLAGDNN_HYGON_TRITON_JIT_ROOT
FLAGDNN_HYGON_TRITON_JIT_DIR
FLAGDNN_HYGON_TRITON_JIT_LIBRARY
FLAGDNN_HYGON_TRITON_JIT_INCLUDE_DIR
FLAGDNN_HYGON_TRITON_JIT_SCRIPT_DIR
```

通常只设置 `ROOT` 或 `DIR`；默认从 `${PROJECT_SOURCE_DIR}/../libtriton_jit` 发现。resolver 只接受同一个 HCU `TritonJITConfig.cmake` 推导出的完整 build、installed-lib 或 installed-lib64 布局，并校验 exported target location、SONAME、header、library 和 scripts。单独覆盖 library/include/scripts 时，它们仍必须属于同一布局；混源和旧 cache 路径直接配置失败。Hygon 不读取或写入全局 `TritonJIT_DIR`，因此 NVIDIA 与 Hygon 的 CMake package 选择不会互相覆盖。

运行时在第一次调用任何 JIT helper 或创建 JIT function 之前完成 fail-closed 校验：对实际已映射 JIT image 做哈希；核对 build/install tree 中的 `standalone_compile.py`、`gen_ssig.py` 和环境 helper；校验已 import Python module 的 `__file__` 与哈希；在隔离 namespace 中执行那一份已哈希的环境 helper 并与 CMake identity 精确比较。`integration.hygon.jit` 还包含预加载伪造 `gen_ssig` module 的子进程回归，要求在使用它之前失败。

runtime compiler identity 查询在同一 `RuntimeContext` 中按 compiler executable/entry/backend/target/engine、相关文件 metadata 和完整排序环境 snapshot 做进程内 memo；`set_compiler` 或环境 snapshot 变化会使 memo 失效。这减少重复 provider subprocess，但不能绕过 identity 校验。cache miss 在临时 artifact 完成并通过 manifest 校验后、原子发布前必须强制绕过 memo 再次查询 identity；compiler 子进程也在 compile request 前后核对完整依赖快照。任何端点可观测的 dependency/environment 变化都清理临时目录并硬失败，不更新 `active_identity`。外部进程仍不得在编译事务期间修改后再恢复 compiler/kernel/tuning/JIT 资源；前后快照无法识别完全恢复的瞬时 A→B→A，因此资源不可变是明确的宿主协作契约。

compiler cache 的离线复用只有一个严格例外：`posix_spawnp` 对 compiler
executable 返回**精确 `ENOENT`**，并被转换为专用的
`CompilerExecutableUnavailable` 类型时，才允许按缓存中已经记录并验证的 active
identity 命中。compiler 能启动但输出 malformed identity、非零退出、timeout、
临时失败，或 dependency/environment snapshot 在采集期间不稳定时，全部硬失败；
不得回退旧 identity、不得把暂态故障伪装成 cache hit。其他 spawn 错误也不属于
“离线 compiler”例外。

kernel、registry、tuning、compiler、JIT library、JIT scripts 或 Python/Triton/Torch 环境变化都必须造成 cache miss。artifact 内容损坏或与原 request Graph 语义不一致必须拒绝，不能带病复用。

Hygon JIT 初始化还涉及 Python/C 的进程全局状态，必须 fail closed：初始化前分别
检查 C environment 和已初始化 Python 的 `os.environ`；
`TRITON_JIT_BACKEND` 只能未设置或精确为 `HCU`，非 HCU 值直接失败。由 FlagDNN
首次发布 `HCU` 后，后续调用必须重新核对这两个 backend view 没有被删除或改写；
受控 `PYTHONPATH`/`sys.path` 的配置必须 once-only、去重，并继续受实际 imported
module path/hash identity 约束。plugin 内部 mutex 只能串行化
FlagDNN 自己的 JIT 初始化/build，不能替应用程序同步 `setenv`、`unsetenv`、
Python `os.environ` 或 `sys.path`；嵌入应用不得在 identity/JIT 初始化与 build
期间并发修改这些状态。需要另一 Triton backend 或无法满足该同步边界时，应使用
独立进程，不能依赖未定义的进程全局状态竞争。

安装树必须包含：

```text
lib/libflagdnn_backend_hygon.so.2
lib/flagdnn/hygon/<exported libtriton_jit SONAME>
lib/flagdnn/share/triton_jit/scripts/standalone_compile.py
lib/flagdnn/share/triton_jit/scripts/gen_ssig.py
lib/flagdnn/share/triton_jit/scripts/flagdnn_python_environment_identity.py
share/flagdnn/compiler/flagdnn_codegen/main.py
share/flagdnn/backends/hygon/compiler.py
share/flagdnn/backends/hygon/dispatch/*.py
share/flagdnn/backends/hygon/codegen/*.py
share/flagdnn/backends/hygon/python_environment_identity.py
share/flagdnn/backends/hygon/flagdnn_hygon_compiler_environment.json
share/flagdnn/backends/hygon/tuning/common.yaml
share/flagdnn/backends/hygon/kernels/registry.json
share/flagdnn/backends/hygon/kernels/binary_minmax.py
share/triton_jit/scripts/standalone_compile.py
share/triton_jit/scripts/gen_ssig.py
share/triton_jit/scripts/flagdnn_python_environment_identity.py
```

构建树必须镜像安装态的私有 JIT 相对布局，而不是把 HCU JIT 放在 plugin
旁边：

```text
build/hygon/backends/hygon/libflagdnn_backend_hygon.so.2
build/hygon/backends/hygon/flagdnn/hygon/<exported libtriton_jit SONAME>
build/hygon/backends/hygon/flagdnn/share/triton_jit/scripts/standalone_compile.py
build/hygon/backends/hygon/flagdnn/share/triton_jit/scripts/gen_ssig.py
build/hygon/backends/hygon/flagdnn/share/triton_jit/scripts/flagdnn_python_environment_identity.py
```

配置阶段将 CMake 实际选择并 canonicalize 后的 library 复制到 Hygon 私有目录，并仅在私有副本中去除 RPATH/RUNPATH 的空项，保留所有非空路径；外部库不做修改。运行时 image SHA 对应处理后的副本，原始依赖仍参与 provenance。安装阶段以 exported SONAME 复制同一私有副本，不能重新搜索另一候选；header metadata 与三个 runtime helper 也必须来自同一 provenance。`share/triton_jit/scripts` 是 compiler identity 使用的 canonical SDK 副本；由于 `libtriton_jit::get_script_dir()` 按已映射 library 的相对位置解析，完全相同的三个脚本还必须安装到 `lib/flagdnn/share/triton_jit/scripts`，不能回退到原源码树。

构建态与安装态 plugin 的动态加载契约完全相同：`DT_RPATH` 必须精确为
`$ORIGIN/flagdnn/hygon`，分别只加载各自树中的私有 HCU JIT。RPATH 不得包含空
路径元素（空元素会把当前工作目录引入搜索）、额外的 `$ORIGIN` 根目录、构建目录、
原 `libtriton_jit` 目录或 DTK 绝对路径。安装 SDK 因而应可脱离原
`libtriton_jit` 构建树运行；Torch/Python 与 DTK/HIP runtime 仍是宿主机先决
条件。生产部署可显式使用 `FLAGDNN_HYGON_PYTHONPATH`，但 installed-consumer
gate 会清除此 override 来证明默认安装资源闭包完整。

必须明确一个 ELF 限制：CUDA 与 HCU 版 `libtriton_jit` 具有相同 SONAME。HCU image 使用 `lib/flagdnn/hygon` 私有路径，不会覆盖 NVIDIA 可能使用的根 lib 路径，因此两种 backend 可以安全安装到同一 prefix；但它们仍不能安全地在同一进程中同时加载，应使用独立进程运行 NVIDIA 与 Hygon。若动态加载器复用已加载的另一平台 image，Hygon runtime identity 会在使用前拒绝，而不是让两套 JIT 共存。

validation executable/reference 不安装。

## 5. Validation 设计

### 5.1 Vendor reference 与受控 CPU functional oracle

Hygon validation 的 vendor reference 只直接使用 HIP runtime 与 hipDNN public C ABI：

- include `hipdnn.h`；
- 链接 `libhipdnn.so`；
- 使用 hipDNN tensor、OpTensor、activation、reduction、convolution、BatchNorm 等 classic primitive；
- composite reference 使用多个 hipDNN primitive 和独立 reference workspace；
- 不使用 hipDNN Graph API，也不假设它存在。

`functional/host_runner.hpp` 管理独立 CPU 参考的输入量化、设备缓冲区、Graph 执行和比较；`host_pointwise.hpp` 与 `fp8_attention_reference.hpp` 实现公共库缺少的标量数学公式。其余算子优先调用 `reference/cpu`。期望值只能依赖输入与公开算子语义，不能读取 DUT 输出，也不能调用 DUT kernel。输入先按实际 dtype 量化，结果按输出 dtype 量化后比较；INT32 算术使用精确整数参考，identity/layout/concatenate 按原始字节比较，保留 NaN payload 和 signed zero。缓冲区检查同时覆盖输入不变性、非连续布局、binding offset 和输出 padding。

functional case 有三种 reference 状态：

| 状态 | 行为 |
|---|---|
| `HIPDNN_SUPPORTED` | 构建并执行 hipDNN primitive/sequence，再与 FlagDNN Graph 输出比较 |
| `CPU_FUNCTIONAL_ORACLE` | 计算独立 CPU 期望值，执行 FlagDNN Graph 后逐逻辑元素比较 |
| `HIPDNN_UNSUPPORTED` | 在构建 DUT 前输出结构化原因并跳过该 case |

capability 可以包含独立的 descriptor/build/real-execute probe，例如 convolution algorithm 和 BatchNorm runtime gate；当前已知无可执行 primitive 的 BF16 reduction 则在 validated allowlist 之前明确 SKIP。native status 分类严格限定为：只有 `HIPDNN_STATUS_NOT_SUPPORTED` 可以作为 runtime capability SKIP；`ARCH_MISMATCH`、`RUNTIME_PREREQUISITE_MISSING`、`ALLOC_FAILED`、`BAD_PARAM`、`INTERNAL_ERROR`、`INVALID_VALUE`、`EXECUTION_FAILED`、`VERSION_MISMATCH` 等全部是硬失败。语义上无法映射的 case 可以在调用 DUT 前由显式 capability reason SKIP，但不能用环境损坏或执行错误伪装 unsupported。capability 已返回 supported 后，正式 hipDNN plan 的任何非 `NOT_SUPPORTED` build/execute 错误都必须 FAIL。

### 5.2 结构化 SKIP contract

统一输出格式：

```text
[SKIP][hipdnn] op=<op> case=<case> reason=<exact reason> <environment> <tensor metadata>
```

规则：

1. reason 至少说明 operation、case、hipDNN header/runtime、arch、dtype、shape/layout 和不支持的 primitive/attribute。
2. 一个 operator 的全部 case 都 unsupported 时，进程返回 77；CTest 使用 `SKIP_RETURN_CODE 77`。
3. 同一 operator 同时有 supported/unsupported case 时，supported case 必须实际运行；unsupported case 写入 `skip_records`。
4. 任一 supported case 数值错误、descriptor 错误、runtime 错误或 timeout 都是 FAIL。
5. benchmark 对 unsupported case 不执行 FlagDNN-only 计时，不输出 reference latency 或 speedup。
6. DTK 升级后 skip reason 或 capability 变化必须评审，不能长期固化过时 SKIP。

SKIP 只说明当前 hipDNN 无法提供精确 oracle。它既不表示 production 不支持，也不构成该 production kernel 的数值正确性证明。CPU-reference functional 的 mismatch、runtime 错误或 timeout 都是 FAIL；缺少等价 hipDNN 对照的 benchmark 仍 SKIP。

### 5.3 Functional 与 benchmark

functional 在构建层面使用单一 `flagdnn_test_hygon_adapter`，由上述 9 个 family runner、9 个 `hipdnn_*.cpp` wrapper 和公共 `flagdnn_test_common_objects` 组成。它链接独立的 `flagdnn_validation_hygon_reference` 与 `flagdnn_validation_cpu_reference`，再通过与 NVIDIA 相同的公共 per-operator suite 注册 contract 生成 61 个 executable/CTest。

functional supported case 的顺序：

1. 校验公共 case 并运行 hipDNN capability；
2. 分别创建 FlagDNN 和 hipDNN 的 device input/output/workspace；
3. 使用相同逻辑输入，但不共享 output 或中间 workspace；
4. 在指定 HCU 和 caller stream 上执行；
5. 按 dimensions/strides/offset gather；
6. 比较数值、NaN/Inf 分类和 padding sentinel；
7. 应用公共 case 的 atol/rtol。

CPU-oracle case 保持相同顺序与检查项，但第 1 至 3 步替换为：准备并保存实际 dtype 量化后的逻辑输入，在 host 上按算子语义和 broadcasting 计算期望值，再按 output dtype 量化。CPU oracle 与 DUT 不共享输出或实现代码；公共 CPU 参考可通过 `reference/tests` 与 NVIDIA/cuDNN 交叉检查；Hygon 私有标量参考使用独立数学实现，不依赖 Hygon DUT。

对 FlagDNN 合法、但 hipDNN OpTensor 无法直接接收相同物理 descriptor 的 strided/broadcast case，允许使用“逻辑等价 packed reference”，但必须满足以下边界：

- DUT 仍使用原始 dimensions/strides/broadcast，并检查非逻辑 padding 未被写坏；
- reference 只把相同逻辑输入重排或显式展开到 packed A/B/C buffer，实际算术仍由 hipDNN primitive 完成，host 不计算期望输出；
- 最后按 DUT 原始 layout gather，与 packed hipDNN 输出逐逻辑元素比较；
- 这种 reference 只证明同一逻辑运算，不把原始任意 strides 宣称为 hipDNN descriptor capability。

hipDNN classic activation 对部分 compact logical shape 还有 descriptor-rank 限制。
validation 必须先按原 tensor 完成 operation/dtype/shape capability 判断；只有所有参与
该 activation primitive 的 tensor 都是 offset=0、相同 dtype/shape/物理映射、
positive-stride 且无 padding/hole/alias 的 compact storage 时，才允许把 descriptor
纯物理重解释为四维 `{batches, channels, 1, 1}`、stride
`{channels, 1, 1, 1}`。DUT descriptor 和 logical compare 均保持原样。sigmoid
backward 的内部 forward/backward 所有 activation tensor 必须一起转换；
`conv_bias_relu` 只转换最终 ReLU descriptor，convolution output 与 bias OpTensor
仍使用原 descriptor。条件不满足就保留原 descriptor 或按 capability SKIP，不能
用重排数据、host oracle 或放宽比较掩盖不兼容。

`integration.hygon.pointwise_smoke` 固定覆盖 `add(alpha=-0.75)`、`sub(alpha=0.5)`、`mul`、`min`、`max`：DUT 使用 left `{2,3,4}`/stride `{31,9,2}`、right `{1,4}`/stride `{13,3}` 和 output stride `{37,11,2}`，hipDNN 侧使用逻辑等价 packed `{2,3,4}` A/B/C。该 gate 同时检查广播展开、数值和 output padding sentinel。

同一个 integration gate 还对 contiguous `min/max` 分别运行 FP32 与 FP16 特殊输入，覆盖 `+0/-0` 的两种顺序、单侧/双侧 NaN 和普通正负值；比较时 NaN 要求分类一致，zero 要求 signbit 与 hipDNN 完全一致。这是第 6.1 节特殊值语义成立的动态资格门禁，不应被普通容差比较吞掉。

benchmark 使用单一 `flagdnn_benchmark_hygon_adapter`，其平台实现由 `runner.cpp`、`hipdnn_provider.cpp`、`pointwise.cpp`、`tensor.cpp`、`convolution.cpp` 和 `normalization.cpp` 组成。case 的 Graph、shape、dtype、layout、输入域、atol/rtol 和 `BenchmarkConfig` 全部来自 `benchmark/common`，与 NVIDIA benchmark 同源；Hygon validation 不允许另建私有 shape 或放宽 tolerance。

FP16 convolution benchmark 有且只有一个精确限定的 oracle policy：先按原
`TensorSpec` 把逻辑输入编码为 FP16 并解码，得到已经发生 FP16 量化的值；再把这些
值写入独立的 FP32 descriptor/buffer/workspace，由相同 public hipDNN convolution
primitive（`conv_bias_relu` 为同一 primitive sequence）生成 correctness oracle。
FlagDNN DUT 仍执行原 FP16 Graph，公共 shape、输入域和 atol/rtol 均不改变。host
只负责 dtype 编解码、layout scatter/gather 与比较，不执行 convolution，因此这不
是 host oracle，也没有引入 MIOpen/BLAS。

同一 case 的性能对照仍必须是 native FP16 hipDNN primitive，并使用另一套原 FP16
descriptors、buffers 和 workspace。每个实际可执行的 performance algorithm 都先与
已保存的 FP32 hipDNN oracle 做 accuracy gate；通过的 candidate 才能进入 warmup、
HIP Graph capture 和 sampling。如果所有可执行 native FP16 candidates 都只因数值
不一致被拒绝，则在 warmup 前输出
`HIPDNN_PERFORMANCE_ACCURACY_EXHAUSTED` 结构化 SKIP，且双方都输出零 timing
samples/records。该例外不扩展到 FP32/BF16、functional 或其他 family；descriptor、
workspace、runtime、非 capability status、capture replay 和 timing 错误仍 hard
FAIL。

公共 `BenchmarkConfig` 默认值为：

| 配置 | 默认值 |
|---|---:|
| warmup iterations | 10 |
| sample count | 20 |
| iterations per sample | 50 |

公共 case 可以在 `benchmark/common` 中显式覆盖这些值；覆盖后 NVIDIA 与 Hygon 仍读取同一份配置。Hygon 计时与 NVIDIA 的批量 graph-launch 口径对齐：

1. 先对 FlagDNN 与 hipDNN 执行相同 correctness gate；
2. 分别 warmup 两个 provider；
3. 分别把各自的 `iterations_per_sample` 次 execute 捕获进独立 HIP Graph；
4. 每个 sample 用 HIP event 测量一次 graph launch，再除以捕获的 iteration 数，得到单次执行微秒数；
5. 偶数 sample 先测 FlagDNN、后测 hipDNN，奇数 sample 反转，避免固定顺序偏差；
6. 排序后以 nearest-rank 规则 `ceil(p*n)-1` 计算 median 与 p90；
7. 每个 provider 输出 JSONL schema v1 的 `steady_state` 记录，字段包含 `provider`、`case`、`unit`、`median`、`p90` 和 `samples`；
8. 另输出文本 `speedup = hipdnn_median / flagdnn_median`。

runner 对每次 provider 调用显式标注 `probe`、`warmup`、`capture-build`、
`capture-replay` 或 `timing` phase，错误必须同时携带 provider 与 phase；每个需要
完成的边界都检查 HIP stream synchronize 与 pending asynchronous error，不能把
前一阶段的异步错误归到后一阶段或继续计时。

hipDNN 存在少数 primitive/shape 可以直接执行但不能进入 HIP Graph capture。只有
在 hipDNN direct probe 与 correctness、FlagDNN correctness、双方 warmup、以及
FlagDNN capture-build 都已成功之后，hipDNN 的 `capture-build` 精确返回
`HIPDNN_STATUS_NOT_SUPPORTED` 或 `HIPDNN_STATUS_EXECUTION_FAILED`，且确认 stream
已经退出 capture 并通过同步/异步错误检查时，才把该 case 结构化 SKIP。该例外只
是 benchmark capture capability，不改变 functional oracle，也不适用于 descriptor、
direct execute、warmup、capture-replay 或 timing；SKIP case 不输出任何 latency 或
speedup。

另有一个按精确 primitive mapping 固化的 qualification 例外：当前 header
7000/runtime 8910 上，`neg` 的 hipDNN `IDENTITY + alpha=-1` direct primitive
correctness 可稳定通过，但重复 HIP Graph timing 会在没有返回 hipDNN status 的情况
下触发进程级设备 VM fault。因此 functional 仍必须执行并通过 direct hipDNN 对照；
benchmark 在 direct correctness 通过后由独立的 `capture-replay` capability gate
结构化 SKIP，不再进程内探测已知危险 replay，也不输出 latency/speedup。该 gate
只匹配 `IDENTITY + alpha=-1`，不得扩展成一般 activation 或 direct-execute 跳过。

correctness oracle 与 performance plan 必须分离。尤其 convolution 先用固定的
hipDNN correctness plan 生成并保存输出，再逐个 real-execute performance
algorithm，与这份 oracle 比较；数值不准确的 algorithm 可以显式拒绝。FP16
candidate 的纯数值耗尽只适用上面的零样本结构化 SKIP；其他 primitive
build/execute 错误仍按第 5.1 节硬失败。即使 pre-timing correctness 已通过，计时
结束后也必须无条件重新读取 FlagDNN output 和实际被计时的 hipDNN output，并让
二者分别与保存的 hipDNN correctness oracle 比较；任一 post-timing mismatch 都
使 case FAIL，不能输出 latency/speedup。convolution v7 metadata query 只有返回
`HIPDNN_STATUS_NOT_SUPPORTED` 或 `HIPDNN_STATUS_EXECUTION_FAILED` 时，才允许回退
到 public enum candidate 列表；这只是候选发现兼容层，真实 workspace query、
primitive execute 和 oracle 比较仍执行第 5.1 节的严格状态分类。

只有 capability gate（包括受控的 real-execute probe、严格限定的 hipDNN
benchmark capture-build gate、上一段精确的 NEG capture-replay qualification，
以及 FP16 convolution performance 的纯数值 candidate 耗尽）明确返回 unsupported
才允许 SKIP。最终 capability
为 supported 后，reference/DUT build、direct execute、warmup、FlagDNN capture、
双方 capture-replay 或 event timing 任一失败都必须 FAIL，不能降级为 SKIP。没有
精确 hipDNN reference 的 case 不执行任何 provider 计时，也不产生 FlagDNN-only
latency 或 speedup。

attention 功能测试使用独立 CPU 参考，包括 FP8 中间量的量化、scale/descale 和 amax。性能用例保持公共注册集合，因没有等价 hipDNN 对照而 SKIP，不输出 FlagDNN-only speedup。

### 5.4 直接依赖与 vendor 传递闭包

必须区分本仓库选择的直接接口与预编译 vendor library 的传递闭包：

- production plugin 的直接依赖是 HCU `libtriton_jit`、Python、HIP runtime `libgalaxyhip` 和系统库；它不直接链接或调用 hipDNN、MIOpen、hipBLAS、rocBLAS。
- validation target 直接链接 FlagDNN、HIP runtime、hipDNN 和仓库内静态 CPU oracle；CPU oracle 不增加动态 vendor 依赖。validation 不直接 include/link/call MIOpen、hipBLAS、rocBLAS，也不把它们当作 oracle。
- 当前 `libtriton_jit`/Torch 的运行时闭包可能传递加载 MIOpen、hipBLAS、rocBLAS；`libhipdnn` 自身也可能有 DTK 管理的传递依赖。这是 vendor-managed closure，不是 FlagDNN 选择的 reference API。
- 因此不能声称“进程绝不会加载 MIOpen/BLAS”；审计必须同时报告 direct `DT_NEEDED` 和完整 runtime closure，并明确二者含义。

依赖门禁的目标是防止 FlagDNN production/reference 直接越界，不是篡改或隐藏 vendor 预编译库内部依赖。

## 6. 当前 reference capability 矩阵

以下矩阵保留 2026-08 的 hipDNN primitive 对照能力记录。表内 SKIP 不再代表功能算子未适配：当前独立 CPU 参考可覆盖其中多类算子；功能支持范围以 operator-support.md 为准。

### 6.1 Pointwise/composite

基线支持 FP32/FP16；BF16 不在当前已验证 pointwise capability 内。

| 状态 | 算子 | 精确 reference 与约束 |
|---|---|---|
| SUPPORTED | `add sub mul min max` | hipDNN OpTensor；FP32/FP16、rank 3/4；普通 functional reference 要求 A/B/C 同 dimensions 且同 strides；布局扩展由逻辑等价 packed reference gate 覆盖 |
| SUPPORTED | `add_square` | hipDNN MUL → ADD sequence，独立中间 workspace |
| SUPPORTED（`neg` benchmark capture SKIP） | `abs identity neg scale` | activation identity/abs 与 alpha 映射；`neg` 使用 identity + `-1`，direct functional oracle supported，但当前 qualification stack 禁止其 HIP Graph replay timing |
| SUPPORTED | `elu relu sigmoid tanh softplus swish` | hipDNN activation；softplus 仅 `beta=1`；attribute 必须有精确映射 |
| SUPPORTED | `sigmoid_backward` | hipDNN sigmoid forward/backward sequence，独立 workspace |
| FUNCTIONAL CPU / BENCHMARK SKIP | `div pow mod cmp_eq` | hipDNN 无精确 public primitive；functional 使用 `reference/cpu` 并真实执行 DUT，benchmark 不计时且不报告 speedup |
| SKIP | `ceil cos erf exp floor gelu gelu_approx_tanh leaky_relu log logical_not reciprocal rsqrt sin sqrt tan` | primitive 缺失或已验证 DTK 语义/attribute 不精确 |
| SKIP | `binary_select cmp_ge cmp_gt cmp_le cmp_lt cmp_neq logical_and logical_or` | 无精确 public primitive |

补充约束：

- pointwise tensor dtype 必须一致，binding offset 必须 element-aligned，dimensions/strides 必须能用 hipDNN int metadata 表示；
- activation input/output dimensions 必须相同；
- `leaky_relu` 不能因 hipDNN enum 存在就标 supported，当前 slope setter 不能证明精确；
- 不把原始 strided/broadcast tensor 直接宣称为 hipDNN descriptor 能力；普通 per-operator case 只在 A/B/C dimensions 与 strides 完全一致时使用直接 reference，第 5.3 节的 integration gate 才通过显式展开/重排验证逻辑等价布局；
- 当前 DTK 上的 `min/max` 已由 Hygon 私有 kernel 与 hipDNN OpTensor 做 FP32/FP16 特殊值对照：unordered comparison 选择 B/right（右侧为 NaN 时结果为 NaN，左侧为 NaN 且右侧为数值时返回右值）；MIN 的 zero/zero tie 统一为 `+0`，MAX 的 zero sign 按 hipDNN 结果逐元素精确比较。该 gate 检查 NaN 分类和 zero signbit，不能用通用 finite-only comparator 替代；更换 DTK/hipDNN 后必须重新执行该 gate，不能仅凭本文沿用结论。

### 6.2 Tensor

| 算子 | 状态 | 精确 reference 与约束 |
|---|---|---|
| `reshape` | SKIP | hipDNN 无独立、精确的 reshape primitive |
| `transpose` | SKIP | hipDNN 无独立、精确的 transpose primitive |
| `slice` | SUPPORTED（受限） | FP32、rank 2/3、正 step、non-overlapping input、dense output；使用 pointer offset + `hipdnnTransformTensor` |
| `reduction` | SUPPORTED（受限） | ADD/AVG/MUL 的 FP32/FP16；axis/keep-dim descriptor 必须精确可表示 |
| `matmul` | SKIP | hipDNN 无 public MatMul primitive |

当前 header 7000/runtime 8910 的 hipDNN ReduceTensor BF16 会在 vendor CK 编译阶段稳定返回 `HIPDNN_STATUS_EXECUTION_FAILED`，且没有 per-dtype support query。因此 BF16 不属于当前 validated reduction allowlist，在 descriptor/probe 前即以“无已验证可执行 primitive”结构化 SKIP；不能把一般的 `EXECUTION_FAILED` 降级为 SKIP。未来只有在具体 DTK/hipDNN 组合实测通过后才能把 BF16 加回 allowlist。其他 reduction mode 没有精确 hipDNN primitive 时同样 SKIP。

### 6.3 Convolution

| 算子 | 已验证 baseline | reference |
|---|---|---|
| `conv_fprop` | FP32/FP16，2D rank 4，canonical NCHW/NHWC | `hipdnnConvolutionForward` |
| `conv_dgrad` | FP32/FP16，2D rank 4，canonical NCHW/NHWC | `hipdnnConvolutionBackwardData` |
| `conv_wgrad` | FP32/FP16，2D rank 4，canonical NCHW/NHWC | `hipdnnConvolutionBackwardFilter` |
| `conv_bias_relu` | FP32/FP16，2D rank 4，canonical NCHW/NHWC | convolution → OpTensor ADD → Activation ReLU sequence |

表中的 FP16 表示 DUT 与计时用 hipDNN primitive 均为 native FP16；benchmark
correctness 使用第 5.3 节“源 FP16 量化后再由 hipDNN FP32 primitive 累加”的独立
oracle。functional 与 FP32 路径保持原 dtype primitive 对照。

所有 convolution case 还必须满足：

- input/filter/output dtype 和 canonical layout 一致，不验证 mixed layout；
- pre/post padding 对称；stride、dilation、groups 和 output shape 精确；
- channel/group 关系合法，binding offset element-aligned；
- algorithm query 只是候选发现，真实 execute gate 才是最终 capability；
- `conv_bias_relu` bias 为可精确表达的 `[1,C,1,1]`，两个中间结果使用独立 reference workspace；
- BF16 不属于当前已证明 baseline，不能仅凭 enum/descriptor 宣称支持。

### 6.4 Normalization

| 算子 | 状态 | 原因/约束 |
|---|---|---|
| `layernorm` | SKIP | hipDNN public API 没有精确 LayerNorm primitive |
| `rmsnorm` | SKIP | hipDNN public API 没有精确 RMSNorm primitive |
| `batchnorm` | SUPPORTED（受限） | ForwardTraining；FP32、rank 4、dense NCHW，参数/统计量 `[1,C,1,1]`，epsilon/momentum 合法且通过 real-execute gate |
| `batchnorm_inference` | SKIP | FlagDNN 直接消费 inverse variance，hipDNN inference 消费 variance + epsilon，输入语义不等价 |

BatchNorm training reference 必须独立维护 saved mean、saved inverse variance、next running mean 和 next running variance；不能把 auxiliary output 省略或与 DUT 共享。

### 6.5 Attention

| 算子 | 状态 | 原因 |
|---|---|---|
| `sdpa` | SKIP | hipDNN 无 public MatMul/SDPA/Graph primitive 可构造精确 forward reference |
| `sdpa_backward` | SKIP | hipDNN 无精确 backward primitive/sequence |
| `sdpa_fp8` | SKIP | hipDNN 无 FP8 MatMul/SDPA/Graph primitive |
| `sdpa_fp8_backward` | SKIP | hipDNN 无 FP8 backward primitive/sequence |

四个 attention production plan 仍由 `compiler_nn.py` 生成；functional test 的职责是确认 capability reason 稳定、CTest 正确标记 SKIP。由于没有精确 reference，当前公共 benchmark manifest 不包含 attention。

## 7. CMake、CTest 与公共修改边界

`backends/hygon/CMakeLists.txt` 负责：

- 只允许 `libtriton_jit` engine；
- 发现 HIP、HCU `libtriton_jit`、Python/Triton/Torch headers/ABI；
- 构建并安装 `flagdnn_backend_hygon`；
- 安装 compiler/environment-identity Python 文件、Hygon kernel registry/override、tuning YAML、实际选中的 `libtriton_jit` SONAME 以及同一 provenance 的 JIT helper scripts；
- 让 build/install tree 镜像同一私有 JIT 相对布局，并将两者的 plugin RPATH 都精确固定为 `$ORIGIN/flagdnn/hygon`，不得出现空路径、额外 search path 或绝对 link/build path；
- 不查找 hipDNN。

`backends/hygon/validation/CMakeLists.txt` 是唯一 hipDNN 接入点。当前构建结构与 NVIDIA 对齐：

- 一个 `flagdnn_test_hygon_adapter`，包含 9 个 family runner 与 9 个 `hipdnn_*.cpp` wrapper；
- 一个独立 `flagdnn_validation_hygon_reference`，由 functional 和 benchmark 共同链接；
- 一个 `flagdnn_benchmark_hygon_adapter`，包含统一 runner、hipDNN provider 和四个 family benchmark 实现；
- functional 与 benchmark 都调用公共 per-operator suite 注册函数，不维护 Hygon 私有 operator shape/case 清单。

CTest 注册为：

- 61 个 `functional.hygon.<op>`；
- 57 个 `benchmark.hygon.<op>`；
- `integration.hygon.jit_candidate_compatibility_contract`；
- `integration.hygon.jit_global_state_contract`；
- `integration.hygon.convolution_validation_static_contract`；
- `integration.hygon.cmake_configuration_contract`；
- `integration.hygon.validation_contract`；
- `integration.hygon.normalization_stability`；
- `integration.hygon.dependency_boundary`；
- `integration.hygon.compiler_contract`；
- `integration.hygon.reference_dependency_boundary`；
- `integration.hygon.jit`；
- `integration.hygon.pointwise_smoke`；
- `integration.hygon.installed_consumer`；
- `integration.hygon.runtime`；
- `integration.hygon.graph`。

重新配置后的 catalog 算术为 12 core + 2 benchmark catalog + 61 functional + 14 Hygon integration + 57 benchmark，共 146 项。这个数字来自 `build/hygon` 的 `ctest -N` 注册面，只是结构快照；attention 只应出现四个 functional 名称，不应出现 `benchmark.hygon.sdpa*`。

`tools/run_tests.py` 仍只把 functional 与 benchmark 计入 118 个 operator-suite 状态，但在 Hygon 上默认先执行强制 preflight：12 个 core、2 个 benchmark catalog 和上面的 14 个 `integration.hygon.*`，共 28 项。runner 会先读取 CTest JSON catalog；缺少任一必需名称、任一项失败、超时或整项 SKIP 都使最终退出码非零。可用 `--no-preflight` 做局部调试，但不能用于正式验收。

这些注册项承担的不只是 smoke：

- `integration.hygon.runtime` 对 artifact 做 Graph/request 语义复核，检查 function dense/strided 选择、argument ABI、`n_elements`、完整 DIM/stride 常量、`OP_KIND`、bit-exact `ALPHA`、`BLOCK_SIZE`、grid、workspace alignment/range/DAG、materialized source 与 source identity；并注册 node count、source hash、argument ABI、scratch overlap、workspace alignment、materialized source、op kind、alpha、element count、stride、block size 和 grid 的 mutation rejection case。
- `integration.hygon.jit` 绑定 CMake 选中的 JIT image、scripts、Python module/root/hash 和环境 identity，并验证污染 `gen_ssig` module 的子进程必须在使用前失败。
- `integration.hygon.installed_consumer` 自动安装到隔离 SDK，清空 `FLAGDNN_BACKEND_ROOT`、kernel/tuning root、全部 Hygon JIT/compiler override、legacy JIT selector、compiler entry override、`PYTHONPATH`/`PYTHONHOME`，再检查安装资源并使用安装态 `FlagDNNConfig.cmake` 构建 C/C++ consumer。辅助 identify 必须报告 SDK 内的实际 compiler entry、Hygon provider、environment JSON、私有 JIT 与 canonical/private 两组脚本；真实 GPU `installed.hygon_add` 通过 ABI 无关的 dynamic link-map 断言实际映射 JIT 位于 SDK 私有目录，而脚本实际解析继续由同 ABI 编译的 plugin/JIT 路径和 identity 校验覆盖，不能从 consumer 硬编码 C++ mangled symbol。Add consumer 使用未显式指定 backend/compiler 的默认 `flagdnn::Handle`，走 Graph、autotune、安装态 compiler/JIT，并以 hipDNN OpTensor Add 对照；它是默认 preflight gate，不再只是手工示例。
- `core.run_tests_contract` 固化 runner 的 catalog、结构化 summary、benchmark JSON、SKIP/accounting 和 timeout contract。

runner 对每个 Hygon PASS/SKIP suite 要求且只允许一个 accounting summary，并核对 `executed + skipped == cases`、结构化 `skip_records` 数量以及 benchmark record group。benchmark executed case 必须且只能有 `{flagdnn, hipdnn}` 两个 provider；样本必须为正有限值，median/p90 按 nearest-rank 重算，重复或 schema 不完整均失败。CTest 状态与 summary 状态必须一致，SKIP benchmark 不得产生 timing。

runner 的 summary schema 为 v2，顶层必须包含 `overall_status` 与 `exit_code`。
显式 `--output` 时，runner 在完整参数校验前先以同目录临时文件加
`os.replace` 原子发布 `running`，正常或已知失败路径再原子发布终态，避免旧的
successful summary 形成 false green。空 `--ops`、纯逗号、空/纯注释 op-list 和
最终零 invocation 都 fail closed；case/skip 归属按最长完整 manifest operator
匹配。accounting marker 必须精确等于当前 operator 与 suite；唯一例外是与 NVIDIA
架构一致的 `conv_fprop/conv_dgrad/conv_wgrad` functional 共用
`FLAGDNN_CONVOLUTION_FUNCTIONAL` family marker，benchmark 仍逐算子。不能让
`add` 接受 `add_square` 的记录。

设备选择由测试命令显式控制。本轮正式测试统一传入 `--device 0`，使 operator
suites 与 preflight 在 GPU 0 上运行；这只是本次测试运行范围，不定义 Hygon
backend 的设备架构或平台验收契约。省略 `--device` 时 runner 保留调用进程已有的
设备可见性，不能把这种行为解释成隐式选择 GPU 0。

本轮适配涉及的公共修改白名单必须按职责而不是按 Hygon 特判理解：

- 根 `CMakeLists.txt` 与中立 `cmake/FlagDNNDefaultBackend.cmake`、default-backend
  contract：只定义单/多 backend 的通用默认选择和错误处理；
- `src/runtime` 的 compiler client/cache/context contract 与
  `compiler/flagdnn_codegen/main.py` identity protocol：只提供所有 backend 可用的
  compiler identity、dependency snapshot、typed executable-unavailable 和严格
  cache fail-closed 语义；
- `tools/build.sh`、`tools/install.sh`、`tools/run_tests.py`：通用 backend build-tree
  选择、安装 manifest 解析、CTest catalog/SKIP/accounting/device contract；
- `tests/CMakeLists.txt`、`tests/core/**` 和 installed-consumer contract：验证默认
  backend、compiler/cache、安装 discovery 和 C/C++ 公共 API；其中 Hygon GPU
  consumer 可以是平台 adapter，但不能改变公共 API 语义；
- `cmake/VerifyNoVendorDependencies.cmake`、
  `cmake/VerifyPerOperatorTestLayout.cmake`、run-tests/default-backend 等中立 contract
  脚本：执行 direct dependency、平台泄漏、catalog 与配置回归检查。

以上公共修改必须保持 backend-neutral，并由 core/NVIDIA 不回归门禁约束；它们不是
在公共 Graph/lowering 中加入 Hygon 分支的许可。禁止为了 Hygon 修改
`benchmark/CMakeLists.txt`、公共 case 语义、Graph lowering 或任何
`backends/nvidia/**` 文件，NVIDIA staged/unstaged diff 都必须为空。`tools/tests`
不应存在；runner 验证通过现有 CTest/core contract 完成。

## 8. 后续新增或优化算子的固定流程

每个新增算子或新 case 按以下顺序实施：

1. 确认公共 Graph API、Graph IR 和 platform-neutral case 已存在；缺少公共 contract 时单独设计，不加入 Hygon 分支。
2. 在 `compiler.py`、`compiler_tensor.py` 或 `compiler_nn.py` 中增加严格 schema、dtype、shape、stride、attribute 校验。
3. 设计 `NodePlan`、stage DAG、workspace 和 runtime scalar ABI；多阶段 output/dependency 必须可在 artifact 中审计。
4. 优先复用 common kernel；新增 Hygon override 时提供 compile/correctness/performance 证据和无隐式回退测试。
5. 在 `gfx936` 上完成 Triton compile、libtriton_jit load、caller-stream launch、边界 shape、非连续 layout 和 default/autotune smoke。
6. 检查 hipDNN public primitive 是否能精确表达同一语义：能则实现独立 C++ plan；不能则默认实现稳定 capability reason 和结构化 SKIP。新增 CPU functional oracle 必须显式评审并加入有限 allowlist，且不得改变 benchmark capability。
7. 同时更新单一 functional adapter 的对应 family runner/reference hook、单一 benchmark adapter 的 hipDNN provider（若公共 manifest 有 benchmark）和本文 capability 矩阵；shape/dtype/tolerance/`BenchmarkConfig` 仍只能来自公共目录。
8. 运行单算子 functional/benchmark、依赖门禁、安装 discovery，再运行全量 runner。
9. 检查 NVIDIA staged/unstaged 均为零差异，公共 platform-leak gate 通过。

任何阶段都不得用 MIOpen、BLAS、自定义 device oracle 或 FlagDNN kernel 填补 hipDNN reference 空白。host 计算仅允许位于 `reference/cpu`、仅供显式 allowlist 的 functional test 使用，且不得进入 benchmark。

## 9. 正式构建、安装与测试命令

### 9.1 环境与默认值

`tools/build.sh` 的默认配置已经是：

- build type：`Release`；
- generator：`Ninja`；
- engine：`libtriton_jit`；
- tests：ON；
- benchmarks：ON；
- warnings-as-errors：ON；
- jobs：`min(nproc, 8)`；
- build directory：`build/<backend>`。

默认 backend policy 为 `auto`：只构建 Hygon 时选择 Hygon；multi-backend 列表包含 NVIDIA 时优先 NVIDIA，否则选择第一个 backend。显式 `--default-backend NAME` 也可选择不在本构建列表中的外部 plugin；运行时必须通过 `FLAGDNN_BACKEND_PATH` 提供该 plugin，否则 Handle 创建会 fail-closed。

因此常规 Hygon 工作流只需要：

```bash
./tools/build.sh --backends hygon
./tools/install.sh
```

确保 codegen Python 能导入 Torch、Triton、PyYAML，并能发现 HCU 版 `libtriton_jit`。Hygon resolver 默认检查仓库同级的 `${PROJECT_SOURCE_DIR}/../libtriton_jit`；自定义位置只使用 Hygon 命名空间变量，例如：

```bash
./tools/build.sh --backends hygon -- \
  -DFLAGDNN_HYGON_TRITON_JIT_ROOT=/path/to/hcu/libtriton_jit
```

也可以提供 `FLAGDNN_HYGON_TRITON_JIT_DIR` 指向 HCU package/config 布局。不要为 Hygon 设置全局 `TritonJIT_DIR` 或 `LIBTRITON_JIT_ROOT`；Hygon resolver 有意不读取它们，以免覆盖 NVIDIA package 选择。所选 HCU JIT 布局必须同时提供 exported target/header/library、`standalone_compile.py` 和 `gen_ssig.py`，任一缺失或 provenance 混用都在配置阶段失败。环境 identity helper 由 FlagDNN 自身安装并校验，它不是外部 JIT package 必须提供的 provenance 输入。

无参数 `install.sh` 会在未显式设置 `FLAGDNN_BUILD_DIR`/`FLAGDNN_BACKENDS` 时，从 `build/*/CMakeCache.txt` 选择唯一已配置的构建树。完成一次上述 Hygon 构建且没有其他 configured build tree 时，它会自动选择 `build/hygon`，无需导出 `FLAGDNN_BACKENDS`。如果存在多个 configured build tree，脚本会 fail closed，此时显式使用 `--build-dir build/hygon`（或设置明确的 backend）消除歧义；如果一个 configured tree 都不存在，其历史默认仍是 `build/nvidia`，不能把这种情况当成 Hygon 安装。

### 9.2 构建

正式简化命令：

```bash
./tools/build.sh --backends hygon
```

预期构建目录为 `build/hygon`，并同时构建 production plugin、functional targets 和 benchmark targets。

默认 `Ninja` 是 single-config generator，`--build-type` 通过
`CMAKE_BUILD_TYPE` 生效。若显式使用 `Ninja Multi-Config`、Xcode 或 Visual
Studio generator，脚本不会再伪设 `CMAKE_BUILD_TYPE`，而会把同一个
`--build-type` 作为 `cmake --build --config` 的配置名传递。例如：

```bash
./tools/build.sh --backends hygon \
  --generator "Ninja Multi-Config" --build-type Debug
./tools/install.sh --build-dir build/hygon --config Debug
```

multi-config 产物位于对应配置子目录；`install.sh` 不再用 single-config 的
`src/libflagdnn.so` 路径提前误判，而由 `cmake --install --config` 对所选配置
执行并严格报告缺失产物。

### 9.3 安装

未显式设置安装选择环境变量、且只有 `build/hygon` 一个 configured build tree 时直接执行：

```bash
./tools/install.sh
```

唯一 configured build tree 为 Hygon 时，默认安装前缀是 `build/hygon/install`。存在多个构建树时显式执行：

```bash
./tools/install.sh --build-dir build/hygon
```

安装后至少检查：

```bash
test -e build/hygon/install/lib/libflagdnn_backend_hygon.so.2
find build/hygon/install/lib/flagdnn/hygon -maxdepth 1 \
  -type f -name 'libtriton_jit.so*' -print
test -f build/hygon/install/lib/flagdnn/share/triton_jit/scripts/standalone_compile.py
test -f build/hygon/install/lib/flagdnn/share/triton_jit/scripts/gen_ssig.py
test -f build/hygon/install/lib/flagdnn/share/triton_jit/scripts/flagdnn_python_environment_identity.py
test -f build/hygon/install/share/flagdnn/compiler/flagdnn_codegen/main.py
test -f build/hygon/install/share/triton_jit/scripts/standalone_compile.py
test -f build/hygon/install/share/triton_jit/scripts/gen_ssig.py
test -f build/hygon/install/share/triton_jit/scripts/flagdnn_python_environment_identity.py
test -f build/hygon/install/share/flagdnn/backends/hygon/compiler.py
test -f build/hygon/install/share/flagdnn/backends/hygon/compiler_tensor.py
test -f build/hygon/install/share/flagdnn/backends/hygon/compiler_nn.py
test -f build/hygon/install/share/flagdnn/backends/hygon/compiler_identity.py
test -f build/hygon/install/share/flagdnn/backends/hygon/python_environment_identity.py
test -f build/hygon/install/share/flagdnn/backends/hygon/flagdnn_hygon_compiler_environment.json
test -f build/hygon/install/share/flagdnn/backends/hygon/tuning/common.yaml
test -f build/hygon/install/share/flagdnn/backends/hygon/kernels/registry.json
test -f build/hygon/install/share/flagdnn/backends/hygon/kernels/binary_minmax.py
```

实际 JIT 文件名以 package exported SONAME 为准，由配置与 installed-consumer contract 精确核对；上面的 `find` 只用于人工列出结果，不能把未版本化 `libtriton_jit.so` symlink 当作契约。还应使用 `readelf -d build/hygon/backends/hygon/libflagdnn_backend_hygon.so.2` 和 `readelf -d build/hygon/install/lib/libflagdnn_backend_hygon.so.2`，确认构建态、安装态 `DT_RPATH` 都精确且只含 `$ORIGIN/flagdnn/hygon`；并核对构建态 SONAME 实际位于 `build/hygon/backends/hygon/flagdnn/hygon/`，不存在空 RPATH 元素、plugin-adjacent JIT 或绝对构建路径。

安装态完整 Graph/JIT/autotune/GPU consumer 已是自动 CTest gate，正式检查直接运行：

```bash
ctest --test-dir build/hygon -j1 --output-on-failure \
  -R '^integration\.hygon\.installed_consumer$'
```

该 gate 自己创建隔离 SDK、执行安装，在清理后的子进程环境中依次执行 installed identify、configure、build 和 CTest，并运行 `installed.c`、`installed.cpp` 与 `installed.hygon_add`。它会校验 build/install 两种 RPATH、私有 JIT bytes、canonical/private helper 哈希、identity dependency 实际路径，以及 ABI 无关的运行时 mapped JIT 路径；实际 graph build 同时覆盖 JIT 内部脚本解析与 identity。`installed.hygon_add` 必须从默认 `Handle()` 选择 Hygon，使用安装资源生成且仅生成一份 manifest/source/autotune selection，在真实 HIP stream 上执行 Add 并与 hipDNN OpTensor Add 比较。手工配置 `tests/core/installed_consumer` 只用于失败定位，不替代这个自动 gate。

validation target 不安装，测试仍从 `build/hygon` 运行。

### 9.4 单算子 smoke

先覆盖 supported、mixed/limited 和 all-SKIP family：

```bash
ctest --test-dir build/hygon -j1 -V -R '^functional\.hygon\.add$'
ctest --test-dir build/hygon -j1 -V -R '^functional\.hygon\.slice$'
ctest --test-dir build/hygon -j1 -V -R '^functional\.hygon\.conv_fprop$'
ctest --test-dir build/hygon -j1 -V -R '^functional\.hygon\.batchnorm$'
ctest --test-dir build/hygon -j1 -V -R '^functional\.hygon\.sdpa$'
```

四个 CPU-oracle pointwise 算子必须运行逐算子 functional；production compile/launch smoke 仍作为额外链路门禁：

```bash
ctest --test-dir build/hygon -j1 --output-on-failure \
  -R '^functional\.hygon\.(div|pow|mod|cmp_eq)$'
ctest --test-dir build/hygon -j1 --output-on-failure \
  -R '^integration\.hygon\.pointwise_smoke$'
```

再检查 catalog 与不属于 operator runner 的 core/integration 边界：

```bash
ctest --test-dir build/hygon -N
ctest --test-dir build/hygon -j1 --output-on-failure \
  -R '^(core\.|benchmark\.catalog_|integration\.hygon\.)'
```

重新配置后 `build/hygon` 应由 `ctest -N` 列出 146 项；最后一条命令覆盖 12 个 core、2 个 benchmark catalog contract 和 14 个 Hygon integration，共 28 项。`ctest -N` 不执行测试。正式 Hygon `tools/run_tests.py` 会自动先运行同一 preflight；单独命令仍便于定位。

### 9.5 全量 functional + benchmark

正式全量命令：

```bash
python3 tools/run_tests.py \
  --platform hygon \
  --ops all \
  --suites all \
  --device 0 \
  --timeout 3600 \
  --output build/hygon/hygon-all-final-summary.json
```

正式命令显式使用 `--device 0`，不依赖调用环境的设备 mask 或隐式默认设备。
需要把同一测试运行定向到其他单卡时可改为 `--device N`；这不改变 Hygon backend
的架构契约。该命令先执行 28 项 preflight，随后调度 61 个 functional 和 57 个
benchmark，共 118 个 operator-suite invocation。验收时要求：

- `schema_version == 2`、`overall_status == "passed"`、`exit_code == 0`；
- `status_counts.failed == 0`；
- `status_counts.timeout == 0`；
- `status_counts.not_found == 0`；
- `preflight.status == "passed"`，且 `missing_tests` 为空；
- `status_counts.passed + status_counts.skipped == 118`；
- `coverage.benchmark_record_errors == 0`、`coverage.hipdnn_skip_record_errors == 0`、`coverage.hygon_case_accounting_errors == 0`；
- `performance.metric == "hipdnn_median_us/flagdnn_median_us"`，且所有可比 case 都形成完整 FlagDNN/hipDNN pair；当前不设置最低 speedup 门禁，性能只做回归记录；
- 四个 attention functional 明确 SKIP，且不存在 attention benchmark；
- mixed capability target 即使 CTest 总状态为 passed，也必须保留逐 case `skip_records`；
- 每个 benchmark supported case 同时包含 `flagdnn` 与 `hipdnn` 记录；
- unsupported case 不出现 FlagDNN-only reference latency/speedup。

`--no-preflight` 只用于局部调试，不能用于正式验收。

### 9.6 当前正式验收记录（2026-08-13）

该记录早于四个 CPU functional oracle 的引入，以下 passed/skipped 数量仅是历史
基线，不能代表当前代码；当前变更后的正式全量数字必须重新执行第 9.5 节生成。

以下记录来自目标 Hygon/DTK 环境的正式单卡全量运行。命令没有显式传入性能
阈值，因此证明构建、功能、配对、SKIP 与 accounting 闭合，并记录当前性能；
不把 speedup 用作通过条件。按当前测试范围复现时使用以下单卡命令：

```bash
./tools/build.sh --backends hygon
./tools/install.sh
python3 tools/run_tests.py \
  --platform hygon --ops all --suites all --device 0 \
  --timeout 3600 \
  --output build/hygon/hygon-all-final-summary.json
```

本次实测结果如下：

- build 与 install 均成功；默认安装目录为 `build/hygon/install`；
- 当前 CTest catalog 为 146 项：61 functional、57 benchmark、12 core、2 benchmark catalog 和 14 Hygon integration；
- summary 为 schema v2，`overall_status=passed`、`exit_code=0`；
- 28/28 preflight 全部通过，耗时 1148.450 秒，`missing_tests` 与 `errors` 均为空；
- 118 个 operator-suite invocation 全部闭合：46 passed、72 skipped、0 failed、0 timeout、0 not_found；其中 functional 为 24 passed/37 skipped，benchmark 为 22 passed/35 skipped；
- benchmark 可比目录声明 22 个算子、368 个 case pair；实测观察 368/368 pair、736 条 provider record，`missing_case_count=0`、`extra_pair_case_count=0`，每个 pair 都同时包含 `flagdnn` 与 `hipdnn`，没有单边 latency；
- `performance.threshold=null`，368 个 case 全部通过覆盖/记录 gate。当前 kernel speedup（`hipdnn_median_us/flagdnn_median_us`）最小值为 0.915433，中位数为 2.875924，几何均值为 2.926073；这些值只作为本次回归基线，不作为本阶段通过/失败条件；
- 共保留 1614 条逐 case hipDNN SKIP 记录；`benchmark_record_errors`、`hipdnn_skip_record_errors` 和 `hygon_case_accounting_errors` 均为 0；
- 四个 attention functional 均为显式 SKIP，未注册 attention benchmark；
- qualification identity 为 hipDNN header 7000、runtime 8910、`gfx936:sramecc+:xnack-`；完整日志未出现 VM fault、进程崩溃、CTest failure 或 timeout 标记。

本次正式证据为 `build/hygon/hygon-all-final-summary.json`；后续代码或环境变化后
必须按第 9.5 节重新生成。72 个 SKIP 只表示当前 public hipDNN primitive
无法提供精确 oracle，不能解释成相应 production kernel 已得到数值正确性证明。

静态复审还保留一个不阻塞首批 10 算子交付的 P2 覆盖缺口：当前
`div`、`pow`、`mod`、`cmp_eq` 已由逐算子 functional 使用 CPU oracle 实际打通
production compiler/artifact/JIT/autotune/caller-stream 链路；`integration.hygon.pointwise_smoke`
仍覆盖首批 10 算子的补充链路（`add_square` 由独立 composite integration 覆盖）。
其余在 hipDNN capability 阶段整体 SKIP、且未列入 CPU allowlist 的算子仍不会在
functional/benchmark 中构建 DUT。后续扩算子时应增加逐算子 production-chain
smoke/静态对账，或单独评审 CPU functional oracle；不得引入 MIOpen、BLAS 或
自定义 device fallback oracle。

未指定 `--ops` 时，runner 的开发默认集合仍是：

```text
add sub mul div pow max min mod add_square cmp_eq
```

该 10 算子集合适合快速开发回归，但不能替代 `--ops all`。其中 `div`、`pow`、`mod`、`cmp_eq` 没有 hipDNN 数值 primitive：functional 使用 CPU semantic oracle 并可形成相对该 oracle 的数值 PASS/FAIL，benchmark 按严格 contract SKIP，不能输出性能结论。

## 10. Definition of Done

Hygon 全量适配只有同时满足以下条件才能交付：

- `./tools/build.sh --backends hygon` 在目标 DTK/HCU 环境完成；
- `./tools/install.sh` 安装完整 plugin/compiler/tuning 资源，安装树可独立发现 Hygon provider；
- 自动 `integration.hygon.installed_consumer` 通过，隔离安装态 C/C++ consumer 和真实 GPU Add consumer 均成功；
- 61 functional、57 benchmark、14 Hygon integration、12 core、2 benchmark catalog 和总计 146 个 CTest catalog 与公共 manifest 一致；
- public Graph 经 Hygon compiler、artifact、libtriton_jit、Triton HCU kernel、autotune 和 caller HIP stream 的真实链路运行；
- artifact request/Graph/source/ABI/tuning/workspace mutation gate 全部拒绝受损输入；
- pointwise、tensor、convolution、normalization、attention 的 default/autotune plan 都满足 stage ABI 和 workspace contract；
- `div`、`pow`、`mod`、`cmp_eq` 的逐算子 functional 使用 CPU oracle 验证数值并真实执行 compiler/JIT/autotune/caller-stream；其 benchmark 必须 SKIP；
- `min/max` 在当前 qualification DTK 上以 hipDNN 对照 FP32/FP16 NaN 分类和 signed-zero，并保留 Hygon 私有语义 override；
- 所有 hipDNN-supported case 完成 FlagDNN-vs-hipDNN 数值比较；
- 除四个显式 CPU functional allowlist 算子外，所有 hipDNN-unsupported case 输出稳定、可审计的 SKIP，且没有其他 oracle fallback；
- benchmark 完全使用公共 case/`BenchmarkConfig`，先正确性门禁，再按第 5.3 节 HIP Graph batch 口径计时，并只报告有 hipDNN reference 的 case；
- production direct dependency 中没有 hipDNN/MIOpen/BLAS；validation 的 vendor 接口只直接调用 HIP 与 hipDNN，并静态链接平台无关 CPU oracle；vendor transitive closure 被如实报告；
- `execute` 不 compile、allocate、autotune 或 synchronize；
- compiler/runtime identity 与 cache 能隔离 kernel、tuning、provider、实际 JIT image/scripts、Python/Triton/Torch module tree 和环境变化；
- compiler cache 仅在 compiler spawn 精确返回 `ENOENT` 时允许按已验证 active identity 离线命中，其他 identity/protocol/timeout/tempfail 全部硬失败；cache miss 还必须在 compiler request 前后核对依赖快照，并在临时 artifact 校验后、原子发布前强制刷新 identity，拒绝端点可观测的 dependency 漂移；宿主同时保证编译事务期间资源不可变；
- Hygon CMake 只使用 `FLAGDNN_HYGON_TRITON_JIT_*` 变量，build/install resource 全部来自同一 JIT provenance、镜像同一私有相对布局，两个 plugin 的 `DT_RPATH` 都精确为 `$ORIGIN/flagdnn/hygon`；
- Hygon JIT 对非 HCU 或被改写的 `TRITON_JIT_BACKEND` fail closed，应用遵守 process environment/Python 全局状态的非并发修改边界；
- CUDA/HCU JIT 相同 SONAME 的限制被遵守：NVIDIA 与 Hygon qualification 使用独立进程，不声称同一进程支持两套 JIT；
- `backends/nvidia/**` staged/unstaged 均无差异；
- platform-neutral dependency/layout gates 通过；
- 正式 runner 的 suite/accounting/benchmark JSON 与 SKIP 记录相互闭合，不能以缺记录、重复记录或错误 CTest 状态形成 false green；
- 在声明“全量通过”前，必须按本次单卡测试范围实际执行第 9.5 节 `tools/run_tests.py --platform hygon --ops all --suites all --device 0`，并满足 preflight、operator suite、可比 case 配对和零功能失败标准；不设置最低 speedup 阈值，默认 10 算子基线也不能替代该全量检查。该命令只证明 GPU 0 上的本次测试结果，不扩展为平台设备架构结论。

对 hipDNN 无 reference 且未列入 CPU functional allowlist 的算子，结构化 SKIP 只代表在限定 reference 下无法做数值对比。四个 allowlist 算子的 Hygon PASS 证明其输出与当前 CPU semantic oracle 一致；`reference/tests` 的四项 NVIDIA/cuDNN CTest 另行证明该 oracle 的对应语义。交付报告必须记录这两层测试及其实际环境，不能用其中一层替代另一层。

## 11. 提交前审计清单

```bash
git diff --exit-code -- backends/nvidia
git diff --cached --exit-code -- backends/nvidia
git diff --check
cmake -DSOURCE_ROOT="$PWD" \
  -P cmake/VerifyPerOperatorTestLayout.cmake
test ! -e tools/tests
```

同时人工确认：

- Hygon production source 没有 include/import `backends/nvidia` 或 validation；
- production CMake 没有查找/链接 hipDNN、MIOpen、hipBLAS、rocBLAS；
- validation 没有直接 include/link/call MIOpen/BLAS；
- `reference/cpu` 没有 include/link 任一平台 SDK，四项 `reference/tests` NVIDIA/cuDNN 独立校验已执行并记录；
- Hygon JIT CMake selector 只使用 `FLAGDNN_HYGON_TRITON_JIT_*`，没有复用全局 `TritonJIT_DIR`/`LIBTRITON_JIT_ROOT`；
- 安装态 compiler、kernel/tuning、JIT library/scripts 与 environment identity 都来自同一 provenance；
- HCU JIT 在 build/install tree 中分别位于 plugin 的 `flagdnn/hygon` 私有子目录，两个 plugin 的 RPATH 都精确为 `$ORIGIN/flagdnn/hygon`，JIT 实际 script directory 指向各自树中的私有资源；
- RPATH 没有空元素、当前工作目录、plugin 根目录或绝对 build/DTK/libtriton_jit 路径；
- compiler cache 没有把 malformed/nonzero/timeout/tempfail 当作离线命中，只有 spawn 精确 `ENOENT` 进入 typed unavailable 分支；
- autotune 只接受类型化 `HygonError` 的 HIP 白名单，或上游普通 `std::runtime_error` 对应白名单的完整固定格式；未知异常不得作为 candidate 拒绝处理；
- embedding 应用在 compiler identity/JIT 初始化和 build 期间不并发修改 process environment、Python `os.environ` 或 `sys.path`；
- 外部不同 UID binding 不受 Hygon 特有的 pointer-range blanket rejection，仍遵循公共/NVIDIA alias 语义；
- NVIDIA 与 Hygon 测试按进程隔离，不假设相同 SONAME 的 CUDA/HCU JIT 可以在同一进程共存；
- capability matrix、CTest catalog 和本文同步；
- `integration.hygon.installed_consumer` 属于默认 28 项 preflight，不依赖手工 consumer 步骤；
- 交付测试命令显式使用 `--device 0`，报告只据此陈述本次单卡运行结果，不把它表述成 runner 默认设备行为或平台设备架构结论；
- 没有把 vendor transitive closure 误写成 FlagDNN reference，也没有声称进程绝不会加载 vendor 内部依赖；
- 没有提前填写尚未实际执行的全量 passed/skipped 数字。

## 12. 主要风险与处理

| 风险 | 处理 |
|---|---|
| hipDNN header/runtime 版本不同 | 同时记录两者；以 descriptor/build/execute probe 为准 |
| enum 存在但 runtime 不支持 | real-execute capability gate；正式执行失败为 FAIL |
| 大量 SKIP 掩盖验证空白 | 单列 skip_records、覆盖矩阵和缺少 oracle 的算子 |
| 非 `NOT_SUPPORTED` 错误被伪装为 capability SKIP | 严格 native status 分类；环境、参数、执行与版本错误一律 FAIL |
| Hygon 误用 NVIDIA 私有实现 | Hygon 自有目录；双重 NVIDIA zero-diff gate |
| common kernel 在 HCU 编译但不正确 | 所有有 reference 的 case 比较 selected variant；无 reference 时明确空白 |
| CPU semantic oracle 自身实现错误 | allowlist 限定为四个 functional 算子；在 `reference/tests` 中逐算子与 cuDNN 独立对照，qualification 重跑并记录结果；不用于性能结论 |
| DTK 升级改变 MIN/MAX 特殊值语义 | 每个 qualification stack 重跑 FP32/FP16 hipDNN NaN/signed-zero gate，私有 override 与 identity 同步失效 |
| CUDA warp32/TF32 假设泄漏 | wave64 tuning、严格 IEEE matmul、HCU 实机 compile/launch |
| runtime scalar 被错误 constexpr 特化 | 明确 scalar_i32/scalar_f32 ABI，校验 full/runtime signature |
| 多阶段 workspace 重叠或依赖错误 | 统一 workspace pack、合成 UID、artifact range/DAG 校验 |
| libtriton_jit library/scripts/include 混用 | Hygon namespaced resolver 要求单一 build/install provenance，混合 cache fail closed |
| libtriton_jit ABI/环境不匹配 | mapped image + scripts + Python/Triton/Torch tree identity，helper 使用前 fail closed |
| compiler 暂态故障被旧 cache 掩盖 | 仅精确 spawn `ENOENT` 可走 typed offline hit；malformed/nonzero/timeout/tempfail 硬失败 |
| autotune 误吞 compiler/JIT/runtime 回归 | 只接受类型化 `HygonError` 的六个明确 device-incompatible HIP status；其他异常硬失败 |
| Python/C 进程全局环境被并发改写 | 初始化前后 fail-closed 校验；应用负责不并发修改 env/`os.environ`/`sys.path`，不同 Triton backend 使用独立进程 |
| CUDA/HCU JIT 相同 SONAME 被同进程复用 | 两个平台用独立进程；runtime identity 在使用前拒绝错误 image |
| CUDA/HCU JIT 同名安装相互覆盖 | HCU library 使用 build/install 各自的 `flagdnn/hygon` 私有目录；两态 plugin 都固定精确私有 RPATH |
| RPATH 空元素或绝对路径引入 cwd/build-tree 污染 | build/install 镜像私有布局；两者只允许精确 `$ORIGIN/flagdnn/hygon` |
| 安装包遗漏或从源码树偷取 compiler/kernel/JIT 资源 | installed-consumer 清空所有资源 override 和 Python path，从隔离 SDK identify/configure/build/test，并核对实际路径与哈希 |
| vendor closure 被误认成 reference | direct dependency 与 runtime closure 分开审计和报告 |
| benchmark 在 unsupported case 给出误导结论 | capability 先行；无 hipDNN reference 就 SKIP，不计时 |
| suite/benchmark/SKIP 记录缺失导致 false green | runner 强制单一 accounting summary、provider pair、统计重算、CTest 状态一致性 |

本指南是后续 Hygon 扩算子和优化的执行 contract。任何偏离上述 reference、依赖、workspace、ABI、catalog 或 NVIDIA 边界的变更，都必须先更新设计并单独评审。
