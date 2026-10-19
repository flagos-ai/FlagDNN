# FlagDNN 功能与性能验证

## 1. 一个算子清单，两种平台无关 workload

`cmake/Operators.cmake` 定义公共功能与性能算子清单。平台可以通过
`backends/<platform>/validation/benchmark/additional_operators.txt` 为公共功能算子
补充性能入口；目录契约同时检查这些平台清单。功能与性能入口分别位于：

```text
tests/test_<op>.cpp
benchmark/test_<op>.cpp
```

顶层入口是薄 `main`：功能入口读取 `tests/common` case 并调用功能 runner contract；
性能入口读取 `benchmark/common` workload 并调用 benchmark runner contract。入口不
包含 CUDA、cuDNN、CANN、ACL 或任何其他平台 SDK。

`cmake/VerifyPerOperatorTestLayout.cmake` 检查入口数量、目录边界、平台依赖隔离，并
禁止重新出现 Python 功能测试、Python benchmark 和旧平台目录。
`tests/core/run_tests_contract.py` 是公共 tests 目录的显式例外：它只验证同为 Python
的统一测试 runner，不定义算子 case，也不执行算子。平台目录还可包含编译器的
主机端契约，例如 `backends/nvidia/validation/compiler_contract.py`；
这些契约不替代 C++ 设备功能测试。

## 2. 平台无关目录

```text
tests/
├── CMakeLists.txt          # common/core target 与功能 suite 注册函数
├── test_<op>.cpp           # 每个算子一个薄入口
├── common/                 # case、FlagDNN Graph builder、runner contract
└── core/                   # C/C++ API、ABI、runtime/backend/安装 contract

benchmark/
├── CMakeLists.txt          # workload target 与性能 suite 注册函数
├── test_<op>.cpp           # 每个算子一个薄入口
├── common/                 # case/workload、provider 与 runner contract
└── result.schema.json      # 稳定 JSONL 输出格式

reference/
├── cpu/                    # 无平台 SDK 依赖的受控 CPU 功能语义 oracle
└── tests/                  # 使用 cuDNN 独立验证 CPU oracle
```

`tests/common` 只定义 shape、dtype、layout、tolerance、autotune case 和被测 FlagDNN
Graph；`benchmark/common` 只定义 workload、采样配置及 provider contract。两者都不
决定 reference SDK、device memory、stream 或 event timer。

## 3. 平台验证只占一个目录

所有平台私有验证代码与生产 backend 同属一个平台根目录：

```text
backends/nvidia/validation/
├── CMakeLists.txt          # NVIDIA 唯一验证装配入口
├── cuda_driver.hpp         # 功能/性能共享的 context/stream/buffer/event
├── tensor_io.cpp/.hpp   # 功能/性能共享的布局、编解码、padding 校验
├── functional/             # cuDNN Graph/legacy reference、功能 runner
└── benchmark/              # cuDNN provider、正确性比较、GPU event 计时、JSONL
```

不存在 `tests/platforms`、`benchmark/platforms` 或 `test_support`。适配 Ascend 时只需
增加或修改 `backends/ascend/validation/**`；平台目录内部可以按功能/性能分文件，但
开发者只有一个平台接入点。

NVIDIA 的功能和性能比较必须执行真实 cuDNN 算子。优先使用 cuDNN Frontend
Graph；没有合适的 Graph 执行计划时，使用 cuDNN 共享库中的实际 API，例如
`cudnnReduceTensor`、`cudnnTransformTensor`、`cudnnOpTensor` 和空间采样器。
不得将 CPU 数学公式或自行实现的 GPU kernel 标记为 cuDNN 对照。
双方使用相同的量化输入，并按各自 SDK 的存储格式编码；输出和 workspace 独立
分配。输出通过校验后才计时。

功能和性能的共享范围以当前合格 NVIDIA 栈（H100、cuDNN 9.24、Frontend 1.26）
实际可执行的 cuDNN 对照为准。先使用 Frontend，缺少对应执行计划时检查原生
cuDNN API；两者均不支持的算子或配置从 `cmake/Operators.cmake`、
`tests/common` 及功能/性能入口移除，所有平台使用同一份精简后的用例。
不能只在 NVIDIA runner 内过滤，再让其他平台用未经 cuDNN 验证的 CPU 公式补齐。
生产 Graph API、编译器和 Triton kernel 不因测试范围收缩而删除或修改。

2026-10-19 清理的范围及依据：

| 范围 | 处理 |
| --- | --- |
| GenStats、RNG、RoPE 前向/反向、MoE 反向 | 无独立可执行 cuDNN 对照，删除共享功能/性能测试、对应 CPU 实现及平台适配调用；保留 BN Finalize 和 MoE 前向 |
| 精确 INT32 逐元素算术、比较、选择和 AddSquare | INT32 compute 无计划；FLOAT compute 存在整数舍入/精度差异，删除该测试范围和 CPU 整数公式；保留已验证的 INT32 identity/拼接等存储操作 |
| FP32 Q/K/V 的 SDPA | 删除共享用例，保留 FP16/BF16 及已有可执行的 FP8 对照 |
| 四维交叉广播 MatMul | 原用例无原生执行计划，移除该用例，保留二维/三维及精度用例 |
| BF16 reduction 输出 BF16，以及不支持的混合输出配置 | Graph 与原生 ReduceTensor 均不支持的配置移除；保留 FP32/FP16，以及已验证的 BF16/INT32 SUM 到 FP32 |
| 带 padding 的 Binary Select | 原生 Graph 无计划且无对应 legacy API，移除该配置 |
| 带 padding 的 NEG | 保留，通过 cuDNN 原生 TransformTensor 的负缩放进行对照 |

本次共享变更需要同步修改其他平台 validation：删除已不存在的 CPU 调用、
无入口的测试 adapter 和注册项，并按实际共享生成器收缩 THead/天数能力表。
保留条目的平台资格和测量值不变，不将删除用例统计为通过。用例在生成后收缩的
位置保留既有 tensor UID，以免改变其他平台已有基准的 workload 标识。
相比保留空壳 CPU 函数或返回 SKIP，这样能直接消除编译依赖和两套测试范围。
现有 Graph/编译协议的 host contract 仍可保护保留的生产实现，不代表这些配置
已通过数值或性能验收。

已注册 NVIDIA 用例在计划构建、执行或比较时失败，必须报错，不能动态跳过或
替换成 CPU 对照。具备 cuDNN 对照的算子/dtype 继续使用共享 shape 集合。

`reference/tests` 为保留的 `reference/cpu` 提供独立 CPU/cuDNN 功能测试，
覆盖矩阵、卷积、reduction、归一化、attention、索引/布局、浮点逐元素、BN Finalize、
重采样、因果卷积、FP8/MoE 前向及通用公式。原有 `div/pow/mod/cmp_eq` 的
24 条广播对照保留。测试直接执行 cuDNN，比较完整逻辑输出，并检查输入、
输出保护区和 padding；INT32 identity 等按原始字节比较。
IEEE FP32 与 TF32 使用独立进程及对应的 `NVIDIA_TF32_OVERRIDE` 设置。
BN Finalize 同时校验四/六输出，归一化检查各反向输出和 running statistics。
CPU/cuDNN 测试不再以“无计划”作为成功或跳过；保留测试都必须真正完成对照。

CPU 数学函数没有存储 dtype 的路径，使用 cuDNN 支持的 dtype 和相同量化输入验证，
例如 attention 的 FP16/BF16。代表性小用例用于验证实现，不等同于穷举所有参数。
`reference/cpu/benchmark.cpp` 的公式也做功能校验；对应的
`reference.cpu.formulas.cudnn` 不调用性能 runner、不测 CPU 耗时。
所有平台的 CPU 参考只用于功能正确性，性能比较仍使用厂商 DNN 或规定的等效性能。

此次 H100 验证中，16 个 CPU/cuDNN 测试分组全部通过（493 条数值/存储对照），
另有 4 个 CPU host contract 通过；无失败、无跳过。


其他平台缺少厂商功能对照时，仅可调用保留且经过 cuDNN 对照的 CPU 功能。
Hygon、Iluvatar 和 THead 的浮点逐元素回退保留；MThreads 保留浮点激活反向回退；
Ascend 保留原始字节 identity 等已验证路径。整数算术回退及依赖它的功能测试
随本次 CPU API 删除一起移除。数值不匹配、非法参数和执行错误仍然失败。

## 4. CMake 装配顺序

根 `CMakeLists.txt` 的顺序固定为：

1. 构建平台无关 core 和生产 backend。
2. 按开关加载 `reference/`、`tests/`、`benchmark/`，定义 oracle、公共 workload 和注册函数。
3. 读取唯一的 `FLAGDNN_BACKENDS` 列表。
4. 加载 `backends/<platform>/validation/CMakeLists.txt`。

平台 validation CMake 独立查找设备 SDK，并按已启用的模式调用：

- `flagdnn_register_functional_suite(...)`
- `flagdnn_register_benchmark_suite(...)`

功能和性能可以独立构建；benchmark 不再要求 `FLAGDNN_BUILD_TESTS=ON`。validation
target 不安装、不导出，生产 backend 不链接 validation。

例如同时选择 NVIDIA 与 Ascend：

```bash
-DFLAGDNN_BACKENDS='nvidia;ascend'
```

## 5. 性能 case 执行语义

每个 benchmark case 依次：

1. 构建 FlagDNN Graph executable。
2. 构建平台 DNN reference；NVIDIA 必须取得实际可执行的 cuDNN 对照。
3. 使用相同输入运行并比较输出。
4. 正确后执行 warmup。
5. 交替测量 FlagDNN 与 reference，降低执行顺序偏差。
6. 输出 `steady_state` JSONL 样本和 speedup。

NVIDIA 主性能入口和 Attention 性能用例启用 FlagDNN autotune；补充 dtype 和
扩展算子复用其 case 的构建选项。双方的构建策略随 workload 固定。
同一 executable 的稳态采样不会重新进行候选选择；首次和再次构建耗时单独记录。

## 6. NVIDIA 全量构建

推荐入口：

```bash
tools/build.sh
```

默认根据 backend 生成 `build/<backend>` Release 构建，并启用功能与性能测试。公共
脚本不发现平台 SDK；NVIDIA 的 libtriton_jit、Torch 和 cuDNN 发现由
`backends/nvidia/` 负责。使用 `tools/build.sh --help` 查看 Python、backend、构建
目录和并行度等公共选项，其他 CMake 参数放在 `--` 后。下面的命令是等价的手动
CMake 配置示例。

```bash
cmake -S . -B /tmp/flagdnn-build-nvidia \
  -DFLAGDNN_BACKENDS=nvidia \
  -DFLAGDNN_BUILD_TESTS=ON \
  -DFLAGDNN_BUILD_BENCHMARKS=ON \
  -DFLAGDNN_EXECUTION_ENGINE=libtriton_jit \
  -DFLAGDNN_CODEGEN_PYTHON=/path/to/python
cmake --build /tmp/flagdnn-build-nvidia -j
```

只构建平台无关 core contract 时设置 `-DFLAGDNN_BACKENDS=`。

只构建并运行 CPU reference 的 cuDNN 独立验证：

```bash
reference/tests/build.sh
reference/tests/run.sh

# 严格验收：只要有 CPU 功能缺少可执行的 cuDNN 对照，也返回失败。
FLAGDNN_REFERENCE_REQUIRE_ALL=1 reference/tests/run.sh
```

通过 `FLAGDNN_REFERENCE_BUILD_DIR` 选择构建目录，通过 `FLAGDNN_BUILD_TYPE` 选择
单配置或多配置生成器中的配置。

## 7. 运行方法

批量执行默认的 `DEFAULT_OPERATORS`，按 GPU 分配算子，每个算子依次执行精度和
性能测试（同一 GPU 内串行，不同 GPU 并行）：

```bash
python3 tools/run_tests.py --dump-output \
  --output-dir logs_result_20260918_flagdnn --gpus 1,2
```

后端从 `FLAGDNN_BENCHMARK_PLATFORM` 或 CMake 构建信息识别。存在多个后端时
使用 `--platform nvidia|iluvatar|hygon|thead|ascend|mthreads` 选择；非默认构建目录
使用 `--build-dir` 或 `FLAGDNN_BUILD_DIR` 指定。测试必须在构建时使用的运行环境中
执行。批量模式默认运行两个 suite，单个命令超时为 4800 秒，可用 `--timeout`
调整；只有显式传入 `--preflight` 才执行前置契约测试。

THead 的每个 GPU worker 同时设置相同的 `CUDA_VISIBLE_DEVICES` 和
`HGGC_VISIBLE_DEVICES`。当前 PPU CUDA 兼容驱动实际读取前者；原生测试与 JIT
内部使用进程可见的逻辑设备 `0`，由掩码映射到 `--gpus` 分配的物理卡。
控制台的 `[GPU n]` 是调度编号，不能单独作为物理设备使用情况的证据。

THead 批量模式默认将编译 artifact 保存在 `<build-dir>/cache/run_tests`，在功能、
性能 suite、多 GPU worker 和后续批次之间复用；非空 `FLAGDNN_CACHE_PATH` 可覆盖
该目录，空值视为未配置。缓存不随结果目录清理，首次运行仍需编译，后续运行可以
复用与当前编译器身份、设备和请求匹配的 artifact。所有精度 case、数值比较、
warmup 和性能采样仍正常执行，不复用测试结果或性能数据。需要冷缓存对比时，将
`FLAGDNN_CACHE_PATH` 指向新的空目录；其他平台和 `--output` 串行兼容模式的缓存
策略不变。确认没有测试使用该目录后，可删除 `<build-dir>/cache/run_tests` 清理缓存。

输出目录与 FlagBLAS 批量测试格式一致，项目环境字段改为 `flag_dnn`：

```text
logs_result_20260918_flagdnn/
├── add/
│   ├── accuracy_stdout.log
│   ├── accuracy_stderr.log
│   ├── accuracy_result.json
│   ├── performance_stdout.log
│   ├── performance_stderr.log
│   └── performance_result.log
├── sub/
├── ...
├── summary1.json
├── summary2.json
└── summary.json
```

`--dump-output` 控制四个 stdout/stderr 日志，结果文件始终生成。`summary<GPU>.json`
在每个算子完成后原子更新，即使 GPU 未分配到算子也会生成空汇总；`summary.json`
合并当前运行的分片。中断时保留已完成算子的汇总并清理测试子进程。精度 JSON
记录逐用例结果；只有总数的原生输出使用明确标为 `native_accounting` 的匿名索引。
性能日志使用 `[INFO]` JSON 记录，延迟以毫秒表示，汇总详情字段为
`base`、`gems`、`speedup`。`--color` 支持 `auto/always/never`，重定向时默认使用
普通文本进度。原有 `--output <file>` 串行兼容模式仍可用。

结构、API 和依赖 contract：

```bash
ctest --test-dir /tmp/flagdnn-build-nvidia \
  -R '^core\.|^integration\.nvidia\.(compiler_contract|dependency_boundary|reference_dependency_boundary)$' \
  -j1 --output-on-failure
```

全部功能测试严格串行：

```bash
ctest --test-dir /tmp/flagdnn-build-nvidia \
  -L functional -j1 --output-on-failure
```

全部性能测试严格串行：

```bash
ctest --test-dir /tmp/flagdnn-build-nvidia \
  -L benchmark -j1 --output-on-failure
```

按算子运行：

```bash
python3 tools/run_tests.py \
  --build-dir /tmp/flagdnn-build-nvidia \
  --ops matmul \
  --suites functional,benchmark \
  --platform nvidia \
  --output /tmp/flagdnn-matmul-results.json
```

当前不提供 `--ops` 时，默认依次运行 `add`、`sub`、`mul`、`div`、`pow`、`max`、
`min`、`mod`、`add_square` 和 `cmp_eq`。使用 `--ops all` 运行所选 suite 在
平台目录中注册的全部算子；使用 `--list` 打印所选平台和 suite 的算子列表。

NVIDIA 测试集合是 FlagDNN 与当前 cuDNN 栈的可执行交集，不能从公共 API 类型
矩阵直接推算测试数量。测试适配器和 CMake 读取同一份算子排除清单；各算子 runner
进一步限定已验证的 dtype、shape 和布局。完整数量以当前 case 工厂和实际运行结果
为准，旧的 CPU 对照或无 cuDNN 性能数据不计入验收。

当前 H100（SM90）、cuDNN 9.24 环境不注册独立 `genstats`、`rng`、`rope`、
`rope_backward` 和 `moe_grouped_matmul_bwd` 比较。BF16 reduction 使用 cuDNN
可执行的 SUM → FP32；INT32 算术/归约、非浮点布局转换和 FP32 SDPA 等不支持的
配置不加入 NVIDIA 比较集合。Identity、gen_index 的部分非浮点类型有实际 cuDNN
对照，继续保留。SDK 或设备变化后应重新执行支持性检查。

- `causal_conv1d` 使用实际 `cudnnCausalConv1dForward`，覆盖 FP32/FP16/BF16；
  该 API 无 TF32 精度选项，仅保留 dilation=1、带 bias 的连续布局 case。
- `bn_finalize` 仅保留可执行的 FP32 case；不使用 CPU 补足 FP16/BF16 对照。
- `resample` 保留三种 pooling 的 FP32/FP16/BF16，以及空间采样器支持的 FP32
  bilinear 配置；nearest 和无法对齐坐标语义的配置不加入 NVIDIA 比较集合。
- E8M0 用于 MXFP8 的 block scale；E8M0 identity 没有可执行对照，不计入 identity
  的 dtype 覆盖。

| 未注册的独立算子 | 当前验证栈上的原因 |
| --- | --- |
| `genstats` | 实际 GENSTATS 图在 FP32/FP16/BF16 下均未取得可执行计划 |
| `rng` | uniform、normal、Bernoulli 的独立图未取得可执行计划 |
| `rope`、`rope_backward` | cuDNN 的 RoPE 接口用于 SDPA 融合，不能作为这里的独立算子对照 |
| `moe_grouped_matmul_bwd` | FP16/BF16 对照要求 cuBLASLt ≥13.5，当前实际加载 12.8.4；FP8 不在 cuDNN backward 支持表内 |

MoE 的依赖及类型要求见 [cuDNN 支持矩阵](https://docs.nvidia.com/deeplearning/cudnn/latest/operations/MoeGroupedMatmul.html#support-matrix)。

这些限制只决定 NVIDIA 验证集合，不移除 FlagDNN 的对应生产能力。MXFP8 在当前
H100 上有实际 cuDNN block-scale dequantize + matmul 对照，继续参与功能和性能
测试。MoE SCATTER 中出现 cuDNN 未完整写出输出的路由配置已替换为可执行配置，
替换后仍覆盖 top-k、空专家和每种输入类型的 14 个 shape。

SDPA 前向、反向及两种 FP8 Attention 均有独立性能基准。前向与反向的统计量由
实际 cuDNN 前向生成，性能阶段复用同一套输出检查。FP16/BF16 前向与反向均保留，
FP8 前向与反向覆盖 E4M3/E5M2。逐算子 suite 的超时包含编译和 cuDNN 计划构建。

IEEE 和 TF32 matmul/convolution 分进程执行；IEEE 进程在启动前设置
`NVIDIA_TF32_OVERRIDE=0`，避免 cuDNN 默认计划隐式使用 TF32。TF32 组选择
Tensor Core 计划。范围和相消的 IEEE contract 也使用独立进程。

NVIDIA 的逐用例性能验收可添加 `--min-speedup 0.9`。指标为 cuDNN GPU median
除以 FlagDNN GPU median；每个可比较用例必须分别达到阈值，不使用算子平均值。
报告的 `performance` 列出不达标用例；`comparable_coverage` 核对每个 case 的
FlagDNN/cuDNN 配对。缺失或失败的任务、缺失 provider、重复 provider 和非法时间
均判失败。不传阈值时仍输出性能统计，但不因加速比低于阈值而失败。

完整验收时应取消 `FLAGDNN_BENCHMARK_CASE` 过滤并使用 `--ops all`；coverage
描述所选任务及其实际输出记录，不应把过滤后的结果宣称为全量用例覆盖。

## 汇总 JSON 兼容格式

`tools/run_tests.py --output <path>/summary.json` 生成与原批量测试程序相同的
汇总结构，供现有可视化程序读取：

- 顶层字段为 `timestamp`、`env`、`selected_suites`、`result`。
- `functional` 写入 `result[算子].accuracy`，`benchmark` 写入
  `result[算子].performance`；`selected_suites` 使用相同名称。
- 状态使用 `Passed`、`Failed`、`Skipped`、`Timeout`、`NotFound`、`Error`；
  `duration` 单位为秒，功能记录包含实际 case 的通过、失败、跳过等计数。
- 性能数据为 `data[dtype].details[case]`，每项包含 `base`、`flag_dnn`、
  `speedup`；前两项单位为毫秒，dtype 层的 `speedup` 为各 case 加速比的算术平均。
  dtype 名称沿用 `fp32/fp16/bf16` 等；detail 键保留完整 case 名称，包含 shape、
  布局或模式等信息，避免同 shape 的不同配置相互覆盖。
- 有性能数据时，`data_file` 使用相对路径指向每次发布独立保存的性能文件，
  其结构与原程序一致，后续发布不会覆盖旧汇总引用的数据。

为兼容 FlagGems 的 `tools/psum_text`，未选择的分类以 `Skipped` 和零计数占位；
`selected_suites` 仍只记录实际选择。`gems` 是 `flag_dnn` 耗时的别名，
`env.flag_gems` 是 `env.flag_dnn` 的别名，均表示本次被测 FlagDNN。
FP8 使用 `f8-e4m3fn/f8-e5m2`；脚本固定的汇总列不包含 TF32、E8M0，
这些类型仍完整保存在 JSON 和单算子详情中。对包含 `summary.json` 的目录执行：

```bash
python /path/to/FlagGems/tools/psum_text /path/to/results -f markdown
```

原始微秒样本、执行命令、覆盖率、性能门槛和失败诊断保存在旁边的
`summary.native.json`，不混入兼容汇总。指定其他文件名时，诊断文件名为
`<stem>.native.json`。程序退出码继续反映测试和性能门槛结果；批次运行中、参数
错误或中断时，兼容文件不会保留上次成功结果，具体状态见诊断文件。

日常开发使用默认算子集合或 `--ops add` 等小范围选择，无需使用 `--ops all`。

## 8. 串行与结果纪律

同一设备上的性能测试必须使用 `-j1`，且整个测量期间不能同时运行其他 GPU 任务。
并行进程会影响 autotune winner、首次编译成本、GPU event 样本和平台 DNN plan
选择；主机提交及构建耗时测量期间也应避免其他高负载 CPU 任务。

benchmark 的稳定机器接口由 `benchmark/result.schema.json` 定义。控制台文本只供
开发者阅读，不应作为持续集成的数据接口。

NVIDIA 使用 schema v3，在保留 CUDA Graph GPU 稳态时间的同时，记录独立的主机
提交、首次/再次构建耗时和 workspace。缓存范围与指标解释见
[NVIDIA 实现与开发导航](nvidia-implementation.md#性能记录)。
NVIDIA CI 使用 `--ops all --suites functional,benchmark --preflight` 的统一入口，
并保留安装后 consumer 检查。

NVIDIA 只支持 `libtriton_jit` 执行引擎，配置其他引擎会明确报错。
安装后的 NVIDIA consumer 检查实际 GPU 输出、manifest 和 autotune selection，
并要求 FlagDNN artifact 目录中没有外部 cubin；实际编译缓存由 Triton/JIT 管理。

## 补充类型与配对性能记录

新增算子的薄性能入口使用 `benchmark/common/native_runner.hpp`，复用
`tests/common` 的 shape 与 FlagDNN Graph builder；cuDNN 对照、设备管理、输出
检查和计时均属于 NVIDIA adapter。`native` 表示 C++ 入口，不表示无对照。

`benchmark/dtype_main.cpp` 注册 `.boolean`、`.copy`、`.ieee`、`.tf32` 和
`.fp32_output` 补充组。`benchmark/dtype_only_operators.txt` 中的算子由这些
补充组完整提供性能入口。普通 matmul 的 FP8 组注册为
`functional.nvidia.matmul.fp8` 和 `benchmark.nvidia.matmul.fp8`，与
`matmul_fp8` API 的 case 集合互斥。统一 runner 按算子选取全部已注册分支。

NVIDIA v3 记录必须成对包含 `flagdnn` 和 `cudnn`。`comparison: "none"`、
单方记录或缺失任一方运行开销字段都会失败。GPU 稳态时间之外，双方均记录
host submit、首次构建、复用构建与 workspace，不允许生成 CPU reference latency。

```bash
python3 tools/run_tests.py --platform nvidia --build-dir build/nvidia \
  --ops all --suites all --preflight --output /tmp/flagdnn-all-results.json
```

未实现新增执行链的平台使用平台拥有的 capability 列表注册 C++ gate，返回 77
并标记跳过。此状态不能作为该平台实现或通过功能测试的证据。
