# FlagDNN THead/真武 PPU 适配指南

本文记录当前 THead backend 的架构、能力边界、构建测试方法和后续扩展规则。完整的 61
算子状态与逐项 case 计数见 [THead 算子覆盖实施清单](../../docs/superpowers/plans/2026-09-03-thead-operator-coverage.md)，
底层协议设计见 [THead 全链路适配设计](../../docs/superpowers/specs/2026-09-02-thead-adaptation-design.md)。

## 1. 结论

THead 平台不需要提供与 cuDNN Frontend Graph 一一对应的 graph API。验证模型是两条独立
执行链路使用同一输入：

```text
输入
  ├─ FlagDNN Frontend Graph -> THead plugin -> Triton kernel -> 输出 A
  └─ THead acDNN C/C++ primitive/backend descriptor ----------> 输出 B
                                                            A/B 比较
```

FlagDNN Graph 是被测接口，acDNN 是唯一 reference。只有完整语义可以被 acDNN 精确表达且
经真机认证的 case 才执行；其余 case 以稳定原因结构化 skip。不能为了提高通过数量而改用
BLAS、CPU、PyTorch、cuDNN 或 FlagDNN 自身作为 oracle。

当前公共目录 61 个 functional 算子全部闭合：51 个算子的 1052 个公共 case 已全部完成真机
功能认证，另 10 个算子的 90 个 case 完成 reference 能力审计并 skip。功能目录中已不存在
`probe_required`。57 个 benchmark 算子的公共目录也已闭合为 51 个算子的 1125 个 paired case
与 6 个算子的 57 个 audited skip，目录中同样不存在待认证 probe。准确的逐 case 条件由以下
文件定义：

- `validation/capability.json`：functional 唯一能力事实源；
- `validation/benchmark/comparable_cases.json`：benchmark schema v2 唯一可比较事实源。

## 2. 不可改变的边界

- 适配代码只放在 `backends/thead/**`；不在公共 Graph/lowering 中加入 THead 特判，也不修改
  `backends/nvidia/**` 或其他平台专用代码。
- 使用系统安装的 PPU 定制 Triton：`/usr/local/lib/python3.12/site-packages/triton`。
  `/home/wbj/FlagTree` 当前问题单独延后。
- `libtriton_jit` 使用它已有的 CUDA backend。THead 适配只消费其稳定 CMake/ABI 产物，不修改
  `/home/wbj/libtriton_jit` 源码，也不把它的内部传递依赖当成 FlagDNN reference 依赖。
- PPU Triton 内部使用 `nvidia`/`cuda` 兼容 backend 并生成 `.cubin`；这只是 PPU 工具链协议。
  FlagDNN backend 名必须始终是 `thead`，设备指纹必须是 `ppu_*`。
- production plugin 不 include、调用或链接 acDNN；acDNN 只进入 validation 和 benchmark target。
- 编译、artifact、JIT、驱动、非法内存、Graph、cache 或意外 acDNN 错误必须失败。只有
  capability catalog 明确声明的能力缺口可输出 `[SKIP][acdnn]` 并返回 77。
- 公共 `tests/core/run_tests_contract.py` 的问题按要求暂缓，最终 runner 使用
  `--no-preflight`；THead 自有 integration tests 必须独立全部通过。

## 3. 全链路架构

```text
FlagDNN Frontend Graph
  -> 公共 validation/lowering
  -> Graph IR schema v3, backend="thead"
  -> compiler.py
       -> 完整 IR/schema/UID/shape/stride/attribute 验证
       -> common registry kernel 优先，THead kernel 最小补充
       -> PPU-aware Triton 编译真实 cubin
       -> schema v1 execution-program artifact
  -> ABI v2 plugin
       -> artifact.cpp 严格复验 hash/path/signature/launch
       -> libtriton_jit(CUDA compatibility backend)
       -> caller-owned PPU stream
       -> Graph capture/replay
       -> autotune miss/hit/corrupt-cache recovery
```

`compiler.py` 只接受 execution engine `libtriton_jit`，对未知字段、错误 backend/target、
compiler identity 不匹配、重复 UID、非法 tensor geometry、未认证语义和不安全 artifact path
全部 fail closed。artifact 固定记录源文件、entry point、signature、grid、block、shared memory、
候选 identity 和内容 hash；运行时重新校验后才创建 executable。

THead runtime 通过 PPU CUDA Driver compatibility API 初始化设备、retain primary context，并在
调用者传入的 native stream 上 launch。target/device identity 包含 PPU 型号、capability、SDK、
driver、PCI/UUID，用于 compiler 与 autotune cache 隔离。

## 4. 代码职责

| 路径 | 职责 |
|---|---|
| `CMakeLists.txt`, `cmake/**` | PPU SDK、Triton、libtriton_jit 的严格发现与安装边界 |
| `backend.cpp`, `context.*` | backend ABI v2、PPU context、设备/目标 identity |
| `compiler.py` | Graph IR family validator、kernel 选择、Triton 编译与 artifact 发布 |
| `artifact.*` | execution-program schema、hash、path、launch 参数运行时复验 |
| `engines/**` | libtriton_jit executable、raw argument ABI、capture-safe launch |
| `kernels/**` | common kernel 无法覆盖时的最小 THead Triton 实现 |
| `tuning/common.yaml` | family 候选、资源限制和稳定 autotune identity |
| `validation/acdnn_*_reference.*` | 仅测试侧使用的 acDNN RAII reference plan |
| `validation/functional/**` | Graph/acDNN 同输入执行、数值比较和 case accounting |
| `validation/benchmark/**` | 同 stream 的成对 warmup/measurement 和 provider record |
| `validation/installed_consumer/**` | 隔离安装、find_package、ABI/Graph/JIT/autotune 真机验证 |
| `validation/run_tests_adapter.py` | 118 个功能/性能任务、skip/pair/coverage/性能汇总 |

实现优先复用 `kernels/common`。THead 私有 kernel 目前只承担公共 kernel 无法精确覆盖的
add-square、convolution 和 normalization 等 family；不得复制 NVIDIA backend 实现。

## 5. 算子覆盖

已完成全部公共 case 功能认证的 family：

- pointwise/activation：Add、Mul、Sub、Min、Max、Scale、Relu、Sigmoid、Tanh、Elu、
  Identity、Gelu、LeakyRelu、Sqrt、Neg、Abs、Ceil、Floor、Exp、Log、Rsqrt、Sin、Cos、Tan、
  Softplus、Swish、GeluApproxTanh、Div、Pow、SigmoidBackward、Reciprocal 和 6 个比较算子；
- composite/layout/reduction：AddSquare、Reshape、Transpose、Slice、Reduction；
- normalization：BatchNorm inference/training、LayerNorm、RMSNorm；
- dense/spatial：MatMul、ConvFprop、ConvDgrad、ConvWgrad、ConvBiasRelu。

当前 SDK 2.0.0-715aa1、acDNN header/runtime 1400 下的 10 个 audited skip 是：

- BinarySelect、Erf：没有精确 primitive/backend descriptor；
- Mod、LogicalNot：runtime 在 descriptor finalization 阶段拒绝；
- LogicalAnd、LogicalOr：受控真值输入得到错误语义；
- SDPA、SDPA FP8：仅有 legacy projection-oriented MHA forward，无法表达 raw-QKV、GQA、
  bias/mask/statistics，以及 FP8 descale/scale/amax；
- SDPA backward、SDPA FP8 backward：SDK 没有能生成所需梯度和 amax 的 backward API。

`qualified` 在本文中表示对应算子的全部当前公共 case 已认证，不外推到 manifest 之外的新
dtype、layout、shape 或 attribute。支持边界仍以两个 JSON 事实源中的精确 case 为准；新增
case 必须重新走 probe、功能和成对性能认证，不能根据同 family 的既有结果推定支持。

## 6. acDNN-only reference 与性能规则

稳定 primitive 优先，包括 OpTensor、Activation、TransformTensor、ReduceTensor、BatchNorm、
Convolution。只有 primitive 无法表达时才使用经 compile/create/execute/numeric 探针认证的
acDNN backend descriptor。复合算子 reference 是这些 acDNN 操作构成的显式 DAG，包括 BF16
转换、非 2 的幂尾段、批量 MatMul 分段以及非对称卷积的 symmetric-superset/slice 或
zero-pad/backward 序列。LeakyRelu 使用 TransformTensor、ReLU 和 OpTensor 组合出精确 slope
语义，规避当前 runtime 对直接 leaky descriptor 的错误实现。

MatMul reference 使用 acDNN MATMUL backend descriptor，不调用任何 BLAS。FlagDNN THead
functional/benchmark target 的依赖检查会拒绝 BLAS/cuDNN 符号或动态库；production plugin 的
依赖检查还会拒绝 acDNN。

非对称卷积 reference 先把问题变换为 acDNN 可表达的 symmetric-superset：Fprop 对结果做
slice，Dgrad/Wgrad 先把输入梯度 zero-pad 到 superset 再调用 acDNN backward primitive。
FP32/FP16 的 slice、zero、pad 使用分段 `acdnnTransformTensor`；BF16 因 runtime 1400 对该
primitive 返回 `NOT_SUPPORTED`，改用已真机认证的 `POINTWISE_IDENTITY_FWD` backend
descriptor。benchmark executable wrapper 必须把 `prepare()` 转发给底层 reference，否则
zero buffer 等一次性准备动作不会执行。

acDNN runtime 1400 在同一进程连续切换 backward-convolution geometry 时会污染后续 reference
状态；单 case 新进程的结果稳定且与 FlagDNN 一致。因此 functional 和 benchmark 的
ConvDgrad/ConvWgrad 由父进程逐 case `fork/exec` 隔离，子进程共享只读语义的 production
artifact cache，父进程负责唯一的 suite accounting marker。这是 validation-only 隔离，不改变
production plugin、Graph、Triton kernel、stream 或成对计时规则。

性能测试要求：

1. 两侧使用同一输入、同一 caller-owned stream 和相同 case metadata；
2. 分别 warmup，再交替采样 FlagDNN/acDNN median；
3. 一个 case 只有两侧记录同时存在才形成 pair；
4. 汇总指标为 `acdnn_median_us/flagdnn_median_us`；
5. 未传 `--min-speedup` 时只验证链路、记录和统计完整性，不声明性能阈值达标。

## 7. 构建和安装

当前已验证配置的关键 cache 值：

```text
FLAGDNN_BACKENDS=thead
FLAGDNN_DEFAULT_BACKEND=thead
FLAGDNN_EXECUTION_ENGINE=libtriton_jit
FLAGDNN_CODEGEN_PYTHON=/usr/local/bin/python3
FLAGDNN_THEAD_PPU_SDK_ROOT=/usr/local/PPU_SDK
FLAGDNN_THEAD_TRITON_JIT_ROOT=/home/wbj/libtriton_jit
FLAGDNN_THEAD_TRITON_ROOT=/usr/local/lib/python3.12/site-packages
FLAGDNN_BUILD_TESTS=ON
FLAGDNN_BUILD_BENCHMARKS=ON
FLAGDNN_WARNINGS_AS_ERRORS=ON
```

Triton codegen 需要 PPU SDK 的 compiler/tool 路径；CMake/CTest contract 会构造受控环境，
不要通过把任意 site-packages 放到 `PYTHONPATH` 最前面来碰运气。安装验证必须从临时 prefix
执行 `find_package(FlagDNN)`，并清除开发树 compiler override，防止误用源码树文件。

公共 `libflagdnn` 为同时支持 build tree 运行，会编译进 `FLAGDNN_DEFAULT_CODEGEN_COMPILER`
对应的构建树 fallback 字符串；安装态 `default_compiler_config()` 会先由已加载 core library 的
位置解析 `../share/flagdnn/compiler/flagdnn_codegen/main.py`，只有安装资源缺失时才考虑 fallback。
因此不能仅凭对 core ELF 做 raw-string 搜索判定安装泄漏。隔离 consumer 已清空全部 compiler、
resource、`PYTHONPATH` 和 loader override，并验证实际 artifact 使用安装前缀内资源；安装文本、
CMake metadata、plugin RPATH/RUNPATH 和 THead plugin 均不引用源码树或 FlagTree。

## 8. 测试入口

当前 THead CTest 注册 247 项：129 个 integration、61 个 functional 和 57 个 benchmark。
完整验证命令：

```bash
cmake --build build/thead -j2
ctest --test-dir build/thead --output-on-failure -L thead -j1
PYTHONDONTWRITEBYTECODE=1 /usr/local/bin/python3 tools/run_tests.py \
  --platform thead --build-dir build/thead \
  --ops all --suites all --no-preflight \
  --output build/thead/thead-run-tests.json
PYTHONDONTWRITEBYTECODE=1 /usr/local/bin/python3 \
  backends/thead/validation/compiler_contract.py --case all --compile
```

2026-09-07 综合审计后最终树的验证结果：

- 从 `/tmp/flagdnn-thead-audit.49EwAf` 全新配置，warnings-as-errors 的 THead build 共 374 个
  Ninja step 全部通过；`tools/install.sh` 安装成功，manifest 为 40 项、安装树实际 37 个文件；
- 安装后的 private compiler/kernel/tuning 资源可独立使用。清除开发树 compiler override 后，
  `find_package(FlagDNN)` consumer 在真机完成 37 个 Graph/JIT/autotune 场景，497.09 秒通过；
- 同一源码另做 common-only build，共 24 个 build step 通过，证明未选择 THead 时不会引入其
  SDK、Triton、libtriton_jit 或 acDNN 依赖；
- 除已明确延期文件关联的两项外，公共 CTest 为 13/13 通过；单独复验
  `core.test_architecture` 证明其唯一报错对象是 `tests/core/run_tests_contract.py`，不是 THead
  目录或 backend 行为；
- 当前 THead CTest 共 247 项。`tools/run_tests.py` 执行其中 61 个 functional 和 57 个
  benchmark，剩余 129 个 integration 由同一 build tree 的 CTest 完整执行且 129/129 通过。
  合并结果为 231 passed、16 个 capability 预期 skip；
- `tools/run_tests.py`：schema v2、overall `passed`、exit 0，118 项为 102 passed、16 skipped；
- functional：`1142 = 1052 executed + 90 skipped`；benchmark：
  `1182 = 1125 paired + 57 skipped`；两个目录都没有待认证 probe；
- 51 个真正形成 pair 的 comparable operator，其 1125/1125 required pair 全部观测到，2250 条
  provider record 完整，missing/extra/record/accounting error 均为 0；147 条 acDNN skip
  record 全部通过 schema 校验；
- exhaustive Triton 到真实 cubin 的 CTest 在 399.40 秒通过；production plugin 的直接 ELF
  依赖不含 acDNN/BLAS，reference target 只直接使用 acDNN、不直接使用 BLAS/cuDNN；
- ratio `acdnn_median_us/flagdnn_median_us` 的几何均值为 `1.676735`、中位数为
  `1.115608`、最小值为 `0.017919`。本次 threshold 为 null，只证明成对测试和统计链路完整，
  不表示所有 case 达成性能目标。

结果 JSON：`build/thead/thead-run-tests-audit.json`。公共
`tests/core/run_tests_contract.py` 仍按既定决定单独延期，本次用 `--no-preflight` 执行最终 runner；
它也连带使禁止 Python test 文件的 `core.test_architecture` 报错。THead adapter 自身合同和所有
THead integration 均已通过。

## 9. 2026-09-07 综合代码审计与修复

本次逐文件审计保留 92 个 THead 源码、能力数据、合同和文档文件。每个文件均可由 CMake、安装
规则、Python import、kernel registry、能力目录、合同或文档入口到达；未发现应删除的重复或
无引用文件，也未留下 symlink、空文件、`__pycache__`、`.pyc`、临时日志或编辑器文件。最终
`git status` 的允许边界仍只有 `backends/thead/**` 与 `docs/superpowers/**`，公共代码、
`backends/nvidia/**` 及其他平台代码均未修改。

审计中发现并修复的高风险问题均采用 THead 局部修复和聚焦回归：

- compiler 现在拒绝 tensor storage span 超过有符号 64 位、卷积输出/M/reduction 超过 Triton
  int32 索引范围，以及 runtime JSON 无法安全表示的 geometry，避免整数回绕和越界 launch；
- artifact 对 pointer dtype/alignment、typed tensor-transfer offset 乘法溢出、launch 参数和
  workspace offset 做严格复验；fused SSA virtual 不再错误物化 workspace，只有跨 stage virtual
  分配存储，因此 AddSquare 和 ConvBiasRelu 的融合路径 workspace 为 0；
- MatMul 的 `INPUT_IS_FLOAT32` 只对 FP32 输入置位，FP16/BF16 不再走错误精度分支；BatchNorm
  对 batch 大于 256 的 case 使用 general kernel，并显式约束 int32 索引；删除了不可达的旧
  no-bias convolution kernel；
- autotune variant ABI 现在把 scalar 的类型和值都纳入兼容性判定，防止候选改变语义；compiler
  identity 只包含稳定且影响语义的环境，并显式覆盖 registry、platform、kernel 与 tuning
  依赖，避免无关测试变量污染 cache，同时保证真实输入变化必然失效；
- 源码树从 compiler 的相邻布局推导 `libtriton_jit`，安装树使用 private relocatable 布局；安装
  集合补齐 add-square、convolution、normalization kernel，避免开发树可用而安装后缺文件。

修复后的 compiler contract 增加 storage-size overflow 与三类 convolution int32-index overflow
用例，artifact/Graph/JIT/autotune/install/functional/benchmark 的受影响路径均已重新验证。当前
审计未发现剩余的已知严重缺陷；新 shape、dtype、layout、attribute 仍必须按精确能力目录重新认证。

## 10. 后续扩展规则

扩大 dtype/layout/shape 或重新打开 audited skip 时，必须按以下顺序：

1. 审计公共 manifest 和当前 SDK，写出精确 acDNN reference plan 与能力边界；
2. 先增加失败的 compiler/capability/reference contract；
3. 只在 THead compiler/kernel 中加入最小实现，公共代码和其他 backend 保持不变；
4. 真机完成 acDNN plan 的 compile/create/execute/numeric 资格认证；
5. functional 通过后再声明 paired benchmark，禁止单边性能记录；
6. 重跑 artifact、JIT、Graph、autotune、installed consumer、完整 CTest 和最终 runner；
7. 只有 case accounting、catalog closure 和 comparable coverage 同时闭合才更新支持状态。

不能仅凭 header 存在 enum 就声明算子支持；SDK、header/runtime 或目标设备变化后，所有
backend descriptor 资格必须重新验证。
