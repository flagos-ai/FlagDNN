# FlagDNN THead/真武 PPU 全链路适配设计

> 状态：架构和 61 算子目录闭包已实现；算子级资格范围见
> [THead 算子覆盖实施清单](../plans/2026-09-03-thead-operator-coverage.md)。
>
> 所有权边界：`backends/thead/**`

## 1. 目标

在不修改平台无关 Graph/lowering、不修改其他 backend 专用代码的前提下，实现完整 THead
backend：配置、编译、安装、Frontend Graph、Triton kernel、libtriton_jit、运行时、autotune、
acDNN-only 功能测试、acDNN-only 成对性能测试以及 `tools/run_tests.py` 汇总。

THead 没有 cuDNN Frontend Graph 风格的 graph API 不是阻塞条件。设计把 FlagDNN Frontend
Graph 作为被测接口，把 THead acDNN C/C++ primitive/backend descriptor 作为独立 oracle：

```text
同一输入
  ├─ FlagDNN Graph -> THead compiler/plugin/Triton -> output A
  └─ acDNN C/C++ reference plan --------------------> output B
                                                  A/B 数值比较
```

acDNN 不能精确表达的语义必须结构化 skip，不得使用 BLAS 或其他 reference 填空。

## 2. 设计约束

1. Backend identity 固定为 `thead`，使用 backend ABI v2，插件 SONAME 为
   `libflagdnn_backend_thead.so.2`。
2. 使用 `/usr/local/lib/python3.12/site-packages/triton` 的 PPU 定制 Triton；当前不使用
   `/home/wbj/FlagTree`。
3. PPU Triton 通过 `nvidia`/`cuda` compatibility backend 生成 `.cubin`。这些内部标识不能
   泄漏成 FlagDNN backend 或设备身份。
4. `libtriton_jit` 使用其既有 CUDA backend；FlagDNN 不修改其源码。
5. FlagDNN THead functional/benchmark 的唯一数值 reference 是 acDNN，不调用 acBLAS、
   cuBLAS、其他 BLAS、cuDNN、PyTorch、CPU reference 或 FlagDNN 自身。
6. production plugin 不 include、调用或链接 acDNN；acDNN 仅进入测试目标。
7. 只修改 `backends/thead/**` 和 THead 文档；不修改公共代码、`backends/nvidia/**` 或其他
   平台专用代码。
8. capability skip 与系统错误分离：只有明确的能力缺口可返回 77；compiler/JIT/artifact/
   driver/Graph/cache/acDNN 意外错误必须失败。
9. `tests/core/run_tests_contract.py` 的公共契约问题继续暂缓，不能为此放宽 THead 设计。
10. 不创建 commit。

## 3. 已验证环境

以下是 2026-09-04 的资格认证快照，不应永久硬编码为其他机器的事实：

| 项目 | 值 |
|---|---|
| 设备 | 2 × PPU-ZW810E |
| 设备节点 | `/dev/alixpu*` |
| PPU-SMI / Driver / HGGC | 1.22 / 1.3.2-d7f5a2 / 13.0 |
| PPU SDK | `/usr/local/PPU_SDK`, 2.0.0-715aa1 |
| acDNN header/runtime | 1400 / 1400 |
| Python | `/usr/local/bin/python3` |
| Triton distribution/module | 3.5.0+ppu2.0.0.oe / 3.5.0 |
| Triton root | `/usr/local/lib/python3.12/site-packages` |
| libtriton_jit | `/home/wbj/libtriton_jit`, backend CUDA |

此前从受限 sandbox 得出“无驱动或无设备节点”的判断不是目标容器事实。设备验证必须在透传
`/dev/alixpu*` 的环境运行。

## 4. 总体架构

```text
Frontend Graph API
  -> 公共 validation/lowering
  -> Graph IR schema v3, backend="thead"
  -> THead compiler provider
       -> IR/family capability validation
       -> registry/tuning resolution
       -> PPU-aware Triton -> cubin
       -> execution-program artifact schema v1
  -> THead ABI v2 plugin
       -> artifact runtime validation
       -> libtriton_jit executable
       -> caller-owned PPU stream
       -> Graph capture/replay
       -> persistent autotune winner cache
```

```text
backends/thead/
├── CMakeLists.txt, cmake/**        dependency discovery and install rules
├── backend.cpp                     ABI v2 entry point
├── context.*                       PPU context and device identity
├── compiler.py                     Graph IR validator/compiler provider
├── artifact.*                      artifact parser and runtime validation
├── engines/**                      libtriton_jit engine
├── kernels/**                      minimal THead-specific Triton kernels
├── tuning/common.yaml              candidate definitions
└── validation/**                   contracts, acDNN refs, functional, benchmark, runner
```

### 4.1 与 NVIDIA backend 的架构对应关系

NVIDIA 目录只作为只读架构参照，不复用或修改其平台专用源码：

| NVIDIA 架构接缝 | THead 对应设计 |
|---|---|
| `backend.cpp` + `context.*` 的动态 ABI plugin | 独立 THead ABI v2 plugin；底层改为 PPU CUDA Driver compatibility context |
| `compiler.py` 与 `artifact.*` 分离 | 保留 provider/artifact 边界，但重新实现 THead family validator、identity 和 PPU launch metadata |
| kernel registry + tuning YAML | 复用公共 registry 协议和 common kernel；THead 私有 registry 只列最小补充 kernel |
| `engines/libtriton_jit.cpp` | 对齐稳定 engine/raw-argument 接缝；消费 CUDA-backend libtriton_jit 以适配 PPU compatibility ABI |
| autotune candidate/cache 生命周期 | 复用公共 policy 语义，THead 独立实现 PPU candidate identity、测量、winner-only cache 和恢复 |
| cuDNN functional/benchmark reference | 替换为 acDNN primitive/backend descriptor/DAG；不要求 acDNN Graph API |
| validation adapter 与安装态 consumer | THead 独立注册完整 catalog、structured skip、paired record 和隔离安装门禁 |

因此借鉴的是模块边界和生命周期，而不是 NVIDIA kernel、cuDNN helper、设备身份或 CMake
实现。两套 backend 可同时演进，THead 变更不会改变 NVIDIA 行为。

公共 kernel/registry protocol 可以按稳定接口复用。只有公共 kernel 无法满足 PPU 编译或语义时，
才在 `backends/thead/kernels` 增加最小实现；禁止复制 NVIDIA backend。

## 5. 配置与依赖发现

当前配置关键项：

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

`ResolvePPUSDK.cmake` 从单一 SDK root 解析 CUDA driver compatibility、HGGC、acDNN 和版本
信息。只有 validation/benchmark 开启时才要求 acDNN。

`ResolveTritonJIT.cmake` 验证外部 CMake config、header、shared library、SONAME、provenance 和
`BACKEND=CUDA` identity。FlagDNN 只消费这个外部产物，不改源码或二进制。

Triton identity contract 必须验证指定 Python 的 `triton.__file__` 位于指定 root、distribution
version 含 `ppu`、CUDA compatibility codegen 存在、target/arch/warp 合法、PPU compiler 标记
存在且产物扩展名为 `cubin`。不能把普通上游 Triton 或 `/home/wbj/FlagTree` 静默混入。

Codegen 所需环境由 CTest/installed consumer 构造，等价关键变量为：

```bash
export PPU_SDK=/usr/local/PPU_SDK
export PPU_HOME=/usr/local/PPU_SDK
export CUDA_PATH=/usr/local/PPU_SDK/CUDA_SDK
export TRITON_PTXAS_PATH=/usr/local/PPU_SDK/CUDA_SDK/bin/ptxas
export TRITON_IR_FORMATTER_PATH=/usr/local/PPU_SDK/bin/llvm-irformatter
export TRITON_JIT_BACKEND=CUDA
```

Vendor tool 的运行时搜索路径必须受控包含 PPU SDK target/lib、SDK lib 和 CUDA_SDK/lib64。

## 6. Backend ABI、context 与 identity

`backend.cpp` 只导出 `flagdnnBackendGetApiV2`。生命周期为 create context、查询 target、创建
executable、execute、destroy。`TheadContext` 使用 PPU CUDA Driver compatibility API：

- 初始化 driver、枚举设备并 retain/release primary context；
- 查询型号、compute capability、UUID、PCI bus id 和 driver version；
- 生成 `ppu_<normalized-model>_cc<capability>` target fingerprint；
- 生成包含设备、SDK 与 driver 的 device identity，隔离编译和 autotune cache。

执行必须使用调用方传入的 native `CUstream`，不能偷换成默认 stream，也不能在稳态路径创建
隐式全局 context。

## 7. Compiler provider

`compiler.py` 是唯一 THead compiler provider，只接受 Graph IR schema v3、backend `thead` 和
engine `libtriton_jit`。解析使用 duplicate-key 和非有限数值保护；以下条件全部 fail closed：

- 未知/缺失字段、错误 schema/backend/target/version/identity；
- 重复或缺失 tensor/node UID、错误 ports、环或非法 virtual tensor 生命周期；
- 非正 extent、越界存储、重叠/未认证 stride、alignment 不足；
- 未认证 dtype/layout/broadcast/attribute/operation sequence；
- 不安全的 source/output/cache path；
- registry、source、tuning 或 compiler identity hash 不一致。

Compiler 按 family 分发：binary/unary/comparison pointwise、AddSquare DAG、layout、reduction、
BatchNorm、LayerNorm/RMSNorm、MatMul、Convolution FProp/DGrad/WGrad 和 ConvBiasRelu DAG。
每个 validator 只接受 capability catalog 已认证的最小切片。

Kernel 选择优先使用公共 registry；THead 私有 `add_square.py`、`normalization.py`、
`convolution.py` 只补足必要 family。真实 Triton 编译的 metadata 必须反映到 artifact launch；
例如 LayerNorm/RMSNorm 对 normalized suffix 17、20、768 的 shared memory 由精确候选和真实
编译结果确定，不能用估算值冒充。

## 8. Execution-program artifact

Artifact schema 与 execution program version 当前均为 1。顶层固定记录 backend、target、engine、
request hash、compiler provider/version/identity、external UID、tensor、workspace 和 stages。

每个 tensor 记录 dtype、dimensions、strides、alignment、virtual 标志和 storage size。只有跨
stage 存活、因而确实需要物化的 virtual tensor 才额外记录 workspace offset；单个融合 stage
内部的 SSA 临时值没有外部存储。每个 stage 记录：

- source node/dependency DAG、operation、provider、ownership；
- function、registry/source relative path 与 hash；
- variant full/runtime signature、typed arguments、grid/block/shared memory；
- num_warps、num_stages、受控 PPU compiler options；
- autotune warmup/repetitions、candidate identity 和 winner-cache path。

Runtime parser 使用严格字段集和数值范围，重新计算 hash，限制 source/cache 落在受控 root，
验证 workspace 无溢出/重叠和 stage 拓扑后才创建 executable。artifact 不可信，不能把 Python
provider 的输出直接传给 driver。

## 9. libtriton_jit execution engine

Engine 对每个 variant 创建 libtriton_jit kernel，在 execute 时把 external binding、workspace
tensor 和 typed scalar 组装为 raw argument array，并使用 artifact 中已验证的 launch 参数。

关键契约：

- external UID 必须恰好绑定一次，pointer/alignment/workspace size 合法；
- stage 按 dependency 拓扑执行；跨 stage 的 virtual tensor 只落在 artifact workspace，融合
  stage 内部的 virtual tensor 不创建伪 workspace binding；
- 只传 libtriton_jit 当前 ABI 支持的 options；不支持的 `maxnreg` 等必须拒绝；
- 执行期间不得写 artifact/source/cache，不得进行 Python/Triton 编译；
- steady-state launch 不产生未经允许的 host allocation；
- libtriton_jit 内部依赖不改变 FlagDNN functional/benchmark 的 acDNN-only oracle 规则。

## 10. Graph capture/replay

Runtime 必须支持在 caller-owned stream 上 capture、instantiate、重复 replay。测试覆盖单节点和
多 stage Graph、非默认 stream、跨 stream event 依赖、重复 capture/replay、结果一致性和
steady-state 无 JIT/cache 变化。

Compiler/JIT/autotune 只能发生在 executable build 阶段；capture 内只允许已构建 kernel launch。
Graph capture 失败、非法访问或输出差异是测试失败，不得 skip。

## 11. Autotune

`tuning/common.yaml` 定义 binary/unary、reduction、BatchNorm、LayerNorm/RMSNorm、MatMul 和
convolution 的保守候选。完整候选 metadata 都参与 candidate identity。

Autotune 事务：

1. cache miss 时在调用方 stream 上 warmup/measurement；
2. 每次候选执行前恢复可变输入/输出，避免状态污染；
3. 校验候选输出后原子发布 winner-only cache；
4. cache hit 只构建/执行 winner，不重新测量 loser；
5. JSON 损坏、identity/hash/schema 不匹配时安全 retune；
6. device/compiler/registry/source/tuning/Graph 变化必须产生不同 cache identity。

`maxnreg` 等当前 JIT ABI 无法传递的 option 必须为 null 或明确拒绝，不能无声忽略。

## 12. acDNN-only 功能验证

`validation/capability.json` 对每个公共 functional case 给出 status、path、reference plan、
constraints、reason code 和 detail。普通运行不能把 `probe_required` 自动当成支持；CTest 只对
绑定的已审计 case 注入资格开关。

每个 functional runner：

1. 读取公共 case metadata 和 THead capability；
2. 对未认证 case 输出完整 `[SKIP][acdnn]`；
3. 对认证 case分配同一输入，分别执行 FlagDNN Graph 和 acDNN plan；
4. 使用非默认 PPU stream，检查 padding/stride 并按 dtype 容差逐元素比较；
5. 输出 `cases = executed + skipped`，accounting 不等即失败。

Reference 层使用 RAII 管理 acDNN descriptor/handle/workspace。稳定 primitive 优先；backend
descriptor 必须先通过 compile/create/execute/numeric 真机资格认证。复合 reference 是显式
acDNN DAG，不能包含 BLAS 或 host-computed oracle。

## 13. 成对性能测试

`benchmark/comparable_cases.json` schema v2 是唯一 benchmark 能力事实源。一个声明为
`probe_required` 的 case 必须输出两条记录：`provider=flagdnn` 和 `provider=acdnn`；两条记录
缺一、metadata 不同、重复或出现额外 pair 都失败。

两侧使用同一输入与 stream，分别 warmup 后交替采样，记录 median。汇总指标为
`acdnn_median_us/flagdnn_median_us`。没有 `--min-speedup` 时只验证采样和统计链路完整，不表示
达到性能目标。

MatMul reference 走 acDNN MATMUL backend descriptor；所有 THead functional/benchmark target
都禁止链接或调用任何 BLAS。reference dependency contract 检查链接命令、ELF NEEDED 和符号。

## 14. Catalog、runner 与 skip

公共 manifest 有 61 个 functional operator 和 57 个 benchmark operator。THead CMake 必须为
每个入口注册真实 adapter 或 unsupported adapter，名字与公共 catalog 完全一致。

`run_tests_adapter.py` 执行 118 个任务并解析：

- suite 级 passed/skipped/failed；
- functional/benchmark case accounting；
- `[SKIP][acdnn]` 字段和 SDK/header/runtime/target；
- benchmark provider pair、重复/缺失/额外 record；
- comparable catalog coverage 和可选性能阈值。

Unsupported 不是“测试不存在”。每个 case 都必须有稳定 reason；编译或运行错误不能被包装成
skip。最终 JSON schema v2 是机器可读验证记录。

## 15. 安装验证

Installed consumer contract 在临时 prefix 执行安装，清除开发树 compiler override，然后用
`find_package(FlagDNN)` 构建独立 C/C++ consumer。它验证：

- headers、CMake exports、plugin ABI/SONAME、私有 runtime 依赖；
- 安装态 compiler/Triton identity 和 artifact source path；
- 非默认 PPU stream、Graph、JIT、acDNN 数值对照；
- autotune 首次发布、第二次 winner cache hit 和内容不变。

任何依赖源码树绝对路径才能运行的结果都不算安装完成。

## 16. 当前覆盖与 audited skip

完整 61 行清单及逐算子 case 分布见配套算子计划。当前 51 个 operator 的 1052 个公共
functional case 与 1125 个公共 benchmark case 已分别完成真机功能认证和成对性能认证；
10 个 audited-skip operator 为 BinarySelect、Erf、LogicalAnd/Not/Or、Mod、SDPA、
SDPA backward、SDPA FP8、SDPA FP8 backward。两个能力目录均不存在待认证 probe。

Attention 审计的关键结论：SDK 只暴露 legacy projection-oriented
`acdnnMultiHeadAttnForward`，不能精确表达 FlagDNN raw-QKV SDPA 的 GQA、broadcast bias、
diagonal mask、statistics，以及 FP8 descale/scale/amax；SDK 也没有相应 backward API。因此这些
算子不能拿“名字相似”的 API 做错误对照。

## 17. 失败语义

| 条件 | 结果 |
|---|---|
| capability 明确不支持 dtype/layout/shape/attribute | `[SKIP][acdnn]`, return 77 |
| 无精确 acDNN reference 或已证实语义错误 | `[SKIP][acdnn]`, return 77 |
| compiler/IR/artifact/hash/path/identity 错误 | fail |
| Triton/libtriton_jit/driver/非法内存/Graph 错误 | fail |
| 已认证 acDNN plan 意外失败或输出不一致 | fail |
| benchmark 单边/重复/缺失/额外 record | fail |
| case accounting 或 catalog closure 不闭合 | fail |

## 18. 标准验证命令

```bash
cmake --build build/thead -j2
ctest --test-dir build/thead --output-on-failure -L thead -j1
PYTHONDONTWRITEBYTECODE=1 /usr/local/bin/python3 tools/run_tests.py \
  --platform thead --build-dir build/thead \
  --ops all --suites all --no-preflight \
  --output build/thead/thead-run-tests.json
PYTHONDONTWRITEBYTECODE=1 /usr/local/bin/python3 \
  backends/thead/validation/compiler_contract.py --case all --compile
git diff --check -- backends/thead docs/superpowers
```

最终完成需要同一代码版本满足 warnings-as-errors build、129 个 integration、61 个 functional、
57 个 benchmark、118-task runner、真实 Triton compile、安装 consumer、依赖边界和 diff hygiene。
真机结果记录在算子覆盖计划与 `backends/thead/adaptation-recommendations.md`。

2026-09-06 的最终同版本验证满足该定义：247 项 THead CTest 最终闭合为 231 passed、
16 expected skip（完整 label run 中旧 300 秒 compile timeout 经 900 秒属性精确复验通过）；
118-task runner 为 102 passed、16 skipped、overall passed；1052/1142 functional case 和
1125/1182 paired benchmark case 真实执行；1125/1125 required pair、2250 条 provider
record 完整；独立 compiler all 与 CTest exhaustive compile 均完成真实 cubin 编译。

## 19. 后续扩展准则

扩大能力必须按“能力证据 -> RED contract -> 最小 compiler/kernel -> acDNN 真机资格 -> functional
-> paired benchmark -> JIT/Graph/autotune/install -> 全量回归”的顺序。仅出现 header enum 或
API 名称不能证明语义正确；SDK/header/runtime/设备发生变化时，backend descriptor 与 audited
skip 均需重新审计。
