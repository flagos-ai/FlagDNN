# FlagDNN THead 全栈适配实施计划

> 状态：完成。基础设施、算子覆盖和验证链路均已实现，同版本最终真机门禁全部通过。
>
> 执行约束：当前 workspace 内顺序实施；禁止子代理；不创建 commit；不修改公共代码、
> `backends/nvidia/**`、FlagTree 或 libtriton_jit 源码。

详细设计见 [THead 全链路适配设计](../specs/2026-09-02-thead-adaptation-design.md)，61 个算子
的逐项状态见 [THead 算子覆盖实施清单](2026-09-03-thead-operator-coverage.md)。

## 1. 目标

交付可配置、可编译、可安装、可执行和可验证的 THead backend，并打通：

```text
Frontend Graph -> Graph IR -> THead compiler -> PPU Triton cubin
  -> libtriton_jit -> PPU stream -> Graph capture/replay -> autotune
  -> isolated install consumer -> acDNN functional/benchmark
  -> tools/run_tests.py
```

测试模型不是“对比两个 graph API”，而是用 FlagDNN Graph 运行结果与 THead acDNN C/C++
reference plan 比较。不能精确对照的 case 结构化 skip。

## 2. 技术栈与环境

| 项目 | 已验证值 |
|---|---|
| 编译语言 | C++20、CMake/CTest、Python 3.12 |
| 设备 | PPU-ZW810E |
| PPU SDK | 2.0.0-715aa1 |
| acDNN | header/runtime 1400 |
| Triton | 系统 PPU 版 3.5.0+ppu2.0.0.oe |
| Triton root | `/usr/local/lib/python3.12/site-packages` |
| JIT | `/home/wbj/libtriton_jit`, backend CUDA |
| FlagDNN backend ABI | v2, identity `thead` |
| Graph/artifact schema | Graph IR v3 / execution program v1 |

`cuda`/`nvidia` 只作为 PPU 工具链内部兼容标识；对外 backend 和 target identity 必须分别为
`thead` 和 `ppu_*`。

## 3. 执行规则

- 只在 `backends/thead/**` 和 THead 文档中实现。
- common kernel 优先；需要平台 kernel 时只增加最小 THead 文件。
- FlagDNN functional/benchmark 的唯一 oracle 是 acDNN，不使用任何 BLAS。
- production plugin 不链接 acDNN，reference 目标不链接 BLAS/cuDNN。
- 每个接口 fail closed；未知字段、错误 identity/hash/path/geometry 不允许回退。
- device tests 串行，并使用 caller-owned 非默认 PPU stream。
- 只有 capability 声明的缺口可 skip；所有意外错误必须失败。
- 公共 `tests/core/run_tests_contract.py` 问题继续暂缓，最终 runner 使用 `--no-preflight`。

## 4. 已完成工作分解

### Task 1：环境与依赖边界

- [x] 单一 PPU SDK root 解析 driver compatibility、HGGC、acDNN、工具和版本。
- [x] 识别系统 PPU Triton distribution、module、backend catalog、target 和 `.cubin` 产物。
- [x] 验证 libtriton_jit CMake/provenance/SONAME 和 `BACKEND=CUDA` identity。
- [x] 配置阶段拒绝混用普通 Triton、FlagTree 或错误 SDK。
- [x] production/reference dependency boundary contract。

### Task 2：ABI v2 plugin 与 PPU context

- [x] `flagdnnBackendGetApiV2` 和完整 create/build/execute/destroy 生命周期。
- [x] primary context retain/release、设备信息、target/device identity。
- [x] caller-owned stream 执行和 binding/workspace/alignment 校验。
- [x] 插件 SONAME、错误传播和安装导出。

### Task 3：Graph IR 与 artifact

- [x] schema v3 严格解析、duplicate-key/非有限数值保护。
- [x] tensor/node UID、ports、DAG、virtual tensor、geometry、stride、alignment 验证。
- [x] execution-program v1，记录 source/signature/arguments/launch/autotune/hash。
- [x] runtime 重新验证 hash、相对路径、workspace、stage topology 和 compiler identity。

### Task 4：Triton compiler 与 libtriton_jit

- [x] Python provider 选择公共或 THead kernel，并用 PPU Triton 真实编译 cubin。
- [x] compiler identity 覆盖 provider、registry、source、tuning、Triton 和 target。
- [x] typed tensor/workspace/scalar raw arguments 与 libtriton_jit ABI 对齐。
- [x] binary/unary/comparison、composite、layout、reduction、normalization、MatMul、convolution
  family 的 compiler contract 和负向变异。

### Task 5：Runtime Graph

- [x] 非默认 stream 上真实 launch。
- [x] 单 stage、多 stage 和跨 stream event 依赖。
- [x] capture、instantiate、多次 replay 和重复 capture。
- [x] capture/steady-state 无 Python、JIT、cache 写入和意外 host allocation。

### Task 6：Autotune

- [x] 保守候选表和完整 candidate identity。
- [x] miss 时 warmup/measurement/正确性验证和原子 winner 发布。
- [x] hit 时 winner-only build，稳态不重新调优。
- [x] 损坏、schema/hash/identity 不匹配时安全 retune。
- [x] stateful Graph 候选间恢复输入输出。

### Task 7：acDNN-only functional

- [x] capability schema、61 operator catalog closure 和稳定 skip reason。
- [x] stable primitive、backend descriptor 与复合 DAG reference RAII 层。
- [x] 同输入、同 stream、stride-aware compare 和 padding 检查。
- [x] 每个 suite 强制 `cases = executed + skipped`。
- [x] 51 operator 的全部公共 case 完成真实执行，10 operator 完成 audited skip。

### Task 8：acDNN-only paired benchmark

- [x] comparable catalog schema v2 和 57 operator adapter closure。
- [x] 同输入/stream、分别 warmup、交替采样和 median 记录。
- [x] 每个 comparable case 强制 FlagDNN/acDNN 双记录，无单边结果。
- [x] MatMul reference 走 acDNN descriptor，不使用 BLAS。
- [x] 51 operator 的 1125 个公共 case 完成成对认证，6 operator 的 57 个 case 完成 audited skip。
- [x] 可选 `--min-speedup` 门禁；未设置时不声明性能目标。

### Task 9：安装态 consumer

- [x] 临时 prefix install 和独立 `find_package(FlagDNN)`。
- [x] 清除开发树 compiler override，验证安装态 source/runtime 路径。
- [x] C/C++ API、plugin ABI、Graph、JIT、非默认 stream、acDNN 数值对照。
- [x] autotune 初次发布、第二次 winner hit 和 cache 内容不变。

### Task 10：runner 与全量退出

- [x] `tools/run_tests.py` THead adapter，执行 61 functional + 57 benchmark 任务。
- [x] summary v2 解析 case accounting、skip、provider pair、coverage 和性能统计。
- [x] CTest 注册 129 integration + 61 functional + 57 benchmark，共 247 项。
- [x] 同一最终版本完成 build、完整 CTest、118-task runner、真实 compiler all、依赖审计和
  `git diff --check`，并回填最终数字。

## 5. 算子实施模板

后续扩大任何 dtype/layout/shape/attribute 或重新打开 audited skip，仍按以下步骤：

1. 从公共 functional/benchmark manifest 列出全部 case。
2. 在当前 acDNN header/runtime 中确认精确 primitive/descriptor/DAG；记录完整约束。
3. 先添加 compiler/capability/reference 的失败 contract，并确认 RED 原因准确。
4. 优先复用 common kernel；只在 THead 目录增加最小实现和 tuning candidate。
5. 真实 Triton compile，校验 artifact source/signature/grid/block/shared/hash。
6. 真机认证 acDNN plan 的 compile/create/execute/numeric 语义。
7. functional 完整记账并通过数值比较。
8. 有精确 benchmark reference 时添加成对目录和双 provider 记录。
9. 重跑 JIT、Graph capture/replay、autotune 和 installed consumer。
10. 完成完整 CTest、runner、dependency 和 diff hygiene 后才能更新状态。

仅 header 中出现 enum 不算能力证据；SDK/header/runtime/目标设备变化时必须重新资格认证。

## 6. 完成定义

本计划完成需要同时满足：

- 61 个 functional operator 与 57 个 benchmark operator 无缺失入口；
- 51 个 qualified operator 的 1052 个 functional case 真实执行，10 个 audited-skip operator
  的 90 个 case 有可复现证据；
- comparable catalog 的 1125 个 case 全部成对认证，剩余 57 个 benchmark case 有稳定 skip；
- 真实 PPU cubin、libtriton_jit、Graph、autotune、安装 consumer 全部通过；
- 生产 plugin 无 acDNN/BLAS 依赖，functional/benchmark 无 BLAS/cuDNN reference 依赖；
- warnings-as-errors build、247 项 THead CTest、118-task runner 和 compiler all 零失败；
- 文档数字来自同一最终运行，不把未设置阈值的性能采样写成性能达标。

## 7. 已知延后项

- `/home/wbj/FlagTree`：当前不使用，问题后续单独处理。
- `tests/core/run_tests_contract.py` / `core.test_architecture`：公共问题按要求暂缓。
- 10 个 audited-skip operator：等待未来 SDK 提供精确且真机语义正确的 acDNN reference。
- 更宽 dtype/layout/shape/attribute：按 capability 逐 slice 扩大，不做隐式全覆盖声明。

## 8. 最终真机证据（2026-09-06）

- warnings-as-errors build：通过；
- THead CTest：247 项最终闭合为 231 passed、16 expected skip；129 个 integration 全部通过。
  完整 label run 唯一未通过项是以旧 300 秒属性启动的 exhaustive Triton compile；属性改为
  900 秒后精确复验在 401.38 秒通过；
- runner：118 项，102 passed、16 skipped、0 failed/timeout/not-found，overall passed、exit 0；
- functional：1052/1142 case 执行，90 个 capability skip；
- benchmark：1125/1182 case 成对执行，57 个 capability skip；
- comparable：51 个形成 pair 的 operator、1125/1125 pair、2250 provider records，零
  missing/extra/record/accounting error；147 条 skip record 零 schema error；
- compiler：`--case all --compile` 真实 cubin 独立复验通过（398.59 秒），CTest 内同一矩阵也通过；
- dependency：production plugin 直接依赖不含 acDNN/BLAS，FlagDNN reference target 直接依赖
  只使用 acDNN、不使用 BLAS；
- 性能 ratio 几何均值 1.623491、中位数 0.980640、最小值 0.017787；未设置阈值，不做性能
  达标声明；
- 完整报告：`build/thead/thead-run-tests.json`。
