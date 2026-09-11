# THead 算子覆盖实施清单

> 状态：完成。61 个公共 functional 算子已逐项实现或完成 reference 能力审计，最终真机全量门禁全部通过。
>
> 实施日期：2026-09-03 至 2026-09-06
>
> 代码边界：`backends/thead/**`；未修改公共代码、`backends/nvidia/**`、FlagTree 或 libtriton_jit 源码。

## 1. 目标和完成定义

本计划把 THead backend 从基础设施打通推进到公共算子目录闭包。FlagDNN Frontend Graph
是被测接口，THead acDNN C/C++ primitive 或经真机认证的 acDNN backend descriptor 是唯一
数值 reference。THead 不需要、也不假设存在与 cuDNN Frontend Graph 对等的 graph API。

状态定义：

- `qualified`：至少一个明确 functional 切片在 PPU 上执行，FlagDNN Graph 输出与 acDNN
  reference 数值一致；全部公共 case 均有 executed/skip 记账；有可精确比较的公共 benchmark
  时完成成对采样；相关 compiler/artifact/JIT/Graph/autotune/install 契约通过。
- `audited-skip`：当前 SDK 无法提供精确、可靠的 acDNN reference。算子入口和全部 case
  仍纳入目录闭包，以稳定 reason 结构化 skip，不能用其他库替代 oracle。
- `F`：`executed/manifest total`；`B`：`paired benchmark/manifest total`。`—` 表示公共
  benchmark manifest 没有该算子。

能力事实源是 `backends/thead/validation/capability.json` 和
`backends/thead/validation/benchmark/comparable_cases.json`，本文只记录汇总，不复制每个
case 的 dtype、layout、shape、stride 和 attribute 条件。

## 2. 不可变约束

- 使用 `/usr/local/lib/python3.12/site-packages/triton` 中的 PPU 定制 Triton；
  `/home/wbj/FlagTree` 的问题继续延后。
- `libtriton_jit` 按既有 CUDA backend 构建和调用，不修改其源码。这里的 CUDA 名称属于
  PPU 兼容工具链协议，FlagDNN backend 对外始终是 `thead`。
- FlagDNN 功能和性能测试只与 acDNN 比较，不调用 acBLAS、cuBLAS、其他 BLAS、cuDNN、
  PyTorch、CPU reference 或 FlagDNN 自身作为 oracle。
- production plugin 不链接 acDNN；只有 validation/benchmark target 链接 acDNN。
- compiler/JIT/artifact/driver/Graph/acDNN 意外错误必须失败；只有能力目录中声明的缺口
  可以返回 77，并输出完整 `[SKIP][acdnn]`。
- device test 串行；每个已执行 benchmark case 必须同时产生 FlagDNN 和 acDNN 两条记录。
- `tests/core/run_tests_contract.py` 的公共问题按要求暂缓；不能为绕过它修改公共代码。
- 不创建 commit。

## 3. 实施门禁

- [x] G1：闭合 61 个 functional operator 与 57 个 benchmark operator 的公共 manifest。
- [x] G2：每个已实现 family 有 canonical compiler contract、非法 mode/dtype/layout/shape/
  stride/attribute 负例和严格 Graph IR 校验。
- [x] G3：Triton 真实编译产出 `.cubin`，artifact 对 source/signature/launch/hash/version fail closed。
- [x] G4：functional reference 仅由 acDNN primitive/backend descriptor 构成，能力切片外结构化 skip。
- [x] G5：在 caller-owned 非默认 PPU stream 上执行并做 stride-aware 数值比较和 case accounting。
- [x] G6：可比较 benchmark 使用同输入、同 stream、成对 warmup/measurement，禁止单边记录。
- [x] G7：JIT、Graph capture/replay、跨 stream、安全边界及 steady-state 不变式通过。
- [x] G8：autotune 覆盖 miss、winner-only hit、损坏 cache 恢复和设备/编译器 identity 隔离。
- [x] G9：隔离安装后用 `find_package(FlagDNN)` 验证 ABI、compiler、JIT、Graph 和 autotune。
- [x] G10：warnings-as-errors build、完整 `ctest -L thead`、最终 `tools/run_tests.py`、依赖
  边界和 diff hygiene 取得同一版本的最终证据。

## 4. 完整算子清单

Reference 路径缩写：`P` 为稳定 acDNN primitive，`D` 为经真机认证的 acDNN backend
descriptor，`DAG` 为仅由这些 acDNN 操作构成的精确序列。

| # | 算子 | 状态 | F | B | acDNN reference |
|---:|---|---|---:|---:|---|
| 01 | add | qualified | 5/5 | 24/24 | P: OpTensor ADD；支持 alpha、broadcast、typed/strided |
| 02 | mul | qualified | 24/24 | 24/24 | P: OpTensor MUL |
| 03 | sub | qualified | 30/30 | 24/24 | P: OpTensor ADD，右输入系数为负 |
| 04 | min | qualified | 24/24 | 24/24 | P/DAG: OpTensor MIN；BF16 经 FP32 转换 |
| 05 | max | qualified | 24/24 | 24/24 | P/DAG: OpTensor MAX；BF16 经 FP32 转换 |
| 06 | scale | qualified | 24/24 | 24/24 | P: OpTensor MUL；生产路径为 Mul alias |
| 07 | relu | qualified | 24/24 | 24/24 | P/DAG: Activation RELU；BF16 经 FP32 转换 |
| 08 | sigmoid | qualified | 24/24 | 24/24 | P: Activation SIGMOID |
| 09 | tanh | qualified | 24/24 | 24/24 | P: Activation TANH |
| 10 | elu | qualified | 24/24 | 24/24 | P: Activation ELU，alpha=1 |
| 11 | identity | qualified | 27/27 | 33/33 | P/D: TransformTensor 或 POINTWISE_IDENTITY |
| 12 | gelu | qualified | 24/24 | 24/24 | P: Activation GELU |
| 13 | leaky_relu | qualified | 24/24 | 24/24 | DAG: negate、双 ReLU、带 slope 的 ADD |
| 14 | sqrt | qualified | 24/24 | 24/24 | P: OpTensor SQRT |
| 15 | neg | qualified | 25/25 | 24/24 | P: TransformTensor，alpha=-1 |
| 16 | abs | qualified | 24/24 | 24/24 | D: POINTWISE_ABS |
| 17 | ceil | qualified | 24/24 | 6/6 | D: POINTWISE_CEIL |
| 18 | floor | qualified | 24/24 | 3/3 | D: POINTWISE_FLOOR |
| 19 | exp | qualified | 24/24 | 24/24 | D: POINTWISE_EXP |
| 20 | log | qualified | 24/24 | 24/24 | D: POINTWISE_LOG |
| 21 | rsqrt | qualified | 24/24 | 24/24 | D: POINTWISE_RSQRT |
| 22 | sin | qualified | 24/24 | 3/3 | D: POINTWISE_SIN |
| 23 | cos | qualified | 24/24 | 3/3 | D: POINTWISE_COS |
| 24 | tan | qualified | 24/24 | 3/3 | D: POINTWISE_TAN |
| 25 | softplus | qualified | 24/24 | 24/24 | D: POINTWISE_SOFTPLUS_FWD，beta=1 |
| 26 | swish | qualified | 24/24 | 24/24 | D: POINTWISE_SWISH_FWD，beta=1.25 |
| 27 | gelu_approx_tanh | qualified | 24/24 | 24/24 | D: POINTWISE_GELU_APPROX_TANH_FWD |
| 28 | div | qualified | 24/24 | 24/24 | D: POINTWISE_DIV |
| 29 | pow | qualified | 24/24 | 24/24 | D: POINTWISE_POW |
| 30 | mod | audited-skip | 0/27 | 0/6 | runtime 拒绝 POINTWISE_MOD descriptor |
| 31 | cmp_eq | qualified | 12/12 | 24/24 | D: POINTWISE_CMP_EQ |
| 32 | cmp_neq | qualified | 12/12 | 24/24 | D: POINTWISE_CMP_NEQ |
| 33 | cmp_gt | qualified | 12/12 | 24/24 | D: POINTWISE_CMP_GT |
| 34 | cmp_ge | qualified | 12/12 | 24/24 | D: POINTWISE_CMP_GE |
| 35 | cmp_lt | qualified | 12/12 | 24/24 | D: POINTWISE_CMP_LT |
| 36 | cmp_le | qualified | 12/12 | 24/24 | D: POINTWISE_CMP_LE |
| 37 | logical_not | audited-skip | 0/4 | 0/8 | runtime 拒绝 POINTWISE_LOGICAL_NOT descriptor |
| 38 | logical_and | audited-skip | 0/4 | 0/8 | descriptor 对受控 true 输入返回错误语义 |
| 39 | logical_or | audited-skip | 0/4 | 0/8 | descriptor 对受控 true 输入返回错误语义 |
| 40 | binary_select | audited-skip | 0/13 | 0/24 | 无精确 ternary select primitive/descriptor |
| 41 | sigmoid_backward | qualified | 24/24 | 24/24 | D: POINTWISE_SIGMOID_BWD |
| 42 | reciprocal | qualified | 24/24 | 6/6 | D: POINTWISE_DIV，分子为 1 |
| 43 | erf | audited-skip | 0/24 | 0/3 | 无 ERF primitive/descriptor |
| 44 | add_square | qualified | 24/24 | 24/24 | DAG: OpTensor MUL -> ADD |
| 45 | reshape | qualified | 9/9 | 15/15 | P/D: flatten TransformTensor/typed identity |
| 46 | transpose | qualified | 9/9 | 15/15 | P/DAG: permuted TransformTensor；BF16 经 FP32 转换 |
| 47 | slice | qualified | 9/9 | 12/12 | P: 分段 TransformTensor |
| 48 | reduction | qualified | 23/23 | 9/9 | P: ReduceTensor ADD |
| 49 | batchnorm_inference | qualified | 12/12 | 24/24 | DAG: ADD/MUL 精确推理公式 |
| 50 | batchnorm | qualified | 6/6 | 24/24 | P: BatchNormalizationForwardTraining |
| 51 | layernorm | qualified | 9/9 | 15/15 | DAG: Reduce/OpTensor/RSQRT descriptor |
| 52 | rmsnorm | qualified | 9/9 | 15/15 | DAG: Reduce/OpTensor/RSQRT descriptor |
| 53 | conv_fprop | qualified | 24/24 | 51/51 | P/DAG: IMPLICIT_GEMM；非对称输出 superset/slice |
| 54 | conv_dgrad | qualified | 27/27 | 45/45 | P/DAG: ALGO_0；非对称梯度 zero-pad |
| 55 | conv_wgrad | qualified | 27/27 | 45/45 | P/DAG: ALGO_0；非对称梯度 zero-pad |
| 56 | conv_bias_relu | qualified | 30/30 | 30/30 | P: ConvolutionBiasActivationForward RELU |
| 57 | matmul | qualified | 27/27 | 24/24 | D: MATMUL；高维 broadcast 分批执行；未使用 BLAS |
| 58 | sdpa | audited-skip | 0/4 | — | legacy projection MHA 与 raw-QKV SDPA 语义不等价 |
| 59 | sdpa_backward | audited-skip | 0/4 | — | SDK 无可产生 dQ/dK/dV/dBias 的 backward API |
| 60 | sdpa_fp8 | audited-skip | 0/4 | — | 无 descale/scale/amax/statistics 等精确 FP8 SDPA API |
| 61 | sdpa_fp8_backward | audited-skip | 0/2 | — | SDK 无 FP8 attention backward API |

汇总：51 个算子的全部当前公共 functional case 已真机认证，10 个算子完成 audited skip。
functional 能力目录为 `1142 = 1052 executed + 90 skipped`，不存在待认证 probe。benchmark
目录为 `1182 = 1125 paired + 57 skipped`，也不存在待认证 probe。所有 comparable 状态均在
真机成对正确性与计时认证通过后写入。

## 5. 实现架构

```text
Frontend Graph
  -> 公共 validation/lowering（不改）
  -> Graph IR schema v3, backend=thead
  -> backends/thead/compiler.py
       -> 严格 family validator
       -> common kernel 优先 / THead 最小 kernel 补充
       -> PPU-aware Triton -> cubin
       -> versioned execution-program artifact
  -> THead ABI v2 plugin
       -> libtriton_jit(CUDA compatibility backend)
       -> caller-owned PPU stream
       -> Graph capture/replay + autotune cache
```

测试侧分离为：

```text
同一输入
  ├─ FlagDNN Graph -> THead plugin/Triton -> output A
  └─ acDNN C/C++ reference plan ----------> output B
                                        stride-aware compare
```

Family-specific kernel 和 reference 位于 `backends/thead/kernels/` 与
`backends/thead/validation/acdnn_*_reference.*`。通用 case accounting、skip schema、paired
benchmark 和 runner 汇总分别由 capability catalog、functional runners、benchmark runner 和
`run_tests_adapter.py` 负责。

## 6. 资格边界

“算子 qualified”表示该算子的全部当前公共 manifest case 已完成认证，不表示未来新增的任意
dtype/layout/shape 均自动支持。精确边界始终由两个 JSON 事实源逐 case 定义：

- pointwise 覆盖公共 FP32/FP16/BF16、dense/NHWC/显式 stride、broadcast 和公开 attribute；
- layout、reduction 和 normalization 覆盖当前 manifest 的全部几何、axis 和 normalized extent；
- MatMul 覆盖当前公开 rank-2、batched、broadcast 以及高维 batch 分段 case；
- convolution 覆盖当前公开 1D/2D/3D、FP32/FP16/BF16、NCHW/channels-last、stride、dilation、
  symmetric/asymmetric padding；非对称 reference 使用纯 acDNN symmetric-superset DAG；
- manifest 之外的新 dtype、stride、broadcast、attribute 或 shape 一律先 fail closed，必须重新
  probe，不能静默复用其他 case 的资格结论。

## 7. 最终验证命令与记录

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

2026-09-06 最终真机结果：

- warnings-as-errors build 成功，重新配置后 `ninja: no work to do`；
- THead CTest：247 项最终闭合为 `231 passed + 16 expected skip`。完整 label run 的唯一问题是
  以旧 300 秒属性启动的 exhaustive Triton compile 超时；属性改为 900 秒后，该项精确复验在
  401.38 秒通过。129 个 integration 全通过，61 个 functional 为 51 passed/10 skip，
  57 个 benchmark 为 51 passed/6 skip；
- `tools/run_tests.py`：overall `passed`、exit code 0，`118 = 102 passed + 16 skipped`；
- functional case：`1142 = 1052 executed + 90 skipped`；benchmark case：
  `1182 = 1125 paired + 57 skipped`；两个目录均为零 probe；
- comparable coverage：51 个形成 pair 的 operator、1125/1125 required pair 全部观测到，2250 条 provider
  record 完整，missing/extra/record/accounting error 均为 0；
- 共解析 147 条 acDNN skip record，字段错误为 0；
- ratio `acdnn_median_us/flagdnn_median_us`：1125 个 case 的几何均值 `1.623491`、中位数
  `0.980640`、最小值 `0.017787`；threshold 为 null，因此不声明性能阈值达标；
- 独立 `compiler_contract.py --case all --compile` 真实 cubin 编译通过（398.59 秒），CTest
  内同一 exhaustive matrix 也通过；production plugin
  的直接 ELF 依赖无 acDNN/BLAS，reference target 只直接引入 acDNN、不引入 BLAS；
- `git diff --check -- backends/thead docs/superpowers` 通过。

## 8. 后续 SDK 升级时的重新认证流程

只有 SDK/header/runtime 或目标设备发生变化时才重新打开 audited-skip 或扩大切片：先在
capability catalog 添加精确候选并观察 RED，再完成 compiler/artifact contract、acDNN reference
真机探针、functional、paired benchmark、JIT/Graph/autotune/install 和全量退出门禁。不能仅因
header 出现 enum 就声明支持，也不能用非 acDNN reference 填补空缺。
