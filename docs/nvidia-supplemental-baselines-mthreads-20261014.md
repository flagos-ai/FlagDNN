# MThreads 等效性能所需 NVIDIA 补测任务（2026-10-14）

本文记录需要在 NVIDIA 环境完成的工作，当前尚未执行这些补测。目标是为以下 MThreads 批量测试提供可验证的等效性能参考：

```bash
python3 tools/run_tests.py --dump-output \
  --output-dir logs_result_20260920_flagdnn --gpus 0 --appoint
```

计算规则见 [峰值与等效性能说明](equivalent-performance-baseline.md)，现有数据见 [performance-baseline.json](performance-baseline.json)。本清单依据仓库 `092d6b72d66c585ac025c22a03aa4dcd65cf30d4` 及 2026-10-13 的 MThreads 适配工作区核对；交付新测量时还需要保存实际使用的源码快照和哈希。

## 1. 需要补齐的数据

原先因 muDNN 不支持而跳过的 204 条性能用例中，99 条已有有效 NVIDIA 参考，并已在 MTT S5000 GPU 0 完成等效性能验证。剩余 **105 条**如下：

| 算子及目录 | 数量 | 当前 NVIDIA 状态 | 需要完成的工作 |
|---|---:|---|---|
| `elu_backward` 的 native 小形状及属性用例 | 45 | `not_collected` | 扩展采集器，验证并采集同名、同 dtype、同布局、同属性用例 |
| `softplus_backward` 的 native 小形状及属性用例 | 45 | `not_collected` | 同上 |
| `pow` 的普通 INT32 用例 | 14 | `validation_failed` | 先解决精确整数参考实现，再采样 |
| `pow_int32_broadcast_padded_2x3x17` | 1 | `build_failed` | 先解决原始广播及 padded 布局的参考执行，再完成精确验证和采样 |

ELU/softplus 已有的各 45 条 `*_perf_*` 大形状基线不属于本次缺口，不能借用其时间替代 native 小形状。上述数量是需要补测的目录数量，不代表已成功执行或一定被 cuDNN 支持。

### 1.1 完整目录的确定规则

共同的 14 个普通形状为：

```text
1x1x16 2x4x8 1x4x8x16 2x4x8x16 1x3x17 3x5x7
1x3x5x7 2x3x5x7 1x1x31 1x1x32 1x1x33 2x7x16
1x1x1023 1x1x1025
```

- ELU backward：`elu_backward_{fp32,fp16,bfloat16}_{shape}`，共 42 条；再加 `elu_backward_attributes_{fp32,fp16,bfloat16}_1x1x65`，共 3 条。
- Softplus backward：`softplus_backward_{fp32,fp16,bfloat16}_{shape}`，共 42 条；再加 `softplus_backward_attributes_{fp32,fp16,bfloat16}_1x1x65`，共 3 条。
- INT32 pow：`pow_int32_{shape}`，共 14 条；再加 `pow_int32_broadcast_padded_2x3x17`，共 1 条。

第 4 节提供生成全部 105 个准确名称并检查采集目录的脚本。该集合已与 MThreads 缺口审计逐条核对。

### 1.2 属性、输入和布局要求

用例定义以 [tests/common/pointwise.cpp](../tests/common/pointwise.cpp) 的 `make_binary_pointwise_cases` 为准，不能只根据名字重建为默认连续张量。

- 普通反向用例：两个输入均为 `kReal`，使用目录中的默认属性。
- 属性反向用例：第一个输入 DY 为 `kReal`，第二个输入 X 为 `kScaled`；ELU 的 `elu_alpha=0.25`，softplus 的 `softplus_beta=0.5`。必须保留属性 flags。
- FP16/BF16：先按实际存储 dtype 编码，再解码后交给 CPU oracle；预期输出也按输出 dtype 量化。保留用例本身的误差容限。
- INT32：使用精确整数 oracle，不能经由浮点数组比较整数结果。
- 每个张量的 UID、role、dtype、shape、strides、binding byte offset 都必须入档；保留输入字节与输出 padding 的校验。

INT32 pow 的 padded 用例必须保留以下布局，binding offset 和 UID 从共享目录导出：

| 张量 | shape | strides |
|---|---|---|
| 底数输入 | `[2,3,17]` | `[137,41,2]` |
| 指数输入 | `[1,3,17]` | `[127,37,1]` |
| 输出 | `[2,3,17]` | `[151,43,2]` |

静默改为连续布局会改变测量任务，不能将其耗时写回原 case。

## 2. NVIDIA 环境和采集器准备

现有 NVIDIA 峰值来自 **NVIDIA H100 80GB HBM3，700 W**；当前参考声明为 **cuDNN 9.24.0**。优先在匹配设备、功耗设置及软件环境上补采，并独占所选 GPU。记录实际 GPU/架构、驱动、CUDA、cuDNN、编译器、功耗和时钟状态。若使用其他 GPU 或参考提供方，需要另建对应证据和峰值，不能把新设备的时间与旧 H100 峰值直接组合。

现有工具：

- [benchmark_nvidia_pointwise_gaps.cpp](../tools/performance_baseline/benchmark_nvidia_pointwise_gaps.cpp)：C++ 采集器，支持 `--op`、`--case`、`--group`、`--probe-first`、`--list`、`--catalog`。
- [build_benchmark_nvidia_pointwise_gaps.sh](../tools/performance_baseline/build_benchmark_nvidia_pointwise_gaps.sh)：编译上述程序，需要已配置的 NVIDIA 构建目录、CUDA、cuDNN 及 cuDNN frontend headers。
- [run_benchmark_nvidia_pointwise_gaps.py](../tools/performance_baseline/run_benchmark_nvidia_pointwise_gaps.py)：已有 Python 包装器不支持 `--op`/`--case`；下面的逐 case 命令直接调用 C++ 程序。

**当前采集器还不能直接补齐 90 条 ELU/softplus backward 用例。** 在 NVIDIA 开发环境中先完成以下修改：

1. 扩展 `catalog()`：以 `FLAGDNN_POINTWISE_ELU_BWD`、`FLAGDNN_POINTWISE_SOFTPLUS_BWD` 调用共享 `make_binary_pointwise_cases`，各保留完整 45 条，不能沿用仅筛选 `_attributes_` 的旧逻辑。
2. 扩展 `emit_catalog()` 和 `emit()`：记录 `elu_alpha`、`softplus_beta`、完整属性 flags、逐输入 `input_domains`、输入生成器身份及版本。普通 `[Real, Real]` 与属性 `[Real, Scaled]` 必须区分。
3. 浮点反向用例按实际量化后的输入做 CPU 全量对照，使用 `reference/` 下的独立实现。必要时扩展 CPU oracle 的接入，不能把设备结果作为预期值。
4. 当前 `measure()` 仅在计时前检查数值，采样后直接结束。补上最后一次采样后的全量输出检查、输入不变和 padding 检查，以及重置输出后的一次 Graph 回放和全量检查。重置必须覆盖所有逻辑输出；写入值经实际 dtype 编码后，每个元素都必须不能通过对应的 CPU 正确性检查，浮点值还须超出原数值容差。仅与预期值略有不同不足以排除空回放或部分回放。
5. 增加三阶段校验结果和覆盖数量的输出字段；任一阶段失败，均不得输出可入库的有效时间。所有 CPU 校验、读回和重置必须在计时区间之外。

以上为待完成的 NVIDIA 工作，本次 MThreads 修改没有实现或验证这些 NVIDIA 采集器扩展。

## 3. INT32 pow：先证明参考实现正确，再测性能

### 3.1 现有失败的原因

14 条普通用例在计时前的精确校验中出现 `16777217 ** 1` 实际得到 `16777216`。检查 [cudnn_pointwise.cpp](../backends/nvidia/validation/functional/cudnn_pointwise.cpp) 的 `build_pointwise_reference`：当前 POW 的 graph intermediate/compute dtype 为 FLOAT，`2^24+1` 经过 FP32 表示时丢失精度。CPU 整数 oracle 的预期值正确。

第 15 条 padded 广播用例在构建 cuDNN graph/engine 时失败，已有日志包含 `CUDNN_STATUS_NOT_SUPPORTED_LAYOUT`。这些记录均无可复用的有效耗时。

### 3.2 NVIDIA 环境中的排查顺序

1. 用第 4.1 节命令保存最小精度失败及 padded 布局失败的原始输出，记录 cuDNN 错误、设备和软件版本。
2. 对 INT32 POW 单独探测精确计算路径：检查输入、输出、中间值、compute dtype 和选中的 engine。尝试 INT32 compute/intermediate 只是能力探测，不能假定当前 cuDNN 支持 POW。避免全局更改其他算子的计算类型。
3. 用 [reference/cpu/pointwise.cpp](../reference/cpu/pointwise.cpp) 的 `pointwise_integer_reference` 做逐元素精确比较。保留 `2^24+1`、负底数、负指数、`0**0`、大指数和按 32 位回绕等仓库已定义语义；先通过边界测试，再运行全部 15 条目录用例。
4. 对 padded 广播用例验证原始 shape、strides、offset；若 engine 不支持该布局，记录准确不支持原因，不能改成 dense 后沿用原名称入库。
5. 精确数值、输入不变、输出 padding 及重置后的 Graph 回放全部通过后，才进行正式计时并交付样本。

不能通过减小输入范围、删除边界值、放宽整数容差、使用 FP32 POW 耗时或将 `validation_failed` 改为成功来填补基线。若 cuDNN 确实没有符合语义的路径，应保留“不支持”的结论。若后续决定采用另一个 NVIDIA GPU 参考实现，必须明确提供方、源码、语义、计时方式及独立正确性验证，并扩展基线契约后再使用，不能标成 cuDNN 数据。

### 3.3 还需解决输入分布和缩放指标

当前 [MThreads pointwise_runner.cpp](../backends/mthreads/validation/functional/pointwise_runner.cpp) 的 POW 输入是底数 `(i % 9) - 4`、指数 `i % 6`；NVIDIA gaps 采集器使用共享 `pointwise_integer_input` 的整数边界表。POW 执行时间可能依赖指数，两端仅名称、dtype 和布局相同还不足以证明耗时可比。

后续接入前必须统一并记录两端的性能输入生成规则，同时保留强整数边界功能测试。若边界验证与性能数据使用不同分布，要分别命名、记录并让匹配器检查，不能在同一 case 名称下静默替换输入以绕过已知错误。

当前 `platforms.mthreads` 未启用 INT32 pow 缩放策略，`vector_int32` 峰值也未测量。NVIDIA 已有 H100 `vector_int32=30.733` TOP/s，但不能因此自动选择该指标：需先根据最终实现确定合理的缩放代理、说明局限，并在 MThreads 补测对应指标。若选用其他代理，也必须提供依据和两端同口径数据。现有浮点反向算子的 `vector_fp32` 策略不能直接套到 INT32 pow。

## 4. NVIDIA 主机上的执行步骤

### 4.1 现有工具即可执行的能力探测

以下命令在仓库根目录运行；替换真实安装路径，确保 `NVIDIA_BUILD_DIR` 已配置且包含 NVIDIA validation targets：

```bash
export CUDA_ROOT=/usr/local/cuda
export CUDNN_ROOT=/path/to/cudnn
export CUDNN_FRONTEND_INCLUDE_DIR=/path/to/cudnn_frontend/include
export NVIDIA_BUILD_DIR=build/nvidia
out=$(mktemp -d /tmp/flagdnn-nv-supplement-XXXXXX)

git rev-parse HEAD > "$out/source-head.txt"
git diff --binary > "$out/working-tree.patch"
nvidia-smi -q > "$out/nvidia-smi-before.txt"
bash tools/performance_baseline/build_benchmark_nvidia_pointwise_gaps.sh "$out/nv_gaps"

# 这两个模式不初始化 CUDA；输出准确目录与元数据。
"$out/nv_gaps" --op pow --list > "$out/pow-list.tsv"
"$out/nv_gaps" --op pow --catalog > "$out/pow-catalog.jsonl"

# 原实现预计失败；分别保存退出码，失败数据不得入库。
CUDA_VISIBLE_DEVICES=0 "$out/nv_gaps" --case pow_int32_1x1x16 \
  > "$out/pow-int32-probe.jsonl" 2> "$out/pow-int32-probe.log"
printf '%s\n' "$?" > "$out/pow-int32-probe.exit"
CUDA_VISIBLE_DEVICES=0 "$out/nv_gaps" --case pow_int32_broadcast_padded_2x3x17 \
  > "$out/pow-padded-probe.jsonl" 2> "$out/pow-padded-probe.log"
printf '%s\n' "$?" > "$out/pow-padded-probe.exit"
```

上面的探测命令应在未启用 `set -e` 的交互 shell 运行，或用显式 `if` 保存失败退出码，确保两个失败都能留档。`git diff` 不包含未跟踪文件，新加或修改的采集器、CPU reference、NVIDIA builder 还须另外保存源码快照及 SHA256。

### 4.2 扩展采集器后，校验完整 105 条目录

本节需要先完成第 2 节的目录扩展并重新编译；当前原始采集器执行 `--op elu_backward` 或 `--op softplus_backward` 会报无匹配用例。沿用上面的 `out` 目录：

```bash
python3 - <<'PY' > "$out/required-cases.txt"
shapes = """1x1x16 2x4x8 1x4x8x16 2x4x8x16 1x3x17 3x5x7
1x3x5x7 2x3x5x7 1x1x31 1x1x32 1x1x33 2x7x16
1x1x1023 1x1x1025""".split()
names = []
for op in ("elu_backward", "softplus_backward"):
    for dtype in ("fp32", "fp16", "bfloat16"):
        names += [f"{op}_{dtype}_{shape}" for shape in shapes]
        names.append(f"{op}_attributes_{dtype}_1x1x65")
names += [f"pow_int32_{shape}" for shape in shapes]
names.append("pow_int32_broadcast_padded_2x3x17")
assert len(names) == len(set(names)) == 105
print("\n".join(sorted(names)))
PY

# 放在子 shell 中，以免改变上面能力探测的退出码处理方式。
(
  set -e
  for op in elu_backward softplus_backward pow; do
    "$out/nv_gaps" --op "$op" --list > "$out/$op-list.tsv"
    "$out/nv_gaps" --op "$op" --catalog > "$out/$op-catalog.jsonl"
  done
  cat "$out/elu_backward-list.tsv" "$out/softplus_backward-list.tsv" \
    "$out/pow-list.tsv" | cut -f3 | LC_ALL=C sort > "$out/actual-cases.txt"
  diff -u "$out/required-cases.txt" "$out/actual-cases.txt"
)
```

目录检查必须无差异：45 条 ELU、45 条 softplus、15 条 INT32 pow。元数据另行检查，名称相等不能替代布局、属性及输入域的一致性。

### 4.3 正式逐用例采集

完成采集器校验扩展后，可先补 90 条浮点反向用例；INT32 pow 在第 3 节问题解决之前继续作为能力探测并保留失败。以下循环不会把失败转成通过，而是为每个 case 保留独立退出码：

```bash
while IFS= read -r case; do
  if CUDA_VISIBLE_DEVICES=0 "$out/nv_gaps" --case "$case" \
      > "$out/$case.jsonl" 2> "$out/$case.log"; then
    printf '0\n' > "$out/$case.exit"
  else
    rc=$?
    printf '%s\n' "$rc" > "$out/$case.exit"
  fi
done < "$out/required-cases.txt"
nvidia-smi -q > "$out/nvidia-smi-after.txt"
```

保持现有 `native` 计时口径：10 次预热、20 个样本、每样本 CUDA Graph 内 50 次执行、1 次 Graph 预热回放；使用设备事件时间，每个样本除以 50。保留原始 batch 时间和逐次执行时间，排序后第 10 个样本为 nearest-rank P50、第 18 个为 P90；不使用中间两个样本的均值替代该 P50 定义。

## 5. 交付与验收

每个有效 case 至少交付：

- 完整目录元数据：准确名称、op/mode、tensor UID/role/dtype/shape/strides/offset、属性和 flags、逐输入域及生成规则。
- NVIDIA 设备、架构、软件版本、测量时间、源码 revision、工作区修改、源码快照和 SHA256。
- 三阶段 CPU 全量校验、输入和 padding 校验、逻辑输出覆盖数量；浮点误差或 INT32 精确比较结果。
- 完整 20 个有效样本、Graph 执行数、warmup/profile、P50/P90、构建耗时、workspace、原始 stdout/stderr 和退出码。
- 失败的 case 继续交付准确失败阶段和原因；若换参考提供方，独立记录身份，不能沿用 `provider=cudnn`。

将经过验证的实测数据合入 `performance-baseline.json` 时，仅新增有效 case，并更新对应 coverage/测量来源。历史失败证据保留在原始测量归档；当前不可用目录仅在新证据成立后更新。不得覆盖已有有效 NVIDIA 记录、修改 Hygon 的数据/策略，或降低 0.8 效率门槛来制造通过。

补采成功后 MThreads 仍需完成接入：核对 90 条 native case 的平台内 matcher，补齐 ELU/softplus 属性用例的 `[Real, Scaled]` 检查，并扩展结果 schema/adapter 的准确名称范围及契约；对 INT32 pow 完成输入规则对齐、对应峰值实测与策略定义、参考提供方检查，再进行 GPU 验证。当前完整接入并验证通过的范围为此前的 99 条等效性能路径，单独添加 JSON 记录不会使其余用例自动通过。

最终验收以原始批量命令的真实输出为准：全部要求的功能和性能用例执行、无不支持跳过、无缺基线错误；并分别保留原生性能比较和等效性能效率判定。未指定 `--min-speedup` 时，批量执行状态与效率是否达到 0.8 是两个报告维度，应同时检查，不能仅看进程退出码。
