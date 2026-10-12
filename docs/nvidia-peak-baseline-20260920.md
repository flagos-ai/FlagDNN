# 跨平台性能换算基准

[performance-baseline.json](performance-baseline.json) 保存**各平台峰值**和 **NVIDIA 原生参考耗时**。目标平台的本次算子耗时由测试程序提供，不写回文件。

- `metric_units`：指标单位；`platforms.<平台>.peaks`：该设备的峰值，未测为 `null`。
- `reference.cases.<case_name>`：`time_us` 为参考耗时；`tensors` 记录每个输入/输出的 dtype、shape、strides，`attributes` 记录算子参数。存在 `reference_strides` 时表示 cuDNN 使用了原生对照路径中的布局转换，该转换不计时。
- `platforms.<平台>.scaling.case_policies.<op>.scale_metric`：该目标平台显式选用的换算峰值。共享 NVIDIA 记录中的 `scale_metric` 保持原始 `null`，不承载目标平台策略。
- `timing_profile`：计时配置；`validation`：`cpu_full` 全量 CPU 校验、`cpu_sampled` CPU 采样、`bit_exact` 按位校验或 `sanity` 输出写入/有限值/填充区完整性检查。
- `reference.coverage`：覆盖统计与没有可用 NVIDIA 对照的算子；`reference.unavailable_cases`：被目录排除、构建/执行失败或校验失败的用例，不提供可用于验收的时间。

依据 [算子支持表](operator-support.md)，当前包含 **16 项峰值、82 个算子的 3264 个参考用例**。除最初四算子外，已补比较/逻辑/选择、一元运算、激活及反向、布局、矩阵乘、卷积、归一化和注意力的可测缺口；具体 dtype/属性覆盖以 `case_name`、`coverage.catalog_limits` 和不可用记录为准。INT32 额外尝试了共享用例，只有通过精确校验的用例才保存时间，不代表完整 INT32 语义均受支持。

2026-10-12 按[补测清单](nvidia-performance-missing-cases-20261012.md)完成 **371/371 条 NVIDIA 原生采集，0 ERROR**：core 291 条、aux 80 条，其中 `cpu_full` 335 条、`bit_exact` 36 条；每条均保留计时前后及重置输出后回放校验。历史 2893 条有效参考和 491 条不可用记录完整保留，所有平台对象与峰值保持原样。来源见[测量摘要](../tools/performance_baseline/measurements/nvidia-missing-20261012.json)和[原始归档](../tools/performance_baseline/measurements/nvidia-missing-20261012.tar.gz)。这批数据不代表 Hygon 策略已接入或 371 个 Hygon 用例已通过；后续仍需适配、匹配计时 profile 并正式回归。

不可用状态：`excluded` 表示目录过滤，`build_failed` 表示 cuDNN 构建失败，`execution_failed` 表示执行失败，`validation_failed` 表示结果校验失败。原先笼统标为 `ERROR` 的 213 条中，198 条是 INT32 经 FLOAT 参考路径产生精度差异，15 条无法构建执行计划；这些记录没有 `time_us`，不能参与换算，也不能据此判断 FlagDNN 实现失败。

峰值口径：取扫描配置中最高的中位吞吐；FMA/MAD 每个标量计 2 次运算，稠密 GEMM 计 `2×M×N×K`，带宽计实际读取与写入的逻辑字节数（十进制 TB/s）。

其他平台需填写自己的 `device`、`measured_on` 和 `peaks`，并在自己的 `scaling.case_policies` 中显式登记所选算子的换算策略及依据；峰值定义及单位必须与 NVIDIA 一致。仅填写峰值不会启用等效性能，也不能借用 Hygon 策略。运行时匹配相同用例和计时配置，按 **`T_limit = T_nv × P_nv / (0.8 × P_target)`** 计算上限，再与本次 `measured_us` 比较。

| 算子类型 | 峰值选择建议（先确认瓶颈） |
|---|---|
| 布局/复制、索引生成 | 显存受限时分别考虑 `hbm_copy`、`hbm_write`。 |
| 比较、逻辑等逐元素运算 | 显存受限时考虑 `hbm_triad`；小用例可能受缓存或调度限制。 |
| 激活、幂、除法、归一化等 | 检查向量计算或带宽；cuDNN 低精度输入不等于低精度计算。特殊函数按 FMA 峰值缩放只是近似。 |
| MatMul、卷积、Attention | 张量计算受限时选实际计算精度对应的 `tensor_*`；显式 TF32 与普通 FP32 用例分开记录。 |

NVIDIA 性能计数器权限受限（`ERR_NVGPUCTRPERM`），因此不把瓶颈猜测写成实测结论。下述 Hygon 等效性能覆盖的 40 个算子显式采用 FP32 FMA 或逻辑带宽代理，各代理的局限仅记录在 `platforms.hygon.scaling.case_policies`。共享 `reference` 保留历史 2893 条 NVIDIA 记录并追加本轮 371 条，3264 条用例的 `scale_metric` 均为 `null`；目标平台策略不改写这些参考记录。混合 FP8 峰值 `tensor_fp8_e5m2_e4m3` 对应 E5M2 × E4M3；不能直接当作任意 FP8/MXFP8 配置的峰值。

读取示例（先显式配置目标平台策略、填写目标设备峰值；缺少该平台的算子策略时报错，不回退到 Hygon 或共享参考）：

```python
import json
import math


def check_performance(data, platform, case_name, measured_us, traffic_ratio=1.0):
    ref = data["reference"]
    case = ref["cases"][case_name]
    try:
        metric = data["platforms"][platform]["scaling"]["case_policies"][case["op"]]["scale_metric"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"目标平台 {platform} 尚未为 {case['op']} 显式配置换算策略") from error
    if not isinstance(metric, str) or not metric:
        raise ValueError("目标平台策略缺少有效 scale_metric")
    p_nv = data["platforms"][ref["platform"]]["peaks"][metric]
    p_target = data["platforms"][platform]["peaks"][metric]
    t_nv, alpha = case["time_us"], data["min_relative_efficiency"]
    values = (t_nv, p_nv, p_target, measured_us, alpha, traffic_ratio)
    if any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in values):
        raise ValueError("缺少有效峰值、耗时或换算参数")
    if alpha > 1 or (traffic_ratio != 1 and not metric.startswith("hbm_")):
        raise ValueError("达标比例应在 (0,1] 内，流量修正仅适用于带宽模型")
    limit = t_nv * p_nv / (alpha * p_target) * traffic_ratio
    return {"limit_us": limit, "passed": measured_us <= limit}

with open("docs/performance-baseline.json") as f:
    data = json.load(f)
# result = check_performance(data, platform, case_name, measured_us)
```

`traffic_ratio` 默认 1；仅在带宽模型成立且实际读写量不同的情况下，传入目标/NVIDIA 的流量比。例：两个 FP32 输入的比较运算，cuDNN 输出 1 bit、目标输出 1 byte，比值为 `72/65`；FP16/BF16 为 `40/33`。布尔参考位宽见各 tensor 的 `reference_storage_bits`。

计时排除建图、拷贝和校验。`native` 为预热 10 次、采样 20 批、Graph 每批 50 次；`attention` 为 5 次、10 批、每批 20 次。2026-10-12 新增的 24 条 batchnorm 训练用例使用独立 `batchnorm_training` profile，同为预热 5 次、采样 10 批、每批 20 次，不修改已有 profile。三者均有一次 Graph 预热回放，并取 nearest-rank P50；目标平台须匹配对应配置。`sanity` 不代表完整数值正确性。未提供 NVIDIA 对照的 `genstats`、`rng`、`rope`、`rope_backward`、`moe_grouped_matmul_bwd` 不使用其他算子的时间替代。

复测命令见 [工具目录](../tools/performance_baseline/README.md)。工具：[向量/带宽](../tools/performance_baseline/benchmark_nvidia_vector_memory.cu)、[张量](../tools/performance_baseline/benchmark_nvidia_tensor.py)、[核心用例](../tools/performance_baseline/benchmark_nvidia_cudnn.cpp)、[显式 TF32](../tools/performance_baseline/benchmark_nvidia_precision.cpp)、[扩展/Attention](../tools/performance_baseline/benchmark_nvidia_extended.py)、[属性/整数/FP8 复制](../tools/performance_baseline/run_benchmark_nvidia_pointwise_gaps.py)。Hygon 在下述范围内仅对缺少可用 hipDNN 性能对照的用例接入此 JSON；已有原生性能对照继续保留。

## Hygon div 等效性能（2026-10-10）

先在空闲 GPU 0（HIP 设备名 `BW`，`gfx936:sramecc+:xnack-`，80 CU，64 GiB 显存）实测峰值，再填写 JSON 并接入 div。扫描 128/256 threads、2/4/8 blocks/CU，内存每缓冲区 256/1024 MiB，均大于 8 MiB L2；每配置预热 20 次、5 轮各 20 样本。54 组配置共 5400 个样本通过校验，FP32 FMA 全量检查，内存窗口和 read checksum 采样检查。[测量摘要](../tools/performance_baseline/measurements/hygon-20261010.json) 保留全部配置的中位耗时、吞吐和原始 CSV 哈希。

| 指标 | 实测峰值 |
|---|---:|
| `vector_fp32` | 14.60953232 TFLOP/s |
| `hbm_triad` | 1.134362154 TB/s |
| `hbm_copy` | 1.132065845 TB/s |
| `hbm_read` | 1.189452346 TB/s |
| `hbm_write` | 1.125421181 TB/s |

Hygon 的 div 策略为对应的 45 个参考用例统一选 `vector_fp32`，不修改这些共享参考记录。cuDNN 的 `benchmark/pointwise.cpp` 和 Hygon 的 pointwise dispatch/codegen 均指定 FP32 计算，包含 FP16/BF16 存储输入；因此不能按存储 dtype 选低精度 FMA 峰值。这里使用 **FP32 FMA 峰值作为除法性能的近似代理**，并在 JSON 的 `platforms.hygon.scaling.case_policies.div` 标注。它不是已证实的瓶颈结论：倒数/除法的专用吞吐、缓存和小用例启动延迟均可能影响结果。本次带宽实测作为诊断数据，不用于宣称全部 div 用例受 HBM 限制。未测的其他精度峰值仍为 `null`。

`benchmark.hygon.div` 从 JSON 匹配同名用例，并核对 dtype、shape、strides、设备型号/架构、计算参数和计时配置。全量 CPU 校验及 padding 检查均在计时外；除原计时前后校验外，计时结束后还重置逻辑输出并重新回放 Graph，避免空图或部分写入复用旧输出。HIP Graph 使用 10 次预热、每批 50 次、20 批、一次 Graph 预热，取 nearest-rank P50。按 `T_limit = T_nv × 60.424 / (0.8 × 14.60953232)` 记录性能评价，低于基准标记为 `BELOW_BASELINE`，不影响原生测试的成功退出。缺基线、峰值无效、配置不匹配、精度错误或运行异常仍直接报错，不转为 SKIP。

本次 GPU 0 验证执行 45/45 个用例、0 SKIP，全部通过前后 CPU 校验。首轮性能为 42 通过、3 失败，失败均为 `2x256x63x65` 的 FP32/FP16/BF16，效率分别为 0.79455/0.78125/0.77381；各定点复测一次后为 0.83031/0.82970/0.81981，均通过，表明这些用例在 0.8 门槛附近存在波动。门槛和 NVIDIA 耗时保持原值，两轮证据保存在 `build/hygon/performance-baseline-20261010/div-result.json` 及其引用的日志中。

输出 schema v4 的 `samples` 仅包含本次 FlagDNN 实测；`equivalent_baseline` 保存原 NVIDIA 耗时、峰值、等效耗时、上限、相对效率和基准评价，不伪造参考样本。`tools/run_tests.py` 在汇总中保留这些信息。相对效率为 `T_nv × P_nv / (P_hygon × measured_us)`；JSON 中的 0.8 用于基准评价，**未指定 `--min-speedup` 时只报告性能，不因未达基准而把测试标为 Failed 或返回非零**。此时 `equivalent_performance.gate_enabled=false`、`gate_passed=null`，明细中的 `performance_gate.enabled=false`；`equivalent_baseline.passed=false` 仍如实保留，不代表执行失败。显式传入 `--min-speedup` 才启用强制性能门槛，等效用例要求 `max(0.8, --min-speedup)`，不会放宽基准最低值；原生对照仍按原生加速比判定。默认读取源码树中的基线，可用 `FLAGDNN_PERFORMANCE_BASELINE=/absolute/path/baseline.json` 指定其他已测设备基线；设备信息仍须与运行设备一致。更新此逻辑后须重新构建 benchmark 可执行文件，不能仅更新 Python 脚本。

```bash
cmake --build build/hygon --target flagdnn_benchmark_hygon_div flagdnn_benchmark_hygon_baseline_contract -j8
ctest --test-dir build/hygon --output-on-failure -R '^integration\.hygon\.div_performance_baseline$'
ctest --test-dir build/hygon -V -R '^benchmark\.hygon\.div$'
python3 tools/run_tests.py --platform hygon --ops div --suites benchmark --device 0 --output build/hygon/div-equivalent-summary.json
# 可选：显式启用性能门槛，未达标时返回失败。
python3 tools/run_tests.py --platform hygon --ops div --suites benchmark --device 0 --min-speedup 0.8
```

## Hygon 缺失性能测试的等效补齐

沿用上述五项 Hygon 实测峰值，新增范围没有依赖尚未测量的 FP16、BF16、INT32 向量峰值。仅在 `platforms.hygon.scaling.case_policies.<op>` 中为以下 40 个算子登记 Hygon 专属策略，其他平台不会自动启用。1597 是接入当时这 40 项策略所覆盖的历史 NVIDIA 记录数，不是本轮新增记录数；当时未修改共享 `reference`。这些记录包含额外属性/整数探针，数量不等于 Hygon 默认测试目录大小。

| 指标 | 算子 |
|---|---|
| `hbm_copy` | identity（仅 FP8 E4M3/E5M2）、reshape、transpose、slice、concatenate、logical_not、neg、ceil、floor、leaky_relu |
| `hbm_write` | gen_index |
| `hbm_triad` | cmp_eq/neq/gt/ge/lt/le、logical_and/or、binary_select、leaky_relu_backward |
| `vector_fp32` | div、pow、mod、sqrt、rsqrt、reciprocal、exp、log、erf、sin、cos、tan、elu_backward、gelu、gelu_backward、gelu_approx_tanh、gelu_approx_tanh_backward、softplus、softplus_backward |

这是一组显式近似模型，不是已证实的 HBM 或 FMA 瓶颈结论；小形状重复缓冲区、缓存、启动开销和特殊函数指令吞吐都可能影响换算。选择指标不依赖本次 FlagDNN 耗时，也不通过修改门槛消除性能失败。

带宽模型按输入/输出逻辑元素数及各自位宽计算 `traffic_ratio = target_bits / reference_bits`，不计 padding；相同存储位宽时为 1。纯布尔逻辑为 8；比较 FP32 为 72/65、FP16/BF16 为 40/33；binary_select FP32 为 104/97、FP16/BF16 为 56/49。完整公式为：

```text
equivalent_time_us = T_nv × P_nv / P_hygon × traffic_ratio
limit_us = equivalent_time_us / min_relative_efficiency
relative_efficiency = equivalent_time_us / measured_us
```

FP32 FMA 模型要求 `traffic_ratio = 1`。所有等效用例均做计时前后全量 CPU 结果校验及 padding 检查；NVIDIA 原始 `validation` 通过结果中的 `reference_validation` 保留，尤其布局、concatenate、gen_index 的 `sanity` 不能被描述成 NVIDIA 全量正确性校验。

实际注册的布局 `.copy` 和 logical_not `.boolean` 测试已采用与 NVIDIA 相同的原生目录，分别各 36 和 12 条用例；不借用这些时间验收另一个大形状目录。concatenate 有 42 条浮点参考，gen_index 有 28 条（FP32/INT32）。`elu_backward`、`softplus`、`softplus_backward` 保留已有 hipDNN 对照，仅在静态能力、构建、执行探测或 Graph 捕获明确不支持时回退；数值错误或设备执行错误不转成等效通过。

concatenate 的 70 条 INT32/BOOL/FP8 扩展用例仍没有可用 NVIDIA 时间，列为 `EXCLUDED`，定点选择时为 `SKIP`。logical_and/or 的附加 `.boolean` 小形状目录现已新增 24 条 NVIDIA 参考，但 Hygon 对应路径仍按旧逻辑报告缺参考 `SKIP`，尚需接入新记录；其原有主性能目录继续运行等效测试。缺失文件、错误峰值、同名用例参数不匹配仍是硬错误，不作为缺参考跳过。

2026-10-12 的 CPU 基线契约检查了 40 个算子的 **1570 条实际目录用例**（原有 1546 条及新增接入的 24 条 FP8 identity），并通过错误参数、流量、设备和来源信息的反例检查。identity 的两种 FP8 描述符已由独立探针确认被当前 hipDNN 返回 `NOT_SUPPORTED`；等效路径复用既有 NVIDIA 六值输入，计时前后及重置输出后的 Graph 回放均按原始字节校验复制，并检查输入未变和 padding。此路径不启用其他 identity dtype，不改动 NVIDIA 记录或任何平台的峰值。

2026-10-10 的历史定点验证在 GPU 0 按当时的算子顺序执行 **110 个定点用例**，覆盖各算子的默认 dtype，选取 FP32 小形状、FP16 中间形状、BF16 末尾大形状（布尔及索引类型按各自目录选择）。其中 104 个等效用例均通过计时前后全量 CPU 校验，6 个原生用例通过 hipDNN 对照校验；所有输出记录通过 JSON Schema 检查。当时结果为 **101 通过、9 个性能失败、0 SKIP**，这不是当前完整目录的通过结论；当前默认命令未指定性能阈值时只报告低于基线，不因性能数值失败。

9 个失败均为 `2x256x63x65` 的 BF16：六个比较算子的相对效率为 0.616–0.623，binary_select 为 0.584，leaky_relu_backward 为 0.538，softplus 为 0.754，均低于 0.8。没有调整门槛或参考时间，也没有用复测覆盖失败。完整样本、原生/等效分类及失败信息保存在 `build/hygon/performance-baseline-20261010/expanded/smoke-result.json` 和它引用的 110 份原始日志中。

定点复测示例（完整目录运行不设置此过滤变量）：

```bash
FLAGDNN_BENCHMARK_CASE=cmp_eq_perf_bfloat16_2x256x63x65_by_2x256x63x65 \
  ctest --test-dir build/hygon -V -R '^benchmark\.hygon\.cmp_eq$'
```
