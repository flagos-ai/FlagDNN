# 跨平台性能换算基准

[performance-baseline.json](performance-baseline.json) 保存**各平台峰值**和 **NVIDIA 原生参考耗时**。目标平台的本次算子耗时由测试程序提供，不写回文件。

- `metric_units`：指标单位；`platforms.<平台>.peaks`：该设备的峰值，未测为 `null`。
- `reference.cases.<case_name>`：`time_us` 为参考耗时；`tensors` 记录每个输入/输出的 dtype、shape、strides，`attributes` 记录算子参数。存在 `reference_strides` 时表示 cuDNN 使用了原生对照路径中的布局转换，该转换不计时。
- `scale_metric`：换算所用峰值；`timing_profile`：计时配置；`validation`：`cpu_full` 全量 CPU 校验、`cpu_sampled` CPU 采样、`bit_exact` 按位校验或 `sanity` 输出写入/有限值/填充区完整性检查。
- `reference.coverage`：覆盖统计与没有可用 NVIDIA 对照的算子；`reference.unavailable_cases`：被目录排除、构建/执行失败或校验失败的用例，不提供可用于验收的时间。

依据 [算子支持表](operator-support.md)，当前包含 **16 项峰值、65 个算子的 2893 个参考用例**。除最初四算子外，已补比较/逻辑/选择、一元运算、激活及反向、布局、矩阵乘、卷积、归一化和注意力的可测缺口；具体 dtype/属性覆盖以 `case_name`、`coverage.catalog_limits` 和不可用记录为准。INT32 额外尝试了共享用例，只有通过精确校验的用例才保存时间，不代表完整 INT32 语义均受支持。

不可用状态：`excluded` 表示目录过滤，`build_failed` 表示 cuDNN 构建失败，`execution_failed` 表示执行失败，`validation_failed` 表示结果校验失败。原先笼统标为 `ERROR` 的 213 条中，198 条是 INT32 经 FLOAT 参考路径产生精度差异，15 条无法构建执行计划；这些记录没有 `time_us`，不能参与换算，也不能据此判断 FlagDNN 实现失败。

峰值口径：取扫描配置中最高的中位吞吐；FMA/MAD 每个标量计 2 次运算，稠密 GEMM 计 `2×M×N×K`，带宽计实际读取与写入的逻辑字节数（十进制 TB/s）。

其他平台只需填写自己的 `device`、`measured_on` 和 `peaks`，峰值定义及单位必须与 NVIDIA 一致。运行时匹配相同用例和计时配置，按 **`T_limit = T_nv × P_nv / (0.8 × P_target)`** 计算上限，再与本次 `measured_us` 比较。

| 算子类型 | 峰值选择建议（先确认瓶颈） |
|---|---|
| 布局/复制、索引生成 | 显存受限时分别考虑 `hbm_copy`、`hbm_write`。 |
| 比较、逻辑等逐元素运算 | 显存受限时考虑 `hbm_triad`；小用例可能受缓存或调度限制。 |
| 激活、幂、除法、归一化等 | 检查向量计算或带宽；cuDNN 低精度输入不等于低精度计算。特殊函数按 FMA 峰值缩放只是近似。 |
| MatMul、卷积、Attention | 张量计算受限时选实际计算精度对应的 `tensor_*`；显式 TF32 与普通 FP32 用例分开记录。 |

本机性能计数器权限仍受限（`ERR_NVGPUCTRPERM`），因此不把瓶颈猜测写成实测结论，`scale_metric` 暂留 `null`。混合 FP8 峰值 `tensor_fp8_e5m2_e4m3` 对应 E5M2 × E4M3；不能直接当作任意 FP8/MXFP8 配置的峰值。

读取示例（先确认指标、填写目标设备峰值）：

```python
import json
import math


def check_performance(data, platform, case_name, measured_us, traffic_ratio=1.0):
    ref = data["reference"]
    case = ref["cases"][case_name]
    metric = case["scale_metric"]
    if metric is None:
        raise ValueError("请先确认该用例的瓶颈并填写 scale_metric")
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

计时排除建图、拷贝和校验。本基准将普通用例统一为预热 10 次、Graph 每批 50 次、采样 20 批，部分用例与原测试默认配置不同，目标平台需匹配；Attention 沿用原生配置 5/20/10，均取 nearest-rank P50。`sanity` 不代表完整数值正确性。未提供 NVIDIA 对照的 `genstats`、`rng`、`rope`、`rope_backward`、`moe_grouped_matmul_bwd` 不使用其他算子的时间替代。

复测命令见 [工具目录](../tools/performance_baseline/README.md)。工具：[向量/带宽](../tools/performance_baseline/benchmark_nvidia_vector_memory.cu)、[张量](../tools/performance_baseline/benchmark_nvidia_tensor.py)、[核心用例](../tools/performance_baseline/benchmark_nvidia_cudnn.cpp)、[显式 TF32](../tools/performance_baseline/benchmark_nvidia_precision.cpp)、[扩展/Attention](../tools/performance_baseline/benchmark_nvidia_extended.py)、[属性/整数/FP8 复制](../tools/performance_baseline/run_benchmark_nvidia_pointwise_gaps.py)。现有平台测试入口尚未接入此 JSON 替代基准。
