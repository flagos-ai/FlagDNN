# 跨平台等效性能基准

等效性能用 NVIDIA 参考耗时和两平台实测峰值，估算目标平台的性能基准。它依赖显式选择的计算或带宽模型，不能代替目标平台的正确性验证；已有可用原生性能对照时仍使用原生对照。

当前公共 concatenate / pow 用例范围以 NVIDIA 可验证参考为准：2026-10-15 已删除 70 条非浮点 concatenate 和 15 条 INT32 pow。历史排除记录不代表当前必测目录，详见下文“公共用例调整及跨平台影响”。

## 数据来源

正式数据统一保存在 [performance-baseline.json](performance-baseline.json)，本次 FlagDNN 实测耗时由测试程序提供，不写回基线。

| 字段 | 用途 |
|---|---|
| `platforms.<平台>.peaks`、`metric_units` | 设备实测峰值及单位；未测项为 `null` |
| `platforms.<平台>.scaling.case_policies.<op>` | 该平台的换算指标 `scale_metric`、适用范围和依据 |
| `reference.cases.<case_name>` | NVIDIA 参考耗时 `time_us`、张量与算子参数、输入及计时配置、验证等级 |
| `reference.measurement.timing_profiles` | 预热次数、采样数、每批执行次数及 Graph 预热配置 |
| `min_relative_efficiency` | 最低相对效率，当前为 `0.8` |
| `reference.coverage`、`reference.unavailable_cases` | 参考覆盖范围及不可用用例；不可用记录不能参与换算 |
| `measurement_summaries` | 测量配置、来源和原始证据哈希 |

各平台必须登记自己的设备身份、测量日期、峰值和算子策略。仅有峰值不会启用等效性能，不得借用其他平台策略，也不得改写共享 NVIDIA 参考记录中的 `scale_metric`。

## 换算公式

```text
equivalent_time_us = T_nv × P_nv / P_target × traffic_ratio
limit_us = equivalent_time_us / α
relative_efficiency = equivalent_time_us / measured_us
passed = measured_us <= limit_us
```

`T_nv` 为匹配用例的 NVIDIA 耗时；`P_nv`、`P_target` 为同一指标、同一单位的两平台峰值；`measured_us` 为本次目标平台耗时；`α` 取 `min_relative_efficiency`。耗时、峰值和流量比必须为有限正数，且 `0 < α <= 1`。

`traffic_ratio` 默认 `1`，仅带宽模型允许按目标/NVIDIA 的逻辑读写位数修正，不计 padding。例如两个 FP32 输入、一个比较输出，NVIDIA 输出 1 bit、目标输出 1 byte 时，比例为 `72/65`。参考存储位宽见 tensor 的 `reference_storage_bits`；计算模型始终取 `1`。

## 峰值选择

- 复制、布局和逐元素算子：仅在带宽模型适用时选择 `hbm_copy`、`hbm_write` 或 `hbm_triad`。
- 向量计算：按实际计算精度选择指标，不能仅根据输入存储 dtype 判断。特殊函数使用 FP32 FMA 峰值只是近似代理。
- 矩阵乘、卷积和注意力：选择与实际计算模式匹配的 `tensor_*` 指标；TF32、FP32 和不同 FP8 组合不能混用。

峰值按各配置的全样本耗时中位数计算吞吐，再取最高配置值。FMA/MAD 每标量计 2 次运算，稠密 GEMM 计 `2×M×N×K`，带宽按逻辑读写字节计数，单位为十进制 TB/s。缓存、启动开销和特殊指令吞吐可能限制实际性能，策略应记录代理模型的依据与局限，不能根据本次测量结果调整指标或门槛。

## 使用条件

1. 匹配设备型号和架构、用例名称、dtype、shape、strides、算子属性及输入生成方式。同名但参数不同的用例不能共用参考耗时，也不能借用其他 dtype 或算子的时间。
2. 目标平台必须匹配参考 `timing_profile`，包括预热、Graph 回放、采样和统计方式。当前参考取 nearest-rank P50；编译、建图、数据传输和校验均在计时外。Graph 与直接提交批次的计时结果不能混用。
3. 先验证结果，再计时；计时后以及重置输出后重新回放，检查输出、输入完整性与 padding。FP8 纯复制按原始字节比较。
4. 保留 NVIDIA 原始验证等级：`cpu_full` 为全量 CPU 校验，`cpu_sampled` 为采样校验，`bit_exact` 为按位校验，`sanity` 仅检查执行、输出写入及边界，不代表完整数值正确性。
5. 缺基线文件或平台策略、无效峰值、身份或参数不匹配、校验失败和执行错误均为失败。已明确不在参考覆盖范围的用例可按平台目录规则排除或跳过，不能伪造参考耗时。

## 结果与平台支持

输出仅保存本次真实 FlagDNN 样本，`equivalent_baseline` 单独记录参考来源、峰值、等效耗时、上限、相对效率及性能评价，不生成虚构的原生对照样本。

默认批测中，执行及正确性通过与性能达标分别记录：低于基线仍保留 `equivalent_baseline.passed=false`，不会仅因性能不足将测试判为失败。

- **Hygon**：按本平台 `case_policies` 启用等效路径。显式传入 `--min-speedup` 才启用性能失败门槛，等效用例要求 `max(0.8, --min-speedup)`；原生用例按原生加速比判定。
- **Ascend**：当前仅对 identity 的 FP8 E4M3/E5M2 用例启用 `hbm_copy` 等效策略，不支持 `--min-speedup`。其他 FP8 布局或拼接不能借用 identity 的参考耗时。
- **MThreads**：按本平台策略为 muDNN 不支持且参考完整的用例启用等效路径；`--min-speedup` 门槛规则同 Hygon。90 条 ELU/softplus backward native 参考已补齐，平台运行及报告接入仍需单独完成，见下文。

峰值采集、构建和复测命令见[性能基准工具说明](../tools/performance_baseline/README.md)；具体覆盖范围、峰值数值和测量来源以正式 JSON 为准。

## 公共用例调整及跨平台影响（2026-10-15）

经开发决策，公共 concatenate / pow 测试以已验证的 NVIDIA cuDNN 参考范围为准，从共享工厂删除 85 条无有效参考的用例。仅在 NVIDIA 端过滤会造成平台覆盖不一致；共享工厂统一集合后，其他平台只需同步静态目录和测试契约，无需修改 kernel 或运行逻辑。

| 公共目录 | 调整前 | 调整后 |
|---|---:|---:|
| concatenate 功能及 native 性能 | 112 | 42 |
| pow 功能 | 57 | 42 |
| pow 常规浮点性能 | 45 | 45 |

concatenate 删除 INT32、BOOL、FP8 E4M3/E5M2/E8M0 各 14 条；pow 删除 14 条普通 INT32 和 1 条 padded 广播用例。保留 FP32、FP16、BF16。gen_index 和其他逐点算子的整数用例不受影响；算子 API、Triton kernel 和 CPU reference 的已有能力不变。

`tests/common/index.cpp` 调整 concatenate 类型集合；`tests/common/pointwise.cpp` 排除 INT32 POW 并保留原 UID 分配间隔。NVIDIA index runner 删除重复过滤。修改前后导出共享索引、逐点工厂和 pow 性能目录，2370 条变为 2285 条；差集恰好为上述 85 条，其余名称、UID、dtype、shape、strides、binding offset、属性、输入域、容差和 autotune 标记逐项相同。

跨平台同步范围：

- Thead：`validation/capability.json` 删除 85 个功能名称，`validation/benchmark/comparable_cases.json` 删除 70 个 concatenate 性能名称，保留条目的能力判定不变。覆盖契约不再要求退役用例的 70 条跳过；历史跳过检查改用独立合成目录。
- MThreads：仅更新主机 `equivalent_baseline_contract.cpp`。原来“105 条都缺参考”的断言已过时：90 条激活反向参考已入库，15 条 INT32 POW 已移除。现在验证 84 条普通记录能按原元数据匹配，6 条 Real/Scaled 属性记录仍被现有输入域规则拒绝。报告适配器尚未接受新增 native 名称，不能把主机契约通过视为平台接入完成。
- 公共批测契约：有效参考数由 3264 更新为 3354；按 Hygon 策略归类的参考数量增加 90，实际运行目录仍为 1941，策略和门槛不变。

85 条历史失败记录保留在 `reference.unavailable_cases` 中，`status=excluded`、`catalog_status=removed_from_shared_cases`，实测失败类型存于 `measurement_status`。它们不是当前待补测任务；3354 条有效参考、平台峰值及 0.8 门槛不受目录调整影响。[测量摘要](../tools/performance_baseline/measurements/nvidia-supplement-20261015.json)和[工具说明](../tools/performance_baseline/README.md)保留原始补测事实与复现入口。

NVIDIA 相关构建和仓库内 SDK 安装通过；公共批测契约、Thead/MThreads 报告契约、MThreads 主机基线契约通过。目录调整后定点 GPU 功能 15 条、性能 6 条通过，6 条性能的 0.8 门槛检查通过；90 条激活反向采集回归通过。未运行其他平台设备测试或全仓全量测试；主机验证不能替代目标平台设备验证。
