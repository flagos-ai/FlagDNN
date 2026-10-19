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

## 2026-10-19 Iluvatar 补测与共享目录

Iluvatar 提出的 401 条缺口已逐条核查：92 条补测成功，309 条在 H100、
cuDNN 9.24 上没有可用的原生对照。另核查核心目录中过滤的 8 条用例，
保留并补测 2 条单元素 identity，删除 6 条不支持的布尔形状用例。
新增 94 条参考数据，批次为 `nvidia-iluvatar-20261019`；历史 3354 条有效
数据、各平台峰值、缩放策略和通过阈值均未改变。

不能将目录过滤视为 cuDNN 不支持。14 条 BF16 reshape/transpose/slice
以及 3 条单元素 identity 实际可由原生 `cudnnTransformTensor` 执行，
核心 NVIDIA 引用现在复用已有功能测试的原生回退路径。slice 回退的字节
偏移按真实 dtype 大小计算。所有补测均保持原始 dtype、shape、stride、
绑定偏移和输入生成规则，未使用类型重解释、CPU 耗时或其他算子代替 cuDNN。

按照统一 NVIDIA 基准的约定，以下已验证不支持的用例在共享生成器中删除：

- 69 条比较、选择及逻辑算子的特定布尔形状性能用例：cuDNN 要求每次访问
  至少 8 个布尔元素，原始连续维度不满足条件。
- 36 条 BF16 输入、BF16 输出的归约性能用例及对应功能用例：
  原生图没有可用计划，`cudnnReduceTensor` 返回 `CUDNN_STATUS_NOT_SUPPORTED`。
- 54 条 BF16/INT32 到 FP32 的混合归约用例：图和原生归约均不能执行。
  保留全部已验证的 FP32/FP16 用例，以及 rank-two 或末轴的 BF16/INT32 SUM，
  共 97 条混合输出用例；其中原有 20 条、新增 77 条。
- 144 条 INT32/FP8 布局用例：原始 dtype 的图没有可用计划；INT32 transform
  返回不支持，FP8 tensor descriptor 返回 `CUDNN_STATUS_BAD_PARAM`。
- 12 条 FP8 E8M0 identity：原生 cuDNN 图无法建立执行计划。

共享性能目录合计删除 315 条，功能目录删除 246 条。删除在原始枚举之后完成，
或先推进原始 UID 再决定保留，因此保留用例的 UID 和张量元数据不变。
NVIDIA/Hygon 的重复核心目录过滤已取消。THead 仅删除静态目录中的对应条目
和 overrides；Iluvatar 扩展已实测归约名单；Iluvatar/Hygon/MThreads 的
基准读取器仅增加这一明确批次及其 94 条记录的校验，不放宽元数据验证。
这些同步是共享目录/数据变化所必需，不修改其他平台 kernel 或运算实现。

Hygon 原有类型目录逐条对齐 NVIDIA，因此同步加入实测支持的 2 条 BF16
末轴 SUM 和 9 条 INT32 SUM，避免 NVIDIA 扩展后目录契约失败；当 hipDNN
不支持时，使用原有 CPU oracle 和新增实测等效参考。仅对应类型选择、
等效入口和报告 v4 的 INT32 SUM 名称校验扩展，原生 FP32/FP16 路径保持原策略。
Hygon 等效主机目录为 1953 条；NVIDIA/Hygon 类型探针的 16 组 437 条元数据
逐项一致。共享报告 schema 的扩展仍由各平台适配器约束，其他平台不会自动
启用 Hygon 策略。401 条请求均已闭环，原始请求 JSON 已从 docs 清理，
原文在 Git 历史和本地补测证据目录保留。

补测采用 10 次预热、20 个样本、每样本 50 次 CUDA Graph 执行，统计 nearest-rank
P50。计时前、计时后、清空输出后重放均对照 CPU oracle 全量验证，并检查输入、
padding 和边界。原始样本、失败原因、源码快照和二进制留在
`build/nvidia/iluvatar-followup-20261019/`；路径及 SHA-256 记录在正式基准的
`measurement_summaries.nvidia-iluvatar-20261019` 中，不添加压缩归档到源码。
各平台性能仍应使用自己的硬件运行验证；共享记录和主机契约通过不代表已完成
其他平台的设备性能测试。


本次回归：NVIDIA 完整配置构建和仓库内 SDK 安装通过；定点功能 26 条、
成对性能 22 条通过，22 条均满足 0.8 门槛，最低加速比约 1.44。
公共测试入口完整主机契约、公共性能目录契约、Iluvatar/Hygon/MThreads
基线读取契约、THead 当前目录覆盖契约通过；采集器 21 项主机测试通过。
历史 3354 条参考、平台策略和门槛逐项未变，证据 SHA-256 已复核。
未运行其他平台设备测试或全量 NVIDIA 算子测试。
