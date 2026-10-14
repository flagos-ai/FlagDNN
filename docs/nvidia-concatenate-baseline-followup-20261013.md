# NVIDIA concatenate 缺失基线补测清单（2026-10-13）

本文定义需要在 NVIDIA 真机补测的 **70 条 concatenate 参考耗时**、完整张量配置、校验与交付要求。它是待执行清单，不是测量结果；目前没有这 70 条的有效 `time_us`，也没有“全部性能测试通过”的结论。

[performance-baseline.json](performance-baseline.json) 中 concatenate 的 112 个同名目录项可精确分为：42 条 FP32/FP16/BF16 有效参考，以及 `reference.unavailable_cases` 中 70 条 `excluded`。后者为 INT32、BOOL、FP8 E4M3、FP8 E5M2、FP8 E8M0 各 14 条，排除原因均为：

```text
Excluded by NVIDIA native index runner: concatenate supports only fp32/fp16/bfloat16 reference cases
```

已有 42 条的验证等级是 `sanity`；不能把它们描述为经过全量 CPU 校验。当前 Thead 的功能通过、CPU 索引验证及设备峰值测量均不能替代缺失的 NVIDIA 原生耗时。等效换算方法见[跨平台基准说明](equivalent-performance-baseline.md)。

## 目录来源与核对方式

唯一用例定义来源是 [tests/common/index.cpp](../tests/common/index.cpp) 的 `make_index_cases("concatenate")`，结构定义见 [index.hpp](../tests/common/index.hpp)。本清单通过 CPU 程序直接调用当前构建的共享工厂提取 112 个完整 `TestTensor`，再与实际 Thead benchmark 的 `--dump-cases` 逐名核对；结果满足：

```text
实际目录 112 = reference.cases 42 + reference.unavailable_cases 70
两个参考集合无交集，无重复名，无遗漏名
每个缺失 dtype 恰好 14 条
```

以下命令只读取已构建二进制的目录，不启动 GPU 测试：

```bash
build/thead/backends/thead/validation/flagdnn_benchmark_thead_concatenate --dump-cases
```

没有该构建目录时，可依据共享工厂生成目录，或在已配置 NVIDIA 环境中使用下文“现有采集工具可做与缺少的工作”中的原生 `--catalog`；不能把目录导出当作 NVIDIA 测量成功。

case 名里的维度是工厂的**起始形状**，并非总输出形状。例如 `concatenate_int32_2_1025_axis-1` 有 `[2,1025]` 和 `[2,1026]` 两个输入，实际输出为 `[2,2051]`。不得仅解析名字并把输入/输出全部设成同一形状。

## 精确用例清单：5 种 dtype 各 14 条

每个代码块的第 1 至 14 行依次对应下方 G00 至 G13。除 dtype 外，同一组的输入次序、UID、shape、strides、axis 和绑定偏移完全相同。

### int32

```text
concatenate_int32_1_axis0
concatenate_int32_17_axis-1
concatenate_int32_33_axis0
concatenate_int32_2_3_axis-1
concatenate_int32_3_7_axis0
concatenate_int32_2_3_5_axis-1
concatenate_int32_1_3_17_axis0
concatenate_int32_2_4_33_axis-2
concatenate_int32_2_3_4_5_axis0
concatenate_int32_1_7_3_9_axis-3
concatenate_int32_2_2_2_2_3_axis0
concatenate_int32_1_2_1_3_1_2_1_5_axis-5
concatenate_int32_3_257_axis0
concatenate_int32_2_1025_axis-1
```

### bool

```text
concatenate_bool_1_axis0
concatenate_bool_17_axis-1
concatenate_bool_33_axis0
concatenate_bool_2_3_axis-1
concatenate_bool_3_7_axis0
concatenate_bool_2_3_5_axis-1
concatenate_bool_1_3_17_axis0
concatenate_bool_2_4_33_axis-2
concatenate_bool_2_3_4_5_axis0
concatenate_bool_1_7_3_9_axis-3
concatenate_bool_2_2_2_2_3_axis0
concatenate_bool_1_2_1_3_1_2_1_5_axis-5
concatenate_bool_3_257_axis0
concatenate_bool_2_1025_axis-1
```

### fp8_e4m3

```text
concatenate_fp8_e4m3_1_axis0
concatenate_fp8_e4m3_17_axis-1
concatenate_fp8_e4m3_33_axis0
concatenate_fp8_e4m3_2_3_axis-1
concatenate_fp8_e4m3_3_7_axis0
concatenate_fp8_e4m3_2_3_5_axis-1
concatenate_fp8_e4m3_1_3_17_axis0
concatenate_fp8_e4m3_2_4_33_axis-2
concatenate_fp8_e4m3_2_3_4_5_axis0
concatenate_fp8_e4m3_1_7_3_9_axis-3
concatenate_fp8_e4m3_2_2_2_2_3_axis0
concatenate_fp8_e4m3_1_2_1_3_1_2_1_5_axis-5
concatenate_fp8_e4m3_3_257_axis0
concatenate_fp8_e4m3_2_1025_axis-1
```

### fp8_e5m2

```text
concatenate_fp8_e5m2_1_axis0
concatenate_fp8_e5m2_17_axis-1
concatenate_fp8_e5m2_33_axis0
concatenate_fp8_e5m2_2_3_axis-1
concatenate_fp8_e5m2_3_7_axis0
concatenate_fp8_e5m2_2_3_5_axis-1
concatenate_fp8_e5m2_1_3_17_axis0
concatenate_fp8_e5m2_2_4_33_axis-2
concatenate_fp8_e5m2_2_3_4_5_axis0
concatenate_fp8_e5m2_1_7_3_9_axis-3
concatenate_fp8_e5m2_2_2_2_2_3_axis0
concatenate_fp8_e5m2_1_2_1_3_1_2_1_5_axis-5
concatenate_fp8_e5m2_3_257_axis0
concatenate_fp8_e5m2_2_1025_axis-1
```

### fp8_e8m0

```text
concatenate_fp8_e8m0_1_axis0
concatenate_fp8_e8m0_17_axis-1
concatenate_fp8_e8m0_33_axis0
concatenate_fp8_e8m0_2_3_axis-1
concatenate_fp8_e8m0_3_7_axis0
concatenate_fp8_e8m0_2_3_5_axis-1
concatenate_fp8_e8m0_1_3_17_axis0
concatenate_fp8_e8m0_2_4_33_axis-2
concatenate_fp8_e8m0_2_3_4_5_axis0
concatenate_fp8_e8m0_1_7_3_9_axis-3
concatenate_fp8_e8m0_2_2_2_2_3_axis0
concatenate_fp8_e8m0_1_2_1_3_1_2_1_5_axis-5
concatenate_fp8_e8m0_3_257_axis0
concatenate_fp8_e8m0_2_1025_axis-1
```

## 14 组共享张量几何

`I0`、`I1`、`I2`、`I3` 按拼接次序对应输入 UID 100、101、102、103；`O` 的 UID 为 200。所有张量的 `binding_byte_offset=0`。表中的 stride 单位是**逻辑元素**，不是字节；INT32 为 32 位，三种 FP8 为 8 位，Thead BOOL 当前为 8 位。NVIDIA BOOL 的物理编码必须实测确认并单独记录，不能由这里的目标位宽推断。

`axis` 保留目录里的带符号值，括号内为归一化的非负轴。参考 JSON 的 `attributes.axis` 必须与原带符号值相同，即使原生执行时先归一化。

| 组 | 名称后缀 | axis（归一化） | 输入数 | 输出逻辑元素数 |
|---|---|---:|---:|---:|
| G00 | `1_axis0` | 0（0） | 1 | 1 |
| G01 | `17_axis-1` | -1（0） | 2 | 35 |
| G02 | `33_axis0` | 0（0） | 3 | 102 |
| G03 | `2_3_axis-1` | -1（1） | 4 | 36 |
| G04 | `3_7_axis0` | 0（0） | 1 | 21 |
| G05 | `2_3_5_axis-1` | -1（2） | 2 | 66 |
| G06 | `1_3_17_axis0` | 0（0） | 3 | 306 |
| G07 | `2_4_33_axis-2` | -2（1） | 4 | 1452 |
| G08 | `2_3_4_5_axis0` | 0（0） | 1 | 120 |
| G09 | `1_7_3_9_axis-3` | -3（1） | 2 | 405 |
| G10 | `2_2_2_2_3_axis0` | 0（0） | 3 | 216 |
| G11 | `1_2_1_3_1_2_1_5_axis-5` | -5（3） | 4 | 360 |
| G12 | `3_257_axis0` | 0（0） | 1 | 771 |
| G13 | `2_1025_axis-1` | -1（1） | 2 | 4102 |

| 组 | 张量 | shape | strides |
|---|---|---|---|
| G00 | I0 | `[1]` | `[1]` |
| G00 | O | `[1]` | `[1]` |
| G01 | I0 | `[17]` | `[2]` |
| G01 | I1 | `[18]` | `[1]` |
| G01 | O | `[35]` | `[2]` |
| G02 | I0 | `[33]` | `[1]` |
| G02 | I1 | `[34]` | `[2]` |
| G02 | I2 | `[35]` | `[1]` |
| G02 | O | `[102]` | `[2]` |
| G03 | I0 | `[2,3]` | `[9,2]` |
| G03 | I1 | `[2,4]` | `[4,1]` |
| G03 | I2 | `[2,5]` | `[13,2]` |
| G03 | I3 | `[2,6]` | `[6,1]` |
| G03 | O | `[2,18]` | `[18,1]` |
| G04 | I0 | `[3,7]` | `[7,1]` |
| G04 | O | `[3,7]` | `[17,2]` |
| G05 | I0 | `[2,3,5]` | `[42,13,2]` |
| G05 | I1 | `[2,3,6]` | `[18,6,1]` |
| G05 | O | `[2,3,11]` | `[78,25,2]` |
| G06 | I0 | `[1,3,17]` | `[51,17,1]` |
| G06 | I1 | `[2,3,17]` | `[114,37,2]` |
| G06 | I2 | `[3,3,17]` | `[51,17,1]` |
| G06 | O | `[6,3,17]` | `[51,17,1]` |
| G07 | I0 | `[2,4,33]` | `[279,69,2]` |
| G07 | I1 | `[2,5,33]` | `[165,33,1]` |
| G07 | I2 | `[2,6,33]` | `[417,69,2]` |
| G07 | I3 | `[2,7,33]` | `[231,33,1]` |
| G07 | O | `[2,22,33]` | `[1521,69,2]` |
| G08 | I0 | `[2,3,4,5]` | `[60,20,5,1]` |
| G08 | O | `[2,3,4,5]` | `[168,55,13,2]` |
| G09 | I0 | `[1,7,3,9]` | `[465,66,21,2]` |
| G09 | I1 | `[1,8,3,9]` | `[216,27,9,1]` |
| G09 | O | `[1,15,3,9]` | `[405,27,9,1]` |
| G10 | I0 | `[2,2,2,2,3]` | `[24,12,6,3,1]` |
| G10 | I1 | `[3,2,2,2,3]` | `[93,45,21,9,2]` |
| G10 | I2 | `[4,2,2,2,3]` | `[24,12,6,3,1]` |
| G10 | O | `[9,2,2,2,3]` | `[93,45,21,9,2]` |
| G11 | I0 | `[1,2,1,3,1,2,1,5]` | `[243,120,117,38,35,16,13,2]` |
| G11 | I1 | `[1,2,1,4,1,2,1,5]` | `[80,40,40,10,10,5,5,1]` |
| G11 | I2 | `[1,2,1,5,1,2,1,5]` | `[395,196,193,38,35,16,13,2]` |
| G11 | I3 | `[1,2,1,6,1,2,1,5]` | `[120,60,60,10,10,5,5,1]` |
| G11 | O | `[1,2,1,18,1,2,1,5]` | `[1383,690,687,38,35,16,13,2]` |
| G12 | I0 | `[3,257]` | `[257,1]` |
| G12 | O | `[3,257]` | `[257,1]` |
| G13 | I0 | `[2,1025]` | `[2053,2]` |
| G13 | I1 | `[2,1026]` | `[1026,1]` |
| G13 | O | `[2,2051]` | `[4105,2]` |

这些数值直接来自工厂，而不是按 dense 布局重新计算。其生成规则可用于交叉检查：以 G00 为序号 0；第 `i` 组有 `1 + i % 4` 个输入，第 `j` 个输入在归一化 axis 上的维度为起始维度加 `j`；输出在该轴上的维度是所有输入之和。输入在 `(i+j)%2 != 0` 时带 padding，输出在 `i%3 != 0` 时带 padding。dense stride 从末轴以 `span=1` 累乘；padded stride 从 `span=2` 开始，每处理一轴后执行 `span = span * shape[axis] + 3`。不要把单输入组优化成不写输出：I0 和 O 的存储独立，且部分组的布局不同。

对正 stride，物理 storage 元素数应从 `1 + sum((shape[d]-1)*strides[d])` 计算；逻辑元素数为 `product(shape)`。两者都需保存，padding 不应误计为逻辑吞吐。所有输入和输出使用同一 dtype。

## 真机、原生路线与峰值要求

优先使用现有参考对应的 **NVIDIA H100 80GB HBM3、700 W** 测量环境；现有 provider 为 `cuDNN 9.24.0`，NVIDIA `hbm_copy` 峰值为 **2.858 TB/s**，来源日期为 2026-09-25。需要记录本次实际 GPU 型号、UUID、compute capability、显存/L2、驱动、CUDA、cuDNN/Frontend 版本、功耗限制、时钟/运行状态、测量 UTC 时间、源码 revision 及脏工作树文件哈希。没有真机记录，不能确认新时间与历史峰值属于可比较的设备环境。

先在 NVIDIA 真机逐 dtype 证明实际原生路线能够保持本清单的张量语义和 CUDA Graph 执行。现有 [build_cudnn_index](../backends/nvidia/validation/functional/cudnn_extended.cpp) 先尝试 cuDNN graph concatenate，异常时尝试 [CudnnConcatenateTransform](../backends/nvidia/validation/functional/cudnn_layout.cpp)，后者按输入调用 `cudnnTransformTensor(alpha=1,beta=0)` 写入输出子视图。目录排除不等于已证明这些类型永远不受支持；同样，删除排除条件也不证明它们已经受支持。必须记录实际选中的路线、描述符 dtype、输入/输出物理表示、调用数，以及 Graph 捕获与重放结果。

对于 INT32 和 FP8，数值转换后再拼接不能证明原始位模式的复制语义。若提议改用字节重解释、其他原生库或新的多原语组合，应作为独立参考实现先审查、校验并记录来源和额外开销；不能直接伪装为已认证的原 dtype cuDNN concatenate。用 FlagDNN 自己的 NVIDIA 内核充当原生对照、借用其他 dtype/identity 的时间、根据 Thead 时间倒推 NVIDIA 时间，均不能填补本清单。若原生路线仍不支持，保留明确失败/不可用原因，不产生有效耗时。

测量期间 GPU 独占、串行运行。`hbm_copy` 使用逻辑读加写流量和十进制 TB/s；采集器为 [benchmark_nvidia_vector_memory.cu](../tools/performance_baseline/benchmark_nvidia_vector_memory.cu)。如需复核峰值，按其原有配置扫描超过 L2 的缓冲区，保留所有样本和校验结果，取配置内中位吞吐，再选配置间最高中位值；不能取最快单样本，也不能用产品宣传带宽代替。

如果换用另一种 NVIDIA GPU、功耗配置或不可与历史峰值对应的设备环境，必须为该环境提供同口径 `hbm_copy` 实测并单独保留来源。当前共享 JSON 只有一个 `platforms.nvidia.peaks.hbm_copy`，不能把其他设备时间混入后继续默默乘 H100 的 2.858，也不能覆盖该全局峰值而改变既有参考的换算。此时需先设计并验证按测量来源选择峰值的 schema/loader，或使用能正确绑定设备和峰值的独立基线文档，再入库。此次补测的算子代理只需要 `hbm_copy`，不需要虚构 FP8 Tensor Core 峰值。

## 数据、全量验证与计时协议

### 输入与独立 CPU 参考

CPU 数学/索引依据为 [reference/cpu/index.hpp](../reference/cpu/index.hpp) 和 [reference/cpu/index.cpp](../reference/cpu/index.cpp)。迁移后的 `evaluate_index_bytes` 接收各输入的连续逻辑字节串，生成全部逻辑输出；concatenate 保留 INT32/FP8 表示位，`index_source` 提供单点逻辑映射。采集器负责按各张量的 stride、绑定偏移收集/散布数据；若 NVIDIA 使用 bitpacked BOOL，需先解包为每逻辑值一字节的参考输入，再按真实编码比较 GPU 结果，不能把 packed buffer 直接传给目前按 1 字节 BOOL 定义的 `evaluate_index_bytes`。CPU 参考不得调用被测 NVIDIA 实现生成期望结果，也不得通过 FlagDNN 输出生成期望结果。

为了使未来 Thead 性能输入与 NVIDIA 参考一致，建议沿用当前目标端非浮点 concatenate 的原始输入规则并在补测时固定版本：第 `k` 个输入（从 0 开始）的连续逻辑字节串，INT32/FP8 第 `b` 字节为 `(b*37+k*83)%256`；BOOL 第 `i` 个逻辑值为 `(i+k)%2`。按该输入张量的 stride 散布后再上传。INT32 以原始 4 字节保存，不经过 float；需记录端序。NVIDIA 如果要求 bitpacked BOOL，应先按同一逻辑值编码并记录位序，不能把 byte BOOL 的物理 buffer 直接当成 packed buffer。[Thead 输入与布局适配](../backends/thead/validation/cpu_validation.cpp)可供输入兼容性核查；期望值数学与拼接语义由上述共享 `reference/cpu/` 原语提供。

额外的非计时校验数据应包含 INT32 的 0、±1、±16777217、INT32_MIN/MAX，以及 FP8 的全部 256 种字节编码；较小张量可分多个 fixture 覆盖。FP8 的 NaN、无穷及其他特殊编码在复制语义下按原始字节检查，不做浮点近似比较，也不能用“输出必须 finite”排除合法位模式。BOOL 使用合法 0/1 值并检查真实 packed/byte 编码和未使用位。

每条用例必须完成以下三次**全量**验证，并保存检查元素数/字节数、失败位置及结果：

1. 计时前执行，逐逻辑元素比较 CPU 期望；INT32/FP8 精确比字节，BOOL 精确比逻辑位并验证其物理存储约定。
2. 最后一批计时后重新读回全部输出，再做相同检查，同时确认输入、padding、边界 guard 未被修改。
3. 重置输出为逐位置保证与期望不同的合法哨兵，回放同一个已捕获 Graph，再全量验证写入完整性。INT32/FP8 可按预期字节取反；BOOL 应用相反的合法逻辑值并正确编码。固定哨兵如果恰好等于期望值，不能检测空执行或部分漏写。

输入初始化、CPU 索引计算、拷贝、校验、输出重置均在计时区间之外。未通过任一阶段的条目只能作为失败记录保留，不能留下可用于等效性能的 `time_us`。全量 CPU 逐字节验证的正式参考等级填写 `validation="cpu_full"`，精确比较方法和实际计数另存在原始记录中。

### 与现存 `native` profile 一致的计时

| 项 | 必须值/要求 |
|---|---|
| 预热 | 10 次完整 concatenate 执行并同步 |
| Graph | 捕获 50 次完整执行，实例化后先预热回放 1 次并同步 |
| 正式样本 | 20 个 GPU event 样本，每个包围一次 50 次执行的 Graph 回放 |
| 单次样本 | `samples_us[i] = sample_batch_us[i] / 50` |
| 统计 | `nearest_rank_p50`；20 样本排序后第 10 个为 P50，第 18 个为 P90（按 1 开始计数） |
| 入库时间 | `time_us = median_us = nearest_rank_p50(samples_us)`，单位 µs，必须有限且大于 0 |
| 不计时 | 图构建/编译、workspace 分配、数据生成与传输、CPU 校验 |
| 计时内容 | 一次逻辑 concatenate 的全部原生调用；如有多个子操作，不能只计其中一段 |

原始记录保留未排序的 20 个逐样本值、20 个批次值、P50/P90、workspace 字节数及独立 build 时间。不得改成最小值、host wall time、无 Graph 的直接提交或其他 warmup/batch 配置后仍标为 `native`。当前 concat 形状较小且反复使用相同 buffer，缓存和提交开销可能主导；`hbm_copy` 等效模型仅为约定代理，不是已经证明的硬件瓶颈。

## 现有采集工具可做与缺少的工作

[benchmark_nvidia_extended.py](../tools/performance_baseline/benchmark_nvidia_extended.py) 与 [C++ 采集器](../tools/performance_baseline/benchmark_nvidia_extended.cpp) 已有 `--op concatenate`、`--case NAME`、`--shared-context` 等选项；但 C++ catalog 对本文 70 条明确设置 `blocked`，Python 会直接保存 blocked 结果而不运行测量。**现有这些参数无法完成本文补测。** `--extra-variants` 是 attention 变体探测选项，也不能启用这些 concatenate dtype。

在已经配置 `build/nvidia`、已构建其 native 验证库、设置 `CUDNN_ROOT` 和 `CUDNN_FRONTEND_INCLUDE_DIR`（可选 `CUDA_ROOT`）的 NVIDIA 开发环境，可以用以下现有命令核对目录；这里不声称它会采集这 70 条：

```bash
python3 tools/performance_baseline/benchmark_nvidia_extended.py \
  --build-only --binary /tmp/benchmark_nvidia_extended \
  --output /tmp/nvidia-concatenate-followup/catalog-inspection.jsonl
/tmp/benchmark_nvidia_extended --catalog
```

`--build-only` 会编译并导出 `.catalog.jsonl`，不进行实际测量；C++ `--catalog` 不初始化 CUDA。该 Python 构建过程链接现有 `build/nvidia` 库，不自动替你构建所依赖的 CMake 目标。完整工具环境说明见[工具 README](../tools/performance_baseline/README.md)。

补测前仍需实现并审查：逐 dtype 原生能力认证；可保留 INT32/FP8 所有位模式的输入与 I/O；独立全量 CPU 字节参考；三阶段校验；实际 provider/路线与 BOOL 物理位宽记录；恰好 70 条的精确选择和失败保留。当前 extended collector 的通用输入为 `vector<float>`，concat 的 `indexed_inputs` 会先生成浮点数；默认输出校验只做写入、finite 和 padding sanity。它也没有为 concat 注册 `exact` 回调或计时后/重置后全量验证。因此删除 `blocked` 并使用旧流程不满足本文要求。

已有 `run_benchmark_nvidia_missing.py` 的 371 条补测目录不包含本文新增的 70 条；不能拿该工具原有成功记录宣称此缺口已解决。本文不提供一个尚不存在的“70 条一键采集命令”；新采集器完成后，交付物必须记录实际使用的完整构建与运行命令。

## 交付记录与入库 schema

建议以独立测量归档交付，不直接覆盖历史记录。至少包含：恰好 70 条目标 manifest、完整原始 catalog、逐用例 JSONL、stdout/stderr、三阶段校验结果、设备/软件环境、实际命令、采集器源码快照及 SHA-256、原始文件 SHA-256，以及峰值来源或本次峰值原始测量。失败项也保留，不用后一次成功静默擦掉。

### 原始逐用例记录的必需内容

| 字段组 | 内容 |
|---|---|
| 身份 | `case_name`、`op="concatenate"`、真实 `provider`、catalog、测量来源 ID、设备/软件环境引用 |
| 张量 | 按输入次序后接输出；`uid`、role、真实 dtype、dimensions、strides、binding_byte_offset、logical_elements、storage_elements、实际存储位宽和 BOOL encoding/位序 |
| 属性 | `attributes.axis` 为目录带符号值；另可记录归一化执行轴 |
| 输入 | 原始字节/逻辑值生成公式、dtype 转换规则、端序、padding/guard 规则、生成器版本/哈希 |
| 路线 | 实际 cuDNN graph/transform 等实现、描述符 dtype、调用数、所有必要布局或编码处理及是否计时 |
| 验证 | `status`、`correctness`、全量校验方法；计时前/后/重置回放各自 PASS、实际校验量及错误；失败时 stage/reason |
| 计时 | timing_method、10/20/50/1 四个计数、nearest_rank、20 个 samples_us、20 个 sample_batch_us、median_us、p90_us、workspace_bytes、build_us |

现有 extended 原始字段用 `dimensions`，规范基线用 `shape`；原始 BOOL dtype 写 `boolean`，当前规范 loader 要求 `bool`。现有 collector 在配置 `exact` 回调时会输出 `validation_scope="cpu_exact"`，这只是原始字段；正式 `reference.cases` 应写已有 loader 认可的 `validation="cpu_full"`，并以真实全量检查证据支撑归一化，不能仅改标签。不要为正式基线引入未经 loader 支持的新 validation 枚举。

### 规范参考条目示意

下面是 G01 INT32 的**结构模板**。`time_us: null` 表示尚未测量，不能作为有效基线导入；只有真实成功记录才能用其 P50 替换该值。

```json
{
  "concatenate_int32_17_axis-1": {
    "op": "concatenate",
    "tensors": [
      {"uid": 100, "role": "input", "dtype": "int32", "shape": [17], "strides": [2], "binding_byte_offset": 0, "reference_storage_bits": 32},
      {"uid": 101, "role": "input", "dtype": "int32", "shape": [18], "strides": [1], "binding_byte_offset": 0, "reference_storage_bits": 32},
      {"uid": 200, "role": "output", "dtype": "int32", "shape": [35], "strides": [2], "binding_byte_offset": 0, "reference_storage_bits": 32}
    ],
    "time_us": null,
    "scale_metric": null,
    "timing_profile": "native",
    "validation": "cpu_full",
    "attributes": {"axis": -1}
  }
}
```

正式 JSON 的 schema version 当前为 4。共享条目的 `scale_metric` 保持 `null`；目标平台的策略在 `platforms.thead.scaling.case_policies.concatenate` 显式选择 `hbm_copy`。补测的日期、provider、manifest、文件指纹和关联峰值来源需追加到 `reference.measurement` 的可追溯测量信息；不要改写历史 42 条、其他算子或其他平台峰值。成功条目入 `reference.cases` 后，需消除同名 `unavailable_cases` 的当前状态冲突，同时把原来的 excluded 记录和原因保留在测量历史中；只有成功且全部核验通过的名字才能迁移。

## 补齐后仍需的接入与验收

取得 70 个正数耗时只是前提，当前代码并不自动启用这些 dtype。须在单独实现变更中完成并验证以下事项；本文没有修改这些生产代码或正式 JSON。

1. **显式启用范围。** [Thead native_index.cpp](../backends/thead/validation/benchmark/native_index.cpp) 目前对非 FP32/FP16/BF16 concatenate 直接报 `nvidia_reference_unavailable`。新数据核验后才能移除相应已补齐用例的静态跳过，并扩大 Thead concatenate 的策略 scope。缺失、失败或元数据不匹配的项继续明确失败/跳过，不能笼统放行所有 dtype。
2. **Thead loader 的 dtype 契约。** [equivalent_baseline.cpp](../backends/thead/validation/benchmark/equivalent_baseline.cpp) 当前只接受 concatenate FP32/FP16/BF16；INT32、BOOL 和三种 FP8 均需在取得有效数据后显式扩展。需要针对新增范围补齐，保留其他算子的限制，并验证精确 dtype 名、位宽、输入数、UID/顺序、shape/stride、绑定偏移和带符号 axis。
3. **BOOL 的存储与流量。** 当前 Thead 等效路径只启用参考与目标位宽相同、`traffic_ratio=1` 的用例，尚未启用 BOOL concatenate。补测后应按真实参考编码扩展 loader：目标 BOOL 按 8 位计；只有真机确认 bitpacked 的参考才能填写 `reference_storage_bits=1`，此时逻辑流量比为 8。若实际原生路线为 byte BOOL，应如实填 8，此时流量比为 1。两种来源均须单独验证，不能为了通过加载检查把 byte 时间标成 1 bit。数据记录、编码适配、reference stride 与模型的假设都需一致；必要编码转换的计时边界必须说明。
4. **结果 schema、adapter 与覆盖契约。** 当前 [benchmark/result.schema.json](../benchmark/result.schema.json) 的 Thead 分支只接受 concatenate FP32/FP16/BF16，要求 `reference_validation="sanity"` 和 `traffic_ratio=1`；通用 v4 case 模式也未允许 concatenate FP8 名称。[Thead run_tests_adapter.py](../backends/thead/validation/run_tests_adapter.py) 的 `equivalent_case_operator`、`validate_benchmark_record` 和覆盖统计使用相同的旧范围/验证等级/流量限制。因此补齐数据后，还必须有针对性地扩展 schema、adapter 和覆盖契约，允许新 dtype 的真实 `cpu_full` 证据及符合 BOOL 编码的流量比，保留历史 42 条的 `sanity`；不能仅修改 C++ loader。补充五种 dtype 正例，以及错 dtype、stride、axis、位宽、profile、非正时间、样本不足等反例。原始 `cpu_exact`/精确比较证据按上文归一化为 `cpu_full`，并确认原 NVIDIA 时间、参考验证等级、真实 FlagDNN 样本和流量修正完整进入汇总。现有结果对象不接受随意添加的存储位宽字段；逐张量物理位宽保存在参考/原始采集元数据，如要在结果中新增字段需同步扩展 schema。计时 profile 必须仍为 `native`。
5. **目录和 NVIDIA 结果闭环。** 新 manifest 与本文 70 名称集合完全相等，五种 dtype 各 14，所有张量与共享工厂逐项一致，无重复、缺失或额外条目；每条有 20 个真实样本、全部三阶段校验成功、统计值可复算、来源哈希可验证。最终 concatenate 应有 112 条有效参考；如果只补齐部分，继续报告剩余真实缺口。
6. **Thead 真机执行。** 先对五种 dtype 以及负轴、rank 8、padded stride 做定点功能和等效性能验证，再重建必要目标运行完整用户命令。不得把 NVIDIA 测量完成或 CPU 契约通过直接视为 Thead GPU 通过。

等效换算仍为：

```text
T_equiv = T_nv × P_nv(hbm_copy) / P_thead(hbm_copy) × traffic_ratio
relative_efficiency = T_equiv / T_thead
limit_us = T_equiv / 0.8
```

当前 Thead `hbm_copy=2.033910078 TB/s`；目标设备必须继续匹配记录的 `PPU-ZW810E` 和 `sm_80`。INT32/FP8 若参考和目标都保持相同位宽，`traffic_ratio=1`；BOOL 按真实参考编码处理。不能根据补测后的达标情况调整峰值、阈值、dtype 或参考时间。

完整回归命令为：

```bash
python3 tools/run_tests.py --dump-output --output-dir logs_result_20260920_flagdnn --gpus 0 --appoint
```

在目录和其他能力不变、70 条全部接入的前提下，`--appoint` 静态预期仍是 59 算子、67 个 benchmark 套件、3141 条性能用例；参考路径应由 **2956 原生 + 115 等效 + 70 缺参考**变为 **2956 原生 + 185 等效 + 0 缺参考**。这是补齐后的验收预期，不是本次已运行结果。还必须读取完整功能/性能日志，确认实际执行数、失败数、跳过数、目录覆盖和进程退出状态。

用户原命令未指定 `--min-speedup`，默认只报告性能，执行 `passed` 不证明所有相对效率都达到 0.8。若验收要求性能阈值参与失败判定，另行显式使用 `--min-speedup 0.8`，并检查全部等效记录的 `passed` 与门禁结果；阈值开关不影响数值正确性、异常、元数据匹配和缺参考处理。本文没有声称这 70 条已完成 NVIDIA 补测，也没有声称完整批测已经全部通过。
