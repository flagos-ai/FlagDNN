# NVIDIA 性能基准补测清单

## 2026-10-12 NVIDIA 补测结果

附录一的 **371 条 NVIDIA 原生参考已全部完成采集，371 PASS、0 ERROR**。采集器在计时前、计时后及重置输出后回放时验证全部输出、输入不变和 padding/边界；本轮没有新增 `sanity` 记录。

| 项目 | 结果 |
|---|---:|
| 核心采集器：主逐元素、identity 主目录、batchnorm 训练 | 291 PASS |
| 辅助采集器：identity copy、logical boolean、混合输出 reduction | 80 PASS |
| 新增校验等级 | `cpu_full` 335 条；`bit_exact` 36 条 |
| 有效参考总量 | 2893 条历史记录 + 371 条新增 = **3264 条、82 个算子** |
| 历史不可用记录 | 491 条完整保留；本轮未新增失败记录 |

新增 batchnorm 的 24 条使用独立 `batchnorm_training` profile（预热 5 次、采样 10 批、每批 20 次）；其余 347 条使用原有 `native`（10/20/50），Graph 均预热回放一次。历史有效记录、不可用记录、所有平台对象及峰值保持原样；新增记录的共享 `scale_metric` 为 `null`。

原始记录独立验收通过 140080 项断言及 16 个反例检查；这是对记录证据的复核，不是另一次 GPU 测量。来源和证据见[测量摘要](../tools/performance_baseline/measurements/nvidia-missing-20261012.json)及[原始记录与日志归档](../tools/performance_baseline/measurements/nvidia-missing-20261012.tar.gz)。精确 371 条的构建、无 GPU 目录检查和串行复测命令见[工具说明](../tools/performance_baseline/README.md#2026-10-12-缺失目录补测)。下文保留补测前背景、hipDNN 探测和验收要求；附录一仍是准确复测清单，附录三、四没有计入本轮。

**本轮只补 NVIDIA 数据，不表示 Hygon 等效策略已接入或 371 个 Hygon 用例已通过。** Hygon 的 logical_and/or 辅助 `.boolean` 路径仍按旧逻辑报告缺参考，需要改为加载这批新记录；其他待补路径也须完成策略、元数据和 profile 适配后再正式回归。

## 补测前背景与 Hygon 状态

2026-10-12 修复前的 59 算子 Hygon 性能回归有 425 个跳过用例。独立验证 hipDNN 后，此前 Hygon 修复已实现 30 个 reduction 的原生对照，并将 24 个 identity FP8 用例接入已有 NVIDIA 等效参考；按同一目录核对，当时剩余 **371 个用例需要补齐参考路径**。这是目录与实现范围的核算，不是新一轮 59 算子全量通过结果；正式回归中止情况见下文。这些用例构成本轮已完成的 NVIDIA 补测范围。

补测前，原清单的 401 条均没有 NVIDIA 记录：其中 30 条已有原生 hipDNN 对照，不再需要为解除跳过而补测 NVIDIA，并非新增了 30 条 NVIDIA 数据。剩余 371 条也不能全部称为“已实测 hipDNN 不支持”；本轮验证范围、实际错误与尚未逐例验证的范围见下节。

范围不包含 `DEFAULT_OPERATORS` 中注释的算子。concatenate 的 70 个排除用例、identity E8M0 的 12 个目录外用例另列为扩展任务，不混入当前 371 或历史 425 的计数。本轮正式回归计划范围为 identity、add、div、reduction、batchnorm，未重新执行全部 59 算子。

## 本轮补测范围（按补测前缺口登记）

| 分组 | 算子和 dtype | 本轮新增 NVIDIA 记录数 |
|---|---|---:|
| 主逐元素性能目录 | 下列 17 个算子的 BF16，各 15 个 | 255 |
| identity 主目录 | BF16 | 12 |
| identity 辅助 copy 目录 | BF16、INT32、BOOL，各 12 个 | 36 |
| batchnorm 训练 | FP16、BF16，各 12 个 | 24 |
| reduction 混合类型输出 | FP16 → FP32 中间轴共 6 个，BF16 → FP32 共 14 个 | 20 |
| logical_and 和 logical_or 辅助 boolean 目录 | BOOL，各 12 个 | 24 |
| 合计 | 22 个算子 | **371** |

17 个主逐元素算子为：

`abs`、`add`、`add_square`、`elu`、`max`、`min`、`mul`、`relu`、`relu_backward`、`scale`、`sigmoid`、`sigmoid_backward`、`sub`、`swish`、`swish_backward`、`tanh`、`tanh_backward`。

补测前按完整的 `(op, case_name)` 与 [performance-baseline.json](performance-baseline.json) 核对，371 条均无对应有效记录，也不在当时的 `reference.unavailable_cases` 中；本轮按这些精确 ID 新增记录。不是把现有的其他 dtype、近似 shape 或其他目录记录当成缺测用例的替代品。

## Hygon 历史跳过原因与 hipDNN 支持的区别

“映射”指测试适配器如何把 FlagDNN 的操作、每个张量的 dtype/shape/strides、参数和输出，转换成语义一致的 hipDNN descriptor 与调用。API 有对应名称，不代表当前适配器已经实现了全部组合；适配器跳过，也不等于库实际返回不支持。

| 补测前待补范围 | 独立探测与适配状态 | 可以得出的结论 |
|---|---|---|
| 303 个逐元素用例 | 原生 dtype 白名单保持不变；BF16 的 17 个算子各测 1 个代表用例，另测 identity 多 dtype 与 FP32/FP16 控制组，详见下表 | 已测低精度组合出现 API 拒绝或内核编译失败，不能直接删除白名单；没有逐一穷举全部 shape |
| 24 个 batchnorm 用例 | 对 `1x128x8x8` 代表 shape 测试标准、Ex 和参数转换候选；仅原目录 FP32 及改变参数 dtype 的诊断变体通过 | 低精度原目录组合未找到五个输出都正确的路径；不是 rank 缺口，也不是 24 条都逐例测过 |
| 20 个 reduction 用例 | 原 50 条中 30 条 FP16 → FP32 已补齐原生对照；余下 6 条 FP16 中间轴存在错误结果，14 条 BF16 均实测执行失败 | 不能使用只保留原始高维 descriptor 的混合输出实现，也不能为中间轴套用首/末轴二维视图 |
| 24 个 logical 辅助 BOOL 用例 | [native_dtype.cpp](../backends/hygon/validation/benchmark/native_dtype.cpp) 明确报告缺少对应 NVIDIA 参考 | 主性能目录已测过同名算子，但不是这批 case，不能复用耗时 |

原 327 个逐元素跳过用例中的 24 个 identity FP8 E4M3/E5M2，现已使用已有 NVIDIA `bit_exact` 记录；当时逐元素部分仍缺 303 条参考，现已包含在本轮 NVIDIA 新增数据中。该路径在计时前后及重新填充输出后回放时，检查原始编码、输入未改变和 padding 未被破坏，不以 FLOAT 中转替代按位校验。

### 2026-10-12 Hygon 独立验证证据（历史）

环境：GPU `BW`，`arch=gfx936:sramecc+:xnack-`，HIP runtime/driver `60125085`，hipDNN `header=7000 runtime=8910`。以下结论仅针对该安装版本与已测配置，不代表所有 hipDNN 版本。

| 探测范围 | 实际结果 | 本轮处理 |
|---|---|---|
| 17 个 BF16 逐元素代表用例 | 7 个二元/组合图在 OpTensor 返回 `8 (EXECUTION_FAILED)`，日志为内核不识别 `bfloat16`；6 个前向 Activation 返回 `9 (NOT_SUPPORTED)`；4 个反向路径在准备前向 y 时已返回 9，尚未进入 Backward | 保留限制，不把反向准备失败写成 Backward 接口自身拒绝 |
| identity 多 dtype | BF16/INT32 的 Activation 返回 9，Transform 返回 8；BOOL/FP8 在设置 descriptor 时返回 9；FP16 Transform 虽返回成功但结果错误，现有 FP16 Activation 可正确执行 | 仅将已有有效 NVIDIA 数据的两种 FP8 接入等效路径；不放行错误的原生候选 |
| FP32/FP16 OpTensor 控制组 | add/mul/min/max × 两种 dtype × 对齐、奇数尺寸、channels-last 三种布局，共 24 个配置，直接执行与 Graph 均正确 | 保持原生支持；此前 add 的失败不能解释成 hipDNN 不支持 FP16/FP32 add |
| batchnorm 标准/Ex/转换候选，共 9 个配置 | FP32 标准与 Ex 正确；原目录 FP16 标准只保证 y 正确，四组 FP32 统计输出错误且部分未写；BF16 标准及低精度 Ex 返回 9；逐次 Transform 参数转换返回 8 | 不把 API 返回成功或只检查 y 当成整体通过。主机预先提升 scale/bias 后的 FP16 诊断虽通过，但改变了原目录参数 dtype，未作为原生适配 |
| FP16 → FP32 reduction 原始 descriptor，共 36 个 | 21 个二维配置正确，15 个高维配置结果错误；另有同 shape 的 36 个 FP32 控制组全部正确 | 混合类型输出需限制 descriptor 组合，不能一律放行 |
| 连续首/末轴 reduction 的二维视图，共 9 个 | 无拷贝地保留输入数据、绑定偏移、输出顺序及 FP32 类型；每批 50 次，直接执行、Graph、重新填充后回放及边界检查全部正确 | 与 21 个二维配置一起，新增 30 个原生对照。其余 6 个中间轴不适用此变换 |
| BF16 → FP32 reduction，共 14 个 | 全部独立进程实测返回执行失败，日志含 `ck::get_datatype_from_enum<ck::BFloat16>` 未定义模板实例；未进入 Graph 执行/回放 | 保留错误证据，等待有效参考，不生成耗时 |

逐元素共运行 52 个接口/配置组合（48 个唯一 case）；batchnorm 仅使用一个代表 shape 做上述 9 种配置，并未逐例执行其余 11 个低精度 shape。标准 FP16 batchnorm 与部分高维混合 reduction 的统计/输出缓冲区检查显示半精度写入及尾部未写迹象；即使调用返回成功也不能用于对照。

可复现工具与参数见 [hipDNN 探针说明](../tools/hipdnn_capability/README.md)。本机原始探测、构建与正式回归产物位于 `build/hygon/hipdnn-capability-20261012-HaHtKP/`，其中 `pointwise/`、`batchnorm/`、`reduction/` 保留 API 状态与正确性日志，`regression/` 保存正式测试输出。探针不是峰值测试，不据此覆盖公共 NVIDIA 基线或 Hygon 峰值。

### Hygon 正式回归状态与后续验收（历史状态保留）

- Hygon 全量构建成功；8 项 CPU 契约测试全部通过，涵盖目录、dtype、原生 descriptor、基线加载、计时策略和输出完整性。Python 批量脚本的解析、统计、设备及多平台策略契约也通过。
- GPU 正式回归在功能测试阶段中止：identity 已输出 37 个通过，add 已输出 37 个通过，尚未进入 performance，不能据此称这两个完整 suite 或全部新增路径已正式通过。独立探测完成后，另一个项目的作业开始分别使用 GPU 0、1；为避免干扰其执行及后续性能采样，仅中止了本次启动的批量进程，退出码 130，原日志保留。
- 新增 30 个 reduction 的 hipDNN 路径已完成独立 GPU 校验及生产 descriptor 的 CPU 契约验证；新增 24 个 identity FP8 的加载、元数据、原始字节及输出完整性逻辑通过 CPU 契约。两者仍需在空闲 GPU 上完成正式 benchmark 回归；未把未完成的测试改记为 passed。
- 此前 Hygon 修复仅新增私有 identity 换算策略；当时完整公共 `reference`、所有非 Hygon 平台对象和其余公共字段指纹一致，没有改 NVIDIA 耗时、峰值或公共 profile。本轮 NVIDIA 补测另行追加新记录，不改写这段历史验证结论。

获得独占测试 GPU 后，使用新的输出目录重新运行，不覆盖中止日志：

```bash
regression_dir=$(mktemp -d /tmp/flagdnn-hipdnn-regression-XXXXXX)
python3 tools/run_tests.py --platform hygon --build-dir build/hygon \
  --ops identity,add,div,reduction,batchnorm --suites all --gpus 0,1 \
  --dump-output --output-dir "$regression_dir"
```

验收需检查新增 24 条 identity FP8 等效记录的计时前后按位校验，以及 30 条 reduction 的完整 FlagDNN/hipDNN 配对。没有传 `--min-speedup` 时，低于性能基线只作报告；数值、执行、记录缺失等真实错误仍应失败。正式回归完成前，本清单不宣称已经达到全部功能与性能用例零跳过。

## 用例定义与不能省略的元数据

附录一列出本轮已补测、可精确复测的 371 个 case ID。shape、strides、张量角色、属性和输入生成规则应直接复用以下目录，不要只根据 case 名称重新拼装测试。

| 分组 | 权威用例定义 | 关键约束 |
|---|---|---|
| 17 个主逐元素算子和 identity 主目录 | [benchmark/test_*.cpp 的各算子入口](../benchmark)、[pointwise_cases.cpp](../benchmark/common/pointwise_cases.cpp)、[binary_pointwise_cases.cpp](../benchmark/common/binary_pointwise_cases.cpp)、[pointwise_graph_cases.cpp](../benchmark/common/pointwise_graph_cases.cpp)、[Hygon 目录选择器](../backends/hygon/validation/benchmark/cases.hpp) | 四维逐元素数据通常使用 channels-last strides，不能一律改成 contiguous；保留入口的属性覆盖与输入域 |
| identity copy、logical boolean | [tests/common/pointwise.cpp](../tests/common/pointwise.cpp)、[native_dtype.cpp](../backends/hygon/validation/benchmark/native_dtype.cpp) | identity 数值类型使用 `identity_shapes`，BOOL 使用另一组 `packed_boolean_shapes`；同一算子的两个目录不可混用 |
| batchnorm | [batchnorm_inference_cases.cpp](../benchmark/common/batchnorm_inference_cases.cpp) 中 `make_training_case`、`batchnorm_benchmark_cases` | 这是训练前向，不是 inference；有十个张量、五个输出，`epsilon=1e-3`、`momentum=0.1` |
| reduction | [tests/common/reduction.cpp](../tests/common/reduction.cpp) 的 `make_reduction_cases`、[native_dtype.cpp](../backends/hygon/validation/benchmark/native_dtype.cpp) 的 `fp32_output` 选择 | 保存 reduction mode、axis、keep_dimensions、输出 shape；输入绑定偏移 16 字节、输出 32 字节，不可丢失 |

需要特别保留的语义：

- `scale` 当前性能目录是两个同 shape 输入的 MUL，不是单个输入乘一个标量。
- `add_square` 是 `left + right * right` 的图，不能当成普通 ADD 或单个 unary 算子。
- batchnorm 的 x、scale、bias、y 使用该 case 的 FP16/BF16；输入运行统计及输出统计为 FP32。补测必须覆盖所有输出与运行统计更新，不能只测 y。
- batchnorm 的共享目录采用 NCHW contiguous，而当前 NVIDIA benchmark 适配会把 x/y 改为 channels-last 参考布局，转换不计入采样。必须同时保留逻辑 `strides` 与实际 `reference_strides`；后续 Hygon 策略应显式处理或声明该布局差异，不能声称两边物理工作负载完全相同。实现见 [NVIDIA runner.cpp](../backends/nvidia/validation/benchmark/runner.cpp) 的 `cudnn_tensor_specification`。
- reduction 原目录 FP16 → FP32 为 12 个 shape × sum/avg/mul，其中 10 个 shape 已接原生对照；本轮仅补 `3x5x7 axis=1`、`2x7x5x3 axis=2` 的 6 条。另有 BF16 → FP32 的 14 个二维 sum 用例，不应扩成同目录所有 mode 后误计入这 20 条。
- BOOL 要记录 NVIDIA 实际存储位宽。若参考为 bit-packed、Hygon 为 byte，后续带宽换算需按实际流量修正；不能把布尔逻辑张量元素数直接当成两边相同字节数。
- INT32 identity 与 FP8 identity 应按原始编码校验复制结果，不能经 FLOAT 中转后声称整数/FP8 按位正确。

### 计时配置

计时只包含设备执行，排除建图、编译、分配、主机与设备拷贝、输入准备和校验。保留原始样本，使用 GPU event 测量 Graph 批执行，取单次耗时的 nearest-rank P50；单位为微秒。

| 分组 | 预热执行次数 | 样本批数 | 每批执行次数 | Graph 预热回放 |
|---|---:|---:|---:|---:|
| 除 batchnorm 外的本清单用例 | 10 | 20 | 50 | 1 |
| batchnorm 24 个补测用例 | 5 | 10 | 20 | 1 |

batchnorm 按共享性能用例配置补测，并登记为独立的 `batchnorm_training` profile；不得标成固定为 10/20/50 的 `native`。Hygon 等效接入仍须显式匹配这个新 profile。采集器现在保留各 case 的实际配置，未修改已有 `native` 或 `attention` 的含义。

NVIDIA 等效参考维持上述 Graph 口径。本轮 Hygon 部分原生 hipDNN 配对采用的 `hip_event_direct_batch` 是原生对照兼容策略，不应混入 NVIDIA Graph 基准或伪装成同口径数据。

## NVIDIA 侧采集与交付要求

### 采集前准备

1. 使用与现有基准匹配的 NVIDIA H100 80GB HBM3 设备配置，并记录驱动、CUDA、cuDNN、cuDNN Frontend、设备标识、功率限制和源码版本。若更换设备，须使用独立的设备基准及对应峰值，不能把新设备耗时与旧 H100 峰值拼接。
2. 先导出采集器的用例目录，与附录一逐项核对。仓库中的正式 NVIDIA 测试目录、独立采集器目录和已入库数据是三个不同集合；重新执行旧采集命令不保证会覆盖新增用例。
3. 使用已扩展的 core 和新增 aux 采集入口，复用上节的原始 case 定义；不得将 FlagDNN 实现本身当作 NVIDIA 对照。
4. 保证 GPU 空闲、测量串行运行。对每个 case 先检查全量正确性和完整写入，再采样；失败须保留阶段、返回码及原因，不能写入有效 `time_us`。

已有工具及编译前提见 [工具说明](../tools/performance_baseline/README.md)。核心入口为 [benchmark_nvidia_cudnn.cpp](../tools/performance_baseline/benchmark_nvidia_cudnn.cpp) 与 [benchmark_nvidia_core_catalog.hpp](../tools/performance_baseline/benchmark_nvidia_core_catalog.hpp)；扩展入口为 [benchmark_nvidia_extended.cpp](../tools/performance_baseline/benchmark_nvidia_extended.cpp)；属性、整数、FP8 复制探测入口为 [benchmark_nvidia_pointwise_gaps.cpp](../tools/performance_baseline/benchmark_nvidia_pointwise_gaps.cpp)。

补测前的采集器缺口及本轮实现：

- core catalog 原先未捕获上述 17 个主逐元素入口及 identity、batchnorm 训练；本轮已捕获这些真实入口，共采集 291 条。
- reduction 的 20 条来自独立的 `fp32_output` 辅助目录，本轮由 aux 复用真实 shared cases 的精确选择；加入普通 `test_reduction.cpp` 不能替代这些用例。其余 30 条已有原生 hipDNN 对照，不计入本轮。
- gaps 的 identity 分支只有 FP8 E4M3/E5M2，extended 只有 logical_not 辅助目录；本轮 aux 补齐 36 条 BF16/INT32/BOOL identity copy 和 24 条 logical_and/or boolean，不借用其他目录耗时。
- core 原先统一为 `10/20/50`；本轮已保留 batchnorm 的 `5/10/20` 和五个输出校验证据。core/aux 导出完整属性、逐输入生成规则、张量元数据、三阶段校验及原始样本。

以下保留单入口构建与列目录示例；完整的 core + aux 精确 371 条调度命令以[工具说明](../tools/performance_baseline/README.md#2026-10-12-缺失目录补测)为准。先创建独立临时目录，避免覆盖旧测量产物：

```bash
export CUDNN_ROOT=/path/to/cudnn
export CUDNN_FRONTEND_INCLUDE_DIR=/path/to/cudnn_frontend/include
nv_measure_dir=$(mktemp -d /tmp/flagdnn-nv-missing-XXXXXX)
bash tools/performance_baseline/build_benchmark_nvidia_cudnn.sh "$nv_measure_dir/core"
"$nv_measure_dir/core" --list
python3 tools/performance_baseline/benchmark_nvidia_extended.py --build-only \
  --binary "$nv_measure_dir/extended" --output "$nv_measure_dir/extended.jsonl"
```

### 每个用例需要交付的数据

- 完整的 `op`、`case_name`，不得重命名成其他目录的已有用例。
- 所有输入、输出、图中间张量的角色、UID、dtype、shape、strides、绑定字节偏移；使用了参考侧布局转换时，另外保存 `reference_strides` 及转换是否计时。
- 所有算子属性及输入生成规则；例如 pointwise flags/alpha/swish_beta，batchnorm epsilon/momentum，reduction mode/axis/keep_dimensions。
- 预热次数、样本数、每批执行次数、Graph 预热次数、计时方法、全部样本、P50/P90 和单位；仅提供一个最优时间不够。
- 校验类型、误差容限、最大误差、校验元素数、计时前后结果，以及重置输出后回放的完整写入验证；整数、BOOL、复制应采用相应精确校验。
- 原始日志、源码版本、设备与软件环境、测量日期。采用 `sanity` 时必须如实标注，不能称为全量精度通过。

本轮补测内容是算子参考耗时与上述元数据，没有重新测量或覆盖已有 16 项 NVIDIA 峰值。后续选择换算指标时，应依据实际计算路径与数据流量；不能仅因为存储类型是 BF16，就自动选 `vector_bf16`。

### 入库与验收

- 新增成功记录前检查 `(op, case_name)`、全部张量元数据、属性及计时配置一致；不使用相近 shape、其他 dtype 或其他算子的时间代替。
- 仅追加经过校验的新记录与必要的来源、profile、coverage 信息，保留原有 2893 条 NVIDIA 有效记录、峰值和已有 profile；不要批量重写公共数据。
- 新记录的共享 `scale_metric` 保持 `null`。Hygon 的换算策略只写入其私有 `platforms.hygon.scaling.case_policies`，不能借补测修改其他平台使用的参考数据。
- 如果 NVIDIA 原生接口也无法构建、执行或通过校验，应保留 `unavailable_cases` 原因，不提供虚假的有效时间。对这类用例，仍需另行解决参考路径，不能靠改状态实现零跳过。
- 补齐 NVIDIA 数据只是前置条件。Hygon 还需要新增对应等效策略、dtype/属性加载与元数据检查，并实际复跑；371 条数据入库不等于 371 个 Hygon 用例已经通过。

## 已有数据与目录外任务

| 范围 | 数量 | 处理方式 |
|---|---:|---|
| identity FP8 E4M3/E5M2 | 24 | 已有 `bit_exact`、`native` profile 的有效 NVIDIA 记录，见附录二；本轮已接入 Hygon 等效路径，无需补测 |
| concatenate INT32/BOOL/FP8 E4M3/E5M2/E8M0 | 70 | 当前已有 `unavailable_cases.status=excluded`，无有效时间；若目标覆盖这些排除用例，先扩展 NVIDIA 原生参考目录再尝试测量，见附录三 |
| identity FP8 E8M0 | 12 | 功能目录包含，但 NVIDIA/Hygon 性能选择器都过滤掉；不在 425 个 skip 内；若纳入全 dtype 性能验收，需先建立可验证的参考路径，见附录四 |

70 条 concatenate 是 5 个 dtype 各 14 条。上述目录外任务不应悄悄计入当前 371 条补测完成率。当前注释算子及其他未注册的 dtype/shape 也不在本清单的完成范围。

concatenate 的精确定义见 [tests/common/index.cpp](../tests/common/index.cpp) 的 `make_index_cases`。必须保留输入个数、axis、各输入输出 shape 与带 padding 的 strides；case 名称中的 shape 不等于最终拼接输出 shape。

## 附录一 剩余需 NVIDIA 参考路线补齐的 371 个用例

以下保留补测前登记的完整 case ID，共 22 组、371 条；现已完成 NVIDIA 采集，仍用作精确复测清单。名称中的 `bfloat16` 即 BF16；`_perf_` 与无该前缀的辅助目录必须保留区别。

### abs 15 个

```text
abs_perf_bfloat16_1x1x1024
abs_perf_bfloat16_8x16x32
abs_perf_bfloat16_4x16x64x128
abs_perf_bfloat16_8x16x64x128
abs_perf_bfloat16_1x1x1000
abs_perf_bfloat16_3x257x513
abs_perf_bfloat16_3x7x65x129
abs_perf_bfloat16_5x7x65x129
abs_perf_bfloat16_1x1x4096
abs_perf_bfloat16_3x8x128
abs_perf_bfloat16_2x32x17x19
abs_perf_bfloat16_4x64x31x33
abs_perf_bfloat16_16x32x64
abs_perf_bfloat16_8x128x128
abs_perf_bfloat16_2x256x63x65
```

### add 15 个

```text
add_perf_bfloat16_1x1x1024_by_1x1x1024
add_perf_bfloat16_8x16x32_by_8x16x32
add_perf_bfloat16_4x16x64x128_by_4x16x64x128
add_perf_bfloat16_8x16x64x128_by_8x16x64x128
add_perf_bfloat16_1x1x1000_by_1x1x1000
add_perf_bfloat16_3x257x513_by_3x257x513
add_perf_bfloat16_3x7x65x129_by_3x7x65x129
add_perf_bfloat16_5x7x65x129_by_5x7x65x129
add_perf_bfloat16_1x1x4096_by_1x1x4096
add_perf_bfloat16_3x8x128_by_3x8x128
add_perf_bfloat16_2x32x17x19_by_2x32x17x19
add_perf_bfloat16_4x64x31x33_by_4x64x31x33
add_perf_bfloat16_16x32x64_by_16x32x64
add_perf_bfloat16_8x128x128_by_8x128x128
add_perf_bfloat16_2x256x63x65_by_2x256x63x65
```

### add_square 15 个

```text
add_square_perf_bfloat16_1x1x1024
add_square_perf_bfloat16_8x16x32
add_square_perf_bfloat16_4x16x64x128
add_square_perf_bfloat16_8x16x64x128
add_square_perf_bfloat16_1x1x1000
add_square_perf_bfloat16_3x257x513
add_square_perf_bfloat16_3x7x65x129
add_square_perf_bfloat16_5x7x65x129
add_square_perf_bfloat16_1x1x4096
add_square_perf_bfloat16_3x8x128
add_square_perf_bfloat16_2x32x17x19
add_square_perf_bfloat16_4x64x31x33
add_square_perf_bfloat16_16x32x64
add_square_perf_bfloat16_8x128x128
add_square_perf_bfloat16_2x256x63x65
```

### batchnorm 24 个

```text
batchnorm_perf_fp16_8x32x32x32
batchnorm_perf_bfloat16_8x32x32x32
batchnorm_perf_fp16_16x64x16x16
batchnorm_perf_bfloat16_16x64x16x16
batchnorm_perf_fp16_4x128x16x16
batchnorm_perf_bfloat16_4x128x16x16
batchnorm_perf_fp16_8x64x56x56
batchnorm_perf_bfloat16_8x64x56x56
batchnorm_perf_fp16_16x128x28x28
batchnorm_perf_bfloat16_16x128x28x28
batchnorm_perf_fp16_16x256x14x14
batchnorm_perf_bfloat16_16x256x14x14
batchnorm_perf_fp16_8x512x7x7
batchnorm_perf_bfloat16_8x512x7x7
batchnorm_perf_fp16_32x1024x1x1
batchnorm_perf_bfloat16_32x1024x1x1
batchnorm_perf_fp16_2x16x31x33
batchnorm_perf_bfloat16_2x16x31x33
batchnorm_perf_fp16_4x32x15x17
batchnorm_perf_bfloat16_4x32x15x17
batchnorm_perf_fp16_2x64x7x9
batchnorm_perf_bfloat16_2x64x7x9
batchnorm_perf_fp16_1x128x8x8
batchnorm_perf_bfloat16_1x128x8x8
```

### elu 15 个

```text
elu_perf_bfloat16_1x1x1024
elu_perf_bfloat16_8x16x32
elu_perf_bfloat16_4x16x64x128
elu_perf_bfloat16_8x16x64x128
elu_perf_bfloat16_1x1x1000
elu_perf_bfloat16_3x257x513
elu_perf_bfloat16_3x7x65x129
elu_perf_bfloat16_5x7x65x129
elu_perf_bfloat16_1x1x4096
elu_perf_bfloat16_3x8x128
elu_perf_bfloat16_2x32x17x19
elu_perf_bfloat16_4x64x31x33
elu_perf_bfloat16_16x32x64
elu_perf_bfloat16_8x128x128
elu_perf_bfloat16_2x256x63x65
```

### identity 48 个

```text
identity_perf_bfloat16_2x3x4
identity_perf_bfloat16_8x16x32
identity_perf_bfloat16_64x64x64
identity_perf_bfloat16_16x64x128
identity_perf_bfloat16_16x256x256
identity_perf_bfloat16_32x128x256
identity_perf_bfloat16_4x1024x1024
identity_perf_bfloat16_16x512x1024
identity_perf_bfloat16_2x2048x2048
identity_perf_bfloat16_8x1024x2048
identity_perf_bfloat16_1x1x32
identity_perf_bfloat16_3x257x513
identity_bfloat16_2x3x4
identity_int32_2x3x4
identity_bool_1x1x16
identity_bfloat16_4x5x6
identity_int32_4x5x6
identity_bool_2x8x8
identity_bfloat16_1x8x16
identity_int32_1x8x16
identity_bool_1x8x4x8
identity_bfloat16_3x1x17
identity_int32_3x1x17
identity_bool_2x8x4x8
identity_bfloat16_2x4x8
identity_int32_2x4x8
identity_bool_1x1x8
identity_bfloat16_5x7x11
identity_int32_5x7x11
identity_bool_1x1x24
identity_bfloat16_1x33x65
identity_int32_1x33x65
identity_bool_1x1x128
identity_bfloat16_2x16x257
identity_int32_2x16x257
identity_bool_1x1x256
identity_bfloat16_4x32x128
identity_int32_4x32x128
identity_bool_3x5x32
identity_bfloat16_1x1x2
identity_int32_1x1x2
identity_bool_1x16x3x5
identity_bfloat16_1x1x3
identity_int32_1x1x3
identity_bool_2x16x7x9
identity_bfloat16_2x3x17
identity_int32_2x3x17
identity_bool_3x32x5x7
```

### logical_and 12 个

```text
logical_and_bool_1x1x16
logical_and_bool_2x8x8
logical_and_bool_1x8x4x8
logical_and_bool_2x8x4x8
logical_and_bool_1x1x8
logical_and_bool_1x1x24
logical_and_bool_1x1x128
logical_and_bool_1x1x256
logical_and_bool_3x5x32
logical_and_bool_1x16x3x5
logical_and_bool_2x16x7x9
logical_and_bool_3x32x5x7
```

### logical_or 12 个

```text
logical_or_bool_1x1x16
logical_or_bool_2x8x8
logical_or_bool_1x8x4x8
logical_or_bool_2x8x4x8
logical_or_bool_1x1x8
logical_or_bool_1x1x24
logical_or_bool_1x1x128
logical_or_bool_1x1x256
logical_or_bool_3x5x32
logical_or_bool_1x16x3x5
logical_or_bool_2x16x7x9
logical_or_bool_3x32x5x7
```

### max 15 个

```text
max_perf_bfloat16_1x1x1024_by_1x1x1024
max_perf_bfloat16_8x16x32_by_8x16x32
max_perf_bfloat16_4x16x64x128_by_4x16x64x128
max_perf_bfloat16_8x16x64x128_by_8x16x64x128
max_perf_bfloat16_1x1x1000_by_1x1x1000
max_perf_bfloat16_3x257x513_by_3x257x513
max_perf_bfloat16_3x7x65x129_by_3x7x65x129
max_perf_bfloat16_5x7x65x129_by_5x7x65x129
max_perf_bfloat16_1x1x4096_by_1x1x4096
max_perf_bfloat16_3x8x128_by_3x8x128
max_perf_bfloat16_2x32x17x19_by_2x32x17x19
max_perf_bfloat16_4x64x31x33_by_4x64x31x33
max_perf_bfloat16_16x32x64_by_16x32x64
max_perf_bfloat16_8x128x128_by_8x128x128
max_perf_bfloat16_2x256x63x65_by_2x256x63x65
```

### min 15 个

```text
min_perf_bfloat16_1x1x1024_by_1x1x1024
min_perf_bfloat16_8x16x32_by_8x16x32
min_perf_bfloat16_4x16x64x128_by_4x16x64x128
min_perf_bfloat16_8x16x64x128_by_8x16x64x128
min_perf_bfloat16_1x1x1000_by_1x1x1000
min_perf_bfloat16_3x257x513_by_3x257x513
min_perf_bfloat16_3x7x65x129_by_3x7x65x129
min_perf_bfloat16_5x7x65x129_by_5x7x65x129
min_perf_bfloat16_1x1x4096_by_1x1x4096
min_perf_bfloat16_3x8x128_by_3x8x128
min_perf_bfloat16_2x32x17x19_by_2x32x17x19
min_perf_bfloat16_4x64x31x33_by_4x64x31x33
min_perf_bfloat16_16x32x64_by_16x32x64
min_perf_bfloat16_8x128x128_by_8x128x128
min_perf_bfloat16_2x256x63x65_by_2x256x63x65
```

### mul 15 个

```text
mul_perf_bfloat16_1x1x1024_by_1x1x1024
mul_perf_bfloat16_8x16x32_by_8x16x32
mul_perf_bfloat16_4x16x64x128_by_4x16x64x128
mul_perf_bfloat16_8x16x64x128_by_8x16x64x128
mul_perf_bfloat16_1x1x1000_by_1x1x1000
mul_perf_bfloat16_3x257x513_by_3x257x513
mul_perf_bfloat16_3x7x65x129_by_3x7x65x129
mul_perf_bfloat16_5x7x65x129_by_5x7x65x129
mul_perf_bfloat16_1x1x4096_by_1x1x4096
mul_perf_bfloat16_3x8x128_by_3x8x128
mul_perf_bfloat16_2x32x17x19_by_2x32x17x19
mul_perf_bfloat16_4x64x31x33_by_4x64x31x33
mul_perf_bfloat16_16x32x64_by_16x32x64
mul_perf_bfloat16_8x128x128_by_8x128x128
mul_perf_bfloat16_2x256x63x65_by_2x256x63x65
```

### reduction 20 个

```text
reduction_sum_bfloat16_to_fp32_3_7
reduction_sum_bfloat16_to_fp32_2_17
reduction_sum_bfloat16_to_fp32_4_31
reduction_sum_bfloat16_to_fp32_2_33
reduction_sum_bfloat16_to_fp32_3_65
reduction_sum_bfloat16_to_fp32_2_127
reduction_sum_bfloat16_to_fp32_2_257
reduction_sum_fp16_to_fp32_3_5_7
reduction_avg_fp16_to_fp32_3_5_7
reduction_mul_fp16_to_fp32_3_5_7
reduction_sum_fp16_to_fp32_2_7_5_3
reduction_avg_fp16_to_fp32_2_7_5_3
reduction_mul_fp16_to_fp32_2_7_5_3
reduction_sum_bfloat16_to_fp32_8_8
reduction_sum_bfloat16_to_fp32_9_16
reduction_sum_bfloat16_to_fp32_16_32
reduction_sum_bfloat16_to_fp32_17_64
reduction_sum_bfloat16_to_fp32_32_128
reduction_sum_bfloat16_to_fp32_33_256
reduction_sum_bfloat16_to_fp32_64_512
```

### relu 15 个

```text
relu_perf_bfloat16_1x1x1024
relu_perf_bfloat16_8x16x32
relu_perf_bfloat16_4x16x64x128
relu_perf_bfloat16_8x16x64x128
relu_perf_bfloat16_1x1x1000
relu_perf_bfloat16_3x257x513
relu_perf_bfloat16_3x7x65x129
relu_perf_bfloat16_5x7x65x129
relu_perf_bfloat16_1x1x4096
relu_perf_bfloat16_3x8x128
relu_perf_bfloat16_2x32x17x19
relu_perf_bfloat16_4x64x31x33
relu_perf_bfloat16_16x32x64
relu_perf_bfloat16_8x128x128
relu_perf_bfloat16_2x256x63x65
```

### relu_backward 15 个

```text
relu_backward_perf_bfloat16_1x1x1024_by_1x1x1024
relu_backward_perf_bfloat16_8x16x32_by_8x16x32
relu_backward_perf_bfloat16_4x16x64x128_by_4x16x64x128
relu_backward_perf_bfloat16_8x16x64x128_by_8x16x64x128
relu_backward_perf_bfloat16_1x1x1000_by_1x1x1000
relu_backward_perf_bfloat16_3x257x513_by_3x257x513
relu_backward_perf_bfloat16_3x7x65x129_by_3x7x65x129
relu_backward_perf_bfloat16_5x7x65x129_by_5x7x65x129
relu_backward_perf_bfloat16_1x1x4096_by_1x1x4096
relu_backward_perf_bfloat16_3x8x128_by_3x8x128
relu_backward_perf_bfloat16_2x32x17x19_by_2x32x17x19
relu_backward_perf_bfloat16_4x64x31x33_by_4x64x31x33
relu_backward_perf_bfloat16_16x32x64_by_16x32x64
relu_backward_perf_bfloat16_8x128x128_by_8x128x128
relu_backward_perf_bfloat16_2x256x63x65_by_2x256x63x65
```

### scale 15 个

```text
scale_perf_bfloat16_1x1x1024_by_1x1x1024
scale_perf_bfloat16_8x16x32_by_8x16x32
scale_perf_bfloat16_4x16x64x128_by_4x16x64x128
scale_perf_bfloat16_8x16x64x128_by_8x16x64x128
scale_perf_bfloat16_1x1x1000_by_1x1x1000
scale_perf_bfloat16_3x257x513_by_3x257x513
scale_perf_bfloat16_3x7x65x129_by_3x7x65x129
scale_perf_bfloat16_5x7x65x129_by_5x7x65x129
scale_perf_bfloat16_1x1x4096_by_1x1x4096
scale_perf_bfloat16_3x8x128_by_3x8x128
scale_perf_bfloat16_2x32x17x19_by_2x32x17x19
scale_perf_bfloat16_4x64x31x33_by_4x64x31x33
scale_perf_bfloat16_16x32x64_by_16x32x64
scale_perf_bfloat16_8x128x128_by_8x128x128
scale_perf_bfloat16_2x256x63x65_by_2x256x63x65
```

### sigmoid 15 个

```text
sigmoid_perf_bfloat16_1x1x1024
sigmoid_perf_bfloat16_8x16x32
sigmoid_perf_bfloat16_4x16x64x128
sigmoid_perf_bfloat16_8x16x64x128
sigmoid_perf_bfloat16_1x1x1000
sigmoid_perf_bfloat16_3x257x513
sigmoid_perf_bfloat16_3x7x65x129
sigmoid_perf_bfloat16_5x7x65x129
sigmoid_perf_bfloat16_1x1x4096
sigmoid_perf_bfloat16_3x8x128
sigmoid_perf_bfloat16_2x32x17x19
sigmoid_perf_bfloat16_4x64x31x33
sigmoid_perf_bfloat16_16x32x64
sigmoid_perf_bfloat16_8x128x128
sigmoid_perf_bfloat16_2x256x63x65
```

### sigmoid_backward 15 个

```text
sigmoid_backward_perf_bfloat16_1x1x1024_by_1x1x1024
sigmoid_backward_perf_bfloat16_8x16x32_by_8x16x32
sigmoid_backward_perf_bfloat16_4x16x64x128_by_4x16x64x128
sigmoid_backward_perf_bfloat16_8x16x64x128_by_8x16x64x128
sigmoid_backward_perf_bfloat16_1x1x1000_by_1x1x1000
sigmoid_backward_perf_bfloat16_3x257x513_by_3x257x513
sigmoid_backward_perf_bfloat16_3x7x65x129_by_3x7x65x129
sigmoid_backward_perf_bfloat16_5x7x65x129_by_5x7x65x129
sigmoid_backward_perf_bfloat16_1x1x4096_by_1x1x4096
sigmoid_backward_perf_bfloat16_3x8x128_by_3x8x128
sigmoid_backward_perf_bfloat16_2x32x17x19_by_2x32x17x19
sigmoid_backward_perf_bfloat16_4x64x31x33_by_4x64x31x33
sigmoid_backward_perf_bfloat16_16x32x64_by_16x32x64
sigmoid_backward_perf_bfloat16_8x128x128_by_8x128x128
sigmoid_backward_perf_bfloat16_2x256x63x65_by_2x256x63x65
```

### sub 15 个

```text
sub_perf_bfloat16_1x1x1024_by_1x1x1024
sub_perf_bfloat16_8x16x32_by_8x16x32
sub_perf_bfloat16_4x16x64x128_by_4x16x64x128
sub_perf_bfloat16_8x16x64x128_by_8x16x64x128
sub_perf_bfloat16_1x1x1000_by_1x1x1000
sub_perf_bfloat16_3x257x513_by_3x257x513
sub_perf_bfloat16_3x7x65x129_by_3x7x65x129
sub_perf_bfloat16_5x7x65x129_by_5x7x65x129
sub_perf_bfloat16_1x1x4096_by_1x1x4096
sub_perf_bfloat16_3x8x128_by_3x8x128
sub_perf_bfloat16_2x32x17x19_by_2x32x17x19
sub_perf_bfloat16_4x64x31x33_by_4x64x31x33
sub_perf_bfloat16_16x32x64_by_16x32x64
sub_perf_bfloat16_8x128x128_by_8x128x128
sub_perf_bfloat16_2x256x63x65_by_2x256x63x65
```

### swish 15 个

```text
swish_perf_bfloat16_1x1x1024
swish_perf_bfloat16_8x16x32
swish_perf_bfloat16_4x16x64x128
swish_perf_bfloat16_8x16x64x128
swish_perf_bfloat16_1x1x1000
swish_perf_bfloat16_3x257x513
swish_perf_bfloat16_3x7x65x129
swish_perf_bfloat16_5x7x65x129
swish_perf_bfloat16_1x1x4096
swish_perf_bfloat16_3x8x128
swish_perf_bfloat16_2x32x17x19
swish_perf_bfloat16_4x64x31x33
swish_perf_bfloat16_16x32x64
swish_perf_bfloat16_8x128x128
swish_perf_bfloat16_2x256x63x65
```

### swish_backward 15 个

```text
swish_backward_perf_bfloat16_1x1x1024_by_1x1x1024
swish_backward_perf_bfloat16_8x16x32_by_8x16x32
swish_backward_perf_bfloat16_4x16x64x128_by_4x16x64x128
swish_backward_perf_bfloat16_8x16x64x128_by_8x16x64x128
swish_backward_perf_bfloat16_1x1x1000_by_1x1x1000
swish_backward_perf_bfloat16_3x257x513_by_3x257x513
swish_backward_perf_bfloat16_3x7x65x129_by_3x7x65x129
swish_backward_perf_bfloat16_5x7x65x129_by_5x7x65x129
swish_backward_perf_bfloat16_1x1x4096_by_1x1x4096
swish_backward_perf_bfloat16_3x8x128_by_3x8x128
swish_backward_perf_bfloat16_2x32x17x19_by_2x32x17x19
swish_backward_perf_bfloat16_4x64x31x33_by_4x64x31x33
swish_backward_perf_bfloat16_16x32x64_by_16x32x64
swish_backward_perf_bfloat16_8x128x128_by_8x128x128
swish_backward_perf_bfloat16_2x256x63x65_by_2x256x63x65
```

### tanh 15 个

```text
tanh_perf_bfloat16_1x1x1024
tanh_perf_bfloat16_8x16x32
tanh_perf_bfloat16_4x16x64x128
tanh_perf_bfloat16_8x16x64x128
tanh_perf_bfloat16_1x1x1000
tanh_perf_bfloat16_3x257x513
tanh_perf_bfloat16_3x7x65x129
tanh_perf_bfloat16_5x7x65x129
tanh_perf_bfloat16_1x1x4096
tanh_perf_bfloat16_3x8x128
tanh_perf_bfloat16_2x32x17x19
tanh_perf_bfloat16_4x64x31x33
tanh_perf_bfloat16_16x32x64
tanh_perf_bfloat16_8x128x128
tanh_perf_bfloat16_2x256x63x65
```

### tanh_backward 15 个

```text
tanh_backward_perf_bfloat16_1x1x1024_by_1x1x1024
tanh_backward_perf_bfloat16_8x16x32_by_8x16x32
tanh_backward_perf_bfloat16_4x16x64x128_by_4x16x64x128
tanh_backward_perf_bfloat16_8x16x64x128_by_8x16x64x128
tanh_backward_perf_bfloat16_1x1x1000_by_1x1x1000
tanh_backward_perf_bfloat16_3x257x513_by_3x257x513
tanh_backward_perf_bfloat16_3x7x65x129_by_3x7x65x129
tanh_backward_perf_bfloat16_5x7x65x129_by_5x7x65x129
tanh_backward_perf_bfloat16_1x1x4096_by_1x1x4096
tanh_backward_perf_bfloat16_3x8x128_by_3x8x128
tanh_backward_perf_bfloat16_2x32x17x19_by_2x32x17x19
tanh_backward_perf_bfloat16_4x64x31x33_by_4x64x31x33
tanh_backward_perf_bfloat16_16x32x64_by_16x32x64
tanh_backward_perf_bfloat16_8x128x128_by_8x128x128
tanh_backward_perf_bfloat16_2x256x63x65_by_2x256x63x65
```

## 附录二 已有 NVIDIA 记录的 24 个 identity FP8 用例

以下用例已接入 Hygon 等效性能，不计入当前 371 条补测清单。

```text
identity_fp8_e4m3_2x3x4
identity_fp8_e5m2_2x3x4
identity_fp8_e4m3_4x5x6
identity_fp8_e5m2_4x5x6
identity_fp8_e4m3_1x8x16
identity_fp8_e5m2_1x8x16
identity_fp8_e4m3_3x1x17
identity_fp8_e5m2_3x1x17
identity_fp8_e4m3_2x4x8
identity_fp8_e5m2_2x4x8
identity_fp8_e4m3_5x7x11
identity_fp8_e5m2_5x7x11
identity_fp8_e4m3_1x33x65
identity_fp8_e5m2_1x33x65
identity_fp8_e4m3_2x16x257
identity_fp8_e5m2_2x16x257
identity_fp8_e4m3_4x32x128
identity_fp8_e5m2_4x32x128
identity_fp8_e4m3_1x1x2
identity_fp8_e5m2_1x1x2
identity_fp8_e4m3_1x1x3
identity_fp8_e5m2_1x1x3
identity_fp8_e4m3_2x3x17
identity_fp8_e5m2_2x3x17
```

## 附录三 当前排除的 70 个 concatenate 用例

以下用例需要先建立可用参考，不能把现有 excluded 记录改成测量成功。

```text
concatenate_int32_1_axis0
concatenate_bool_1_axis0
concatenate_fp8_e4m3_1_axis0
concatenate_fp8_e5m2_1_axis0
concatenate_fp8_e8m0_1_axis0
concatenate_int32_17_axis-1
concatenate_bool_17_axis-1
concatenate_fp8_e4m3_17_axis-1
concatenate_fp8_e5m2_17_axis-1
concatenate_fp8_e8m0_17_axis-1
concatenate_int32_33_axis0
concatenate_bool_33_axis0
concatenate_fp8_e4m3_33_axis0
concatenate_fp8_e5m2_33_axis0
concatenate_fp8_e8m0_33_axis0
concatenate_int32_2_3_axis-1
concatenate_bool_2_3_axis-1
concatenate_fp8_e4m3_2_3_axis-1
concatenate_fp8_e5m2_2_3_axis-1
concatenate_fp8_e8m0_2_3_axis-1
concatenate_int32_3_7_axis0
concatenate_bool_3_7_axis0
concatenate_fp8_e4m3_3_7_axis0
concatenate_fp8_e5m2_3_7_axis0
concatenate_fp8_e8m0_3_7_axis0
concatenate_int32_2_3_5_axis-1
concatenate_bool_2_3_5_axis-1
concatenate_fp8_e4m3_2_3_5_axis-1
concatenate_fp8_e5m2_2_3_5_axis-1
concatenate_fp8_e8m0_2_3_5_axis-1
concatenate_int32_1_3_17_axis0
concatenate_bool_1_3_17_axis0
concatenate_fp8_e4m3_1_3_17_axis0
concatenate_fp8_e5m2_1_3_17_axis0
concatenate_fp8_e8m0_1_3_17_axis0
concatenate_int32_2_4_33_axis-2
concatenate_bool_2_4_33_axis-2
concatenate_fp8_e4m3_2_4_33_axis-2
concatenate_fp8_e5m2_2_4_33_axis-2
concatenate_fp8_e8m0_2_4_33_axis-2
concatenate_int32_2_3_4_5_axis0
concatenate_bool_2_3_4_5_axis0
concatenate_fp8_e4m3_2_3_4_5_axis0
concatenate_fp8_e5m2_2_3_4_5_axis0
concatenate_fp8_e8m0_2_3_4_5_axis0
concatenate_int32_1_7_3_9_axis-3
concatenate_bool_1_7_3_9_axis-3
concatenate_fp8_e4m3_1_7_3_9_axis-3
concatenate_fp8_e5m2_1_7_3_9_axis-3
concatenate_fp8_e8m0_1_7_3_9_axis-3
concatenate_int32_2_2_2_2_3_axis0
concatenate_bool_2_2_2_2_3_axis0
concatenate_fp8_e4m3_2_2_2_2_3_axis0
concatenate_fp8_e5m2_2_2_2_2_3_axis0
concatenate_fp8_e8m0_2_2_2_2_3_axis0
concatenate_int32_1_2_1_3_1_2_1_5_axis-5
concatenate_bool_1_2_1_3_1_2_1_5_axis-5
concatenate_fp8_e4m3_1_2_1_3_1_2_1_5_axis-5
concatenate_fp8_e5m2_1_2_1_3_1_2_1_5_axis-5
concatenate_fp8_e8m0_1_2_1_3_1_2_1_5_axis-5
concatenate_int32_3_257_axis0
concatenate_bool_3_257_axis0
concatenate_fp8_e4m3_3_257_axis0
concatenate_fp8_e5m2_3_257_axis0
concatenate_fp8_e8m0_3_257_axis0
concatenate_int32_2_1025_axis-1
concatenate_bool_2_1025_axis-1
concatenate_fp8_e4m3_2_1025_axis-1
concatenate_fp8_e5m2_2_1025_axis-1
concatenate_fp8_e8m0_2_1025_axis-1
```

## 附录四 尚未注册性能测试的 12 个 identity E8M0 用例

这些名称对应共享功能目录；当前性能选择器不会执行它们。

```text
identity_fp8_e8m0_2x3x4
identity_fp8_e8m0_4x5x6
identity_fp8_e8m0_1x8x16
identity_fp8_e8m0_3x1x17
identity_fp8_e8m0_2x4x8
identity_fp8_e8m0_5x7x11
identity_fp8_e8m0_1x33x65
identity_fp8_e8m0_2x16x257
identity_fp8_e8m0_4x32x128
identity_fp8_e8m0_1x1x2
identity_fp8_e8m0_1x1x3
identity_fp8_e8m0_2x3x17
```
