# 跨平台性能基准工具

这里集中保存峰值和 cuDNN 参考耗时的测量工具。数据格式、换算公式和校验范围见 [基准说明](../../docs/equivalent-performance-baseline.md)，正式数据保存在 [performance-baseline.json](../../docs/performance-baseline.json)。

测量摘要统一保存在正式基线末尾的 `measurement_summaries.<measurement_id>`，
包含平台峰值配置明细和 NVIDIA 补测记录。平台的 `measurement.summary` 与
NVIDIA 补测的 `manifest` 使用 JSON Pointer 引用相应摘要；正式峰值和参考耗时仍保留原有结构。
原始 CSV、日志、源码快照和编译产物留在本地 `build/`；已有原始归档位于
`build/performance-baseline/evidence/`，路径和 SHA-256 保存在摘要中，不随源码分发。
运行时只读取正式基线，不依赖本地原始归档。

需要已配置的 `build/nvidia`、CUDA、cuDNN 和 cuDNN Frontend；张量峰值脚本还需要支持 CUDA 的 PyTorch。构建前设置：

```bash
export CUDNN_ROOT=/path/to/cudnn
export CUDNN_FRONTEND_INCLUDE_DIR=/path/to/cudnn_frontend/include
# 可选：CUDA_ROOT；Shell 构建脚本还支持 NVIDIA_BUILD_DIR（默认 build/nvidia）
```

从仓库根目录运行：

```bash
# 核心算子与显式 TF32
bash tools/performance_baseline/build_benchmark_nvidia_cudnn.sh /tmp/nv_core
bash tools/performance_baseline/build_benchmark_nvidia_cudnn.sh /tmp/nv_tf32 precision
/tmp/nv_core --list
/tmp/nv_tf32 --self-test

# 扩展算子、Attention（--build-only 仅构建并列目录）
python3 tools/performance_baseline/benchmark_nvidia_extended.py --build-only
python3 tools/performance_baseline/benchmark_nvidia_extended.py --shared-context --output /tmp/nv_extended.jsonl

# 特殊属性、精确 INT32 和 FP8 identity 探测
bash tools/performance_baseline/build_benchmark_nvidia_pointwise_gaps.sh /tmp/nv_gaps
python3 tools/performance_baseline/run_benchmark_nvidia_pointwise_gaps.py --binary /tmp/nv_gaps --shared-groups --output /tmp/nv_gaps.jsonl

# 张量峰值
python3 tools/performance_baseline/benchmark_nvidia_tensor.py --output /tmp/nv_tensor.json
```

## 2026-10-12 缺失目录补测

[正式基线](../../docs/performance-baseline.json)中标记 `measurement_id=nvidia-missing-20261012` 的 371 条由两个采集器提供：核心采集器覆盖 291 条主逐元素、identity 主目录和 batchnorm 训练用例；[辅助采集器](benchmark_nvidia_missing_aux.cpp)覆盖 80 条 identity copy、logical boolean 和混合输出 reduction 用例。[调度脚本](run_benchmark_nvidia_missing.py)默认从正式基线读取这批用例，按完整 `(op, case_name)` 匹配目录，拒绝遗漏和重复，串行运行并在失败后重启进程。仍可通过 `--report` 显式指定含“附录一”的 Markdown 补测清单。

构建目录须已配置 NVIDIA 后端，并启用 `FLAGDNN_BUILD_TESTS` 和 `FLAGDNN_BUILD_BENCHMARKS`；沿用上面的 CUDA/cuDNN 环境变量。辅助构建脚本只链接已有验证库，不会自动运行 CMake 构建：

```bash
nv_missing_dir=$(mktemp -d /tmp/flagdnn-nv-missing-XXXXXX)
cmake --build "${NVIDIA_BUILD_DIR:-build/nvidia}" --target \
  flagdnn_test_nvidia_adapter flagdnn_validation_nvidia_tensor_io \
  flagdnn_validation_cpu_reference
bash tools/performance_baseline/build_benchmark_nvidia_cudnn.sh "$nv_missing_dir/core"
bash tools/performance_baseline/build_benchmark_nvidia_missing_aux.sh "$nv_missing_dir/aux"

# 无 GPU 目录核对：只生成清单、目录元数据和文件指纹，不测量。
python3 tools/performance_baseline/run_benchmark_nvidia_missing.py \
  --core-binary "$nv_missing_dir/core" --aux-binary "$nv_missing_dir/aux" \
  --list-only --output-dir "$nv_missing_dir/catalog"

# GPU 空闲且独占后，串行采集准确的 371 条；输出目录必须尚不存在。
CUDA_VISIBLE_DEVICES=0 python3 tools/performance_baseline/run_benchmark_nvidia_missing.py \
  --core-binary "$nv_missing_dir/core" --aux-binary "$nv_missing_dir/aux" \
  --output-dir "$nv_missing_dir/measurement" \
  > "$nv_missing_dir/collection.log" 2>&1
```

两采集器均支持无 GPU 的 `--list`、JSONL `--catalog`，以及精确过滤 `--case NAME` / `--case-file PATH`。调度输出包含 `collection-manifest.json`、两组 `.cases` / `.catalog.jsonl`、测量 `.jsonl` 和 `.logs/`；失败阶段和原始日志保留，不自动修改正式基线。数值及完整写入检查包括计时前、计时后、重置输出后回放；BOOL 保留参考存储位宽，batchnorm 使用独立的 5/10/20 配置。入库前需逐项核对元数据、校验结果和计时配置；补测归档说明见[正式基线](../../docs/performance-baseline.json)的 `measurement_summaries.nvidia-missing-20261012`，不能将目录条数当作测量通过数。

向量和显存带宽工具的编译/运行参数见 [benchmark_nvidia_vector_memory.cu](benchmark_nvidia_vector_memory.cu) 文件开头。CPU 采样参考自测的编译命令见 [check_benchmark_nvidia_sampled_reference.cpp](check_benchmark_nvidia_sampled_reference.cpp)。

测量时保证 GPU 独占，所有 GPU 测试串行执行。上述命令输出原始记录，不自动覆盖正式 JSON；部分能力探测预期会失败，失败记录不能作为有效耗时。完整数值校验、抽样校验和 `sanity` 基础检查应分别看待。

异常恢复与 JSON 输出的 CPU 回归测试：

```bash
python3 -m unittest discover -s tools/performance_baseline/tests -v
```

Hygon 向量 FP32 与显存带宽测量（DTK，以下为实测 gfx936 构建命令）：

```bash
/opt/dtk/bin/hipcc -O3 -std=c++17 --offload-arch=gfx936 \
  tools/performance_baseline/benchmark_hygon_vector_memory.hip.cpp \
  -o /tmp/flagdnn_hygon_vector_memory
mkdir -p build/hygon/performance-baseline-20261010
/tmp/flagdnn_hygon_vector_memory --device 0 \
  > build/hygon/performance-baseline-20261010/vector-memory.csv \
  2> build/hygon/performance-baseline-20261010/vector-memory.log
```

默认扫描和计数口径与 NVIDIA 向量/带宽工具一致：8 条独立寄存器链、每轮 32 次展开的 FP32 FMA（每条 2 FLOP），内存按逻辑读写字节计数。配置峰值取 `record=summary` 的最高中位吞吐，不能取最快单样本；CSV 的 GB/s 除以 1000 后填写 JSON 的 TB/s。工具不自动修改正式基线，不测量的精度保留 `null`。首次实测全部配置摘要见[正式基线](../../docs/performance-baseline.json)的 `measurement_summaries.hygon-20261010`，div 的代理指标及限制见[基准说明](../../docs/equivalent-performance-baseline.md#峰值选择)。

Hygon 的等效性能回退现覆盖布局、索引及逐元素/激活共 40 个算子（identity 仅 FP8 E4M3/E5M2），平台行为见[结果与平台支持](../../docs/equivalent-performance-baseline.md#结果与平台支持)，具体指标与覆盖范围以正式基线为准。换算策略仅在 `platforms.hygon.scaling.case_policies.<op>` 中显式启用，均为有局限的近似模型，不是已验证的瓶颈结论；公共 NVIDIA `reference` 保留历史 2893 条有效记录及已有峰值、profile，可追加经过校验的新测量；共享 `scale_metric` 保持 `null`。其他平台只能读取自身 `scaling.case_policies`，即使峰值齐全，没有自己的算子策略仍须报错，不能回退到 Hygon 策略或修改公共参考。已有可用 hipDNN 对照保持原样；无对照用例输出带实际样本、流量修正及 NVIDIA 验证等级的 schema v4。默认只报告性能评价，不因低于 0.8 而失败；仅显式指定 `--min-speedup` 后才启用性能失败门槛。精度错误、运行异常或无效数据仍然失败。修改退出语义后必须重建对应 benchmark 目标，原始历史报告不重写。示例：

```bash
# CPU 基线契约名称沿用最初的 div，但已覆盖 40 算子的 1570 条实际目录用例。
cmake --build build/hygon --target flagdnn_benchmark_hygon_baseline_contract -j8
ctest --test-dir build/hygon --output-on-failure -R '^integration\.hygon\.div_performance_baseline$'
# 构建对应性能目标后，通过正式入口执行完整选中目录：
python3 tools/run_tests.py --platform hygon --ops reshape,transpose,slice,concatenate,gen_index \
  --suites benchmark --device 0 --output build/hygon/layout-index-equivalent-summary.json
```

## Hygon 原生对照的计时模式

当前 DTK/hipDNN 的 activation HIP Graph 路径会漏写部分非 256 整除形状的尾部。
`identity`、`abs`，以及 `relu`、`elu`、`sigmoid`、`swish`、`tanh`、`softplus`
的前向/反向原生对照因此统一采用 `hip_event_direct_batch`：FlagDNN 和 hipDNN
双方在同一 stream 上分别执行相同批次数，用 GPU event 计时；其余原生对照保持
`hip_graph_batch`。这是执行前明确选择的兼容策略，不会在任意精度失败后自动降级。
两种计时口径不同，direct 模式还包含 GPU 等待 CPU 提交的间隙，不能与历史 Graph
耗时当作同口径数据混比。schema v5 原生记录保存 `timing_mode`、`timing_reason`
和 `benchmark_config`；双方模式、原因及配置不一致时拒绝比较。等效性能仍使用
原来的 Graph 计时和 schema v4，NVIDIA 基准数据不变。

计时后的检查先验证最后一次采样，再填入经实际 dtype 编解码且必定超出原误差容限
的有限哨兵，重新执行整批并全量检查输出。这样既避免 `beta=0` 时 `0*NaN` 的误报，
又保留空执行/部分写入检测。日志分别标识 `last-timed-sample`、`fresh-capture`
和 `fresh-direct-batch`；coverage 报告区分配对已输出但套件失败、部分配对、未输出
和未运行，不因缺少 `--min-speedup` 而忽略精度或执行错误。

## Ascend 峰值采集

使用匹配的 `torch`、`torch_npu` 和 Ascend Triton，在已分配的空闲设备上运行：

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
python3 tools/performance_baseline/benchmark_ascend_vector_memory.py \
  --device 7 --operations hbm_copy --output-dir build/ascend/peak-measurement
```

工具拒绝覆盖已有 CSV/summary。默认扫描每缓冲区 256/1024 MiB、block 4096/8192/16384，
并检查缓冲区大于实际 L2。copy/triad/write 分别计 2/3/1 份逻辑字节流量，单位为十进制 TB/s。
每配置预热 20 次，Graph 内含 10 次执行，Graph 预热一次，再采样 5 轮各 20 次，
各配置先以全样本耗时中位数计算吞吐，再取最高值作为峰值；不是先计算各样本吞吐再取中位数。编译、初始化、拷贝、捕获和校验不计时。
内存校验覆盖首尾及 32 个分布窗口，同时检查输入与填充输出后的重新回放。
可选 FP32 FMA 扫描独立运行并检查全部输出，不能由 copy 结果推定向量峰值。

输出包含 `samples.csv`、`summary.json`、采集时源码快照和 `kernels/` 编译证据。
只把已测指标写入对应设备基线，未测项保留 `null`，算子换算策略另行显式登记。
2026-10-13 的 copy 测量为 6 个配置、600 个有效样本，峰值 **1.1794765684348663 TB/s**；
见[正式基线](../../docs/performance-baseline.json)的 `measurement_summaries.ascend-20261013`，其中保留配置明细及本地原始归档的路径与哈希。

### NVIDIA 补测记录与复现（2026-10-15）

原始补测执行 175 个不同用例和 15 次 INT32 compute 探测，共 190 次尝试：ELU backward 45 条、softplus backward 45 条通过；非浮点 concatenate 70 条、INT32 POW 15 条没有有效参考。90 条有效记录已入库，原有 3264 条记录未改变。

失败原因：INT32 concatenate 的 legacy 执行返回 NOT_SUPPORTED，BOOL/FP8 descriptor 返回 BAD_PARAM；原 dtype graph 均无可用引擎。14 条普通 INT32 POW 出现 16777217**1 得到 16777216 的精度错误，1 条 padded 布局构建失败，另测 15 条 INT32 compute 均无可用引擎。没有用其他 dtype 的时间替代。后续已从公共目录移除这 85 条，详见[公共用例调整及跨平台影响](../../docs/equivalent-performance-baseline.md)。

[测量摘要](measurements/nvidia-supplement-20261015.json)记录设备、软件、计数和 SHA256；原始压缩归档不再随源码保存。测量机的忽略目录 `build/performance-baseline/evidence/nvidia-supplement-20261015` 保留解压后的旧任务单、原始日志、样本及源码/二进制快照；该目录不随 Git 分发。原始归档 SHA256 留作来源标识，为 `a9cb3f224672be60a713e906f824610e99700a0aeb5de5bc6d6f9dadde94335d`。摘要中的跨平台范围和 18 条功能/6 条性能回归描述的是原始 NVIDIA 补测阶段，回归算子为 ELU backward 和 softplus backward；后续公共目录变更另见上述说明。

有效时间使用 H100 80GB HBM3 / 700 W、cuDNN 9.24.0。输入按实际 dtype 编解码后交给独立 CPU reference；计时前、计时后、重置输出回放后三阶段全量检查，每阶段 27,120 个输出元素，并检查输入、padding 和 guard。计时口径为 10 次预热、20 个样本、每批 Graph 50 次执行、1 次 Graph 预热，取 nearest-rank P50。构建、传输和校验不计时。原始自检覆盖 175 条/71,642 个元素，入库审计通过 18 项损坏证据拒绝检查。这些 NVIDIA 数据不代表目标平台已完成接入或达到性能门槛。

当前工具仅采集保留的 90 条激活反向用例。先重新构建当前 NVIDIA validation archives，再执行以下命令；CUDA/cuDNN 路径应指向本机已有安装，设备应为空闲且已预留的 H100。工具不会修改功耗设置或自动导入正式基线。

```bash
export CUDA_ROOT=/home/wangbingjie/cuda-local/usr/local/cuda-12.9
export CUDNN_ROOT=/home/wangbingjie/py312-flagdnn/lib/python3.12/site-packages/nvidia/cudnn
export CUDNN_FRONTEND_INCLUDE_DIR=/home/wangbingjie/py312-flagdnn/lib/python3.12/site-packages/include
bash tools/performance_baseline/build_benchmark_nvidia_supplement.sh build/nvidia/nv_supplement
CUDA_VISIBLE_DEVICES=0 python3 tools/performance_baseline/run_benchmark_nvidia_supplement.py --binary build/nvidia/nv_supplement --output build/performance-baseline/new-supplement
python3 tools/performance_baseline/audit_nvidia_supplement.py build/performance-baseline/new-supplement --measurement-id new-supplement
```

输出目录必须位于仓库内且为空或尚不存在，避免覆盖中断测量的证据。采集器保存源码和可执行二进制快照，并从该副本串行执行全部用例。审计核对目录参数与容差、三阶段 CPU 校验、样本、原始日志、源码和二进制哈希，只生成候选数据。新采集必须指定独立 measurement ID，不得复用原始 ID。

复现原始 175 条测量需要另行取得原始 collector、源码与证据；这些文件不再随源码分发。当前共享工厂不再生成退役用例。在测量机已有历史证据目录时，可运行：

```bash
python3 tools/performance_baseline/audit_nvidia_supplement.py --historical-20261015 build/performance-baseline/evidence/nvidia-supplement-20261015
```

当前审计保留历史模式，并新增容差及无效误差检查。测量时源码与采集后补充的自动归档工具分别保存在本地证据目录的 `source-snapshot.tar.gz` 和 `post-collection-tooling/` 中。

## 2026-10-19 Iluvatar 缺口补测

`nvidia-iluvatar-20261019` 新增 94 条经过三阶段全量 CPU 对照的 cuDNN 数据。
401 条请求中 92 条保留并补测，309 条确认不支持后删除共享用例；额外核心核查
保留 2 条、删除 6 条。删减理由和跨平台同步范围见
[共享目录说明](../../docs/equivalent-performance-baseline.md#2026-10-19-iluvatar-补测与共享目录)。

核心采集器现在包含 reduction/reshape/transpose/slice 的实际性能入口，
`--probe-native` 可跳过平台 capability 预判，直接尝试原生 cuDNN 构建与执行。
它不会恢复已从共享生成器删除的历史用例。辅助采集器的 `--all-shared` 扩展
混合输出归约和布局覆盖；不带该参数仍保持原有 80 条目录及输入规则。
两者支持 `--catalog` 和精确 `--case-file`，目录模式不初始化 GPU。
历史失败的完整原始记录和当时的源码/二进制保留在本地证据目录中。

新增数据的最终采集为 17 条核心用例、77 条辅助归约，全部成功。
正式 JSON 只追加有效记录；不支持的用例保留具体失败原因和
`catalog_status=removed_from_shared_cases`，不能作为有效性能耗时使用。
