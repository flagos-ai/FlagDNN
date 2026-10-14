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
