# 跨平台性能基准工具

这里集中保存峰值和 cuDNN 参考耗时的测量工具。数据格式、换算公式和校验范围见 [基准说明](../../docs/nvidia-peak-baseline-20260920.md)，正式数据保存在 [performance-baseline.json](../../docs/performance-baseline.json)。

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

默认扫描和计数口径与 NVIDIA 向量/带宽工具一致：8 条独立寄存器链、每轮 32 次展开的 FP32 FMA（每条 2 FLOP），内存按逻辑读写字节计数。配置峰值取 `record=summary` 的最高中位吞吐，不能取最快单样本；CSV 的 GB/s 除以 1000 后填写 JSON 的 TB/s。工具不自动修改正式基线，不测量的精度保留 `null`。首次实测全部配置摘要见 [hygon-20261010.json](measurements/hygon-20261010.json)，div 的代理指标及限制见[基准说明](../../docs/nvidia-peak-baseline-20260920.md#hygon-div-等效性能2026-10-10)。

Hygon 的等效性能回退现覆盖布局、索引及逐元素/激活共 40 个算子（identity 仅 FP8 E4M3/E5M2），具体指标和缺参考目录见[覆盖说明](../../docs/nvidia-peak-baseline-20260920.md#hygon-缺失性能测试的等效补齐)。换算策略仅在 `platforms.hygon.scaling.case_policies.<op>` 中显式启用，均为有局限的近似模型，不是已验证的瓶颈结论；公共 NVIDIA `reference` 保持原始内容，2893 条记录的 `scale_metric` 均为 `null`。其他平台只能读取自身 `scaling.case_policies`，即使峰值齐全，没有自己的算子策略仍须报错，不能回退到 Hygon 策略或修改公共参考。已有可用 hipDNN 对照保持原样；无对照用例输出带实际样本、流量修正及 NVIDIA 验证等级的 schema v4。默认只报告性能评价，不因低于 0.8 而失败；仅显式指定 `--min-speedup` 后才启用性能失败门槛。精度错误、运行异常或无效数据仍然失败。修改退出语义后必须重建对应 benchmark 目标，原始历史报告不重写。示例：

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
