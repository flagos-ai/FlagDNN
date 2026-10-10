# NVIDIA 性能基准工具

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
