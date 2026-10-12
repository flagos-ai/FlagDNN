# hipDNN 接口组合独立验证

这些探针直接调用已安装的 hipDNN，不经过 FlagDNN 适配器的 dtype 白名单。
用途是区分接口拒绝、数值错误、输出漏写和 Graph 兼容问题，不能将其输出作为性能基线。
探针不修改 `docs/performance-baseline.json`，也不会把失败结果改成通过。

## 构建与运行

先构建仓库的 Hygon 测试与 benchmark 依赖，再在独立目录构建探针：

```bash
probe_dir=$(mktemp -d /tmp/flagdnn-hipdnn-probes-XXXXXX)
bash tools/hipdnn_capability/build_probes.sh "$probe_dir" build/hygon
"$probe_dir/pointwise_probe_contract"
"$probe_dir/pointwise_probe" --list
"$probe_dir/batchnorm_probe" --list
"$probe_dir/reduction_probe" --list
```

构建及 `--list` 不初始化 GPU。SDK 不在 `/opt/dtk` 时设置 `DTK_ROOT`、
`HIPDNN_ROOT`；可通过 `CXX` 指定 C++ 编译器。每个 GPU 用例应独立启动进程，
防止前一个 HIP 错误污染后续结果。不要并发占用同一张 GPU，也不要覆盖历史日志。

```bash
# 逐元素：直接调用、重新填充输出后重跑、Graph 与重新填充后的回放。
"$probe_dir/pointwise_probe" --device 0 \
  --case add_perf_bfloat16_1x1x1024_by_1x1x1024 --path both
# identity 的另一种原生 API 候选，不改变原始位模式要求。
"$probe_dir/pointwise_probe" --device 0 \
  --case identity_int32_2x3x4 --path both --identity-api transform
# batchnorm 保持原目录的五个输入和五个输出。
"$probe_dir/batchnorm_probe" --device 0 \
  --case batchnorm_perf_fp16_1x128x8x8
# Ex 的 BN-only 候选，同样检查全部五个输出。
"$probe_dir/batchnorm_probe" --device 0 \
  --case batchnorm_perf_fp16_1x128x8x8 --api ex
# 混合输出 reduction：默认每批 50 次，检查全部输出与缓冲区边界。
"$probe_dir/reduction_probe" --device 0 \
  --case reduction_sum_fp16_to_fp32_3_7 --iterations 50
```

## 校验边界

- 用例来自共享目录，保留 dtype、shape、strides、绑定字节偏移和属性。
- 普通数值用例使用实际量化输入的 CPU 全量参考与原误差容限；identity 按原始字节比较，INT32 不经过 FLOAT 中转。
- 直接执行结果不能替代 Graph 校验。重新填入有限、可区分的输出哨兵后再运行，检测空执行、尾部漏写和输入或 padding 被改动。
- `pointwise_probe` 的 `--path direct` 用于区分 API 支持和 Graph 支持；direct 通过不自动证明 Graph 通过。
- batchnorm 的 `--fp32-parameters` 是诊断变体，元数据标记为不等同原目录，不能据此放行低精度参数用例。`--transform-parameters` 则每次执行都调用 hipDNN 转换，转换也进入 Graph；转换失败即失败。
- reduction 的 `--rank2-boundary-view` 只尝试连续存储首轴或末轴的无拷贝二维视图。原输入数据、输出类型、绑定偏移和 CPU 参考不变；中间轴及不连续布局拒绝改写。`--descriptor-self-test` 是 CPU 等价性检查。

结果中的 API 名称、返回码、失败阶段和数值检查结果应一起保留。`NOT_SUPPORTED`、
`EXECUTION_FAILED`、错误结果和超时是不同结论；不得将数值或执行错误笼统归为“不支持”。
只有完成相应支持矩阵及正式测试回归后，才应修改生产适配范围。原生不可用且缺少 NVIDIA
有效记录时，按 [补测清单](../../docs/nvidia-performance-missing-cases-20261012.md) 补齐数据，
不能借用其他 dtype 或 shape 的时间。
