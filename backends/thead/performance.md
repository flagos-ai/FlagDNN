# THead speedup ≥ 0.9 优化记录

验收口径：每个已认证可比较 case 的 `acdnn_median_us / flagdnn_median_us ≥ 0.9`，
并通过同输入正确性检查和完整 case accounting。没有 acDNN reference 的算子不计为达标。
FlagDNN 计算 kernel 仅使用 Triton。

## 修改前统计

输入为 `build/thead/thead-run-tests-audit.json`，记录时间为
`2026-09-07T04:21:45.857265+00:00`，SHA256 为
`c5e5f41299dfe4557bea7d04f90207822c95c33d4401bd134c4bfb6d00a34654`。
这份历史记录未设置性能阈值；重新按 0.9 统计后：

- 51 个算子形成 1125 个可比较 case，其中 459 个 case、42 个算子未达标。
- 另外 6 个 benchmark 算子 reference 不可用，4 个 SDPA family 算子没有 benchmark。
- 最小 speedup 为 0.017919，对应 FP16 YOLO-X P5 ConvDgrad。
- ConvDgrad、ConvWgrad、ConvFprop 分别有 37/45、39/45、32/51 个 case 未达标。

生成全部 61 个算子和逐 case 的 Markdown、CSV、JSON 统计：

```bash
PYTHONDONTWRITEBYTECODE=1 /usr/local/bin/python3 \
  backends/thead/validation/performance_report.py \
  --input build/thead/thead-run-tests-audit.json \
  --output-dir build/thead/performance-baseline --min-speedup 0.9
```

## 当前修改

`compiler.py` 识别输入输出具有相同物理布局的连续张量，包括 NHWC 和任意连续轴排列，
派发已有的 Triton linear kernel。存在 padding 或不同物理布局的张量继续使用 strided
kernel。singleton 维度的 stride 不影响物理地址，因此不阻止 linear kernel 的使用。
AddSquare 的虚拟中间结果不参与外部存储布局判定。

Add、Unary、AddSquare 在元素数至少 65536 时使用 1024 元素分块，其余保持 256。
编译器 identity 自动包含该源码变化，不会复用旧派发的 artifact。
新 compiler contract 覆盖连续 NHWC、padding、混合布局、singleton 和大尺寸分块；
原 AddSquare strided contract 改用真正带行间 padding 的张量。

`engines/libtriton_jit.cpp` 去除 launch 参数数组的全量清零：4096 个参数槽位及其指针
原来每次清零约 96 KiB，现在只写入 kernel ABI 实际使用的参数槽位。两个 scratch 参数
仍明确初始化为零；未使用槽位不会被读取。固定栈存储、并发调用隔离和 Graph capture
行为保持不变，计算仍由原 Triton kernel 完成。

已验证全量 compiler contract、扩展布局 contract、Release 构建及 Graph capture 静态
contract 通过。Abs、AddSquare、ReLU 真机功能回归共 72/72 个 case 通过；
`integration.thead.graph`、`integration.thead.runtime` 均通过，包含真实 capture/replay
及运行时错误检查。日志保存在 `build/thead/performance-baseline/`。

## 测量条件与未完成项

修改前在真机复测：FP16 ReLU `[1,1,1024]` 为 3.1168 µs / acDNN 1.5648 µs，
speedup 0.5021；FP32 NHWC ReLU `[8,16,64,128]` 为 13.6656 µs / acDNN
2.6368 µs，speedup 0.1930。

修改后该大 ReLU 的同输入正确性通过，但性能样本存在约 48 µs 到 3.8 µs 的跳变，
同时 acDNN 从约 30.7 µs 跳到 2.7 µs。`ppu-smi` 在本次 benchmark 已退出时仍显示
两张 PPU 利用率 100%，有其他进程在运行。因此当前不能据此发布优化后稳定 speedup，
也没有完成全算子达标验收。需要可独占、无其他计算任务的设备时段才能继续调优和验收。

加入 launch 参数初始化优化后，代表性复测结果如下。表内是完整原始样本的中位数，
未剔除干扰样本；两侧 p90 仍有明显尖峰，因此这些结果不能替代稳定环境的全量验收。

| ReLU case | 修改前 FlagDNN µs | 修改后 FlagDNN µs | 修改后 acDNN µs | 修改前 speedup | 修改后 speedup |
|---|---:|---:|---:|---:|---:|
| FP16 `[1,1,1024]` | 3.1168 | 1.5616 | 1.6120 | 0.5021 | 1.0323 |
| FP32 NHWC `[8,16,64,128]` | 13.6656 | 3.7184 | 2.7464 | 0.1930 | 0.7386 |

对应日志为 `relu-small-after.log`、`relu-large-after.log`。大 ReLU 的 4096 元素分块
实验为 3.9696 µs、speedup 0.6697，未采用；代码保留 1024 分块。

后续仍需完成短算子启动开销分析、MatMul/卷积/BatchNorm 等未达标 case 的 Triton
调优，并在稳定设备环境按以下命令重新生成完整结果：

```bash
PYTHONDONTWRITEBYTECODE=1 /usr/local/bin/python3 tools/run_tests.py \
  --platform thead --build-dir build/thead --ops all --suites all \
  --no-preflight --min-speedup 0.9 \
  --output build/thead/thead-performance-final.json
```

`--no-preflight` 沿用现有 THead 适配文档中的公共 runner contract 延期决定，
不跳过逐算子的正确性检查、性能采样或 speedup 门禁。
