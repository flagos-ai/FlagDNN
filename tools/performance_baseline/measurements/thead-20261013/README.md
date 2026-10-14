# Thead PPU 峰值采集（2026-10-13）

GPU 0：PPU-ZW810E，CUDA 兼容能力 `sm_80`，64 计算单元，64 MiB L2，102676561920 字节设备内存。PPU SDK `2.1.0-a5f865`，兼容 runtime/driver 版本均为 13000。`sm_80` 是兼容接口能力标识，不代表 NVIDIA 硬件。

仓库根目录复测（GPU 空闲且独占时）：

```bash
/usr/local/PPU_SDK/CUDA_SDK/bin/nvcc -O3 -std=c++17 -arch=sm_80 \
  tools/performance_baseline/benchmark_thead_vector_memory.cu \
  -o /tmp/flagdnn_thead_vector_memory
/tmp/flagdnn_thead_vector_memory --device 0 \
  > /tmp/thead-vector-memory.csv 2> /tmp/thead-vector-memory.log
```

原始记录无损保存在 [vector-memory.csv.gz](vector-memory.csv.gz) 和 [vector-memory.log](vector-memory.log)，全部配置摘要与 SHA-256 指纹见 [thead-20261013.json](../thead-20261013.json)。gzip 使用压缩级别 9、固定 `mtime=0`，不嵌入原文件名；`raw_csv_sha256` 对应归档文件，`raw_csv_uncompressed_sha256` 和 `raw_csv_uncompressed_bytes` 对应解压后的原 CSV。采集器编译成功；保留的 `build.log` 中弃用警告来自 SDK 头文件。

扫描 128/256 threads、2/4/8 blocks/计算单元，内存每个缓冲区 256/1024 MiB，均超过 L2。每配置预热 20 次，5 轮各 20 次独立事件计时，共 54 组、5400 样本。取每配置全部 100 样本的中位吞吐，再取配置间最大值；不能使用最快单样本。构建、传输、初始化、校验均不计时。

FP32 使用 8 条独立寄存器链、每轮 32 次展开，volatile FMA 和运行时系数防止工作被消除；每条标量 FMA 计 2 FLOP。每轮检查全部计算输出的精确结果。内存初始化包含完整随机化尾数位，copy/write/triad 每轮校验首、中、尾共 768 个 16 字节向量，read 校验横跨 grid 的 64 条完整 checksum。全部检查通过。带宽是逻辑读写量，不是内存控制器物理事务测量；read 的小 checksum 写入不计入 work，sink 字节数保留在每个配置中。

| 指标 | 本次最高中位吞吐 |
|---|---:|
| `vector_fp32` | 21.95791017 TFLOP/s |
| `hbm_copy` | 2.033910078 TB/s |
| `hbm_read` | 1.834828781 TB/s |
| `hbm_write` | 2.037074237 TB/s |
| `hbm_triad` | 2.101283396 TB/s |

其他精度/张量峰值没有测量，仍为 `null`。本采集只建立资源峰值，不自动为任何算子选择代理指标或证明瓶颈。


从仓库根目录校验归档、原始数据和确定性压缩；不调用 GPU：

```bash
python3 - <<'PY'
import csv, gzip, hashlib, io, json
from collections import Counter
from pathlib import Path

meta = json.loads(Path("tools/performance_baseline/measurements/thead-20261013.json").read_text())
archive = Path(meta["raw_csv"]).read_bytes()
assert hashlib.sha256(archive).hexdigest() == meta["raw_csv_sha256"]
raw = gzip.decompress(archive)
assert hashlib.sha256(raw).hexdigest() == meta["raw_csv_uncompressed_sha256"]
assert len(raw) == meta["raw_csv_uncompressed_bytes"]
rows = list(csv.DictReader(io.StringIO(raw.decode())))
assert Counter(row["record"] for row in rows) == {"sample": 5400, "round": 270, "summary": 54}
assert all(row["validation"] == "pass" for row in rows)
repacked = io.BytesIO()
with gzip.GzipFile(fileobj=repacked, mode="wb", filename="", mtime=0, compresslevel=9) as stream:
    stream.write(raw)
assert repacked.getvalue() == archive
print("PASS: archive, original CSV, 5400 samples and deterministic gzip verified")
PY
```
