# FlagDNN 阿里 T-Head 真武 PPU 适配建议

> 本文面向 T-Head SAIL/PPU SDK。建议借鉴 `backends/nvidia` 的职责划分和
> `backends/iluvatar` 已验证的闭环，但所有 PPU SDK、compiler、kernel、runtime、
> reference 和测试代码都应留在 `backends/thead`；不得修改或包含其他平台专用源码。

## 1. 首要前置条件：先打通 SAIL 版 `libtriton_jit`

T-Head 官方 `triton-for-sail` 面向 PPU0010/PPU0015，编译链为
`ttir -> ttgir -> llir -> hgbin`。当前 `/home/wbj/libtriton_jit` 的 backend 列表没有
SAIL/PPU，因此 FlagDNN 适配不能直接开始于复制 NVIDIA launcher，也不能因为 SAIL
支持 CUDA 语言/API 就把 PPU 当成 `BACKEND=CUDA`。

建议先在 `libtriton_jit` 项目独立完成并安装 PPU backend，至少证明：

1. 能识别 `triton-for-sail` 的 target、编译选项、entry metadata 和 `hgbin`；
2. 能用 HGGC/PPU Driver API 装载 module、取得 function、传入原生 stream 并 launch；
3. 外部 C++ consumer 可在非默认 stream 上调用最小 kernel，且错误码、资源和 module
   生命周期正确；
4. 安装包导出的 CMake target、头文件、共享库和 Python/Triton 环境身份完整。

这四项通过后，FlagDNN 的 `engines/libtriton_jit.cpp` 才应接入该已安装 backend。若需要
修改 `libtriton_jit`，应在它自己的仓库完成，不在 FlagDNN 中维护私有补丁副本。

## 2. Backend 与 SDK 边界

- backend 名称固定为 `thead`，使用
  `flagdnnCreateHandleWithBackendName(..., "thead", ...)`；按现有 loader 契约使用 ABI
  v2 和 `libflagdnn_backend_thead.so.2`，无需修改公共枚举或 loader。
- SDK 发现全部放在本目录 `CMakeLists.txt`/`cmake/`。配置阶段确认 PPU/SAIL SDK 根、
  HGGC runtime/driver、`hgcc`/HGRTC、`triton-for-sail`、已安装 PPU
  `libtriton_jit`、Python，以及 validation 用的 acDNN/acBLAS。
- 不以头文件函数名或“CUDA compatible”为 ABI 证据。原生 stream、event、module、Graph
  类型与调用约定必须用当前 SDK 的 C/C++ contract 实机验证；production plugin 不得
  意外链接 NVIDIA `libcuda`/`libcudnn`。
- acDNN/acBLAS 只属于 validation，不进入安装的 production plugin；公共构建脚本和
  其他 backend 不添加 PPU SDK 探测。

## 3. 架构、compiler 与 kernel 策略

- target fingerprint 必须至少区分 PPU0010/`ppu_10` 与
  PPU0015/`ppu_15`，并纳入实际设备、SDK/driver 版本；两代架构不能共享 artifact 或
  autotune winner。
- `compiler.py` 负责把公共 Graph IR 变为 PPU Triton plan。compiler identity 应覆盖
  `triton-for-sail`、Python、SDK、`libtriton_jit`、kernel/registry/tuning 文件及所有
  影响 codegen 的环境变量。
- 先跑 common kernel。只有公共 Triton 语义在 PPU 上无法编译，或有稳定可复现收益时，
  才在 `backends/thead/kernels` 增加 override；平台 registry 登记后失败必须硬失败，
  不得隐式退回 common。
- PPU 的 `tl.aiu_load` 是适合矩阵类 kernel 的平台优化点，但有指针对齐、二维连续、
  tile 和 K 字节倍数约束，而且 PPU0010 与 PPU0015 的低精度能力不同。应把它放在带
  capability guard 的 PPU 私有 candidate 中，不能为使用 AIU 去修改 common kernel。
- tuning 必须按架构分层；不要照搬 NVIDIA 的 warp、shared-memory、stage、tile 参数。
  candidate prepare 只可过滤明确的资源/架构不兼容，编译协议、ABI、hash、非法访问等
  错误必须终止 build。

## 4. Reference 选择要先消除“cuDNN 兼容”歧义

SAIL SDK 提供 acDNN，并在部分版本提供独立 cuDNN 兼容包，但这不代表 NVIDIA cuDNN
Frontend Graph、所有 classic primitive、SONAME 或语义都完整等价。编码前应冻结一种
明确政策：

- DNN primitive 优先使用当前 SDK 文档支持的原生 acDNN C/C++ 接口；matmul 如获批准可
  使用独立 acBLAS provider；
- 若选择 cuDNN 兼容包，必须探测实际 header/runtime 版本、SONAME、导出 symbol 和每个
  primitive 的行为，不能只检查文件名叫 `cudnn`；
- 不允许在同一结果中悄悄混用 acDNN、兼容 cuDNN、PyTorch 和 CPU oracle。每条结果都要
  带明确 provider identity。

只有 dtype、layout/stride、broadcast、NaN、rounding、alpha/beta、padding/group、
accumulation 等语义精确一致的 case 才能比较。无法精确表达时结构化 `SKIP`，同时仍要
执行 FlagDNN production graph；记录 op、case、reason、reference 头/运行时版本、SAIL
SDK、PPU target、dtype、layout 和 shape。禁止用若干 reference 算子近似拼接后宣称一个
fused Graph 已获等价验证。

## 5. Runtime Graph、autotune 与性能

- 区分 FlagDNN Frontend Graph 和 HGGC runtime Graph。Graph capture 只包围已完成 JIT、
  module load、workspace 确定和 winner 选择后的 `execute()`；capture 中不得编译、分配、
  autotune 或同步。
- `create_executable()` 可准备 candidate、在私有 stream/event 上调优并固化 winner；
  `execute()` 必须尊重调用者 stream，不得退回默认 stream。测试非默认 stream、跨 stream
  依赖、重复 capture/replay 和 destroy-before-completion 的生命周期约束。
- 使用 PPU/HGGC device event，在相同 stream 和相同输入上交替测量 FlagDNN 与
  reference；计时窗口不包含编译、首次 cache、数据初始化或 reference plan build。GPU/
  PPU 测试统一串行运行。
- 目标仍建议为 `speedup = reference_median_us / flagdnn_median_us >= 0.9`。优先处理
  pointwise/layout/reduction，再处理 normalization、matmul/convolution，最后处理
  attention/composite；PPU0010 和 PPU0015 分别统计，不能用一代架构的平均值掩盖另一代
  回归。

## 6. 必须建立的 contract

除每算子 functional/benchmark 外，至少建立以下平台 contract：

- SDK 单根目录、生产/reference 依赖隔离及 plugin 导出/SONAME；
- compiler identity、artifact hash/schema/stage DAG/metadata；
- PPU context、target fingerprint、非默认 stream 与错误传播；
- installed `libtriton_jit` consumer、最小 JIT kernel、autotune cache 失效；
- HGGC Graph capture/replay；
- installed FlagDNN consumer；
- catalog closure、结构化 SKIP、benchmark JSONL 和 `run_tests` adapter。

`validation/run_tests_adapter.py` 只承担 T-Head 的设备可见性、preflight、SKIP 和结果
校验。不要把 PPU 分支写进 `tools/run_tests.py`；若确需通用扩展点，必须有 contract
证明其他已支持平台行为不变。

## 7. 推荐交付顺序

1. 在外部 `libtriton_jit` 完成 SAIL/PPU backend 并通过 C++ smoke；
2. SDK resolver、ABI plugin、context/stream、target fingerprint、安装后 consumer；
3. compiler identity、单个 common Add kernel、artifact/JIT；
4. HGGC Graph、autotune、cache contract；
5. acDNN/acBLAS exact reference、功能测试、性能测试、结构化 SKIP；
6. 扩展全算子并最终运行：

```bash
tools/build.sh --backends thead --build-dir build/thead
tools/install.sh --build-dir build/thead
python3 tools/run_tests.py \
  --build-dir build/thead \
  --platform thead \
  --ops all \
  --suites functional,benchmark
```

不要在 SAIL 版 `libtriton_jit` 尚未通过独立 smoke 时批量复制算子文件；否则后续很难
区分是 Graph/compiler/artifact 问题，还是 PPU loader/launch 本身的问题。

## 8. 参考资料（核对日期：2026-08-24）

- [T-Head `triton-for-sail`](https://github.com/t-head/triton-for-sail)
- [T-Head HGGC samples](https://github.com/t-head/hggc-samples)
- [T-Head ACTLIZE for PPU](https://github.com/t-head/actlize)
- [真武 PPU SAIL SDK v2.1 release note](https://help.aliyun.com/zh/document_detail/3030339.html)
- [FlagOS `libtriton_jit`](https://github.com/flagos-ai/libtriton_jit)
