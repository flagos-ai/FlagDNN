# FlagDNN 摩尔线程 MUSA 适配建议

> 本文是适配前的重点检查清单，不是算子支持清单。建议以
> `backends/nvidia` 的职责划分为参考，以 `backends/iluvatar`、
> `backends/hygon` 已验证的依赖隔离和测试闭环为经验，但不要复用或修改这些
> 平台的专用源码。

## 1. 先冻结平台契约

- backend 名称固定为 `mthreads`，通过
  `flagdnnCreateHandleWithBackendName(..., "mthreads", ...)` 选择；当前无需为它修改
  公共枚举或 `src/backend_loader.cpp`。
- 使用 backend ABI v2，生成并安装
  `libflagdnn_backend_mthreads.so.2`；`execute()` 只能在调用者传入的原生 MUSA
  stream 上排队，不能编译、autotune、分配持久资源或做全设备同步。
- 在编码前记录一套经实机确认的版本矩阵：MUSA Driver、MUSA Toolkit、muDNN、
  `torch_musa`、Triton、Python、`libtriton_jit`、目标架构及设备型号。不要把某一代
  卡的架构、warp size 或资源上限写死为全平台事实。

## 2. 依赖发现必须由 `backends/mthreads` 独占

- 在本目录的 `CMakeLists.txt` 和 `cmake/` 中发现 SDK；公共
  `tools/build.sh`、`tools/install.sh`、根 CMake 及其他 backend 不负责寻找 MUSA。
- 至少分别确认原生 driver/runtime 头和库（通常包括 `musa.h`、
  `musa_runtime.h`、`libmusa`、`libmusart`），不要因为 API 与 CUDA 相似而链接系统
  CUDA 库或包含 CUDA 头。
- production plugin 只链接运行所需的 MUSA driver/runtime、MUSA 版
  `libtriton_jit` 及其明确依赖。muDNN、muBLAS 等 reference 库只能被
  `validation/` target 使用，不能进入安装的 backend 依赖闭包。
- 配置阶段输出所有已选择资源的规范化 realpath、SONAME/版本和 Python
  executable，并拒绝来自不同 SDK 根目录或不同 Python 环境的混搭。特别防止 pip
  解析出公共 CUDA PyTorch/Triton，替换 MUSA 配套 wheel。

## 3. 优先复用现有 MUSA `libtriton_jit`，但先做四项门禁

当前 `/home/wbj/libtriton_jit` 已提供 `BACKEND=MUSA`，其 Triton backend 名称为
`mtgpu`。因此不建议在 FlagDNN 中再造一套 launcher；应先独立证明：

1. 安装后的 CMake package、头文件和共享库可被干净的外部 C++ consumer 找到；
2. `musaStream_t`、MUSA Driver stream 与 JIT launcher 的 stream 传递方式完全兼容；
3. 最小 Triton kernel 能完成 Python 编译、artifact 装载、C++ 调用及非默认 stream
   执行；
4. module/function/metadata 的生命周期、错误码和卸载行为在重复 build/execute/destroy
   后无泄漏。

任何一项失败都应先在 `libtriton_jit` 的 MUSA backend 修复；不要在 FlagDNN
通用层增加 MUSA 特判。`torchada` 适合应用迁移，但 production plugin 应使用原生
MUSA API，不能依赖运行时把 `torch.cuda` 或 CUDA symbol 动态改写为 MUSA。

## 4. Compiler、artifact 与 cache identity

- 在 `compiler.py` 中消费公共 Graph IR，平台 capability 在这里判断；先尝试
  common registry，仅在有明确性能收益或语义差异时增加 MUSA 私有 kernel override。
- 平台 registry 一旦登记某算子就拥有解析权。私有 kernel 编译或加载失败必须明确
  失败，不能静默回退 common kernel。
- target fingerprint 至少区分实际 MUSA 架构和设备；compiler/measurement identity
  还应覆盖 SDK/driver、Python、`torch_musa`、Triton、`libtriton_jit`、kernel source、
  registry 和 tuning YAML。环境变化后不能复用旧 artifact 或 autotune winner。
- artifact parser 必须校验 schema、文件 hash、入口、参数顺序、stage DAG、launch
  metadata、workspace 和 candidate ID；不能信任 compiler 输出，也不能按目录中“第一个
  文件”猜测二进制。

## 5. Reference 与 SKIP 纪律

- 建议以原生 muDNN C/C++ primitive 作为 DNN 算子的正确性和性能 reference；若项目
  批准对 matmul 使用 muBLAS，应使用独立 provider 并在结果中明确 provider identity，
  不能混成一个含糊的“vendor reference”。不要用 PyTorch/`torch_musa` 作最终 reference，
  它可能发生 fusion、fallback 或选择不同底层库。
- 仅在 dtype、layout、stride、broadcast、NaN、rounding、alpha/beta、padding、group、
  accumulation 等语义可精确表达时比较。无法精确表达的 case 必须结构化 `SKIP`，禁止
  用多个算子拼出近似 reference、偷偷搬到 CPU 或放宽误差掩盖语义差异。
- reference `SKIP` 不等于不运行 FlagDNN：仍应构建并执行 production graph，结果中记录
  op、case、原因、muDNN 头/运行时版本、MUSA SDK、target、dtype、layout 和 shape。
- 在 `validation/` 内集中实现 buffer、stream、event、tensor 编解码和 padding guard，
  functional 与 benchmark 共享同一套底层设施，避免两套布局转换产生假通过。

## 6. Graph、autotune 与性能

- 明确区分 FlagDNN Frontend Graph 与 MUSA runtime graph capture。Graph 测试必须捕获
  已完成编译和候选选择的 executable；capture 期间不得首次 JIT、分配、autotune 或
  同步。
- autotune 只能在 `create_executable()` 阶段进行。先准备全部候选，只过滤有明确类型的
  架构/资源不兼容错误；Python、协议、hash、ABI、非法内存访问等错误必须硬失败。
- 使用 MUSA device event 在同一 stream 上计时；依赖 stage 在计时窗口外执行，FlagDNN
  与 reference 交替采样。不得用 host wall clock，也不得并行运行同卡 benchmark。
- 全量性能目标建议保持 `speedup = reference_median_us / flagdnn_median_us >= 0.9`。
  优化顺序可按 pointwise/layout、reduction、normalization、matmul/convolution、attention/
  composite 由易到难；先修 launch、布局和错误 tuning，再增加私有 kernel。

## 7. 最小交付顺序

1. SDK/CMake、ABI plugin、context/stream、target fingerprint；
2. compiler identity、单个 common Add kernel、artifact/JIT/安装后 consumer；
3. runtime graph capture、autotune 与 cache 失效 contract；
4. muDNN reference、功能 suite、benchmark suite、结构化 SKIP；
5. 扩展全算子，最后运行：

```bash
tools/build.sh --backends mthreads --build-dir build/mthreads
tools/install.sh --build-dir build/mthreads
python3 tools/run_tests.py \
  --build-dir build/mthreads \
  --platform mthreads \
  --ops all \
  --suites functional,benchmark
```

在 `validation/run_tests_adapter.py` 中只放 MUSA 的设备可见性、preflight、SKIP 和结果
校验策略。不要把这些规则写入 `tools/run_tests.py`；若确有新的通用扩展点，必须先证明
它对 NVIDIA、Hygon、Ascend、Iluvatar 的默认行为无变化。

## 8. 参考资料（核对日期：2026-08-24）

- [Moore Threads `torch_musa`](https://github.com/MooreThreads/torch_musa)
- [Moore Threads `vllm-musa`](https://github.com/MooreThreads/vllm-musa)
- [FlagOS `libtriton_jit`](https://github.com/flagos-ai/libtriton_jit)

