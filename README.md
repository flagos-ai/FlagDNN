# FlagDNN

FlagDNN 是一个面向多种加速器的 Native C/C++ DNN Graph Runtime。对外接口按照
cuDNN Frontend 的 Graph 风格设计，算子 kernel 使用 Triton 实现；平台可以提供专用
kernel 覆盖通用实现。当前完整实现的平台是 NVIDIA，执行引擎默认使用
`libtriton_jit`。

## 项目目标

- 提供稳定的 C ABI 和 cuDNN-Frontend-style C++ Graph API。
- Graph 构建、lowering、backend 选择和执行生命周期全部由 C++ 管理。
- 通用 Triton kernel 可被不同平台复用。
- NVIDIA、Ascend、Iluvatar 等平台可以独立提供 compiler、kernel、tuning 和 runtime。
- 在 `libtriton_jit` 原生支持 autotune 前，由 FlagDNN 提供平台无关的调优策略和
  平台计时适配。

FlagDNN 不再提供 Python eager API。仓库中的 Python 仅用于 Triton kernel 与外部
codegen；外部 codegen 在独立进程执行。当前 `libtriton_jit` 本身仍依赖嵌入式
Python，因此选择该执行引擎时 NVIDIA backend 会加载 Python runtime，但不会引入
FlagDNN Python eager API 或 Torch。

## 目录结构

```text
FlagDNN/
├── include/                 # 唯一公开的 C/C++ API
├── src/                     # 平台无关的 Graph、lowering 和 runtime
├── compiler/                # 独立外部 Triton 编译器
├── kernels/common/          # 跨平台通用 Triton kernel
├── backends/
│   ├── autotune_policy.*    # 跨平台 autotune 选择策略
│   ├── nvidia/              # NVIDIA 生产实现及唯一 validation/ 接入点
│   ├── ascend/              # Ascend 平台 registry/后续实现位置
│   └── iluvatar/            # Iluvatar 平台 registry/后续实现位置
├── tests/                   # C++ Graph 功能测试
├── benchmark/               # C++ Graph 性能测试
├── cmake/                   # 构建、安装、平台和测试注册
├── docs/                    # 架构、算子开发和测试文档
└── tools/                   # Native 测试与环境辅助工具
```

每个概念只有一个事实来源：

- 公开算子清单：`cmake/Operators.cmake`
- 通用 kernel registry：`kernels/registry.json`
- 平台 kernel registry：`backends/<platform>/kernels/registry.json`
- NVIDIA tuning：`backends/nvidia/tuning/common.yaml`
- 平台功能/性能适配：`backends/<platform>/validation/`
- C++ API：`include/flagdnn/` 与 `include/flagdnn_frontend.h`

历史拼写 `flagnn_frontend.h`/`flagnn_frontend` 仅作为源码兼容别名保留；新代码统一使用
`flagdnn_frontend.h`/`flagdnn_frontend`。

## 算子执行链

以 Add 为例：

```text
flagdnn_frontend::graph::Graph::pointwise(ADD)
  -> Graph::build()
  -> src/graph/lowering/pointwise.cpp
  -> external compiler request
  -> platform kernel registry
       NVIDIA 有 Add override: backends/nvidia/kernels/
       否则: kernels/common/
  -> tuning candidate expansion
  -> FlagDNN autotune candidate selection
  -> libtriton_jit compile/load
  -> Triton kernel launch on the caller stream
```

平台 registry 一旦登记某个算子，该平台实现就拥有该算子的解析权；平台 kernel 的编译
失败不会静默回退到 common。只有平台 registry 没有登记该算子时才选择通用 kernel。

## 构建

基础要求：

- CMake 3.23+
- 支持 C++20 的编译器
- Python 3（包含 Triton/FlagTree 与 PyYAML，用于外部 codegen）
- NVIDIA 构建需要 CUDA Driver、CUDA Toolkit 和 CUDA 版 `libtriton_jit`
- 功能测试和性能测试需要 cuDNN Frontend

推荐直接使用平台无关的仓库脚本。构建目录默认根据 backend 生成
`build/<backend>`，构建类型为 `Release`，并启用功能测试和性能测试。仓库当前默认
backend 是 NVIDIA，因此无参数时仍生成 `build/nvidia` 并使用 `libtriton_jit`：

```bash
source /path/to/python-env/bin/activate
tools/build.sh
```

公共脚本不包含 CUDA、cuDNN、CANN 或其他平台 SDK 的发现逻辑。每个平台在
`backends/<platform>/` 内独立发现自己的 compiler、runtime 和 validation 依赖。
例如 NVIDIA backend 会从所选 Python 环境发现 Triton、PyYAML、Torch、cuDNN
Frontend，并尝试发现仓库同级的 `libtriton_jit/build`。非标准 CMake 依赖位置可以
在 `--` 后传递，例如：

```bash
tools/build.sh \
  --python /path/to/python \
  --build-dir /tmp/flagdnn-build-nvidia \
  -- -DTritonJIT_DIR=/path/to/libtriton_jit/build
```

使用 `tools/build.sh --help` 查看全部选项。只需要生产库时可增加
`--no-tests --no-benchmarks`。

其他平台完成 `backends/<platform>/CMakeLists.txt` 和 validation 接入后使用同一个
脚本，例如：

```bash
tools/build.sh --backends ascend --build-dir build/ascend
tools/install.sh --build-dir build/ascend
```

仅构建平台无关核心：

```bash
cmake -S . -B /tmp/flagdnn-build \
  -DFLAGDNN_BACKENDS= \
  -DFLAGDNN_BUILD_TESTS=ON
cmake --build /tmp/flagdnn-build -j
ctest --test-dir /tmp/flagdnn-build --output-on-failure
```

构建 NVIDIA backend、功能测试和 benchmark：

```bash
cmake -S . -B /tmp/flagdnn-build-nvidia \
  -DFLAGDNN_BACKENDS=nvidia \
  -DFLAGDNN_BUILD_TESTS=ON \
  -DFLAGDNN_BUILD_BENCHMARKS=ON \
  -DFLAGDNN_EXECUTION_ENGINE=libtriton_jit \
  -DFLAGDNN_CODEGEN_PYTHON=/path/to/python
cmake --build /tmp/flagdnn-build-nvidia -j
```

将编译结果安装成完整 SDK：

```bash
tools/install.sh
```

安装位置默认为所选构建目录下的 `install`；默认 NVIDIA 构建对应
`build/nvidia/install`。指定其他位置：

```bash
tools/install.sh --prefix /path/to/flagdnn-sdk
```

安装目录中的 `lib/`、`include/`、`lib/cmake/FlagDNN/` 和 `share/flagdnn/` 共同构成
完整 SDK；不能只复制共享库而丢弃 compiler、kernel、registry 和 tuning 资源。

安装后的 runtime 会从库文件相对位置查找随 SDK 安装的 codegen/kernel 资源，并默认
从 `PATH` 解析 `python3`。部署环境可以通过 `FLAGDNN_COMPILER_EXECUTABLE`、
`FLAGDNN_CODEGEN_COMPILER` 或 `Handle::set_compiler()` 显式覆盖。
外部编译器默认超时为 1800 秒，可通过 `FLAGDNN_COMPILER_TIMEOUT_SECONDS` 设置为
1–86400 秒。compiler/kernel/tuning、Python/Triton/PyYAML、ptxas 或目标设备身份变化时，
对应编译和 autotune cache key 会变化；损坏的已缓存 artifact 会被隔离并重编译一次。

GPU 测试必须串行执行：

```bash
ctest --test-dir /tmp/flagdnn-build-nvidia -L functional -j1 --output-on-failure
ctest --test-dir /tmp/flagdnn-build-nvidia -L performance -j1 --output-on-failure
```

也可以按算子运行：

```bash
python3 tools/run_tests.py \
  --build-dir /tmp/flagdnn-build-nvidia \
  --ops sin \
  --suites benchmark \
  --platform nvidia
```

## C++ Graph 示例

```cpp
#include <flagdnn_frontend.h>

namespace fe = flagdnn_frontend;

fe::graph::Graph graph;
graph.set_name("add")
    .set_io_data_type(fe::DataType_t::HALF)
    .set_compute_data_type(fe::DataType_t::FLOAT)
    .set_autotune(true);

auto x = graph.tensor(fe::graph::Tensor_attributes()
                          .set_uid(1)
                          .set_dim({2, 3, 4})
                          .set_stride({12, 4, 1}));
auto y = graph.tensor(fe::graph::Tensor_attributes()
                          .set_uid(2)
                          .set_dim({1, 3, 4})
                          .set_stride({12, 4, 1}));
auto z = graph.pointwise(
    x,
    y,
    fe::graph::Pointwise_attributes()
        .set_mode(fe::PointwiseMode_t::ADD)
        .set_compute_data_type(fe::DataType_t::FLOAT));
z->set_uid(3).set_output(true);
```

Graph 随后通过 `build(handle, ...)` 构建，通过 UID-to-device-pointer bindings 和调用者
stream 执行。

## 开发文档

- [架构边界](docs/architecture.md)
- [算子开发](docs/operator-development.md)
- [功能与性能测试](docs/testing.md)

## License

Apache License 2.0，见 [LICENSE](LICENSE)。
