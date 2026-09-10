# FlagDNN

FlagDNN 是一个面向多种加速器平台的 C/C++ DNN Graph Runtime。不同平台的环境依赖
和构建参数请参考对应的安装章节。

## 平台安装

### Hygon

#### 环境要求

- Linux、海光 DCU，以及相互匹配的驱动、DTK（HIP）和 hipDNN。
- CMake 3.25+（FlagDNN 本身最低要求为 3.23，编译 `libtriton_jit` 需 3.25）、
  Ninja、支持 C++20 的编译器、Python 3 及 Python 开发头文件。
- 同一 Python 环境中的海光版 PyTorch、FlagTree、PyYAML、NumPy、packaging 和
  pybind11。
- HCU 版 `libtriton_jit`。

依赖的编译和安装方法请参考：

- [FlagTree](https://github.com/flagos-ai/flagtree)：用于替代 Triton，无需单独安装 Triton。
- [`libtriton_jit`](https://github.com/flagos-ai/libtriton_jit)：需编译为 HCU 版本。

请选择与当前 DTK 兼容的依赖版本。建议将 FlagDNN 与 `libtriton_jit` 放在同一级
目录。

#### 编译

在 FlagDNN 根目录执行：

```bash
tools/build.sh \
  --backends hygon \
  --build-dir build/hygon
```

该命令以 `Release` 模式编译 FlagDNN，并同时构建功能测试和性能测试。

#### 安装

```bash
# 默认安装到 build/hygon/install
tools/install.sh --build-dir build/hygon

# 指定安装目录
tools/install.sh \
  --build-dir build/hygon \
  --prefix /path/to/flagdnn-sdk
```

#### 批量测试

```bash
set -o pipefail
python3 tools/run_tests.py \
    --platform hygon \
    --device 0 \
    --suites functional,benchmark \
    --no-preflight \
    --verbose \
    --output build/hygon/run-tests.json \
    2>&1 | tee build/hygon/run-tests.log
```

### Iluvatar（天数智芯）

#### 环境要求

- Linux、天数智芯 GPU，以及相互匹配的驱动和 CoreX SDK。当前后端仅支持
  `corex_71` 设备，适配环境为 CoreX 4.4.0。
- 功能测试和性能测试依赖同一 CoreX SDK 中的 cuDNN 兼容库，当前要求
  `cudnn.h` 版本为 7.6.5、动态库为 `libcudnn.so.7`。
- CMake 3.25+（FlagDNN 本身最低要求为 3.23，编译 `libtriton_jit` 需 3.25）、
  Ninja、支持 C++20 的编译器、Python 3.10+ 及 Python 开发头文件；
  检查 SDK 动态库还需要 `readelf` 和 `nm`（或对应的 LLVM 工具）。
- 同一 Python 环境中的天数智芯版 PyTorch、支持 Iluvatar 后端的 FlagTree、
  PyYAML、NumPy、packaging 和 pybind11。
- IX 版 `libtriton_jit`。

依赖的编译和安装方法请参考：

- [FlagTree](https://github.com/flagos-ai/flagtree)：选择 Iluvatar 后端，提供
  `triton` Python 模块，无需单独安装 Triton。
- [`libtriton_jit`](https://github.com/flagos-ai/libtriton_jit)：编译时设置
  `-DBACKEND=IX`，安装动态库、头文件、脚本和 CMake 配置文件。

请选择与当前 CoreX 兼容的依赖版本，并在编译和运行测试前激活对应的 Python 环境。
默认从 `/usr/local/corex` 查找 CoreX，从 `/usr/local/lib/cmake/TritonJIT`
查找已安装的 IX 版 `libtriton_jit`。

#### 编译

在 FlagDNN 根目录执行：

```bash
tools/build.sh \
  --backends iluvatar \
  --build-dir build/iluvatar
```

该命令以 `Release` 模式编译 FlagDNN，并同时构建功能测试和性能测试。
若依赖安装在其他位置，可显式指定 Python、CoreX 和 `libtriton_jit` 路径：

```bash
tools/build.sh \
  --backends iluvatar \
  --build-dir build/iluvatar \
  --python /path/to/python3 \
  -- \
  -DFLAGDNN_ILUVATAR_COREX_ROOT=/path/to/corex \
  -DFLAGDNN_ILUVATAR_TRITON_JIT_DIR=/path/to/libtriton_jit/lib/cmake/TritonJIT
```

`FLAGDNN_ILUVATAR_TRITON_JIT_DIR` 应指向包含 `TritonJITConfig.cmake` 的目录。

#### 安装

```bash
# 默认安装到 build/iluvatar/install
tools/install.sh --build-dir build/iluvatar

# 指定安装目录
tools/install.sh \
  --build-dir build/iluvatar \
  --prefix /path/to/flagdnn-sdk
```

#### 批量测试

```bash
set -o pipefail
python3 tools/run_tests.py \
    --platform iluvatar \
    --device 0 \
    --suites functional,benchmark \
    --no-preflight \
    --verbose \
    --output build/iluvatar/run-tests.json \
    2>&1 | tee build/iluvatar/run-tests.log
```

CoreX cuDNN 不支持的 reference case 会记录结构化 `SKIP`，应结合 JSON 汇总中的
覆盖情况和跳过原因解读结果。更多测试说明见[功能与性能验证](docs/testing.md)。
