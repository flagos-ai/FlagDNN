# FlagDNN

FlagDNN 是一个面向多种加速器平台的 C/C++ DNN Graph Runtime。不同平台的环境依赖
和构建参数请参考对应的安装章节。

## 平台安装

### NVIDIA（英伟达）

#### 环境要求

- Linux、NVIDIA GPU，以及相互匹配的驱动和 CUDA Toolkit。当前验证设备为
  H100（SM90），CUDA 版本为 12.9。
- 功能测试和性能测试依赖 cuDNN 头文件、动态库（含 `libcudnn_ext.so.9`）和
  cuDNN Frontend 头文件。当前验证版本为 cuDNN `9.24.0.43`、Frontend `1.26.0`。
- CMake 3.25+（FlagDNN 本身最低要求为 3.23，编译 `libtriton_jit` 需 3.25）、
  Ninja、支持 C++20 的编译器、Python 3.10+ 及 Python 开发头文件和动态库；
  检查动态库依赖还需要 `readelf` 和 `nm`。当前验证使用 Python 3.12。
- 同一 Python 环境中的 CUDA 版 PyTorch、支持 NVIDIA 后端的 FlagTree、
  PyYAML、NumPy、packaging 和 pybind11。
- CUDA 版 `libtriton_jit`，以及配套的头文件、编译脚本和 CMake 配置文件；
  需与 FlagDNN 使用相同的 Python、PyTorch 和 FlagTree 环境。

依赖的编译和安装方法请参考：

- [FlagTree](https://github.com/flagos-ai/flagtree)：选择 NVIDIA 后端，提供
  `triton` Python 模块，无需单独安装 Triton。
- [`libtriton_jit`](https://github.com/flagos-ai/libtriton_jit)：编译时设置
  `-DBACKEND=CUDA`。

建议将 FlagDNN 与 `libtriton_jit` 放在同一级目录，默认从
`../libtriton_jit/build` 查找 JIT 依赖。cuDNN 和 Frontend 可从所选 Python 环境中的
`nvidia-cudnn-cu12`、`nvidia-cudnn-frontend` 安装目录自动发现。

#### 编译

在已激活上述 Python 环境的 FlagDNN 根目录执行：

```bash
tools/build.sh \
  --backends nvidia \
  --build-dir build/nvidia
```

该命令以 `Release` 模式编译 FlagDNN，并同时构建功能测试和性能测试。
NVIDIA 后端仅支持默认的 `libtriton_jit` 执行引擎。若依赖安装在其他位置，
可显式指定 Python、CUDA、`libtriton_jit` 和 cuDNN 路径：

```bash
tools/build.sh \
  --backends nvidia \
  --build-dir build/nvidia \
  --python /path/to/python3 \
  -- \
  -DCUDAToolkit_ROOT=/path/to/cuda \
  -DTritonJIT_DIR=/path/to/libtriton_jit/build \
  -DCUDNN_ROOT=/path/to/cudnn \
  -DCUDNN_FRONTEND_INCLUDE_DIR=/path/to/cudnn-frontend/include
```

`TritonJIT_DIR` 应指向包含 `TritonJITConfig.cmake` 的目录，已安装的包通常位于
`<prefix>/lib/cmake/TritonJIT`。`CUDNN_ROOT` 应包含配套的头文件和动态库；
`CUDNN_FRONTEND_INCLUDE_DIR` 应包含 `cudnn_frontend.h`。

#### 安装

```bash
# 默认安装到 build/nvidia/install
tools/install.sh --build-dir build/nvidia

# 指定安装目录
tools/install.sh \
  --build-dir build/nvidia \
  --prefix /path/to/flagdnn-sdk
```

安装后的 SDK 仍依赖兼容的 CUDA、`libtriton_jit` 和构建时选定的 Python 环境；
功能测试和性能测试保留在构建目录中，不随 SDK 安装。

#### 批量测试

在与编译时一致的 Python 环境中执行：

```bash
python3 tools/run_tests.py \
    --platform nvidia \
    --gpus 1,2 \
    --dump-output \
    --output-dir logs_result_20260918_flagdnn
```

测试细节见[功能与性能验证](docs/testing.md)

### Hygon（海光）

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

请选择与当前 DTK 兼容的依赖版本。`libtriton_jit` 可放在任意目录；未指定路径时，
默认从 FlagDNN 同级的 `../libtriton_jit` 查找其构建产物或安装文件。

#### 编译

在 FlagDNN 根目录执行：

```bash
tools/build.sh \
  --backends hygon \
  --build-dir build/hygon
```

该命令以 `Release` 模式编译 FlagDNN，并同时构建功能测试和性能测试。

若 `libtriton_jit` 位于其他目录，在 `--` 后传入 Hygon 专用的 CMake 参数：

```bash
tools/build.sh \
  --backends hygon \
  --build-dir build/hygon \
  -- \
  -DFLAGDNN_HYGON_TRITON_JIT_DIR=/path/to/libtriton_jit/build
```

`FLAGDNN_HYGON_TRITON_JIT_DIR` 指向包含 `TritonJITConfig.cmake` 的目录，
不是 `.so` 文件。源码构建通常位于 `<repo>/build`，安装包通常位于
`<prefix>/lib/cmake/TritonJIT` 或 `<prefix>/lib64/cmake/TritonJIT`。
也可改用 `-DFLAGDNN_HYGON_TRITON_JIT_ROOT=/path/to/libtriton_jit` 指定已构建的
源码仓库根目录或完整安装前缀。

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
python3 tools/run_tests.py \
    --platform hygon \
    --gpus 1,2 \
    --dump-output \
    --output-dir logs_result_20260918_flagdnn
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
python3 tools/run_tests.py \
    --platform iluvatar \
    --gpus 1,2 \
    --dump-output \
    --output-dir logs_result_20260918_flagdnn
```

更多测试说明见[功能与性能验证](docs/testing.md)。

### THEAD（阿里平头哥）

#### 环境要求

- Linux、平头哥 PPU，以及相互匹配的驱动、PPU SDK 和 acDNN。当前验证设备为
  PPU-ZW810E，CUDA compatibility capability 为 8.0；验证环境为 PPU SDK
  `2.1.0-a5f865`、acDNN header/runtime `1400`。
- CMake 3.25+（FlagDNN 本身最低要求为 3.23，编译 `libtriton_jit` 需 3.25）、
  Ninja、支持 C++20 的编译器、Python 及对应开发头文件；当前验证使用 Python 3.12。
  检查 SDK 动态库还需要 `readelf` 和 `nm`（或对应的 LLVM 工具）。
- 同一 Python 环境中的 PPU 版 PyTorch、支持 PPU 的 FlagTree、PyYAML、NumPy、
  packaging 和 pybind11。当前构建及安装验证使用 FlagTree `0.7.0+ppu.git22f4ff0e`，
  对应 Triton API `3.6.0`。
- CUDA 后端的 `libtriton_jit`，使用 PPU SDK 的 CUDA 兼容环境构建，且与所选
  Python、PyTorch 的 ABI 一致。THEAD 当前使用 `libtriton_jit` 执行 Triton kernel。

依赖的编译和安装方法请参考：

- [FlagTree](https://github.com/flagos-ai/flagtree)：选择 PPU 发行版。发行包名为
  `flagtree`，对外提供的 Python 模块仍名为 `triton`，因此代码使用 `import triton`；
  无需另装上游 Triton。
- [`libtriton_jit`](https://github.com/flagos-ai/libtriton_jit)：编译时设置
  `-DBACKEND=CUDA`，使用 PPU SDK 的 CUDA 兼容头文件和动态库；需保留动态库、
  头文件、编译脚本和 `TritonJITConfig.cmake`。

默认从 `/usr/local/PPU_SDK` 查找 SDK，从 FlagDNN 同级的
`libtriton_jit` 目录查找 JIT 依赖；FlagTree 路径由所选 Python 环境自动发现。
功能和性能测试严格使用同一 PPU SDK 中的 acDNN 作为参考。

#### 编译

在已安装 PPU 版 FlagTree 及上述依赖的 Python 环境中，于 FlagDNN 根目录执行。
可直接使用系统 Python，无需创建或激活虚拟环境：

```bash
tools/build.sh \
  --backends thead \
  --build-dir build/thead
```

该命令以 `Release` 模式编译 FlagDNN，并同时构建功能测试和性能测试。

显式指定 Python、PPU SDK、`libtriton_jit` 和 FlagTree 路径：

```bash
tools/build.sh \
  --backends thead \
  --default-backend thead \
  --build-dir build/thead \
  --python /path/to/ppu-env/bin/python3 \
  -- \
  -DFLAGDNN_THEAD_PPU_SDK_ROOT=/path/to/PPU_SDK \
  -DFLAGDNN_THEAD_TRITON_JIT_ROOT=/path/to/libtriton_jit \
  -DFLAGDNN_THEAD_TRITON_ROOT=/path/to/ppu-env/lib/python3.12/site-packages
```

`FLAGDNN_THEAD_TRITON_JIT_ROOT` 可指向已构建的源码仓库根目录或完整安装前缀。
`FLAGDNN_THEAD_TRITON_ROOT` 应指向同时包含 `triton/` 和对应发行包元数据的
Python 包根目录，通常为 `site-packages`；可省略此参数以自动发现。
`--python` 也可指定系统 Python（例如 `/usr/local/bin/python3`），无需激活虚拟环境。
解释器与 FlagTree 包目录可以分别选择；所选 Python 仍需能导入 PPU 版 PyTorch
及其余依赖。

#### 安装

```bash
# 默认安装到 build/thead/install
tools/install.sh --build-dir build/thead

# 指定安装目录
tools/install.sh \
  --build-dir build/thead \
  --prefix /path/to/flagdnn-sdk
```

#### 批量测试

在与编译时一致的 Python 环境中执行：

```bash
python3 tools/run_tests.py \
    --platform thead \
    --gpus 1,2 \
    --dump-output \
    --output-dir logs_result_20260918_flagdnn
```

如需使用原有 `psum_text`，改用 `--output build/thead/psum-report/summary.json`
串行兼容模式（不传 `--gpus`、`--dump-output` 或 `--output-dir`）：

```bash
./tools/psum_text /path/to/FlagDNN/build/thead/psum-report/
```

通用测试说明见
[功能与性能验证](docs/testing.md)。

### MThreads（摩尔线程）

#### 环境要求

- Linux、摩尔线程 GPU，以及相互匹配的驱动、MUSA Toolkit 和 muDNN。当前适配
  基线为 MTT S5000、MUSA 4.3.5 和 muDNN 3.1.5。
- CMake 3.25+（FlagDNN 本身最低要求为 3.23，编译 `libtriton_jit` 需 3.25）、
  Ninja、支持 C++20 的编译器、Python 3.10+ 及 Python 开发头文件。
- 同一 Python 环境中的 PyTorch、与其匹配的 `torch_musa`、支持 MUSA 后端的
  FlagTree、PyYAML、NumPy、packaging 和 pybind11。
- MUSA 版 `libtriton_jit`，以及可从 `PATH` 找到的 `mcc`、`patchelf` 和
  `readelf`。

依赖的编译和安装方法请参考：

- [FlagTree](https://github.com/flagos-ai/flagtree)：选择 MUSA（`mtgpu`）后端，
  提供 `triton` Python 模块，无需单独安装 Triton。
- [`libtriton_jit`](https://github.com/flagos-ai/libtriton_jit)：编译时设置
  `-DBACKEND=MUSA`，安装动态库、头文件、脚本和 CMake 配置文件。

请选择与当前 MUSA 兼容的依赖版本，并在编译和运行测试前激活对应的 Python 环境。
确保 MUSA、Torch/MKL 及 `libtriton_jit` 所依赖的其他动态库可被加载；环境变量
配置示例见[摩尔线程适配设计](docs/mthreads-adaptation-design.md)。

#### 编译

```bash
tools/build.sh \
  --backends mthreads \
  --build-dir build/mthreads
```

在 FlagDNN 根目录执行，以下示例假设 MUSA 安装在 `/usr/local/musa`，MUSA 版
`libtriton_jit` 安装在 `/usr/local`：

```bash
export MUSA_HOME=/usr/local/musa
export PATH="$MUSA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$MUSA_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

tools/build.sh \
  --backends mthreads \
  --build-dir build/mthreads \
  -- \
  -DFLAGDNN_MTHREADS_MUSA_ROOT="$MUSA_HOME" \
  -DFLAGDNN_MTHREADS_TRITON_JIT_DIR=/usr/local/lib/cmake/TritonJIT
```

该命令以 `Release` 模式编译 FlagDNN，并同时构建功能测试和性能测试。

`FLAGDNN_MTHREADS_MUSA_ROOT` 和 `FLAGDNN_MTHREADS_TRITON_JIT_DIR` 均须显式
指定，后者应指向包含 `TritonJITConfig.cmake` 的目录。若依赖安装在其他位置，
请调整上述路径；指定 Python 时，在 `--` 前添加 `--python /path/to/python3`。

#### 安装

```bash
# 默认安装到 build/mthreads/install
tools/install.sh --build-dir build/mthreads

# 指定安装目录
tools/install.sh \
  --build-dir build/mthreads \
  --prefix /path/to/flagdnn-sdk
```

#### 批量测试

```bash
python3 tools/run_tests.py \
    --platform mthreads \
    --gpus 1,2 \
    --dump-output \
    --output-dir logs_result_20260918_flagdnn
```

如需使用原有 `psum_text`，改用 `--output build/mthreads/psum-report/summary.json`
串行兼容模式（不传 `--gpus`、`--dump-output` 或 `--output-dir`）：

```bash
./tools/psum_text /path/to/FlagDNN/build/mthreads/psum-report/
```

### Ascend（昇腾）

#### 环境要求

- Linux（x86_64 或 aarch64）、昇腾 NPU，以及相互匹配的驱动、固件和 CANN 9.x
  （要求 `>= 9.0` 且 `< 10.0`）。支持的设备型号见
  [Ascend 能力配置](backends/ascend/capabilities.json)中的 `codegen_arches`。
- CANN 中的 ACL 头文件、`libascendcl.so` 和 `libruntime.so`；功能测试和性能测试
  还依赖同一 CANN 安装中的 ACLNN 头文件、`libnnopbase.so`、`libopapi_math.so`
  和 `libopapi_nn.so`。
- CMake 3.25+（FlagDNN 本身最低要求为 3.23，编译 `libtriton_jit` 需 3.25）、
  Ninja、支持 C++20 的编译器、Python 3 及 Python 开发头文件和动态库；
  检查动态库依赖还需要 `readelf`。
- 同一 Python 环境中的 PyTorch、与之匹配的 `torch_npu`、支持 Ascend 后端的
  FlagTree、PyYAML、NumPy、packaging 和 pybind11。
- NPU 版 `libtriton_jit`，以及配套的头文件、`standalone_compile.py` 和 CMake
  配置文件。其链接的 Python 动态库必须与 FlagDNN 编译时选择的 Python 一致。

依赖的编译和安装方法请参考：

- [FlagTree](https://github.com/flagos-ai/flagtree)：选择 Ascend 后端，提供
  `triton` Python 模块，无需单独安装 Triton。
- [`libtriton_jit`](https://github.com/flagos-ai/libtriton_jit)：编译时设置
  `-DBACKEND=NPU`，并使用与 FlagDNN 相同的 Python 环境。

FlagDNN 从 `ASCEND_HOME_PATH` 或 `ASCEND_TOOLKIT_HOME` 获取 CANN 路径，也可在
编译时显式指定 `CANN_ROOT`。建议将 FlagDNN 与 `libtriton_jit` 放在同一级目录，
默认会尝试从 `../libtriton_jit/build` 查找 NPU 版 `libtriton_jit`。

#### 编译

在 FlagDNN 根目录执行：

```bash
tools/build.sh \
  --backends ascend \
  --build-dir build/ascend
```

该命令以 `Release` 模式编译 FlagDNN，并同时构建功能测试和性能测试。
Ascend 后端仅支持默认的 `libtriton_jit` 执行引擎。若依赖安装在其他位置，
可显式指定 Python、CANN 和 `libtriton_jit` 路径：

```bash
tools/build.sh \
  --backends ascend \
  --build-dir build/ascend \
  --python /path/to/python3 \
  -- \
  -DCANN_ROOT=/path/to/cann-9.x \
  -DTritonJIT_DIR=/path/to/libtriton_jit/build
```

`CANN_ROOT` 应为 CANN 安装目录的绝对路径；`TritonJIT_DIR` 应指向包含
`TritonJITConfig.cmake` 的目录，已安装的包通常为 `<prefix>/lib/cmake/TritonJIT`。
若无法自动找到配套的 `standalone_compile.py`，可在 `--` 后追加
`-DFLAGDNN_TRITON_JIT_STANDALONE_COMPILER=/path/to/standalone_compile.py`。

#### 安装

```bash
# 默认安装到 build/ascend/install
tools/install.sh --build-dir build/ascend

# 指定安装目录
tools/install.sh \
  --build-dir build/ascend \
  --prefix /path/to/flagdnn-sdk
```

安装后的 SDK 仍依赖兼容的 CANN 和构建时选定的 Python 环境；功能测试和性能测试
保留在构建目录中，不随 SDK 安装。

#### 批量测试

```bash
python3 tools/run_tests.py \
    --platform ascend \
    --gpus 1,2 \
    --dump-output \
    --output-dir logs_result_20260918_flagdnn
```

更多测试说明见[功能与性能验证](docs/testing.md)。
