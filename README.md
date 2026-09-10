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
python3 tools/run_tests.py \
    --platform hygon \
    --device 0 \
    --suites functional,benchmark \
    --no-preflight \
    --verbose \
    --output build/hygon/run-tests.json \
    2>&1 | tee build/hygon/run-tests.log

