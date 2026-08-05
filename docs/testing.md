# FlagDNN 功能与性能验证

## 1. 一个算子清单，两种平台无关 workload

`cmake/Operators.cmake` 是 C++ 功能测试和性能测试的唯一算子清单。每个条目必须同时
存在：

```text
tests/test_<op>.cpp
benchmark/test_<op>.cpp
```

顶层入口是薄 `main`：功能入口读取 `tests/common` case 并调用功能 runner contract；
性能入口读取 `benchmark/common` workload 并调用 benchmark runner contract。入口不
包含 CUDA、cuDNN、CANN、ACL 或任何其他平台 SDK。

`cmake/VerifyPerOperatorTestLayout.cmake` 检查入口数量、目录边界、平台依赖隔离，并
禁止重新出现 Python 测试和旧平台目录。

## 2. 平台无关目录

```text
tests/
├── CMakeLists.txt          # common/core target 与功能 suite 注册函数
├── test_<op>.cpp           # 每个算子一个薄入口
├── common/                 # case、FlagDNN Graph builder、runner contract
└── core/                   # C/C++ API、ABI、runtime/backend/安装 contract

benchmark/
├── CMakeLists.txt          # workload target 与性能 suite 注册函数
├── test_<op>.cpp           # 每个算子一个薄入口
├── common/                 # case/workload、provider 与 runner contract
└── result.schema.json      # 稳定 JSONL 输出格式
```

`tests/common` 只定义 shape、dtype、layout、tolerance、autotune case 和被测 FlagDNN
Graph；`benchmark/common` 只定义 workload、采样配置及 provider contract。两者都不
决定 reference SDK、device memory、stream 或 event timer。

## 3. 平台验证只占一个目录

所有平台私有验证代码与生产 backend 同属一个平台根目录：

```text
backends/nvidia/validation/
├── CMakeLists.txt          # NVIDIA 唯一验证装配入口
├── cuda_driver.hpp         # 功能/性能共享的 context/stream/buffer/event
├── tensor_io.cpp/.hpp   # 功能/性能共享的布局、编解码、padding 校验
├── functional/             # cuDNN Graph reference、host oracle、功能 runner
└── benchmark/              # cuDNN provider、正确性比较、GPU event 计时、JSONL
```

不存在 `tests/platforms`、`benchmark/platforms` 或 `test_support`。适配 Ascend 时只需
增加或修改 `backends/ascend/validation/**`；平台目录内部可以按功能/性能分文件，但
开发者只有一个平台接入点。

平台 reference 不支持某个 primitive 时，必须注册明确的 capability test，或使用
经过审查的 host oracle，不能跳过 FlagDNN 的真实设备执行。当前 NVIDIA reference
主要使用 cuDNN Frontend Graph API。

## 4. CMake 装配顺序

根 `CMakeLists.txt` 的顺序固定为：

1. 构建平台无关 core 和生产 backend。
2. 按开关加载 `tests/`、`benchmark/`，定义公共 workload 和注册函数。
3. 读取唯一的 `FLAGDNN_BACKENDS` 列表。
4. 加载 `backends/<platform>/validation/CMakeLists.txt`。

平台 validation CMake 独立查找设备 SDK，并按已启用的模式调用：

- `flagdnn_register_functional_suite(...)`
- `flagdnn_register_benchmark_suite(...)`

功能和性能可以独立构建；benchmark 不再要求 `FLAGDNN_BUILD_TESTS=ON`。validation
target 不安装、不导出，生产 backend 不链接 validation。

例如同时选择 NVIDIA 与 Ascend：

```bash
-DFLAGDNN_BACKENDS='nvidia;ascend'
```

## 5. 性能 case 执行语义

每个 benchmark case 依次：

1. 构建 FlagDNN Graph executable。
2. 构建平台 DNN reference，或返回明确 capability reason。
3. 使用相同输入运行并比较输出。
4. 正确后执行 warmup。
5. 交替测量 FlagDNN 与 reference，降低执行顺序偏差。
6. 输出 `steady_state` JSONL 样本和 speedup。

每个逐算子可执行文件的第一个 case 设置 `Graph::set_autotune(true)`；后续 case 仍走
libtriton_jit，但不会重复同一 executable 的完整候选计时。

## 6. NVIDIA 全量构建

推荐入口：

```bash
tools/build.sh
```

默认根据 backend 生成 `build/<backend>` Release 构建，并启用功能与性能测试。公共
脚本不发现平台 SDK；NVIDIA 的 libtriton_jit、Torch 和 cuDNN 发现由
`backends/nvidia/` 负责。使用 `tools/build.sh --help` 查看 Python、backend、构建
目录和并行度等公共选项，其他 CMake 参数放在 `--` 后。下面的命令是等价的手动
CMake 配置示例。

```bash
cmake -S . -B /tmp/flagdnn-build-nvidia \
  -DFLAGDNN_BACKENDS=nvidia \
  -DFLAGDNN_BUILD_TESTS=ON \
  -DFLAGDNN_BUILD_BENCHMARKS=ON \
  -DFLAGDNN_EXECUTION_ENGINE=libtriton_jit \
  -DFLAGDNN_CODEGEN_PYTHON=/path/to/python
cmake --build /tmp/flagdnn-build-nvidia -j
```

只构建平台无关 core contract 时设置 `-DFLAGDNN_BACKENDS=`。

## 7. 运行方法

结构、API 和依赖 contract：

```bash
ctest --test-dir /tmp/flagdnn-build-nvidia \
  -L contract -j1 --output-on-failure
```

全部功能测试严格串行：

```bash
ctest --test-dir /tmp/flagdnn-build-nvidia \
  -L functional -j1 --output-on-failure
```

全部性能测试严格串行：

```bash
ctest --test-dir /tmp/flagdnn-build-nvidia \
  -L performance -j1 --output-on-failure
```

按算子运行：

```bash
python3 tools/run_tests.py \
  --build-dir /tmp/flagdnn-build-nvidia \
  --ops matmul \
  --suites functional,benchmark \
  --platform nvidia \
  --output /tmp/flagdnn-matmul-results.json
```

当前不提供 `--ops` 时，默认依次运行 `add`、`sub`、`mul`、`div`、`pow`、`max`、
`min`、`mod`、`add_square` 和 `cmp_eq`。使用 `--ops all` 运行所选 suite 在
manifest 中注册的全部算子；使用 `--list` 只打印全部 manifest 算子列表。

## 8. 串行与结果纪律

同一设备上的 GPU 测试必须使用 `-j1`。并行 GPU 进程会污染 autotune winner、cache
首次编译成本、GPU event 样本和平台 DNN plan 选择。

benchmark 的稳定机器接口由 `benchmark/result.schema.json` 定义。控制台文本只供
开发者阅读，不应作为持续集成的数据接口。
