# FlagDNN 架构边界

## 1. 总体分层

```text
Public C/C++ API
      |
      v
Platform-neutral Graph + lowering + runtime
      |
      v
External compiler protocol
      |
      +--> platform kernel registry（优先）
      +--> common kernel registry（平台未登记时回退）
      |
      v
Execution engine（当前为 libtriton_jit）
      |
      v
Platform stream + GPU/NPU
```

生产依赖只能向下。公共 API、Graph、lowering、公共 autotune policy、功能 case 和
benchmark workload 都不能包含 CUDA、cuDNN、ACLNN 或其他平台 SDK 头文件。

## 2. 生产目录

### `include/`

唯一公开接口：

- `flagdnn/flagdnn.h`：稳定 C ABI。
- `flagdnn/flagdnn.hpp`：面向内部 lowering 和底层集成者的 C++ descriptor/RAII wrapper。
- `flagdnn/frontend.hpp`：header-only Frontend Graph API。
- `flagdnn_frontend.h`：推荐使用的 cuDNN-Frontend-style 公开入口。
- `flagnn_frontend.h`：历史拼写兼容入口，仅转发到 canonical header。

应用开发者以 `flagdnn_frontend::graph::Graph` 为主入口；`flagdnn::Graph::add()` 只是把已经
构造好的 descriptor 挂入底层图，并不是 `graph.add()` 算子 API。Add 与 cuDNN Frontend
一样通过 `Graph::pointwise(... PointwiseMode_t::ADD)` 表达。

平台 ABI、compiler 协议和内部 lowering 头文件不得安装为公开 API。

### `src/`

平台无关 C++ 核心：

- `src/graph/`：Graph IR、tensor/operation attributes 和 build 生命周期。
- `src/graph/lowering/`：把 Frontend operation 降为 backend 可消费描述。
- `src/runtime/`：compiler client、artifact、cache、executable 和 context。
- `src/api.cpp`：稳定 C ABI 实现。
- `src/backend_loader.*`：backend plugin 发现和加载。

`src/graph` 内部按职责而不是按平台组织：

```text
src/graph/
├── graph.cpp                 # build 编排，不保存算子规则
├── validation.cpp/.hpp      # 图拓扑、UID、binding 与依赖验证
├── ir.cpp/.hpp              # versioned backend IR 序列化
├── tensor.cpp / types.hpp   # 平台无关 Graph 数据模型
└── lowering/
    ├── dispatch.cpp         # operation 到算子族的唯一分发
    ├── helpers.hpp / lowering.hpp
    └── pointwise.cpp, reduction.cpp, matmul.cpp, layout.cpp,
        convolution.cpp, normalization.cpp
```

lowering 只检查公开 Graph 语义并生成 `int64` 描述；warp、block、共享内存、最大 extent
等 kernel 能力由 `backends/<platform>/compiler.py` 判断。因此新增平台不会修改这些文件。

这里不保存 Triton kernel，也不实现具体设备 launch。

Core 在每次 build 时查询 compiler identity，因此同一个 `Handle` 也能观察 compiler、
kernel 或 tuning 修改。artifact 以 graph、target 和 compiler identity 分层缓存；命中项若在
backend 解析/校验阶段报告 `COMPILATION_FAILED`，Core 会隔离该项并只重编译一次。外部编译器
运行在独立进程组中，`FLAGDNN_COMPILER_TIMEOUT_SECONDS` 到期后会终止完整进程树并回收子进程。

### `compiler/` 与 `kernels/common/`

`compiler/` 是独立进程运行的外部编译器，负责解析 versioned request、选择 registry、
展开 tuning candidate，并返回 JIT plan 或外部 artifact manifest。它不是 Python 用户
API。

`kernels/common/` 保存平台无关 Triton 算法，按 binary、unary、reduction、layout、
matmul、convolution、normalization 等算法族复用。common kernel 只能依赖 Triton 和
Python 标准库，不能依赖 backend runtime、Torch 或旧 FlagDNN Python wrapper。

### `backends/` 根目录与 `backends/<platform>/`

`backends/autotune_policy.*` 保存候选 ID、cache、repetition、选择策略等跨平台 backend
机制；它是轻量的根级公共策略，不再为两个文件建立 `common/` 子目录。设备 event、stream
和 module 不属于这一层。

一个平台的生产实现和验证实现统一归属同一个平台根目录：

```text
backends/<platform>/
├── CMakeLists.txt                 # 生产 backend plugin
├── cmake/                         # 可选的平台 SDK Find 模块
├── compiler.py
├── kernels/
│   └── registry.json
├── tuning/
├── engines/
├── *.cpp / *.hpp                 # 生产 runtime/plugin
└── validation/                   # 不安装、不被生产 target 链接
    ├── CMakeLists.txt             # 该平台唯一测试装配入口
    ├── <device>_driver.hpp        # 功能/性能共享的 stream/buffer/event
    ├── tensor_io.cpp/.hpp          # 功能/性能共享的布局、编解码、padding 校验
    ├── functional/                # DNN reference、host oracle、功能 runner
    └── benchmark/                 # DNN provider、计时与结果输出
```

NVIDIA 对应 `backends/nvidia/validation/`。`functional/` 和 `benchmark/` 是同一平台目录
内部的两种验证模式，不是两个平台接入点。

`tools/build.sh` 和 `tools/install.sh` 只负责公共 CMake 配置、构建与安装，不得探测
CUDA、cuDNN、CANN 或其他平台 SDK。生产依赖由 `backends/<platform>/CMakeLists.txt`
发现，validation/reference 依赖由同一平台的 `validation/CMakeLists.txt` 或
`cmake/` Find 模块发现。新增平台不修改公共脚本。

## 3. 平台无关验证目录

```text
tests/
├── CMakeLists.txt
├── test_<op>.cpp
├── common/                 # case、FlagDNN Graph builder、runner contract
└── core/                   # 公共 API、ABI、runtime/backend contract

benchmark/
├── CMakeLists.txt
├── test_<op>.cpp
├── common/                 # workload、provider/case/runner contract
└── result.schema.json
```

`tests/` 只表达“正确性测什么”，`benchmark/` 只表达“性能测什么”，均不知道由哪个设备
SDK 完成验证。不存在 `tests/platforms/`、`benchmark/platforms/` 或 `test_support/`。
唯一算子集合来自 `cmake/Operators.cmake`。

根 CMake 的唯一平台选择项是：

```text
FLAGDNN_BACKENDS=nvidia;ascend
```

`backends/CMakeLists.txt` 使用同一列表装配生产插件；根目录在加载平台无关
tests/benchmark contract 后，再自动加载每个
`backends/<platform>/validation/CMakeLists.txt`。因此新增 Ascend 时，生产、compiler、kernel、
tuning 与验证代码都只增加或修改 `backends/ascend/**`；通用 Graph/lowering、已有 common 和
NVIDIA 源码都无需修改。

## 4. Kernel 解析规则

解析 `(backend, operation)` 时：

1. 查询 `backends/<backend>/kernels/registry.json`。
2. 若平台登记该 operation，返回平台 candidate。
3. 否则查询 `kernels/registry.json` 并返回 common candidate。
4. 两处均没有则明确报错。

平台 registry 一旦登记算子，就拥有该平台上的解析权。平台 candidate 不支持输入或
编译失败时必须返回 capability/compile 错误，不能失败后隐式切换 common，否则行为、
cache identity 和调优空间会不确定。

## 5. Autotune 与 libtriton_jit

- YAML/registry 定义候选空间。
- backend compiler 将候选展开为独立 JIT variant。
- `backends/autotune_policy.cpp` 实现候选 ID、cache、repetition 和选择策略。
- 平台 backend 记录 device event、同步 caller stream 并提供时间样本。
- `libtriton_jit` 编译、加载和启动选中的 Triton kernel。

NVIDIA backend 先准备每个 JIT variant，过滤无法加载的候选。多个有效候选进入公共
autotune，一个候选直接选择，零个候选返回聚合错误。未来 libtriton_jit 原生支持
autotune 后，只替换候选选择 adapter，不改变 Graph、registry 或验证契约。

持久化选择只有在 policy identity、candidate identity、measurement identity 和 device
identity（包含 SM、设备、CUDA Driver）全部匹配时才命中。libtriton_jit 路径还把实际 JIT
共享库及 Python/Triton/Torch 环境加入 measurement identity；计时只覆盖当前候选 stage，
依赖 stage 在 event 计时之前执行。

## 6. Validation 依赖隔离

- 生产 backend target 不链接 validation target。
- validation target 不安装、不导出。
- cuDNN/CANN 等 reference SDK 只由
  `backends/<platform>/validation/CMakeLists.txt` 查找。
- `tests/common`、`benchmark/common` 和 core library 的依赖边界由 CTest contract 检查。
- 同一设备上的 GPU 功能测试和性能测试必须串行执行。

## 7. 禁止重新引入的结构

- Python eager API、Torch wrapper 和 Python Graph frontend
- `src/flag_dnn` 或 `python/flag_dnn`
- `tests/platforms`、`benchmark/platforms`、`test_support`
- `benchmark/cases`（workload 已统一在 `benchmark/common`）
- kernel 旁的 runtime wrapper/decorator
- 按算子复制的 binary/unary Triton 模板
- 根目录之外的 README
- 仓库内 build、cache、coverage 和 benchmark result
- 同时维护 CMake manifest、Markdown 算子清单和 Python inventory
