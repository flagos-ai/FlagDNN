# FlagDNN mthreads（摩尔线程 MUSA）适配设计

> 文档状态：设计已实装；功能与覆盖全量回归完成，性能门禁的剩余差距已记录
> 基线日期：2026-08-24；架构复核日期：2026-09-01
> 目标 backend 名称：mthreads
> 目标 production plugin：libflagdnn_backend_mthreads.so.2

本文定义 FlagDNN 在摩尔线程 MUSA 平台上的目标架构、实现边界、算子策略、
验证口径、安装闭包和最终验收标准，并记录 2026-08-26 已完成实现的实际结构与
实机证据。目标设计仍作为后续维护契约；文末实装记录用于区分已验证事实、当前 SDK
边界和仍需执行的发布门禁。

当前 `backends/mthreads` 已具备 production backend、compiler provider、MUSA
libtriton_jit engine、autotune、validation adapter、61 个 functional target、57 个
benchmark target、环境 identity helper 以及隔离安装态 consumer。实现中若发现本文与真实
SDK 行为不一致，仍应先用最小实机 probe 固化证据，再更新本文；不能静默放宽测试或
在公共层添加 MUSA 特判。

## 1. 结论与不可变约束

采用“mthreads 独立原生 backend”方案：

1. FlagDNN Frontend Graph API 是用户入口和被测接口。
2. mthreads provider 独立把公共 Graph IR 降低为一个或多个 MUSA Triton kernel
   stage。
3. production 使用 MUSA 版 libtriton_jit 编译、加载并在调用者 MUSA stream 上
   launch kernel。
4. validation 直接使用 muDNN C++ primitive 或严格等价的 primitive sequence
   作为数值 reference。
5. 公共 Graph、Graph IR、lowering、tests、benchmark、构建脚本和批量测试 runner
   默认零修改。
6. 不修改 backends/nvidia 或任何其他平台的专用代码，也不从 mthreads 反向 include
   其他 backend 的私有文件。
7. 使用字符串 backend 名称 mthreads 和现有 backend ABI v2，不新增公共 backend
   enum。
8. 第一阶段 production 只支持 libtriton_jit execution engine。对
   external_artifact 的请求在 mthreads 配置阶段明确失败，不能运行时悄悄降级。
9. muDNN、muBLAS、torch_musa 测试辅助代码不得链接进已安装的 production plugin。
   torch_musa 只允许作为 MUSA Triton/libtriton_jit 编译环境依赖。
10. 最终验收必须真实打通编译、安装、Graph、libtriton_jit、Triton MUSA kernel、
    autotune、功能测试、性能测试和 tools/run_tests.py；不能用 mock、host reference
    或大面积 SKIP 代替。

这些约束优先于“快速复用”或局部性能优化。如果新增 mthreads backend 暴露出公共
扩展点缺失，应单独提交中立设计，并为所有受影响 backend 增加回归测试；在该设计
获批前，mthreads 工作不得修改公共或其他平台代码。

## 2. 方案选择

### 2.1 采用方案：独立 provider + common kernel 优先 + MUSA 私有 override

该方案复用的是 NVIDIA backend 的职责边界和协议，不复制其 CUDA 实现：

- 复用公共 Graph IR、compiler provider loader、kernel registry、artifact cache、
  backend ABI 和测试 case；
- mthreads 自己实现 Graph lowering、artifact parser、MUSA context、JIT engine 和
  validation；
- common registry 中的 Triton kernel 先经过 MUSA 兼容性验证；
- 只有语义不兼容、编译不兼容或有实测性能理由时，才在
  backends/mthreads/kernels 增加 platform override；
- 一个 operation 一旦注册 platform override，就由 mthreads 完全拥有，override
  编译失败不能隐式回退到 common kernel。

这能保持公共接口和算子语义一致，同时把 SDK、架构、编译器和性能差异限制在
backends/mthreads 内。

### 2.2 不采用：复制 NVIDIA backend 后替换 CUDA API

NVIDIA compiler 和 kernels 包含 CUDA libdevice、SM 架构、TMA、WGMMA、Hopper
descriptor、CUDA Driver module 及 CUDA Graph 等专用逻辑。机械复制会把 NVIDIA
策略误当作平台中立契约，并产生难以审计的隐式行为。mthreads 可以参考其文件职责、
错误处理和 manifest 校验，但不能链接、导入或运行 NVIDIA 私有实现。

### 2.3 后备方案：自建 Triton 编译与 launcher

只有当 MUSA libtriton_jit 无法满足下列硬门禁，并且问题不能在其上游修复时，才重新
评审自建 launcher：

- caller stream 透传；
- 安装态 CMake/package/resource 可发现；
- module/function 生命周期有界；
- MUSA runtime Graph capture 可重放；
- 多次 executable build/destroy 不产生无界资源增长。

在这之前不得在 FlagDNN 中复制 libtriton_jit 的编译、缓存或 launcher 逻辑。

## 3. 范围与非目标

### 3.1 本设计覆盖

- backend 配置、构建与安装；
- Graph IR 到 execution program 的 mthreads compiler；
- common kernel 复用和 MUSA kernel override；
- artifact 完整性、ABI、workspace 和 stage DAG；
- libtriton_jit 的 MUSA backend 集成；
- MUSA stream、event、runtime Graph capture；
- build-time autotune 和 winner cache；
- 61 个 functional operator、57 个 benchmark operator 的注册和执行策略；
- muDNN C++ reference、结构化 capability 和严格 case accounting；
- 源码树、构建树、安装树三种运行形态；
- tools/run_tests.py 的 mthreads 平台适配和最终全量验收。

### 3.2 本设计不做

- 不为 muDNN 仿造一个 Graph builder；
- 不改变 FlagDNN 公共 C/C++ Graph API；
- 不把 torch、torch_musa 或 torchada 作为数值 reference；
- 不依赖 CUDA symbol 翻译或运行时 monkey patch；
- 不承诺当前 muDNN 头文件中出现的 primitive 一定支持所有公共 case；
- 不以 host 计算、FlagDNN 自身输出或另一个自研 Triton kernel充当独立 oracle；
- 不把安装包做成包含完整 MUSA SDK、Python 和 PyTorch 的完全静态发行物；
- 不把 validation/reference 代码安装进 production SDK。

## 4. 当前实机基线与配置门禁

以下内容是 2026-08-24 在当前机器上的观测值，只是首个开发基线，不是硬编码的永久
支持矩阵：

| 项目 | 当前观测 |
| --- | --- |
| 设备 | MTT S5000 |
| MUSA Toolkit | /usr/local/musa，4.3.5 |
| MUSA Driver | 3.3.5-server，Driver/Runtime API 4.3 |
| 设备能力 | compute capability 3.1，warp size 32 |
| muDNN | header/runtime 3.1.5，SONAME libmudnn.so.3 |
| muBLAS | 1.10.5；不作为默认 reference |
| Python | /home/wangbingjie/flagdnn，Python 3.10.12 |
| Torch/torch_musa | 2.9.0 系列 MUSA wheel |
| Triton | 3.6.0，已安装 mthreads backend |
| libtriton_jit | /usr/local，CMake backend MUSA，内部 backend 名 mtgpu |
| Python 辅助依赖 | PyYAML 6.0.2，配置期直接校验 import 和模块 identity |

配置阶段与 NVIDIA 后端保持相同职责边界：直接解析选中的 SDK、Python 和
TritonJIT CMake package，并在 build tree 中生成机器可读的工具链 environment
identity；不要求用户预先运行独立 probe 或提供外部报告。配置期至少校验：

1. MUSA header、runtime library、driver library来自同一个规范化 MUSA 根目录；
2. 编译期和运行期 MUSA 主版本兼容；
3. muDNN header version、runtime version 和 SONAME 一致；
4. codegen Python 可导入 torch、torch_musa、triton 和 yaml；
5. TritonJITConfig.cmake 声明的 backend 是 MUSA；
6. libtriton_jit library、header、CMake config、standalone_compile.py 以及它使用的
   helper scripts 来自同一 provenance；
7. libtriton_jit 的直接和间接 ELF 依赖在构建树与安装树中都可解析。

设备可见性、实际 Triton target、caller stream、Graph capture 和数值正确性属于
运行能力，不在 CMake 配置期访问 GPU；它们由 compiler identity 和正式
`validation` 测试在需要设备时校验。设备架构、warp size、shared-memory 限制和最大
block threads 由运行时查询，不能从设备名称猜测。

任一门禁失败时 CMake 必须 fail closed，并显示选中资源的 realpath、版本、SONAME 和
Python executable。不得回退到系统中的 CUDA PyTorch/Triton，也不得混用两个 MUSA
SDK 或两个虚拟环境。

官方 libtriton_jit 提供 MUSA build backend，torch_musa 提供 MUSA PyTorch 运行环境；
实现和问题定位应优先与这些上游契约对齐：

- https://github.com/flagos-ai/libtriton_jit
- https://github.com/MooreThreads/torch_musa

## 5. 总体数据流

### 5.1 production

    FlagDNN public C/C++ Graph API
      -> 公共 Graph validation/lowering
      -> versioned Graph IR（backend=mthreads）
      -> compiler/flagdnn_codegen/main.py
      -> backends/mthreads/compiler.py provider
      -> common registry 或 mthreads platform registry
      -> materialized Triton source + execution-program manifest
      -> libflagdnn_backend_mthreads.so.2
      -> MUSA libtriton_jit
      -> MUSA module/function
      -> 调用者提供的 musaStream_t

公共 core 只知道 backend 名称、Graph IR、artifact directory 和 ABI v2。MUSA SDK、
Triton target、kernel entry、launch metadata 和 module 生命周期全部由 mthreads
plugin 负责。

### 5.2 functional validation

    公共 FunctionalCase
      +-> 相同 seed 的 DUT 输入 -> FlagDNN Graph -> output A
      +-> 独立 reference 输入 -> muDNN primitive/sequence -> output B
      -> 比较全部 logical output、padding canary 和输入未修改状态

reference 和 DUT 可以共享逻辑输入值，但必须使用独立的可变输出、workspace 和
中间存储。任何一侧失败都不能使用另一侧结果继续构造 oracle。

### 5.3 Graph 的两个含义

本文严格区分：

1. FlagDNN Frontend Graph：被测的算子表达和多节点执行语义；
2. MUSA runtime Graph：对已构建 executable 的 stream capture、instantiate 和
   replay。

muDNN 没有 cuDNN Frontend Graph 对等接口不构成阻塞。单节点 FlagDNN Graph 可映射
为一个 muDNN primitive；融合或多节点 Graph 可映射为严格等价的 primitive
sequence。MUSA runtime Graph 测试必须在首次 JIT 和 autotune 已完成后开始 capture。

## 6. 目录与职责

建议目录如下。文件可以在实现计划中按测试切片逐步加入，但最终职责不得跨越边界：

    backends/mthreads/
    ├── CMakeLists.txt
    ├── cmake/
    │   ├── MthreadsDependencies.cmake
    │   └── MthreadsTritonJIT.cmake
    ├── backend.cpp
    ├── context.cpp
    ├── context.hpp
    ├── error.cpp
    ├── error.hpp
    ├── artifact.cpp
    ├── artifact.hpp
    ├── autotune.cpp
    ├── autotune.hpp
    ├── compiler.py
    ├── compiler_identity.py
    ├── compiler_graph.py
    ├── compiler_tensor.py
    ├── compiler_nn.py
    ├── execution_plan.py
    ├── engines/
    │   ├── engine.cpp
    │   ├── engine.hpp
    │   └── libtriton_jit.cpp
    ├── kernels/
    │   ├── registry.json
    │   └── platform_*.py       # 仅在确有需要时增加 MUSA override
    ├── tuning/
    │   └── mthreads.yaml
    └── validation/
        ├── CMakeLists.txt
        ├── musa_driver.cpp/.hpp
        ├── tensor_io.cpp/.hpp
        ├── mudnn_reference.cpp/.hpp
        ├── functional/
        │   ├── adapter.cpp
        │   ├── pointwise.cpp
        │   ├── layout.cpp
        │   ├── reduction.cpp
        │   ├── matmul.cpp
        │   ├── convolution.cpp
        │   ├── normalization.cpp
        │   ├── attention.cpp
        │   └── composite.cpp
        ├── benchmark/
        │   ├── runner.cpp
        │   ├── flagdnn_provider.cpp/.hpp
        │   ├── mudnn_provider.cpp/.hpp
        │   └── comparable_cases.json
        ├── integration/
        │   └── contract_*.cpp/.py
        └── run_tests_adapter.py

compiler.py 是公共 provider loader 的唯一入口，其他 compiler 模块只用于控制复杂度。
安装时必须一起复制，并进入 compiler identity。validation 整体不安装。

允许直接复用但不修改的中立组件包括：

- backends/backend_api.h；
- backends/autotune_policy.cpp/.hpp；
- src/runtime/json.cpp 和 sha256.cpp；
- compiler/flagdnn_codegen 下的 provider loader、kernel registry 和资源解析；
- kernels/common；
- tests/common 和 benchmark/common；
- cmake/Operators.cmake 中的算子 manifest。

禁止 include、import 或复制后同步下列实现：

- backends/nvidia；
- backends/hygon、backends/ascend、backends/iluvatar 等其他平台私有代码；
- 其他平台 validation/reference。

可以阅读它们理解契约，但 mthreads 必须拥有自己的实现和测试。

## 7. CMake、依赖边界与安装闭包

### 7.1 接入现有构建架构

仓库根 CMake 和 backends/CMakeLists.txt 已按 FLAGDNN_BACKENDS 动态进入
backends/<platform>。因此实现只需新增 backends/mthreads/CMakeLists.txt：

- 调用现有 flagdnn_add_backend_plugin(mthreads INSTALL ...)；
- production target 名为 flagdnn_backend_mthreads；
- 安装 ABI 主版本为 2，得到 libflagdnn_backend_mthreads.so.2；
- 当只配置 mthreads 时，FLAGDNN_DEFAULT_BACKEND=auto 可以沿用现有规则选中它；
- validation/CMakeLists.txt 使用现有
  flagdnn_register_functional_suite 和 flagdnn_register_benchmark_suite；
- 不修改根 CMakeLists.txt、backends/CMakeLists.txt、tests/CMakeLists.txt 或
  benchmark/CMakeLists.txt。

如果实现时发现根构建无法进入一个已有 backends/<name>/CMakeLists.txt，应先把它
作为架构缺陷单独报告，不能把临时特判混进 mthreads 变更。

### 7.2 变量命名和发现规则

所有用户可配置项使用 mthreads 命名空间，避免污染多 backend 构建：

- FLAGDNN_MTHREADS_MUSA_ROOT；
- FLAGDNN_MTHREADS_MUDNN_ROOT，仅 validation 使用；
- FLAGDNN_MTHREADS_TRITON_JIT_ROOT；
- FLAGDNN_MTHREADS_TRITON_JIT_DIR；
- FLAGDNN_MTHREADS_BUNDLE_TRITON_JIT，默认开启；
- FLAGDNN_MTHREADS_REQUIRE_REFERENCE_COVERAGE，正式验收默认开启。

MUSA_HOME 可以作为 FLAGDNN_MTHREADS_MUSA_ROOT 的环境默认值。mthreads resolver
不得读取 CUDA_HOME，也不得依赖其他 backend 设置的全局 TritonJIT_DIR、
LIBTRITON_JIT_ROOT 或 Python_EXECUTABLE。

查找顺序必须确定：

1. 显式 mthreads CMake cache 变量；
2. MUSA_HOME 或平台专属环境变量；
3. 经过验证的标准安装位置；
4. 找不到即失败。

resolver 必须使用 NO_DEFAULT_PATH 或等价的受控发现方式，拒绝“找到任意同名库即可”。
同一 CMake build 同时启用多个 backend 时，mthreads 不能复用一个已经指向 CUDA/HIP
实现的 TritonJIT imported target。应在 mthreads 子目录内捕获并校验自己的 imported
location、include、compile definition 和 link interface，再链接到平台私有 target。

### 7.3 production 与 validation 依赖隔离

| 组件 | MUSA driver/runtime | libtriton_jit | Python/Torch/Triton | muDNN |
| --- | --- | --- | --- | --- |
| flagdnn core | 否 | 否 | 否 | 否 |
| mthreads plugin | 是 | 是 | JIT 明确需要的闭包 | 否 |
| mthreads compiler resources | 否 | helper scripts | 是 | 否 |
| functional adapter | 是 | 通过 DUT | 编译环境 | 是 |
| benchmark adapter | 是 | 通过 DUT | 编译环境 | 是 |
| installed consumer gate | 是 | 是 | 是 | 仅 GPU reference consumer |

必须提供自动 dependency-boundary 测试：

- readelf/ldd 检查 core 不出现 MUSA、muDNN、Torch 或 libtriton_jit；
- production plugin 不出现 libmudnn；
- validation adapter 必须真实依赖 libmudnn，防止 reference 被意外替换；
- 安装树中不允许出现源码树或 build tree 的绝对 RPATH；
- 不允许空 RPATH 元素、当前工作目录或 plugin 根目录被当作 JIT 搜索目录。

### 7.4 私有 JIT 安装布局

为避免不同 backend 的同名 libtriton_jit.so 相互覆盖，mthreads 使用平台私有布局：

    <prefix>/lib/libflagdnn_backend_mthreads.so.2
    <prefix>/lib/flagdnn/mthreads/<实际 libtriton_jit SONAME>
    <prefix>/share/flagdnn/backends/mthreads/*.py
    <prefix>/share/flagdnn/backends/mthreads/flagdnn_mthreads_environment.json
    <prefix>/lib/flagdnn/share/triton_jit/scripts/*
    <prefix>/share/flagdnn/backends/mthreads/kernels/*
    <prefix>/share/flagdnn/backends/mthreads/tuning/*

plugin 的安装 RPATH 精确为 $ORIGIN/flagdnn/mthreads。实际文件名以
TritonJITConfig.cmake 导出的库和 ELF SONAME 为准，不能假定一定是未版本化
libtriton_jit.so。

只私有安装 libtriton_jit 本身及其 helper resources；MUSA SDK、Python、Torch 和
torch_musa 作为已记录版本的外部依赖。environment.json 至少包含：

- schema version；
- backend=MUSA 和 compiler backend=mtgpu/musa 的规范化标识；
- JIT library realpath、SONAME、SHA-256 和相对安装路径；
- scripts 清单及 SHA-256；
- Python executable、Python ABI；
- torch、torch_musa、Triton 版本和包根目录 hash；
- MUSA toolkit、driver/runtime ABI；
- 生成该文件的 FlagDNN version。

build-tree plugin 也使用独立的 flagdnn/mthreads 私有 JIT 目录，确保 build 和 install
测试覆盖相同加载模型。

MThreads 在生成私有 `standalone_compile.py` 时适配 descriptor 签名，以及缺少
`translate_llvmir_to_mubin` 的 Triton 编译接口。旧接口不可调用时，只有缓存中按
loader 优先级选中的 `.mubin`、`.o`、`.so` 或 `.llir` 非空且与本次
`triton.compile()` 返回的 `ccinfo.asm` 内容一致，才跳过额外转换；没有匹配产物则
明确报错。`.llir` 沿用 MUSA loader 的驱动 JIT 路径。旧接口仍可调用时保留原流程。
这些适配只写入 FlagDNN 的私有副本，不修改外部 `libtriton_jit` 源码，也不改变其他
backend 的 helper。

私有复制前必须证明 JIT 的 helper 解析可重定位：library 应相对自身或显式的私有
resource root 找到 scripts，不能继续硬编码 /usr/local/share 或原构建 prefix。被复制
JIT 的 DT_RPATH/DT_RUNPATH 也不能含源码树、build tree 或绝对虚拟环境路径；不满足时
应以正确 install prefix 重建或在 libtriton_jit 上游补齐可重定位支持，不能用全局
PYTHONPATH、LD_LIBRARY_PATH 或保留原安装树来伪造安装态通过。

平台私有目录只解决安装文件覆盖，不自动解决同一进程加载两个同 SONAME、不同 backend
JIT 的 ELF 复用问题。多 backend SDK 必须通过 global-state/SONAME collision gate；
可靠方案是上游提供 backend-qualified SONAME，或证明所用 loader isolation 不会串用。
在该门禁通过前，只能声明单 JIT backend 进程支持，不能宣称进程内多 backend 共存。

### 7.5 安装态 consumer

自动 integration.mthreads.installed_consumer 必须：

1. 安装到隔离的临时 prefix；
2. 清除 FLAGDNN_BACKEND_ROOT、FLAGDNN_KERNEL_SOURCE_ROOT、
   FLAGDNN_TUNING_ROOT、PYTHONPATH、PYTHONHOME 和全部 mthreads resource override；
3. 使用安装态 FlagDNNConfig.cmake 配置外部 C 和 C++ consumer；
4. C++ consumer 使用 flagdnn::Handle("mthreads", 0) 构建 Add Graph；
5. 默认-backend consumer 使用 Handle() 验证安装时的默认选择；
6. 从安装资源生成 manifest/source/autotune selection；
7. 在真实 MUSA stream 上执行，并与 muDNN Binary Add 比较；
8. 检查进程实际映射的 libtriton_jit 位于安装 prefix 的平台私有目录；
9. 检查 compiler identity 中的每个依赖都来自安装树或已声明外部 SDK；
10. 确认没有从源码树、构建树或调用者当前目录偷取资源。

validation target 本身不安装，但 installed-consumer integration test 从 build tree
驱动上述隔离流程。

## 8. mthreads compiler 与 kernel 选择

### 8.1 provider 接口

compiler/flagdnn_codegen/main.py 会根据请求中的 backend=mthreads 加载
backends/mthreads/compiler.py。该模块至少实现现有 provider 协议要求的：

- compiler_identity(target, execution_engine)；
- compiler_identity_dependencies(target, execution_engine)，若 provider 协议支持；
- compile_request(request_path, output_directory, execution_engine)。目标设备名和
  指纹从 request 中读取并与 compiler_identity 结果交叉校验，不额外扩展通用
  provider 调用签名。

实现必须消费公共 versioned Graph IR，不能读取 C++ Graph 对象、validation case 或
muDNN descriptor。输入字段、dtype、shape、stride、attribute、UID、virtual tensor
和 dependency 都要先完整校验，再创建内部 plan。

### 8.2 compiler 内部分层

建议使用小型中间结构，而不是把全部逻辑堆进 compiler.py：

- TensorSpec：dtype、dims、strides、alignment、UID、virtual/external；
- NodeSpec：公共 operation、输入输出 UID、attributes；
- KernelStagePlan：source、function、constants、runtime arguments、grid 和候选；
- WorkspaceRegion：synthetic UID、offset、size、alignment、lifetime；
- ExecutionPlan：有序 stage、显式 dependency、binding UIDs 和 workspace size。

compiler_graph.py 负责 Graph/node validation 与 dependency；
compiler_tensor.py 负责 pointwise、layout、reduction、matmul 的参数化；
compiler_nn.py 负责 convolution、normalization、attention 和 composite expansion；
execution_plan.py 负责 stage DAG、workspace packing、argument ABI 和 manifest
serialization。compiler.py 只负责 provider 入口、错误上下文和原子输出。

### 8.3 common kernel 与 platform override

选择顺序沿用现有 kernel_registry.py：

1. 查找 backend=mthreads 的 platform candidate；
2. 未注册 platform candidate 时使用 common candidate；
3. 两者都不存在则编译失败。

第一批工作不是假定 common kernel 可用，而是对 kernels/common 做真实 MUSA
compatibility sweep。每个 registry entry 至少覆盖：

- module 可导入；
- entry function 存在且 signature 匹配；
- MUSA Triton compile 成功；
- 最小、典型和边界 shape launch 成功；
- contiguous、broadcast 和测试 manifest 中的 strided layout；
- num_warps、num_stages、block size、shared memory 在设备限制内；
- 结果通过独立 muDNN reference；
- default 配置和至少两个有效 autotune candidate。

common kernel 不允许包含 CUDA-only import。当前审计显示 CUDA libdevice、SM90、TMA
等代码位于 backends/nvidia/kernels，而 kernels/common 没有直接 CUDA import；
这只是静态初筛，不能替代实机 compile/launch。

需要 MUSA override 时，registry entry 必须写明：

- ownership=platform；
- backend=mthreads；
- 被覆盖 operation；
- provider、source、functions；
- tuning source/table/key/strategy；
- override 原因和对应 regression case。

禁止在 override compile 失败后回退 common，因为这会让 source identity、性能结果和
capability 随异常路径变化。

### 8.4 多 stage、workspace 与依赖

一个 Graph node 可以展开为多个 kernel stage，例如：

- reduction 的 partial/reduce；
- convolution 的 im2col/matmul/reorder；
- convolution backward 的 split/reduce；
- batchnorm 的统计量和输出；
- SDPA backward 的 delta、dQ、dK、dV；
- FP8 attention 的 amax 初始化、主计算和归约；
- composite 的多个 primitive stage。

所有内部 tensor 使用与外部 Graph UID 不冲突的 synthetic UID。workspace packer 必须：

1. 分离公共 Graph virtual tensor 与 provider-local temporary；
2. 按每个 region 的 alignment 向上取整；
3. 检查加法、乘法和 offset 计算溢出；
4. 保证 region 不重叠且不超出总 workspace；
5. 保证 mutable global/profile scratch 不被不同并发执行共享；
6. 把 stage dependency 显式写入 execution program；
7. 只在最后一个生产 Graph output 的 stage 完成后暴露该 output。

runtime scalar 使用明确的 i32 或 f32 ABI，不能把 Python int 大小、C++ size_t 或
设备指针布局作为隐式协议。每个 scalar 的名称、类型、bit-exact value 和 argument
位置都进入 manifest 和 identity。

### 8.5 编译输出的原子性

compiler 在临时目录生成全部 source、metadata 和 manifest，完成 hash 与自校验后再
原子发布。失败时不能留下可被 cache 命中的半成品。输出目录中未被 manifest 引用的
可执行 source/binary 视为错误；manifest 引用的文件必须有固定 size 和 SHA-256。

compiler 不信任用户可控路径：

- relative path 不能为绝对路径；
- 不能包含 ..；
- canonical path 必须仍在 artifact/resource root 内；
- symlink 解析不能逃逸；
- 文件数量和大小有上限；
- JSON 拒绝重复 key、非有限数和类型混淆。

## 9. Target、identity 与 cache

### 9.1 target fingerprint

get_target_fingerprint() 输出不超过
FLAGDNN_BACKEND_MAX_TARGET_FINGERPRINT 的文件系统安全字符串。建议格式为：

    musa-{architecture}-cc{major}{minor}-w{warp-size}

fingerprint 只描述决定 artifact 可执行性的目标能力，至少覆盖：

- 实际设备 architecture/ISA；
- compute capability；
- warp size；
- codegen 会使用的 feature bits；
- 会限制合法 grid、block、shared memory 或 instruction 的 device limits。

fingerprint 不包含 device UUID、PCI 地址、ordinal 或 driver patch version，使同一目标
架构的设备可以安全共享编译 artifact。context 另建私有 device_identity，覆盖 target、
driver/runtime 和稳定 UUID；它只用于 autotune measurement identity、运行诊断和
设备特定 cache，不作为 Graph artifact target。

设备 marketing name 只用于诊断，不作为唯一 key。target 获取失败必须返回明确的
runtime error，不能退回 generic-musa。

### 9.2 compiler identity

compiler identity 覆盖所有可能改变 artifact 的输入：

- FlagDNN version、Graph schema、artifact schema、execution-program version；
- execution engine；
- mthreads compiler 全部 Python 模块；
- compiler/flagdnn_codegen entry 和 kernel registry loader；
- 实际选中的 common/platform registry；
- 实际物化的 kernel source；
- 实际选中的 tuning YAML/table；
- Python ABI、torch、torch_musa、Triton；
- libtriton_jit library、header、helper scripts 和 build identity；
- MUSA toolkit/compiler version；
- target fingerprint。

identity dependency 必须由文件内容和稳定 metadata 计算，不能只依赖 mtime。公共
compiler runner 的“读取前后 snapshot 不一致即重试/失败”机制继续生效。

### 9.3 autotune measurement identity

autotune winner 比 artifact identity 还要额外区分：

- device UUID 或等价稳定标识；
- driver/runtime；
- 时钟/功耗策略若可查询；
- candidate 列表及顺序无关的 canonical identity；
- warmup、repetition 和计时策略版本；
- dependency replay plan；
- workspace layout。

环境变化后旧 winner 必须 miss；损坏或字段不完整的 selection cache 必须删除或拒绝，
不能选第一个候选继续运行。cache 写入使用同目录临时文件、fsync 和原子 rename，并在
多进程竞争时保证最终文件完整。

## 10. Artifact 与 backend ABI

### 10.1 ABI 选择

mthreads 使用 flagdnnBackendApiV2：

- backend_name 精确为 mthreads；
- 导出 flagdnnBackendGetApiV2；
- create_context(device_ordinal)；
- get_target_fingerprint()；
- create_executable()；
- execute()；
- 对应 destroy 函数和 thread-local get_last_error()。

公共 C++ 调用方式为：

    flagdnn::Handle handle("mthreads", 0);

不能为了 mthreads 修改 flagdnnBackend_t enum。默认 backend 由现有
FLAGDNN_DEFAULT_BACKEND=mthreads 配置完成。

### 10.2 manifest 必需语义

mthreads artifact parser 独立实现，但保持与现有 execution-program 语义一致。至少
校验：

- schema_version、artifact_kind=flagdnn_execution_program；
- backend、target、FlagDNN version；
- request SHA-256、compiler identity；
- engine=libtriton_jit；
- external binding UID 的唯一性和完整性；
- workspace size、alignment 和每个 region；
- stage 数量及上限；
- stage dependency 是无环且引用合法；
- source path、function name 和 source hash；
- variant ID 唯一；
- grid、block、num_warps、num_stages、shared memory；
- runtime argument ABI、tensor storage size、scalar bit pattern；
- autotune schema、candidate identity、warmup、repetitions 和 selection path。

parser 必须独立复核 Graph IR 与 manifest 的关键对应关系，不能因为两者由同一个
compiler 生成就信任：

- operation/node 数与 stage ownership；
- external input/output UID；
- dtype、rank、dims、strides、alignment；
- runtime scalar；
- virtual tensor 和 workspace；
- dependency 顺序；
- target/compiler/request identity。

### 10.3 错误映射

MUSA 和 JIT 错误映射必须保留原始 status、operation 和上下文：

| 情况 | backend result |
| --- | --- |
| null、重复 UID、workspace 不足、错误 stream | INVALID_VALUE |
| device allocation 失败 | ALLOC_FAILED |
| 已验证的 Graph/case capability 不支持 | NOT_SUPPORTED |
| MUSA launch/module/event/context 错误 | RUNTIME_ERROR |
| compiler、manifest、source、JIT compile/load 错误 | COMPILATION_FAILED |
| 不变量破坏或未知异常 | INTERNAL_ERROR |

get_last_error() 的消息不参与控制流；测试和 autotune 必须依据类型化 result/status，
不得匹配错误字符串。


## 11. MUSA runtime 与 libtriton_jit

### 11.1 context 和 stream

context.cpp 使用 RAII 管理指定 device ordinal 对应的 MUSA primary context，且必须与
MUSA runtime、调用者 musaStream_t 和 libtriton_jit 使用同一设备上下文。需要实机
验证 runtime stream 与 Driver MUstream 的二进制和上下文互操作，不能只靠
reinterpret_cast 编译成功作为证据。

create_context() 应完成：

- MUSA driver/runtime 初始化；
- device ordinal 范围检查；
- primary context retain；
- 读取目标与 device limits；
- 建立不可变的 target/environment snapshot。

execute() 接收的 native_stream 解释为 musaStream_t。它必须：

- 验证 stream 非空以及属于 executable 的设备；
- 在该 stream 上按 execution program 顺序排队全部 stage；
- 不切换到默认 stream；
- 不创建内部执行 stream；
- 不调用 musaDeviceSynchronize 或隐式全设备同步；
- 返回前不等待 kernel 完成。

多线程和多 stream 使用同一个 executable 时，executable 元数据和 winner 必须只读。
每次 launch 的 argument 指针数组使用固定上限的栈存储或预先分配且线程安全的存储，
不能在 execute() 首次或按执行次数增长。所有可变 device scratch 来自调用者传入的
workspace，不能使用一个跨 stream 共享的持久 scratch buffer。

### 11.2 create_executable 与 execute 分界

| 操作 | create_executable | execute |
| --- | --- | --- |
| 读取并校验 artifact | 允许 | 禁止 |
| Python/Triton JIT compile | 允许 | 禁止 |
| module/function load | 允许 | 禁止 |
| candidate autotune | 允许 | 禁止 |
| 临时 stream/event/allocation | 允许，返回前释放 | 禁止 |
| 固化 winner 和 immutable launch metadata | 必须 | 只读取 |
| 在 caller stream launch | 仅 probe 时使用私有 stream | 必须 |
| synchronize | 仅私有 build/autotune stream | 禁止 |
| 持久或随执行次数增长的分配 | 禁止 | 禁止 |

create_executable() 返回前必须使所有后续 execute 路径完全 warm，包括 Python runtime、
Triton extension、JIT cache、module/function lookup 和任何 lazy initialization。
Graph capture 测试会在 capture 期间检测首次初始化和非法 API。

### 11.3 libtriton_jit backend identity

MUSA package 的 CMake backend 名是 MUSA，而 libtriton_jit 内部 Triton backend 名
可能是 mtgpu，当前 Triton target backend 名是 musa。三者是不同层级，必须通过
environment identity 明确记录，不能只比较一个字符串。

mthreads engine 只接受经 CMake 验证的 MUSA JIT target。正式
`integration.mthreads.jit_add` 与 installed-consumer 覆盖：

1. 用选中的 Python、Triton 和 helper script 编译 vector Add；
2. 生成目标架构可加载的 MUSA artifact，当前实现通常是 .mubin；
3. 通过 libtriton_jit 创建 function；
4. 在调用者创建的非默认 musaStream_t 上 launch；
5. 用该 stream 上的 event 等待并验证结果；
6. 在另一个 stream 重复，确认没有落到默认 stream；
7. 销毁 function/executable 后重复至少数千次，记录 host/device/module 资源增长；
8. 在安装 tree 的隔离 consumer 中重复以上流程。

如果 Python C extension 需要全局 Python symbols，mthreads engine 可以独立实现受控的
RTLD_GLOBAL promotion，但不能 include NVIDIA 的实现。该行为必须只执行一次、验证
加载的是配置时记录的 Python runtime，并纳入 installed-consumer 测试。

### 11.4 module 生命周期

当前 MUSA libtriton_jit 可能使用进程级 module cache。实现前必须确定并记录：

- cache key 是否覆盖 target、source、compile option 和 entry；
- 相同 key 是否只加载一次；
- executable destroy 是否释放 function wrapper；
- module 是否能安全 unload；
- 不能 unload 时 cache 是否有明确上限或可回收策略；
- fork、并发 build 和进程退出行为。

允许“进程生命周期 module cache”作为明确契约，但不允许 key 缺失导致错误复用，也
不允许对无限数量动态 Graph 无界增长。生命周期门禁失败时优先修复 libtriton_jit；
FlagDNN 不通过隐藏全局 singleton 或跳过 destroy 测试规避。

### 11.5 binding、workspace 与 alias

execute() 必须在 launch 前完成纯 host 侧验证：

- binding_count 与 artifact 外部 UID 集合完全一致；
- UID 无重复、无缺失、无额外项；
- device pointer 非空并符合 tensor alignment；
- workspace_size 足够；
- 非零 workspace base 满足 256-byte alignment 或 manifest 中更严格要求；
- binding 不与禁止 alias 的 workspace region 重叠；
- Graph contract 未声明合法的 input/output alias 时不自行支持。

允许的 in-place 或 alias 语义只能来自公共 Graph/operator contract。validation 必须
包含合法 alias、非法 alias、错位 pointer、workspace 少一个字节、错误 UID 和重复 UID
的 mutation case。

### 11.6 MUSA runtime Graph capture

integration.mthreads.graph_capture 的顺序固定：

1. 创建 Graph、开启 autotune 并 build executable；
2. 在 capture 外执行一次 warmup，并同步该 caller stream；
3. 记录 artifact、JIT compile 和 autotune cache 的文件数量；
4. musaStreamBeginCapture；
5. 调用同一 executable 的 execute()；
6. musaStreamEndCapture；
7. musaGraphInstantiate；
8. 多次 musaGraphLaunch 到原 caller stream；
9. 用 stream event 等待并验证结果；
10. 检查 capture/replay 前后没有新增 JIT、artifact 或 autotune 文件；
11. 检查没有额外 allocation、module load 或同步；
12. 销毁 graph exec、graph、event 和 stream。

capture 失败如果来自 FlagDNN execute 中的首次 JIT、allocation 或非法同步，必须硬
失败。只有明确由 MUSA runtime 返回并经白名单确认的“该 SDK 不支持此 capture
操作”才可记录 capability；最终“完整适配”验收仍要求该 gate 通过。

## 12. Autotune 设计

### 12.1 build-time 流程

autotune=true 时，每个可调 stage 在 create_executable() 中执行：

1. 解析并验证全部 candidate；
2. 为所有 external tensor、workspace region 和依赖 stage 准备独立 tuning buffer；
3. 初始化输入和 canary，不能使用用户 execution binding；
4. compile/load 每个 candidate；
5. 在独立非默认 MUSA stream 上按真实 dependency plan warmup；
6. 使用同一 stream 上的 MUSA event 计时；
7. 每个 candidate 执行相同 warmup/repetition；
8. 检查 launch error、输出有限性以及 canary；
9. 选择 median 最小的合法 candidate；
10. 原子写入 winner selection；
11. 释放临时 allocation、stream 和 event；
12. 把 winner function 与不可变 launch metadata 存入 executable。

多 stage plan 只对当前待调 stage 计时，但其输入所需 dependency 必须在计时窗口外
按真实顺序重放。不得把依赖成本错误计入某一个 candidate，也不得给 candidate 未初始化
输入。

### 12.2 candidate 错误策略

候选淘汰只接受类型化、可证明与单个 launch configuration 有关的资源错误。初始白名单
应保持最窄：

- MUSA_ERROR_LAUNCH_OUT_OF_RESOURCES；
- MUSA_ERROR_INVALID_RESOURCE_CONFIGURATION。

MUSA_ERROR_NO_BINARY_FOR_GPU、MUSA_ERROR_INVALID_IMAGE、MUSA_ERROR_ILLEGAL_ADDRESS、
MUSA_ERROR_LAUNCH_FAILED、JIT compile error、source/signature/hash 不一致和未知异常
全部硬失败，因为它们通常表示 target、compiler、artifact 或 kernel 缺陷，而非普通
tuning candidate 不优。

如果未来 SDK 返回新的候选级 capability status，必须先添加可复现测试和设计更新，
不能按错误消息文本扩展白名单。若所有 candidate 都被淘汰，create_executable()
返回失败，不得选择未经测量的 default。

### 12.3 正确性和稳定性

autotune winner 除速度外还必须满足：

- 输出与 default candidate/muDNN oracle 一致；
- 输入、padding 和只读 workspace 未修改；
- 重复测量结果有限且非零；
- 不留下 pending asynchronous error；
- 同 identity 冷启动和热启动选择一致；
- candidate 列表换序不改变 identity 或 winner；
- cache corruption、截断、重复 key 和并发写入被拒绝；
- FLAGDNN_PRINT_AUTOTUNING 只影响日志，不影响选择。

正式 functional case 同时覆盖 autotune=false 和 true。benchmark 使用 autotune=true，
但计时不包含 Graph build、JIT、autotune 或首次 module load。

## 13. 算子覆盖设计

### 13.1 单一算子目录

cmake/Operators.cmake 是算子 manifest 唯一来源。当前有 61 个 functional operator，
其中 57 个同时进入 benchmark；4 个 attention operator 目前只做 functional。

| 家族 | manifest operator | production 目标 | muDNN reference 目标 |
| --- | --- | --- | --- |
| unary pointwise | abs, ceil, cos, elu, erf, exp, floor, gelu, gelu_approx_tanh, identity, leaky_relu, log, logical_not, neg, reciprocal, relu, rsqrt, sigmoid, sin, softplus, sqrt, swish, tan, tanh | common unary 或 MUSA override | musa::dnn::Unary；无单 primitive 时用等价 Unary/Binary sequence |
| binary pointwise | add, cmp_eq, cmp_ge, cmp_gt, cmp_le, cmp_lt, cmp_neq, div, logical_and, logical_or, max, min, mod, mul, pow, sigmoid_backward, sub | common binary 或 MUSA override | musa::dnn::Binary 或等价 primitive sequence |
| scalar pointwise | scale | unary/scalar stage | Unary scale 或 Binary 与 scalar tensor |
| ternary | binary_select | common ternary 或 override | musa::dnn::Ternary |
| layout | reshape, slice, transpose | common layout copy/view-aware stage | Permute、strided Tensor view 加 identity；必须保持逻辑索引 |
| reduction | reduction | sum/avg/mul plan | musa::dnn::Reduce |
| matmul | matmul | common matmul 起步，MUSA tuned override | musa::dnn::MatMul 或 BatchMatMul |
| convolution | conv_fprop, conv_dgrad, conv_wgrad | 1D/2D/3D、多 stage | musa::dnn::Convolution forward/backward-data/backward-filter |
| normalization | batchnorm, batchnorm_inference, layernorm, rmsnorm | common normalization 或 override | musa::dnn::BatchNorm、LayerNorm、RMSNorm |
| attention | sdpa, sdpa_backward, sdpa_fp8, sdpa_fp8_backward | common attention 或 MUSA override | musa::dnn::ScaledDotProductAttention |
| composite | add_square, conv_bias_relu | 显式多节点/stage Graph | Binary/Convolution/Unary 的等价 sequence |

表中“reference 目标”不是已经确认的 capability。每一个公共 case 都必须经过当前
muDNN header/runtime 和真实设备执行探测后，才能加入 comparable case catalog。

### 13.2 每类语义门禁

pointwise：

- dtype conversion、NaN、Inf、signed zero 和整数/布尔语义；
- broadcast 与最多公共 case 要求的 rank/stride；
- div、mod、pow 的边界行为；
- comparison 输出 dtype；
- logical 的“非零为真”定义；
- GELU exact 与 tanh approximation 不可混淆；
- sigmoid_backward 的输入含义和公式。

layout：

- reshape 是 view 还是 materialized copy由公共 Graph contract 决定；
- transpose permutation 必须完整且无重复；
- slice 的 start/stop/step、空维度和非连续输出；
- physical padding 不能被覆盖；
- 负 stride 若公共 contract 不允许，应在 Graph/compiler validation 统一拒绝，而不是
  mthreads 私下改变语义。

reduction：

- ADD、AVG、MUL 分别映射；
- axis 排序、重复、负轴规范化；
- keep-dim、空 reduction、accumulation dtype；
- 非连续输入和多输出元素；
- FP16/BF16 的累加精度。

matmul：

- rank、batch broadcast、transpose/stride；
- M/N/K 为 1 和非 tile 整数倍；
- FP16/BF16/FP32 accumulation 与输出转换；
- 任意合法 alignment；
- muDNN algorithm/workspace 选择不得改变结果 contract。

convolution：

- 1D、2D、3D；
- NCHW/NCDHW 与仓库实际 case layout；
- group、stride、dilation；
- pre-padding 与 post-padding 分别校验；
- cross-correlation 与 convolution mode；
- fprop、dgrad、wgrad 的 tensor role；
- 非对称 padding 如果 muDNN 只有对称接口，必须用严格等价的 muDNN sequence 表达；
  表达不了就记录 reference gap，不能悄悄改成对称 padding；
- conv_bias_relu reference 优先使用 Convolution + Binary bias + Unary ReLU，避免不同
  fusion policy 改变 oracle。

normalization：

- normalized axes、epsilon、scale/bias；
- training 与 inference；
- saved mean、inverse variance、running mean/variance 等全部输出；
- variance 定义和 accumulation precision；
- 极小 variance、常量输入、非连续 layout；
- 不允许只比较主输出而忽略统计量。

attention：

- Q/K/V layout、head 数、GQA/MQA broadcast；
- scale、bias、causal/bottom-right band 和 alignment；
- sequence length 边界与 head dimension；
- forward stats 与 backward delta；
- deterministic 属性；
- FP8 descale/scale、amax 的方向、dtype 和全部输出；
- muDNN RunFlash/RunFlashBwd/Math 等入口只能在语义完全等价时选择；
- 不允许用非 FP8 reference 验证 FP8 scaling/amax 后宣称 FP8 case 通过。

composite：

- add_square 必须验证中间 virtual tensor 不暴露且 workspace 正确；
- conv_bias_relu 验证 node dependency、bias broadcast、activation 和最终输出；
- reference sequence 使用独立中间 buffer；
- Graph node 执行顺序不能从输出碰巧正确推断，必须有 artifact DAG mutation test。

### 13.3 capability 的三层状态

每个 case 分别维护：

1. production_status：FlagDNN mthreads 是否能 build 和 execute；
2. reference_status：当前 muDNN 是否能严格等价执行；
3. performance_status：两侧是否都能在同一公平计时模式下比较。

reference 不支持不能改变 production_status。benchmark capture 不支持也不能抹掉
functional correctness 结果。正式报告要分别统计三种覆盖率，不能只给一个 passed
数字。

开发期间可以输出结构化 SKIP 定位 reference gap；最终“完美适配”要求当前 manifest
所有 functional case 的 production_status=passed 且 reference_status=comparable。
任何 reference SKIP 都会让正式全量验收失败，即使 CTest 本身把单项 SKIP 视为成功。

## 14. muDNN functional validation

### 14.1 reference 原则

muDNN C++ API 是唯一默认数值 oracle。禁止：

- 调用 FlagDNN kernel 生成 expected；
- 用 torch_musa eager 作为最终 expected；
- 用同一 mthreads Triton kernel 的另一个 launch 作为 expected；
- reference 失败后回退 host 公式并仍标记为 mudnn；
- 为提高覆盖率静默改 shape、dtype、layout 或 attribute。

如果 muDNN 无单一 primitive，可使用严格等价的 muDNN primitive sequence。sequence
的每一步、临时 tensor 和 workspace 都由 reference provider 独立拥有，并在日志中
标记 sequence=true。

### 14.2 capability 判断

capability 判断分三步：

1. 静态语义检查：当前 case 能否由已知 muDNN primitive/sequence 精确表达；
2. descriptor/build probe：创建 descriptor、查询 workspace、选择 algorithm；
3. runtime probe：用独立 buffer 在真实 stream 上至少执行一次并检查 status。

只有精确的 muDNN NOT_SUPPORTED status 可以变成 reference SKIP。
INVALID_PARAMETER、NOT_INITIALIZED、ALLOC_FAILED、INTERNAL_ERROR、ARCH_MISMATCH、
EXECUTION_FAILED 及未知 status 全部硬失败。不能把“可能是 capability”作为依据。

结构化记录至少为一行：

    [SKIP][mudnn] op=<manifest-op> case=<case-id> reason=<canonical-reason> mudnn=<header/runtime> musa=<toolkit/driver> target=<fingerprint> dtype=<dtype> layout=<layout> shape=<shape>

字段需要可靠转义或使用现有 runner 支持的 JSON 扩展，case ID 必须与公共 case 唯一
对应。重复、缺字段、未知 op/case 或泛化 reason 都由 run_tests_adapter.py 判失败。

### 14.3 数据和比较

每个 case 的执行顺序：

1. 从公共 case 和 seed 生成 host logical input；
2. 分别分配 DUT 和 reference physical buffer；
3. 用 mthreads tensor_io 写入相同逻辑值和 padding canary；
4. 构建 muDNN reference；
5. 构建 FlagDNN Graph，分别覆盖 autotune=false/true；
6. 在显式非默认 stream 上执行 reference 和 DUT；
7. 只在测试边界同步各自 stream；
8. 复制所有输出和相关输入回 host；
9. 使用公共 case tolerance 比较 logical element；
10. 检查 padding canary、只读输入和未使用 workspace；
11. 释放资源并检查 pending MUSA error。

DUT 和 reference 的执行先后要在 case 间交替或隔离，避免温度、cache 和异步错误总是
偏向一方。数值比较报告首个错误 element、logical index、physical offset、actual、
expected、absolute/relative error 和 tolerance。

### 14.4 Graph 专项 functional tests

除公共单算子 tests 外，至少包含：

- ReLU -> virtual tensor -> Add 多节点 Graph；
- Add -> reduction 多 stage Graph；
- convolution -> bias -> ReLU composite；
- normalization 的多输出 Graph；
- SDPA backward 或 FP8 attention 的多 stage Graph；
- 同一 compiled Graph 使用两组 binding 重放；
- 两条非默认 stream 上独立 workspace 并发执行；
- 合法与非法 alias；
- workspace size/alignment 边界；
- node、UID、dependency、source hash 和 runtime scalar mutation rejection。

这些测试比较完整 Graph output 与 muDNN sequence，而不是只验证 kernel 能 launch。

## 15. Benchmark 设计

### 15.1 provider 与记录

mthreads benchmark adapter 使用两个 provider 名：

- flagdnn；
- mudnn。

每个可比 case 必须发出且只发出一对 JSONL record，case ID 完全相同。record 至少包含
公共 benchmark schema 要求的 provider、case、samples 和 median；平台扩展可以增加
MUSA/muDNN/target 信息，但不得改变公共解析字段。

run_tests_adapter.py 维护 versioned
backends/mthreads/validation/benchmark/comparable_cases.json：

- 只在真实 correctness + timing probe 通过后加入 case；
- 声明 platform=mthreads、suite=benchmark；
- metric=mudnn_median_us/flagdnn_median_us；
- operator 和 case 数量必须与内容一致；
- case 不能重复或归属到错误 operator；
- catalog 中每个被选 case 必须观察到两个 provider；
- 未声明的额外 pair 也要报告，避免 catalog 漂移。

### 15.2 正确性先于计时

每个 benchmark case 先执行第 14 节相同的 correctness、padding 和 input-mutation
门禁。只有通过后才能计时和发出 latency record。失败或 reference capability gap
不能留下单边 latency。

计时排除：

- Graph build；
- compiler subprocess；
- JIT compile/module load；
- autotune；
- buffer allocation 和初始化；
- host/device copy；
- 第一次 lazy initialization。

这些 cold-start 成本由独立 integration metric 报告，不能混进 steady-state speedup。

### 15.3 MUSA event 和 Graph batch

计时使用同一非默认 stream 上的 musaEventRecord 和 musaEventElapsedTime。为降低 host
launch overhead，可用 MUSA runtime Graph capture 批量 replay，但必须遵守公平性：

1. 两个 provider 都 capture-safe 时，分别 capture 等量执行并使用相同 batch size；
2. 任一 provider 不支持 capture 时，两侧都使用 direct event loop；
3. 不能只给 FlagDNN 使用 Graph batch；
4. capture 前两侧都完成 warmup；
5. start/stop event 包围相同数量的 steady-state execute；
6. 每轮后检查 pending error；
7. 采样数、warmup、batch size 来自公共 BenchmarkConfig 或平台统一策略。

### 15.4 性能判定

speedup 定义为：

    mudnn_median_us / flagdnn_median_us

数值大于 1 表示 FlagDNN 更快。分阶段门禁：

- 开发阶段：threshold 为空，只要求正确、成对、完整记录；
- 性能收敛阶段：对 comparable catalog 运行 --min-speedup 0.9；
- 最终“完美适配”验收：catalog 100% 配对，且每个 qualified case speedup >= 0.9。

最低值、中位数、几何均值和失败 case 都写入 summary。不能用总体平均掩盖单 case 低于
threshold，也不能把 reference SKIP 算作性能通过。


## 16. CTest 注册与 tools/run_tests.py

### 16.1 operator suite 注册

Phase 6 Release 配置中的 backends/mthreads/validation/CMakeLists.txt 必须把完整
manifest 传给公共注册函数，得到：

- functional.mthreads.<op>：61 个 manifest operator；
- benchmark.mthreads.<op>：57 个 manifest operator。

Phase 1–5 的垂直切片允许用显式 --ops 列表验证当期集合，但每阶段必须记录已注册集合，
不能开启 FILTER_REGISTERED_TESTS 隐藏缺项。Phase 6 Release 配置禁止只注册“已完成”
的子集；未实现 target、target 名拼错、CTest 不可发现或测试二进制缺失都必须在
configure/preflight 阶段失败，而不是被 runner 当作未选择。

每个 suite 输出且只输出一个公共 accounting marker：

    FLAGDNN_<OP>_FUNCTIONAL: PASS cases=<n> executed=<n> skipped=0
    FLAGDNN_<OP>_BENCHMARK: PASS cases=<n> executed=<n> skipped=0

开发阶段允许 executed + skipped = cases，但每个 skip 都必须有唯一结构化记录。最终
严格覆盖模式要求 skipped=0。

### 16.2 mthreads runner adapter

backends/mthreads/validation/run_tests_adapter.py 使用现有动态 hook，不修改
tools/run_tests.py。目标策略：

- DEFAULT_TIMEOUT=1800；
- PREFLIGHT_BY_DEFAULT=True；
- SUPPORTS_MIN_SPEEDUP=True；
- FILTER_REGISTERED_TESTS=False；
- configure_environment() 在指定 --device 时清除 CUDA、HIP、ROCR、既有 MUSA 等
  visibility，设置唯一的 MUSA_VISIBLE_DEVICES，并让测试进程使用映射后的 logical
  device 0；
- 未指定 --device 时保留调用进程已有的全部 visibility，不擅自改卡；
- 清理遗留的 FLAGDNN_<OP>_CASE 单 case filter，确保 --ops all 真正运行全部 case；
- test_expression() 使用标准 functional.mthreads.<op> 和
  benchmark.mthreads.<op>，不制造平台别名；
- preflight_tests() 返回完整 mthreads integration gate；
- prepare() 严格加载 comparable case catalog 和 build environment identity；
- postprocess_result() 解析 accounting、mudnn SKIP 和 benchmark JSONL；
- finalize() 校验 functional/reference coverage、provider pairs、speedup 和缺失测试；
- result_diagnostics() 输出可行动的失败原因；
- status_is_success() 在正式严格模式只接受 passed。

adapter 不能按 stdout 中的自由文本猜测 capability。所有解析记录必须有固定 marker、
字段和 schema。显式 --output 使用同目录临时文件和原子替换；失败也要生成结构完整且
标明 incomplete 的 summary，不能留下截断 JSON。

### 16.3 最终 summary

JSON summary 除公共字段外至少增加：

- environment：MUSA、muDNN、Python、Torch、torch_musa、Triton、JIT、target；
- preflight：期望、发现、通过、失败、超时、跳过；
- operator_registration：61 functional、57 benchmark 的发现状态；
- functional_coverage：suite、case、executed、reference skip；
- benchmark_coverage：catalog case、pair、missing、extra；
- performance：metric、threshold、minimum、median、geometric mean、failures；
- cache：artifact/winner hit/miss 和 mutation results；
- installed_consumer：prefix、mapped JIT、compiler/resource provenance；
- final_gate：每项布尔值和最终原因。

精确 all/all 命令生成的 mthreads-results.json 只有在第 20 节全部 Definition of
Done 成立时才能出现 overall_status=passed 且 final_gate.passed=true。局部 --ops
开发报告可以用 overall_status=passed 表示所选范围通过，但必须同时写
final_gate.full_release_selection=false，且不能作为“完美适配”验收证据。

## 17. 强制 integration/preflight

目标测试名和职责如下。实现计划可以把一个测试拆成更小测试，但不能删除覆盖内容：

| CTest 名 | 必须证明 |
| --- | --- |
| integration.mthreads.cmake_configuration_contract | mthreads 独立发现变量、错误 backend/JIT/Python 混搭均配置失败 |
| integration.mthreads.dependency_boundary | core/plugin ELF 依赖边界和 RPATH |
| integration.mthreads.reference_dependency_boundary | muDNN 只在 validation，reference target 真实链接它 |
| integration.mthreads.environment_identity | build/install environment JSON 完整、hash 和 realpath 一致 |
| integration.mthreads.compiler_contract | provider 协议、Graph IR validation、common/override ownership |
| integration.mthreads.artifact | 正常 execution program 和所有 mutation rejection |
| integration.mthreads.jit | 最小 Add 的真实 Triton compile、module load 和 launch |
| integration.mthreads.jit_stream | caller stream、非默认 stream、双 stream 顺序 |
| integration.mthreads.jit_lifecycle | 重复 build/destroy、module cache 和资源增长有界 |
| integration.mthreads.jit_global_state | JIT helper 可重定位、SONAME 和进程内 backend 隔离 |
| integration.mthreads.runtime | binding、workspace、multi-stage、错误映射和 execute 稳态契约 |
| integration.mthreads.graph | 多节点 Frontend Graph 与 muDNN sequence |
| integration.mthreads.graph_capture | warm 后 MUSA capture/instantiate/replay，capture 内零 JIT/alloc/sync |
| integration.mthreads.autotune | candidate prepare、event timing、winner、类型化错误白名单 |
| integration.mthreads.cache | identity invalidation、损坏、并发和原子发布 |
| integration.mthreads.validation_contract | muDNN status、SKIP schema、case accounting、canary |
| integration.mthreads.benchmark_contract | provider pair、JSONL、catalog、正确性先于计时 |
| integration.mthreads.operator_catalog | 61/57 target 均注册且无未知 operator |
| integration.mthreads.installed_consumer | 隔离安装态 C/C++/GPU Graph/JIT/autotune/reference 全链路 |

另外继续运行现有 core tests 和 benchmark manifest/catalog contract。mthreads preflight
通过的定义是期望名称全部在 CTest JSON catalog 中、全部实际执行、无失败、无 timeout、
无整项 SKIP；ctest -N 只用于发现，不能替代执行。

artifact mutation 至少包括：

- schema/backend/target/version/request/compiler identity；
- 重复 JSON key、类型错误和非有限 number；
- 绝对路径、..、symlink escape；
- source size/hash/function/signature；
- stage 数过大、空 program、非法 dependency、cycle；
- duplicate/missing/extra binding UID；
- scalar kind/value/position；
- grid/block/warp/stage/shared-memory 越界；
- workspace offset/size/alignment/overflow/overlap；
- candidate count、重复 ID、不同 ABI、selection corruption。

所有 rejection case 必须在 module launch 前失败，并返回预期 result class。

## 18. 分阶段实施顺序

后续详细 implementation plan 应按垂直切片拆分，每一阶段都同时完成 production、
validation、安装和测试，不能先堆完 compiler 再补 oracle。

### Phase 0（历史开发阶段）：外部依赖与上游资格确认

工作：

- 固化当前版本矩阵和工具链 environment identity；
- 补齐 PyYAML；
- 验证 MUSA JIT package、helper scripts、stream、event、Graph capture；
- 验证 libtriton_jit module 生命周期；
- 验证 bundled JIT 的 helper/RUNPATH 可重定位和同 SONAME global-state 行为；
- 编写最小 muDNN Binary Add C++ probe；
- 确认设备 visibility 和 logical ordinal 规则。

这些探索性检查完成后必须迁入正式配置或 `validation`，不得在最终后端中保留独立
`backends/mthreads/probes` 构建体系，也不得让正式构建依赖预生成 probe summary。
退出条件：Add JIT 和 muDNN reference 都在指定非默认 stream 上正确运行；安装态 JIT
consumer 可发现；生命周期和 capture 门禁有明确结论。失败项优先在相应上游修复。

### Phase 1：backend skeleton 与 Add 端到端

工作：

- mthreads CMake、plugin ABI、context/error；
- compiler provider、identity、最小 artifact；
- common Add source materialization；
- libtriton_jit engine；
- functional Add、autotune Add；
- build/install/installed consumer；
- dependency 和 artifact mutation 基础测试。

退出条件：源码树、build tree、install tree 都能通过 public Graph Add 与 muDNN Add，
并验证 caller stream、cache miss/hit 和默认 backend。

### Phase 2：pointwise、layout、reduction

工作：

- unary/binary/ternary/scalar lowering；
- contiguous、broadcast、strided ABI；
- reshape/transpose/slice；
- reduction sum/avg/mul 和必要多 stage；
- 对应 muDNN reference 和 benchmark pairs；
- common kernel MUSA sweep，必要 override 与 tuning。

退出条件：本阶段所有 manifest operator 的全部公共 case 通过 default/autotune
functional；qualified benchmark 成对记录；无 reference SKIP。

### Phase 3：matmul 与 convolution

工作：

- matmul/batched/broadcast/strided；
- convolution fprop/dgrad/wgrad 1D/2D/3D；
- group、dilation、非对称 padding、mode；
- multi-stage workspace/DAG；
- muDNN algorithm/workspace；
- MUSA 特定 tuning 和性能收敛。

退出条件：全部 matmul/conv case 正确；artifact mutation 覆盖多 stage；benchmark
catalog 完整且达到阶段性能目标。

### Phase 4：normalization 与 composite Graph

工作：

- batchnorm training/inference 全部输出；
- layernorm、rmsnorm；
- add_square、conv_bias_relu；
- virtual tensor、统计量、sequence reference；
- 多节点 Graph、并发 stream 和 alias。

退出条件：全部输出、padding、input mutation 和 DAG contract 通过。

### Phase 5：SDPA 与 FP8

工作：

- SDPA forward/backward；
- bias、mask/band、alignment、GQA/MQA；
- FP8 scales、descales、amax；
- 多 stage attention workspace；
- muDNN Flash/Math capability 和严格语义选择；
- deterministic 和数值稳定性。

退出条件：4 个 attention functional suite 的全部公共 case 通过，无 reference SKIP；
FP8 不用降精度 oracle 冒充通过。

### Phase 6：全量性能、安装与发布门禁

工作：

- 全 57 benchmark comparable catalog；
- MUSA Graph batch 与 direct timing 公平性；
- --min-speedup 0.9；
- cache/lifecycle/长时间重复；
- clean build、clean install、isolated consumer；
- tools/run_tests.py --ops all --suites all。

退出条件：第 20 节所有 DoD 成立并保存机器可读报告。

每阶段加入 platform override 前先保留 common baseline 的正确性和性能数据。override
必须至少解决一个已记录的编译、语义或性能问题；不得提前复制大量 NVIDIA kernel。

## 19. 构建、安装和测试命令

### 19.1 环境

建议使用绝对虚拟环境路径，并把虚拟环境的 lib、Torch 的独立 library 目录都放入
动态库路径，因为当前 libtriton_jit 间接依赖该环境中的 Torch/MKL libraries：

    source /home/wangbingjie/flagdnn/bin/activate
    export MUSA_HOME=/usr/local/musa
    export PATH="$MUSA_HOME/bin:$PATH"
    FLAGDNN_TORCH_LIB="$(
      python -c 'from importlib.util import find_spec; from pathlib import Path; print(Path(find_spec("torch").origin).resolve().parent / "lib")'
    )"
    FLAGDNN_LD_PATH="/usr/local/openmpi-4.1.6/lib:$VIRTUAL_ENV/lib:$FLAGDNN_TORCH_LIB"
    FLAGDNN_LD_PATH="$FLAGDNN_LD_PATH:$MUSA_HOME/lib:/usr/lib/x86_64-linux-gnu"
    if printenv LD_LIBRARY_PATH >/dev/null; then
      FLAGDNN_LD_PATH="$FLAGDNN_LD_PATH:$LD_LIBRARY_PATH"
    fi
    export LD_LIBRARY_PATH="$FLAGDNN_LD_PATH"
    unset FLAGDNN_LD_PATH FLAGDNN_TORCH_LIB

用户提供的 /usr/lib/x86_64-linux-gn 看起来被截断；正式命令使用本机实际目录
/usr/lib/x86_64-linux-gnu。配置前先执行：

    python -c "import torch, torch_musa, triton, yaml; print(torch.__version__)"
    mthreads-gmi
    mcc --version
    ldd /usr/local/lib/libtriton_jit.so

当前验证环境使用 PyYAML 6.0.2；其他部署环境若缺少 PyYAML，CMake 只报告缺失，
不能联网或自动修改虚拟环境。

### 19.2 clean configure/build

首次正式构建使用新的 build/mthreads；如果该目录包含旧配置，应移动到可恢复备份或
使用另一个新目录，不执行 git reset 或删除其他 backend 构建：

    ./tools/build.sh \
      --build-dir build/mthreads \
      --build-type Release \
      --backends mthreads \
      --default-backend mthreads \
      --engine libtriton_jit \
      --python "$VIRTUAL_ENV/bin/python" \
      --tests \
      --benchmarks \
      -- \
      -DFLAGDNN_MTHREADS_MUSA_ROOT="$MUSA_HOME" \
      -DFLAGDNN_MTHREADS_TRITON_JIT_DIR=/usr/local/lib/cmake/TritonJIT \
      -DFLAGDNN_MTHREADS_REQUIRE_REFERENCE_COVERAGE=ON

预期 build tree 同时产生 core、libflagdnn_backend_mthreads.so.2、61 个 functional
suite target、57 个 benchmark suite target 和 mthreads integration tests。

### 19.3 安装

    ./tools/install.sh --build-dir build/mthreads

默认 prefix 为 build/mthreads/install。至少检查：

    find build/mthreads/install -type f -o -type l
    readelf -d build/mthreads/backends/mthreads/libflagdnn_backend_mthreads.so.2
    readelf -d build/mthreads/install/lib/libflagdnn_backend_mthreads.so.2
    ldd build/mthreads/install/lib/libflagdnn_backend_mthreads.so.2

检查目标不是“ldd 没有任何 not found”这么简单，还要确认 JIT 的实际 mapped path、
RPATH、SONAME、resource hash 和 production/reference 依赖边界。

### 19.4 CTest 发现与 preflight

    ctest --test-dir build/mthreads --show-only=json-v1
    ctest --test-dir build/mthreads \
      --output-on-failure \
      -R "^integration[.]mthreads[.]"

第一条只验证 catalog；第二条才执行 integration。正式全量 runner 会默认执行相同
preflight，因此 --no-preflight 只能用于局部开发定位。

### 19.5 单阶段开发回归

示例：

    python3 tools/run_tests.py \
      --build-dir build/mthreads \
      --platform mthreads \
      --device 0 \
      --ops add,relu,reduction \
      --suites all \
      --preflight \
      --output build/mthreads/mthreads-smoke.json

单算子列表不能替代最终 --ops all。

### 19.6 最终全量验收

    python3 tools/run_tests.py \
      --build-dir build/mthreads \
      --platform mthreads \
      --device 0 \
      --ops all \
      --suites all \
      --min-speedup 0.9 \
      --preflight \
      --output build/mthreads/mthreads-results.json

该命令只证明当前记录版本和 logical device 0 上的结果。支持其他 MUSA Toolkit、driver、
muDNN 或设备架构前，必须新增相应矩阵运行，不能从 S5000 单卡结果外推。

## 20. Definition of Done

只有以下条件全部成立，才能宣称“mthreads 完美适配”和 tools/run_tests.py 全链路跑通：

### 20.1 修改边界

- production、validation 和平台 CMake 修改只位于 backends/mthreads；
- 设计/用户文档只位于 docs/mthreads-adaptation-design.md 或 mthreads 自有目录；
- git diff -- backends/nvidia 为空；
- 其他 backend 专用目录 diff 为空；
- 公共目录零修改；若有经单独批准的公共扩展，必须有所有平台 no-regression 证据。

### 20.2 构建与安装

- Release clean configure/build 成功；
- 配置输出完整、单一 provenance 的版本矩阵；
- libflagdnn_backend_mthreads.so.2 ABI、SONAME、backend name 正确；
- production plugin 不链接 muDNN；
- build/install plugin RPATH 只使用 mthreads 私有 JIT 目录；
- bundled JIT 不含 source/build/绝对虚拟环境 RUNPATH，helper 从私有资源解析；
- backend-qualified SONAME 或经测试的 loader isolation 防止进程内错误复用其他 JIT；
- compiler、kernel、tuning、environment 和 JIT helper resources 安装完整；
- 隔离安装态 C/C++/GPU consumer 通过，不读取 source/build tree。

### 20.3 production 执行

- public Graph 真实经过 mthreads compiler、artifact、libtriton_jit 和 MUSA Triton
  kernel；
- 61 个 functional operator target 全部注册；
- 每个公共 functional case 的 production_status=passed；
- default/autotune 两条路径均覆盖；
- 多 stage、workspace、runtime scalar 和 DAG 经 mutation tests；
- execute 不 compile、不 autotune、不 allocate、不 synchronize；
- caller stream、双 stream、重复 binding 和 concurrency contract 通过；
- warm executable 可被 MUSA runtime Graph capture 和 replay。

### 20.4 reference 与功能正确性

- muDNN C++ primitive/sequence 是唯一 reference；
- 每个公共 functional case 的 reference_status=comparable；
- 所有 logical output、辅助 output、padding canary 和 input mutation 检查通过；
- structured reference skip 数为 0；
- 没有 host、Torch、FlagDNN self-reference 或语义降级。

### 20.5 autotune 与 cache

- candidate compile、warmup、dependency replay、MUSA event timing 在 build 阶段；
- 只过滤两种已批准的类型化资源配置错误；
- winner correctness 通过且 cache 原子、可重复、可失效；
- target、compiler、kernel、tuning、JIT、Python/Triton、driver 变化触发 miss；
- execute 只使用已固化 winner；
- 并发 build 和损坏 cache 测试通过。

### 20.6 benchmark 与 runner

- 57 个 benchmark operator target 全部注册；
- 每个公共 benchmark case 都在 comparable catalog 中；
- 每个 case 恰有 flagdnn/mudnn 两条 record；
- correctness gate 在 timing 前通过；
- missing pair、extra pair、单边 latency、重复 case 均为 0；
- qualified case 全部满足 mudnn_median_us/flagdnn_median_us >= 0.9；
- preflight、118 个 operator-suite invocation 均无 failure、timeout 或 SKIP；
- tools/run_tests.py 最终退出码为 0；
- mthreads-results.json 完整且 final_gate 全部为 true。

### 20.7 生命周期与可维护性

- 重复 Graph build/destroy、JIT function/module、stream/event/workspace 资源增长有界；
- thread-local error 在并发调用中不串扰；
- mthreads 不依赖其他 backend 私有源文件；
- 每个 platform override 有原因、case、baseline 和 tuning provenance；
- 当前版本矩阵、capability catalog 和最终报告一起归档。

tools/run_tests.py 退出 0 但存在 reference SKIP、缺失 benchmark pair、未执行 preflight
或安装态 consumer 未通过，不满足本 DoD。

## 21. 风险与处理

| 风险 | 处理 |
| --- | --- |
| 配置误选 CUDA/HIP libtriton_jit | mthreads namespaced resolver，校验 BACKEND=MUSA、library/hash/scripts provenance |
| Python/Torch/Triton 来自不同环境 | environment identity 和实际 import path fail closed |
| PyYAML 缺失 | configure preflight 明确失败，不在 CMake 中联网安装 |
| LD_LIBRARY_PATH 缺少 Torch 或虚拟环境 lib | canonical env 加入 torch/lib 和 VIRTUAL_ENV/lib，安装 consumer 做 ldd/mapped-path 检查 |
| MUSA runtime stream 与 MUstream 不兼容 | `integration.mthreads.jit_add` 在非默认 caller stream 真实 launch，失败修复 JIT 上游 |
| libtriton_jit module cache 无界 | 生命周期压力测试；优先上游增加有界缓存/释放 |
| common kernel 有隐藏 MUSA 不兼容 | registry entry 级 compile/launch sweep；私有 override，不改 common |
| 复制 NVIDIA 架构特化 | mthreads 独立 source，静态 dependency/include 禁止测试 |
| capture 中触发首次 JIT/分配 | create_executable 全预热，capture 前后 cache/resource 计数 |
| muDNN primitive 名存在但语义不完整 | case 级 descriptor + runtime probe，不凭 header 宣称支持 |
| reference SKIP 被当成“通过” | 正式 REQUIRE_REFERENCE_COVERAGE=ON，finalize 要求零 SKIP |
| autotune 吞掉 kernel/compiler 缺陷 | 只允许两个类型化资源错误，其余硬失败 |
| 多 stage workspace 重叠或依赖错 | 单一 packer、synthetic UID、range/DAG validator 和 mutation tests |
| 并发 stream 共享 scratch | mutable scratch 全部放调用者 workspace，executable metadata immutable |
| JIT 安装文件与其他 backend 冲突 | build/install 使用 flagdnn/mthreads 私有目录和精确 plugin RPATH |
| 同 SONAME JIT 在进程内被错误复用 | backend-qualified SONAME 或经测试的 loader isolation；否则明确限制单 JIT backend 进程 |
| multi-backend CMake imported target 冲突 | mthreads 子目录受控解析并捕获自己的 target properties |
| benchmark 只优化少量 case | 全量 comparable catalog、missing/extra pair gate、单 case threshold |
| 热降频或时钟漂移 | 交替 provider、足够 warmup/sample、报告设备状态；不改变 correctness |
| FP8 用非 FP8 oracle 冒充 | 强制比较 scales/descales/amax 和 FP8 语义，无法等价则不满足 DoD |
| 为 MUSA 修改公共/其他平台代码 | 路径 allowlist、git diff gate、独立公共 RFC |

## 22. 后续文档与评审规则

历史 implementation plan 将适配拆成 Phase 0–6；最终代码结构以本文和 NVIDIA 对齐
后的 production/validation 职责边界为准。
implementation plan 必须为每个步骤写出：

- 精确文件路径；
- 先失败的测试；
- 最小实现；
- 验证命令和预期结果；
- 对应本文的退出条件；
- 不触碰公共和其他 backend 的检查。

实施中更新 capability 或 comparable catalog 时，提交说明必须附真实设备、版本、
case、muDNN status 和结果证据。最终实测数据写入单独的验收记录，不能反向把某次
机器结果改写成无条件的平台承诺。

仓库内相关规范：

- docs/architecture.md；
- docs/testing.md；
- docs/add-functional-execution-chain.md；
- backends/backend_api.h；
- cmake/Operators.cmake；
- backends/mthreads/adaptation-recommendations.md。

外部实现依据：

- FlagOS libtriton_jit：https://github.com/flagos-ai/libtriton_jit
- Moore Threads torch_musa：https://github.com/MooreThreads/torch_musa

本文若与上述公共 FlagDNN contract 冲突，以版本化公共 API/ABI 和仓库测试为准；
若与当前 MUSA/muDNN 实机行为冲突，以可复现 probe 为准，并先更新设计再实现。

## 23. 2026-08-26 实装记录

本节记录当前工作区的 as-built 状态。第 1–22 节仍是维护和发布契约；若两者有差异，
本节明确标出实机结论或尚未完成的门禁，不能据此静默降低目标。

### 23.1 已实现链路

当前实现已经打通：

    public FlagDNN C/C++ Graph
      -> versioned Graph IR
      -> mthreads compiler provider
      -> common/MThreads Triton source materialization
      -> versioned execution-program artifact
      -> libflagdnn_backend_mthreads.so.2
      -> private MUSA libtriton_jit build/load
      -> immutable stage launch metadata
      -> caller musaStream_t 上的 muLaunchKernel

对应实现位于 `backends/mthreads`，包括 backend ABI v2、MUSA context/error、artifact
parser、stage-DAG executor、autotune、compiler provider、kernel registry/override、
tuning policy、配置期环境 identity、functional/benchmark validation 和安装态 consumer。
production steady-state `execute()` 不再进入 Python/JIT；实测 JIT 计数为：dense build
2、autotune miss 8、autotune hit 2、steady execute 0。warm executable 已通过 MUSA
runtime Graph capture、instantiate 和 replay。

autotune selection 使用每个 stage 自己的 `tuning/stage-N.json`。实现阶段修复了此前
多 stage 读取 stage-0 winner 的风险；runtime 现在按精确 stage ID 固化 winner，
selection 损坏、identity 不匹配或候选缺失均 fail closed。

### 23.2 数值 reference 的实际口径

普通 pointwise、layout、reduction、matmul、convolution、normalization、composite 和
普通 SDPA 前后向，均使用独立 buffer 上的直接 muDNN C++ primitive 或严格等价
primitive sequence；FlagDNN Graph 结果不参与 expected 构造。

当前 muDNN 3.1.5 的 `ScaledDotProductAttention::RunFlash/RunFlashBwd` 没有
descale、scale 或 amax 参数，并在实机上拒绝 FP8 FlashAttention，因而不能表达
FlagDNN FP8 Attention 的完整公开契约。`sdpa_fp8` 与 `sdpa_fp8_backward` 明确使用
validation-only、与 DUT 独立的高精度数学 oracle，逐项模拟 FP8 编解码并检查全部
scale/descale、output/stats、gradient 和 amax；日志会明确打印该例外，不能将其表述为
muDNN FP8 primitive。4 个 Attention suite 共 14 个 functional case；仓库 manifest
有意不为这 4 个算子注册 benchmark，因此总数为 61 functional、57 benchmark。

### 23.3 可重定位安装闭包

build tree 与 install tree 都采用平台私有 JIT 布局：

    <lib>/libflagdnn_backend_mthreads.so.2
    <lib>/flagdnn/mthreads/libtriton_jit.so
    <lib>/flagdnn/share/triton_jit/scripts/{gen_ssig.py,standalone_compile.py}
    <share>/flagdnn/backends/mthreads/{compiler modules,kernels,tuning,identity}

plugin 的 ELF `DT_RPATH` 精确为 `$ORIGIN/flagdnn/mthreads`，并使用 old-dtags 防止
调用者 `LD_LIBRARY_PATH` 抢占私有 JIT。`flagdnn_mthreads_install.json` 记录 JIT 与
helper 的安装相对路径、SONAME、provenance 和 SHA-256；compiler identity 与 C++
runtime 都在使用前复核这些 hash。environment identity 根据 canonical SDK/JIT/Python
资源内容生成，不包含软链请求路径或动态设备状态。当前私有 JIT SHA-256 为
`38c354e2cde47a2441d2b5ec1ed4f4d3ff41360de37458c18cff2d2389e1d926`。

隔离 installed-consumer 同时覆盖纯 C ABI consumer 和 `find_package(FlagDNN)` C++
consumer；验证显式/default MThreads Handle、安装态 compiler 自动发现、cold/hot
autotune、实际 mapped plugin/JIT、MUSA Graph capture/replay、FlagDNN/default 与
直接 muDNN Add 的 bitwise 对照，并拒绝 CUDA cubin/source/build-tree 泄漏。官方
`tools/install.sh` 已在 NOCONFIG 单配置下成功安装到隔离 prefix，并生成
`FlagDNNTargets-noconfig.cmake`。

### 23.4 已完成的实机门禁

当前 MTT S5000、MUSA 4.3.5、muDNN 3.1.5、Python 3.10.12、Torch 2.9.0、
torch_musa 2.9.0、Triton 3.6.0、PyYAML 6.0.2 环境已得到下列证据：

- 配置期工具链 environment identity 生成与 canonical-path 稳定性测试通过；
- CTest catalog：144 个测试，含 61 functional、57 benchmark targets 和环境 identity 单元测试；
- final-run preflight：23/23 通过；runner 正则之外的 benchmark catalog
  contract 另行 2/2 通过；
- compiler contract：48 个有效请求通过，103 个非法请求被拒绝；
- artifact contract：47 个有效 artifact 通过，371 个 mutation 被拒绝；
- convolution：fprop 24/24、dgrad 27/27、wgrad 27/27；
- Attention：4 个 suite、14/14 functional cases；
- isolated installed C/C++ consumers：2/2 通过；
- official `tools/install.sh`：通过。

共享 convolution runner 输出 `FLAGDNN_CONVOLUTION_FUNCTIONAL`；MThreads runner
adapter 仅对 fprop/dgrad/wgrad functional accounting 使用受控别名，并拒绝错误方向
或 benchmark 别名。Attention runner 已统一输出标准 `cases/executed/skipped` marker。

最终 all/all 报告为
`build/mthreads-audit-20260831/mthreads-final-verified-results.json`：118/118 个
operator-suite 进程通过，0 failed、0 skipped、0 timeout；benchmark 覆盖 1182/1182
个 FlagDNN/muDNN case pair、2364 条 provider record，0 record error，覆盖标记为
complete。61 个 functional suite 全部通过。runner 最终退出码为 1，原因仅是
24/1182 个 benchmark case 未满足
`mudnn_median_us/flagdnn_median_us >= 0.9`；最低值 0.642837，来自 MatMul，不能将
这个结果表述为性能发布门禁通过。

本轮定位并修复了一个会系统性误选候选的性能缺陷：MThreads engine 原先用重复
`muLaunchKernel` 直接 launch 给候选排序，而 benchmark 的 steady state 通过 MUSA
Graph replay，二者在小 kernel 上的排序并不一致。当前实现与 NVIDIA/Hygon engine
对齐，在 32-node captured Graph batch 上 warmup 和计时，按实际执行次数折算单次
耗时，并将 measurement identity 升级为
`mthreads-libtriton-jit-captured-mugraph-batch32-v1` 以失效旧 winner。代表性 Log
case `fp16_8x16x32` 从 0.546 左右恢复到约 1.05；完整 all/all 的未达标 case 从修复前
118 个降为 24 个。

随后按 NVIDIA 优化 unary 的实现将 Log 改为 `log2(x) * ln(2)`。当前 compiler
identity 为
`502ea6487b57d08d1bc85c9d80dca26d2e141d2e63d1d93240d22656676b19ce`。
`build/mthreads-audit-20260831/mthreads-log-log2-experiment.json` 证明完整 Log
functional suite 通过、24/24 benchmark pair 和 48/48 provider record 完整；全部
24 个 FlagDNN 中位数相对上一实现改善 3.6%–29.8%，没有超过 5% 的回退。Log 的
3 个 `3x257x513` case 仍未过 0.9，最低值从 0.768 提升至 0.855；因此全局未达标
数量仍为 24，最低值仍由 MatMul 决定。

### 23.5 Matmul TLE 实装与局部门禁（2026-08-29）

dense、非 broadcast 的 FP16/BF16 Matmul 已增加 MTGPU TLE 路径。kernel 使用独立
A/B SPSC pipe、descriptor-to-shared-memory copy 和 warp-specialized WGMMA；正式
launch 记录 16 个 consumer warps，producer worker 使用 4 warps、32 registers，
最终 block 为 20 warps。producer 对 A、B 依次执行 acquire/copy/commit，consumer
逐 reduction tile 执行 wait、WGMMA、`wgmma_wait(0)` 和 release。输出仍使用普通
pointer store，避免当前实机上 output descriptor 路径的挂起问题。

只有下列已实测 shape 进入 `matmul_tle_kernel`；其他 descriptor-compatible shape
继续进入 `matmul_descriptor_kernel`，strided/broadcast shape 继续进入
`matmul_strided_kernel`：

| B×M×N×K | BLOCK_M×BLOCK_N×BLOCK_K | stages | panel width |
| --- | --- | ---: | ---: |
| 32×512×512×512 | 128×128×32 | 3 | 2 |
| 16×1024×1024×1024 | 256×256×32 | 3 | 2 |
| 16×2048×2048×512 | 256×256×32 | 3 | 2 |
| 8×2048×2048×2048 | 256×256×64 | 3 | 2 |
| 32×1024×1024×4096 | 256×256×64 | 3 | 4 |
| 4×4096×4096×4096 | 256×256×64 | 3 | 2 |

512 shape 从原配置调整到 128×128×32 后，两次正式 Graph benchmark 均确认收益。
调优前报告 `build/mthreads/reports/matmul-tle-final.json` 与最终配置报告
`build/mthreads/reports/matmul-tle-final-strict.json` 的中位数如下：

| dtype | FlagDNN 调优前 | FlagDNN 最终 | mudnn/FlagDNN 调优前 | mudnn/FlagDNN 最终 |
| --- | ---: | ---: | ---: | ---: |
| FP16 | 81.568 μs | 79.432 μs | 0.780404 | 0.821634 |
| BF16 | 81.000 μs | 77.256 μs | 0.802667 | 0.845501 |

当前局部正确性与覆盖证据为：

- `functional.mthreads.matmul` 执行 27/27，零 skip；
- `benchmark.mthreads.matmul` 执行 24/24 case，每个 case 都有一条 FlagDNN 和一条
  muDNN steady-state record，共 48 条；
- 所有 benchmark accuracy、capture、postcheck 和 provider-pair 检查通过；TLE
  FP16/BF16 case 的 `max_abs`、`max_rel` 均为 0；
- compiler/artifact contract 同时覆盖 FP16 与 BF16 TLE 完整签名，runtime 参数仍为
  A、B、output 三个 tensor binding。

性能发布门禁尚未完成。当前正式配置的严格报告
`build/mthreads-audit-20260831/mthreads-final-verified-results.json` 在阈值
`mudnn_median_us/flagdnn_median_us >= 0.9` 下有 9/24 case 未通过，最低值为
0.642837；CTest 自身通过，runner 因性能门禁返回 1。失败项为：

| case | speedup |
| --- | ---: |
| BF16 16×2048×512 by 16×512×2048 | 0.642837 |
| FP16 16×2048×512 by 16×512×2048 | 0.710556 |
| BF16 16×1024×1024 | 0.785530 |
| BF16 8×2048×2048 | 0.820624 |
| FP16 32×512×512 | 0.821955 |
| FP16 16×1024×1024 | 0.829478 |
| BF16 32×512×512 | 0.836349 |
| FP16 8×2048×2048 | 0.870362 |
| BF16 32×1024×4096 by 32×4096×1024 | 0.899233 |

本轮还否决了两个只在局部测量中看似有利的候选：1024 FP16 的 BLOCK_K=64 在
同模式 Graph 对比中没有收益，已恢复 BLOCK_K=32；4096 BF16 的 panel width=4
在临时 direct Graph 中看似更快，但正式 24-case 路径只得到 0.880994，低于
panel width=2 基线的 0.899205，因此 dtype override 与对应实验夹具均已撤销。候选
报告保留在 `build/mthreads/reports/matmul-tle-dtype-final-strict.json`，不能作为当前
配置或发布结果引用。

当前安装版 TLE 的 `wgmma_wait` 只支持 pending=0，pipe 也只提供单 payload 的 SPSC
语义，无法直接表达多 async WGMMA group 的计算/搬运重叠。后续性能工作应优先处理
两种 rectangular case 和 1024/2048 square case，或先扩展上游 TLE primitive；不得
通过降低 0.9 门槛掩盖剩余差距。本节只是 Matmul 局部门禁证据，不替代 23.4 的
最终 all/all 验收。

### 23.6 中断恢复后的性能修复与剩余阻塞（2026-09-03）

恢复开发后重新检查了工作树、旧 all/all 报告和当前构建产物。production 功能链、
61/57 个 per-operator target、compiler/artifact contract 与安装边界均已实装；旧全量
报告的 118/118 个 operator-suite 进程和 1182/1182 个 benchmark case pair 也均完整。
尚未完成的是 `mudnn_median_us/flagdnn_median_us >= 0.9` 性能发布门禁。

本轮保留以下两类最小修复：

- dense contiguous unary 对完整 program 使用无 mask load/store，只在尾部 program
  保留 mask；Log 使用 `log2(x) * ln(2)`，Tanh、GELU 和近似 GELU 使用 MUSA
  libdevice 的 `fast_tanh`/`fast_gelu`；
- TLE Matmul 仅对门禁已保证完整的 output tile 去除冗余 store mask。当前稳定
  `backends/mthreads/kernels/matmul.py` SHA-256 为
  `f151121e90fd1bc50ec161f4156d247cda93b81469e30e7ba6647426ad886b43`。

当前专项证据如下：

- `build/mthreads-tools-audit/reports/mthreads-unary-fastmath.json`：4 个 unary 算子的
  functional+benchmark 共 8/8 suite 通过，96/96 case pair 完整，0 性能失败，最低
  speedup 为 0.902741；
- `build/mthreads-tools-audit/reports/mthreads-identity-unmasked.json`：preflight 通过，
  Identity functional+benchmark 2/2 suite、33/33 case pair、66/66 provider record
  完整，最低 speedup 为 0.907056；
- `build/mthreads-tools-audit/reports/mthreads-matmul-unmasked.json`：Matmul functional
  27/27 case、benchmark 24/24 case pair 和 48/48 provider record 均正确完整；性能
  失败从旧报告的 9 个降到 8 个，最低 speedup 从 0.642837 提升到 0.724776。

最新 Matmul 未达标项为：

| case | speedup |
| --- | ---: |
| FP16 16×2048×512 by 16×512×2048 | 0.724776 |
| BF16 16×2048×512 by 16×512×2048 | 0.768055 |
| FP16 32×512×512 | 0.820541 |
| BF16 32×512×512 | 0.830828 |
| FP16 16×1024×1024 | 0.840790 |
| BF16 16×1024×1024 | 0.864166 |
| FP16 8×2048×2048 | 0.894123 |
| BF16 8×2048×2048 | 0.896554 |

本轮还用最差 BF16 rectangular case 验证并撤销了以下候选：

- `maxnreg=128` 和 `160` 分别为 314.048 μs（0.758304）和 309.120 μs
  （0.763872），不足以形成可保留收益；
- `BLOCK_M×BLOCK_N=128×256` 为 382.104 μs（0.619036），显著退化；
- 将值与地址分别 `reshape(..., can_reorder=True)` 会因 SQMMA 与 blocked 源布局的
  物理置换不同而产生错误结果；用累加器零值锚定地址仍在 element 8 复现同一错误；
- 关闭引擎强制的 `TRITON_MUSA_ENABLE_LLC_OPT` 虽正确，但退化到 419.896 μs
  （0.563073），证明该开关必须保留。

其他已撤销候选包括 BK=64、pipeline stage 变化、panel width 变化、128×128 或
256×512 tile、减少 consumer/producer warps、改变 producer acquire/commit 顺序、
取消 loop unroll、两组 async WGMMA 后再 wait、output cache modifier、block-pointer
  store 和 output descriptor。后者在当前设备上会挂起；多 async group 又受限于
  `wgmma_wait(0)` 与单 payload SPSC pipe。

因此当前代码已达到这套安装版 FlagTree/TLE 原语下的已验证局部最优，但最终 all/all
性能门禁仍未通过，不能标记为 release complete。下一步需要先在上游扩展安全的
多 pending WGMMA/pipe 语义或修复 SQMMA 到连续输出布局转换，再回到本仓库复测上述
8 个 Matmul case；不得降低阈值、删除案例或链接 muDNN/muBLAS 绕过 production 边界。

### 23.7 最终代码与全链路审计（2026-09-03）

最终审计在实际 MTT S5000（不是 910B4-1）、MUSA 4.3.5、muDNN 3.1.5 环境完成。
本轮发现并修复了两个不能由已有性能报告替代的 production 正确性问题：

- Identity、dense Reshape 和物理 copy Transpose 的 D2D copy 快速路径原先会绕过
  `RawArguments`，因而没有执行每个 stage 的 UID、alignment 和 runtime ABI 校验。
  当前快速路径同样构造并校验 runtime arguments；功能回归传入对齐地址偏移，确认
  public execute API 会拒绝非法 binding，而不是提交 copy。
- 嵌入式 Python JIT 原先会在安装 SDK 的私有 compiler/kernel 目录生成
  `__pycache__`，改变完整性保护的 environment identity，导致下一次 graph build
  重建 artifact 并丢失 autotune winner。引擎现在只在 build mutex 保护的 JIT
  调用期间同时抑制环境和已初始化解释器的 bytecode 写入，退出时恢复调用者原始
  环境、`sys.dont_write_bytecode` 和 C API flag。隔离 installed-consumer 明确
  清除外部 `PYTHONDONTWRITEBYTECODE` 后验证 cold/hot cache 命中、selection 时间戳
  不变、SDK/cache 中无 `.pyc`/`__pycache__`，且临时设置没有泄漏到调用进程。

runner 也改为从 CMake 的 functional/benchmark operator manifest 生成目标集合，
不再依赖手写列表或静默过滤已注册测试；指定 device 时清除 CUDA/HIP/ROCR 等跨平台
visibility 变量，并只设置 MUSA visibility。安装说明通过 `find_spec("torch")` 定位
Torch，而不在补齐动态库路径前导入它；private JIT 所需 Torch/OpenMPI 路径已写入示例。

最终全量报告为 `build/mthreads-final-audit-20260903/mthreads-final-results-20260903.json`：
preflight 24/24 通过，118/118 个 operator-suite 进程通过，0 failed、0 skipped、
0 timeout；61 个 functional suite 全部通过；57 个 benchmark suite 覆盖
1182/1182 个 case pair 和 2364 条 provider record，0 record error，coverage
complete。runner 仍因严格 `mudnn_median_us/flagdnn_median_us >= 0.9` 门禁返回 1。
全量长跑记录到 11 个低于阈值的 case，其中 8 个是稳定的 Matmul 短板：

| case | speedup |
| --- | ---: |
| FP16 16×2048×512 by 16×512×2048 | 0.725865 |
| BF16 16×2048×512 by 16×512×2048 | 0.765379 |
| BF16 32×512×512 | 0.820991 |
| FP16 32×512×512 | 0.831161 |
| FP16 16×1024×1024 | 0.838979 |
| BF16 16×1024×1024 | 0.845709 |
| FP16 8×2048×2048 | 0.892736 |
| BF16 8×2048×2048 | 0.897848 |

另外 3 个是微秒级边界记录：BinarySelect FP32 `1×1×1024` 两侧样本在长跑中先后
发生约 4→7 μs 与 5→8 μs 的频率漂移，导致两个独立 median 落在不同区间。在空闲
设备上独立复测 BinarySelect 和 Identity 的 57 个 case 后，BinarySelect 全部过线，
Identity 仅 BF16 `16×64×128` 为 0.899300，距阈值约 0.08%。复测报告为
`build/mthreads-final-audit-20260903/mthreads-targeted-nonmatmul-performance-20260903.json`。

为排除错误实现选择，还临时关闭了 Identity D2D copy 并重跑完整功能和 33 个性能
case：功能通过，但纯 Triton 路径产生 6 个性能失败、最低 0.851875，关键形状慢
4%–6%，因此实验补丁已撤销并保留当前更快的 copy 路径。

后端关闭的独立干净构建完成 77 个目标且 15/15 core CTest 通过；MThreads Release
`-Werror` 构建完成 371 个目标。公共代码变化仅包括 per-operator 布局校验对
`core/run_tests_contract.py` 的精确豁免和生成物 ignore，不引入 MUSA 头文件、链接项
或运行时分支，未发现影响 NVIDIA/Hygon/Cambricon 等其他平台的路径。

因此 correctness、coverage、编译、安装、工具与集成契约已完成；性能发布门禁仍不能
标记完成。稳定阻塞仍是上述 8 个 Matmul case，需要先扩展上游 FlagTree/TLE 的多
pending WGMMA/pipe 能力或修复 SQMMA 到连续输出的安全转换，然后重新执行 all/all；
不得用放宽阈值、删除 case 或 production 链接 muDNN/muBLAS 的方式规避。
