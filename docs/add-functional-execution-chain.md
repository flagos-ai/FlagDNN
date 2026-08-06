# Add 功能测试到 GPU 的完整执行链路

本文面向刚接触 FlagDNN 的开发者，以 Add 算子为例，解释一次功能测试如何从 C++ Graph API 出发，经过 Graph lowering、编译器、kernel 选择、FlagDNN autotune 和 `libtriton_jit`，最终通过 CUDA Driver 在 GPU 上运行，并与 cuDNN Frontend 的结果比较。

## 1. 先记住结论

当前 Add 的生产执行链路是：

```text
C++ 功能测试
  -> FlagDNN Frontend Graph::pointwise(ADD)
  -> 原生 OperationDescriptor
  -> Pointwise Lowering
  -> Graph IR
  -> NVIDIA 编译器生成 JIT manifest
  -> 选择 common Add Triton kernel
  -> FlagDNN 展开并执行 autotune
  -> libtriton_jit 编译 Triton kernel
  -> CUDA Driver cuLaunchKernel
  -> GPU 执行
  -> 与 cuDNN Frontend Graph 的输出比较
```

Add 的源 Triton kernel 是：

- [`kernels/common/binary.py`](../kernels/common/binary.py#L43)

它不使用已经删除的旧 Python 包或 portable kernel。cuDNN 只在功能测试中充当独立参考实现，不参与 FlagDNN 的生产执行链路。

## 2. 如何运行 Add 功能测试

如果还没有编译：

```bash
cd /home/wangbingjie/FlagDNN

tools/build.sh \
    --backends nvidia \
    --engine libtriton_jit \
    --tests \
    --no-benchmarks
```

运行 Add 的全部功能测试：

```bash
python3 tools/run_tests.py \
    --build-dir build/nvidia \
    --suites functional \
    --platform nvidia \
    --ops add \
    --verbose
```

只运行启用 autotune 的 Add case，并显示调优日志：

```bash
FLAGDNN_EXECUTION_ENGINE=libtriton_jit \
FLAGDNN_PRINT_AUTOTUNING=1 \
FLAGDNN_ADD_CASE=contiguous_autotune \
python3 tools/run_tests.py \
    --build-dir build/nvidia \
    --suites functional \
    --platform nvidia \
    --ops add \
    --verbose
```

参数含义：

- `--ops add`：只运行 Add。
- `--verbose`：显示 `tools/run_tests.py` 捕获的 CTest 输出。
- `FLAGDNN_ADD_CASE=contiguous_autotune`：只运行名字包含该字符串的 case。
- `FLAGDNN_PRINT_AUTOTUNING=1`：打印候选耗时、cache hit 和最终选择。

测试启动逻辑在 [`tools/run_tests.py`](../tools/run_tests.py#L175)。它最终运行的 CTest 是：

```text
functional.nvidia.add
```

测试可执行文件通常位于：

```text
build/nvidia/backends/nvidia/validation/flagdnn_test_nvidia_add
```

## 3. 功能测试的代码如何组织

Add 测试入口是 [`tests/test_add.cpp`](../tests/test_add.cpp#L7)：

```cpp
int main(int argc, char** argv) {
  const auto cases = flagdnn::testing::make_add_cases();
  return flagdnn::testing::run_add_functional_test(
      argc, argv, std::span(cases));
}
```

它只负责：

1. 创建 Add 测试用例。
2. 将这些用例交给当前平台的测试适配器。

相关代码分为三层：

```text
tests/common/add.cpp
  平台无关的 case 数据、FlagDNN C++ Graph 构造

backends/nvidia/validation/functional/add_runner.cpp
  NVIDIA CUDA 内存、stream、执行和结果比较

backends/nvidia/validation/functional/cudnn_add.cpp
  NVIDIA 测试中的独立 cuDNN Frontend 参考实现
```

通用测试 target 在 [`tests/CMakeLists.txt`](../tests/CMakeLists.txt#L24) 注册；NVIDIA 测试适配器在 [`backends/nvidia/validation/CMakeLists.txt`](../backends/nvidia/validation/CMakeLists.txt#L43) 组装。

这种拆分意味着未来增加其他平台时，`tests/test_add.cpp` 和 `tests/common/add.cpp` 可以继续复用，平台目录只提供自己的 runner、设备 I/O 和参考实现。

## 4. Add 测试用例

测试用例定义在 [`tests/common/add.cpp`](../tests/common/add.cpp#L250)，当前包括：

```text
add_contiguous_autotune_fp32
add_odd_extent_fp16
add_nhwc_layout_bf16
add_alpha_half_fp32
add_alpha_negative_bf16
```

只有第一个 case 启用了 autotune：

```cpp
make_case(
    "contiguous_autotune",
    {2, 4, 8},
    {2, 4, 8},
    FLAGDNN_DATA_FLOAT32,
    100,
    1.0,
    true);
```

每个 case 还包含：

- 输入、输出 Tensor 的 UID。
- 数据类型。
- dimensions 和 strides。
- Add 的 `alpha`。
- 绝对误差和相对误差阈值。

UID 可以理解为 Graph 中 Tensor 的插槽编号。Graph build 只处理 Tensor 元数据；Graph execute 时，UID 才与真实 GPU 指针绑定。

## 5. Add 的 C++ Graph API 入口

测试侧构造 FlagDNN Graph 的类是 [`FlagdnnAddExecutable`](../tests/common/add.cpp#L183)。关键代码为：

```cpp
graph_->set_name(test_case.name)
    .set_io_data_type(...)
    .set_intermediate_data_type(...)
    .set_compute_data_type(...)
    .set_autotune(test_case.autotune);

const auto left = make_tensor(graph_, test_case.left, "left");
const auto right = make_tensor(graph_, test_case.right, "right");

auto output = graph_->pointwise(
    left,
    right,
    fe::graph::Pointwise_attributes()
        .set_name("add")
        .set_mode(fe::PointwiseMode_t::ADD)
        .set_compute_data_type(fe::DataType_t::FLOAT)
        .set_alpha(test_case.alpha));
```

因此当前对外 Add API 是：

```cpp
graph.pointwise(
    left,
    right,
    Pointwise_attributes().set_mode(PointwiseMode_t::ADD));
```

而不是 `graph.add(left, right)`。这是 cuDNN Frontend 风格的接口：Add 是 Pointwise operation 的一种 mode。

### 为什么生产代码中没有 add.cpp

Add、Sub、Mul、Div 等算子具有相同的结构：

- 都是 binary pointwise。
- Tensor、广播和 stride 校验相同。
- kernel 参数结构相同。
- 只需要由 `PointwiseMode` 区分计算语义。

所以生产 lowering 集中在 [`src/graph/lowering/pointwise.cpp`](../src/graph/lowering/pointwise.cpp#L27)，GPU 算法集中在共享的 `binary.py`。`tests/common/add.cpp` 是 Add 功能测试的 Graph 构造代码，不是生产算子实现。

## 6. Graph::pointwise 只是在记录计算图

公开的 Frontend Graph 定义在 [`include/flagdnn/frontend.hpp`](../include/flagdnn/frontend.hpp#L1370)，二元 `Graph::pointwise` 位于 [`include/flagdnn/frontend.hpp`](../include/flagdnn/frontend.hpp#L1466)。

这个函数主要完成：

1. 检查左右输入 Tensor。
2. 检查 mode 是否是受支持的二元 Pointwise mode。
3. 检查 `alpha`。
4. 推导广播后的输出 shape。
5. 创建输出 Tensor 描述。
6. 向 `nodes_` 中加入一个二元 Pointwise Node。

关键动作是：

```cpp
nodes_.push_back(
    Node::make_binary_pointwise(left, right, output, attributes));
```

这一阶段不会：

- 分配 GPU Tensor 内存。
- 编译 Triton kernel。
- 调用 `libtriton_jit`。
- 启动 GPU kernel。

它只是记录：

```text
output = ADD(left, right, alpha)
```

## 7. Graph::build 才开始准备可执行程序

测试随后调用：

```cpp
graph_->build(handle_, {fe::HeurMode_t::A});
```

调用位置是 [`tests/common/add.cpp`](../tests/common/add.cpp#L213)，Frontend 实现在 [`Graph::build`](../include/flagdnn/frontend.hpp#L2094)。

它依次执行：

```text
validate()
  -> build_operation_graph()
  -> create_execution_plans()
  -> check_support()
  -> build_plans()
```

对应代码位置：

- [`validate()`](../include/flagdnn/frontend.hpp#L1989)
- [`build_operation_graph()`](../include/flagdnn/frontend.hpp#L2001)
- [`create_execution_plans()`](../include/flagdnn/frontend.hpp#L2020)
- [`check_support()`](../include/flagdnn/frontend.hpp#L2050)
- [`build_plans()`](../include/flagdnn/frontend.hpp#L2068)

最关键的是 `check_support()`：

```cpp
auto candidate = std::make_unique<flagdnn::Executable>(
    handle, *native_graph_, &options);
```

创建 `Executable` 时才真正进入 Graph IR、编译器、kernel 选择、JIT 和 autotune。

`set_autotune(true)` 最终在 [`selected_build_options()`](../include/flagdnn/frontend.hpp#L4253) 中变成：

```cpp
FLAGDNN_BUILD_OPTION_AUTOTUNE
```

## 8. Frontend Graph 转换为原生 Graph

Frontend 通过 [`lower_to_native_graph()`](../include/flagdnn/frontend.hpp#L3526) 将高层 Node 转成低层 `OperationDescriptor`。

Add 走二元 Pointwise 分支 [`NodeKind::kBinaryPointwise`](../include/flagdnn/frontend.hpp#L3556)：

```cpp
flagdnn::OperationDescriptor operation(
    FLAGDNN_OPERATION_POINTWISE);

operation.set_pointwise(
    input,
    second,
    FLAGDNN_POINTWISE_ADD,
    output,
    alpha);

native_graph->add(operation);
```

这里的：

```cpp
native_graph->add(operation);
```

不是算术加法，而是“将一个 operation 加入 Graph”。低层包装位于 [`flagdnn::Graph::add`](../include/flagdnn/flagdnn.hpp#L636)，最终调用 [`flagdnnGraphAddOperation`](../src/api.cpp#L1810)。

真正代表加法语义的是：

```text
FLAGDNN_OPERATION_POINTWISE
+ FLAGDNN_POINTWISE_ADD
```

## 9. 进入原生 Graph 编译流程

Frontend 创建低层 `flagdnn::Executable` 后会调用 [`flagdnnBuildExecutable`](../src/api.cpp#L1859)，然后进入 [`build_graph_executable()`](../src/graph/graph.cpp#L26)。

它的核心流程是：

```cpp
validate_graph(graph);

for (operation : graph.operations) {
  lowered.push_back(lower_operation(operation));
}

graph_ir = make_graph_ir(...);
artifact = prepare_artifact_package(context, graph_ir);
backend_executable = context.create_executable(artifact);
```

可以理解为：

```text
OperationDescriptor
  -> LoweredOperation
  -> Graph IR
  -> 编译/JIT artifact
  -> BackendExecutable
```

## 10. Pointwise Lowering

统一分发入口是 [`lower_operation()`](../src/graph/lowering/dispatch.cpp#L54)。Pointwise 算子进入 [`lower_pointwise()`](../src/graph/lowering/pointwise.cpp#L263)。

Add、Sub、Mul、Div 等都会复用 [`lower_binary_pointwise()`](../src/graph/lowering/pointwise.cpp#L27)，它负责：

- 输入、输出数量检查。
- 数据类型检查。
- stride 非重叠检查。
- 广播 shape 检查。
- 计算输出元素数量。
- 提取 Pointwise mode。
- 提取 `alpha`。

Add 最终得到的主要参数类似：

```text
n_elements     = 64
pointwise_mode = FLAGDNN_POINTWISE_ADD
alpha          = 1.0
```

[`pointwise_operation_name()`](../src/graph/lowering/pointwise.cpp#L118) 将 `FLAGDNN_POINTWISE_ADD` 映射为字符串 `"add"`，后续 kernel registry 使用这个字符串查找实现。

## 11. Graph IR

Graph IR 由 [`make_graph_ir()`](../src/graph/ir.cpp#L170) 生成。简化后类似：

```json
{
  "backend": "nvidia",
  "target": "sm_90",
  "build_options": {
    "heuristic_modes": ["A"],
    "autotune": true
  },
  "graph": {
    "nodes": [
      {
        "type": "add",
        "inputs": [
          {"name": "left", "uid": 100},
          {"name": "right", "uid": 101}
        ],
        "outputs": [
          {"name": "output", "uid": 102}
        ],
        "attributes": {
          "n_elements": 64,
          "pointwise_mode": 1,
          "alpha": 1.0
        }
      }
    ]
  }
}
```

Graph IR 是平台无关核心和平台编译器之间的协议。核心层不需要知道 CUDA 的 `CUfunction`、Triton 函数名或 `BLOCK_SIZE`。

## 12. Handle 和 NVIDIA backend plugin

测试 runner 创建：

```cpp
flagdnn::Handle handle(FLAGDNN_BACKEND_NVIDIA, 0);
```

位置是 [`add_runner.cpp`](../backends/nvidia/validation/functional/add_runner.cpp#L254)。Handle 初始化进入 [`RuntimeContext::initialize`](../src/runtime/context.cpp#L110)，并由 [`BackendLibrary::load`](../src/backend_loader.cpp#L157) 动态加载：

```text
libflagdnn_backend_nvidia.so.<ABI版本>
```

CTest 通过 `FLAGDNN_BACKEND_PATH` 告诉核心库 plugin 所在目录，设置位置在 [`backends/nvidia/validation/CMakeLists.txt`](../backends/nvidia/validation/CMakeLists.txt#L75)。

因此生产库分为：

```text
libflagdnn.so
  平台无关的 API、Graph、IR、cache 和 plugin loader

libflagdnn_backend_nvidia.so
  NVIDIA CUDA、artifact parser、autotune 计时和执行引擎
```

## 13. handle.set_compiler() 的作用

测试 runner 调用：

```cpp
handle.set_compiler(
    argv[1],
    argv[2],
    cache.path().string());
```

三个参数分别是：

```text
argv[1] = Python 解释器
argv[2] = compiler/flagdnn_codegen/main.py
第三项  = 本次测试的 artifact cache 目录
```

公开接口位于 [`Handle::set_compiler`](../include/flagdnn/flagdnn.hpp#L72)，底层保存到 [`RuntimeContext::set_compiler`](../src/runtime/context.cpp#L146)。

这里要区分两种 Python 使用：

1. FlagDNN 外部编译器进程：读取 Graph IR、选择 kernel、展开 tuning 候选、生成 manifest。
2. `libtriton_jit` 内嵌 Python：真正调用 Triton compiler 编译 kernel。

它们不是同一个阶段。

## 14. Artifact cache 和外部编译器

Graph IR 交给 [`prepare_artifact_package()`](../src/runtime/cache.cpp#L168)。缓存结构大致为：

```text
<cache>/
  nvidia/
    sm_90/
      <graph_hash>/
        <compiler_identity>/
          request.json
          manifest.json
          generated_stage_0.py
          .flagdnn-autotune-....json
```

编译器子进程由 [`run_compiler_process()`](../src/runtime/compiler_client.cpp#L86) 启动。系统先调用 [`query_compiler_identity()`](../src/runtime/compiler_client.cpp#L211)，cache miss 时再调用 [`compile_external_artifact()`](../src/runtime/compiler_client.cpp#L240)。

`compile_external_artifact` 是历史/通用命名。当 execution engine 是 `libtriton_jit` 时，它不会提前生成最终 cubin，而是生成：

- materialized Triton source。
- kernel 函数名。
- runtime/full signature。
- launch grid。
- 参数 ABI。
- autotune variants。
- `manifest.json`。

真正的 Triton 编译留给 `libtriton_jit`。

通用编译器入口是 [`compiler/flagdnn_codegen/main.py`](../compiler/flagdnn_codegen/main.py#L55)。它根据 Graph IR 的 `backend` 加载对应平台 provider；NVIDIA provider 是 [`backends/nvidia/compiler.py`](../backends/nvidia/compiler.py#L9384)。

## 15. Add 选择哪个 Triton kernel

kernel registry 选择函数是 [`select_kernel_candidate()`](../compiler/flagdnn_codegen/kernel_registry.py#L134)，顺序为：

```text
1. 查 backends/nvidia/kernels/registry.json
2. NVIDIA 没有该算子时，查 kernels/registry.json
```

当前 NVIDIA binary override 只覆盖 `pow` 和 `sigmoid_backward`，见 [`backends/nvidia/kernels/registry.json`](../backends/nvidia/kernels/registry.json#L70)。它没有 Add，因此 Add 回退到 [`kernels/registry.json`](../kernels/registry.json#L5)：

```json
{
  "operations": [
    "add", "sub", "mul", "div", "min", "max", "mod", "pow",
    "cmp_eq", "cmp_neq", "cmp_gt", "cmp_ge", "cmp_lt", "cmp_le"
  ],
  "source": "binary.py",
  "functions": [
    "binary_contiguous_kernel",
    "binary_strided_kernel"
  ]
}
```

所以当前 Add 的源 kernel 明确是：

```text
kernels/common/binary.py
```

如果未来在 NVIDIA registry 中注册 Add，它会自动优先选择 `backends/nvidia/kernels/` 下的平台优化实现，不需要修改 Graph 或通用 lowering。

## 16. binary.py 如何复用多个算子

实际计算逻辑位于 [`_apply_binary_operation()`](../kernels/common/binary.py#L43)：

```python
if OP_KIND == POINTWISE_ADD:
    result = left + ALPHA * right
```

因此 FlagDNN Add 的数学语义是：

```text
output = left + alpha * right
```

同一个 kernel 通过编译期常量 `OP_KIND` 复用多个算子：

```text
OP_KIND=ADD -> 加法
OP_KIND=SUB -> 减法
OP_KIND=MUL -> 乘法
OP_KIND=DIV -> 除法
...
```

这就是旧架构中 binary kernel 复用思想在当前 C++ 架构中的实现。

## 17. 连续和 strided kernel 的选择

NVIDIA 编译器处理 binary 配置的位置是 [`backends/nvidia/compiler.py`](../backends/nvidia/compiler.py#L5298)。默认会准备：

```text
OP_KIND    = ADD
ALPHA      = test_case.alpha
BLOCK_SIZE = 256
```

然后通过 [`_can_use_dense_binary_kernel()`](../backends/nvidia/compiler.py#L444) 判断 Tensor 是否具有相同 shape、相同 strides 且物理 dense。

满足条件时选择 [`binary_contiguous_kernel`](../kernels/common/binary.py#L88)，否则选择 [`binary_strided_kernel`](../kernels/common/binary.py#L111)。

连续 kernel 的核心逻辑是：

```python
program_id = tl.program_id(0)
offsets = program_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
mask = offsets < n_elements

left = tl.load(x_ptr + offsets, mask=mask)
right = tl.load(y_ptr + offsets, mask=mask)
result = _apply_binary_operation(left, right, OP_KIND, ALPHA)
tl.store(out_ptr + offsets, result, mask=mask)
```

每个 Triton program 处理 `BLOCK_SIZE` 个元素，grid 为：

```text
ceil(n_elements / BLOCK_SIZE)
```

当前 5 个 Add case 的输入和输出具有相同且物理 dense 的布局，所以当前测试实际走 `binary_contiguous_kernel`。`binary_strided_kernel` 支持广播和不同 strides，但这 5 个 Add case 没有覆盖真正的广播路径。

## 18. Autotune 候选展开

Add 使用 [`backends/nvidia/tuning/common.yaml`](../backends/nvidia/tuning/common.yaml#L1010) 中的 `binary` 表：

```yaml
binary:
  - gen: true
    param_map:
      META:
        BLOCK_SIZE: block_size
      num_warps: warps
      num_stages: stages
    block_size:
      - 128
      - 256
      - 512
      - 1024
      - 2048
      - 4096
    warps:
      - 4
      - 8
      - 16
    stages:
      - 2
      - 3
```

候选数量是：

```text
6 个 BLOCK_SIZE
* 3 个 num_warps
* 2 个 num_stages
= 36 个候选
```

[`_expand_generated_tuning_entry()`](../backends/nvidia/compiler.py#L7779) 使用 `itertools.product` 展开完整笛卡尔积，随后 [`_prepare_tuning_variants()`](../backends/nvidia/compiler.py#L8559) 为每个候选重新生成：

- constexpr 常量。
- `num_warps`、`num_stages`。
- launch grid。
- full signature。
- candidate identity。

因此 Add 不是只尝试几个写死的 block，而是完整消费 YAML 的全部调优空间。

## 19. 当前 autotune 与 libtriton_jit 的职责

`binary.py` 使用普通的：

```python
@triton.jit
```

并没有使用 `@triton.autotune`，当前 `libtriton_jit` 也不负责选择最快配置。

职责划分是：

```text
FlagDNN 编译器
  展开 36 个候选

libtriton_jit
  编译每个被请求的候选 kernel

FlagDNN NVIDIA engine
  使用 CUDA Event 测量候选 GPU 时间

FlagDNN autotune_policy
  选择最快候选并缓存
```

平台无关的选择和缓存策略位于 [`backends/autotune_policy.cpp`](../backends/autotune_policy.cpp#L227)，负责：

- winner cache 查询。
- 校准单次 kernel 时间。
- warmup。
- 多次采样和中位数统计。
- 对最快的 3 个候选再次确认。
- 原子写入 winner cache。

NVIDIA 的 CUDA Event 计时位于 [`LibTritonJitEngine::select_candidate`](../backends/nvidia/engines/libtriton_jit.cpp#L654)。它使用 CUDA Graph 批量 launch，并通过 `cuEventElapsedTime` 测量纯 GPU 时间。

注意 YAML 中的 `warmup: 5` 和 `repetitions: 10` 在当前策略中作为毫秒预算使用，不是简单地固定执行 5 次和 10 次；策略会先校准，再计算应执行多少次。

## 20. libtriton_jit 如何接入 NVIDIA backend

NVIDIA backend 在 [`backends/nvidia/CMakeLists.txt`](../backends/nvidia/CMakeLists.txt#L104) 中执行：

```cmake
find_package(TritonJIT 0.1.0 CONFIG REQUIRED)
```

随后将 `TritonJIT::triton_jit` 链接进 `libflagdnn_backend_nvidia.so`，并在 [`backends/nvidia/CMakeLists.txt`](../backends/nvidia/CMakeLists.txt#L194) 定义：

```cmake
FLAGDNN_HAS_LIBTRITON_JIT=1
```

顶层默认执行引擎在 [`CMakeLists.txt`](../CMakeLists.txt#L27) 中设置为：

```text
libtriton_jit
```

运行时也可以通过环境变量选择：

```bash
export FLAGDNN_EXECUTION_ENGINE=libtriton_jit
```

## 21. libtriton_jit 如何编译 Triton kernel

NVIDIA backend 解析 `manifest.json` 后创建 [`LibTritonJitEngine`](../backends/nvidia/engines/libtriton_jit.cpp#L529)。每个 stage 首先调用：

```cpp
const JitFunction& function = JitFunction::get_instance(
    stage.source.string(), stage.function_name);
```

对 Add 来说参数大致为：

```text
source   = <artifact>/generated_stage_0.py
function = binary_contiguous_kernel
```

`generated_stage_0.py` 是根据 `kernels/common/binary.py` 物化到 artifact 目录中的运行时源文件。

FlagDNN 随后通过 [`launch_jit()`](../backends/nvidia/engines/libtriton_jit.cpp#L405) 调用：

```cpp
function.launch_with_raw_args(
    stream,
    grid_x,
    grid_y,
    grid_z,
    num_warps,
    num_stages,
    full_signature,
    arguments,
    argument_count);
```

`libtriton_jit` 的 raw API 位于 [`triton_jit_function.h`](../../libtriton_jit/include/triton_jit/triton_jit_function.h#L420)。它调用 `get_kernel(full_signature, options, device)`；实际 cache miss 编译逻辑位于 [`triton_jit_function.cpp`](../../libtriton_jit/src/triton_jit_function.cpp#L96)：

1. 初始化内嵌 Python。
2. import `standalone_compile`。
3. 调用 `compile_a_kernel(...)`。
4. 让 Triton 编译 Python kernel。
5. 加载 GPU kernel。
6. 写入 `libtriton_jit` 的 overload cache。

因此实际关系是：

```text
FlagDNN C++ backend
  -> libtriton_jit C++ API
  -> 内嵌 Python
  -> Triton compiler
  -> CUDA kernel
```

用户侧仍然只使用 C++ API。

## 22. Tensor 指针和 kernel 参数如何绑定

[`run_case()`](../backends/nvidia/validation/functional/add_runner.cpp#L193) 创建的 binding 类似：

```cpp
{
    {100, left_gpu_pointer},
    {101, right_gpu_pointer},
    {102, output_gpu_pointer},
}
```

manifest 中保存 argument ABI。Add 连续 kernel 的可见参数类似：

```text
参数 0：UID=100 的 Tensor 指针
参数 1：UID=101 的 Tensor 指针
参数 2：UID=102 的 Tensor 指针
参数 3：n_elements，i32
参数 4：global scratch
参数 5：profile scratch
```

参数整理发生在 [`RawArguments`](../backends/nvidia/engines/libtriton_jit.cpp#L290)。

其中：

- `x_ptr`、`y_ptr`、`out_ptr`、`n_elements` 是运行时参数。
- `OP_KIND`、`ALPHA`、`BLOCK_SIZE` 是 Triton `constexpr`，包含在 full signature 中，在编译时特化。

例如一个候选可能是：

```text
OP_KIND    = ADD
ALPHA      = 1.0
BLOCK_SIZE = 512
num_warps  = 8
num_stages = 3
```

## 23. 稳态 execute 为什么不再进入 Python

Graph build 阶段，FlagDNN 通过 `libtriton_jit` 准备好 kernel 后，会在 [`prepare_cuda_launch()`](../backends/nvidia/engines/libtriton_jit.cpp#L424) 中使用 CUDA Graph capture 捕获一次 launch，再通过 `cuGraphKernelNodeGetParams` 取得：

```text
CUfunction
grid
block
shared_memory
```

这些内容保存为 `PreparedCudaLaunch`。

之后正常执行不会重新进入 Python，也不会重新调用 Triton compiler。稳态调用链是：

```text
Graph::execute()
  -> flagdnnExecuteAsync()
  -> native::Executable::execute()
  -> BackendExecutable::execute()
  -> NVIDIA plugin execute()
  -> LibTritonJitEngine::execute()
  -> launch_prepared_cuda()
  -> cuLaunchKernel()
```

关键位置：

- [`Graph::execute`](../include/flagdnn/frontend.hpp#L2142)
- [`flagdnnExecuteAsync`](../src/api.cpp#L1908)
- [`native::Executable::execute`](../src/runtime/executable.cpp#L30)
- [`BackendExecutable::execute`](../src/backend_loader.cpp#L313)
- [`LibTritonJitEngine::execute`](../backends/nvidia/engines/libtriton_jit.cpp#L617)
- [`launch_prepared_cuda`](../backends/nvidia/engines/libtriton_jit.cpp#L506)

最终进入 GPU 的调用是 CUDA Driver API：

```cpp
cuLaunchKernel(...);
```

## 24. build 与 execute 的区别

| 阶段 | 主要工作 | Python | GPU |
|---|---|---:|---:|
| `graph.tensor()` | 记录 Tensor 元数据 | 否 | 否 |
| `graph.pointwise()` | 向 Graph 加入 Add Node | 否 | 否 |
| 首次 `graph.build()` | Lowering、IR、JIT、autotune | 会 | 会执行调优候选 |
| cache hit 的 `graph.build()` | 读取 artifact/winner、准备赢家 | 可能查询编译器/JIT cache | 会准备赢家 |
| `graph.execute()` | 使用真实指针启动已准备 kernel | 否 | 会 |
| 重复 `graph.execute()` | 直接 `cuLaunchKernel` | 否 | 会 |

因此功能测试显示的几十秒不是 Add kernel latency。功能测试总耗时包括 Python compiler、36 个 JIT 候选、autotune、cuDNN Graph build、内存复制和数值比较。性能 benchmark 才测稳态 kernel 时间。

## 25. Cache 和失效规则

当前至少有三层相关 cache：

1. Graph artifact cache：保存 manifest 和 materialized source。
2. FlagDNN autotune winner cache：保存特定设备的获胜 variant。
3. `libtriton_jit` kernel cache：保存 full signature 和编译选项对应的 kernel。

[`compiler_identity.py`](../backends/nvidia/compiler_identity.py#L22) 会哈希：

- kernel registry。
- Triton kernel 源码。
- tuning YAML。
- 编译器代码。
- Triton 版本。
- ptxas 版本。
- execution engine。

因此修改 `kernels/common/binary.py` 或 `backends/nvidia/tuning/common.yaml` 会改变 compiler identity，旧 artifact 和调优结果不会被错误复用。

Add 功能测试使用 [`TemporaryCache`](../backends/nvidia/validation/functional/add_runner.cpp#L34)，进程退出时会删除整个目录。因此：

- 同一 Graph 重复 execute 不会重新调优。
- 同一次测试进程中 cache 可以生效。
- 重新启动 Add 功能测试时通常会重新调优。

生产环境可以指定持久化目录：

```bash
export FLAGDNN_CACHE_DIRECTORY=/path/to/persistent/flagdnn-cache
```

## 26. 与 cuDNN Frontend 的结果比较

NVIDIA runner 的 [`run_case()`](../backends/nvidia/validation/functional/add_runner.cpp#L193) 分别创建：

```cpp
auto flagdnn = build_flagdnn_add(handle, test_case);
auto reference = build_add_reference(test_case);
```

两条执行链完全独立：

```text
FlagDNN 分支
  FlagDNN Frontend
  -> Triton
  -> libtriton_jit
  -> CUDA kernel

参考分支
  cuDNN Frontend
  -> cuDNN execution plan
  -> cuDNN kernel
```

cuDNN 参考实现是 [`CudnnAddExecutable`](../backends/nvidia/validation/functional/cudnn_add.cpp#L203)。对于 `alpha != 1`，参考 Graph 显式构造：

```text
scaled_right = right * alpha
output = left + scaled_right
```

runner 最后：

1. 给两个实现传入相同输入。
2. 分别写入不同输出缓冲区。
3. 同步 CUDA stream。
4. 将结果复制回 CPU。
5. 检查 padding 是否被破坏。
6. 按 `atol/rtol` 比较数值。

通过后输出：

```text
add_xxx: FlagDNN Graph vs cuDNN Graph PASS
```

## 27. 两条最重要的调用链

Graph build 链路：

```text
tests/common/add.cpp
  -> Graph::pointwise(ADD)
  -> Graph::build()
  -> Graph::check_support()
  -> flagdnnBuildExecutable()
  -> native::build_graph_executable()
  -> lower_pointwise()
  -> make_graph_ir()
  -> prepare_artifact_package()
  -> compiler/flagdnn_codegen/main.py
  -> backends/nvidia/compiler.py
  -> select_kernel_candidate()
  -> kernels/common/binary.py
  -> LibTritonJitEngine()
  -> TritonJITFunction::get_instance()
  -> launch_with_raw_args()
  -> FlagDNN autotune
  -> PreparedCudaLaunch
```

稳态 execute 链路：

```text
Graph::execute()
  -> flagdnnExecuteAsync()
  -> native::Executable::execute()
  -> BackendExecutable::execute()
  -> NVIDIA backend execute()
  -> LibTritonJitEngine::execute()
  -> launch_prepared_cuda()
  -> cuLaunchKernel()
  -> binary_contiguous_kernel()
  -> GPU
```

## 28. 一句话总结

当前 Add 由 cuDNN-Frontend 风格的 `Graph::pointwise(ADD)` 描述语义，由通用 Pointwise Lowering 生成 `"add"` Graph IR；NVIDIA registry 因没有 Add 专用覆盖而选择 `kernels/common/binary.py`；FlagDNN 从 YAML 展开和测量 autotune 候选；`libtriton_jit` 负责将候选编译成 CUDA kernel；准备完成后，每次 `Graph::execute()` 都直接通过 `cuLaunchKernel` 在 GPU 上运行。
