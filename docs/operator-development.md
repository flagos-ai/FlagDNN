# FlagDNN 算子开发

## 1. 先确定算子语义

FlagDNN 对外提供 cuDNN-Frontend-style Graph primitive，而不是独立 eager 函数。
新增算子前先确定它属于已有 Graph family 还是需要新 primitive：

- pointwise：binary、unary、comparison、activation、ternary
- reduction
- layout/view
- matmul
- convolution
- normalization
- composite graph

例如 Add 是 `Graph::pointwise(... PointwiseMode_t::ADD)`，不会新增
`Graph::add()`。

## 2. 修改位置

一个完整算子通常涉及：

1. `include/flagdnn/frontend.hpp`
   - 增加公开 enum/attributes/Graph method，或复用现有 family。
2. `src/graph/`
   - 定义 operation descriptor、shape/dtype/attribute 校验。
3. `src/graph/lowering/`
   - 将公开 Graph operation 降为稳定 backend operation。
4. `kernels/common/`
   - 增加或复用通用 Triton 算法。
5. `kernels/registry.json`
   - 登记 common operation、source 和 entry function。
6. `backends/<platform>/kernels/`
   - 仅当平台确实有专用算法时增加 override。
7. `backends/<platform>/tuning/`
   - 登记候选空间，不在 C++ 或 Python 代码中硬编码候选。
8. `tests/` 与 `benchmark/`
   - 增加同名的 C++ 薄入口和 family case。

不要为同一模板族复制 kernel。例如 Add、Sub、Mul、Div、比较和逻辑运算应复用
binary kernel 模板，通过 compile-time operation 选择具体表达式。

## 3. Common kernel

通用 kernel 必须：

- 位于 `kernels/common/<family>.py`
- 只依赖 Triton 与 Python 标准库
- 接受由 compiler 明确传入的 constexpr/meta 参数
- 支持 registry 声明的 dtype/layout/shape
- 不读取平台 runtime 全局状态
- 不包含 Torch tensor allocation、decorator registry 或 eager wrapper

在 `kernels/registry.json` 中登记后，所有没有平台 override 的 backend 都可以解析它。

## 4. 平台优化 kernel

平台优化 kernel 位于：

```text
backends/<platform>/kernels/<family>.py
backends/<platform>/kernels/registry.json
```

平台 registry 对 operation 的登记是显式覆盖。覆盖后：

- 该平台总是先选择平台 candidate。
- capability 判断必须在 compiler policy 中完成。
- 编译或加载失败必须返回错误。
- 不能以异常为条件隐式回退 common。

若平台只需要不同 tuning 配置而不需要不同算法，应复用 common source，只在 tuning
所有权允许的位置增加平台配置。

## 5. Tuning candidate

调优空间来自 YAML/registry。新增或修改 candidate 时检查：

- tuning table/key 能唯一定位算子族
- 所有 YAML 维度都被展开
- candidate ID 稳定且包含影响生成代码的配置
- cache identity 包含 kernel source、compiler、target 和 config
- 共享内存、warp/stage 等平台限制可在 JIT prepare 阶段明确过滤

不要把 `BLOCK_SIZE=128/256/512` 之类的列表重复写入 C++。

## 6. C++ Graph 与 lowering

Frontend 层负责用户可见语义；lowering 负责 backend operation，不负责 launch。

推荐命名：

- `lower_<family>()`：Graph IR 到 backend descriptor
- `build()`：完成 Graph validation、lowering 和 executable 创建
- `execute()`：绑定 UID、workspace 和 stream 后运行

不要使用 `prepare_<op>()` 表达 lowering，也不要为 family 内每个 mode 创建一个
`src/graph/lowering/<op>.cpp`。只有参数结构和 lowering 算法真正独立时才拆文件。

## 7. 测试要求

将算子加入 `cmake/Operators.cmake` 后，必须同时提供：

```text
tests/test_<op>.cpp
benchmark/test_<op>.cpp
```

功能测试验证 FlagDNN Graph 与平台 DNN Graph/明确 host oracle 的输出。性能测试先执行
正确性门禁，再执行 warmup 和设备 event 计时。顶层入口不得包含平台头文件。

推荐验证顺序：

```bash
cmake --build /tmp/flagdnn-build -j
ctest --test-dir /tmp/flagdnn-build \
  -R 'core.test_architecture|catalog_contract' \
  --output-on-failure
python3 tools/run_tests.py \
  --build-dir /tmp/flagdnn-build \
  --ops <op> \
  --suites functional,benchmark \
  --platform nvidia
```

GPU 功能测试、autotune 和 benchmark 必须串行运行。
