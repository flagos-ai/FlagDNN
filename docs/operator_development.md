# 算子开发指南

本文档是新增或扩展高性能 FlagDNN 算子的开发入口。它把 eager Triton 实现、graph capture schema、graph 编译期 fast path、自动调优配置、功能测试和性能 benchmark 分开说明，目标是让算子开发者明确知道每一层应该在哪些文件添加哪些代码。

## 开发分层

| 层级 | 文件位置 | 什么时候修改 | 性能注意事项 |
|---|---|---|---|
| Eager 算子实现 | `src/flag_dnn/ops/<op>.py` | 必改。定义 public API、输入校验、输出分配、Triton kernel dispatch，并显式拒绝暂不支持的 cuDNN 功能。 | 这是算子语义和 eager 性能的源头。热路径 dispatch 要保持简单；只有 benchmark 证明有收益时，才保留 shape-specific fast path。 |
| Eager 导出 | `src/flag_dnn/ops/__init__.py`、`src/flag_dnn/__init__.py` | 新增 public 算子时修改。 | 导出层不应增加运行时开销。 |
| Graph wrapper | `src/flag_dnn/graph/wrappers.py` | 算子需要被 `@flag_dnn.graph` capture 时，把算子名加入 `GRAPH_AWARE_OPS`。 | wrapper 主要发生在 capture 阶段，不要在这里添加 graph replay 的运行时逻辑。 |
| Graph schema | `src/flag_dnn/graph/registry.py` | 添加 normalize、shape inference、fallback run function 和 `OpSchema`。 | normalize 和 shape inference 在 graph 构建期执行；fallback run function 应调用 eager 算子，保证 prepared fast path 不适用时仍然正确。 |
| Graph 编译期 fast path | `src/flag_dnn/graph/prepared/core.py`、`src/flag_dnn/graph/prepared/ops.py`、`src/flag_dnn/graph/prepared/<family>.py` | 只有性能敏感 graph 路径需要静态绑定 shape、stride、attrs、grid、kernel constexpr 时才添加。框架 API 放在 `prepared/core.py`；`prepared/ops.py` 只做 side-effect import；具体 op/family preparer 放在 `prepared/<family>.py`。op-specific fast path 用 `@register_prepared_run_fn("<op>")` 注册；多个 op 共享的 preparer 用 `@register_generic_prepared_run_fn` 注册。 | 这是 graph 性能优化的关键层。返回的闭包会在 replay 中执行，因此应在闭包外预计算 scalar tail、grid、输出 stride 和 kernel cache。约束不满足时返回 `None` 或调用 `default_run_fn`。 |
| 自动调优配置 | `src/flag_dnn/runtime/backend/_nvidia/tune_configs.yaml` | 每个存在 tunable `BLOCK_*`、tile、split、`num_warps`、`num_stages` 或 grouping meta 参数的 Triton kernel，都要新增或更新 tune key。 | kernel 中使用 `runtime.get_tuned_config()`。只要自动调优适用，最终性能路径不能长期停留在硬编码 meta 参数上。 |
| 功能测试 | `tests_graph/test_graph_<op>.py`、`tests_graph/consts.py` | 添加 cuDNN graph 对标测试和 shape 集合。 | 测试要证明 graph capture 命中了目标 op，并且输出语义与参考实现一致。 |
| 性能 benchmark | `benchmark_graph/test_<op>_perf.py`、`benchmark_graph/consts.py` | 添加该算子的必测性能 shape。 | 对比 FlagDNN graph 与 cuDNN graph baseline，并报告 `speedup = baseline_time / candidate_time`。 |

## Graph 算子接入清单

1. 先在 `src/flag_dnn/ops/<op>.py` 实现 eager 算子。public API 声称支持的语义必须对齐 cuDNN frontend。
2. 在 `src/flag_dnn/ops/__init__.py` 和 `src/flag_dnn/__init__.py` 导出该算子。
3. 如果需要 graph capture，在 `src/flag_dnn/graph/wrappers.py` 的 `GRAPH_AWARE_OPS` 中加入算子名。
4. 在 `src/flag_dnn/graph/registry.py` 添加 graph schema：
   - `_normalize_<op>`：把位置参数和关键字参数转换成 graph input ids 与不可变 attrs。
   - `_<op>_shape`：返回输出 `TensorSpec`。
   - `_run_<op>`：调用 eager 算子，作为正确性 fallback。
   - `OpSchema(name="<op>", ...)`：注册 graph op。
5. 只有当 prepared fast path 能减少 replay 开销或启用专用 kernel 时，才增加 prepared run function。通用框架 API 来自 `src/flag_dnn/graph/prepared/core.py`；op-specific 代码放在对应 `prepared/<family>.py`，新 family 则新增 `prepared/<family>.py` 并在 `prepared/ops.py` 导入触发注册。op-specific preparer 推荐用 decorator 注册：

```python
@register_prepared_run_fn("<op>")
def _prepare_<op>(
    attrs: dict[str, Any],
    input_specs: Sequence[TensorSpec],
    default_run_fn: RunFn,
) -> Optional[RunFn]:
    if not_supported_by_fast_path:
        return None

    static_tail = (...)
    grid = (...)
    cache: list[Any] = [None, None]

    def run(inputs: Sequence[Any], run_attrs: dict[str, Any]) -> Any:
        if runtime_constraints_changed:
            return default_run_fn(inputs, run_attrs)
        ...

    return run
```

如果一个 preparer 要覆盖一组 op，例如 pointwise，可以使用 generic 注册：

```python
@register_generic_prepared_run_fn
def _prepare_family(
    op_type: str,
    attrs: dict[str, Any],
    input_specs: Sequence[TensorSpec],
    default_run_fn: RunFn,
) -> Optional[RunFn]:
    ...
```

6. 根据 replay 结构选择 prepared spec。单个 launch 且只需要一个输出时用 `PreparedSingleKernelRunSpec`；固定顺序多 kernel、需要 workspace 或多步结果合成时用 `PreparedKernelPipelineSpec`。两条路径的完整代码模板见“新算子 Prepared 模板”，不要在接入清单里复制维护第二份示例。

7. 明确 prepared fast path 的缓存边界。第一次 replay 可以触发 Triton compile/autotune；之后应复用 compiled kernel、静态 grid 和静态参数。若某个 pipeline step 的首次 launch 返回 `(compiled_kernel, metadata)`，设置 `first_launch_returns_metadata=True` 并提供 `build_cached_call`；若 replay direct launcher 使用固定 grid/args，用 `make_static_cached_call(grid, args)`。

8. 第一次正式性能结论前，在 `_nvidia/tune_configs.yaml` 添加或更新对应 tune key。
9. 添加 graph 功能测试和性能 benchmark。benchmark shape 集合要小而有代表性，且必须覆盖性能敏感场景。

## Prepared 代码布局

Prepared 代码分三层：

- `src/flag_dnn/graph/prepared/core.py` 是框架层：定义 `RunFn`、`PrepareRunFn`、注册表、`prepare_run_fn`、`RuntimeTensorCheck`、单 kernel spec、pipeline spec、cached-call helper、tensor cache helper、launcher/run-fn 生成器和注册 decorators。普通算子开发者通常只导入这些 API，不在这里写 op-specific 分支。
- `src/flag_dnn/graph/prepared/common.py` 放跨算子族共享的轻量 helper，例如 runtime device 检查、静态 shape 读取和 unsupported path 报错。不要在这里放 op-specific dispatch。
- `src/flag_dnn/graph/prepared/<family>.py` 是算子族层：通过 decorators 注册具体 prepared fast path。复杂算子族可以继续拆细，例如 `prepared/sdpa_forward.py` 和 `prepared/sdpa_backward.py`。`prepared/ops.py` 是注册入口，负责导入这些子模块触发注册，并 re-export `prepare_run_fn` 给 executor 使用。

当前算子族模块：

1. `prepared/pointwise.py`：pointwise family preparer 及其内部 helper。
2. `prepared/sdpa_forward.py`：`sdpa` forward 的 graph 编译期 fast path。
3. `prepared/sdpa_backward.py`：`sdpa_backward` 的 graph 编译期 fast path。
4. `prepared/conv.py`：convolution helpers，以及 `conv_dgrad`、`conv_wgrad`、`conv_fprop` 的 prepared fast path。

新增算子时，如果属于已有 family，放进对应 `prepared/<family>.py`；如果是新的复杂 family，新建 `prepared/<family>.py` 并在 `prepared/ops.py` 导入该模块以触发注册。不要再把大量 op-specific 代码直接写回 `prepared/ops.py`。

## 新算子 Prepared 模板

假设新增算子 `op_a`。下面两个代码模板分别覆盖单 kernel fast path 和固定顺序多 kernel pipeline，开发时按实际路径二选一，不要对同一组条件同时注册两个 preparer。这个模板只展示 prepared 层；eager 算子、graph schema、测试和 benchmark 仍按前面的接入清单完成。

1. 如果 `op_a` 属于已有 family，把代码放进对应 `prepared/<family>.py`。如果是新 family，新建 `src/flag_dnn/graph/prepared/op_a.py`，并在 `prepared/ops.py` 增加一行 side-effect import：

```python
from flag_dnn.graph.prepared import op_a as _prepared_op_a  # noqa: F401
```

2. 在 `prepared/op_a.py` 中写 op-specific preparer。优先使用 `runtime_tensor_checks_from_specs`、`PreparedSingleKernelRunSpec` 和 `make_single_kernel_run_fn`：

```python
from __future__ import annotations

from typing import Any, Optional, Sequence

import torch
import triton

from flag_dnn.graph.prepared import (
    PreparedSingleKernelRunSpec,
    PreparedSingleKernelSpec,
    RunFn,
    make_single_kernel_run_fn,
    register_prepared_run_fn,
    runtime_tensor_checks_from_specs,
)
from flag_dnn.graph.prepared.common import _static_shape
from flag_dnn.graph.tensor import TensorSpec, torch_dtype
from flag_dnn.ops.op_a import _op_a_kernel


@register_prepared_run_fn("op_a")
def _prepare_op_a(
    attrs: dict[str, Any],
    input_specs: Sequence[TensorSpec],
    default_run_fn: RunFn,
) -> Optional[RunFn]:
    if len(input_specs) != 1:
        return None

    shape = _static_shape(input_specs[0])
    if shape is None:
        return None
    checks = runtime_tensor_checks_from_specs(input_specs, (0,))
    if checks is None:
        return None

    dtype = torch_dtype(input_specs[0].dtype)
    stride = tuple(input_specs[0].stride or ())
    if not stride:
        return None

    n = int(shape[-1])
    out_shape = shape
    static_tail = (n, *stride)
    constexpr_kwargs = {"BLOCK_D": triton.next_power_of_2(n)}

    def grid(meta: dict[str, Any]) -> tuple[int, int, int]:
        return (triton.cdiv(n, meta["BLOCK_M"]), 1, 1)

    def build_cached_call(
        meta: dict[str, Any],
    ) -> tuple[tuple[int, ...], tuple[Any, ...]]:
        block_m = int(meta["BLOCK_M"])
        return (triton.cdiv(n, block_m), 1, 1), static_tail + (block_m,)

    def make_output(inputs: Sequence[Any]) -> torch.Tensor:
        x = inputs[0]
        assert isinstance(x, torch.Tensor)
        return torch.empty(out_shape, dtype=dtype, device=x.device)

    def runtime_args(
        inputs: Sequence[Any], output: torch.Tensor
    ) -> tuple[Any, ...]:
        return inputs[0], output

    return make_single_kernel_run_fn(
        PreparedSingleKernelRunSpec(
            kernel=PreparedSingleKernelSpec(
                kernel=_op_a_kernel,
                grid=grid,
                static_args=static_tail,
                constexpr_kwargs=constexpr_kwargs,
                build_cached_call=build_cached_call,
            ),
            input_checks=checks,
            output_factory=make_output,
            runtime_args=runtime_args,
        ),
        default_run_fn,
    )
```

3. 固定顺序多 kernel fast path 使用 `PreparedKernelPipelineSpec`。典型场景是 `zero -> compute`、`split -> combine` 或 `zero -> compute -> reduce`。开发者只描述 context、每个 step 的 runtime args 和 cached call；框架会为每个 step 独立缓存 compiled kernel：

```python
from typing import Any, NamedTuple, Optional, Sequence

import torch
import triton

from flag_dnn.graph.prepared import (
    PreparedKernelPipelineSpec,
    PreparedPipelineStepSpec,
    RunFn,
    make_kernel_pipeline_run_fn,
    make_static_cached_call,
    register_prepared_run_fn,
    runtime_tensor_checks_from_specs,
)
from flag_dnn.graph.prepared.common import _static_shape
from flag_dnn.graph.tensor import TensorSpec, torch_dtype
from flag_dnn.ops.op_a import _op_a_compute_kernel, _op_a_zero_kernel


class OpAContext(NamedTuple):
    output: torch.Tensor
    workspace: torch.Tensor


@register_prepared_run_fn("op_a")
def _prepare_op_a_pipeline(
    attrs: dict[str, Any],
    input_specs: Sequence[TensorSpec],
    default_run_fn: RunFn,
) -> Optional[RunFn]:
    if len(input_specs) != 1:
        return None
    shape = _static_shape(input_specs[0])
    if shape is None:
        return None
    checks = runtime_tensor_checks_from_specs(input_specs, (0,))
    if checks is None:
        return None

    dtype = torch_dtype(input_specs[0].dtype)
    n = int(shape[-1])
    out_shape = shape
    workspace_shape = (triton.cdiv(n, 256), 256)
    workspace_numel = workspace_shape[0] * workspace_shape[1]
    zero_grid = (triton.cdiv(workspace_numel, 1024), 1, 1)
    zero_tail = (workspace_numel, 1024)
    compute_tail = (n,)
    compute_constexpr = {"BLOCK_D": triton.next_power_of_2(n)}

    def compute_grid(meta: dict[str, Any]) -> tuple[int, int, int]:
        return (triton.cdiv(n, meta["BLOCK_M"]), 1, 1)

    def build_compute_cached_call(
        meta: dict[str, Any],
    ) -> tuple[tuple[int, ...], tuple[Any, ...]]:
        block_m = int(meta["BLOCK_M"])
        return (triton.cdiv(n, block_m), 1, 1), compute_tail + (block_m,)

    def make_context(inputs: Sequence[Any]) -> OpAContext:
        x = inputs[0]
        assert isinstance(x, torch.Tensor)
        output = torch.empty(out_shape, dtype=dtype, device=x.device)
        workspace = torch.empty(
            workspace_shape, dtype=torch.float32, device=x.device
        )
        return OpAContext(output=output, workspace=workspace)

    def zero_args(
        inputs: Sequence[Any], context: Any
    ) -> tuple[Any, ...]:
        return (context.workspace,)

    def compute_args(
        inputs: Sequence[Any], context: Any
    ) -> tuple[Any, ...]:
        return inputs[0], context.workspace, context.output

    return make_kernel_pipeline_run_fn(
        PreparedKernelPipelineSpec(
            steps=(
                PreparedPipelineStepSpec(
                    kernel=_op_a_zero_kernel,
                    grid=zero_grid,
                    runtime_args=zero_args,
                    static_args=zero_tail,
                    build_cached_call=make_static_cached_call(
                        zero_grid, zero_tail
                    ),
                ),
                PreparedPipelineStepSpec(
                    kernel=_op_a_compute_kernel,
                    grid=compute_grid,
                    runtime_args=compute_args,
                    static_args=compute_tail,
                    constexpr_kwargs=compute_constexpr,
                    build_cached_call=build_compute_cached_call,
                    first_launch_returns_metadata=True,
                ),
            ),
            input_checks=checks,
            context_factory=make_context,
            result=lambda context: context.output,
        ),
        default_run_fn,
    )
```

4. 选择模板时遵循下面规则：单个 launch 且只需要一个输出时用 `PreparedSingleKernelRunSpec`；kernel 顺序固定、需要 workspace 或多步结果合成时用 `PreparedKernelPipelineSpec`；kernel 选择会随 replay 输入动态变化时，把不同模式拆成多个 preparer 条件，条件不满足时返回 `None` 或 fallback。所有可调 `BLOCK_*`、tile、split、warps/stages 参数都应接入 `tune_configs.yaml` 和 `runtime.get_tuned_config()`，不要在最终性能路径长期硬编码。

5. 添加 `tests_graph/test_graph_op_a.py` 和 `benchmark_graph/test_op_a_perf.py`。如果 prepared path 是性能关键路径，benchmark 必须覆盖触发 prepared fast path 的 shape，并确认没有低于 `speedup >= 0.9` 的必测 case。

## Prepared Fast Path 规则

- `prepare_run_fn` 只负责按注册表顺序分发：先尝试 generic preparer，再尝试 op-specific preparer。新算子不应再向 `prepare_run_fn` 添加新的 hard-coded 分支。
- 当静态 shape、stride、dtype、attrs、device 或 layout 不满足 fast path 约束时，preparer 必须返回 `None`。内部 helper 例如 `_prepare_binary_pointwise` 不直接注册；它们由已注册的 family preparer 调用。
- prepared 闭包必须重新检查 replay 时可能变化的 runtime 假设，例如 tensor stride。假设变化时调用 `default_run_fn`。如果检查只来自 `TensorSpec` 的静态 shape/stride/dtype，优先用 `runtime_tensor_checks_from_specs(input_specs, indices)` 生成 `RuntimeTensorCheck`，不要重复手写检查元组。
- Triton compiled kernel 应缓存在闭包内部。单 kernel 路径优先通过 `PreparedSingleKernelRunSpec` 自动生成 replay closure；固定顺序的多 kernel 路径优先通过 `PreparedKernelPipelineSpec` 自动生成 replay closure。需要手写特殊 replay 逻辑时，再通过底层 launcher 或手写 closure 缓存 compiled kernel、静态 grid 和静态参数 tail。第一次 replay 可以触发 autotune 或 compile，后续 replay 直接 launch。
- 如果 graph replay 需要复用稳定的输出或 workspace tensor，例如 CUDA graph 友好的 convolution prepared path，优先使用 `PreparedTensorCache` 和 `get_cached_empty_tensor`，不要在每个算子里重复写 cache get/create 样板。
- graph replay 闭包中只做轻量工作：输出分配、廉价假设校验和 kernel launch。不要在闭包里重复 shape 解析、config lookup 或复杂 dispatch。
- eager dispatch 与 prepared dispatch 对同一受支持 shape 应选择等价算法。如果新增 graph-only fast path，必须在测试和 benchmark 中说明它为什么只存在于 graph 路径。

## 以 SDPA 为参考

SDPA 比多数算子复杂，它的实现闭环包括：

- `src/flag_dnn/ops/sdpa.py`：forward eager 语义和 Triton kernels。
- `src/flag_dnn/ops/sdpa_backward.py`：backward eager 语义和 Triton kernels。
- `src/flag_dnn/graph/registry.py`：`sdpa` 和 `sdpa_backward` 的 graph schema。
- `src/flag_dnn/graph/prepared/sdpa_forward.py`、`src/flag_dnn/graph/prepared/sdpa_backward.py`：graph 编译期 fast path，例如 decode、dense exact、GQA causal 和 fused atomic backward。
- `src/flag_dnn/runtime/backend/_nvidia/tune_configs.yaml`：SDPA 相关 tune keys。
- `tests_graph/test_graph_sdpa.py`、`tests_graph/test_graph_sdpa_backward.py`、`benchmark_graph/test_sdpa_perf.py`、`benchmark_graph/test_sdpa_backward_perf.py`：正确性和性能覆盖。

可以参考 SDPA 学习复杂算子如何在不改变 public semantics 的前提下专门优化 graph replay。但不要把 SDPA 的复杂度直接复制到简单算子中。通常应先完成 eager 语义、graph schema、自动调优、功能测试和 benchmark，再根据实测瓶颈添加 prepared fast path。

## 性能守则

- Triton kernel 存在可调 meta 参数时，最终性能路径必须接入仓库级自动调优机制。
- 不要用最终硬编码 `triton.Config` 列表替代 `tune_configs.yaml` 和 `runtime.get_tuned_config()`。
- 归因 kernel 慢之前，尽量先量化 graph path、direct eager path 和 prepared path 的差异。小算子可能主要受 graph replay 或 dispatch 开销影响。
- 只有所有必测性能 shape 都达到 `speedup >= 0.9`，性能才算合格。
- 不要把低 speedup 实现报告为已完成。需要基于 benchmark、PTX/SASS/NCU 等证据定位真实瓶颈，并继续迭代。
