# 算子开发指南

新增一个 FlagDNN graph 算子要改哪些文件、各加什么代码。按下表顺序做即可；大多数算子只涉及前 3 步加测试。

## 文件清单

| # | 文件 | 加什么 | 何时改 |
|---|---|---|---|
| 1 | `src/flag_dnn/ops/<op>.py` | eager 算子：public API、输入校验、输出分配、Triton kernel dispatch | 必改 |
| 2 | `src/flag_dnn/ops/__init__.py`、`src/flag_dnn/__init__.py` | `import` 并导出该算子 | 必改 |
| 3 | `src/flag_dnn/graph/registry/schemas/<family>.py` | 该算子的 `_normalize` + `_shape` + `_run`，并在 `register()` 加一行注册 | 需要被 `@flag_dnn.graph` capture 时 |
| 4 | `src/flag_dnn/graph/prepared/<family>.py` | `@register_prepared_run_fn` 注册一个编译期 fast path | 可选，仅当 replay 是性能瓶颈 |
| 5 | `src/flag_dnn/runtime/backend/_nvidia/tune_configs.yaml` | kernel 的 tune key | kernel 有可调 `BLOCK_*`/`num_warps` 等 meta 参数时 |
| 6 | `tests_graph/test_<op>.py`、`tests_graph/consts.py` | cuDNN graph 对标测试 + shape 集合 | 必改 |
| 7 | `benchmark_graph/test_<op>_perf.py`、`benchmark_graph/consts.py` | 必测性能 shape | 必改 |

## 第 3 步：graph 算子定义（最常见的工作）

在算子所属 family 的 `schemas/<family>.py` 里**一处**完成下面三个函数，再到该文件的 `register(flag_ops)` 末尾加一行注册。不需要改动任何其它文件——capture wrapper 会自动从注册信息派生。

```python
# 1) 参数 → (graph input ids, attrs)；简单算子直接复用 _normalize_unary / _normalize_binary
def _normalize_<op>(ctx, args, kwargs):
    return [ctx.as_value(args[0], "input")], {...}

# 2) 输出 TensorSpec 推断；输出形状同输入时直接用 common._shape_like_first
def _<op>_shape(input_specs, attrs):
    return [TensorSpec(...)]

# 3) eager 回退（prepared fast path 不适用时保证正确）；共享 helper 见 _run_common
def _run_<op>(flag_ops):
    def run(inputs, attrs):
        _require_runtime_backend(inputs, "<op>")
        return flag_ops.<op>(inputs[0], ...)
    return run

# 4) 在 register(flag_ops) 里加一行
register_op_def(OpDef(
    name="<op>",
    normalize=_normalize_<op>,
    shape=_<op>_shape,
    run=_run_<op>(flag_ops),
    # 可选：num_outputs=<int> 或 (1, 2)；fusible=True；output_keys=(...)（eager 返回 dict 时）
))
```

现有 family 文件：`pointwise` / `utility` / `conv` / `matmul_attention` / `norm_reduction` / `fused`。共享 helper：normalize/shape 在 `schemas/common.py`，run 在 `schemas/_run_common.py`。

**新增整个 family** 时：新建 `schemas/<family>.py`（含上述函数和一个 `register(flag_ops)`），并在 `registry/ops.py` 的 `register_default_ops()` 加一行 `<family>.register(flag_ops)`。

## 第 4 步：编译期 fast path（可选）

只有当 graph replay 成为性能瓶颈、需要静态绑定 shape/stride/grid/kernel constexpr 时才加。在 `prepared/<family>.py` 用 `@register_prepared_run_fn("<op>")`（或多算子共享的 `@register_generic_prepared_run_fn`）注册一个 preparer：在闭包外预计算静态量、返回一个轻量的 `run(inputs, attrs)` 闭包；约束不满足时返回 `None` 或调用 `default_run_fn`。新 family 需在 `prepared/ops.py` 加一行 side-effect import 触发注册。

可直接复制的写法参考：单 kernel 用 `PreparedSingleKernelRunSpec` + `make_single_kernel_run_fn`（见 `prepared/conv.py`）；固定顺序多 kernel 用 `PreparedKernelPipelineSpec`。kernel 的可调 meta 参数务必接入 `tune_configs.yaml` 与 `runtime.get_tuned_config()`，不要长期硬编码。

## 验证

- 改动 registry 后先跑 `tests_graph/test_registry.py`（自检 schema 与 capture wrapper 一致）。
- 完整复杂算子参考 SDPA：`ops/sdpa.py` + `schemas/matmul_attention.py` + `prepared/sdpa_forward.py` + `tests_graph/test_sdpa.py`。
