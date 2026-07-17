from __future__ import annotations

from typing import Any, Optional, Sequence

import torch

from flag_dnn import runtime
from flag_dnn.graph.device import is_runtime_device_tensor
from flag_dnn.graph.prepared import (
    PreparedSingleKernelRunSpec,
    PreparedSingleKernelSpec,
    RunFn,
    make_single_kernel_run_fn,
    register_generic_prepared_run_fn,
    runtime_tensor_checks_from_specs,
)
from flag_dnn.graph.prepared.common import (
    _is_runtime_device_spec,
    _require_runtime_backend,
    _unsupported_triton_path,
)
from flag_dnn.graph.tensor import TensorSpec, torch_dtype

# Pointwise prepared paths

_POINTWISE_BINARY_OPS = {
    "add",
    "sub",
    "mul",
    "div",
    "mod",
    "max",
    "min",
    "minimum",
    "maximum",
    "eq",
    "ne",
    "lt",
    "le",
    "gt",
    "ge",
}

_POINTWISE_CMP_REVERSE = {
    "eq": "eq",
    "ne": "ne",
    "lt": "gt",
    "le": "ge",
    "gt": "lt",
    "ge": "le",
}

_POINTWISE_CMP_ALIASES = {
    "cmp_eq": "eq",
    "cmp_neq": "ne",
    "cmp_lt": "lt",
    "cmp_le": "le",
    "cmp_gt": "gt",
    "cmp_ge": "ge",
}

_POINTWISE_BINARY_KERNEL_OPS = {
    "add": "add",
    "sub": "sub",
    "mul": "mul",
    "div": "div",
    "mod": "mod",
    "max": "max",
    "min": "minimum",
    "minimum": "minimum",
    "maximum": "maximum",
    "eq": "eq",
    "ne": "ne",
    "lt": "lt",
    "le": "le",
    "gt": "gt",
    "ge": "ge",
}

_POINTWISE_COMPARISON_OPS = {"eq", "ne", "lt", "le", "gt", "ge"}
_POINTWISE_FAST_DTYPES = {"float16", "bfloat16", "float32"}


@register_generic_prepared_run_fn
def _prepare_pointwise(
    op_type: str,
    attrs: dict[str, Any],
    input_specs: Sequence[TensorSpec],
    default_run_fn: RunFn,
) -> Optional[RunFn]:
    if not input_specs or not all(
        _is_runtime_device_spec(spec) for spec in input_specs
    ):
        return None

    actual_op_type = _POINTWISE_CMP_ALIASES.get(
        op_type, attrs.get("op_type", op_type)
    )
    if actual_op_type in _POINTWISE_BINARY_OPS:
        return _prepare_binary_pointwise(
            actual_op_type, attrs, input_specs, default_run_fn
        )
    if actual_op_type == "binary_select":
        return _prepare_binary_select_pointwise(input_specs, default_run_fn)
    if actual_op_type == "add_square":
        return _prepare_add_square_pointwise(
            attrs, input_specs, default_run_fn
        )
    if actual_op_type == "pow":
        prepared = _prepare_dense_tensor_pow(
            attrs, input_specs, default_run_fn
        )
        if prepared is not None:
            return prepared
        return _prepare_pow_pointwise(attrs, default_run_fn)
    if op_type == "abs":
        from flag_dnn.ops.abs import abs as abs_op

        def run(inputs: Sequence[Any], _attrs: dict[str, Any]) -> Any:
            _require_runtime_backend(inputs, "abs")
            return abs_op(inputs[0])

        return run
    if op_type == "sigmoid":
        from flag_dnn.ops.sigmoid import sigmoid

        compute_data_type = attrs.get("compute_data_type")
        name = attrs.get("name", "")

        def run(inputs: Sequence[Any], _attrs: dict[str, Any]) -> Any:
            _require_runtime_backend(inputs, "sigmoid")
            return sigmoid(
                inputs[0], compute_data_type=compute_data_type, name=name
            )

        return run
    if op_type == "sigmoid_backward":
        from flag_dnn.ops.sigmoid_backward import sigmoid_backward

        compute_data_type = attrs.get("compute_data_type")
        name = attrs.get("name", "")

        def run(inputs: Sequence[Any], _attrs: dict[str, Any]) -> Any:
            _require_runtime_backend(inputs, "sigmoid_backward")
            return sigmoid_backward(
                inputs[0],
                inputs[1],
                compute_data_type=compute_data_type,
                name=name,
            )

        return run
    return None


def _pointwise_operands(
    inputs: Sequence[Any], attrs: dict[str, Any]
) -> tuple[Any, Any]:
    left = inputs[0]
    if len(inputs) > 1:
        right = inputs[1]
    else:
        right = attrs["other"]
    if attrs.get("reverse"):
        return right, left
    return left, right


def _static_int_shape(spec: TensorSpec) -> Optional[tuple[int, ...]]:
    shape = tuple(spec.shape)
    if not all(isinstance(dim, int) for dim in shape):
        return None
    return tuple(int(dim) for dim in shape)


def _is_dense_flat_spec(spec: TensorSpec) -> bool:
    return spec.layout in ("contiguous", "nhwc") and spec.stride is not None


def _prepare_dense_tensor_binary(
    op_type: str,
    attrs: dict[str, Any],
    input_specs: Sequence[TensorSpec],
    default_run_fn: RunFn,
) -> Optional[RunFn]:
    kernel_op_type = _POINTWISE_BINARY_KERNEL_OPS.get(op_type)
    if kernel_op_type is None or attrs.get("reverse"):
        return None
    if len(input_specs) != 2:
        return None
    if attrs.get("rounding_mode") is not None:
        return None

    try:
        alpha = float(attrs.get("alpha", 1))
    except (TypeError, ValueError):
        return None

    left_spec, right_spec = input_specs
    shape = _static_int_shape(left_spec)
    if shape is None or shape != _static_int_shape(right_spec):
        return None
    if (
        not _is_dense_flat_spec(left_spec)
        or not _is_dense_flat_spec(right_spec)
        or left_spec.stride != right_spec.stride
        or left_spec.dtype != right_spec.dtype
        or left_spec.dtype not in _POINTWISE_FAST_DTYPES
    ):
        return None

    input_checks = runtime_tensor_checks_from_specs(
        input_specs,
        (0, 1),
        require_shape=True,
        require_stride=True,
        require_dtype=True,
    )
    if input_checks is None:
        return None

    from flag_dnn.ops.binary import binary_tensor_kernel

    out_dtype = (
        torch.bool
        if kernel_op_type in _POINTWISE_COMPARISON_OPS
        else torch_dtype(left_spec.dtype)
    )
    stride = tuple(int(item) for item in left_spec.stride or ())
    n_elements = 1
    for dim in shape:
        n_elements *= dim
    output_cache: dict[tuple[Any, ...], torch.Tensor] = {}

    def output_factory(inputs: Sequence[Any]) -> torch.Tensor:
        source = inputs[0]
        assert isinstance(source, torch.Tensor)
        key = (
            source.device.type,
            source.device.index,
            out_dtype,
            shape,
            stride,
        )
        output = output_cache.get(key)
        if output is None:
            output = torch.empty_strided(
                shape, stride, device=source.device, dtype=out_dtype
            )
            output_cache[key] = output
        return output

    def default_runtime_args(
        inputs: Sequence[Any], output: torch.Tensor
    ) -> tuple[Any, ...]:
        if runtime.device.vendor_name == "ascend" and kernel_op_type == "add":
            return (inputs[0], inputs[1], output, alpha)
        return (inputs[0], inputs[1], output, n_elements, alpha)

    runtime_args = default_runtime_args
    kernel = binary_tensor_kernel
    constexpr_kwargs: dict[str, Any] = {
        "ROUND_MODE": 0,
        "OP_TYPE": kernel_op_type,
    }

    if runtime.device.vendor_name == "ascend" and kernel_op_type == "add":
        from flag_dnn.runtime.backend._ascend.ops.binary import (
            add_tensor_aligned_core_loop_kernel,
            add_tensor_core_loop_kernel,
            can_use_aligned_core_loop,
            get_add_block_size,
            make_core_loop_grid,
        )

        block_size = get_add_block_size(
            n_elements, left_spec.dtype, left_spec.device
        )
        grid = make_core_loop_grid(n_elements, left_spec.device)
        alpha_is_one = alpha == 1.0
        if alpha_is_one and can_use_aligned_core_loop(n_elements, block_size):
            kernel = add_tensor_aligned_core_loop_kernel
            program_count = grid({"BLOCK_SIZE": block_size})[0]
            blocks_per_program = n_elements // block_size // program_count
            constexpr_kwargs = {
                "BLOCKS_PER_PROGRAM": blocks_per_program,
                "BLOCK_SIZE": block_size,
                "num_warps": 4,
                "num_stages": 1,
            }

            def aligned_runtime_args(
                inputs: Sequence[Any], output: torch.Tensor
            ) -> tuple[Any, ...]:
                return (inputs[0], inputs[1], output)

            runtime_args = aligned_runtime_args

            def build_cached_call(
                constexprs: dict[str, Any]
            ) -> tuple[tuple[int, ...], tuple[Any, ...]]:
                static_grid = (*grid({"BLOCK_SIZE": block_size}), 1, 1)
                return static_grid, (blocks_per_program, block_size)

        else:
            kernel = add_tensor_core_loop_kernel
            constexpr_kwargs = {
                "N_ELEMENTS": n_elements,
                "ALPHA_IS_ONE": alpha_is_one,
                "ALIGNED_BLOCKS": (
                    n_elements >= 262144 and n_elements % block_size == 0
                ),
                "BLOCK_SIZE": block_size,
                "num_warps": 4,
                "num_stages": 1,
            }

            def build_cached_call(
                constexprs: dict[str, Any]
            ) -> tuple[tuple[int, ...], tuple[Any, ...]]:
                static_grid = (*grid({"BLOCK_SIZE": block_size}), 1, 1)
                return static_grid, (
                    n_elements,
                    alpha_is_one,
                    n_elements >= 262144 and n_elements % block_size == 0,
                    block_size,
                )

    else:

        def grid(meta: dict[str, Any]) -> tuple[int, ...]:
            block_size = int(meta["BLOCK_SIZE"])
            return ((n_elements + block_size - 1) // block_size,)

        def build_cached_call(
            constexprs: dict[str, Any]
        ) -> tuple[tuple[int, ...], tuple[Any, ...]]:
            block_size = int(constexprs["BLOCK_SIZE"])
            static_grid = (
                (n_elements + block_size - 1) // block_size,
                1,
                1,
            )
            return static_grid, (0, kernel_op_type, block_size)

    def extra_check(inputs: Sequence[Any]) -> bool:
        left, right = inputs
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and is_runtime_device_tensor(left)
            and is_runtime_device_tensor(right)
            and left.device == right.device
        )

    return make_single_kernel_run_fn(
        PreparedSingleKernelRunSpec(
            kernel=PreparedSingleKernelSpec(
                kernel=kernel,
                grid=grid,
                static_args=(),
                constexpr_kwargs=constexpr_kwargs,
                build_cached_call=build_cached_call,
            ),
            input_checks=input_checks,
            output_factory=output_factory,
            runtime_args=runtime_args,
            extra_check=extra_check,
            validate_inputs=bool(attrs.get("_validate_inputs", True)),
        ),
        default_run_fn,
    )


def _prepare_binary_select_pointwise(
    input_specs: Sequence[TensorSpec],
    default_run_fn: RunFn,
) -> Optional[RunFn]:
    if len(input_specs) != 3:
        return None

    input0_spec, input1_spec, mask_spec = input_specs
    shape = _static_int_shape(input0_spec)
    if (
        shape is None
        or shape != _static_int_shape(input1_spec)
        or shape != _static_int_shape(mask_spec)
    ):
        return None
    if (
        not _is_dense_flat_spec(input0_spec)
        or not _is_dense_flat_spec(input1_spec)
        or not _is_dense_flat_spec(mask_spec)
        or input0_spec.stride != input1_spec.stride
        or input0_spec.stride != mask_spec.stride
        or input0_spec.dtype != input1_spec.dtype
        or input0_spec.dtype not in _POINTWISE_FAST_DTYPES
    ):
        return None

    input_checks = runtime_tensor_checks_from_specs(
        input_specs,
        (0, 1, 2),
        require_shape=True,
        require_stride=True,
        require_dtype=True,
    )
    if input_checks is None:
        return None

    from flag_dnn.ops.binary_select import binary_select_tensor_kernel

    out_dtype = torch_dtype(input0_spec.dtype)
    stride = tuple(int(item) for item in input0_spec.stride or ())
    n_elements = 1
    for dim in shape:
        n_elements *= dim
    output_cache: dict[tuple[Any, ...], torch.Tensor] = {}

    def output_factory(inputs: Sequence[Any]) -> torch.Tensor:
        source = inputs[0]
        assert isinstance(source, torch.Tensor)
        key = (
            source.device.type,
            source.device.index,
            out_dtype,
            shape,
            stride,
        )
        output = output_cache.get(key)
        if output is None:
            output = torch.empty_strided(
                shape, stride, device=source.device, dtype=out_dtype
            )
            output_cache[key] = output
        return output

    def runtime_args(
        inputs: Sequence[Any], output: torch.Tensor
    ) -> tuple[Any, ...]:
        return (inputs[0], inputs[1], inputs[2], output, n_elements)

    def grid(meta: dict[str, Any]) -> tuple[int, ...]:
        block_size = int(meta["BLOCK_SIZE"])
        return ((n_elements + block_size - 1) // block_size,)

    def build_cached_call(
        constexprs: dict[str, Any]
    ) -> tuple[tuple[int, ...], tuple[Any, ...]]:
        block_size = int(constexprs["BLOCK_SIZE"])
        static_grid = ((n_elements + block_size - 1) // block_size, 1, 1)
        return static_grid, (block_size,)

    def extra_check(inputs: Sequence[Any]) -> bool:
        input0, input1, mask = inputs
        return (
            isinstance(input0, torch.Tensor)
            and isinstance(input1, torch.Tensor)
            and isinstance(mask, torch.Tensor)
            and is_runtime_device_tensor(input0)
            and is_runtime_device_tensor(input1)
            and is_runtime_device_tensor(mask)
            and input0.device == input1.device
            and input0.device == mask.device
        )

    return make_single_kernel_run_fn(
        PreparedSingleKernelRunSpec(
            kernel=PreparedSingleKernelSpec(
                kernel=binary_select_tensor_kernel,
                grid=grid,
                static_args=(),
                constexpr_kwargs={},
                build_cached_call=build_cached_call,
            ),
            input_checks=input_checks,
            output_factory=output_factory,
            runtime_args=runtime_args,
            extra_check=extra_check,
        ),
        default_run_fn,
    )


def _prepare_dense_tensor_pow(
    attrs: dict[str, Any],
    input_specs: Sequence[TensorSpec],
    default_run_fn: RunFn,
) -> Optional[RunFn]:
    if attrs.get("reverse") or len(input_specs) != 2:
        return None

    left_spec, right_spec = input_specs
    shape = _static_int_shape(left_spec)
    if shape is None or shape != _static_int_shape(right_spec):
        return None
    if (
        not _is_dense_flat_spec(left_spec)
        or not _is_dense_flat_spec(right_spec)
        or left_spec.stride != right_spec.stride
        or left_spec.dtype != right_spec.dtype
        or left_spec.dtype not in _POINTWISE_FAST_DTYPES
    ):
        return None

    input_checks = runtime_tensor_checks_from_specs(
        input_specs,
        (0, 1),
        require_shape=True,
        require_stride=True,
        require_dtype=True,
    )
    if input_checks is None:
        return None

    from flag_dnn.ops.pow import pow_tensor_kernel

    out_dtype = torch_dtype(left_spec.dtype)
    stride = tuple(int(item) for item in left_spec.stride or ())
    n_elements = 1
    for dim in shape:
        n_elements *= dim
    output_cache: dict[tuple[Any, ...], torch.Tensor] = {}

    def output_factory(inputs: Sequence[Any]) -> torch.Tensor:
        source = inputs[0]
        assert isinstance(source, torch.Tensor)
        key = (
            source.device.type,
            source.device.index,
            out_dtype,
            shape,
            stride,
        )
        output = output_cache.get(key)
        if output is None:
            output = torch.empty_strided(
                shape, stride, device=source.device, dtype=out_dtype
            )
            output_cache[key] = output
        return output

    def runtime_args(
        inputs: Sequence[Any], output: torch.Tensor
    ) -> tuple[Any, ...]:
        return (inputs[0], inputs[1], output, n_elements)

    def grid(meta: dict[str, Any]) -> tuple[int, ...]:
        block_size = int(meta["BLOCK_SIZE"])
        return ((n_elements + block_size - 1) // block_size,)

    def build_cached_call(
        constexprs: dict[str, Any]
    ) -> tuple[tuple[int, ...], tuple[Any, ...]]:
        block_size = int(constexprs["BLOCK_SIZE"])
        static_grid = ((n_elements + block_size - 1) // block_size, 1, 1)
        return static_grid, (block_size,)

    def extra_check(inputs: Sequence[Any]) -> bool:
        left, right = inputs
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and is_runtime_device_tensor(left)
            and is_runtime_device_tensor(right)
            and left.device == right.device
        )

    return make_single_kernel_run_fn(
        PreparedSingleKernelRunSpec(
            kernel=PreparedSingleKernelSpec(
                kernel=pow_tensor_kernel,
                grid=grid,
                static_args=(),
                constexpr_kwargs={},
                build_cached_call=build_cached_call,
            ),
            input_checks=input_checks,
            output_factory=output_factory,
            runtime_args=runtime_args,
            extra_check=extra_check,
        ),
        default_run_fn,
    )


def _prepare_add_square_pointwise(
    attrs: dict[str, Any],
    input_specs: Sequence[TensorSpec],
    default_run_fn: RunFn,
) -> Optional[RunFn]:
    if len(input_specs) != 2:
        return None

    left_spec, right_spec = input_specs
    shape = _static_int_shape(left_spec)
    if shape is None or shape != _static_int_shape(right_spec):
        return None
    if (
        not _is_dense_flat_spec(left_spec)
        or not _is_dense_flat_spec(right_spec)
        or left_spec.stride != right_spec.stride
        or left_spec.dtype != right_spec.dtype
        or left_spec.dtype not in _POINTWISE_FAST_DTYPES
    ):
        return None

    input_checks = runtime_tensor_checks_from_specs(
        input_specs,
        (0, 1),
        require_shape=True,
        require_stride=True,
        require_dtype=True,
    )
    if input_checks is None:
        return None

    from flag_dnn.ops.add_square import (
        _compute_uses_float32,
        add_square_tensor_kernel,
    )

    compute_float32 = _compute_uses_float32(attrs.get("compute_data_type"))
    out_dtype = torch_dtype(left_spec.dtype)
    stride = tuple(int(item) for item in left_spec.stride or ())
    n_elements = 1
    for dim in shape:
        n_elements *= dim
    output_cache: dict[tuple[Any, ...], torch.Tensor] = {}

    def output_factory(inputs: Sequence[Any]) -> torch.Tensor:
        source = inputs[0]
        assert isinstance(source, torch.Tensor)
        key = (
            source.device.type,
            source.device.index,
            out_dtype,
            shape,
            stride,
        )
        output = output_cache.get(key)
        if output is None:
            output = torch.empty_strided(
                shape, stride, device=source.device, dtype=out_dtype
            )
            output_cache[key] = output
        return output

    def runtime_args(
        inputs: Sequence[Any], output: torch.Tensor
    ) -> tuple[Any, ...]:
        return (inputs[0], inputs[1], output, n_elements)

    def grid(meta: dict[str, Any]) -> tuple[int, ...]:
        block_size = int(meta["BLOCK_SIZE"])
        tiles = int(meta["TILES_PER_PROGRAM"])
        return ((n_elements + block_size * tiles - 1) // (block_size * tiles),)

    def build_cached_call(
        constexprs: dict[str, Any]
    ) -> tuple[tuple[int, ...], tuple[Any, ...]]:
        block_size = int(constexprs["BLOCK_SIZE"])
        tiles = int(constexprs["TILES_PER_PROGRAM"])
        static_grid = (
            (n_elements + block_size * tiles - 1) // (block_size * tiles),
            1,
            1,
        )
        return static_grid, (compute_float32, block_size, tiles)

    def extra_check(inputs: Sequence[Any]) -> bool:
        left, right = inputs
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and is_runtime_device_tensor(left)
            and is_runtime_device_tensor(right)
            and left.device == right.device
        )

    return make_single_kernel_run_fn(
        PreparedSingleKernelRunSpec(
            kernel=PreparedSingleKernelSpec(
                kernel=add_square_tensor_kernel,
                grid=grid,
                static_args=(),
                constexpr_kwargs={"COMPUTE_FLOAT32": compute_float32},
                build_cached_call=build_cached_call,
            ),
            input_checks=input_checks,
            output_factory=output_factory,
            runtime_args=runtime_args,
            extra_check=extra_check,
        ),
        default_run_fn,
    )


def _prepare_binary_pointwise(
    op_type: str,
    attrs: dict[str, Any],
    input_specs: Sequence[TensorSpec],
    default_run_fn: RunFn,
) -> RunFn:
    prepared = _prepare_dense_tensor_binary(
        op_type, attrs, input_specs, default_run_fn
    )
    if prepared is not None:
        return prepared

    from flag_dnn.ops.binary import binary

    alpha = attrs.get("alpha", 1)
    rounding_mode = attrs.get("rounding_mode")

    def run(inputs: Sequence[Any], _attrs: dict[str, Any]) -> Any:
        _require_runtime_backend(inputs, op_type)
        left, right = _pointwise_operands(inputs, attrs)
        if op_type == "add":
            if isinstance(left, torch.Tensor):
                return binary(left, right, alpha=alpha, op_type="add")
            if isinstance(right, torch.Tensor):
                return binary(right, left, alpha=alpha, op_type="add")
        elif op_type == "sub":
            if isinstance(left, torch.Tensor):
                return binary(left, right, alpha=alpha, op_type="sub")
        elif op_type == "mul":
            if isinstance(left, torch.Tensor):
                return binary(left, right, op_type="mul")
            if isinstance(right, torch.Tensor):
                return binary(right, left, op_type="mul")
        elif op_type == "div":
            if isinstance(left, torch.Tensor):
                return binary(
                    left,
                    right,
                    rounding_mode=rounding_mode,
                    op_type="div",
                )
        elif op_type == "mod":
            if isinstance(left, torch.Tensor):
                return binary(left, right, op_type="mod")
        elif op_type == "max":
            if isinstance(left, torch.Tensor):
                return binary(left, right, op_type="max")
            if isinstance(right, torch.Tensor):
                return binary(right, left, op_type="max")
        elif op_type in ("min", "minimum"):
            if isinstance(left, torch.Tensor):
                return binary(left, right, op_type="minimum")
            if isinstance(right, torch.Tensor):
                return binary(right, left, op_type="minimum")
        elif op_type == "maximum":
            if isinstance(left, torch.Tensor):
                return binary(left, right, op_type="maximum")
            if isinstance(right, torch.Tensor):
                return binary(right, left, op_type="maximum")
        else:
            if isinstance(left, torch.Tensor):
                return binary(left, right, op_type=op_type)
            if isinstance(right, torch.Tensor):
                return binary(
                    right,
                    left,
                    op_type=_POINTWISE_CMP_REVERSE[op_type],
                )
        _unsupported_triton_path(op_type, "operand combination")

    return run


def _prepare_pow_pointwise(
    attrs: dict[str, Any], default_run_fn: RunFn
) -> RunFn:
    from flag_dnn.ops.pow import pow as pow_op

    compute_data_type = attrs.get("compute_data_type")
    name = attrs.get("name", "")

    def run(inputs: Sequence[Any], _attrs: dict[str, Any]) -> Any:
        _require_runtime_backend(inputs, "pow")
        left, right = _pointwise_operands(inputs, attrs)
        if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
            return pow_op(
                left, right, compute_data_type=compute_data_type, name=name
            )
        _unsupported_triton_path("pow", "two scalar operands")

    return run
