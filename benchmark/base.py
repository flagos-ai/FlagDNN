"""FlagGems-style benchmark base classes for FlagDNN.

New performance tests should import this module and benchmark the same torch
operator twice: baseline PyTorch first, then PyTorch under flag_dnn.use_dnn().
Existing tests that still import performance_utils keep working unchanged.
"""

import math
from typing import Generator

import torch

from flag_dnn.utils import shape_utils

from . import consts
from .conftest import Config
from .performance_utils import (  # noqa: F401
    Benchmark,
    GenericBenchmark,
    GenericBenchmark2DOnly,
    GenericBenchmark4DOnly,
    GenericBenchmarkExcluse1D,
    GenericBenchmarkExcluse3D,
    GenericBenchmarkFilterShapes,
)

MAX_BENCH_BYTES = 8 * 1024**3


def _estimated_tensor_bytes(shape, dtype, tensors):
    element_size = torch.empty((), dtype=dtype).element_size()
    return math.prod(shape) * element_size * tensors


def _within_memory_budget(shape, dtype, tensors):
    return _estimated_tensor_bytes(shape, dtype, tensors) <= MAX_BENCH_BYTES


def generate_tensor_input(shape, dtype, device):
    if dtype.is_floating_point:
        return torch.randn(shape, dtype=dtype, device=device)
    if dtype == torch.bool:
        return torch.randint(0, 2, shape, dtype=dtype, device="cpu").to(device)
    if dtype.is_complex:
        return torch.randn(shape, dtype=dtype, device=device)
    return torch.randint(
        torch.iinfo(dtype).min,
        torch.iinfo(dtype).max,
        shape,
        dtype=dtype,
        device="cpu",
    ).to(device)


class UnaryPointwiseBenchmark(Benchmark):
    DEFAULT_METRICS = consts.DEFAULT_METRICS[:] + ["gbps", "tflops"]

    def set_more_shapes(self):
        special_shapes_1d = [
            (1,),
            (16,),
            (127,),
            (1024,),
            (65536,),
            (1024 * 1024,),
            (16 * 1024 * 1024,),
        ]
        special_shapes_2d = [
            (17, 31),
            (1024, 1025),
            (4096, 4096),
        ]
        model_shapes = [
            (1, 2048, 4096),
            (8, 128, 12288),
            (1, 3, 224, 224),
            (8, 64, 56, 56),
            (32, 256, 14, 14),
            (1, 8, 16, 32, 32),
        ]
        return special_shapes_1d + special_shapes_2d + model_shapes

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            # Unary pointwise: one input read and one output write.
            if not _within_memory_budget(shape, cur_dtype, tensors=2):
                continue
            inp = generate_tensor_input(shape, cur_dtype, self.device)
            if inp.numel() == 0:
                continue
            yield inp,

    def get_gbps(self, args, latency):
        inp = args[0]
        io_amount = shape_utils.size_in_bytes(inp) * 2
        return io_amount * 1e-9 / (latency * 1e-3)

    def get_tflops(self, op, *args, **kwargs):
        return args[0].numel()


class UnaryPointwiseWithArgsBenchmark(UnaryPointwiseBenchmark):
    def __init__(
        self,
        *args,
        extra_args=(),
        extra_arg_cases=None,
        input_range=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.extra_arg_cases = (
            [tuple(extra_args)] if extra_arg_cases is None else extra_arg_cases
        )
        self.input_range = input_range

    def _make_input(self, shape, cur_dtype):
        if self.input_range is None:
            return generate_tensor_input(shape, cur_dtype, self.device)
        low, high = self.input_range
        return torch.empty(shape, dtype=cur_dtype, device=self.device).uniform_(
            low, high
        )

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            # Unary pointwise: one input read and one output write.
            if not _within_memory_budget(shape, cur_dtype, tensors=2):
                continue
            inp = self._make_input(shape, cur_dtype)
            if inp.numel() == 0:
                continue
            for extra_args in self.extra_arg_cases:
                yield (inp, *extra_args)


class BinaryPointwiseBenchmark(Benchmark):
    DEFAULT_METRICS = consts.DEFAULT_METRICS[:] + ["gbps", "tflops"]

    def __init__(
        self,
        *args,
        lhs_range=None,
        rhs_range=None,
        extra_shapes=None,
        output_dtype=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.lhs_range = lhs_range
        self.rhs_range = rhs_range
        self.extra_shapes = extra_shapes or []
        self.output_dtype = output_dtype

    def set_more_shapes(self):
        same_shape_cases = [
            (1024, 1024),
            (32, 256, 1024),
            (32, 64, 112, 112),
            (8, 2048, 64, 64),
        ]
        broadcast_cases = [
            ((1024, 256), (256,)),
            ((32, 256, 1024), (256, 1)),
            ((32, 256, 56, 56), (256, 1, 1)),
            ((32, 256, 56, 56), (1, 256, 1, 1)),
            ((32, 1, 56, 56), (1, 256, 56, 56)),
            ((8, 16, 1, 128), (1, 16, 128, 1)),
        ]
        return same_shape_cases + broadcast_cases + self.extra_shapes

    def _normalize_shape_pair(self, shape_config):
        sequence_types = (list, tuple)
        if (
            isinstance(shape_config, sequence_types)
            and len(shape_config) == 2
            and isinstance(shape_config[0], sequence_types)
            and isinstance(shape_config[1], sequence_types)
        ):
            return tuple(shape_config[0]), tuple(shape_config[1])
        return shape_config, shape_config

    def _make_input(self, shape, cur_dtype, input_range):
        if input_range is None:
            return generate_tensor_input(shape, cur_dtype, self.device)
        low, high = input_range
        return torch.empty(shape, dtype=cur_dtype, device=self.device).uniform_(
            low, high
        )

    def _output_element_size(self, cur_dtype):
        dtype = self.output_dtype or cur_dtype
        return torch.empty((), dtype=dtype).element_size()

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape_config in self.shapes:
            shape_x, shape_y = self._normalize_shape_pair(shape_config)
            out_shape = torch.broadcast_shapes(shape_x, shape_y)
            # Binary pointwise: two input reads and one output write.
            total_bytes = (
                _estimated_tensor_bytes(shape_x, cur_dtype, tensors=1)
                + _estimated_tensor_bytes(shape_y, cur_dtype, tensors=1)
                + math.prod(out_shape) * self._output_element_size(cur_dtype)
            )
            if total_bytes > MAX_BENCH_BYTES:
                continue
            inp1 = self._make_input(shape_x, cur_dtype, self.lhs_range)
            inp2 = self._make_input(shape_y, cur_dtype, self.rhs_range)
            if inp1.numel() == 0 or inp2.numel() == 0:
                continue
            yield inp1, inp2

    def get_gbps(self, args, latency):
        inp1, inp2 = args[:2]
        out_shape = torch.broadcast_shapes(inp1.shape, inp2.shape)
        out_bytes = math.prod(out_shape) * self._output_element_size(inp1.dtype)
        io_amount = (
            shape_utils.size_in_bytes(inp1)
            + shape_utils.size_in_bytes(inp2)
            + out_bytes
        )
        return io_amount * 1e-9 / (latency * 1e-3)

    def get_tflops(self, op, *args, **kwargs):
        out_shape = torch.broadcast_shapes(args[0].shape, args[1].shape)
        return math.prod(out_shape)


class UnaryReductionBenchmark(Benchmark):
    DEFAULT_METRICS = consts.DEFAULT_METRICS[:] + ["gbps"]

    def __init__(self, *args, cases=None, input_range=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.cases = cases
        self.input_range = input_range

    def set_more_metrics(self):
        return ["gbps"]

    def set_more_shapes(self):
        if self.cases is not None:
            return self.cases
        return [
            ((1024 * 1024 * 16,), None, False),
            ((32, 256, 1024), None, False),
            ((1024, 1024), 1, False),
            ((32, 1024, 1024), 2, False),
            ((8, 128, 4096), 2, False),
            ((1024, 1024), 0, False),
            ((32, 1024, 1024), 0, False),
            ((32, 256, 56, 56), (2, 3), False),
            ((128, 256, 56, 56), 1, False),
            ((1, 16, 2048, 2048), (2, 3), False),
            ((64, 512, 512), 2, True),
            ((128, 256, 256), 1, True),
        ]

    def _normalize_case(self, case):
        if (
            isinstance(case, (list, tuple))
            and len(case) == 3
            and isinstance(case[0], (list, tuple))
        ):
            shape, dim, keepdim = case
            return tuple(shape), dim, keepdim
        shape = tuple(case)
        dim = 1 if len(shape) > 1 else None
        return shape, dim, False

    def _make_input(self, shape, cur_dtype):
        if self.input_range is None:
            return generate_tensor_input(shape, cur_dtype, self.device)
        low, high = self.input_range
        return torch.empty(shape, dtype=cur_dtype, device=self.device).uniform_(
            low, high
        )

    def get_input_iter(self, cur_dtype) -> Generator:
        for case in self.shapes:
            shape, dim, keepdim = self._normalize_case(case)
            if not _within_memory_budget(shape, cur_dtype, tensors=1):
                continue
            inp = self._make_input(shape, cur_dtype)
            if inp.numel() == 0:
                continue
            yield inp, dim, keepdim

    def _output_numel(self, inp, dim):
        if dim is None:
            return 1
        dims = [dim] if isinstance(dim, int) else list(dim)
        out_numel = inp.numel()
        for dim_item in {d if d >= 0 else d + inp.ndim for d in dims}:
            out_numel //= inp.shape[dim_item]
        return out_numel

    def get_gbps(self, args, latency):
        inp, dim = args[:2]
        io_amount = shape_utils.size_in_bytes(inp) + (
            self._output_numel(inp, dim) * inp.element_size()
        )
        return io_amount * 1e-9 / (latency * 1e-3)


class UnaryDimwiseBenchmark(Benchmark):
    DEFAULT_METRICS = consts.DEFAULT_METRICS[:] + ["gbps"]

    def __init__(
        self,
        *args,
        cases=None,
        input_range=None,
        output_factor=1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.cases = cases
        self.input_range = input_range
        self.output_factor = output_factor

    def set_more_metrics(self):
        return ["gbps"]

    def set_more_shapes(self):
        if self.cases is not None:
            return self.cases
        return [
            ((1024,), -1, None),
            ((65536,), -1, None),
            ((256, 1000), -1, None),
            ((1024, 100), -1, None),
            ((32, 32000), -1, None),
            ((16, 128256), -1, None),
            ((16, 12, 1024, 1024), -1, None),
            ((8, 32, 2048, 2048), -1, None),
            ((4, 32, 4096, 4096), -1, None),
            ((1, 32, 8192, 8192), -1, None),
        ]

    def _normalize_case(self, case):
        if (
            isinstance(case, (list, tuple))
            and len(case) == 3
            and isinstance(case[0], (list, tuple))
        ):
            shape, dim, out_dtype = case
            return tuple(shape), dim, out_dtype
        return tuple(case), -1, None

    def _make_input(self, shape, cur_dtype):
        if self.input_range is None:
            return generate_tensor_input(shape, cur_dtype, self.device)
        low, high = self.input_range
        return torch.empty(shape, dtype=cur_dtype, device=self.device).uniform_(
            low, high
        )

    def get_input_iter(self, cur_dtype) -> Generator:
        for case in self.shapes:
            shape, dim, out_dtype = self._normalize_case(case)
            input_bytes = _estimated_tensor_bytes(shape, cur_dtype, tensors=1)
            output_bytes = input_bytes * self.output_factor
            if input_bytes + output_bytes > MAX_BENCH_BYTES:
                continue
            inp = self._make_input(shape, cur_dtype)
            if inp.numel() == 0:
                continue
            yield inp, dim, out_dtype

    def get_gbps(self, args, latency):
        inp = args[0]
        io_amount = shape_utils.size_in_bytes(inp) * (1 + self.output_factor)
        return io_amount * 1e-9 / (latency * 1e-3)


class UnaryDimwiseWithIndicesBenchmark(UnaryDimwiseBenchmark):
    def __init__(self, *args, index_dtype=torch.int64, **kwargs):
        super().__init__(*args, **kwargs)
        self.index_dtype = index_dtype

    def _index_element_size(self):
        return torch.empty((), dtype=self.index_dtype).element_size()

    def get_gbps(self, args, latency):
        inp = args[0]
        io_amount = (
            shape_utils.size_in_bytes(inp) * 2
            + inp.numel() * self._index_element_size()
        )
        return io_amount * 1e-9 / (latency * 1e-3)


class BlasBenchmark(Benchmark):
    DEFAULT_METRICS = consts.DEFAULT_METRICS[:] + ["gbps", "tflops"]
    DEFAULT_SHAPES = []

    def __init__(
        self,
        *args,
        input_fn,
        cases=None,
        max_peak_bytes=MAX_BENCH_BYTES,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.input_fn = input_fn
        self.cases = cases
        self.max_peak_bytes = max_peak_bytes

    def set_shapes(self, shape_file_path=None):
        if self.cases is not None:
            self.shapes = self.cases
            return
        super().set_shapes(shape_file_path)

    def set_more_shapes(self):
        return self.cases or []

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            yield from self.input_fn(
                shape,
                cur_dtype,
                self.device,
                self.max_peak_bytes,
            )

    def get_gbps(self, args, latency):
        if self.op_name == "dot":
            x, y = args
            io_amount = shape_utils.size_in_bytes(x) + shape_utils.size_in_bytes(y)
        elif self.op_name == "mv":
            mat, vec = args
            io_amount = (
                shape_utils.size_in_bytes(mat)
                + shape_utils.size_in_bytes(vec)
                + mat.shape[0] * mat.element_size()
            )
        elif self.op_name == "mm":
            a, b = args
            io_amount = (
                shape_utils.size_in_bytes(a)
                + shape_utils.size_in_bytes(b)
                + a.shape[0] * b.shape[1] * a.element_size()
            )
        else:
            io_amount = sum(
                shape_utils.size_in_bytes(arg)
                for arg in args
                if torch.is_tensor(arg)
            )
        return io_amount * 1e-9 / (latency * 1e-3)

    def get_tflops(self, op, *args, **kwargs):
        if self.op_name == "dot":
            return args[0].numel() * 2
        if self.op_name == "mv":
            mat = args[0]
            return mat.shape[0] * mat.shape[1] * 2
        if self.op_name == "mm":
            a, b = args
            return a.shape[0] * a.shape[1] * b.shape[1] * 2
        return super().get_tflops(op, *args, **kwargs)


class UnaryParametricPointwiseBenchmark(Benchmark):
    DEFAULT_METRICS = consts.DEFAULT_METRICS[:] + ["gbps"]
    DEFAULT_SHAPES = []

    def __init__(
        self,
        *args,
        input_fn,
        cases=None,
        output_factor=1.0,
        max_peak_bytes=MAX_BENCH_BYTES,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.input_fn = input_fn
        self.cases = cases
        self.output_factor = output_factor
        self.max_peak_bytes = max_peak_bytes

    def set_more_metrics(self):
        return ["gbps"]

    def set_shapes(self, shape_file_path=None):
        if self.cases is not None:
            self.shapes = self.cases
            return
        super().set_shapes(shape_file_path)

    def set_more_shapes(self):
        return self.cases or []

    def get_input_iter(self, cur_dtype) -> Generator:
        for case in self.shapes:
            yield from self.input_fn(
                case,
                cur_dtype,
                self.device,
                self.max_peak_bytes,
            )

    def get_gbps(self, args, latency):
        inp = args[0]
        io_amount = shape_utils.size_in_bytes(inp) * (1 + self.output_factor)
        for arg in args[1:]:
            if torch.is_tensor(arg):
                io_amount += shape_utils.size_in_bytes(arg)
        return io_amount * 1e-9 / (latency * 1e-3)


class NormalizationBenchmark(Benchmark):
    DEFAULT_METRICS = consts.DEFAULT_METRICS[:] + ["gbps"]
    DEFAULT_SHAPES = []

    def __init__(
        self,
        *args,
        input_fn,
        cases=None,
        output_factor=1.0,
        max_peak_bytes=MAX_BENCH_BYTES,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.input_fn = input_fn
        self.cases = cases
        self.output_factor = output_factor
        self.max_peak_bytes = max_peak_bytes

    def set_more_metrics(self):
        return ["gbps"]

    def set_shapes(self, shape_file_path=None):
        if self.cases is not None:
            self.shapes = self.cases
            return
        super().set_shapes(shape_file_path)

    def set_more_shapes(self):
        return self.cases or []

    def get_input_iter(self, cur_dtype) -> Generator:
        for case in self.shapes:
            yield from self.input_fn(
                case,
                cur_dtype,
                self.device,
                self.max_peak_bytes,
            )

    def get_gbps(self, args, latency):
        inp = args[0]
        io_amount = shape_utils.size_in_bytes(inp) * (1 + self.output_factor)
        for arg in args[1:]:
            if torch.is_tensor(arg):
                io_amount += shape_utils.size_in_bytes(arg)
        return io_amount * 1e-9 / (latency * 1e-3)


class PoolingBenchmark(Benchmark):
    DEFAULT_METRICS = consts.DEFAULT_METRICS[:] + ["gbps"]
    DEFAULT_SHAPES = []

    def __init__(
        self,
        *args,
        input_fn,
        output_numel_fn,
        cases=None,
        max_peak_bytes=MAX_BENCH_BYTES,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.input_fn = input_fn
        self.output_numel_fn = output_numel_fn
        self.cases = cases
        self.max_peak_bytes = max_peak_bytes

    def set_more_metrics(self):
        return ["gbps"]

    def set_shapes(self, shape_file_path=None):
        if self.cases is not None:
            self.shapes = self.cases
            return
        super().set_shapes(shape_file_path)

    def set_more_shapes(self):
        return self.cases or []

    def get_input_iter(self, cur_dtype) -> Generator:
        for case in self.shapes:
            yield from self.input_fn(
                case,
                cur_dtype,
                self.device,
                self.max_peak_bytes,
            )

    def get_gbps(self, args, latency):
        inp = args[0]
        out_numel = self.output_numel_fn(args)
        io_amount = shape_utils.size_in_bytes(inp) + out_numel * inp.element_size()
        return io_amount * 1e-9 / (latency * 1e-3)


class ConvolutionBenchmark(Benchmark):
    DEFAULT_METRICS = consts.DEFAULT_METRICS[:] + ["gbps", "tflops"]
    DEFAULT_SHAPES = []

    def __init__(
        self,
        *args,
        input_fn,
        output_shape_fn,
        flops_fn,
        cases=None,
        fp64_cases=None,
        max_peak_bytes=MAX_BENCH_BYTES,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.input_fn = input_fn
        self.output_shape_fn = output_shape_fn
        self.flops_fn = flops_fn
        self.cases = cases
        self.fp64_cases = fp64_cases
        self.max_peak_bytes = max_peak_bytes
        if self.cases is not None:
            self.DEFAULT_SHAPES = self.cases
            self.shapes = self.cases

    def set_more_metrics(self):
        return ["gbps", "tflops"]

    def set_shapes(self, shape_file_path=None):
        if self.cases is not None:
            self.shapes = self.cases
            return
        super().set_shapes(shape_file_path)

    def set_more_shapes(self):
        return self.cases or []

    def get_input_iter(self, cur_dtype) -> Generator:
        shapes = self.shapes
        if cur_dtype == torch.float64 and self.fp64_cases is not None:
            shapes = self.fp64_cases
        for case in shapes:
            yield from self.input_fn(
                case,
                cur_dtype,
                self.device,
                self.max_peak_bytes,
            )

    def get_gbps(self, args, latency):
        output_shape = self.output_shape_fn(args)
        io_amount = 0
        for arg in args:
            if torch.is_tensor(arg):
                io_amount += shape_utils.size_in_bytes(arg)
        io_amount += math.prod(output_shape) * args[0].element_size()
        return io_amount * 1e-9 / (latency * 1e-3)

    def get_tflops(self, op, *args, **kwargs):
        return self.flops_fn(args)


class EmbeddingBenchmark(Benchmark):
    DEFAULT_METRICS = consts.DEFAULT_METRICS[:] + ["gbps"]

    def __init__(
        self,
        *args,
        input_fn,
        cases=None,
        max_peak_bytes=MAX_BENCH_BYTES,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.input_fn = input_fn
        self.cases = cases
        self.max_peak_bytes = max_peak_bytes
        if self.cases is not None:
            self.DEFAULT_SHAPES = self.cases
            self.shapes = self.cases

    def set_more_metrics(self):
        return ["gbps"]

    def set_more_shapes(self):
        return self.cases or []

    def get_input_iter(self, cur_dtype) -> Generator:
        for case in self.shapes:
            yield from self.input_fn(
                case,
                cur_dtype,
                self.device,
                self.max_peak_bytes,
            )

    def get_gbps(self, args, latency):
        indices, weight = args
        output_bytes = indices.numel() * weight.shape[1] * weight.element_size()
        io_amount = shape_utils.size_in_bytes(indices) + output_bytes * 2
        return io_amount * 1e-9 / (latency * 1e-3)


def unary_input_fn(shape, cur_dtype, device):
    yield generate_tensor_input(shape, cur_dtype, device),


def binary_input_fn(shape, cur_dtype, device):
    yield (
        generate_tensor_input(shape, cur_dtype, device),
        generate_tensor_input(shape, cur_dtype, device),
    )


def reduction_torch_op(op):
    def wrapped(x, dim, keepdim):
        if dim is None:
            return op(x)
        return op(x, dim=dim, keepdim=keepdim)

    return wrapped


def dimwise_dtype_torch_op(op):
    def wrapped(x, dim, out_dtype):
        return op(x, dim=dim, dtype=out_dtype)

    return wrapped


def dimwise_ignore_dtype_torch_op(op):
    def wrapped(x, dim, _out_dtype):
        return op(x, dim=dim)

    return wrapped
