import os

import torch

COMPARE_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

IDENTITY_SHAPES = (
    (1, 1, 1),
    (2, 3, 4),
    (8, 16, 32),
    (64, 64, 64),
    (16, 64, 128),
    (16, 256, 256),
    (32, 128, 256),
    (4, 1024, 1024),
    (16, 512, 1024),
    (2, 2048, 2048),
    (8, 1024, 2048),
)

# cuDNN frontend rejects raw 1D pointwise descriptors and returns
# incorrect values for raw 2D descriptors on this backend. Keep
# runnable 3D wrappers plus 4D channels-last/NHWC cases.
POINTWISE_UNARY_SHAPES = (
    # Power-of-two aligned shapes.
    (1, 1, 1024),
    (8, 16, 32),
    (4, 16, 64, 128),
    (8, 16, 64, 128),
    # Non-power-of-two shapes.
    (1, 1, 1000),
    (3, 257, 513),
    (3, 7, 65, 129),
    (5, 7, 65, 129),
)

ABS_SHAPES = POINTWISE_UNARY_SHAPES
SIGMOID_SHAPES = POINTWISE_UNARY_SHAPES
POINTWISE_BINARY_SHAPES = (
    # Power-of-two aligned shapes.
    ((1, 1, 1024), (1, 1, 1024)),
    ((8, 16, 32), (8, 16, 32)),
    ((4, 16, 64, 128), (4, 16, 64, 128)),
    ((8, 16, 64, 128), (8, 16, 64, 128)),
    # Non-power-of-two shapes.
    ((1, 1, 1000), (1, 1, 1000)),
    ((3, 257, 513), (3, 257, 513)),
    ((3, 7, 65, 129), (3, 7, 65, 129)),
    ((5, 7, 65, 129), (5, 7, 65, 129)),
)

ADD_SHAPES = POINTWISE_BINARY_SHAPES
SUB_SHAPES = POINTWISE_BINARY_SHAPES
MUL_SHAPES = POINTWISE_BINARY_SHAPES
DIV_SHAPES = POINTWISE_BINARY_SHAPES
POW_SHAPES = POINTWISE_BINARY_SHAPES
MAX_SHAPES = POINTWISE_BINARY_SHAPES
CMP_SHAPES = POINTWISE_BINARY_SHAPES
SIGMOID_BACKWARD_SHAPES = POINTWISE_BINARY_SHAPES


def pointwise_layout(tensor):
    # 4D logical NCHW shapes use channels-last strides to match cuDNN NHWC.
    if tensor.dim() == 4:
        return tensor.contiguous(memory_format=torch.channels_last)
    return tensor


def pointwise_randn(shape, dtype, device):
    return pointwise_layout(torch.randn(shape, device=device, dtype=dtype))


def pointwise_rand(shape, dtype, device):
    return pointwise_layout(torch.rand(shape, device=device, dtype=dtype))

RESHAPE_SHAPES = (
    ((8, 16, 32), (128, 32)),
    ((16, 64, 128), (1024, 128)),
    ((16, 256, 256), (4096, 256)),
    ((32, 128, 256), (4096, 256)),
    ((4, 1024, 1024), (4096, 1024)),
)

TRANSPOSE_SHAPES = (
    ((8, 16, 32), (2, 0, 1)),
    ((16, 64, 128), (0, 2, 1)),
    ((32, 128, 256), (1, 0, 2)),
    ((4, 64, 128, 32), (0, 2, 3, 1)),
    ((2, 128, 128, 64), (0, 3, 1, 2)),
)

SLICE_SHAPES = (
    ((8, 16, 32), (slice(None), slice(2, 14, 2), slice(None))),
    ((16, 64, 128), (slice(1, 15), slice(None, None, 2), slice(8, 120, 4))),
    ((32, 128, 256), (slice(None), slice(4, 124, 3), slice(16, 240, 2))),
    (
        (4, 64, 128, 32),
        (slice(None), slice(8, 56, 2), slice(None), slice(4, 28, 2)),
    ),
)

CONCATENATE_SHAPES = (
    (((8, 16, 32), (8, 32, 32)), 1),
    (((16, 64, 128), (16, 64, 64), (16, 64, 32)), 2),
    (((8, 128, 256), (24, 128, 256)), 0),
    (((4, 64, 128, 16), (4, 64, 128, 32)), 3),
)

GEN_INDEX_SHAPES = (
    ((16, 64, 128), 1),
    ((32, 128, 256), 2),
    ((4, 64, 128, 32), 3),
)

CONV_FPROP_SHAPES = (
    # (input_shape, weight_shape, stride, padding, pre_padding, post_padding, dilation)
    ((16, 32, 256), (64, 32, 3), 1, 1, None, None, 1),
    ((8, 64, 255), (96, 64, 5), 2, None, (2,), (1,), 1),
    ((8, 32, 32, 32), (64, 32, 3, 3), 1, 1, None, None, 1),
    ((8, 64, 28, 28), (128, 64, 1, 1), 1, 0, None, None, 1),
    ((8, 64, 56, 56), (128, 64, 3, 3), 2, 1, None, None, 1),
    ((4, 64, 32, 32), (64, 64, 3, 3), 1, 2, None, None, 2),
    ((4, 32, 35, 37), (48, 32, 3, 5), (1, 2), None, (1, 0), (1, 2), 1),
    ((2, 8, 8, 16, 16), (16, 8, 3, 3, 3), 1, 1, None, None, 1),
    ((1, 8, 10, 12, 14), (12, 8, 2, 3, 3), 1, None, (1, 0, 1), (0, 1, 2), 1),
)


def selected_shapes(shapes, env_name):
    only = os.getenv(env_name)
    if not only:
        return shapes
    selected = {int(item) for item in only.split(",") if item.strip()}
    return [shape for index, shape in enumerate(shapes) if index in selected]


def bench_warmup():
    return int(os.getenv("FLAGDNN_CUDNN_PERF_WARMUP", "10"))


def bench_repeat():
    return int(os.getenv("FLAGDNN_CUDNN_PERF_REPEAT", "30"))


def compile_options():
    return {"cache": None, "validate_inputs": False}
