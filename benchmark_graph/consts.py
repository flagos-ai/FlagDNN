import os

import torch

CUDNN_COMPARE_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

CUDNN_IDENTITY_SHAPES = (
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

CUDNN_ABS_SHAPES = CUDNN_IDENTITY_SHAPES

CUDNN_POINTWISE_BINARY_SHAPES = (
    # Power-of-two aligned shapes.
    ((32, 256, 256), (32, 256, 256)),
    ((16, 512, 512), (16, 512, 512)),
    ((32, 512, 512), (32, 512, 512)),
    # Non-power-of-two shapes.
    ((35, 257, 257), (35, 257, 257)),
    ((19, 257, 513), (19, 257, 513)),
    ((29, 257, 513), (29, 257, 513)),
)

CUDNN_ADD_SHAPES = CUDNN_POINTWISE_BINARY_SHAPES
CUDNN_SUB_SHAPES = CUDNN_POINTWISE_BINARY_SHAPES
CUDNN_MUL_SHAPES = CUDNN_POINTWISE_BINARY_SHAPES
CUDNN_DIV_SHAPES = CUDNN_POINTWISE_BINARY_SHAPES
CUDNN_POW_SHAPES = CUDNN_POINTWISE_BINARY_SHAPES
CUDNN_MAX_SHAPES = CUDNN_POINTWISE_BINARY_SHAPES
CUDNN_CMP_SHAPES = CUDNN_POINTWISE_BINARY_SHAPES

CUDNN_RESHAPE_SHAPES = (
    ((8, 16, 32), (128, 32)),
    ((16, 64, 128), (1024, 128)),
    ((16, 256, 256), (4096, 256)),
    ((32, 128, 256), (4096, 256)),
    ((4, 1024, 1024), (4096, 1024)),
)

CUDNN_TRANSPOSE_SHAPES = (
    ((8, 16, 32), (2, 0, 1)),
    ((16, 64, 128), (0, 2, 1)),
    ((32, 128, 256), (1, 0, 2)),
    ((4, 64, 128, 32), (0, 2, 3, 1)),
    ((2, 128, 128, 64), (0, 3, 1, 2)),
)

CUDNN_SLICE_SHAPES = (
    ((8, 16, 32), (slice(None), slice(2, 14, 2), slice(None))),
    ((16, 64, 128), (slice(1, 15), slice(None, None, 2), slice(8, 120, 4))),
    ((32, 128, 256), (slice(None), slice(4, 124, 3), slice(16, 240, 2))),
    (
        (4, 64, 128, 32),
        (slice(None), slice(8, 56, 2), slice(None), slice(4, 28, 2)),
    ),
)

CUDNN_CONCATENATE_SHAPES = (
    (((8, 16, 32), (8, 32, 32)), 1),
    (((16, 64, 128), (16, 64, 64), (16, 64, 32)), 2),
    (((8, 128, 256), (24, 128, 256)), 0),
    (((4, 64, 128, 16), (4, 64, 128, 32)), 3),
)

CUDNN_GEN_INDEX_SHAPES = (
    ((16, 64, 128), 1),
    ((32, 128, 256), 2),
    ((4, 64, 128, 32), 3),
)

CUDNN_CONV_FPROP_SHAPES = (
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
