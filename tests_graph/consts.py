import torch

IDENTITY_SHAPES = (
    (1, 1, 1),
    (2, 3, 4),
    (4, 5, 6),
    (1, 8, 16),
    (3, 1, 17),
    (2, 4, 8),
    (5, 7, 11),
    (1, 33, 65),
    (2, 16, 257),
    (4, 32, 128),
)

# cuDNN frontend rejects raw 1D pointwise descriptors and returns
# incorrect values for raw 2D descriptors on this backend. Keep
# runnable 3D wrappers plus 4D channels-last/NHWC cases.
POINTWISE_UNARY_SHAPES = (
    # Power-of-two aligned shapes.
    (1, 1, 16),
    (2, 4, 8),
    (1, 4, 8, 16),
    (2, 4, 8, 16),
    # Non-power-of-two shapes.
    (1, 3, 17),
    (3, 5, 7),
    (1, 3, 5, 7),
    (2, 3, 5, 7),
)

ABS_SHAPES = POINTWISE_UNARY_SHAPES
EXP_SHAPES = POINTWISE_UNARY_SHAPES
LOG_SHAPES = POINTWISE_UNARY_SHAPES
RSQRT_SHAPES = POINTWISE_UNARY_SHAPES
SIGMOID_SHAPES = POINTWISE_UNARY_SHAPES
RELU_SHAPES = POINTWISE_UNARY_SHAPES
SWISH_SHAPES = POINTWISE_UNARY_SHAPES
LEAKY_RELU_SHAPES = POINTWISE_UNARY_SHAPES
ELU_SHAPES = POINTWISE_UNARY_SHAPES
SOFTPLUS_SHAPES = POINTWISE_UNARY_SHAPES
GELU_APPROX_TANH_SHAPES = POINTWISE_UNARY_SHAPES
RECIPROCAL_SHAPES = POINTWISE_UNARY_SHAPES
CEIL_SHAPES = POINTWISE_UNARY_SHAPES
FLOOR_SHAPES = POINTWISE_UNARY_SHAPES
ERF_SHAPES = POINTWISE_UNARY_SHAPES
SIN_SHAPES = POINTWISE_UNARY_SHAPES
COS_SHAPES = POINTWISE_UNARY_SHAPES
TAN_SHAPES = POINTWISE_UNARY_SHAPES
POINTWISE_BINARY_CASES = (
    # Power-of-two aligned shapes.
    ((1, 1, 16), (1, 1, 16)),
    ((2, 4, 8), (2, 4, 8)),
    ((1, 4, 8, 16), (1, 4, 8, 16)),
    ((2, 4, 8, 16), (2, 4, 8, 16)),
    # Non-power-of-two shapes.
    ((1, 3, 17), (1, 3, 17)),
    ((3, 5, 7), (3, 5, 7)),
    ((1, 3, 5, 7), (1, 3, 5, 7)),
    ((2, 3, 5, 7), (2, 3, 5, 7)),
)

ADD_CASES = POINTWISE_BINARY_CASES
SUB_CASES = POINTWISE_BINARY_CASES
MUL_CASES = POINTWISE_BINARY_CASES
DIV_CASES = POINTWISE_BINARY_CASES
MOD_CASES = POINTWISE_BINARY_CASES
POW_CASES = POINTWISE_BINARY_CASES
MAX_CASES = POINTWISE_BINARY_CASES
MIN_CASES = POINTWISE_BINARY_CASES
ADD_SQUARE_CASES = POINTWISE_BINARY_CASES
LOGICAL_CASES = POINTWISE_BINARY_CASES
CMP_CASES = POINTWISE_BINARY_CASES
BINARY_SELECT_CASES = tuple(
    (x_shape, y_shape, tuple(torch.broadcast_shapes(x_shape, y_shape)))
    for x_shape, y_shape in POINTWISE_BINARY_CASES
)
SIGMOID_BACKWARD_CASES = POINTWISE_BINARY_CASES
SCALE_CASES = POINTWISE_BINARY_CASES
MATMUL_CASES = (
    ((4, 16, 32), (4, 32, 24)),
    ((8, 32, 64), (8, 64, 32)),
    ((16, 32, 128), (16, 128, 64)),
)

# (batch, heads_q, heads_kv, seq_q, seq_kv, head_dim)
SDPA_CASES = (
    (2, 8, 8, 128, 128, 64),
    (1, 8, 8, 512, 512, 128),
    (2, 8, 2, 128, 128, 64),
    (2, 4, 4, 100, 80, 64),
    (2, 4, 4, 64, 64, 72),
    (2, 8, 8, 1, 256, 64),
)

# Masked variants need seq_q > 1 to exercise the diagonal logic.
SDPA_MASKED_CASES = (
    (2, 8, 8, 128, 128, 64),
    (2, 4, 4, 100, 80, 64),
    (2, 4, 4, 100, 256, 64),
    (1, 8, 8, 512, 512, 128),
)


def pointwise_layout(tensor):
    # 4D logical NCHW shapes use channels-last strides to match cuDNN NHWC.
    if tensor.dim() == 4:
        return tensor.contiguous(memory_format=torch.channels_last)
    return tensor


def pointwise_randn(shape, dtype, device):
    return pointwise_layout(torch.randn(shape, device=device, dtype=dtype))


def pointwise_rand(shape, dtype, device):
    return pointwise_layout(torch.rand(shape, device=device, dtype=dtype))


def pointwise_positive(shape, dtype, device, offset=0.1):
    return pointwise_layout(
        torch.rand(shape, device=device, dtype=dtype) + offset
    )


def pointwise_bool(shape, device):
    return pointwise_layout(torch.rand(shape, device=device) > 0.5)


RESHAPE_CASES = (
    ((2, 3, 4), (6, 4)),
    ((1, 8, 16), (4, 32)),
    ((4, 5, 6), (2, 3, 20)),
)

TRANSPOSE_CASES = (
    ((2, 3, 4), (2, 0, 1)),
    ((1, 8, 16), (0, 2, 1)),
    ((2, 3, 4, 5), (0, 2, 3, 1)),
)

SLICE_CASES = (
    ((2, 4, 5), (slice(None), slice(1, 4, 2), slice(None))),
    ((4, 6, 8), (slice(1, 4), slice(None, None, 2), slice(2, 8, 3))),
    ((3, 5, 7, 2), (slice(None), slice(1, 5, 2), slice(None), slice(None))),
)

CONCATENATE_CASES = (
    (((2, 3, 4), (2, 5, 4), (2, 1, 4)), 1),
    (((1, 2), (3, 2), (4, 2)), 0),
    (((2, 3, 1), (2, 3, 5)), 2),
)

GEN_INDEX_CASES = (
    ((2, 3, 4), 1),
    ((2, 3, 4), 2),
)

CONV2D_CASES = (
    ((2, 8, 16, 16), (16, 8, 3, 3), True, 1, 1, 1),
    ((1, 4, 15, 17), (6, 4, 3, 5), False, (2, 1), (1, 2), 1),
    ((2, 3, 8, 8), (5, 3, 1, 1), True, 1, 0, 1),
    ((1, 6, 20, 18), (8, 6, 5, 3), False, (1, 2), (2, 1), 1),
    ((2, 4, 12, 10), (7, 4, 3, 3), True, 2, 1, 1),
    ((1, 5, 19, 21), (9, 5, 3, 3), False, 1, 2, 2),
    ((1, 3, 32, 32), (8, 3, 3, 3), True, 1, 1, 1),
    ((2, 8, 9, 11), (4, 8, 1, 1), False, 1, 0, 1),
    ((1, 4, 18, 18), (4, 4, 3, 3), True, 1, 0, 1),
    ((2, 12, 13, 15), (10, 12, 3, 3), False, 1, 1, 1),
)

CONV_BIAS_RELU_CASES = (
    ((2, 8, 16, 16), (16, 8, 3, 3), 1, 1, 1),
    ((1, 4, 15, 17), (6, 4, 3, 5), (2, 1), (1, 2), 1),
    ((2, 3, 8, 8), (5, 3, 1, 1), 1, 0, 1),
    ((1, 6, 20, 18), (8, 6, 5, 3), (1, 2), (2, 1), 1),
    ((2, 4, 12, 10), (7, 4, 3, 3), 2, 1, 1),
    ((1, 5, 19, 21), (9, 5, 3, 3), 1, 2, 2),
    ((1, 3, 32, 32), (8, 3, 3, 3), 1, 1, 1),
    ((2, 8, 9, 11), (4, 8, 1, 1), 1, 0, 1),
    ((1, 4, 18, 18), (4, 4, 3, 3), 1, 0, 1),
    ((2, 12, 13, 15), (10, 12, 3, 3), 1, 1, 1),
)

BATCHNORM_INFERENCE_CASES = (
    (2, 8, 16, 16),
    (4, 16, 8, 8),
    (2, 32, 7, 9),
)

REDUCTION_CASES = (
    ((2, 4, 8, 8), 1, "ADD"),
    ((2, 4, 8, 8), 1, "AVG"),
    ((2, 4, 8, 8), 1, "MUL"),
)
