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
RECIPROCAL_SHAPES = (
    POINTWISE_UNARY_SHAPES[0],
    POINTWISE_UNARY_SHAPES[5],
)
CEIL_SHAPES = (
    POINTWISE_UNARY_SHAPES[2],
    POINTWISE_UNARY_SHAPES[5],
)
FLOOR_SHAPES = (POINTWISE_UNARY_SHAPES[2],)
ERF_SHAPES = (POINTWISE_UNARY_SHAPES[5],)
SIN_SHAPES = (POINTWISE_UNARY_SHAPES[5],)
COS_SHAPES = (POINTWISE_UNARY_SHAPES[5],)
TAN_SHAPES = (POINTWISE_UNARY_SHAPES[5],)
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
MOD_SHAPES = (
    POINTWISE_BINARY_SHAPES[0],
    POINTWISE_BINARY_SHAPES[4],
)
POW_SHAPES = POINTWISE_BINARY_SHAPES
MAX_SHAPES = POINTWISE_BINARY_SHAPES
MIN_SHAPES = POINTWISE_BINARY_SHAPES
ADD_SQUARE_SHAPES = POINTWISE_BINARY_SHAPES
LOGICAL_SHAPES = POINTWISE_BINARY_SHAPES
CMP_SHAPES = POINTWISE_BINARY_SHAPES
BINARY_SELECT_SHAPES = (
    ((1, 1, 16), (1, 1, 16), (1, 1, 16)),
    ((1, 3, 17), (1, 3, 17), (1, 3, 17)),
    ((3, 5, 7), (3, 5, 7), (3, 5, 7)),
)
SIGMOID_BACKWARD_SHAPES = POINTWISE_BINARY_SHAPES
SCALE_SHAPES = POINTWISE_BINARY_SHAPES
CAUSAL_CONV1D_SHAPES = (
    # (batch, dim, seq_len, kernel_size, activation)
    # Small launch-overhead and non-power-of-two regimes.
    (1, 64, 128, 3, "identity"),
    (3, 192, 257, 4, "silu"),
    # Mamba/SSM-style sequence-mixing regimes.
    (8, 512, 1024, 3, "identity"),
    (4, 768, 2048, 4, "silu"),
    (2, 1024, 4096, 4, "silu"),
    (1, 2048, 8192, 4, "silu"),
    # High-batch serving and wider-channel regimes.
    (16, 1024, 512, 4, "silu"),
    (1, 4096, 2048, 5, "silu"),
)
MATMUL_SHAPES = (
    # Small (launch-overhead regime).
    ((4, 16, 32), (4, 32, 24)),
    ((8, 32, 64), (8, 64, 32)),
    # Compute-bound batched GEMM (LLM-class).
    ((32, 512, 512), (32, 512, 512)),
    ((16, 1024, 1024), (16, 1024, 1024)),
    ((8, 2048, 2048), (8, 2048, 2048)),
    ((4, 4096, 4096), (4, 4096, 4096)),
    # Non-square projection-style GEMM.
    ((16, 2048, 512), (16, 512, 2048)),
    ((32, 1024, 4096), (32, 4096, 1024)),
)

# (batch, heads_q, heads_kv, seq_q, seq_kv, head_dim, causal, generate_stats)
# Each entry is a production-representative regime:
#   1. encoder-style mid-size inference (BERT-large class, d=64)
#   2. GPT-2 class causal training with stats (d=64)
#   3. 7B-class 2k-context causal training with stats (d=128)
#   4. high-batch short-sequence inference (d=128)
#   5. MHA single-token decode against a 2k KV cache
#   6. llama3-8B class GQA (32/8) 4k-context causal training
#   7. GQA single-token decode against an 8k KV cache
#   8. large-batch short-prefill serving burst (launch-overhead regime)
SDPA_SHAPES = (
    (4, 16, 16, 512, 512, 64, False, False),
    (1, 32, 32, 1024, 1024, 64, True, True),
    (2, 16, 16, 2048, 2048, 128, True, True),
    (8, 32, 32, 256, 256, 128, False, False),
    (4, 32, 32, 1, 2048, 128, False, False),
    (1, 32, 8, 4096, 4096, 128, True, True),
    (8, 32, 8, 1, 8192, 128, False, False),
    (32, 16, 16, 128, 128, 64, False, False),
    # ===== YOLO12 attention, imgsz=640, batch=1 =====
    (4, 2, 2, 400, 400, 32, False, False),  # YOLO12n P4 area=4
    (1, 4, 4, 400, 400, 32, False, False),  # YOLO12n P5 area=1
    (4, 4, 4, 400, 400, 32, False, False),  # YOLO12s P4 area=4
    (1, 8, 8, 400, 400, 32, False, False),  # YOLO12s/m/l P5
    (4, 8, 8, 400, 400, 32, False, False),  # YOLO12m/l P4
    (4, 12, 12, 400, 400, 32, False, False),  # YOLO12x P4
    (1, 12, 12, 400, 400, 32, False, False),  # YOLO12x P5
    # ===== YOLO12 attention, imgsz=1280, batch=1 =====
    (4, 8, 8, 1600, 1600, 32, False, False),  # YOLO12m/l P4, large image
    (
        1,
        12,
        12,
        1600,
        1600,
        32,
        False,
        False,
    ),  # YOLO12x P5/P4-like large image
)


# (batch, heads_q, heads_kv, seq_q, seq_kv, head_dim, causal, generate_stats)
# fp8 SDPA forward performance regimes vs cuDNN sdpa_fp8 (all head_dim=128, the
# fp8 attention sweet spot on Hopper):
#   1. encoder-style mid-size bidirectional inference
#   2. GPT-class 1k-context causal training with stats
#   3. 7B-class 2k-context causal training with stats
#   4. high-batch short-sequence bidirectional inference
#   5. llama3-8B class GQA (32/8) 4k-context causal training with stats
#   6. large bidirectional inference (2k context)
#   7. batched 512-context causal inference
#   8. large-head GQA (64/8) 2k-context causal training with stats
SDPA_FP8_SHAPES = (
    (4, 16, 16, 512, 512, 128, False, False),
    (1, 32, 32, 1024, 1024, 128, True, True),
    (2, 16, 16, 2048, 2048, 128, True, True),
    (8, 32, 32, 256, 256, 128, False, False),
    (1, 32, 8, 4096, 4096, 128, True, True),
    (2, 32, 32, 1024, 1024, 128, False, False),
    (4, 32, 32, 512, 512, 128, True, False),
    (1, 64, 8, 2048, 2048, 128, True, True),
)


# (batch, heads_q, heads_kv, seq_q, seq_kv, head_dim, causal)
# fp8 SDPA backward uses the same eight Hopper-focused regimes as fp8 forward,
# with stats always required by the backward op.
SDPA_FP8_BACKWARD_SHAPES = (
    (4, 16, 16, 512, 512, 128, False),
    (1, 32, 32, 1024, 1024, 128, True),
    (2, 16, 16, 2048, 2048, 128, True),
    (8, 32, 32, 256, 256, 128, False),
    (1, 32, 8, 4096, 4096, 128, True),
    (2, 32, 32, 1024, 1024, 128, False),
    (4, 32, 32, 512, 512, 128, True),
    (1, 64, 8, 2048, 2048, 128, True),
)


# (batch, heads_q, heads_kv, seq_q, seq_kv, head_dim, causal)
SDPA_BACKWARD_SHAPES = (
    # ===== existing LLM / encoder shapes =====
    (4, 16, 16, 512, 512, 64, False),
    (1, 32, 32, 1024, 1024, 64, True),
    (2, 16, 16, 2048, 2048, 128, True),
    (8, 32, 32, 256, 256, 128, False),
    # decode-like backward edge case
    (4, 32, 32, 1, 2048, 128, False),
    # LLaMA3 / Mistral / Qwen3-8B class GQA backward
    (1, 32, 8, 4096, 4096, 128, True),
    # GQA decode-like backward edge case
    (8, 32, 8, 1, 8192, 128, False),
    # large-batch short-sequence backward
    (32, 16, 16, 128, 128, 64, False),
    # ===== YOLO12 attention, imgsz=640, batch=1 =====
    (4, 2, 2, 400, 400, 32, False),  # YOLO12n P4 area=4
    (1, 4, 4, 400, 400, 32, False),  # YOLO12n P5 area=1
    (4, 4, 4, 400, 400, 32, False),  # YOLO12s P4 area=4
    (1, 8, 8, 400, 400, 32, False),  # YOLO12s/m/l P5
    (4, 8, 8, 400, 400, 32, False),  # YOLO12m/l P4
    (4, 12, 12, 400, 400, 32, False),  # YOLO12x P4
    (1, 12, 12, 400, 400, 32, False),  # YOLO12x P5
    # ===== YOLO12 attention, imgsz=1280, batch=1 =====
    (4, 8, 8, 1600, 1600, 32, False),  # YOLO12m/l P4, large image
    (1, 12, 12, 1600, 1600, 32, False),  # YOLO12x P5/P4-like large image
)


def pointwise_layout(tensor):
    # cuDNN pointwise benchmarks use NHWC-compatible channels-last strides.
    # Ascend currently supports only the native contiguous conversion here.
    if tensor.dim() == 4 and tensor.device.type == "cuda":
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
    # (input_shape, weight_shape, stride, padding,
    # pre_padding, post_padding, dilation)
    ((16, 32, 256), (64, 32, 3), 1, 1, None, None, 1),
    ((8, 64, 255), (96, 64, 5), 2, None, (2,), (1,), 1),
    ((8, 32, 32, 32), (64, 32, 3, 3), 1, 1, None, None, 1),
    ((8, 64, 28, 28), (128, 64, 1, 1), 1, 0, None, None, 1),
    ((8, 64, 56, 56), (128, 64, 3, 3), 2, 1, None, None, 1),
    ((4, 64, 32, 32), (64, 64, 3, 3), 1, 2, None, None, 2),
    ((4, 32, 35, 37), (48, 32, 3, 5), (1, 2), None, (1, 0), (1, 2), 1),
    ((2, 8, 8, 16, 16), (16, 8, 3, 3, 3), 1, 1, None, None, 1),
    ((1, 8, 10, 12, 14), (12, 8, 2, 3, 3), 1, None, (1, 0, 1), (0, 1, 2), 1),
    # YOLO-n/s/m/x stem + deep stage representative
    ((1, 3, 640, 640), (16, 3, 3, 3), 2, 1, None, None, 1),  # YOLO-n stem
    ((1, 3, 640, 640), (32, 3, 3, 3), 2, 1, None, None, 1),  # YOLO-s stem
    ((1, 3, 640, 640), (64, 3, 3, 3), 2, 1, None, None, 1),  # YOLO-m/l stem
    ((1, 3, 640, 640), (96, 3, 3, 3), 2, 1, None, None, 1),  # YOLO-x stem
    ((1, 128, 40, 40), (256, 128, 3, 3), 2, 1, None, None, 1),  # YOLO-n P5
    ((1, 256, 40, 40), (512, 256, 3, 3), 2, 1, None, None, 1),  # YOLO-s P5
    ((1, 512, 40, 40), (512, 512, 3, 3), 2, 1, None, None, 1),  # YOLO-m/l P5
    ((1, 768, 40, 40), (768, 768, 3, 3), 2, 1, None, None, 1),  # YOLO-x P5
)

CONV_DGRAD_SHAPES = tuple(
    shape
    for index, shape in enumerate(CONV_FPROP_SHAPES)
    if index not in (5, 6)
)

CONV_WGRAD_SHAPES = CONV_DGRAD_SHAPES


def selected_shapes(shapes, env_name, legacy_env_names=()):
    only = os.getenv(env_name)
    if not only:
        for legacy_env_name in legacy_env_names:
            only = os.getenv(legacy_env_name)
            if only:
                break
    if not only:
        return shapes
    selected = {int(item) for item in only.split(",") if item.strip()}
    return [shape for index, shape in enumerate(shapes) if index in selected]


def bench_warmup():
    return int(
        os.getenv(
            "FLAGDNN_PERF_WARMUP",
            os.getenv("FLAGDNN_CUDNN_PERF_WARMUP", "25"),
        )
    )


def bench_repeat():
    return int(
        os.getenv(
            "FLAGDNN_PERF_REPEAT",
            os.getenv("FLAGDNN_CUDNN_PERF_REPEAT", "100"),
        )
    )


def min_speedup():
    # Default to report-only mode. Set this env var, e.g. to 0.9,
    # when graph performance should be used as a hard gate.
    return float(
        os.getenv(
            "FLAGDNN_PERF_MIN_SPEEDUP",
            os.getenv("FLAGDNN_CUDNN_PERF_MIN_SPEEDUP", "0"),
        )
    )


def compile_options():
    return {"cache": None, "validate_inputs": False}


BATCHNORM_INFERENCE_SHAPES = (
    (8, 32, 32, 32),  # launch-sensitive small/medium activation
    (16, 64, 16, 16),  # same element count with larger C
    (4, 128, 16, 16),  # higher C, smaller batch
    (8, 64, 56, 56),  # early CNN/ResNet stage, large spatial
    (16, 128, 28, 28),  # mid-stage activation
    (16, 256, 14, 14),  # late-stage activation
    (8, 512, 7, 7),  # high-channel small spatial
    (32, 1024, 1, 1),  # per-channel 1x1 normalization
)

REDUCTION_SHAPES = (
    ((8, 8, 32, 32), 1, "ADD"),
    ((8, 8, 32, 32), 1, "AVG"),
    ((8, 4, 16, 16), 1, "MUL"),
)


RMSNORM_RHT_AMAX_SHAPES = (
    (64, 2048, 2),
    (128, 4096, 4),
)
