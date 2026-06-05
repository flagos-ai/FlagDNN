"""
DNN operations
"""

from flag_dnn import runtime
from flag_dnn.ops.identity import identity
from flag_dnn.ops.reshape import reshape
from flag_dnn.ops.transpose import transpose
from flag_dnn.ops.slice import slice
from flag_dnn.ops.concatenate import concatenate
from flag_dnn.ops.gen_index import gen_index
from flag_dnn.ops.relu import relu
from flag_dnn.ops.gelu import gelu
from flag_dnn.ops.silu import silu
from flag_dnn.ops.swish import swish
from flag_dnn.ops.leaky_relu import leaky_relu
from flag_dnn.ops.prelu import prelu
from flag_dnn.ops.softmax import softmax
from flag_dnn.ops.batch_norm import batch_norm
from flag_dnn.ops.batch_norm import batch_norm_aten
from flag_dnn.ops.batchnorm import batchnorm
from flag_dnn.ops.batchnorm_inference import batchnorm_inference
from flag_dnn.ops.layer_norm import layer_norm
from flag_dnn.ops.layernorm import layernorm
from flag_dnn.ops.rms_norm import rms_norm
from flag_dnn.ops.rmsnorm import rmsnorm
from flag_dnn.ops.rmsnorm_rht_amax import rmsnorm_rht_amax_wrapper_sm100
from flag_dnn.ops.group_norm import group_norm
from flag_dnn.ops.max_pool2d import max_pool2d
from flag_dnn.ops.avg_pool2d import avg_pool2d
from flag_dnn.ops.adaptive_avg_pool2d import adaptive_avg_pool2d
from flag_dnn.ops.adaptive_max_pool2d import adaptive_max_pool2d
from flag_dnn.ops.add import add
from flag_dnn.ops.sub import sub
from flag_dnn.ops.mul import mul
from flag_dnn.ops.scale import scale
from flag_dnn.ops.div import div
from flag_dnn.ops.mod import mod
from flag_dnn.ops.pow import pow
from flag_dnn.ops.max import max
from flag_dnn.ops.min import min
from flag_dnn.ops.sqrt import sqrt
from flag_dnn.ops.abs import abs
from flag_dnn.ops.neg import neg
from flag_dnn.ops.clamp import clamp
from flag_dnn.ops.sum import sum
from flag_dnn.ops.mean import mean
from flag_dnn.ops.prod import prod
from flag_dnn.ops.reduction import reduction
from flag_dnn.ops.cumsum import cumsum
from flag_dnn.ops.cumprod import cumprod
from flag_dnn.ops.eq import eq
from flag_dnn.ops.ne import ne
from flag_dnn.ops.max_pool1d import max_pool1d
from flag_dnn.ops.max_pool3d import max_pool3d
from flag_dnn.ops.avg_pool1d import avg_pool1d
from flag_dnn.ops.avg_pool3d import avg_pool3d
from flag_dnn.ops.adaptive_avg_pool1d import adaptive_avg_pool1d
from flag_dnn.ops.adaptive_avg_pool3d import adaptive_avg_pool3d
from flag_dnn.ops.adaptive_max_pool1d import adaptive_max_pool1d
from flag_dnn.ops.adaptive_max_pool3d import adaptive_max_pool3d
from flag_dnn.ops.threshold import threshold
from flag_dnn.ops.threshold_ import threshold_
from flag_dnn.ops.leaky_relu_ import leaky_relu_
from flag_dnn.ops.hardtanh import hardtanh
from flag_dnn.ops.hardtanh_ import hardtanh_
from flag_dnn.ops.elu import elu
from flag_dnn.ops.elu_ import elu_
from flag_dnn.ops.rrelu import rrelu
from flag_dnn.ops.rrelu_ import rrelu_
from flag_dnn.ops.mish import mish
from flag_dnn.ops.softplus import softplus
from flag_dnn.ops.softsign import softsign
from flag_dnn.ops.softshrink import softshrink
from flag_dnn.ops.softmin import softmin
from flag_dnn.ops.mv import mv
from flag_dnn.ops.mm import mm
from flag_dnn.ops.matmul import matmul
from flag_dnn.ops.dot import dot
from flag_dnn.ops.conv1d import conv1d
from flag_dnn.ops.conv2d import conv2d
from flag_dnn.ops.conv3d import conv3d
from flag_dnn.ops.conv_fprop import conv_fprop
from flag_dnn.ops.conv_dgrad import conv_dgrad
from flag_dnn.ops.conv_wgrad import conv_wgrad
from flag_dnn.ops.causal_conv1d import causal_conv1d
from flag_dnn.ops.hardswish import hardswish
from flag_dnn.ops.relu6 import relu6
from flag_dnn.ops.selu import selu
from flag_dnn.ops.selu_ import selu_
from flag_dnn.ops.glu import glu
from flag_dnn.ops.celu import celu
from flag_dnn.ops.celu_ import celu_
from flag_dnn.ops.tanh import tanh
from flag_dnn.ops.sigmoid import sigmoid
from flag_dnn.ops.sigmoid_backward import sigmoid_backward
from flag_dnn.ops.logsigmoid import logsigmoid
from flag_dnn.ops.isinf import isinf
from flag_dnn.ops.isnan import isnan
from flag_dnn.ops.square import square
from flag_dnn.ops.add_square import add_square
from flag_dnn.ops.rsqrt import rsqrt
from flag_dnn.ops.positive import positive
from flag_dnn.ops.log import log
from flag_dnn.ops.exp import exp
from flag_dnn.ops.reciprocal import reciprocal
from flag_dnn.ops.ceil import ceil
from flag_dnn.ops.floor import floor
from flag_dnn.ops.erf import erf
from flag_dnn.ops.sin import sin
from flag_dnn.ops.cos import cos
from flag_dnn.ops.tan import tan
from flag_dnn.ops.bitwise_not import bitwise_not
from flag_dnn.ops.minimum import minimum
from flag_dnn.ops.maximum import maximum
from flag_dnn.ops.binary_select import binary_select, where
from flag_dnn.ops.bitwise_and import bitwise_and
from flag_dnn.ops.bitwise_or import bitwise_or
from flag_dnn.ops.bitwise_xor import bitwise_xor
from flag_dnn.ops.logical_and import logical_and
from flag_dnn.ops.logical_or import logical_or
from flag_dnn.ops.logical_not import logical_not
from flag_dnn.ops.one_hot import one_hot
from flag_dnn.ops.kl_div import kl_div
from flag_dnn.ops.mse_loss import mse_loss
from flag_dnn.ops.l1_loss import l1_loss
from flag_dnn.ops.log_softmax import log_softmax
from flag_dnn.ops.interpolate import interpolate
from flag_dnn.ops.any import any
from flag_dnn.ops.all import all
from flag_dnn.ops.fmin import fmin
from flag_dnn.ops.fmax import fmax
from flag_dnn.ops.embedding import embedding
from flag_dnn.ops.embedding import embedding_renorm_
from flag_dnn.ops.cummin import cummin
from flag_dnn.ops.cummax import cummax
from flag_dnn.ops.lt import lt
from flag_dnn.ops.le import le
from flag_dnn.ops.gt import gt
from flag_dnn.ops.ge import ge


def cmp_eq(input, comparison, *, out=None, compute_data_type=None, name=""):
    return eq(
        input,
        comparison,
        out=out,
        compute_data_type=compute_data_type,
        name=name,
    )


def cmp_neq(input, comparison, *, out=None, compute_data_type=None, name=""):
    return ne(
        input,
        comparison,
        out=out,
        compute_data_type=compute_data_type,
        name=name,
    )


def cmp_lt(input, comparison, *, out=None, compute_data_type=None, name=""):
    return lt(
        input,
        comparison,
        out=out,
        compute_data_type=compute_data_type,
        name=name,
    )


def cmp_le(input, comparison, *, out=None, compute_data_type=None, name=""):
    return le(
        input,
        comparison,
        out=out,
        compute_data_type=compute_data_type,
        name=name,
    )


def cmp_gt(input, comparison, *, out=None, compute_data_type=None, name=""):
    return gt(
        input,
        comparison,
        out=out,
        compute_data_type=compute_data_type,
        name=name,
    )


def cmp_ge(input, comparison, *, out=None, compute_data_type=None, name=""):
    return ge(
        input,
        comparison,
        out=out,
        compute_data_type=compute_data_type,
        name=name,
    )


def gelu_approx_tanh(input, *, compute_data_type=None, name=""):
    del compute_data_type, name
    return gelu(input, approximate="tanh")


__all__ = [
    "identity",
    "reshape",
    "transpose",
    "slice",
    "concatenate",
    "gen_index",
    "relu",
    "gelu",
    "gelu_approx_tanh",
    "silu",
    "swish",
    "leaky_relu",
    "prelu",
    "softmax",
    "batch_norm",
    "batch_norm_aten",
    "batchnorm",
    "batchnorm_inference",
    "layernorm",
    "layer_norm",
    "rms_norm",
    "rmsnorm",
    "rmsnorm_rht_amax_wrapper_sm100",
    "group_norm",
    "max_pool2d",
    "avg_pool2d",
    "adaptive_avg_pool2d",
    "adaptive_max_pool2d",
    "add",
    "sub",
    "mul",
    "scale",
    "div",
    "mod",
    "pow",
    "max",
    "min",
    "sqrt",
    "abs",
    "neg",
    "clamp",
    "sum",
    "mean",
    "prod",
    "reduction",
    "cumsum",
    "cumprod",
    "eq",
    "ne",
    "cmp_eq",
    "cmp_neq",
    "cmp_lt",
    "cmp_le",
    "cmp_gt",
    "cmp_ge",
    "max_pool1d",
    "max_pool3d",
    "avg_pool1d",
    "avg_pool3d",
    "adaptive_avg_pool1d",
    "adaptive_avg_pool3d",
    "adaptive_max_pool1d",
    "adaptive_max_pool3d",
    "threshold",
    "threshold_",
    "leaky_relu_",
    "hardtanh",
    "hardtanh_",
    "elu",
    "elu_",
    "rrelu",
    "rrelu_",
    "mish",
    "softplus",
    "softsign",
    "softshrink",
    "softmin",
    "mv",
    "mm",
    "matmul",
    "dot",
    "conv1d",
    "conv2d",
    "conv3d",
    "conv_fprop",
    "conv_dgrad",
    "conv_wgrad",
    "causal_conv1d",
    "hardswish",
    "relu6",
    "selu",
    "selu_",
    "glu",
    "celu",
    "celu_",
    "tanh",
    "sigmoid",
    "sigmoid_backward",
    "logsigmoid",
    "isinf",
    "isnan",
    "square",
    "add_square",
    "rsqrt",
    "positive",
    "log",
    "exp",
    "reciprocal",
    "ceil",
    "floor",
    "erf",
    "sin",
    "cos",
    "tan",
    "bitwise_not",
    "minimum",
    "maximum",
    "binary_select",
    "where",
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "logical_and",
    "logical_or",
    "logical_not",
    "one_hot",
    "kl_div",
    "mse_loss",
    "l1_loss",
    "log_softmax",
    "interpolate",
    "any",
    "all",
    "fmin",
    "fmax",
    "embedding",
    "embedding_renorm_",
    "cummin",
    "cummax",
    "lt",
    "le",
    "gt",
    "ge",
]


runtime.replace_customized_ops(globals())

from flag_dnn.graph.wrappers import install_graph_wrappers  # noqa: E402

install_graph_wrappers(globals())
