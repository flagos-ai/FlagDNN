"""
DNN operations
"""

from flag_dnn.ops.relu import relu
from flag_dnn.ops.gelu import gelu
from flag_dnn.ops.silu import silu
from flag_dnn.ops.leaky_relu import leaky_relu
from flag_dnn.ops.prelu import prelu
from flag_dnn.ops.softmax import softmax
from flag_dnn.ops.batch_norm import batch_norm
from flag_dnn.ops.batch_norm import batch_norm_aten
from flag_dnn.ops.layer_norm import layer_norm
from flag_dnn.ops.rms_norm import rms_norm
from flag_dnn.ops.group_norm import group_norm
from flag_dnn.ops.max_pool2d import max_pool2d
from flag_dnn.ops.avg_pool2d import avg_pool2d
from flag_dnn.ops.adaptive_avg_pool2d import adaptive_avg_pool2d
from flag_dnn.ops.adaptive_max_pool2d import adaptive_max_pool2d
from flag_dnn.ops.add import add
from flag_dnn.ops.sub import sub
from flag_dnn.ops.mul import mul
from flag_dnn.ops.div import div
from flag_dnn.ops.pow import pow
from flag_dnn.ops.sqrt import sqrt
from flag_dnn.ops.abs import abs
from flag_dnn.ops.neg import neg
from flag_dnn.ops.clamp import clamp
from flag_dnn.ops.sum import sum
from flag_dnn.ops.mean import mean
from flag_dnn.ops.prod import prod
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
from flag_dnn.ops.dot import dot


__all__ = [
    "relu",
    "gelu",
    "silu",
    "leaky_relu",
    "prelu",
    "softmax",
    "batch_norm",
    "batch_norm_aten",
    "layer_norm",
    "rms_norm",
    "group_norm",
    "max_pool2d",
    "avg_pool2d",
    "adaptive_avg_pool2d",
    "adaptive_max_pool2d",
    "add",
    "sub",
    "mul",
    "div",
    "pow",
    "sqrt",
    "abs",
    "neg",
    "clamp",
    "sum",
    "mean",
    "prod",
    "cumsum",
    "cumprod",
    "eq",
    "ne",
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
    "dot",
]
