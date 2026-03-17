"""
DNN operations
"""

from flag_dnn.ops.relu import relu
from flag_dnn.ops.gelu import gelu
from flag_dnn.ops.silu import silu
from flag_dnn.ops.leaky_relu import leaky_relu
from flag_dnn.ops.prelu import prelu
from flag_dnn.ops.softmax import softmax

__all__ = [
    "relu",
    "gelu",
    "silu",
    "leaky_relu",
    "prelu",
    "softmax",

]
