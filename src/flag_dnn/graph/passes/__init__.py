from flag_dnn.graph.passes.dead_code import eliminate_dead_nodes
from flag_dnn.graph.passes.fusion import apply_fusion_pass

__all__ = ["apply_fusion_pass", "eliminate_dead_nodes"]
