"""Emit pointer launches for Ascend's extended kernel plans."""

from pathlib import Path
import ast
import hashlib
import math
from .io import _materialize_source
from ..dispatch.extended import plan_graph
from ..dispatch.common import _tensor_storage_size


def emit(graph, output_directory, identity):
    plans, workspace, workspace_size = plan_graph(graph)
    stages = []
    for plan in plans:
        entry, signature, constants, grid, layout = plan["configuration"]
        source = (
            Path(__file__).resolve().parents[1] / "kernels" / (plan["source"] + ".py")
        )
        data = source.read_bytes()
        tree = ast.parse(data)
        function = next(
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == entry
        )
        names = [a.arg for a in function.args.args]
        defaults = (
            dict(zip(names[-len(function.args.defaults) :], function.args.defaults))
            if function.args.defaults
            else {}
        )
        values = dict(constants)
        runtime_names = [name for name in names if name in signature]
        if len(runtime_names) != len(layout):
            raise ValueError("Ascend extended signature and argument layout differ")
        arguments, tokens, tensor_index = [], {}, 0
        annotations = {
            a.arg: ast.unparse(a.annotation) if a.annotation else ""
            for a in function.args.args
        }
        for name, (kind, key) in zip(runtime_names, layout):
            # LTJ's standalone compiler reads constexpr annotations from source.
            # A numeric token must accompany every annotated scalar parameter.
            if kind.startswith("scalar_") and annotations[name] == "tl.constexpr":
                values[name] = plan["parameters"][key]
                continue
            if kind == "tensor" or kind == "tensor_alias":
                index = tensor_index if kind == "tensor" else key
                tensor = plan["tensors"][index]
                if kind == "tensor":
                    tensor_index += 1
                uid = tensor["uid"]
                argument = dict(
                    index=len(arguments),
                    name=name,
                    source="graph_workspace" if tensor["virtual"] else "binding",
                    type="pointer",
                    uid=uid,
                    size=_tensor_storage_size(tensor),
                    alignment=256 if tensor["virtual"] else tensor.get("alignment", 16),
                )
                if tensor["virtual"]:
                    argument["offset"] = workspace[uid][0]
                arguments.append(argument)
                tokens[name] = signature[name]
            elif kind in {"scalar_i32", "scalar_f32"}:
                value = plan["parameters"][key]
                scalar_type = "i32" if kind == "scalar_i32" else "f32"
                arguments.append(
                    dict(
                        index=len(arguments),
                        name=name,
                        source="scalar",
                        type=scalar_type,
                        value=value,
                    )
                )
                tokens[name] = "fp32" if scalar_type == "f32" else scalar_type
            else:
                raise ValueError(f"unsupported Ascend extended argument kind {kind}")
        for name in names:
            if name in tokens:
                continue
            value = values.get(name)
            if value is None and name in defaults:
                value = ast.literal_eval(defaults[name])
            if isinstance(value, bool):
                tokens[name] = "1" if value else "0"
            elif isinstance(value, (int, float)) and math.isfinite(value):
                tokens[name] = repr(value)
            else:
                raise ValueError(
                    f"Ascend constexpr {entry}.{name} is not numeric: {value!r}"
                )
        if len(arguments) > 64 or any(
            type(x) is not int or not 1 <= x <= 2**31 - 1 for x in grid
        ):
            raise ValueError("Ascend extended launch exceeds ABI limits")
        filename, sha = _materialize_source(
            output_directory=output_directory,
            source_bytes=data,
            compiler_identity_sha256=identity["identity_sha256"],
        )
        stages.append(
            dict(
                stage_id=plan["stage_id"],
                operation=plan["operation"],
                kernel_family="extended",
                source_node_ids=plan["source_node_ids"],
                dependencies=plan["dependencies"],
                kernel=dict(
                    source=source.name,
                    entry_point=entry,
                    source_sha256=sha,
                    materialized_source=dict(file=filename, size=len(data), sha256=sha),
                ),
                argument_sources=arguments,
                candidates=[
                    dict(
                        candidate_id="default",
                        launch_abi="ltj_npu_raw_v1",
                        payload=dict(
                            schema_version=1,
                            source_path=filename,
                            source_sha256=sha,
                            entry_point=entry,
                            full_signature=",".join(tokens[name] for name in names),
                            grid=list(grid),
                            compile_options=dict(num_warps=4, num_stages=1),
                        ),
                    )
                ],
                autotune=dict(
                    schema_version=1,
                    enabled=False,
                    selection=dict(state="fixed", candidate_id="default"),
                ),
            )
        )
    return stages, workspace_size
