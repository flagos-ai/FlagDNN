"""Exercise C++ artifact validation with real Python-emitted kernel assets."""

from pathlib import Path
import copy
import hashlib
import importlib
import json
import subprocess
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "compiler"))
from flagdnn_codegen.provider_loader import get_provider

package = get_provider("ascend").__package__
emitter = importlib.import_module(package + ".codegen.extended")
dispatch = importlib.import_module(package + ".dispatch.extended")


def tensor(uid, dtype):
    return dict(
        uid=uid,
        data_type=dtype,
        dimensions=[1, 1, 2, 2],
        strides=[4, 4, 2, 1],
        alignment=16,
        virtual=False,
    )


def main():
    tensors = [tensor(1, "float16"), tensor(2, "float32"), tensor(3, "float16")]
    tensors[1].update(dimensions=[2, 1, 1, 2], strides=[2, 2, 2, 1])
    parameters = dict(rope_dim=2, n_elements=4, output_scale=1.0)
    source, config = dispatch.configuration("rope", parameters, tensors)
    plan = dict(
        configuration=config,
        source=source,
        tensors=tensors,
        parameters=parameters,
        stage_id=0,
        operation="rope",
        source_node_ids=[0],
        dependencies=[],
    )
    request = dict(
        compiler_identity="a" * 64,
        graph=dict(
            node_count=1,
            tensor_count=3,
            tensors=tensors,
            nodes=[
                dict(
                    id=0,
                    type="rope",
                    inputs=[dict(uid=1), dict(uid=2)],
                    outputs=[dict(uid=3)],
                )
            ],
        ),
    )
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        with patch.object(emitter, "plan_graph", return_value=([plan], {}, 0)):
            stages, size = emitter.emit({}, directory, {"identity_sha256": "a" * 64})
        hashes = [s["kernel"]["source_sha256"] for s in stages]
        manifest = dict(
            graph_node_count=1,
            workspace_size=size,
            source_sha256=hashlib.sha256(
                json.dumps(hashes, separators=(",", ":")).encode()
            ).hexdigest(),
            program=dict(schema_version=4, stage_count=1, stages=stages),
        )
        request_path = directory / "request.json"
        request_path.write_text(json.dumps(request))
        arguments = [sys.argv[1], str(request_path), str(directory)]

        def check(value, error=None):
            path = directory / f"manifest-{len(arguments)}.json"
            path.write_text(json.dumps(value))
            arguments.extend([str(path), error or ""])

        check(manifest)
        bad = copy.deepcopy(manifest)
        stage = bad["program"]["stages"][0]
        payload = stage["candidates"][0]["payload"]
        payload["full_signature"] = payload["full_signature"].replace(
            "*fp16", "*fp32", 1
        )
        check(bad, "pointer storage width mismatch")
        bad = copy.deepcopy(manifest)
        bad["program"]["stages"][0]["argument_sources"][0]["uid"] = 99
        check(bad, "tensor binding")
        bad = copy.deepcopy(manifest)
        bad["program"]["stages"][0]["argument_sources"][0]["size"] += 2
        check(bad, "argument metadata mismatch")
        bad = copy.deepcopy(manifest)
        bad["program"]["stages"][0]["kernel"]["source_sha256"] = "b" * 64
        check(bad, "not pinned")
        bad = copy.deepcopy(manifest)
        bad["workspace_size"] = 256
        check(bad, "workspace mismatch")
        bad = copy.deepcopy(manifest)
        bad["program"]["stages"][0]["dependencies"] = [0]
        check(bad, "non-topological")
        result = subprocess.run(arguments, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        print(
            "Ascend extended artifact contract: valid launch and six malformed artifacts checked"
        )


if __name__ == "__main__":
    main()
