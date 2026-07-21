# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

import flag_dnn
from flag_dnn import runtime


def _compile_add(x, y):
    @flag_dnn.graph
    def add_graph(left, right):
        return flag_dnn.add(left, right)

    return flag_dnn.compile(
        add_graph,
        inputs=[x, y],
        options={"cache": None, "validate_inputs": False},
    )


def test_compiled_graph_run_returns_independent_outputs():
    first_left = torch.ones(16, device=flag_dnn.device)
    first_right = torch.ones(16, device=flag_dnn.device)
    compiled = _compile_add(first_left, first_right)

    first = compiled.run(first_left, first_right)
    runtime.torch_device_fn.synchronize()
    first_snapshot = first.cpu().clone()

    second_left = torch.full_like(first_left, 10)
    second_right = torch.full_like(first_right, 20)
    second = compiled.run(second_left, second_right)
    runtime.torch_device_fn.synchronize()

    assert first.data_ptr() != second.data_ptr()
    torch.testing.assert_close(first.cpu(), first_snapshot)
    torch.testing.assert_close(second.cpu(), torch.full((16,), 30.0))


def test_compiled_graph_bind_keeps_static_output_for_fast_replay():
    left = torch.ones(16, device=flag_dnn.device)
    right = torch.ones(16, device=flag_dnn.device)
    compiled = _compile_add(left, right)
    replay = compiled.bind(left, right)

    first = replay()
    runtime.torch_device_fn.synchronize()
    left.fill_(3)
    right.fill_(4)
    second = replay()
    runtime.torch_device_fn.synchronize()

    assert first.data_ptr() == second.data_ptr()
    torch.testing.assert_close(second.cpu(), torch.full((16,), 7.0))
