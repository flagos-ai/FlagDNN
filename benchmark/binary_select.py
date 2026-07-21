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

from __future__ import annotations

import math

import torch

import flag_dnn
from benchmark import consts
from benchmark.base import DnnCompareBenchmark


class BinarySelectBenchmarkBase(DnnCompareBenchmark):
    op_name = "binary_select"

    def make_inputs(self, case, dtype):
        x_shape, y_shape, mask_shape = case
        return (
            consts.pointwise_randn(x_shape, dtype, flag_dnn.device),
            consts.pointwise_randn(y_shape, dtype, flag_dnn.device),
            consts.pointwise_bool(mask_shape, flag_dnn.device),
        )

    def build_baseline_runner(self, inputs):
        x, y, mask = inputs
        return self.baseline.prepare(self.op_name, x, y, mask)

    def build_flag_dnn_runner(self, inputs):
        x, y, mask = inputs

        @flag_dnn.graph
        def binary_select_graph(x, y, mask):
            return flag_dnn.binary_select(
                input0=x,
                input1=y,
                mask=mask,
                compute_data_type="float32",
                name=self.op_name,
            )

        compiled = flag_dnn.compile(
            binary_select_graph,
            inputs=[
                flag_dnn.TensorSpec.from_tensor(x, "x"),
                flag_dnn.TensorSpec.from_tensor(y, "y"),
                flag_dnn.TensorSpec.from_tensor(mask, "mask"),
            ],
            options=consts.compile_options(),
        )
        assert [node.op_type for node in compiled.graph.nodes] == [
            self.op_name
        ]
        return compiled.bind(x, y, mask)

    def transfer_bytes(self, inputs):
        x, y, mask = inputs
        output_shape = torch.broadcast_shapes(x.shape, y.shape, mask.shape)
        return (
            x.numel() * x.element_size()
            + y.numel() * y.element_size()
            + mask.numel() * mask.element_size()
            + math.prod(output_shape) * x.element_size()
        )
