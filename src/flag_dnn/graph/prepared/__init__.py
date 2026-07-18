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

from flag_dnn.graph.prepared.core import (
    GenericPrepareRunFn,
    PrepareRunFn,
    PreparedKernelPipelineSpec,
    PreparedPipelineStepSpec,
    PreparedSingleKernelRunSpec,
    PreparedSingleKernelSpec,
    PreparedTensorCache,
    RunFn,
    RuntimeTensorCheck,
    get_cached_empty_tensor,
    make_kernel_pipeline_launcher,
    make_kernel_pipeline_run_fn,
    make_single_kernel_launcher,
    make_single_kernel_run_fn,
    make_static_cached_call,
    prepare_run_fn,
    register_generic_prepared_run_fn,
    register_prepared_run_fn,
    runtime_tensor_checks_from_specs,
    runtime_tensor_checks_pass,
)

__all__ = (
    "RunFn",
    "PrepareRunFn",
    "GenericPrepareRunFn",
    "PreparedTensorCache",
    "RuntimeTensorCheck",
    "PreparedPipelineStepSpec",
    "PreparedKernelPipelineSpec",
    "PreparedSingleKernelSpec",
    "PreparedSingleKernelRunSpec",
    "make_kernel_pipeline_launcher",
    "make_kernel_pipeline_run_fn",
    "make_static_cached_call",
    "get_cached_empty_tensor",
    "make_single_kernel_launcher",
    "make_single_kernel_run_fn",
    "prepare_run_fn",
    "register_generic_prepared_run_fn",
    "register_prepared_run_fn",
    "runtime_tensor_checks_from_specs",
    "runtime_tensor_checks_pass",
)
