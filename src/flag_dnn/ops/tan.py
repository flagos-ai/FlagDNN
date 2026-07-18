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

from typing import Optional

import torch

from flag_dnn import runtime
from flag_dnn.ops.unary import unary

_PORTABLE_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def tan(
    input: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    compute_data_type=None,
    name: str = "",
) -> torch.Tensor:
    del compute_data_type, name
    if input.dtype not in _PORTABLE_DTYPES:
        raise NotImplementedError(
            f"flag_dnn tan does not support dtype={input.dtype} "
            f"on device={runtime.device.name}"
        )
    if input.device.type != runtime.device.name:
        raise RuntimeError(
            f"flag_dnn tan expected a {runtime.device.name} tensor, "
            f"got device={input.device}"
        )
    return unary(input, out=out, op_type="tan")
