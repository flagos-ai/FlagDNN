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

"""Per-family graph op definitions.

Each module here defines one op family end-to-end in a single place:

* ``normalize`` -- capture-time conversion of user args/kwargs into graph input
  value ids plus immutable attrs;
* ``*_shape`` -- build-time output ``TensorSpec`` inference;
* ``_run_*`` -- the eager-fallback run functions used at replay;
* ``register(flag_ops)`` -- declares the family's ops via ``OpDef``.

To add or change an operator, edit the matching family module; the families are
wired together by ``flag_dnn.graph.registry.ops.register_default_ops``. Shared
run helpers live in ``_run_common``; shared shape/normalize helpers in
``common``.
"""
