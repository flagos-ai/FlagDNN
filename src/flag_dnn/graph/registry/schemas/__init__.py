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
