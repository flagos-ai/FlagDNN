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

"""Regression tests for concurrent graph capture contexts."""

import threading

from flag_dnn.graph.capture import GraphCapture, current_capture
from flag_dnn.graph.tensor import TensorSpec


def test_graph_capture_stack_is_thread_local():
    spec = TensorSpec("x", (8,), "float32", device="cpu")
    first_entered = threading.Event()
    second_entered = threading.Event()
    first_checked = threading.Event()
    second_exited = threading.Event()
    results = {}
    errors = []

    def first_worker():
        try:
            with GraphCapture([spec]) as capture:
                first_entered.set()
                if not second_entered.wait(timeout=5):
                    raise RuntimeError("second capture did not start")
                results["first"] = current_capture() is capture
                first_checked.set()
                if not second_exited.wait(timeout=5):
                    raise RuntimeError("second capture did not exit")
        except BaseException as exc:  # pragma: no cover - thread handoff
            errors.append(exc)
            first_checked.set()

    def second_worker():
        try:
            if not first_entered.wait(timeout=5):
                raise RuntimeError("first capture did not start")
            with GraphCapture([spec]) as capture:
                second_entered.set()
                if not first_checked.wait(timeout=5):
                    raise RuntimeError("first capture was not checked")
                results["second"] = current_capture() is capture
        except BaseException as exc:  # pragma: no cover - thread handoff
            errors.append(exc)
            second_entered.set()
        finally:
            second_exited.set()

    first_thread = threading.Thread(target=first_worker)
    second_thread = threading.Thread(target=second_worker)
    first_thread.start()
    second_thread.start()
    first_thread.join(timeout=10)
    second_thread.join(timeout=10)

    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert not errors
    assert results == {"first": True, "second": True}
