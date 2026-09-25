/* Copyright (c) 2026 FlagOS Contributors. SPDX-License-Identifier: Apache-2.0
 */

#include "backends/hygon/engines/python_runtime.hpp"

#include <Python.h>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace {

std::array<PyObject *, 2> exit_objects{};
std::size_t exit_object_count = 0;
unsigned int generation = 0;

void release_extension_object() {
  if (!Py_IsInitialized() || !PyGILState_Check() || exit_object_count == 0) {
    std::fputs("Hygon extension exit probe has no live Python GIL\n", stderr);
    std::_Exit(86);
  }
  Py_DECREF(exit_objects[--exit_object_count]);
}

} // namespace

extern "C" {

void flagdnn_python_runtime_initialize() {
  flagdnn::hygon::detail::initialize_embedded_python();
}

void flagdnn_python_runtime_guard() {
  flagdnn::hygon::detail::guard_python_shutdown_after_imports();
}

void flagdnn_python_runtime_register_exit_probe() {
  if (exit_object_count == exit_objects.size()) {
    throw std::runtime_error("too many Python extension exit probes");
  }
  const PyGILState_STATE gil = PyGILState_Ensure();
  PyObject *object = reinterpret_cast<PyObject *>(&PyType_Type);
  Py_INCREF(object);
  exit_objects[exit_object_count++] = object;
  PyGILState_Release(gil);
  if (std::atexit(release_extension_object) != 0) {
    throw std::runtime_error("cannot register Python extension exit probe");
  }
}

unsigned int flagdnn_python_runtime_next_generation() { return ++generation; }

} // extern "C"
