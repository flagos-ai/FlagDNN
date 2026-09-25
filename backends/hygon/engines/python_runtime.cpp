/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "backends/hygon/engines/python_runtime.hpp"

#include <Python.h>
#include <dlfcn.h>

#include <cstdint>
#include <cstdlib>
#include <stdexcept>

namespace flagdnn::hygon::detail {
namespace {

bool owns_interpreter = false;
void *runtime_library_handle = nullptr;

void hold_gil_for_extension_destructors() noexcept {
  if (Py_IsInitialized()) {
    // Some Triton builds keep pybind11 types in C++ static objects. Those
    // objects outlive individual engines and still need Python and its GIL
    // during process teardown. Keep the owned interpreter alive and the GIL
    // held for the remaining exit handlers; do not finalize or release here.
    (void)PyGILState_Ensure();
  }
}

void retain_runtime_library() {
  if (runtime_library_handle != nullptr) {
    return;
  }
  Dl_info information{};
  const auto address = reinterpret_cast<void *>(
      reinterpret_cast<std::uintptr_t>(&hold_gil_for_extension_destructors));
  if (dladdr(address, &information) == 0 || information.dli_fname == nullptr) {
    throw std::runtime_error("cannot locate the Hygon Python runtime owner");
  }
  // The core can dlclose its last backend handle before process exit. Retain
  // this DSO so its ownership state and registered callbacks stay valid, and
  // the callback cannot run prematurely at the last graph/handle destruction.
  runtime_library_handle =
      dlopen(information.dli_fname, RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD);
  if (runtime_library_handle == nullptr) {
    throw std::runtime_error("cannot retain the Hygon Python runtime owner");
  }
}

} // namespace

void initialize_embedded_python() {
  if (Py_IsInitialized()) {
    return;
  }
  retain_runtime_library();
  Py_InitializeEx(0);
  if (!Py_IsInitialized()) {
    throw std::runtime_error("cannot initialize Python for Hygon JIT");
  }
  owns_interpreter = true;
  // Runtime work on other host threads must still be able to acquire the GIL.
  (void)PyEval_SaveThread();
}

void guard_python_shutdown_after_imports() {
  if (!owns_interpreter) {
    return;
  }
  if (std::atexit(hold_gil_for_extension_destructors) != 0) {
    throw std::runtime_error("cannot register Hygon Python exit handling");
  }
}

} // namespace flagdnn::hygon::detail
