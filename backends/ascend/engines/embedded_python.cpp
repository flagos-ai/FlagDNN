/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "backends/ascend/engines/embedded_python.hpp"

#include <Python.h>

#include <cstdlib>
#include <stdexcept>
#include <string>

namespace flagdnn::ascend::detail {
namespace {

[[nodiscard]] std::string status_message(const PyStatus& status) {
  std::string result = status.err_msg == nullptr ? "unknown Python error"
                                                  : status.err_msg;
  if (status.func != nullptr) {
    result = std::string(status.func) + ": " + result;
  }
  return result;
}

[[nodiscard]] PyThreadState*& embedded_main_thread_state() noexcept {
  static PyThreadState* state = nullptr;
  return state;
}

[[nodiscard]] bool& owns_embedded_interpreter() noexcept {
  static bool owns = false;
  return owns;
}

void hold_owned_interpreter_gil_at_process_exit() noexcept {
  if (owns_embedded_interpreter() &&
      embedded_main_thread_state() != nullptr && Py_IsInitialized() != 0) {
    /* The pinned extension has process-lifetime C++ static PyObjects. Its
     * later __cxa_atexit handlers DECREF them, so keep the GIL across only the
     * remaining process-exit handlers. */
    (void)PyGILState_Ensure();
  }
}

void register_owned_interpreter_exit_guard() {
  static bool registered = false;
  if (registered) {
    return;
  }
  if (std::atexit(&hold_owned_interpreter_gil_at_process_exit) != 0) {
    throw std::runtime_error(
        "cannot register embedded Python process-exit GIL guard");
  }
  registered = true;
}

}  // namespace

void initialize_embedded_python_from_program(const char* program_name) {
  if (program_name == nullptr || program_name[0] == '\0' ||
      Py_IsInitialized() != 0) {
    throw std::invalid_argument(
        "embedded Python requires a program and a clean runtime");
  }

  PyConfig config;
  PyConfig_InitPythonConfig(&config);
  config.install_signal_handlers = 0;
  PyStatus status =
      PyConfig_SetBytesString(&config, &config.program_name, program_name);
  if (PyStatus_Exception(status)) {
    const std::string message = status_message(status);
    PyConfig_Clear(&config);
    throw std::runtime_error(
        "cannot configure embedded Python program: " + message);
  }
  status = Py_InitializeFromConfig(&config);
  PyConfig_Clear(&config);
  if (PyStatus_Exception(status) || Py_IsInitialized() == 0) {
    throw std::runtime_error(
        "cannot initialize embedded Python: " + status_message(status));
  }
  owns_embedded_interpreter() = true;
}

void release_embedded_python_gil_for_process_lifetime() {
  if (!owns_embedded_interpreter() || Py_IsInitialized() == 0 ||
      embedded_main_thread_state() != nullptr) {
    throw std::runtime_error(
        "embedded Python GIL release requires one owned active interpreter");
  }

  /* Register after extension imports. std::atexit is LIFO, so the guard runs
   * before the imported extension's static PyObject destructors. */
  register_owned_interpreter_exit_guard();
  embedded_main_thread_state() = PyEval_SaveThread();
  if (embedded_main_thread_state() == nullptr) {
    throw std::runtime_error(
        "PyEval_SaveThread returned a null main thread state");
  }
}

}  // namespace flagdnn::ascend::detail
