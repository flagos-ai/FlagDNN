/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "backends/ascend/engines/python_stdout_containment.hpp"

#include <Python.h>

#include <cerrno>
#include <cstring>
#include <exception>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

namespace flagdnn::ascend::detail {
namespace {

[[nodiscard]] std::string consume_python_error() {
  if (PyErr_Occurred() == nullptr) {
    return "unknown Python error";
  }
  PyObject* type = nullptr;
  PyObject* value = nullptr;
  PyObject* traceback = nullptr;
  PyErr_Fetch(&type, &value, &traceback);
  PyErr_NormalizeException(&type, &value, &traceback);

  PyObject* source = value != nullptr ? value : type;
  PyObject* rendered = source == nullptr ? nullptr : PyObject_Str(source);
  std::string result = "unprintable Python error";
  if (rendered != nullptr) {
    const char* text = PyUnicode_AsUTF8(rendered);
    if (text != nullptr) {
      result = text;
    }
  }
  Py_XDECREF(rendered);
  Py_XDECREF(type);
  Py_XDECREF(value);
  Py_XDECREF(traceback);
  PyErr_Clear();
  return result;
}

class ScopedPythonStdoutContainment final {
 public:
  ScopedPythonStdoutContainment() {
    if (Py_IsInitialized() == 0) {
      throw std::runtime_error(
          "cannot contain compiler stdout before Python initialization");
    }
    gil_state_ = PyGILState_Ensure();
    gil_acquired_ = true;
    try {
      if (PyErr_Occurred() != nullptr) {
        throw std::runtime_error(
            "embedded Python entered stdout containment with a pending "
            "exception: " +
            consume_python_error());
      }

      install_non_tle_sentinel();

      original_stdout_ = PySys_GetObject("stdout");  // Borrowed reference.
      if (original_stdout_ == nullptr) {
        throw std::runtime_error(
            "embedded Python has no sys.stdout to contain");
      }
      Py_INCREF(original_stdout_);

      do {
        sink_fd_ = ::open(
            "/dev/null", O_WRONLY | O_CLOEXEC | O_NOFOLLOW);
      } while (sink_fd_ < 0 && errno == EINTR);
      if (sink_fd_ < 0) {
        throw std::runtime_error(
            "cannot open /dev/null for compiler stdout containment: " +
            std::string(std::strerror(errno)));
      }
      struct stat sink_status {};
      if (::fstat(sink_fd_, &sink_status) != 0) {
        throw std::runtime_error(
            "cannot inspect compiler stdout containment sink: " +
            std::string(std::strerror(errno)));
      }
      if (!S_ISCHR(sink_status.st_mode)) {
        throw std::runtime_error(
            "compiler stdout containment sink is not a character device");
      }

      /* FlagDNN owns sink_fd_.  closefd=0 keeps ownership unambiguous on every
       * PyFile_FromFd success/failure path; restoration decrefs the Python text
       * wrapper before closing the descriptor exactly once. */
      sink_ = PyFile_FromFd(sink_fd_,
                            "/dev/null",
                            "w",
                            1,
                            "utf-8",
                            "strict",
                            nullptr,
                            0);
      if (sink_ == nullptr) {
        throw std::runtime_error(
            "cannot create /dev/null compiler stdout text stream: " +
            consume_python_error());
      }
      if (PySys_SetObject("stdout", sink_) != 0) {
        throw std::runtime_error(
            "cannot redirect compiler sys.stdout: " +
            consume_python_error());
      }
      redirected_ = true;
    } catch (...) {
      cleanup_noexcept();
      throw;
    }
  }

  ~ScopedPythonStdoutContainment() { cleanup_noexcept(); }

  ScopedPythonStdoutContainment(const ScopedPythonStdoutContainment&) = delete;
  ScopedPythonStdoutContainment& operator=(
      const ScopedPythonStdoutContainment&) = delete;

  void restore() {
    restore_non_tle_sentinel();
    if (!redirected_) {
      return;
    }
    if (PySys_SetObject("stdout", original_stdout_) != 0) {
      throw std::runtime_error(
          "cannot restore caller sys.stdout after embedded compiler call: " +
          consume_python_error());
    }
    redirected_ = false;
    Py_CLEAR(sink_);
    Py_CLEAR(original_stdout_);
    close_sink();
  }

 private:
  void install_non_tle_sentinel() {
    modules_ = PyImport_GetModuleDict();  // Borrowed reference.
    if (modules_ == nullptr || PyDict_Check(modules_) == 0) {
      throw std::runtime_error("embedded Python sys.modules is unavailable");
    }
    if (PyDict_GetItemString(modules_, "triton.experimental.tle") != nullptr) {
      throw std::runtime_error(
          "Triton TLE is already imported in the non-TLE compiler domain");
    }

    PyObject* dsa = nullptr;
    PyObject* pipeline = nullptr;
    PyObject* parallel = nullptr;
    tle_sentinel_ = PyModule_New("triton.experimental.tle");
    dsa = PyModule_New("triton.experimental.tle.dsa");
    pipeline = PyObject_CallNoArgs(
        reinterpret_cast<PyObject*>(&PyBaseObject_Type));
    parallel = PyObject_CallNoArgs(
        reinterpret_cast<PyObject*>(&PyBaseObject_Type));
    const bool failed =
        tle_sentinel_ == nullptr || dsa == nullptr || pipeline == nullptr ||
        parallel == nullptr ||
        PyObject_SetAttrString(dsa, "pipeline", pipeline) != 0 ||
        PyObject_SetAttrString(dsa, "parallel", parallel) != 0 ||
        PyObject_SetAttrString(tle_sentinel_, "dsa", dsa) != 0 ||
        PyDict_SetItemString(modules_,
                             "triton.experimental.tle",
                             tle_sentinel_) != 0;
    Py_XDECREF(dsa);
    Py_XDECREF(pipeline);
    Py_XDECREF(parallel);
    if (failed) {
      const std::string error = consume_python_error();
      Py_CLEAR(tle_sentinel_);
      throw std::runtime_error(
          "cannot install the scoped non-TLE compiler sentinel: " + error);
    }
    sentinel_installed_ = true;
  }

  void restore_non_tle_sentinel() {
    if (!sentinel_installed_) {
      return;
    }
    PyObject* current =
        PyDict_GetItemString(modules_, "triton.experimental.tle");
    if (current != tle_sentinel_) {
      throw std::runtime_error(
          "scoped non-TLE compiler sentinel changed during compilation");
    }
    if (PyDict_DelItemString(modules_, "triton.experimental.tle") != 0) {
      throw std::runtime_error(
          "cannot remove the scoped non-TLE compiler sentinel: " +
          consume_python_error());
    }
    sentinel_installed_ = false;
    Py_CLEAR(tle_sentinel_);
    modules_ = nullptr;
  }

  void close_sink() {
    if (sink_fd_ < 0) {
      return;
    }
    const int descriptor = sink_fd_;
    sink_fd_ = -1;
    if (::close(descriptor) != 0) {
      throw std::runtime_error(
          "cannot close compiler stdout containment descriptor: " +
          std::string(std::strerror(errno)));
    }
  }

  void close_sink_noexcept() noexcept {
    if (sink_fd_ >= 0) {
      const int descriptor = sink_fd_;
      sink_fd_ = -1;
      (void)::close(descriptor);
    }
  }

  void cleanup_noexcept() noexcept {
    if (gil_acquired_) {
      if (sentinel_installed_ && modules_ != nullptr &&
          PyDict_GetItemString(modules_, "triton.experimental.tle") ==
              tle_sentinel_) {
        if (PyDict_DelItemString(modules_, "triton.experimental.tle") != 0) {
          PyErr_Clear();
        }
      }
      sentinel_installed_ = false;
      Py_CLEAR(tle_sentinel_);
      modules_ = nullptr;
      if (redirected_) {
        if (PySys_SetObject("stdout", original_stdout_) != 0) {
          PyErr_Clear();
        } else {
          redirected_ = false;
        }
      }
      Py_CLEAR(sink_);
      Py_CLEAR(original_stdout_);
      close_sink_noexcept();
      PyGILState_Release(gil_state_);
      gil_acquired_ = false;
    }
  }

  PyGILState_STATE gil_state_ = PyGILState_UNLOCKED;
  PyObject* modules_ = nullptr;
  PyObject* tle_sentinel_ = nullptr;
  PyObject* original_stdout_ = nullptr;
  PyObject* sink_ = nullptr;
  int sink_fd_ = -1;
  bool gil_acquired_ = false;
  bool sentinel_installed_ = false;
  bool redirected_ = false;
};

}  // namespace

void run_with_contained_python_stdout(
    ContainedPythonStdoutOperation operation,
    void* context) {
  if (operation == nullptr) {
    throw std::invalid_argument(
        "contained Python stdout operation must not be null");
  }

  ScopedPythonStdoutContainment containment;
  std::exception_ptr operation_failure;
  try {
    operation(context);
  } catch (...) {
    operation_failure = std::current_exception();
  }

  /* Restore explicitly so a failure is observable by the raw-launch terminal
   * path.  The RAII destructor is a final no-throw recovery attempt only. */
  containment.restore();
  if (operation_failure != nullptr) {
    std::rethrow_exception(operation_failure);
  }
}

}  // namespace flagdnn::ascend::detail
