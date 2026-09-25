/* Copyright (c) 2026 FlagOS Contributors. SPDX-License-Identifier: Apache-2.0
 */

#include <Python.h>

#include <dlfcn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>

namespace {

void require(bool condition, const char *message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

class Fixture {
public:
  explicit Fixture(const char *path) {
    library_ = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (library_ == nullptr) {
      const char *error = dlerror();
      throw std::runtime_error(error == nullptr ? "cannot open fixture"
                                                : error);
    }
    initialize = symbol<void (*)()>("flagdnn_python_runtime_initialize");
    guard = symbol<void (*)()>("flagdnn_python_runtime_guard");
    register_exit_probe =
        symbol<void (*)()>("flagdnn_python_runtime_register_exit_probe");
    next_generation =
        symbol<unsigned int (*)()>("flagdnn_python_runtime_next_generation");
  }

  ~Fixture() {
    if (library_ != nullptr) {
      (void)dlclose(library_);
    }
  }

  void close() {
    require(library_ != nullptr, "fixture was already closed");
    void *library = library_;
    library_ = nullptr;
    require(dlclose(library) == 0, "cannot close fixture");
  }

  void (*initialize)() = nullptr;
  void (*guard)() = nullptr;
  void (*register_exit_probe)() = nullptr;
  unsigned int (*next_generation)() = nullptr;

private:
  template <typename Function> Function symbol(const char *name) {
    void *value = dlsym(library_, name);
    require(value != nullptr, "fixture has no requested symbol");
    return reinterpret_cast<Function>(value);
  }

  void *library_ = nullptr;
};

class GilScope {
public:
  GilScope() : state_(PyGILState_Ensure()) {}
  ~GilScope() { PyGILState_Release(state_); }

private:
  PyGILState_STATE state_;
};

template <typename Operation> void on_worker(Operation operation) {
  std::exception_ptr failure;
  std::thread worker([&] {
    try {
      operation();
    } catch (...) {
      failure = std::current_exception();
    }
  });
  worker.join();
  if (failure != nullptr) {
    std::rethrow_exception(failure);
  }
}

void use_python_on_worker() {
  on_worker([] {
    GilScope gil;
    require(PyGILState_Check(), "worker did not acquire Python GIL");
    PyObject *value = PyLong_FromLong(123);
    require(value != nullptr, "worker could not create a Python object");
    const long actual = PyLong_AsLong(value);
    Py_DECREF(value);
    require(actual == 123, "worker Python object is invalid");
  });
  require(!PyGILState_Check(), "worker leaked Python GIL to its caller");
}

void fail_python_import() {
  GilScope gil;
  PyObject *module = PyImport_ImportModule("_flagdnn_missing_exit_contract");
  if (module != nullptr) {
    Py_DECREF(module);
    throw std::runtime_error("missing-module probe unexpectedly imported");
  }
  const bool expected = PyErr_ExceptionMatches(PyExc_ModuleNotFoundError);
  PyErr_Clear();
  require(expected, "missing-module probe raised the wrong Python error");
}

void run_child(const char *path, std::string_view mode) {
  // A GIL regression must fail promptly instead of hanging the parent runner.
  alarm(15);
  require(!Py_IsInitialized(), "child inherited an initialized interpreter");
  if (mode == "borrowed_host") {
    Py_InitializeEx(0);
    require(Py_IsInitialized(), "host could not initialize Python");
    PyThreadState *state = PyEval_SaveThread();
    {
      Fixture fixture(path);
      fixture.initialize();
      fixture.guard();
      require(!PyGILState_Check(), "borrowed runtime acquired host GIL");
      use_python_on_worker();
      fixture.close();
      require(!PyGILState_Check(), "borrowed runtime changed GIL on unload");
    }
    PyEval_RestoreThread(state);
    require(Py_FinalizeEx() == 0, "host could not finalize its interpreter");
    require(!Py_IsInitialized(), "borrowed runtime kept host Python alive");
    return;
  }

  Fixture fixture(path);
  require(fixture.next_generation() == 1, "fresh fixture retained prior state");
  if (mode == "worker_init") {
    on_worker([&] { fixture.initialize(); });
  } else {
    fixture.initialize();
  }
  require(Py_IsInitialized(), "owned runtime did not initialize Python");
  require(!PyGILState_Check(), "owned initialization did not release the GIL");
  use_python_on_worker();

  if (mode == "unload_reopen") {
    fixture.close();
    use_python_on_worker();
    Fixture reopened(path);
    require(reopened.next_generation() == 2,
            "owned runtime was unloaded when its last handle closed");
    reopened.initialize();
    reopened.register_exit_probe();
    reopened.guard();
    reopened.close();
    use_python_on_worker();
  } else {
    fixture.register_exit_probe();
    if (mode == "import_failure_retry") {
      fail_python_import();
      fixture.guard();
      // A failed call_once can retry and import another extension with a new
      // exit destructor, so the retry must register another shutdown guard.
      fixture.register_exit_probe();
      fail_python_import();
    }
    fixture.guard();
    require(!PyGILState_Check(),
            "shutdown guard retained GIL during execution");
    fixture.close();
    use_python_on_worker();
  }
  // Return normally so the real process/DSO exit callbacks run and DECREF the
  // simulated extension's type objects. An absent or late guard exits with 86.
}

void run_process(const char *executable, const char *path, const char *mode) {
  const pid_t child = fork();
  require(child >= 0, "cannot fork Python runtime contract");
  if (child == 0) {
    execl(executable, executable, path, mode, static_cast<char *>(nullptr));
    _exit(127);
  }
  int status = 0;
  pid_t waited;
  do {
    waited = waitpid(child, &status, 0);
  } while (waited < 0 && errno == EINTR);
  if (waited != child || !WIFEXITED(status) || WEXITSTATUS(status) != 0) {
    throw std::runtime_error(std::string("Python runtime mode failed: ") +
                             mode + " wait_status=" + std::to_string(status));
  }
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc == 3) {
      run_child(argv[1], argv[2]);
      return 0;
    }
    require(argc == 2, "usage: test_python_runtime FIXTURE_DSO [MODE]");
    for (const char *mode : {"owned", "worker_init", "import_failure_retry",
                             "borrowed_host", "unload_reopen"}) {
      run_process(argv[0], argv[1], mode);
    }
    std::cout << "Hygon Python runtime lifecycle contract: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "Hygon Python runtime lifecycle contract: " << error.what()
              << '\n';
    return 1;
  }
}
