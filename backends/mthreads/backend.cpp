/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "backends/backend_api.h"

#include "backends/mthreads/context.hpp"
#include "backends/mthreads/engines/engine.hpp"
#include "backends/mthreads/error.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <memory>
#include <new>
#include <string>
#include <utility>

#if defined(__GNUC__) || defined(__clang__)
#define FLAGDNN_BACKEND_EXPORT __attribute__((visibility("default")))
#else
#define FLAGDNN_BACKEND_EXPORT
#endif

namespace flagdnn::mthreads {
namespace {

class ErrorState final {
 public:
  ErrorState& operator=(const char* message) noexcept {
    try {
      value_ = message == nullptr ? "" : message;
    } catch (...) {
      value_.clear();
    }
    return *this;
  }

  void clear() noexcept { value_.clear(); }
  [[nodiscard]] const char* c_str() const noexcept {
    return value_.c_str();
  }

 private:
  std::string value_;
};

thread_local ErrorState current_error;

template <typename Function>
flagdnnBackendResult_t plugin_call(Function&& function) noexcept {
  current_error.clear();
  try {
    std::forward<Function>(function)();
    return FLAGDNN_BACKEND_RESULT_SUCCESS;
  } catch (const MthreadsError& error) {
    current_error = error.what();
    return error.result();
  } catch (const std::bad_alloc&) {
    current_error = "host memory allocation failed";
    return FLAGDNN_BACKEND_RESULT_ALLOC_FAILED;
  } catch (const std::exception& error) {
    current_error = error.what();
    return FLAGDNN_BACKEND_RESULT_INTERNAL_ERROR;
  } catch (...) {
    current_error = "unknown mthreads backend error";
    return FLAGDNN_BACKEND_RESULT_INTERNAL_ERROR;
  }
}

const char* get_last_error() noexcept {
  return current_error.c_str();
}

flagdnnBackendResult_t create_context(
    std::int32_t device_ordinal, void** context) noexcept {
  return plugin_call([&] {
    require(context != nullptr, "context output pointer is null");
    *context = nullptr;
    auto result = std::make_unique<MthreadsContext>(device_ordinal);
    *context = result.release();
  });
}

void destroy_context(void* context) noexcept {
  delete static_cast<MthreadsContext*>(context);
}

flagdnnBackendResult_t get_target_fingerprint(
    void* context,
    char* buffer,
    std::size_t buffer_size,
    std::size_t* required_size) noexcept {
  return plugin_call([&] {
    require(context != nullptr, "context is null");
    require(required_size != nullptr, "required size output is null");
    const std::string& target =
        static_cast<MthreadsContext*>(context)->target_fingerprint();
    *required_size = target.size() + 1;
    require(
        buffer != nullptr && buffer_size >= *required_size,
        "target fingerprint buffer is too small");
    std::memcpy(buffer, target.c_str(), *required_size);
  });
}

flagdnnBackendResult_t create_executable(
    void* context,
    const flagdnnBackendBuildInputV2* input,
    void** executable,
    std::size_t* workspace_size) noexcept {
  return plugin_call([&] {
    require(context != nullptr, "context is null");
    require(input != nullptr, "build input is null");
    require(
        input->struct_size >= sizeof(flagdnnBackendBuildInputV2),
        "build input structure is too small");
    require(executable != nullptr, "executable output pointer is null");
    require(
        workspace_size != nullptr,
        "workspace size output pointer is null");
    *executable = nullptr;
    *workspace_size = 0;

    const EngineBuildContext build_context =
        static_cast<MthreadsContext*>(context)->engine_build_context();
    ContextGuard guard(
        build_context.context, build_context.device);
    std::unique_ptr<ExecutionEngine> result =
        create_execution_engine(build_context, *input);
    *workspace_size = result->workspace_size();
    *executable = result.release();
  });
}

void destroy_executable(void* executable) noexcept {
  delete static_cast<ExecutionEngine*>(executable);
}

flagdnnBackendResult_t execute(
    void* executable,
    void* native_stream,
    const flagdnnBackendBindingV2 bindings[],
    std::size_t binding_count,
    void* workspace,
    std::size_t workspace_size) noexcept {
  return plugin_call([&] {
    require(executable != nullptr, "executable is null");
    static_cast<ExecutionEngine*>(executable)->execute(
        reinterpret_cast<MUstream>(native_stream),
        bindings,
        binding_count,
        workspace,
        workspace_size);
  });
}

const flagdnnBackendApiV2 api = {
    sizeof(flagdnnBackendApiV2),
    FLAGDNN_BACKEND_ABI_VERSION,
    "mthreads",
    &get_last_error,
    &create_context,
    &destroy_context,
    &get_target_fingerprint,
    &create_executable,
    &destroy_executable,
    &execute,
};

}  // namespace
}  // namespace flagdnn::mthreads

extern "C" FLAGDNN_BACKEND_EXPORT const flagdnnBackendApiV2*
flagdnnBackendGetApiV2(void) {
  return &flagdnn::mthreads::api;
}
