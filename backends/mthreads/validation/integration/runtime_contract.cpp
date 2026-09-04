/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "backends/backend_api.h"

#include <musa.h>

#include <dlfcn.h>

#include <array>
#include <cstddef>
#include <iostream>
#include <regex>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

void require(bool condition, std::string message) {
  if (!condition) {
    throw std::runtime_error(std::move(message));
  }
}

void check_mu(MUresult status, std::string_view operation) {
  if (status == MUSA_SUCCESS) {
    return;
  }
  const char* name = nullptr;
  const char* description = nullptr;
  static_cast<void>(muGetErrorName(status, &name));
  static_cast<void>(muGetErrorString(status, &description));
  throw std::runtime_error(
      std::string(operation) + " failed: " +
      (name == nullptr ? "unknown" : name) + ": " +
      (description == nullptr ? "unknown" : description));
}

class SharedLibrary final {
 public:
  explicit SharedLibrary(const char* path) {
    handle_ = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (handle_ == nullptr) {
      const char* error = dlerror();
      throw std::runtime_error(
          std::string("dlopen failed: ") +
          (error == nullptr ? "unknown" : error));
    }
  }

  ~SharedLibrary() {
    if (handle_ != nullptr) {
      static_cast<void>(dlclose(handle_));
    }
  }

  SharedLibrary(const SharedLibrary&) = delete;
  SharedLibrary& operator=(const SharedLibrary&) = delete;

  [[nodiscard]] void* symbol(const char* name) const {
    static_cast<void>(dlerror());
    void* result = dlsym(handle_, name);
    const char* error = dlerror();
    if (error != nullptr || result == nullptr) {
      throw std::runtime_error(
          std::string("dlsym failed for ") + name + ": " +
          (error == nullptr ? "unknown" : error));
    }
    return result;
  }

 private:
  void* handle_ = nullptr;
};

std::string last_error(const flagdnnBackendApiV2* api) {
  const char* value = api->get_last_error();
  return value == nullptr ? "" : value;
}

}  // namespace

int main(int argc, char* argv[]) {
  try {
    require(argc == 2, "usage: runtime_contract <backend-library>");
    check_mu(muInit(0), "muInit");
    MUcontext initial_context = nullptr;
    check_mu(muCtxGetCurrent(&initial_context), "muCtxGetCurrent(initial)");

    SharedLibrary library(argv[1]);
    const auto get_api =
        reinterpret_cast<flagdnnBackendGetApiV2Function>(
            library.symbol(FLAGDNN_BACKEND_GET_API_V2_SYMBOL));
    const flagdnnBackendApiV2* api = get_api();
    require(api != nullptr, "backend API is null");
    require(
        api->struct_size == sizeof(flagdnnBackendApiV2),
        "backend API struct_size differs");
    require(
        api->abi_version == FLAGDNN_BACKEND_ABI_VERSION,
        "backend ABI version differs");
    require(
        api->backend_name != nullptr &&
            std::string_view(api->backend_name) == "mthreads",
        "backend name differs");

    require(
        api->create_context(0, nullptr) ==
            FLAGDNN_BACKEND_RESULT_INVALID_VALUE,
        "null context output was not rejected");
    require(
        last_error(api).find("output pointer") != std::string::npos,
        "null context output diagnostic is missing");

    void* context = reinterpret_cast<void*>(0x1);
    require(
        api->create_context(-1, &context) ==
            FLAGDNN_BACKEND_RESULT_INVALID_VALUE,
        "negative device ordinal was not rejected");
    require(context == nullptr, "failed context creation returned a handle");
    require(
        last_error(api).find("nonnegative") != std::string::npos,
        "negative ordinal diagnostic is missing");

    require(
        api->create_context(0, &context) ==
            FLAGDNN_BACKEND_RESULT_SUCCESS,
        "device 0 context creation failed: " + last_error(api));
    require(context != nullptr, "context creation returned null");

    MUcontext after_create = nullptr;
    check_mu(
        muCtxGetCurrent(&after_create),
        "muCtxGetCurrent(after create)");
    require(
        after_create == initial_context,
        "context creation changed the caller's current context");

    std::size_t required_size = 0;
    require(
        api->get_target_fingerprint(
            context, nullptr, 0, &required_size) ==
            FLAGDNN_BACKEND_RESULT_INVALID_VALUE,
        "short target buffer was not rejected");
    require(
        required_size > 1 &&
            required_size <= FLAGDNN_BACKEND_MAX_TARGET_FINGERPRINT,
        "target required size is invalid");
    require(
        last_error(api).find("too small") != std::string::npos,
        "short target diagnostic is missing");

    std::array<char, FLAGDNN_BACKEND_MAX_TARGET_FINGERPRINT> target{};
    require(
        api->get_target_fingerprint(
            context, target.data(), target.size(), &required_size) ==
            FLAGDNN_BACKEND_RESULT_SUCCESS,
        "target query failed: " + last_error(api));
    require(
        target.at(required_size - 1) == '\0',
        "target is not NUL terminated");
    require(
        std::regex_match(
            target.data(),
            std::regex(
                R"(^musa-[a-z0-9_.+-]+-cc[0-9]+-w[0-9]+$)")),
        "target fingerprint has an invalid shape: " +
            std::string(target.data()));

    const char graph_ir[] = "{}";
    flagdnnBackendBuildInputV2 input{
        sizeof(flagdnnBackendBuildInputV2),
        graph_ir,
        sizeof(graph_ir) - 1,
        ".",
        "0000000000000000000000000000000000000000000000000000000000000000",
    };
    void* executable = reinterpret_cast<void*>(0x1);
    std::size_t workspace_size = 1;
    require(
        api->create_executable(
            context, &input, &executable, &workspace_size) ==
            FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
        "invalid artifact did not return COMPILATION_FAILED");
    require(executable == nullptr, "failed engine creation returned a handle");
    require(workspace_size == 0, "failed engine creation returned workspace");
    require(
        last_error(api).find("SHA-256") != std::string::npos,
        "artifact trust-boundary diagnostic is missing");

    MUcontext after_build = nullptr;
    check_mu(
        muCtxGetCurrent(&after_build),
        "muCtxGetCurrent(after build)");
    require(
        after_build == initial_context,
        "engine build did not restore the caller's current context");

    api->destroy_context(context);
    context = nullptr;
    MUcontext after_destroy = nullptr;
    check_mu(
        muCtxGetCurrent(&after_destroy),
        "muCtxGetCurrent(after destroy)");
    require(
        after_destroy == initial_context,
        "context destruction changed the caller's current context");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "mthreads runtime contract failed: "
              << error.what() << '\n';
    return 1;
  }
}
