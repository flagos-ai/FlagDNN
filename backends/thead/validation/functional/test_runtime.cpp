// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "backends/backend_api.h"
#include "backends/thead/validation/ppu_driver.hpp"

#include <flagdnn/flagdnn.hpp>

#include <cuda.h>
#include <dlfcn.h>

#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace {

namespace tv = flagdnn::validation::thead;

void require(bool condition, std::string message) {
  if (!condition) {
    throw std::runtime_error(std::move(message));
  }
}

class DynamicLibrary final {
 public:
  explicit DynamicLibrary(const char* path) {
    value_ = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (value_ == nullptr) {
      const char* detail = dlerror();
      throw std::runtime_error(
          std::string("cannot load THead backend plugin: ") +
          (detail == nullptr ? "unknown dynamic-loader error" : detail));
    }
  }

  ~DynamicLibrary() {
    if (value_ != nullptr) {
      (void)dlclose(value_);
    }
  }

  DynamicLibrary(const DynamicLibrary&) = delete;
  DynamicLibrary& operator=(const DynamicLibrary&) = delete;

  template <typename Function>
  [[nodiscard]] Function symbol(const char* name) const {
    dlerror();
    void* address = dlsym(value_, name);
    const char* detail = dlerror();
    if (detail != nullptr || address == nullptr) {
      throw std::runtime_error(std::string("cannot resolve ") + name +
                               ": " +
                               (detail == nullptr ? "null symbol" : detail));
    }
    return reinterpret_cast<Function>(address);
  }

 private:
  void* value_ = nullptr;
};

void require_backend_failure(const flagdnnBackendApiV2& api,
                             flagdnnBackendResult_t actual,
                             flagdnnBackendResult_t expected,
                             std::string_view expected_detail) {
  const char* detail = api.get_last_error();
  require(actual == expected,
          "unexpected THead backend error classification");
  require(detail != nullptr &&
              std::string_view(detail).find(expected_detail) !=
                  std::string_view::npos,
          "THead backend diagnostic did not contain the stable detail: " +
              std::string(expected_detail));
}

template <typename Function>
void require_flagdnn_failure(Function&& function,
                             flagdnnStatus_t expected,
                             std::string_view expected_detail) {
  try {
    std::forward<Function>(function)();
  } catch (const flagdnn::Error& error) {
    require(error.status() == expected,
            "unexpected public FlagDNN error classification");
    require(std::string_view(error.what()).find(expected_detail) !=
                std::string_view::npos,
            "public FlagDNN diagnostic did not contain the stable detail: " +
                std::string(expected_detail));
    return;
  }
  throw std::runtime_error("expected public FlagDNN call to fail");
}

void require_current_context(CUcontext expected, std::string_view stage) {
  require(tv::current_context() == expected,
          "THead changed the caller current context during " +
              std::string(stage));
}

}  // namespace

int main(int argc, char** argv) {
  try {
    require(argc == 2, "usage: test_runtime <thead-backend-plugin>");

    DynamicLibrary library(argv[1]);
    const auto get_api =
        library.symbol<flagdnnBackendGetApiV2Function>(
            FLAGDNN_BACKEND_GET_API_V2_SYMBOL);
    const flagdnnBackendApiV2* api = get_api();
    require(api != nullptr, "THead backend returned a null ABI table");
    require(api->struct_size >= sizeof(flagdnnBackendApiV2),
            "THead backend ABI table is too small");
    require(api->abi_version == FLAGDNN_BACKEND_ABI_VERSION_V2,
            "THead backend ABI version is not v2");
    require(api->backend_name != nullptr &&
                std::string_view(api->backend_name) == "thead",
            "THead backend name is not exactly thead");

    void* invalid_context = reinterpret_cast<void*>(1);
    require_backend_failure(
        *api,
        api->create_context(-1, &invalid_context),
        FLAGDNN_BACKEND_RESULT_INVALID_VALUE,
        "device ordinal must be nonnegative");
    require(invalid_context == nullptr,
            "failed context creation did not clear its output");
    require_backend_failure(*api,
                            api->create_context(0, nullptr),
                            FLAGDNN_BACKEND_RESULT_INVALID_VALUE,
                            "context output pointer is null");
    require_backend_failure(
        *api,
        api->execute(reinterpret_cast<void*>(1),
                     nullptr,
                     nullptr,
                     0,
                     nullptr,
                     0),
        FLAGDNN_BACKEND_RESULT_INVALID_VALUE,
        "native PPU stream is null");

    tv::check_driver(cuInit(0), "cuInit");
    int device_count = 0;
    tv::check_driver(cuDeviceGetCount(&device_count), "cuDeviceGetCount");
    require(device_count > 0, "PPU driver reported no devices");
    CUdevice device = 0;
    tv::check_driver(cuDeviceGet(&device, 0), "cuDeviceGet");

    const CUcontext caller_context = tv::current_context();
    const tv::PrimaryContextState state_before =
        tv::primary_context_state(device);
    {
      tv::PrimaryContext retained(device);
      tv::ScopedCurrentContext current(retained.get());
      require(current.previous() == caller_context,
              "validation context guard captured the wrong caller context");
      tv::Stream caller_stream;
      require(caller_stream.opaque() != nullptr,
              "caller stream must be non-default");
      require(caller_stream.context() == retained.get(),
              "caller stream does not belong to the retained PPU context");

      auto first = std::make_unique<flagdnn::Handle>("thead", 0);
      require(first->backend_name() == "thead",
              "public handle did not report backend thead");
      const std::string target(first->target_fingerprint());
      require(target.starts_with("ppu_"),
              "THead target fingerprint does not start with ppu_");
      require(!target.starts_with("sm_"),
              "THead target fingerprint leaked an NVIDIA sm_ identity");
      require_current_context(retained.get(), "first handle creation");

      auto second = std::make_unique<flagdnn::Handle>("thead", 0);
      require(second->target_fingerprint() == target,
              "two handles reported different PPU target fingerprints");
      require_current_context(retained.get(), "second handle creation");

      require_flagdnn_failure(
          [device_count] {
            flagdnn::Handle invalid("thead",
                                    static_cast<std::int32_t>(device_count));
          },
          FLAGDNN_STATUS_INVALID_VALUE,
          "device ordinal is outside the available PPU range");
      require_current_context(retained.get(), "invalid handle rejection");

      first.reset();
      require(tv::primary_context_state(device).active != 0,
              "destroying one of two handles released the primary context");
      require_current_context(retained.get(), "first handle destruction");
      second.reset();
      require(tv::primary_context_state(device).active != 0,
              "handle destruction invalidated the caller-retained context");
      require_current_context(retained.get(), "second handle destruction");
    }
    require_current_context(caller_context, "complete runtime contract");
    require(tv::primary_context_state(device) == state_before,
            "THead handles did not balance primary-context retain/release");

    std::cout << "PASS THead PPU context, stream, and device identity contract\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
