/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "backends/iluvatar/context.hpp"
#include "backends/iluvatar/error.hpp"

#include <cuda.h>

#include <iostream>
#include <regex>
#include <stdexcept>
#include <string>

namespace {

void expect(bool condition, const char *message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void check_driver(CUresult result, const char *operation) {
  if (result != CUDA_SUCCESS) {
    throw std::runtime_error(std::string(operation) + " failed");
  }
}

} // namespace

int main() {
  try {
    bool rejected_negative_ordinal = false;
    try {
      flagdnn::iluvatar::IluvatarContext invalid(-1);
    } catch (const flagdnn::iluvatar::IluvatarError &error) {
      rejected_negative_ordinal =
          error.result() == FLAGDNN_BACKEND_RESULT_INVALID_VALUE;
    }
    expect(rejected_negative_ordinal,
           "negative device ordinal was not rejected as INVALID_VALUE");

    check_driver(cuInit(0), "cuInit");
    CUcontext original = nullptr;
    check_driver(cuCtxGetCurrent(&original), "cuCtxGetCurrent(original)");

    flagdnn::iluvatar::IluvatarContext context(0);
    expect(context.target_fingerprint() == "corex_71",
           "Iluvatar target fingerprint must be exactly corex_71");
    const auto build_context = context.engine_build_context();
    expect(build_context.context != nullptr,
           "Iluvatar context did not retain a primary context");
    expect(build_context.target_fingerprint == "corex_71",
           "engine build context changed the target fingerprint");
    expect(std::regex_match(build_context.device_identity,
                            std::regex("^corex_71-driver[0-9]+-[0-9a-f]{32}$")),
           "device identity does not match the stable CoreX contract");

    CUcontext retained = nullptr;
    check_driver(cuDevicePrimaryCtxRetain(&retained, build_context.device),
                 "cuDevicePrimaryCtxRetain(test)");
    expect(retained == build_context.context,
           "IluvatarContext did not retain the device primary context");
    check_driver(cuDevicePrimaryCtxRelease(build_context.device),
                 "cuDevicePrimaryCtxRelease(test)");

    check_driver(cuCtxSetCurrent(nullptr), "cuCtxSetCurrent(null)");
    {
      flagdnn::iluvatar::ContextGuard guard(build_context.context);
      CUcontext active = nullptr;
      check_driver(cuCtxGetCurrent(&active), "cuCtxGetCurrent(guarded)");
      expect(active == build_context.context,
             "ContextGuard did not make the retained context current");
    }
    CUcontext restored = build_context.context;
    check_driver(cuCtxGetCurrent(&restored), "cuCtxGetCurrent(restored)");
    expect(restored == nullptr,
           "ContextGuard did not restore the previous null context");
    check_driver(cuCtxSetCurrent(original), "cuCtxSetCurrent(original)");

    std::cout << "PASS Iluvatar CoreX context contract\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
