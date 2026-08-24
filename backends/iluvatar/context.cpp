/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "backends/iluvatar/context.hpp"

#include "backends/iluvatar/error.hpp"

#include <iomanip>
#include <sstream>
#include <utility>

namespace flagdnn::iluvatar {

ContextGuard::ContextGuard(CUcontext context) {
  require(context != nullptr, "CoreX context must not be null");
  CUcontext current = nullptr;
  check_corex(cuCtxGetCurrent(&current), "cuCtxGetCurrent");
  if (current == context) {
    return;
  }
  check_corex(cuCtxPushCurrent(context), "cuCtxPushCurrent");
  pushed_ = true;
}

ContextGuard::~ContextGuard() {
  if (pushed_) {
    CUcontext ignored = nullptr;
    (void)cuCtxPopCurrent(&ignored);
  }
}

IluvatarContext::IluvatarContext(std::int32_t device_ordinal) {
  require(device_ordinal >= 0, "device ordinal must be nonnegative");
  check_corex(cuInit(0), "cuInit");
  check_corex(cuDeviceGet(&device_, device_ordinal), "cuDeviceGet");
  check_corex(cuDevicePrimaryCtxRetain(&context_, device_),
              "cuDevicePrimaryCtxRetain");

  try {
    int major = 0;
    int minor = 0;
    check_corex(
        cuDeviceGetAttribute(
            &major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device_),
        "cuDeviceGetAttribute(compute capability major)");
    check_corex(
        cuDeviceGetAttribute(
            &minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device_),
        "cuDeviceGetAttribute(compute capability minor)");
    require(major == 7 && minor == 1,
            "Iluvatar backend requires qualified CoreX architecture 7.1",
            FLAGDNN_BACKEND_RESULT_NOT_SUPPORTED);
    target_fingerprint_ = "corex_71";

    CUuuid uuid{};
    check_corex(cuDeviceGetUuid(&uuid, device_), "cuDeviceGetUuid");
    int driver_version = 0;
    check_corex(cuDriverGetVersion(&driver_version), "cuDriverGetVersion");
    require(driver_version > 0, "CoreX Driver returned an invalid version",
            FLAGDNN_BACKEND_RESULT_RUNTIME_ERROR);

    std::ostringstream identity;
    identity << target_fingerprint_ << "-driver" << driver_version << '-';
    for (const char byte : uuid.bytes) {
      identity << std::hex << std::setfill('0') << std::setw(2)
               << static_cast<unsigned int>(static_cast<unsigned char>(byte));
    }
    device_identity_ = identity.str();
  } catch (...) {
    (void)cuDevicePrimaryCtxRelease(device_);
    context_ = nullptr;
    throw;
  }
}

IluvatarContext::~IluvatarContext() {
  if (context_ != nullptr) {
    (void)cuDevicePrimaryCtxRelease(device_);
  }
}

const std::string &IluvatarContext::target_fingerprint() const noexcept {
  return target_fingerprint_;
}

EngineBuildContext IluvatarContext::engine_build_context() const {
  return {device_, context_, target_fingerprint_, device_identity_};
}

} // namespace flagdnn::iluvatar
