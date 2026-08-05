/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "backends/nvidia/context.hpp"

#include "backends/nvidia/error.hpp"

#include <iomanip>
#include <sstream>
#include <utility>

namespace flagdnn::cuda {

ContextGuard::ContextGuard(CUcontext context) {
  CUcontext current = nullptr;
  check_cuda(cuCtxGetCurrent(&current), "cuCtxGetCurrent");
  if (current == context) {
    return;
  }
  check_cuda(cuCtxPushCurrent(context), "cuCtxPushCurrent");
  active_ = true;
}

ContextGuard::~ContextGuard() {
  if (active_) {
    CUcontext ignored = nullptr;
    (void)cuCtxPopCurrent(&ignored);
  }
}

CudaContext::CudaContext(std::int32_t device_ordinal) {
  if (device_ordinal < 0) {
    throw CudaError(FLAGDNN_BACKEND_RESULT_INVALID_VALUE,
                    "device ordinal must be nonnegative");
  }
  check_cuda(cuInit(0), "cuInit");
  check_cuda(cuDeviceGet(&device_, device_ordinal), "cuDeviceGet");
  check_cuda(cuDevicePrimaryCtxRetain(&context_, device_),
             "cuDevicePrimaryCtxRetain");
  try {
    int major = 0;
    int minor = 0;
    check_cuda(cuDeviceGetAttribute(
                   &major,
                   CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                   device_),
               "cuDeviceGetAttribute(compute capability major)");
    check_cuda(cuDeviceGetAttribute(
                   &minor,
                   CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                   device_),
               "cuDeviceGetAttribute(compute capability minor)");
    architecture_ = major * 10 + minor;
    target_fingerprint_ = "sm_" + std::to_string(architecture_);

    CUuuid uuid{};
    check_cuda(cuDeviceGetUuid(&uuid, device_), "cuDeviceGetUuid");
    int driver_version = 0;
    check_cuda(cuDriverGetVersion(&driver_version), "cuDriverGetVersion");
    std::ostringstream device_identity;
    device_identity << target_fingerprint_ << "-driver" << driver_version
                    << '-';
    for (const char byte : uuid.bytes) {
      device_identity
          << std::hex << std::setfill('0') << std::setw(2)
          << static_cast<unsigned int>(static_cast<unsigned char>(byte));
    }
    device_identity_ = device_identity.str();
  } catch (...) {
    (void)cuDevicePrimaryCtxRelease(device_);
    context_ = nullptr;
    throw;
  }
}

CudaContext::~CudaContext() {
  if (context_ != nullptr) {
    (void)cuDevicePrimaryCtxRelease(device_);
  }
}

const std::string& CudaContext::target_fingerprint() const noexcept {
  return target_fingerprint_;
}

EngineBuildContext CudaContext::engine_build_context() const {
  return {device_, context_, target_fingerprint_, device_identity_};
}

}  // namespace flagdnn::cuda
