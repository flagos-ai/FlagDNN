/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "backends/mthreads/context.hpp"

#include "backends/backend_api.h"
#include "backends/mthreads/error.hpp"

#include <musa_runtime_api.h>

#include <iomanip>
#include <iostream>
#include <sstream>

namespace flagdnn::mthreads {
namespace {

void report_guard_failure(
    MUresult status, const char* operation) noexcept {
  if (status == MUSA_SUCCESS) {
    return;
  }
  try {
    std::cerr << mu_error(status, operation) << '\n';
  } catch (...) {
  }
}

}  // namespace

ContextGuard::ContextGuard(MUcontext context, MUdevice device)
    : requested_(context) {
  require(context != nullptr, "requested MUSA context is null");
  check_mu(muCtxGetCurrent(&saved_), "muCtxGetCurrent");
  if (saved_ != requested_) {
    check_mu(muCtxPushCurrent(requested_), "muCtxPushCurrent");
    pushed_ = true;
  }
  try {
    MUdevice current_device = 0;
    check_mu(muCtxGetDevice(&current_device), "muCtxGetDevice");
    require(
        current_device == device,
        "current MUSA context belongs to an unexpected device",
        FLAGDNN_BACKEND_RESULT_RUNTIME_ERROR);
  } catch (...) {
    if (pushed_) {
      MUcontext ignored = nullptr;
      static_cast<void>(muCtxPopCurrent(&ignored));
      pushed_ = false;
    }
    throw;
  }
}

ContextGuard::~ContextGuard() {
  if (!pushed_) {
    return;
  }
  MUcontext popped = nullptr;
  const MUresult pop_status = muCtxPopCurrent(&popped);
  report_guard_failure(pop_status, "muCtxPopCurrent");
  if (pop_status == MUSA_SUCCESS && popped != requested_) {
    std::cerr << "muCtxPopCurrent restored an unexpected MUSA context\n";
  }
  MUcontext current = nullptr;
  const MUresult current_status = muCtxGetCurrent(&current);
  report_guard_failure(current_status, "muCtxGetCurrent(after pop)");
  if (current_status == MUSA_SUCCESS && current != saved_) {
    std::cerr << "MUSA context guard did not restore the saved context\n";
  }
}

MthreadsContext::MthreadsContext(std::int32_t device_ordinal) {
  require(device_ordinal >= 0, "device ordinal must be nonnegative");
  check_mu(muInit(0), "muInit");
  check_mu(muDeviceGet(&device_, device_ordinal), "muDeviceGet");
  check_mu(
      muDevicePrimaryCtxRetain(&context_, device_),
      "muDevicePrimaryCtxRetain");
  try {
    int major = 0;
    int minor = 0;
    int warp_size = 0;
    check_mu(
        muDeviceGetAttribute(
            &major,
            MU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            device_),
        "muDeviceGetAttribute(compute capability major)");
    check_mu(
        muDeviceGetAttribute(
            &minor,
            MU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
            device_),
        "muDeviceGetAttribute(compute capability minor)");
    check_mu(
        muDeviceGetAttribute(
            &warp_size, MU_DEVICE_ATTRIBUTE_WARP_SIZE, device_),
        "muDeviceGetAttribute(warp size)");
    require(
        major > 0 && minor >= 0 && warp_size > 0,
        "MUSA target attributes are invalid",
        FLAGDNN_BACKEND_RESULT_RUNTIME_ERROR);

    target_fingerprint_ =
        "musa-mtgpu-cc" + std::to_string(major) +
        std::to_string(minor) + "-w" + std::to_string(warp_size);
    require(
        target_fingerprint_.size() + 1 <=
            FLAGDNN_BACKEND_MAX_TARGET_FINGERPRINT,
        "MUSA target fingerprint exceeds the backend ABI limit",
        FLAGDNN_BACKEND_RESULT_INTERNAL_ERROR);

    MUuuid uuid{};
    check_mu(muDeviceGetUuid(&uuid, device_), "muDeviceGetUuid");
    int driver_version = 0;
    int runtime_version = 0;
    check_mu(muDriverGetVersion(&driver_version), "muDriverGetVersion");
    check_musa(
        musaRuntimeGetVersion(&runtime_version),
        "musaRuntimeGetVersion");

    std::ostringstream identity;
    identity << target_fingerprint_ << "-driver" << driver_version
             << "-runtime" << runtime_version << "-uuid";
    for (const char byte : uuid.bytes) {
      identity << std::hex << std::setfill('0') << std::setw(2)
               << static_cast<unsigned>(
                      static_cast<unsigned char>(byte));
    }
    device_identity_ = identity.str();

    ContextGuard guard(context_, device_);
  } catch (...) {
    static_cast<void>(muDevicePrimaryCtxRelease(device_));
    context_ = nullptr;
    throw;
  }
}

MthreadsContext::~MthreadsContext() {
  if (context_ != nullptr) {
    report_guard_failure(
        muDevicePrimaryCtxRelease(device_),
        "muDevicePrimaryCtxRelease");
  }
}

const std::string& MthreadsContext::target_fingerprint() const noexcept {
  return target_fingerprint_;
}

EngineBuildContext MthreadsContext::engine_build_context() const {
  return {
      device_, context_, target_fingerprint_, device_identity_};
}

}  // namespace flagdnn::mthreads
