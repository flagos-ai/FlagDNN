// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "backends/thead/context.hpp"

#include "backends/backend_api.h"
#include "backends/thead/error.hpp"

#include <cctype>
#include <iomanip>
#include <sstream>
#include <string>
#include <string_view>

#ifndef FLAGDNN_THEAD_PPU_SDK_VERSION
#define FLAGDNN_THEAD_PPU_SDK_VERSION "unknown"
#endif

namespace flagdnn::thead {
namespace {

std::string normalized_model(std::string_view name) {
  std::string model;
  model.reserve(name.size());
  bool previous_separator = false;
  for (const unsigned char byte : name) {
    if (std::isalnum(byte) != 0) {
      model.push_back(static_cast<char>(std::tolower(byte)));
      previous_separator = false;
    } else if (!model.empty() && !previous_separator) {
      model.push_back('_');
      previous_separator = true;
    }
  }
  while (!model.empty() && model.back() == '_') {
    model.pop_back();
  }
  if (model.empty()) {
    model = "device";
  }
  constexpr std::size_t kMaximumModelLength = 80;
  if (model.size() > kMaximumModelLength) {
    model.resize(kMaximumModelLength);
    while (!model.empty() && model.back() == '_') {
      model.pop_back();
    }
  }
  return model;
}

std::string hexadecimal_uuid(const CUuuid& uuid) {
  std::ostringstream output;
  for (const char byte : uuid.bytes) {
    output << std::hex << std::setfill('0') << std::setw(2)
           << static_cast<unsigned int>(static_cast<unsigned char>(byte));
  }
  return output.str();
}

}  // namespace

ContextGuard::ContextGuard(CUcontext context) {
  CUcontext current = nullptr;
  check_driver(cuCtxGetCurrent(&current), "cuCtxGetCurrent");
  if (current == context) {
    return;
  }
  check_driver(cuCtxPushCurrent(context), "cuCtxPushCurrent");
  pushed_ = true;
}

ContextGuard::~ContextGuard() {
  if (pushed_) {
    CUcontext ignored = nullptr;
    (void)cuCtxPopCurrent(&ignored);
  }
}

TheadContext::TheadContext(std::int32_t device_ordinal) {
  require(device_ordinal >= 0, "device ordinal must be nonnegative");
  check_driver(cuInit(0), "cuInit");
  int device_count = 0;
  check_driver(cuDeviceGetCount(&device_count), "cuDeviceGetCount");
  require(device_ordinal < device_count,
          "device ordinal is outside the available PPU range");
  check_driver(cuDeviceGet(&device_, device_ordinal), "cuDeviceGet");
  check_driver(cuDevicePrimaryCtxRetain(&context_, device_),
               "cuDevicePrimaryCtxRetain");
  try {
    int major = 0;
    int minor = 0;
    check_driver(cuDeviceGetAttribute(
                     &major,
                     CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                     device_),
                 "cuDeviceGetAttribute(compute capability major)");
    check_driver(cuDeviceGetAttribute(
                     &minor,
                     CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                     device_),
                 "cuDeviceGetAttribute(compute capability minor)");
    char name[256]{};
    check_driver(cuDeviceGetName(name, sizeof(name), device_),
                 "cuDeviceGetName");
    target_fingerprint_ = "ppu_" + normalized_model(name) + "_cc" +
                          std::to_string(major) + std::to_string(minor);
    require(target_fingerprint_.size() + 1 <=
                FLAGDNN_BACKEND_MAX_TARGET_FINGERPRINT,
            "THead target fingerprint exceeds the backend ABI limit",
            FLAGDNN_BACKEND_RESULT_INTERNAL_ERROR);

    CUuuid uuid{};
    check_driver(cuDeviceGetUuid(&uuid, device_), "cuDeviceGetUuid");
    char pci_bus_id[64]{};
    check_driver(cuDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), device_),
                 "cuDeviceGetPCIBusId");
    int driver_version = 0;
    check_driver(cuDriverGetVersion(&driver_version), "cuDriverGetVersion");
    std::ostringstream identity;
    identity << target_fingerprint_ << "-sdk"
             << FLAGDNN_THEAD_PPU_SDK_VERSION << "-driver"
             << driver_version << "-pci" << pci_bus_id << "-uuid"
             << hexadecimal_uuid(uuid);
    device_identity_ = identity.str();
  } catch (...) {
    (void)cuDevicePrimaryCtxRelease(device_);
    context_ = nullptr;
    throw;
  }
}

TheadContext::~TheadContext() {
  if (context_ != nullptr) {
    (void)cuDevicePrimaryCtxRelease(device_);
  }
}

const std::string& TheadContext::target_fingerprint() const noexcept {
  return target_fingerprint_;
}

EngineBuildContext TheadContext::engine_build_context() const {
  return {device_, context_, target_fingerprint_, device_identity_};
}

}  // namespace flagdnn::thead
