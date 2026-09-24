/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_HYGON_ENGINES_JIT_COMPATIBILITY_HPP_
#define FLAGDNN_BACKENDS_HYGON_ENGINES_JIT_COMPATIBILITY_HPP_

#include "backends/hygon/error.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <iterator>
#include <string>
#include <string_view>
#include <typeinfo>

namespace flagdnn::hygon::detail {

inline constexpr std::array<hipError_t, 6> kJitCandidateHipErrors = {
    hipErrorLaunchOutOfResources, hipErrorInvalidConfiguration,
    hipErrorInvalidDeviceFunction, hipErrorInvalidImage,
    hipErrorNoBinaryForGpu, hipErrorInvalidKernelFile};

[[nodiscard]] inline bool
is_jit_candidate_compatibility_error(const std::exception &error) noexcept {
  if (const auto *hygon_error = dynamic_cast<const HygonError *>(&error)) {
    const auto result = hygon_error->hip_result();
    return result && std::find(kJitCandidateHipErrors.begin(),
                               kJitCandidateHipErrors.end(), *result) !=
                         kJitCandidateHipErrors.end();
  }

  // Upstream HCU exposes plain runtime_error for its HIP calls. Match only
  // those complete messages; compiler exceptions and other derived exception
  // types must not turn a broken compilation into a rejected tuning candidate.
  if (typeid(error) != typeid(std::runtime_error)) {
    return false;
  }
  std::string_view message(error.what());
  constexpr std::string_view launch_prefix = "HCU kernel launch failed: ";
  if (message.starts_with(launch_prefix)) {
    message.remove_prefix(launch_prefix.size());
  }
  for (const hipError_t result : kJitCandidateHipErrors) {
    const char *description = hipGetErrorString(result);
    if (description != nullptr && description[0] != '\0' &&
        message == description) {
      return true;
    }
  }
  return false;
}

inline void validate_jit_kernel_resources(const std::string &kernel_arch,
                                          unsigned int shared_memory,
                                          const hipDeviceProp_t &properties) {
  require(!kernel_arch.empty(), "HCU kernel metadata has no architecture",
          FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);

  const auto device_arch_end =
      std::find(std::begin(properties.gcnArchName),
                std::end(properties.gcnArchName), '\0');
  require(device_arch_end != std::end(properties.gcnArchName),
          "HCU device architecture is not terminated",
          FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
  std::string_view device_arch(
      properties.gcnArchName,
      static_cast<std::size_t>(device_arch_end - properties.gcnArchName));
  device_arch = device_arch.substr(0, device_arch.find(':'));
  require(!device_arch.empty(), "HCU device has no architecture",
          FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
  require(kernel_arch == device_arch,
          "HCU kernel architecture does not match the selected device",
          FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);

  // Check before upstream's lazy module load: its shared-memory validation
  // runs after hipModuleLoad and cannot release that module on rejection.
  if (shared_memory > properties.sharedMemPerBlock) {
    throw HygonError(
        FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
        "HCU kernel requests " + std::to_string(shared_memory) +
            " shared-memory bytes, exceeding the device limit of " +
            std::to_string(properties.sharedMemPerBlock),
        hipErrorLaunchOutOfResources);
  }
}

} // namespace flagdnn::hygon::detail

#endif // FLAGDNN_BACKENDS_HYGON_ENGINES_JIT_COMPATIBILITY_HPP_
