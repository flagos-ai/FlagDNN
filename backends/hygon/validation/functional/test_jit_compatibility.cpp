/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "backends/hygon/engines/jit_compatibility.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

using flagdnn::hygon::HygonError;
using flagdnn::hygon::detail::is_jit_candidate_compatibility_error;
using flagdnn::hygon::detail::validate_jit_kernel_resources;

void expect(bool condition, const char *description) {
  if (!condition) {
    throw std::runtime_error(description);
  }
}

class DerivedRuntimeError final : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

void test_candidate_errors() {
  constexpr std::array accepted = {
      hipErrorLaunchOutOfResources, hipErrorInvalidConfiguration,
      hipErrorInvalidDeviceFunction, hipErrorInvalidImage,
      hipErrorNoBinaryForGpu, hipErrorInvalidKernelFile};
  for (const hipError_t result : accepted) {
    const char *raw_message = hipGetErrorString(result);
    expect(raw_message != nullptr && raw_message[0] != '\0',
           "HIP did not provide a candidate error description");
    const std::string message(raw_message);
    expect(is_jit_candidate_compatibility_error(HygonError(
               FLAGDNN_BACKEND_RESULT_RUNTIME_ERROR, "typed HIP failure",
               result)),
           "typed candidate HIP result was rejected");
    expect(is_jit_candidate_compatibility_error(std::runtime_error(message)),
           "upstream raw HIP error was rejected");
    expect(is_jit_candidate_compatibility_error(
               std::runtime_error("HCU kernel launch failed: " + message)),
           "upstream HCU launch error was rejected");

    for (const std::string &unrelated :
         {"compiler failed: " + message, message + " (metadata)",
          " " + message, message + "\n",
          "HCU kernel launch failed: HCU kernel launch failed: " + message}) {
      expect(!is_jit_candidate_compatibility_error(
                 std::runtime_error(unrelated)),
             "partial or malformed HIP message was accepted");
    }
    expect(!is_jit_candidate_compatibility_error(DerivedRuntimeError(message)),
           "derived runtime_error was accepted by its message");
    expect(!is_jit_candidate_compatibility_error(std::logic_error(message)),
           "logic_error was accepted by its message");
    expect(!is_jit_candidate_compatibility_error(HygonError(
               FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED, message)),
           "untyped Hygon error was accepted by its message");
    expect(!is_jit_candidate_compatibility_error(HygonError(
               FLAGDNN_BACKEND_RESULT_RUNTIME_ERROR, message,
               hipErrorIllegalAddress)),
           "fatal typed HIP result was accepted by its message");
  }

  constexpr std::array rejected = {
      hipSuccess, hipErrorOutOfMemory, hipErrorInvalidValue,
      hipErrorIllegalAddress, hipErrorLaunchFailure, hipErrorUnknown};
  for (const hipError_t result : rejected) {
    expect(!is_jit_candidate_compatibility_error(HygonError(
               FLAGDNN_BACKEND_RESULT_RUNTIME_ERROR, "fatal HIP failure",
               result)),
           "fatal typed HIP result was accepted");
    const char *message = hipGetErrorString(result);
    if (message != nullptr) {
      expect(!is_jit_candidate_compatibility_error(std::runtime_error(message)),
             "fatal upstream HIP error was accepted");
      expect(!is_jit_candidate_compatibility_error(std::runtime_error(
                 std::string("HCU kernel launch failed: ") + message)),
             "fatal upstream launch error was accepted");
    }
  }

  for (const char *message : {
           "", "HCU kernel launch failed: ", "metadata file is missing",
           "Compute architecture mismatch! Device has gfx936, kernel requires gfx90a",
           "OutOfResources: Requested shared memory (65537 bytes) exceeds GPU's maximum (65536 bytes)"}) {
    expect(!is_jit_candidate_compatibility_error(std::runtime_error(message)),
           "unknown or post-load shared-memory failure was accepted");
  }
}

template <typename Operation>
void expect_resource_error(Operation operation, bool recoverable) {
  try {
    operation();
  } catch (const HygonError &error) {
    expect(error.result() == FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
           "resource validation returned the wrong backend result");
    expect(is_jit_candidate_compatibility_error(error) == recoverable,
           "resource validation returned the wrong recovery decision");
    if (recoverable) {
      expect(error.hip_result() == hipErrorLaunchOutOfResources,
             "resource limit error lost its typed HIP result");
    } else {
      expect(!error.hip_result(), "metadata error acquired a HIP result");
    }
    return;
  }
  throw std::runtime_error("resource validation did not reject bad input");
}

void test_kernel_resources() {
  hipDeviceProp_t properties{};
  std::strcpy(properties.gcnArchName, "gfx936:sramecc+:xnack-");
  properties.sharedMemPerBlock = 65536;
  for (const unsigned int shared : {0U, 1U, 65535U, 65536U}) {
    validate_jit_kernel_resources("gfx936", shared, properties);
  }
  expect_resource_error(
      [&] { validate_jit_kernel_resources("gfx936", 65537, properties); },
      true);
  expect_resource_error(
      [&] {
        validate_jit_kernel_resources(
            "gfx936", std::numeric_limits<unsigned int>::max(), properties);
      },
      true);

  // Metadata failures remain fatal even if that candidate also exceeds LDS.
  for (const std::string &architecture :
       {std::string{}, std::string("gfx90a"), std::string("gfx936:extra"),
        std::string("gfx936\0other", 12)}) {
    expect_resource_error(
        [&] { validate_jit_kernel_resources(architecture, 65537, properties); },
        false);
  }
  std::strcpy(properties.gcnArchName, "gfx936");
  validate_jit_kernel_resources("gfx936", 65536, properties);

  properties.sharedMemPerBlock = 0;
  validate_jit_kernel_resources("gfx936", 0, properties);
  expect_resource_error(
      [&] { validate_jit_kernel_resources("gfx936", 1, properties); }, true);

  properties.gcnArchName[0] = '\0';
  expect_resource_error(
      [&] { validate_jit_kernel_resources("gfx936", 0, properties); }, false);
  std::strcpy(properties.gcnArchName, ":xnack-");
  expect_resource_error(
      [&] { validate_jit_kernel_resources("gfx936", 0, properties); }, false);
  std::fill(std::begin(properties.gcnArchName),
            std::end(properties.gcnArchName), 'x');
  expect_resource_error(
      [&] { validate_jit_kernel_resources("gfx936", 0, properties); }, false);
}

} // namespace

int main() {
  try {
    test_candidate_errors();
    test_kernel_resources();
    std::cout << "Hygon JIT compatibility checks passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "Hygon JIT compatibility check failed: " << error.what()
              << '\n';
    return 1;
  }
}
