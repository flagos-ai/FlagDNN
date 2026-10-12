/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#ifndef FLAGDNN_TOOLS_NVIDIA_CORE_VALIDATION_HPP_
#define FLAGDNN_TOOLS_NVIDIA_CORE_VALIDATION_HPP_

#include "../../benchmark/common/case.hpp"
#include "../../backends/nvidia/validation/tensor_io.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace standalone_cudnn {
struct ValidationAccuracy {
  double maximum_absolute = 0.0;
  double maximum_relative = 0.0;
};

// Expected values have already been rounded to their output data type. Their
// sign reversal (or +/-1 near zero) is finite and representable in that type.
// Avoid NaN here: a valid beta=0 implementation may still read the old output.
inline float core_output_poison(float expected, double atol, double rtol) {
  if (!std::isfinite(expected))
    throw std::runtime_error("CPU oracle produced a nonfinite output");
  const float magnitude = std::max(1.0F, std::abs(expected));
  const float poison = std::signbit(expected) ? magnitude : -magnitude;
  const double absolute = std::abs(double(poison) - expected);
  const double relative = absolute / std::max(double(magnitude), std::abs(double(expected)));
  if (absolute <= atol || relative <= rtol)
    throw std::runtime_error("output tolerance cannot distinguish finite poison");
  return poison;
}

// A 256-byte prefix preserves CUDA allocation alignment; the case's declared
// binding offset remains relative to the payload allocation after that guard.
inline constexpr std::size_t kCoreGuardBytes = 256;
struct CoreTensorStorage {
  std::int64_t uid = 0;
  bool input = false;
  std::size_t binding_start = 0;
  std::size_t payload_bytes = 0;
  std::vector<std::uint8_t> initial;
  std::vector<std::uint8_t> writable_bits;
};

inline CoreTensorStorage make_core_tensor_storage(
    const flagdnn::benchmarking::TensorSpec& tensor,
    std::span<const std::uint8_t> encoded, bool input,
    std::size_t element_width) {
  namespace io = flagdnn::validation::nvidia::tensor_io;
  CoreTensorStorage storage;
  storage.uid = tensor.uid;
  storage.input = input;
  storage.binding_start = kCoreGuardBytes + tensor.binding_byte_offset;
  storage.payload_bytes = encoded.size();
  storage.initial.assign(storage.binding_start + encoded.size() + kCoreGuardBytes,
                         0xA5);
  std::copy(encoded.begin(), encoded.end(),
            storage.initial.begin() + storage.binding_start);
  if (!input) {
    storage.writable_bits.assign(encoded.size(), 0);
    for (std::size_t index = 0; index < io::element_count(tensor); ++index) {
      const auto offset = io::logical_offset(index, tensor);
      if (tensor.data_type == FLAGDNN_DATA_BOOLEAN) {
        storage.writable_bits.at(offset / 8) |= 1U << (offset % 8);
      } else {
        for (std::size_t byte = 0; byte < element_width; ++byte)
          storage.writable_bits.at(offset * element_width + byte) = 0xFF;
      }
    }
  }
  return storage;
}

inline std::size_t validate_core_storage(
    const CoreTensorStorage& storage, std::span<const std::uint8_t> current) {
  if (current.size() != storage.initial.size())
    throw std::runtime_error("validation readback size mismatch");
  std::size_t checked = 0;
  for (std::size_t byte = 0; byte < current.size(); ++byte) {
    const bool payload = byte >= storage.binding_start &&
        byte - storage.binding_start < storage.payload_bytes;
    const std::uint8_t writable = !storage.input && payload
        ? storage.writable_bits[byte - storage.binding_start] : 0;
    if (writable == 0xFF) continue;
    ++checked;
    if (((current[byte] ^ storage.initial[byte]) & ~writable) != 0) {
      throw std::runtime_error(
          std::string(storage.input ? "input/guard modified" : "output padding/guard modified") +
          " uid=" + std::to_string(storage.uid) + " allocation_byte=" +
          std::to_string(byte));
    }
  }
  return checked;
}

inline void compare_core_values(std::span<const float> actual,
                                std::span<const float> expected,
                                double absolute_tolerance,
                                double relative_tolerance,
                                ValidationAccuracy& accuracy,
                                std::size_t& checked) {
  if (actual.size() != expected.size())
    throw std::runtime_error("CPU oracle output size mismatch");
  for (std::size_t index = 0; index < actual.size(); ++index) {
    const double value = actual[index], reference = expected[index];
    const double absolute = std::abs(value - reference);
    const double relative = absolute /
        std::max({std::abs(value), std::abs(reference), 1.0e-30});
    ++checked;
    accuracy.maximum_absolute = std::max(accuracy.maximum_absolute, absolute);
    accuracy.maximum_relative = std::max(accuracy.maximum_relative, relative);
    if (!std::isfinite(value) || !std::isfinite(reference) ||
        (absolute > absolute_tolerance && relative > relative_tolerance)) {
      std::ostringstream reason;
      reason << "cuDNN output differs from CPU oracle at element " << index
             << ": actual=" << value << ", expected=" << reference
             << ", abs=" << absolute << ", rel=" << relative;
      throw std::runtime_error(reason.str());
    }
  }
}
}  // namespace standalone_cudnn
#endif
