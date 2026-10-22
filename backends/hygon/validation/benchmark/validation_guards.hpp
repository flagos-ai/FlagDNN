/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_HYGON_VALIDATION_BENCHMARK_VALIDATION_GUARDS_HPP_
#define FLAGDNN_BACKENDS_HYGON_VALIDATION_BENCHMARK_VALIDATION_GUARDS_HPP_

#include "tensor_io.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <ostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace flagdnn::benchmarking {

// Execution success is independent of the reported performance evaluation.
// Explicit --min-speedup enforcement belongs to the Python CLI; numerical,
// launch, timing and baseline-evidence errors still throw before this point.
inline int finish_benchmark_execution(std::string_view suite,
                                      std::size_t matched, std::size_t executed,
                                      std::size_t skipped,
                                      std::ostream &output) {
  if (matched == 0)
    throw std::invalid_argument(
        "FLAGDNN_BENCHMARK_CASE did not match any case");
  if (executed > matched || skipped != matched - executed)
    throw std::invalid_argument(
        "benchmark execution accounting is inconsistent");
  output << suite << ": " << (executed == 0 ? "SKIP" : "PASS")
         << " cases=" << matched << " executed=" << executed
         << " skipped=" << skipped << std::endl;
  return executed == 0 ? 77 : 0;
}

// This path intentionally needs neither a compiler nor a GPU. Invalid filters
// must produce an ordinary failed test, not escape the executable as SIGABRT.
inline int skip_unreferenced_benchmark_cases(
    int argc, std::string_view operation,
    std::span<const std::string_view> case_names, std::string_view filter,
    std::string_view reason, std::ostream &output, std::ostream &error) {
  if (argc != 3)
    return 2;
  std::string marker(operation);
  std::transform(marker.begin(), marker.end(), marker.begin(),
                 [](unsigned char c) { return std::toupper(c); });
  const std::string suite = "FLAGDNN_" + marker + "_BENCHMARK";
  std::size_t matched = 0;
  for (const auto name : case_names) {
    if (!filter.empty() && name != filter)
      continue;
    ++matched;
    output << "[SKIP][nvidia_baseline] op=" << operation << " case=" << name
           << " reason=" << reason << '\n';
  }
  if (matched == 0) {
    error << suite
          << "_FAILED: FLAGDNN_BENCHMARK_CASE did not match any native boolean "
             "case\n";
    return 1;
  }
  output << suite << ": SKIP cases=" << matched
         << " executed=0 skipped=" << matched << '\n';
  return 77;
}

// Match the existing NVIDIA FP8 identity fixture, not a broader distribution
// whose correctness or performance was never recorded in that reference.
inline std::vector<float> fp8_identity_fixture(std::size_t count) {
  std::vector<float> result(count);
  for (std::size_t index = 0; index < count; ++index)
    result[index] = static_cast<float>(1U << (index % 6));
  return result;
}

// Padding is an encoded sentinel: FP8 cannot represent the host value -97
// exactly. Comparing decoded padding with -97 would reject an untouched byte.
template <class Tensor>
void require_encoded_padding_unchanged(std::string_view context,
                                       std::span<const std::uint8_t> actual,
                                       const Tensor &tensor) {
  namespace io = validation::hygon::tensor_io;
  const auto element_size = io::data_type_size(tensor.data_type);
  const auto storage = io::storage_element_count(tensor);
  if (actual.size() != storage * element_size)
    throw std::invalid_argument("encoded padding tensor byte count differs");
  std::vector<bool> occupied(storage, false);
  for (std::size_t index = 0; index < io::element_count(tensor); ++index)
    occupied.at(io::logical_offset(index, tensor)) = true;
  const std::vector<float> sentinel{io::kPaddingSentinel};
  const auto padding = io::encode(sentinel, tensor.data_type);
  for (std::size_t index = 0; index < storage; ++index)
    if (!occupied[index])
      for (std::size_t byte = 0; byte < element_size; ++byte)
        if (actual[index * element_size + byte] != padding[byte])
          throw std::runtime_error(std::string(context) +
                                   " modified encoded output padding at byte " +
                                   std::to_string(index * element_size + byte));
}

// Compare original bytes, never decoded numbers: +0 and -0 (and distinct NaN
// payloads) are numerically indistinguishable but not a bit-exact identity.
// Spans start at each tensor's binding entrance; physical holes are included.
template <class Tensor>
void require_bit_exact_identity(std::string_view context, const Tensor &input,
                                const Tensor &output,
                                std::span<const std::uint8_t> initial_input,
                                std::span<const std::uint8_t> actual_input,
                                std::span<const std::uint8_t> actual_output) {
  namespace io = validation::hygon::tensor_io;
  const auto count = io::element_count(input);
  if (input.data_type != output.data_type || count != io::element_count(output))
    throw std::invalid_argument("bit-exact identity tensor metadata differs");
  const auto element_size = io::data_type_size(input.data_type);
  if (initial_input.size() != io::storage_element_count(input) * element_size ||
      actual_input.size() != initial_input.size() ||
      actual_output.size() != io::storage_element_count(output) * element_size)
    throw std::invalid_argument("bit-exact identity tensor byte count differs");
  for (std::size_t byte = 0; byte < initial_input.size(); ++byte)
    if (actual_input[byte] != initial_input[byte])
      throw std::runtime_error(std::string(context) +
                               " modified identity input at byte " +
                               std::to_string(byte));
  for (std::size_t index = 0; index < count; ++index) {
    const auto source = io::logical_offset(index, input) * element_size;
    const auto destination = io::logical_offset(index, output) * element_size;
    for (std::size_t byte = 0; byte < element_size; ++byte)
      if (actual_output[destination + byte] != initial_input[source + byte])
        throw std::runtime_error(std::string(context) +
                                 " identity output differs bit-exactly at "
                                 "element " +
                                 std::to_string(index) + ", byte " +
                                 std::to_string(byte));
  }
  require_encoded_padding_unchanged(context, actual_output, output);
}

// Poison only logical outputs. Finite values are essential: a valid vendor
// primitive may evaluate beta * old_output even when beta is zero. Every
// selected value must survive the actual dtype codec and fail the same
// absolute/relative comparison as runner.cpp; an unavailable poison is an
// error, never permission to weaken the comparison or reuse stale output.
inline std::vector<float> capture_output_poison(std::span<const float> expected,
                                                flagdnnDataType_t type,
                                                double absolute_tolerance,
                                                double relative_tolerance) {
  namespace io = validation::hygon::tensor_io;
  if (!std::isfinite(absolute_tolerance) || absolute_tolerance < 0 ||
      !std::isfinite(relative_tolerance) || relative_tolerance < 0)
    throw std::invalid_argument(
        "capture poison tolerances must be finite and nonnegative");

  std::vector<float> candidates;
  float maximum = 0;
  switch (type) {
  case FLAGDNN_DATA_BOOLEAN:
    // The reserved 0xA5 byte decodes as -97, not true.
    candidates = {io::kPaddingSentinel};
    break;
  case FLAGDNN_DATA_INT32:
    candidates = {-1073741824.0F, 1073741824.0F, -2147483648.0F, 2147483520.0F};
    break;
  case FLAGDNN_DATA_FLOAT32:
    maximum = std::numeric_limits<float>::max();
    break;
  case FLAGDNN_DATA_FLOAT16:
    maximum = 65504.0F;
    break;
  case FLAGDNN_DATA_BFLOAT16:
    maximum = 0x1.fep127F;
    break;
  case FLAGDNN_DATA_FP8_E4M3:
    maximum = 448.0F;
    break;
  case FLAGDNN_DATA_FP8_E5M2:
    maximum = 57344.0F;
    break;
  case FLAGDNN_DATA_FP8_E8M0:
    // E8M0 represents only positive powers of two, not zero or negatives.
    candidates = {1.0F, 0x1p-127F, 0x1p127F};
    break;
  default:
    throw std::invalid_argument("unsupported capture poison data type");
  }
  if (maximum != 0)
    candidates = {-1.0F, 1.0F, -maximum, maximum};
  const auto roundtrip = [type](std::span<const float> values) {
    return io::decode(io::encode(values, type), type, values.size());
  };
  // Roundtrip this small reusable pool once, not once per tensor element.
  candidates = roundtrip(candidates);
  const auto rejected = [&](float actual, double reference) {
    const double value = actual;
    // A finite value always fails runner.cpp's comparison with infinity.
    // Keep the poison finite so beta-zero writers do not inherit a NaN.
    if (std::isinf(reference))
      return std::isfinite(value);
    const double absolute = std::abs(value - reference);
    const double relative =
        absolute / std::max({std::abs(value), std::abs(reference), 1.0e-30});
    return std::isfinite(value) && std::isfinite(absolute) &&
           absolute > absolute_tolerance && relative > relative_tolerance;
  };

  std::vector<float> result(expected.size());
  for (std::size_t i = 0; i < expected.size(); ++i) {
    const double reference = expected[i];
    if (std::isnan(reference))
      throw std::invalid_argument(
          "capture poison rejects a NaN reference at element " +
          std::to_string(i));
    if (std::isinf(reference) && type != FLAGDNN_DATA_FLOAT32 &&
        type != FLAGDNN_DATA_FLOAT16 && type != FLAGDNN_DATA_BFLOAT16)
      throw std::invalid_argument(
          "capture reference infinity is not preserved by the output codec "
          "at element " + std::to_string(i));
    const auto acceptable = [&](float value) {
      return rejected(value, reference);
    };
    const auto selected =
        std::find_if(candidates.begin(), candidates.end(), acceptable);
    if (selected != candidates.end()) {
      result[i] = *selected;
      continue;
    }
    if (maximum != 0) {
      // Large relative tolerances can require a magnitude close to the
      // reference. For an opposite-sign candidate of magnitude x >= |ref|,
      // the feasible interval is atol-|ref| < x < |ref|/(rtol-1).
      const double magnitude = std::abs(reference);
      const double lower =
          std::max({magnitude, 1.0e-30, absolute_tolerance - magnitude});
      const double upper =
          relative_tolerance > 1
              ? std::min(double(maximum), magnitude / (relative_tolerance - 1))
              : double(maximum);
      const double middle =
          lower < upper ? lower + (upper - lower) / 2 : magnitude;
      const auto opposite = [&](double value) {
        return static_cast<float>(
            std::copysign(std::min(value, double(maximum)), -reference));
      };
      const std::vector<float> adaptive = roundtrip(std::vector<float>{
          opposite(magnitude), opposite(std::max(magnitude, 1.0e-30)),
          opposite(middle)});
      const auto fallback =
          std::find_if(adaptive.begin(), adaptive.end(), acceptable);
      if (fallback != adaptive.end()) {
        result[i] = *fallback;
        continue;
      }
    }
    throw std::runtime_error("no finite capture poison exceeds the configured "
                             "tolerances at element " +
                             std::to_string(i));
  }
  return result;
}

// All four actions are outside event-timed samples. A stale result left by the
// direct probe or warmup must not validate an empty/partially-writing graph.
template <class Reset, class Replay, class Synchronize, class Validate>
decltype(auto) validate_fresh_capture(Reset &&reset, Replay &&replay,
                                      Synchronize &&synchronize,
                                      Validate &&validate) {
  std::forward<Reset>(reset)();
  std::forward<Replay>(replay)();
  std::forward<Synchronize>(synchronize)();
  return std::forward<Validate>(validate)();
}

} // namespace flagdnn::benchmarking

#endif // FLAGDNN_BACKENDS_HYGON_VALIDATION_BENCHMARK_VALIDATION_GUARDS_HPP_
