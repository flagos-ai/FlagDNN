/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_ASCEND_VALIDATION_MOD_CONTRACT_HPP_
#define FLAGDNN_BACKENDS_ASCEND_VALIDATION_MOD_CONTRACT_HPP_

#include "validation/tensor_io.hpp"

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace flagdnn::validation::ascend::mod_contract {

enum class Sentinel : std::size_t {
  kNegativeExactPositiveDivisor,
  kNegativeExactNegativeDivisor,
  kNegativeZero,
  kPositiveZero,
  kMaximumOverHalf,
  kMaximumOverThree,
  kLargeNonzeroOverThree,
  kBelowExactMultiple,
  kAboveExactMultiple,
  kNegativeAboveExactMultiple,
  kNegativeDividendPositiveDivisor,
  kNegativeDividendNegativeDivisor,
  kPositiveDividendNegativeDivisor,
  kPositiveDividendPositiveDivisor,
  kCount,
};

inline constexpr std::size_t kSentinelCount =
    static_cast<std::size_t>(Sentinel::kCount);

[[nodiscard]] inline constexpr std::size_t index(Sentinel sentinel) noexcept {
  return static_cast<std::size_t>(sentinel);
}

struct SemanticInputs {
  std::vector<float> left;
  std::vector<float> right;
};

struct ExactMultipleNeighbors {
  float below = 0.0F;
  float above = 0.0F;
};

[[nodiscard]] inline float maximum_finite(
    flagdnnDataType_t data_type) {
  switch (data_type) {
    case FLAGDNN_DATA_INT32:
      throw std::invalid_argument(
          "INT32 is not supported by this validation adapter");

    case FLAGDNN_DATA_FLOAT32:
      return std::numeric_limits<float>::max();
    case FLAGDNN_DATA_FLOAT16:
      return 65504.0F;
    case FLAGDNN_DATA_BFLOAT16:
      return std::bit_cast<float>(UINT32_C(0x7f7f0000));
    case FLAGDNN_DATA_BOOLEAN:
    case FLAGDNN_DATA_FP8_E8M0:
    case FLAGDNN_DATA_FP8_E4M3:
    case FLAGDNN_DATA_FP8_E5M2:
      break;
  }
  throw std::invalid_argument(
      "Ascend Mod semantic contract data type is unsupported");
}

// FP32 and BF16 maximum finite values are exactly divisible by three.  Use
// their immediately preceding representable value to retain a non-zero
// remainder at the same exponent width.  FP16 maximum finite already has
// remainder two.
[[nodiscard]] inline float large_nonzero_remainder_value(
    flagdnnDataType_t data_type) {
  switch (data_type) {
    case FLAGDNN_DATA_INT32:
      throw std::invalid_argument(
          "INT32 is not supported by this validation adapter");

    case FLAGDNN_DATA_FLOAT32:
      return std::bit_cast<float>(UINT32_C(0x7f7ffffe));
    case FLAGDNN_DATA_FLOAT16:
      return 65504.0F;
    case FLAGDNN_DATA_BFLOAT16:
      return std::bit_cast<float>(UINT32_C(0x7f7e0000));
    case FLAGDNN_DATA_BOOLEAN:
    case FLAGDNN_DATA_FP8_E8M0:
    case FLAGDNN_DATA_FP8_E4M3:
    case FLAGDNN_DATA_FP8_E5M2:
      break;
  }
  throw std::invalid_argument(
      "Ascend Mod semantic contract data type is unsupported");
}

[[nodiscard]] inline ExactMultipleNeighbors exact_multiple_neighbors(
    flagdnnDataType_t data_type) {
  switch (data_type) {
    case FLAGDNN_DATA_INT32:
      throw std::invalid_argument(
          "INT32 is not supported by this validation adapter");

    case FLAGDNN_DATA_FLOAT32:
      return {std::nextafter(6.0F, 0.0F),
              std::nextafter(6.0F,
                             std::numeric_limits<float>::infinity())};
    case FLAGDNN_DATA_FLOAT16:
      return {5.99609375F, 6.00390625F};
    case FLAGDNN_DATA_BFLOAT16:
      return {5.96875F, 6.03125F};
    case FLAGDNN_DATA_BOOLEAN:
    case FLAGDNN_DATA_FP8_E8M0:
    case FLAGDNN_DATA_FP8_E4M3:
    case FLAGDNN_DATA_FP8_E5M2:
      break;
  }
  throw std::invalid_argument(
      "Ascend Mod semantic contract data type is unsupported");
}

[[nodiscard]] inline SemanticInputs semantic_inputs(
    flagdnnDataType_t data_type) {
  const float maximum = maximum_finite(data_type);
  const float large_nonzero = large_nonzero_remainder_value(data_type);
  const ExactMultipleNeighbors neighbors =
      exact_multiple_neighbors(data_type);
  std::vector<float> left = {
      -6.0F,
      -6.0F,
      -0.0F,
      0.0F,
      maximum,
      maximum,
      large_nonzero,
      neighbors.below,
      neighbors.above,
      -neighbors.above,
      -7.0F,
      -7.0F,
      7.0F,
      7.0F,
  };
  std::vector<float> right = {
      3.0F,
      -3.0F,
      2.0F,
      -2.0F,
      0.5F,
      3.0F,
      3.0F,
      3.0F,
      3.0F,
      3.0F,
      3.0F,
      -3.0F,
      -3.0F,
      3.0F,
  };
  SemanticInputs result{
      tensor_io::quantize(left, data_type),
      tensor_io::quantize(right, data_type),
  };
  if (result.left.size() != kSentinelCount ||
      result.right.size() != kSentinelCount) {
    throw std::logic_error("Ascend Mod semantic input count is invalid");
  }
  for (std::size_t offset = 0; offset < kSentinelCount; ++offset) {
    if (!std::isfinite(result.left[offset]) ||
        !std::isfinite(result.right[offset]) ||
        result.right[offset] == 0.0F) {
      throw std::logic_error("Ascend Mod semantic input is unsafe");
    }
  }
  return result;
}

[[nodiscard]] inline std::vector<float> host_reference(
    const SemanticInputs& inputs,
    flagdnnDataType_t data_type) {
  if (inputs.left.size() != kSentinelCount ||
      inputs.right.size() != kSentinelCount) {
    throw std::invalid_argument(
        "Ascend Mod semantic host input count is invalid");
  }
  std::vector<float> result(kSentinelCount);
  for (std::size_t offset = 0; offset < kSentinelCount; ++offset) {
    if (!std::isfinite(inputs.left[offset]) ||
        !std::isfinite(inputs.right[offset]) ||
        inputs.right[offset] == 0.0F) {
      throw std::invalid_argument(
          "Ascend Mod semantic host input is unsafe");
    }
    result[offset] = std::fmod(inputs.left[offset], inputs.right[offset]);
    if (!std::isfinite(result[offset])) {
      throw std::runtime_error(
          "Ascend Mod semantic host result is non-finite");
    }
  }
  return tensor_io::quantize(result, data_type);
}

[[nodiscard]] inline bool strict_zero_matches(float actual,
                                              float reference) noexcept {
  return reference != 0.0F ||
         (actual == 0.0F &&
          std::signbit(actual) == std::signbit(reference));
}

// The explicit semantic cases are a strict std::fmod contract.  Comparing the
// decoded floats by representation makes the signed-zero requirement
// unambiguous and also keeps every non-zero result exact.
[[nodiscard]] inline bool strict_ieee_matches(float actual,
                                              float reference) noexcept {
  return std::bit_cast<std::uint32_t>(actual) ==
         std::bit_cast<std::uint32_t>(reference);
}

struct AclnnReferenceObservation {
  std::size_t matched = 0;
  std::size_t differed = 0;
  std::size_t nonfinite = 0;

  [[nodiscard]] constexpr std::size_t total() const noexcept {
    return matched + differed + nonfinite;
  }
};

// aclnnFmodTensor in the pinned CANN 9 runtime is still required to prepare
// and execute for the semantic cases, but it is not a full std::fmod oracle:
// wide exponent ratios can overflow and negative exact multiples can lose the
// zero sign.  Classify its output explicitly instead of silently skipping it.
// A future ACLNN implementation that matches the host oracle naturally moves
// every element into `matched` and remains valid.
[[nodiscard]] inline AclnnReferenceObservation classify_aclnn_reference(
    std::span<const float> observed,
    std::span<const float> host) {
  if (observed.size() != host.size()) {
    throw std::invalid_argument(
        "Ascend Mod ACLNN observation size is invalid");
  }
  AclnnReferenceObservation result;
  for (std::size_t offset = 0; offset < host.size(); ++offset) {
    if (!std::isfinite(host[offset])) {
      throw std::invalid_argument(
          "Ascend Mod semantic host observation is non-finite");
    }
    if (!std::isfinite(observed[offset])) {
      ++result.nonfinite;
    } else if (strict_ieee_matches(observed[offset], host[offset])) {
      ++result.matched;
    } else {
      ++result.differed;
    }
  }
  return result;
}

}  // namespace flagdnn::validation::ascend::mod_contract

#endif  // FLAGDNN_BACKENDS_ASCEND_VALIDATION_MOD_CONTRACT_HPP_
