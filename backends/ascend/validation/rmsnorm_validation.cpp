/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "validation/rmsnorm_validation.hpp"

#include "validation/tensor_io.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace flagdnn::validation::ascend {
namespace {

namespace tensor_io = flagdnn::validation::ascend::tensor_io;
using benchmarking::BenchmarkCase;
using benchmarking::TensorSpec;
using testing::RmsnormTestCase;
using testing::TestTensor;

constexpr std::array<flagdnnDataType_t, 3> kDataTypes = {
    FLAGDNN_DATA_FLOAT32,
    FLAGDNN_DATA_FLOAT16,
    FLAGDNN_DATA_BFLOAT16,
};

bool is_supported_data_type(flagdnnDataType_t data_type) {
  return std::find(kDataTypes.begin(), kDataTypes.end(), data_type) !=
         kDataTypes.end();
}

std::vector<std::int64_t> contiguous_strides(
    const std::vector<std::int64_t>& dimensions) {
  std::vector<std::int64_t> result(dimensions.size());
  std::int64_t stride = 1;
  for (std::size_t axis = dimensions.size(); axis != 0; --axis) {
    const std::int64_t dimension = dimensions[axis - 1];
    if (dimension <= 0 ||
        stride > std::numeric_limits<std::int64_t>::max() / dimension) {
      throw std::invalid_argument(
          "RMSNorm shape overflows contiguous strides");
    }
    result[axis - 1] = stride;
    stride *= dimension;
  }
  return result;
}

bool is_contiguous(const TestTensor& tensor) {
  return tensor.strides == contiguous_strides(tensor.dimensions);
}

void validate_tensor(const TestTensor& tensor,
                     const char* name,
                     bool require_contiguous = true) {
  tensor_io::validate_layout(tensor);
  if (require_contiguous && !is_contiguous(tensor)) {
    throw std::invalid_argument(std::string("RMSNorm ") + name +
                                " must be contiguous");
  }
  if (tensor.binding_byte_offset % 32U != 0U) {
    throw std::invalid_argument(std::string("RMSNorm ") + name +
                                " binding offset must be 32-byte aligned");
  }
}

std::size_t checked_product(std::span<const std::int64_t> dimensions,
                            const char* description) {
  std::size_t result = 1;
  for (const std::int64_t dimension : dimensions) {
    if (dimension <= 0) {
      throw std::invalid_argument(std::string(description) +
                                  " contains a non-positive dimension");
    }
    result = tensor_io::checked_multiply(
        result, static_cast<std::size_t>(dimension), description);
  }
  return result;
}

void validate_plan(const RmsnormPlan& plan) {
  const std::size_t rank = plan.input.dimensions.size();
  if (rank < 1 || rank > 8) {
    throw std::invalid_argument("RMSNorm input rank must be in [1, 8]");
  }
  validate_tensor(plan.input, "X");
  validate_tensor(plan.scale, "scale");
  validate_tensor(plan.bias, "bias");
  validate_tensor(plan.output, "Y");
  validate_tensor(plan.inverse_variance, "inverse variance");

  if (!is_supported_data_type(plan.input.data_type) ||
      plan.scale.data_type != plan.input.data_type ||
      plan.bias.data_type != plan.input.data_type ||
      plan.output.data_type != plan.input.data_type) {
    throw std::invalid_argument(
        "RMSNorm X/scale/bias/Y data types must match and be floating");
  }
  if (plan.inverse_variance.data_type != FLAGDNN_DATA_FLOAT32) {
    throw std::invalid_argument(
        "RMSNorm inverse variance must use FP32 storage");
  }
  if (plan.output.dimensions != plan.input.dimensions) {
    throw std::invalid_argument("RMSNorm Y shape must match X");
  }
  if (plan.scale.dimensions != plan.bias.dimensions ||
      plan.scale.dimensions.empty() || plan.scale.dimensions.size() > rank) {
    throw std::invalid_argument(
        "RMSNorm scale/bias shapes must match a non-empty X suffix");
  }

  const std::size_t leading = rank - plan.scale.dimensions.size();
  std::size_t normalized_start = rank;
  std::size_t normalized_elements = 1;
  std::vector<std::int64_t> statistic_dimensions = plan.input.dimensions;
  for (std::size_t axis = 0; axis < rank; ++axis) {
    const std::int64_t parameter_dimension =
        axis < leading ? 1 : plan.scale.dimensions[axis - leading];
    const std::int64_t input_dimension = plan.input.dimensions[axis];
    if (parameter_dimension != 1) {
      if (parameter_dimension != input_dimension) {
        throw std::invalid_argument(
            "RMSNorm scale/bias shape does not match X");
      }
      if (normalized_start == rank) {
        normalized_start = axis;
      }
    } else if (normalized_start != rank && input_dimension != 1) {
      throw std::invalid_argument(
          "RMSNorm scale/bias must describe a contiguous suffix");
    }
    if (normalized_start != rank) {
      normalized_elements = tensor_io::checked_multiply(
          normalized_elements,
          static_cast<std::size_t>(input_dimension),
          "RMSNorm normalized element count");
      statistic_dimensions[axis] = 1;
    }
  }
  if (normalized_start == rank ||
      checked_product(plan.scale.dimensions, "RMSNorm scale shape") !=
          normalized_elements ||
      checked_product(plan.bias.dimensions, "RMSNorm bias shape") !=
          normalized_elements) {
    throw std::invalid_argument("RMSNorm scale/bias element count is invalid");
  }

  const std::size_t input_elements = tensor_io::element_count(plan.input);
  const std::size_t rows = input_elements / normalized_elements;
  if (plan.inverse_variance.dimensions != statistic_dimensions ||
      tensor_io::element_count(plan.inverse_variance) != rows) {
    throw std::invalid_argument(
        "RMSNorm inverse variance shape is invalid");
  }
  if (rows > static_cast<std::size_t>(
                 std::numeric_limits<std::int64_t>::max()) ||
      normalized_elements > static_cast<std::size_t>(
                                std::numeric_limits<std::int64_t>::max()) ||
      plan.rows != static_cast<std::int64_t>(rows) ||
      plan.normalized_elements !=
          static_cast<std::int64_t>(normalized_elements)) {
    throw std::invalid_argument("RMSNorm plan decomposition is invalid");
  }
  if (!std::isfinite(plan.epsilon) || plan.epsilon <= 0.0 ||
      plan.epsilon > static_cast<double>(std::numeric_limits<float>::max())) {
    throw std::invalid_argument(
        "RMSNorm epsilon must be positive finite float32");
  }

  const std::array<std::int64_t, 5> uids = {
      plan.input.uid,
      plan.scale.uid,
      plan.bias.uid,
      plan.output.uid,
      plan.inverse_variance.uid,
  };
  if (std::any_of(uids.begin(), uids.end(), [](std::int64_t uid) {
        return uid <= 0;
      })) {
    throw std::invalid_argument("RMSNorm binding UID must be positive");
  }
  for (std::size_t left = 0; left < uids.size(); ++left) {
    for (std::size_t right = left + 1; right < uids.size(); ++right) {
      if (uids[left] == uids[right]) {
        throw std::invalid_argument("RMSNorm binding UID is duplicated");
      }
    }
  }
}

TestTensor convert(const TensorSpec& tensor) {
  return {tensor.uid,
          tensor.data_type,
          tensor.dimensions,
          tensor.strides,
          tensor.binding_byte_offset};
}

RmsnormPlan make_plan(const TestTensor& input,
                      const TestTensor& scale,
                      const TestTensor& bias,
                      const TestTensor& output,
                      const TestTensor& inverse_variance,
                      double epsilon) {
  RmsnormPlan result = {
      input,
      scale,
      bias,
      output,
      inverse_variance,
      0,
      0,
      epsilon,
  };

  const std::size_t rank = input.dimensions.size();
  if (rank >= 1 && !scale.dimensions.empty() &&
      scale.dimensions.size() <= rank) {
    const std::size_t leading = rank - scale.dimensions.size();
    std::size_t normalized_start = rank;
    std::size_t normalized_elements = 1;
    for (std::size_t axis = 0; axis < rank; ++axis) {
      const std::int64_t parameter_dimension =
          axis < leading ? 1 : scale.dimensions[axis - leading];
      if (parameter_dimension != 1 && normalized_start == rank) {
        normalized_start = axis;
      }
      if (normalized_start != rank && input.dimensions[axis] > 0) {
        normalized_elements = tensor_io::checked_multiply(
            normalized_elements,
            static_cast<std::size_t>(input.dimensions[axis]),
            "RMSNorm normalized element count");
      }
    }
    if (normalized_start != rank && normalized_elements != 0) {
      const std::size_t input_elements =
          checked_product(input.dimensions, "RMSNorm input shape");
      const std::size_t rows = input_elements / normalized_elements;
      if (rows <= static_cast<std::size_t>(
                      std::numeric_limits<std::int64_t>::max()) &&
          normalized_elements <= static_cast<std::size_t>(
                                     std::numeric_limits<std::int64_t>::max())) {
        result.rows = static_cast<std::int64_t>(rows);
        result.normalized_elements =
            static_cast<std::int64_t>(normalized_elements);
      }
    }
  }
  validate_plan(result);
  return result;
}

}  // namespace

RmsnormPlan plan_rmsnorm(const RmsnormTestCase& test_case) {
  return make_plan(test_case.x,
                   test_case.scale,
                   test_case.bias,
                   test_case.y,
                   test_case.inv_variance,
                   test_case.epsilon);
}

RmsnormPlan plan_rmsnorm(const BenchmarkCase& test_case) {
  if (test_case.operation != benchmarking::Operation::kRmsnorm ||
      test_case.output_count != 2 || test_case.tensors.size() != 5) {
    throw std::invalid_argument("RMSNorm benchmark case schema is invalid");
  }
  return make_plan(convert(test_case.tensors[0]),
                   convert(test_case.tensors[1]),
                   convert(test_case.tensors[2]),
                   convert(test_case.tensors[3]),
                   convert(test_case.tensors[4]),
                   test_case.normalization.epsilon);
}

std::vector<RmsnormTestCase> make_ascend_rmsnorm_cases(
    std::span<const RmsnormTestCase> common_cases) {
  if (common_cases.empty()) {
    throw std::invalid_argument(
        "common RMSNorm functional catalog must contain cases");
  }
  std::vector<RmsnormTestCase> result(common_cases.begin(), common_cases.end());
  for (const RmsnormTestCase& test_case : result) {
    (void)plan_rmsnorm(test_case);
  }
  return result;
}

std::vector<BenchmarkCase> make_ascend_rmsnorm_benchmark_cases(
    std::span<const BenchmarkCase> common_cases) {
  if (common_cases.empty()) {
    throw std::invalid_argument(
        "common RMSNorm benchmark catalog must contain cases");
  }
  std::vector<BenchmarkCase> result(common_cases.begin(), common_cases.end());
  for (const BenchmarkCase& test_case : result) {
    (void)plan_rmsnorm(test_case);
  }
  return result;
}

void rmsnorm_host_oracle(const RmsnormPlan& plan,
                         std::span<const float> input_storage,
                         std::span<const float> scale,
                         std::span<const float> bias,
                         std::span<float> output_storage,
                         std::span<float> inverse_variance_storage) {
  validate_plan(plan);
  const std::size_t rows = static_cast<std::size_t>(plan.rows);
  const std::size_t normalized =
      static_cast<std::size_t>(plan.normalized_elements);
  if (input_storage.size() < tensor_io::storage_element_count(plan.input) ||
      scale.size() != tensor_io::storage_element_count(plan.scale) ||
      bias.size() != tensor_io::storage_element_count(plan.bias) ||
      output_storage.size() < tensor_io::storage_element_count(plan.output) ||
      inverse_variance_storage.size() <
          tensor_io::storage_element_count(plan.inverse_variance)) {
    throw std::invalid_argument("RMSNorm host oracle storage size is invalid");
  }

  for (std::size_t row = 0; row < rows; ++row) {
    double square_sum = 0.0;
    for (std::size_t column = 0; column < normalized; ++column) {
      const std::size_t logical = row * normalized + column;
      const float value = input_storage[
          tensor_io::physical_offset_unchecked(logical, plan.input)];
      if (!std::isfinite(value)) {
        throw std::invalid_argument("RMSNorm input contains a non-finite value");
      }
      square_sum += static_cast<double>(value) * value;
    }
    const double inverse =
        1.0 / std::sqrt(square_sum / static_cast<double>(normalized) +
                        plan.epsilon);
    if (!std::isfinite(inverse)) {
      throw std::runtime_error("RMSNorm inverse variance is non-finite");
    }
    const std::size_t statistic_offset =
        tensor_io::physical_offset_unchecked(row, plan.inverse_variance);
    inverse_variance_storage[statistic_offset] = static_cast<float>(inverse);

    for (std::size_t column = 0; column < normalized; ++column) {
      const std::size_t logical = row * normalized + column;
      const std::size_t input_offset =
          tensor_io::physical_offset_unchecked(logical, plan.input);
      const std::size_t scale_offset =
          tensor_io::physical_offset_unchecked(column, plan.scale);
      const std::size_t bias_offset =
          tensor_io::physical_offset_unchecked(column, plan.bias);
      const float scale_value = scale[scale_offset];
      const float bias_value = bias[bias_offset];
      if (!std::isfinite(scale_value) || !std::isfinite(bias_value)) {
        throw std::invalid_argument(
            "RMSNorm scale/bias contains a non-finite value");
      }
      const double value =
          static_cast<double>(input_storage[input_offset]) * inverse *
              scale_value +
          bias_value;
      if (!std::isfinite(value)) {
        throw std::runtime_error("RMSNorm output is non-finite");
      }
      output_storage[tensor_io::physical_offset_unchecked(logical,
                                                          plan.output)] =
          static_cast<float>(value);
    }
  }
}

}  // namespace flagdnn::validation::ascend
