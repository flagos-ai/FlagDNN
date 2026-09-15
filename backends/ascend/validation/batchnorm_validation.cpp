/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "validation/batchnorm_validation.hpp"

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

namespace io = flagdnn::validation::ascend::tensor_io;
using benchmarking::BenchmarkCase;
using benchmarking::TensorSpec;
using testing::BatchnormTestCase;
using testing::TestTensor;
using Shape = std::vector<std::int64_t>;

constexpr std::array<flagdnnDataType_t, 3> kDataTypes = {
    FLAGDNN_DATA_FLOAT32,
    FLAGDNN_DATA_FLOAT16,
    FLAGDNN_DATA_BFLOAT16,
};

bool is_supported_data_type(flagdnnDataType_t data_type) {
  return std::find(kDataTypes.begin(), kDataTypes.end(), data_type) !=
         kDataTypes.end();
}

Shape contiguous_strides(const Shape& dimensions) {
  Shape result(dimensions.size());
  std::int64_t stride = 1;
  for (std::size_t axis = dimensions.size(); axis != 0; --axis) {
    const std::int64_t dimension = dimensions[axis - 1];
    if (dimension <= 0 ||
        stride > std::numeric_limits<std::int64_t>::max() / dimension) {
      throw std::invalid_argument(
          "BatchNorm shape overflows contiguous strides");
    }
    result[axis - 1] = stride;
    stride *= dimension;
  }
  return result;
}

std::size_t parameter_elements(const TestTensor& tensor) {
  io::validate_layout(tensor);
  return io::element_count(tensor);
}

void validate_parameter(const TestTensor& tensor,
                        std::size_t channels,
                        flagdnnDataType_t data_type,
                        const char* name) {
  if (tensor.data_type != data_type) {
    throw std::invalid_argument(std::string("BatchNorm ") + name +
                                " data type is invalid");
  }
  if (parameter_elements(tensor) != channels) {
    throw std::invalid_argument(std::string("BatchNorm ") + name +
                                " channel element count is invalid");
  }
  if (tensor.strides != contiguous_strides(tensor.dimensions)) {
    throw std::invalid_argument(std::string("BatchNorm ") + name +
                                " must be contiguous");
  }
  if (tensor.binding_byte_offset % 32U != 0U) {
    throw std::invalid_argument(std::string("BatchNorm ") + name +
                                " binding offset must be 32-byte aligned");
  }
}

void validate_plan(BatchnormPlan& plan) {
  const std::size_t rank = plan.input.dimensions.size();
  if (rank < 2 || rank > 8) {
    throw std::invalid_argument("BatchNorm rank must be in [2, 8]");
  }
  io::validate_layout(plan.input);
  io::validate_layout(plan.output);
  if (!is_supported_data_type(plan.input.data_type) ||
      plan.output.data_type != plan.input.data_type ||
      plan.output.dimensions != plan.input.dimensions) {
    throw std::invalid_argument("BatchNorm X/Y shape or data type is invalid");
  }
  if (plan.input.binding_byte_offset % 32U != 0U ||
      plan.output.binding_byte_offset % 32U != 0U) {
    throw std::invalid_argument(
        "BatchNorm X/Y binding offsets must be 32-byte aligned");
  }
  if (!std::isfinite(plan.epsilon) || plan.epsilon <= 0.0 ||
      !std::isfinite(plan.momentum) || plan.momentum < 0.0 ||
      plan.momentum > 1.0) {
    throw std::invalid_argument("BatchNorm epsilon or momentum is invalid");
  }

  plan.batch = static_cast<std::size_t>(plan.input.dimensions[0]);
  plan.channels = static_cast<std::size_t>(plan.input.dimensions[1]);
  plan.spatial = 1;
  for (std::size_t axis = 2; axis < rank; ++axis) {
    plan.spatial = io::checked_multiply(
        plan.spatial,
        static_cast<std::size_t>(plan.input.dimensions[axis]),
        "BatchNorm spatial element count");
  }
  plan.reduction_elements = io::checked_multiply(
      plan.batch, plan.spatial, "BatchNorm reduction element count");

  validate_parameter(
      plan.scale, plan.channels, plan.input.data_type, "scale");
  validate_parameter(
      plan.bias, plan.channels, plan.input.data_type, "bias");
  validate_parameter(plan.previous_running_mean,
                     plan.channels,
                     FLAGDNN_DATA_FLOAT32,
                     "previous running mean");
  validate_parameter(plan.previous_running_variance,
                     plan.channels,
                     FLAGDNN_DATA_FLOAT32,
                     "previous running variance");
  validate_parameter(
      plan.mean, plan.channels, FLAGDNN_DATA_FLOAT32, "mean");
  validate_parameter(plan.inverse_standard_deviation,
                     plan.channels,
                     FLAGDNN_DATA_FLOAT32,
                     "inverse standard deviation");
  validate_parameter(plan.next_running_mean,
                     plan.channels,
                     FLAGDNN_DATA_FLOAT32,
                     "next running mean");
  validate_parameter(plan.next_running_variance,
                     plan.channels,
                     FLAGDNN_DATA_FLOAT32,
                     "next running variance");

  const std::array<std::int64_t, 10> uids = {
      plan.input.uid,
      plan.scale.uid,
      plan.bias.uid,
      plan.previous_running_mean.uid,
      plan.previous_running_variance.uid,
      plan.output.uid,
      plan.mean.uid,
      plan.inverse_standard_deviation.uid,
      plan.next_running_mean.uid,
      plan.next_running_variance.uid,
  };
  if (std::any_of(uids.begin(), uids.end(), [](std::int64_t uid) {
        return uid <= 0;
      })) {
    throw std::invalid_argument("BatchNorm binding UID must be positive");
  }
  for (std::size_t left = 0; left < uids.size(); ++left) {
    for (std::size_t right = left + 1; right < uids.size(); ++right) {
      if (uids[left] == uids[right]) {
        throw std::invalid_argument("BatchNorm binding UID is duplicated");
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

TensorSpec convert(const TestTensor& tensor) {
  return {tensor.uid,
          tensor.data_type,
          tensor.dimensions,
          tensor.strides,
          tensor.binding_byte_offset};
}

}  // namespace

BatchnormPlan plan_batchnorm(const BatchnormTestCase& test_case) {
  BatchnormPlan result = {
      test_case.x,
      test_case.scale,
      test_case.bias,
      test_case.previous_running_mean,
      test_case.previous_running_variance,
      test_case.y,
      test_case.mean,
      test_case.inv_variance,
      test_case.next_running_mean,
      test_case.next_running_variance,
      test_case.epsilon,
      test_case.momentum,
  };
  validate_plan(result);
  return result;
}

BatchnormPlan plan_batchnorm(const BenchmarkCase& test_case) {
  if (test_case.operation != benchmarking::Operation::kBatchnorm ||
      test_case.output_count != 5 || test_case.tensors.size() != 10) {
    throw std::invalid_argument(
        "BatchNorm benchmark case signature is invalid");
  }
  BatchnormPlan result = {
      convert(test_case.tensors[0]),
      convert(test_case.tensors[1]),
      convert(test_case.tensors[2]),
      convert(test_case.tensors[3]),
      convert(test_case.tensors[4]),
      convert(test_case.tensors[5]),
      convert(test_case.tensors[6]),
      convert(test_case.tensors[7]),
      convert(test_case.tensors[8]),
      convert(test_case.tensors[9]),
      test_case.normalization.epsilon,
      test_case.normalization.momentum,
  };
  validate_plan(result);
  return result;
}

TestTensor batchnorm_reference_data_tensor(const TestTensor& tensor) {
  io::validate_layout(tensor);
  const std::size_t rank = tensor.dimensions.size();
  if (rank < 2 || rank > 8 || !is_supported_data_type(tensor.data_type)) {
    throw std::invalid_argument("BatchNorm reference tensor is invalid");
  }
  std::size_t spatial = 1;
  for (std::size_t axis = 2; axis < rank; ++axis) {
    spatial = io::checked_multiply(
        spatial,
        static_cast<std::size_t>(tensor.dimensions[axis]),
        "BatchNorm reference spatial element count");
  }
  if (spatial >
      static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
    throw std::invalid_argument(
        "BatchNorm reference spatial dimension overflows");
  }
  const Shape dimensions =
      rank == 4
          ? tensor.dimensions
          : Shape{tensor.dimensions[0],
                  tensor.dimensions[1],
                  1,
                  static_cast<std::int64_t>(spatial)};
  return {tensor.uid,
          tensor.data_type,
          dimensions,
          contiguous_strides(dimensions),
          0};
}

BenchmarkCase batchnorm_reference_benchmark_case(
    const BenchmarkCase& test_case) {
  (void)plan_batchnorm(test_case);
  BenchmarkCase result = test_case;
  result.tensors[0] = convert(
      flagdnn::validation::ascend::batchnorm_reference_data_tensor(
          convert(test_case.tensors[0])));
  result.tensors[5] = convert(
      flagdnn::validation::ascend::batchnorm_reference_data_tensor(
          convert(test_case.tensors[5])));
  (void)plan_batchnorm(result);
  return result;
}

std::vector<BatchnormTestCase> make_ascend_batchnorm_cases(
    std::span<const BatchnormTestCase> common_cases) {
  if (common_cases.empty()) {
    throw std::invalid_argument(
        "common BatchNorm functional catalog must contain cases");
  }
  std::vector<BatchnormTestCase> result(common_cases.begin(),
                                        common_cases.end());
  for (const BatchnormTestCase& test_case : result) {
    (void)plan_batchnorm(test_case);
  }
  return result;
}

std::vector<BenchmarkCase> make_ascend_batchnorm_benchmark_cases(
    std::span<const BenchmarkCase> common_cases) {
  if (common_cases.empty()) {
    throw std::invalid_argument(
        "common BatchNorm benchmark catalog must contain cases");
  }
  std::vector<BenchmarkCase> result(common_cases.begin(), common_cases.end());
  for (const BenchmarkCase& test_case : result) {
    (void)plan_batchnorm(test_case);
  }
  return result;
}

void batchnorm_reference_variance_to_invstd(
    std::span<float> saved_variance, double epsilon) {
  if (!std::isfinite(epsilon) || epsilon <= 0.0) {
    throw std::invalid_argument(
        "ACLNN BatchNorm reference epsilon is invalid");
  }
  for (float& variance : saved_variance) {
    if (!std::isfinite(variance) || variance < 0.0F) {
      throw std::runtime_error(
          "ACLNN BatchNorm returned an invalid saved variance");
    }
    variance = static_cast<float>(
        1.0 / std::sqrt(static_cast<double>(variance) + epsilon));
  }
}

void batchnorm_host_oracle(
    const BatchnormPlan& plan_value,
    std::span<const float> input_storage,
    std::span<const float> scale,
    std::span<const float> bias,
    std::span<const float> previous_running_mean,
    std::span<const float> previous_running_variance,
    std::span<float> output_storage,
    std::span<float> mean,
    std::span<float> inverse_standard_deviation,
    std::span<float> next_running_mean,
    std::span<float> next_running_variance) {
  BatchnormPlan plan = plan_value;
  validate_plan(plan);
  if (input_storage.size() < io::storage_element_count(plan.input) ||
      output_storage.size() < io::storage_element_count(plan.output) ||
      scale.size() != plan.channels || bias.size() != plan.channels ||
      previous_running_mean.size() != plan.channels ||
      previous_running_variance.size() != plan.channels ||
      mean.size() != plan.channels ||
      inverse_standard_deviation.size() != plan.channels ||
      next_running_mean.size() != plan.channels ||
      next_running_variance.size() != plan.channels) {
    throw std::invalid_argument("BatchNorm host oracle storage size is invalid");
  }

  std::vector<double> sums(plan.channels, 0.0);
  const std::size_t elements = io::element_count(plan.input);
  for (std::size_t logical = 0; logical < elements; ++logical) {
    const std::size_t channel = (logical / plan.spatial) % plan.channels;
    sums[channel] += input_storage[
        io::physical_offset_unchecked(logical, plan.input)];
  }
  for (std::size_t channel = 0; channel < plan.channels; ++channel) {
    mean[channel] = static_cast<float>(
        sums[channel] / static_cast<double>(plan.reduction_elements));
  }

  std::vector<double> squared_deviations(plan.channels, 0.0);
  for (std::size_t logical = 0; logical < elements; ++logical) {
    const std::size_t channel = (logical / plan.spatial) % plan.channels;
    const double difference =
        static_cast<double>(input_storage[
            io::physical_offset_unchecked(logical, plan.input)]) -
        static_cast<double>(mean[channel]);
    squared_deviations[channel] += difference * difference;
  }
  for (std::size_t channel = 0; channel < plan.channels; ++channel) {
    const double variance = squared_deviations[channel] /
                            static_cast<double>(plan.reduction_elements);
    inverse_standard_deviation[channel] =
        static_cast<float>(1.0 / std::sqrt(variance + plan.epsilon));
    const double unbiased_variance =
        plan.reduction_elements > 1
            ? squared_deviations[channel] /
                  static_cast<double>(plan.reduction_elements - 1)
            : 0.0;
    next_running_mean[channel] = static_cast<float>(
        (1.0 - plan.momentum) * previous_running_mean[channel] +
        plan.momentum * mean[channel]);
    next_running_variance[channel] = static_cast<float>(
        (1.0 - plan.momentum) * previous_running_variance[channel] +
        plan.momentum * unbiased_variance);
  }

  for (std::size_t logical = 0; logical < elements; ++logical) {
    const std::size_t channel = (logical / plan.spatial) % plan.channels;
    const std::size_t input_offset =
        io::physical_offset_unchecked(logical, plan.input);
    const std::size_t output_offset =
        io::physical_offset_unchecked(logical, plan.output);
    output_storage[output_offset] =
        (input_storage[input_offset] - mean[channel]) *
            inverse_standard_deviation[channel] * scale[channel] +
        bias[channel];
  }
}

}  // namespace flagdnn::validation::ascend
