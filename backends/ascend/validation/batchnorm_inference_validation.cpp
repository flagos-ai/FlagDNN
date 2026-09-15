/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "validation/batchnorm_inference_validation.hpp"

#include "validation/tensor_io.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace flagdnn::validation::ascend {
namespace {

namespace tensor_io = flagdnn::validation::ascend::tensor_io;
using benchmarking::BenchmarkCase;
using benchmarking::TensorSpec;
using testing::BatchnormInferenceTestCase;
using testing::TestTensor;
using Shape = std::vector<std::int64_t>;

constexpr std::array<flagdnnDataType_t, 3> kDataTypes = {
    FLAGDNN_DATA_FLOAT32,
    FLAGDNN_DATA_FLOAT16,
    FLAGDNN_DATA_BFLOAT16,
};

std::vector<std::int64_t> contiguous_strides(const Shape& dimensions) {
  std::vector<std::int64_t> result(dimensions.size());
  std::int64_t stride = 1;
  for (std::size_t axis = dimensions.size(); axis != 0; --axis) {
    if (dimensions[axis - 1] <= 0 ||
        stride > std::numeric_limits<std::int64_t>::max() /
                     dimensions[axis - 1]) {
      throw std::invalid_argument(
          "BatchNorm inference shape overflows contiguous strides");
    }
    result[axis - 1] = stride;
    stride *= dimensions[axis - 1];
  }
  return result;
}

bool is_supported_data_type(flagdnnDataType_t data_type) {
  return std::find(kDataTypes.begin(), kDataTypes.end(), data_type) !=
         kDataTypes.end();
}

std::string data_type_name(flagdnnDataType_t data_type) {
  switch (data_type) {
    case FLAGDNN_DATA_INT32:
      throw std::invalid_argument(
          "INT32 is not supported by this validation adapter");

    case FLAGDNN_DATA_FLOAT32:
      return "fp32";
    case FLAGDNN_DATA_FLOAT16:
      return "fp16";
    case FLAGDNN_DATA_BFLOAT16:
      return "bfloat16";
    case FLAGDNN_DATA_BOOLEAN:
    case FLAGDNN_DATA_FP8_E8M0:
    case FLAGDNN_DATA_FP8_E4M3:
    case FLAGDNN_DATA_FP8_E5M2:
      break;
  }
  throw std::invalid_argument("unsupported BatchNorm inference data type");
}

std::size_t parameter_elements(const TestTensor& tensor) {
  tensor_io::validate_layout(tensor);
  std::size_t result = 1;
  for (const std::int64_t dimension : tensor.dimensions) {
    result = tensor_io::checked_multiply(
        result,
        static_cast<std::size_t>(dimension),
        "BatchNorm inference parameter element count");
  }
  return result;
}

void validate_parameter(const TestTensor& tensor,
                        std::size_t channels,
                        const char* name) {
  if (tensor.data_type != FLAGDNN_DATA_FLOAT32) {
    throw std::invalid_argument(
        std::string("BatchNorm inference ") + name + " must use FP32");
  }
  if (parameter_elements(tensor) != channels) {
    throw std::invalid_argument(
        std::string("BatchNorm inference ") + name +
        " channel element count is invalid");
  }
  if (tensor.strides != contiguous_strides(tensor.dimensions)) {
    throw std::invalid_argument(
        std::string("BatchNorm inference ") + name +
        " must be contiguous");
  }
  if (tensor.binding_byte_offset % 32U != 0U) {
    throw std::invalid_argument(
        std::string("BatchNorm inference ") + name +
        " binding offset must be 32-byte aligned");
  }
}

void validate_plan(const BatchnormInferencePlan& plan) {
  const std::size_t rank = plan.input.dimensions.size();
  if (rank < 2 || rank > 8) {
    throw std::invalid_argument("BatchNorm inference rank must be in [2, 8]");
  }
  tensor_io::validate_layout(plan.input);
  tensor_io::validate_layout(plan.output);
  if (!is_supported_data_type(plan.input.data_type) ||
      plan.output.data_type != plan.input.data_type ||
      plan.output.dimensions != plan.input.dimensions) {
    throw std::invalid_argument(
        "BatchNorm inference X/Y shape or data type is invalid");
  }
  if (plan.input.binding_byte_offset % 32U != 0U ||
      plan.output.binding_byte_offset % 32U != 0U) {
    throw std::invalid_argument(
        "BatchNorm inference X/Y binding offsets must be 32-byte aligned");
  }
  const std::size_t channels =
      static_cast<std::size_t>(plan.input.dimensions[1]);
  validate_parameter(plan.mean, channels, "mean");
  validate_parameter(plan.inverse_standard_deviation, channels, "invstd");
  validate_parameter(plan.weight, channels, "scale");
  validate_parameter(plan.bias, channels, "bias");
  const std::array<std::int64_t, 6> uids = {
      plan.input.uid,
      plan.mean.uid,
      plan.inverse_standard_deviation.uid,
      plan.weight.uid,
      plan.bias.uid,
      plan.output.uid,
  };
  if (std::any_of(uids.begin(), uids.end(), [](std::int64_t uid) {
        return uid <= 0;
      })) {
    throw std::invalid_argument(
        "BatchNorm inference binding UID must be positive");
  }
  for (std::size_t left = 0; left < uids.size(); ++left) {
    for (std::size_t right = left + 1; right < uids.size(); ++right) {
      if (uids[left] == uids[right]) {
        throw std::invalid_argument(
            "BatchNorm inference binding UID is duplicated");
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

TestTensor make_tensor(std::int64_t uid,
                       const Shape& dimensions,
                       flagdnnDataType_t data_type,
                       const Shape& strides = {},
                       std::size_t binding_byte_offset = 0) {
  return {uid,
          data_type,
          dimensions,
          strides.empty() ? contiguous_strides(dimensions) : strides,
          binding_byte_offset};
}

BatchnormInferenceTestCase make_local_case(flagdnnDataType_t data_type,
                                           bool rank_five,
                                           std::int64_t uid) {
  BatchnormInferenceTestCase result;
  const Shape shape = rank_five ? Shape{2, 5, 3, 2, 4} : Shape{4, 7};
  const std::int64_t channels = shape[1];
  const Shape parameter_shape = {channels};
  const Shape input_strides =
      rank_five ? Shape{300, 50, 15, 6, 1} : contiguous_strides(shape);
  const Shape output_strides =
      rank_five ? Shape{400, 70, 20, 7, 1} : contiguous_strides(shape);
  result.name = "batchnorm_inference_ascend_" + data_type_name(data_type) +
                (rank_five ? "_rank5_gapped" : "_rank2_nonpower_c");
  result.x = make_tensor(
      uid, shape, data_type, input_strides, rank_five ? 64U : 0U);
  result.mean = make_tensor(uid + 1, parameter_shape, FLAGDNN_DATA_FLOAT32);
  result.inv_variance =
      make_tensor(uid + 2, parameter_shape, FLAGDNN_DATA_FLOAT32);
  result.scale = make_tensor(uid + 3, parameter_shape, FLAGDNN_DATA_FLOAT32);
  result.bias = make_tensor(uid + 4, parameter_shape, FLAGDNN_DATA_FLOAT32);
  result.y = make_tensor(
      uid + 5, shape, data_type, output_strides, rank_five ? 96U : 0U);
  result.absolute_tolerance =
      data_type == FLAGDNN_DATA_FLOAT32
          ? 1.0e-5
          : (data_type == FLAGDNN_DATA_FLOAT16 ? 2.0e-2 : 5.0e-2);
  result.relative_tolerance = result.absolute_tolerance;
  return result;
}

TensorSpec convert(const TestTensor& tensor) {
  return {tensor.uid,
          tensor.data_type,
          tensor.dimensions,
          tensor.strides,
          tensor.binding_byte_offset};
}

}  // namespace

BatchnormInferencePlan plan_batchnorm_inference(
    const BatchnormInferenceTestCase& test_case) {
  BatchnormInferencePlan result = {
      test_case.x,
      test_case.scale,
      test_case.bias,
      test_case.mean,
      test_case.inv_variance,
      test_case.y,
  };
  validate_plan(result);
  return result;
}

BatchnormInferencePlan plan_batchnorm_inference(
    const BenchmarkCase& test_case) {
  if (test_case.operation != benchmarking::Operation::kBatchnormInference ||
      test_case.output_count != 1 || test_case.tensors.size() != 6) {
    throw std::invalid_argument(
        "BatchNorm inference benchmark case signature is invalid");
  }
  BatchnormInferencePlan result = {
      convert(test_case.tensors[0]),
      convert(test_case.tensors[3]),
      convert(test_case.tensors[4]),
      convert(test_case.tensors[1]),
      convert(test_case.tensors[2]),
      convert(test_case.tensors[5]),
  };
  validate_plan(result);
  return result;
}

TestTensor batchnorm_inference_reference_data_tensor(
    const TestTensor& tensor) {
  tensor_io::validate_layout(tensor);
  const std::size_t rank = tensor.dimensions.size();
  if (rank < 2 || rank > 8 || !is_supported_data_type(tensor.data_type)) {
    throw std::invalid_argument(
        "BatchNorm inference reference tensor is invalid");
  }

  std::size_t spatial = 1;
  for (std::size_t axis = 2; axis < rank; ++axis) {
    spatial = tensor_io::checked_multiply(
        spatial,
        static_cast<std::size_t>(tensor.dimensions[axis]),
        "BatchNorm inference reference spatial element count");
  }
  if (spatial >
      static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
    throw std::invalid_argument(
        "BatchNorm inference reference spatial dimension overflows");
  }

  const Shape dimensions = {
      tensor.dimensions[0],
      tensor.dimensions[1],
      1,
      static_cast<std::int64_t>(spatial),
  };
  return {tensor.uid,
          tensor.data_type,
          dimensions,
          contiguous_strides(dimensions),
          0};
}

BenchmarkCase batchnorm_inference_reference_benchmark_case(
    const BenchmarkCase& test_case) {
  (void)plan_batchnorm_inference(test_case);
  BenchmarkCase result = test_case;
  result.tensors[0] = convert(
      batchnorm_inference_reference_data_tensor(convert(test_case.tensors[0])));
  result.tensors[5] = convert(
      batchnorm_inference_reference_data_tensor(convert(test_case.tensors[5])));
  (void)plan_batchnorm_inference(result);
  return result;
}

std::vector<BatchnormInferenceTestCase>
make_ascend_batchnorm_inference_cases(
    std::span<const BatchnormInferenceTestCase> common_cases) {
  if (common_cases.empty()) {
    throw std::invalid_argument(
        "common BatchNorm inference functional catalog must contain cases");
  }
  std::vector<BatchnormInferenceTestCase> result(
      common_cases.begin(), common_cases.end());
  std::int64_t uid = 174000;
  for (const bool rank_five : {false, true}) {
    for (const flagdnnDataType_t data_type : kDataTypes) {
      result.push_back(make_local_case(data_type, rank_five, uid));
      uid += 6;
    }
  }
  for (const BatchnormInferenceTestCase& test_case : result) {
    (void)plan_batchnorm_inference(test_case);
  }
  return result;
}

std::vector<BenchmarkCase>
make_ascend_batchnorm_inference_benchmark_cases(
    std::span<const BenchmarkCase> common_cases) {
  if (common_cases.empty()) {
    throw std::invalid_argument(
        "common BatchNorm inference benchmark catalog must contain cases");
  }
  std::vector<BenchmarkCase> result(common_cases.begin(), common_cases.end());
  const BatchnormInferenceTestCase local =
      make_local_case(FLAGDNN_DATA_FLOAT32, true, 176000);
  BenchmarkCase special;
  special.name = "batchnorm_inference_perf_ascend_fp32_rank5_gapped";
  special.operation = benchmarking::Operation::kBatchnormInference;
  special.tensors = {
      convert(local.x),
      convert(local.mean),
      convert(local.inv_variance),
      convert(local.scale),
      convert(local.bias),
      convert(local.y),
  };
  special.input_domains = {
      benchmarking::InputDomain::kReal,
      benchmarking::InputDomain::kReal,
      benchmarking::InputDomain::kPositive,
      benchmarking::InputDomain::kReal,
      benchmarking::InputDomain::kReal,
  };
  special.absolute_tolerance = local.absolute_tolerance;
  special.relative_tolerance = local.relative_tolerance;
  special.benchmark.warmup_iterations = 5;
  special.benchmark.sample_count = 10;
  special.benchmark.iterations_per_sample = 20;
  result.push_back(std::move(special));
  for (const BenchmarkCase& test_case : result) {
    (void)plan_batchnorm_inference(test_case);
  }
  return result;
}

void batchnorm_inference_host_oracle(
    const BatchnormInferencePlan& plan,
    std::span<const float> input_storage,
    std::span<const float> mean,
    std::span<const float> inverse_standard_deviation,
    std::span<const float> scale,
    std::span<const float> bias,
    std::span<float> output_storage) {
  validate_plan(plan);
  const std::size_t channels =
      static_cast<std::size_t>(plan.input.dimensions[1]);
  if (input_storage.size() < tensor_io::storage_element_count(plan.input) ||
      output_storage.size() < tensor_io::storage_element_count(plan.output) ||
      mean.size() != channels ||
      inverse_standard_deviation.size() != channels ||
      scale.size() != channels || bias.size() != channels) {
    throw std::invalid_argument(
        "BatchNorm inference host oracle storage size is invalid");
  }
  std::size_t inner = 1;
  for (std::size_t axis = 2; axis < plan.input.dimensions.size(); ++axis) {
    inner = tensor_io::checked_multiply(
        inner,
        static_cast<std::size_t>(plan.input.dimensions[axis]),
        "BatchNorm inference inner element count");
  }
  const std::size_t elements = tensor_io::element_count(plan.input);
  for (std::size_t logical = 0; logical < elements; ++logical) {
    const std::size_t channel = (logical / inner) % channels;
    const std::size_t input_offset =
        tensor_io::physical_offset_unchecked(logical, plan.input);
    const std::size_t output_offset =
        tensor_io::physical_offset_unchecked(logical, plan.output);
    output_storage[output_offset] =
        (input_storage[input_offset] - mean[channel]) *
            inverse_standard_deviation[channel] * scale[channel] +
        bias[channel];
  }
}

}  // namespace flagdnn::validation::ascend
