// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "common/runner.hpp"

#include "benchmark/acdnn_provider.hpp"
#include "capability.hpp"
#include "common/flagdnn_provider.hpp"
#include "numeric_types.hpp"
#include "ppu_driver.hpp"
#include "tensor_io.hpp"

#include <acdnn.h>
#include <cuda.h>
#include <flagdnn/flagdnn.hpp>

#include <unistd.h>
#include <sys/wait.h>

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#ifndef FLAGDNN_THEAD_ACDNN_CAPABILITY_CATALOG
#define FLAGDNN_THEAD_ACDNN_CAPABILITY_CATALOG "capability.json"
#endif

#ifndef FLAGDNN_THEAD_BENCHMARK_CATALOG
#define FLAGDNN_THEAD_BENCHMARK_CATALOG "comparable_cases.json"
#endif

#ifndef FLAGDNN_THEAD_PPU_SDK_VERSION
#define FLAGDNN_THEAD_PPU_SDK_VERSION "unknown"
#endif

namespace flagdnn::benchmarking {
namespace {

namespace tv = flagdnn::validation::thead;
namespace tvb = flagdnn::validation::thead::benchmark;

constexpr int kSkipReturnCode = 77;
constexpr float kPaddingSentinel = -64.0F;

class BenchmarkCache final {
 public:
  BenchmarkCache() {
    const char *configured = std::getenv("FLAGDNN_CACHE_PATH");
    if (configured != nullptr && configured[0] != '\0') {
      path_ = configured;
      std::filesystem::create_directories(path_);
      return;
    }
    std::string pattern =
        (std::filesystem::temp_directory_path() /
         "flagdnn-thead-benchmark-XXXXXX")
            .string();
    std::vector<char> writable(pattern.begin(), pattern.end());
    writable.push_back('\0');
    char *created = ::mkdtemp(writable.data());
    if (created == nullptr) {
      throw std::runtime_error("mkdtemp failed for THead benchmark cache");
    }
    path_ = created;
    owned_ = true;
  }

  ~BenchmarkCache() noexcept {
    if (owned_) {
      std::error_code ignored;
      std::filesystem::remove_all(path_, ignored);
    }
  }

  BenchmarkCache(const BenchmarkCache &) = delete;
  BenchmarkCache &operator=(const BenchmarkCache &) = delete;

  [[nodiscard]] const std::filesystem::path &path() const noexcept {
    return path_;
  }

 private:
  std::filesystem::path path_;
  bool owned_ = false;
};

std::size_t checked_multiply(std::size_t left, std::size_t right,
                             std::string_view description) {
  if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right) {
    throw std::overflow_error(std::string(description) + " overflows size_t");
  }
  return left * right;
}

std::size_t element_count(const TensorSpec &tensor) {
  if (tensor.dimensions.empty() ||
      tensor.dimensions.size() != tensor.strides.size()) {
    throw std::invalid_argument("benchmark tensor geometry is invalid");
  }
  std::size_t result = 1;
  for (const std::int64_t dimension : tensor.dimensions) {
    if (dimension <= 0) {
      throw std::invalid_argument("benchmark tensor extent is not positive");
    }
    result = checked_multiply(result, static_cast<std::size_t>(dimension),
                              "benchmark tensor element count");
  }
  return result;
}

std::size_t storage_element_count(const TensorSpec &tensor) {
  if (tensor.dimensions.empty() ||
      tensor.dimensions.size() != tensor.strides.size()) {
    throw std::invalid_argument("benchmark tensor geometry is invalid");
  }
  std::size_t result = 1;
  for (std::size_t axis = 0; axis < tensor.dimensions.size(); ++axis) {
    if (tensor.dimensions[axis] <= 0 || tensor.strides[axis] <= 0) {
      throw std::invalid_argument("benchmark tensor geometry is invalid");
    }
    const std::size_t term = checked_multiply(
        static_cast<std::size_t>(tensor.dimensions[axis] - 1),
        static_cast<std::size_t>(tensor.strides[axis]),
        "benchmark tensor storage span");
    if (result > std::numeric_limits<std::size_t>::max() - term) {
      throw std::overflow_error("benchmark tensor storage span overflows");
    }
    result += term;
  }
  return result;
}

std::size_t logical_offset(std::size_t logical_index,
                           const TensorSpec &tensor) {
  std::size_t result = 0;
  for (std::size_t axis = tensor.dimensions.size(); axis != 0; --axis) {
    const std::size_t current = axis - 1;
    const std::size_t dimension =
        static_cast<std::size_t>(tensor.dimensions[current]);
    const std::size_t coordinate = logical_index % dimension;
    logical_index /= dimension;
    const std::size_t term = checked_multiply(
        coordinate, static_cast<std::size_t>(tensor.strides[current]),
        "benchmark logical offset");
    if (result > std::numeric_limits<std::size_t>::max() - term) {
      throw std::overflow_error("benchmark logical offset overflows");
    }
    result += term;
  }
  return result;
}

bool is_contiguous(const TensorSpec &tensor) {
  std::int64_t expected = 1;
  for (std::size_t axis = tensor.dimensions.size(); axis != 0; --axis) {
    if (tensor.strides[axis - 1] != expected ||
        tensor.dimensions[axis - 1] <= 0 ||
        tensor.dimensions[axis - 1] >
            std::numeric_limits<std::int64_t>::max() / expected) {
      return false;
    }
    expected *= tensor.dimensions[axis - 1];
  }
  return true;
}

std::string data_type_name(flagdnnDataType_t data_type) {
  switch (data_type) {
    case FLAGDNN_DATA_INT32:
      return "int32";
    case FLAGDNN_DATA_FP8_E8M0:
      return "fp8_e8m0";
    case FLAGDNN_DATA_FLOAT32:
      return "fp32";
    case FLAGDNN_DATA_FLOAT16:
      return "fp16";
    case FLAGDNN_DATA_BFLOAT16:
      return "bf16";
    case FLAGDNN_DATA_FP8_E4M3:
      return "fp8_e4m3";
    case FLAGDNN_DATA_FP8_E5M2:
      return "fp8_e5m2";
    case FLAGDNN_DATA_BOOLEAN:
      return "bool";
  }
  return "unknown";
}

std::string shape_name(const TensorSpec &tensor) {
  std::string result;
  for (const std::int64_t dimension : tensor.dimensions) {
    if (!result.empty()) {
      result.push_back('x');
    }
    result += std::to_string(dimension);
  }
  return result.empty() ? "scalar" : result;
}

std::string operation_from_suite(std::string_view suite_name) {
  constexpr std::string_view prefix = "FLAGDNN_";
  constexpr std::string_view suffix = "_BENCHMARK";
  if (!suite_name.starts_with(prefix) || !suite_name.ends_with(suffix) ||
      suite_name.size() <= prefix.size() + suffix.size()) {
    throw std::invalid_argument("malformed THead benchmark suite name");
  }
  std::string operation(
      suite_name.substr(prefix.size(),
                        suite_name.size() - prefix.size() - suffix.size()));
  std::ranges::transform(operation, operation.begin(),
                         [](unsigned char character) {
                           return static_cast<char>(
                               std::tolower(character));
                         });
  return operation;
}

struct BoundTensor {
  TensorSpec specification;
  std::unique_ptr<tv::DeviceBuffer> buffer;
};

struct PreparedBuffers {
  std::vector<BoundTensor> tensors;
  std::vector<flagdnnBinding_t> bindings;
  std::size_t input_count = 0;
};

std::vector<float> make_logical_input(const TensorSpec &tensor,
                                      std::size_t input_index,
                                      flagdnnPointwiseMode_t mode,
                                      InputDomain domain) {
  std::vector<float> result(element_count(tensor));
  for (std::size_t index = 0; index < result.size(); ++index) {
    const int centered =
        static_cast<int>((index * 17 + input_index * 11) % 41) - 20;
    if (domain == InputDomain::kPositive ||
        domain == InputDomain::kDivisor ||
        domain == InputDomain::kPower ||
        domain == InputDomain::kModulo ||
        (domain == InputDomain::kReal &&
         (mode == FLAGDNN_POINTWISE_LOG ||
        mode == FLAGDNN_POINTWISE_RSQRT ||
        mode == FLAGDNN_POINTWISE_DIV ||
        mode == FLAGDNN_POINTWISE_POW ||
        mode == FLAGDNN_POINTWISE_MOD ||
          mode == FLAGDNN_POINTWISE_RECIPROCAL))) {
      result[index] =
          static_cast<float>((index * 17 + input_index * 11) % 41 + 1) /
          static_cast<float>(13 + input_index);
    } else if (domain == InputDomain::kTan ||
               (domain == InputDomain::kReal &&
                mode == FLAGDNN_POINTWISE_TAN)) {
      result[index] = static_cast<float>(centered) / 32.0F;
    } else if (domain == InputDomain::kScaled) {
      result[index] = static_cast<float>(centered) * 0.25F;
    } else if (domain == InputDomain::kLogical) {
      result[index] = (index + input_index) % 3 == 0 ? 0.0F : 1.0F;
    } else {
      result[index] = static_cast<float>(centered) /
                      static_cast<float>(13 + input_index);
    }
  }
  return result;
}

std::vector<float> scatter(std::span<const float> logical,
                           const TensorSpec &tensor) {
  if (logical.size() != element_count(tensor)) {
    throw std::invalid_argument("benchmark logical input size mismatch");
  }
  std::vector<float> result(storage_element_count(tensor), kPaddingSentinel);
  for (std::size_t index = 0; index < logical.size(); ++index) {
    result.at(logical_offset(index, tensor)) = logical[index];
  }
  return result;
}

std::vector<std::uint8_t> scatter_boolean(
    std::span<const float> logical, const TensorSpec &tensor) {
  if (logical.size() != element_count(tensor)) {
    throw std::invalid_argument("benchmark logical boolean size mismatch");
  }
  std::vector<std::uint8_t> result(storage_element_count(tensor), 0x7fU);
  for (std::size_t index = 0; index < logical.size(); ++index) {
    result.at(logical_offset(index, tensor)) =
        logical[index] == 0.0F ? 0U : 1U;
  }
  return result;
}

BoundTensor make_bound_tensor(const TensorSpec &tensor,
                              std::optional<std::span<const float>> logical,
                              CUstream stream) {
  if ((tensor.data_type != FLAGDNN_DATA_FLOAT32 &&
       tensor.data_type != FLAGDNN_DATA_FLOAT16 &&
       tensor.data_type != FLAGDNN_DATA_BFLOAT16 &&
       tensor.data_type != FLAGDNN_DATA_BOOLEAN) ||
      tensor.binding_byte_offset % tv::element_size(tensor.data_type) != 0) {
    throw std::invalid_argument(
        "comparable THead benchmark tensor type or alignment is invalid");
  }
  if (tensor.data_type == FLAGDNN_DATA_BOOLEAN) {
    const std::vector<std::uint8_t> physical =
        logical.has_value()
            ? scatter_boolean(*logical, tensor)
            : std::vector<std::uint8_t>(storage_element_count(tensor),
                                        0x7fU);
    if (tensor.binding_byte_offset >
        std::numeric_limits<std::size_t>::max() - physical.size()) {
      throw std::overflow_error("benchmark bool allocation size overflows");
    }
    auto buffer = std::make_unique<tv::DeviceBuffer>(
        tensor.binding_byte_offset + physical.size());
    tv::copy_to_device_async(
        *buffer,
        std::as_bytes(std::span<const std::uint8_t>(physical.data(),
                                                    physical.size())),
        tensor.binding_byte_offset, stream);
    return {tensor, std::move(buffer)};
  }
  const std::vector<float> physical_values =
      logical.has_value()
          ? scatter(*logical, tensor)
          : std::vector<float>(storage_element_count(tensor),
                               kPaddingSentinel);
  const std::vector<std::byte> physical =
      tv::encode_floating(tensor.data_type, physical_values);
  const std::size_t payload_bytes = physical.size();
  if (tensor.binding_byte_offset >
      std::numeric_limits<std::size_t>::max() - payload_bytes) {
    throw std::overflow_error("benchmark allocation size overflows");
  }
  auto buffer = std::make_unique<tv::DeviceBuffer>(
      tensor.binding_byte_offset + payload_bytes);
  tv::copy_to_device_async(*buffer, physical, tensor.binding_byte_offset,
                           stream);
  return {tensor, std::move(buffer)};
}

PreparedBuffers prepare_buffers(const BenchmarkCase &specification,
                                tv::DeviceStream &stream,
                                bool dense_layout_output = false) {
  const bool is_add = specification.operation == Operation::kAdd;
  const bool is_add_square =
      specification.operation == Operation::kGraph &&
      specification.name.starts_with("add_square_perf_");
  const bool is_conv_bias_relu =
      specification.operation == Operation::kGraph &&
      specification.name.starts_with("conv_bias_relu_perf_");
  const bool is_sub =
      specification.operation == Operation::kPointwise &&
      specification.pointwise_mode == FLAGDNN_POINTWISE_SUB;
  const bool is_mul =
      specification.operation == Operation::kPointwise &&
      specification.pointwise_mode == FLAGDNN_POINTWISE_MUL;
  const bool is_min =
      specification.operation == Operation::kPointwise &&
      specification.pointwise_mode == FLAGDNN_POINTWISE_MIN;
  const bool is_max =
      specification.operation == Operation::kPointwise &&
      specification.pointwise_mode == FLAGDNN_POINTWISE_MAX;
  const bool is_div =
      specification.operation == Operation::kPointwise &&
      specification.pointwise_mode == FLAGDNN_POINTWISE_DIV;
  const bool is_pow =
      specification.operation == Operation::kPointwise &&
      specification.pointwise_mode == FLAGDNN_POINTWISE_POW;
  const bool is_mod =
      specification.operation == Operation::kPointwise &&
      specification.pointwise_mode == FLAGDNN_POINTWISE_MOD;
  const bool is_sigmoid_backward =
      specification.operation == Operation::kPointwise &&
      specification.pointwise_mode == FLAGDNN_POINTWISE_SIGMOID_BWD;
  const bool is_reciprocal =
      specification.operation == Operation::kPointwise &&
      specification.pointwise_mode == FLAGDNN_POINTWISE_RECIPROCAL;
  const bool is_erf =
      specification.operation == Operation::kPointwise &&
      specification.pointwise_mode == FLAGDNN_POINTWISE_ERF;
  const bool is_logical_not =
      specification.operation == Operation::kPointwise &&
      specification.pointwise_mode == FLAGDNN_POINTWISE_LOGICAL_NOT;
  const bool is_logical_binary =
      specification.operation == Operation::kPointwise &&
      (specification.pointwise_mode == FLAGDNN_POINTWISE_LOGICAL_AND ||
       specification.pointwise_mode == FLAGDNN_POINTWISE_LOGICAL_OR);
  const bool is_binary_select =
      specification.operation == Operation::kPointwise &&
      specification.pointwise_mode == FLAGDNN_POINTWISE_BINARY_SELECT;
  const bool is_comparison =
      specification.operation == Operation::kPointwise &&
      (specification.pointwise_mode == FLAGDNN_POINTWISE_CMP_EQ ||
       specification.pointwise_mode == FLAGDNN_POINTWISE_CMP_NEQ ||
       specification.pointwise_mode == FLAGDNN_POINTWISE_CMP_GT ||
       specification.pointwise_mode == FLAGDNN_POINTWISE_CMP_GE ||
       specification.pointwise_mode == FLAGDNN_POINTWISE_CMP_LT ||
       specification.pointwise_mode == FLAGDNN_POINTWISE_CMP_LE);
  const bool is_layout =
      specification.operation == Operation::kReshape ||
      specification.operation == Operation::kTranspose ||
      specification.operation == Operation::kSlice;
  const bool is_reduction =
      specification.operation == Operation::kReduction;
  const bool is_batchnorm =
      specification.operation == Operation::kBatchnorm;
  const bool is_batchnorm_inference =
      specification.operation == Operation::kBatchnormInference;
  const bool is_layernorm =
      specification.operation == Operation::kLayernorm;
  const bool is_rmsnorm =
      specification.operation == Operation::kRmsnorm;
  const bool is_matmul = specification.operation == Operation::kMatmul;
  const bool is_convolution =
      specification.operation == Operation::kConvolutionFprop ||
      specification.operation == Operation::kConvolutionDgrad ||
      specification.operation == Operation::kConvolutionWgrad;
  const bool is_relu = specification.operation == Operation::kRelu;
  const bool is_leaky_relu =
      specification.operation == Operation::kPointwise &&
      specification.pointwise_mode == FLAGDNN_POINTWISE_RELU_FWD &&
      specification.pointwise_attributes.flags ==
          FLAGDNN_POINTWISE_ATTRIBUTE_RELU_LOWER_CLIP_SLOPE &&
      specification.pointwise_attributes.relu_lower_clip_slope == 0.2;
  const bool is_sigmoid =
      specification.operation == Operation::kPointwise &&
      specification.pointwise_mode == FLAGDNN_POINTWISE_SIGMOID_FWD;
  const bool is_tanh =
      specification.operation == Operation::kPointwise &&
      specification.pointwise_mode == FLAGDNN_POINTWISE_TANH_FWD;
  const bool is_elu =
      specification.operation == Operation::kPointwise &&
      specification.pointwise_mode == FLAGDNN_POINTWISE_ELU_FWD;
  const bool is_identity =
      specification.operation == Operation::kPointwise &&
      specification.pointwise_mode == FLAGDNN_POINTWISE_IDENTITY;
  const bool is_gelu =
      specification.operation == Operation::kPointwise &&
      specification.pointwise_mode == FLAGDNN_POINTWISE_GELU_FWD;
  const bool is_sqrt =
      specification.operation == Operation::kPointwise &&
      specification.pointwise_mode == FLAGDNN_POINTWISE_SQRT;
  const bool is_neg =
      specification.operation == Operation::kPointwise &&
      specification.pointwise_mode == FLAGDNN_POINTWISE_NEG;
  const bool is_abs =
      specification.operation == Operation::kPointwise &&
      specification.pointwise_mode == FLAGDNN_POINTWISE_ABS;
  const bool is_ceil =
      specification.operation == Operation::kPointwise &&
      specification.pointwise_mode == FLAGDNN_POINTWISE_CEIL;
  const bool is_floor =
      specification.operation == Operation::kPointwise &&
      specification.pointwise_mode == FLAGDNN_POINTWISE_FLOOR;
  const bool is_exp =
      specification.operation == Operation::kPointwise &&
      specification.pointwise_mode == FLAGDNN_POINTWISE_EXP;
  const bool is_log =
      specification.operation == Operation::kPointwise &&
      specification.pointwise_mode == FLAGDNN_POINTWISE_LOG;
  const bool is_cos =
      specification.operation == Operation::kPointwise &&
      specification.pointwise_mode == FLAGDNN_POINTWISE_COS;
  const bool is_rsqrt =
      specification.operation == Operation::kPointwise &&
      specification.pointwise_mode == FLAGDNN_POINTWISE_RSQRT;
  const bool is_sin =
      specification.operation == Operation::kPointwise &&
      specification.pointwise_mode == FLAGDNN_POINTWISE_SIN;
  const bool is_tan =
      specification.operation == Operation::kPointwise &&
      specification.pointwise_mode == FLAGDNN_POINTWISE_TAN;
  const bool is_softplus =
      specification.operation == Operation::kPointwise &&
      specification.pointwise_mode == FLAGDNN_POINTWISE_SOFTPLUS_FWD;
  const bool is_swish =
      specification.operation == Operation::kPointwise &&
      specification.pointwise_mode == FLAGDNN_POINTWISE_SWISH_FWD;
  const bool is_gelu_approx_tanh =
      specification.operation == Operation::kPointwise &&
      specification.pointwise_mode ==
          FLAGDNN_POINTWISE_GELU_APPROX_TANH_FWD;
  const bool is_activation = is_relu || is_leaky_relu || is_sigmoid ||
                             is_tanh || is_elu || is_identity || is_gelu;
  const bool is_unary =
      is_activation || is_sqrt || is_neg || is_abs || is_ceil || is_floor ||
      is_exp || is_log || is_cos || is_rsqrt || is_sin || is_tan ||
      is_softplus || is_swish || is_gelu_approx_tanh || is_reciprocal ||
      is_erf || is_logical_not;
  const std::size_t expected_tensor_count =
      is_batchnorm ? 10
                   : (is_batchnorm_inference ? 6
                      : (is_layernorm ? 6
                         : (is_rmsnorm ? 5
                            : ((is_conv_bias_relu || is_binary_select) ? 4
                               : ((is_unary || is_layout || is_reduction)
                                      ? 2
                                      : 3)))));
  if ((!is_add && !is_sub && !is_mul && !is_min && !is_max && !is_div &&
       !is_pow && !is_mod && !is_sigmoid_backward && !is_add_square &&
       !is_conv_bias_relu && !is_binary_select && !is_logical_binary &&
       !is_comparison && !is_unary && !is_layout && !is_reduction &&
       !is_batchnorm && !is_batchnorm_inference && !is_layernorm &&
       !is_rmsnorm && !is_matmul &&
       !is_convolution) ||
      specification.output_count !=
          (is_batchnorm ? 5U : (is_layernorm ? 3U : (is_rmsnorm ? 2U : 1U))) ||
      specification.tensors.size() != expected_tensor_count) {
    throw std::invalid_argument(
        "THead paired benchmark requires a catalog-qualified case");
  }
  PreparedBuffers result;
  result.input_count = input_tensor_count(specification);
  result.tensors.reserve(specification.tensors.size());
  std::vector<std::vector<float>> logical_inputs;
  logical_inputs.reserve(result.input_count);
  for (std::size_t index = 0; index < result.input_count; ++index) {
    const flagdnnPointwiseMode_t input_mode =
        is_reduction &&
                specification.reduction_mode == FLAGDNN_REDUCTION_MUL
            ? FLAGDNN_POINTWISE_TAN
            : specification.pointwise_mode;
    const InputDomain input_domain =
        index < specification.input_domains.size()
            ? specification.input_domains[index]
            : specification.input_domain;
    logical_inputs.push_back(make_logical_input(
        specification.tensors[index], index, input_mode, input_domain));
    result.tensors.push_back(make_bound_tensor(
        specification.tensors[index],
        std::span<const float>(logical_inputs.back()), stream.get()));
  }
  for (std::size_t index = 0; index < specification.output_count; ++index) {
    TensorSpec output_specification =
        specification.tensors[result.input_count + index];
    if (dense_layout_output && is_layout) {
      output_specification.strides =
          contiguous_strides(output_specification.dimensions);
    }
    result.tensors.push_back(make_bound_tensor(
        output_specification, std::nullopt, stream.get()));
  }
  result.bindings.reserve(result.tensors.size());
  for (const BoundTensor &tensor : result.tensors) {
    result.bindings.push_back(
        {tensor.specification.uid,
         tensor.buffer->at(tensor.specification.binding_byte_offset)});
  }
  tv::check_driver(cuStreamSynchronize(stream.get()),
                   "cuStreamSynchronize(prepare benchmark buffers)");
  return result;
}

std::vector<float> read_output(PreparedBuffers &buffers,
                               tv::DeviceStream &stream,
                               std::size_t output_index = 0) {
  BoundTensor &output =
      buffers.tensors.at(buffers.input_count + output_index);
  if (output.specification.data_type == FLAGDNN_DATA_BOOLEAN) {
    std::vector<std::uint8_t> bytes(
        storage_element_count(output.specification));
    tv::copy_from_device_async(
        std::as_writable_bytes(std::span<std::uint8_t>(bytes)),
        *output.buffer, output.specification.binding_byte_offset,
        stream.get());
    tv::check_driver(cuStreamSynchronize(stream.get()),
                     "cuStreamSynchronize(read bool benchmark output)");
    std::vector<float> result(bytes.size());
    std::ranges::transform(
        bytes, result.begin(),
        [](std::uint8_t value) { return static_cast<float>(value); });
    return result;
  }
  const std::size_t byte_count = checked_multiply(
      storage_element_count(output.specification),
      tv::element_size(output.specification.data_type),
      "benchmark output bytes");
  std::vector<std::byte> bytes(byte_count);
  tv::copy_from_device_async(bytes, *output.buffer,
                             output.specification.binding_byte_offset,
                             stream.get());
  tv::check_driver(cuStreamSynchronize(stream.get()),
                   "cuStreamSynchronize(read benchmark output)");
  return tv::decode_floating(output.specification.data_type, bytes);
}

std::vector<float> gather(std::span<const float> physical,
                          const TensorSpec &tensor) {
  if (physical.size() != storage_element_count(tensor)) {
    throw std::invalid_argument("benchmark physical output size mismatch");
  }
  std::vector<float> result(element_count(tensor));
  for (std::size_t index = 0; index < result.size(); ++index) {
    result[index] = physical[logical_offset(index, tensor)];
  }
  return result;
}

void require_padding_unchanged(std::span<const float> physical,
                               const TensorSpec &tensor,
                               std::string_view provider) {
  std::vector<bool> occupied(physical.size(), false);
  for (std::size_t index = 0; index < element_count(tensor); ++index) {
    occupied.at(logical_offset(index, tensor)) = true;
  }
  for (std::size_t index = 0; index < physical.size(); ++index) {
    const float sentinel = tensor.data_type == FLAGDNN_DATA_BOOLEAN
                               ? 127.0F
                               : kPaddingSentinel;
    if (!occupied[index] && physical[index] != sentinel) {
      throw std::runtime_error(std::string(provider) +
                               " modified benchmark output padding");
    }
  }
}

void compare_outputs(std::span<const float> actual,
                     std::span<const float> reference,
                     const BenchmarkCase &specification) {
  if (actual.size() != reference.size()) {
    throw std::runtime_error("paired benchmark output sizes differ");
  }
  for (std::size_t index = 0; index < actual.size(); ++index) {
    const double left = actual[index];
    const double right = reference[index];
    if (left == right || (std::isnan(left) && std::isnan(right))) {
      continue;
    }
    const double absolute = std::abs(left - right);
    const double relative =
        absolute / std::max({std::abs(left), std::abs(right), 1.0e-30});
    if (std::isnan(left) != std::isnan(right) || !std::isfinite(absolute) ||
        (absolute > specification.absolute_tolerance &&
         relative > specification.relative_tolerance)) {
      std::ostringstream message;
      message << specification.name << " differs at element " << index
              << ": FlagDNN=" << left << " acDNN=" << right
              << " abs=" << absolute << " rel=" << relative;
      throw std::runtime_error(message.str());
    }
  }
}

void execute(BenchmarkExecutable &executable,
             std::span<const flagdnnBinding_t> bindings,
             tv::DeviceBuffer &workspace, tv::DeviceStream &stream) {
  executable.execute(bindings, workspace.data(), workspace.size(),
                     stream.opaque());
}

void warmup(BenchmarkExecutable &executable,
            std::span<const flagdnnBinding_t> bindings,
            tv::DeviceBuffer &workspace, tv::DeviceStream &stream,
            int iterations) {
  if (iterations < 0) {
    throw std::invalid_argument("benchmark warmup count is invalid");
  }
  for (int iteration = 0; iteration < iterations; ++iteration) {
    execute(executable, bindings, workspace, stream);
  }
  tv::check_driver(cuStreamSynchronize(stream.get()),
                   "cuStreamSynchronize(benchmark warmup)");
}

double measure(BenchmarkExecutable &executable,
               std::span<const flagdnnBinding_t> bindings,
               tv::DeviceBuffer &workspace, tv::DeviceStream &stream,
               int iterations) {
  if (iterations <= 0) {
    throw std::invalid_argument("benchmark iterations per sample is invalid");
  }
  tv::DeviceEvent start;
  tv::DeviceEvent stop;
  start.record(stream.get());
  for (int iteration = 0; iteration < iterations; ++iteration) {
    execute(executable, bindings, workspace, stream);
  }
  stop.record(stream.get());
  stop.synchronize();
  return start.elapsed_microseconds_to(stop) /
         static_cast<double>(iterations);
}

double percentile(std::span<const double> samples, double fraction) {
  if (samples.empty() || !std::isfinite(fraction) || fraction < 0.0 ||
      fraction > 1.0) {
    throw std::invalid_argument("benchmark percentile request is invalid");
  }
  std::vector<double> sorted(samples.begin(), samples.end());
  if (!std::ranges::all_of(sorted, [](double value) {
        return std::isfinite(value) && value > 0.0;
      })) {
    throw std::runtime_error("benchmark samples must be positive and finite");
  }
  std::ranges::sort(sorted);
  const std::size_t rank = std::max<std::size_t>(
      1, static_cast<std::size_t>(
             std::ceil(fraction * static_cast<double>(sorted.size()))));
  return sorted[rank - 1];
}

std::string json_escape(std::string_view input) {
  std::string result;
  for (const char character : input) {
    switch (character) {
      case '\\':
        result += "\\\\";
        break;
      case '"':
        result += "\\\"";
        break;
      case '\n':
        result += "\\n";
        break;
      case '\r':
        result += "\\r";
        break;
      case '\t':
        result += "\\t";
        break;
      default:
        result.push_back(character);
        break;
    }
  }
  return result;
}

void emit_samples(std::string_view provider,
                  const BenchmarkCase &specification,
                  std::span<const double> samples) {
  std::cout << "{\"schema_version\":1,\"kind\":\"steady_state\","
            << "\"provider\":\"" << provider << "\",\"case\":\""
            << json_escape(specification.name)
            << "\",\"unit\":\"us\",\"median\":"
            << percentile(samples, 0.5) << ",\"p90\":"
            << percentile(samples, 0.9) << ",\"samples\":[";
  for (std::size_t index = 0; index < samples.size(); ++index) {
    if (index != 0) {
      std::cout << ',';
    }
    std::cout << samples[index];
  }
  std::cout << "]}" << '\n';
}

void emit_skip(const BenchmarkCase &specification, std::string_view reason,
               std::string_view target, std::string_view operation) {
  if (reason.empty() || specification.tensors.empty()) {
    throw std::runtime_error("THead benchmark skip is malformed");
  }
  const TensorSpec &representative = specification.tensors.front();
  std::cout << "[SKIP][acdnn]"
            << " op=" << operation
            << " case=" << specification.name << " reason=" << reason
            << " sdk=" << FLAGDNN_THEAD_PPU_SDK_VERSION
            << " acdnn_header=" << ACDNN_VERSION
            << " acdnn_runtime=" << acdnnGetVersion()
            << " target=" << target
            << " dtype=" << data_type_name(representative.data_type)
            << " layout="
            << (is_contiguous(representative) ? "contiguous" : "strided")
            << " shape=" << shape_name(representative) << '\n';
}

constexpr char kIsolatedBackwardChild[] =
    "FLAGDNN_THEAD_ISOLATED_BACKWARD_BENCHMARK_CHILD";

// acDNN runtime 1400 retains backward-convolution geometry state across
// plans. A fresh process keeps each paired reference independent; all children
// share the parent's production artifact cache.
int run_isolated_backward_suite(
    char **argv, std::span<const BenchmarkCase> cases,
    std::string_view suite_name, tvb::AcdnnProvider &acdnn_provider) {
  BenchmarkCache cache;
  if (::setenv("FLAGDNN_CACHE_PATH", cache.path().c_str(), 1) != 0 ||
      ::setenv(kIsolatedBackwardChild, "1", 1) != 0) {
    throw std::runtime_error(
        "cannot configure isolated backward benchmark environment");
  }

  std::size_t comparable_executed = 0;
  std::size_t reference_skipped = 0;
  for (const BenchmarkCase &specification : cases) {
    const ProviderCapability capability =
        acdnn_provider.capability(specification);
    if (::setenv("FLAGDNN_BENCHMARK_CASE", specification.name.c_str(), 1) !=
        0) {
      throw std::runtime_error(
          "cannot select isolated backward benchmark case");
    }
    std::cout.flush();
    std::cerr.flush();
    const pid_t child = ::fork();
    if (child < 0) {
      throw std::runtime_error("fork failed for isolated backward benchmark");
    }
    if (child == 0) {
      ::execv(argv[0], argv);
      ::_exit(126);
    }

    int status = 0;
    pid_t waited = -1;
    do {
      waited = ::waitpid(child, &status, 0);
    } while (waited < 0 && errno == EINTR);
    const int expected = capability.supported ? 0 : kSkipReturnCode;
    if (waited != child || !WIFEXITED(status) ||
        WEXITSTATUS(status) != expected) {
      std::cerr << suite_name << ": FAIL isolated_case="
                << specification.name << " expected_exit=" << expected;
      if (waited == child && WIFEXITED(status)) {
        std::cerr << " actual_exit=" << WEXITSTATUS(status);
      } else if (waited == child && WIFSIGNALED(status)) {
        std::cerr << " signal=" << WTERMSIG(status);
      } else {
        std::cerr << " waitpid_error=" << errno;
      }
      std::cerr << '\n';
      return 1;
    }
    if (capability.supported) {
      ++comparable_executed;
    } else {
      ++reference_skipped;
    }
  }
  const bool all_skipped = comparable_executed == 0;
  std::cout << suite_name << ": " << (all_skipped ? "SKIP" : "PASS")
            << " cases=" << cases.size()
            << " comparable_executed=" << comparable_executed
            << " reference_skipped=" << reference_skipped << '\n';
  return all_skipped ? kSkipReturnCode : 0;
}

void run_case(const BenchmarkCase &specification,
              FlagdnnProvider &flagdnn_provider,
              tvb::AcdnnProvider &acdnn_provider,
              tv::DeviceStream &stream) {
  const BenchmarkConfig &config = specification.benchmark;
  if (config.warmup_iterations < 0 || config.sample_count <= 0 ||
      config.iterations_per_sample <= 0) {
    throw std::invalid_argument("benchmark sample configuration is invalid");
  }
  std::unique_ptr<BenchmarkExecutable> production =
      flagdnn_provider.build(specification);
  std::unique_ptr<BenchmarkExecutable> reference =
      acdnn_provider.build(specification);
  const bool is_layout =
      specification.operation == Operation::kReshape ||
      specification.operation == Operation::kTranspose ||
      specification.operation == Operation::kSlice;
  PreparedBuffers production_buffers =
      prepare_buffers(specification, stream);
  PreparedBuffers reference_buffers =
      prepare_buffers(specification, stream, is_layout);
  tv::DeviceBuffer production_workspace(production->workspace_size());
  tv::DeviceBuffer reference_workspace(reference->workspace_size());
  production->prepare(production_buffers.bindings, stream.opaque());
  reference->prepare(reference_buffers.bindings, stream.opaque());

  execute(*production, production_buffers.bindings, production_workspace,
          stream);
  execute(*reference, reference_buffers.bindings, reference_workspace, stream);
  tv::check_driver(cuStreamSynchronize(stream.get()),
                   "cuStreamSynchronize(paired correctness)");
  for (std::size_t output_index = 0;
       output_index < specification.output_count; ++output_index) {
    const TensorSpec &production_output =
        production_buffers.tensors
            .at(production_buffers.input_count + output_index)
            .specification;
    const TensorSpec &reference_output =
        reference_buffers.tensors
            .at(reference_buffers.input_count + output_index)
            .specification;
    const std::vector<float> production_physical =
        read_output(production_buffers, stream, output_index);
    const std::vector<float> reference_physical =
        read_output(reference_buffers, stream, output_index);
    require_padding_unchanged(production_physical, production_output,
                              "FlagDNN");
    require_padding_unchanged(reference_physical, reference_output,
                              "acDNN");
    const std::vector<float> production_logical =
        gather(production_physical, production_output);
    const std::vector<float> reference_logical =
        gather(reference_physical, reference_output);
    compare_outputs(production_logical, reference_logical, specification);
  }
  std::cout << specification.name
            << ": FlagDNN Graph vs acDNN correctness PASS\n";

  warmup(*production, production_buffers.bindings, production_workspace,
         stream, config.warmup_iterations);
  warmup(*reference, reference_buffers.bindings, reference_workspace, stream,
         config.warmup_iterations);
  std::vector<double> production_samples;
  std::vector<double> reference_samples;
  production_samples.reserve(static_cast<std::size_t>(config.sample_count));
  reference_samples.reserve(static_cast<std::size_t>(config.sample_count));
  for (int sample = 0; sample < config.sample_count; ++sample) {
    const auto measure_production = [&] {
      production_samples.push_back(
          measure(*production, production_buffers.bindings,
                  production_workspace, stream,
                  config.iterations_per_sample));
    };
    const auto measure_reference = [&] {
      reference_samples.push_back(
          measure(*reference, reference_buffers.bindings, reference_workspace,
                  stream, config.iterations_per_sample));
    };
    if (sample % 2 == 0) {
      measure_production();
      measure_reference();
    } else {
      measure_reference();
      measure_production();
    }
  }
  if (production_samples.size() != reference_samples.size() ||
      production_samples.size() !=
          static_cast<std::size_t>(config.sample_count)) {
    throw std::runtime_error("paired benchmark sample accounting failed");
  }

  for (std::size_t output_index = 0;
       output_index < specification.output_count; ++output_index) {
    const TensorSpec &production_output =
        production_buffers.tensors
            .at(production_buffers.input_count + output_index)
            .specification;
    const TensorSpec &reference_output =
        reference_buffers.tensors
            .at(reference_buffers.input_count + output_index)
            .specification;
    const std::vector<float> post_production = gather(
        read_output(production_buffers, stream, output_index),
        production_output);
    const std::vector<float> post_reference = gather(
        read_output(reference_buffers, stream, output_index),
        reference_output);
    compare_outputs(post_production, post_reference, specification);
  }

  emit_samples(flagdnn_provider.name(), specification, production_samples);
  emit_samples(acdnn_provider.name(), specification, reference_samples);
  std::cout << specification.name
            << ": median_us flagdnn=" << percentile(production_samples, 0.5)
            << " acdnn=" << percentile(reference_samples, 0.5)
            << " speedup=" << percentile(reference_samples, 0.5) /
                                   percentile(production_samples, 0.5)
            << '\n';
}

}  // namespace

int run_benchmark_suite(int argc, char **argv,
                        std::span<const BenchmarkCase> cases,
                        std::string_view suite_name) {
  try {
    if (argc != 3) {
      throw std::invalid_argument(
          "THead benchmark requires COMPILER_EXECUTABLE COMPILER_ENTRY");
    }
    const tv::CapabilityCatalog capability_catalog =
        tv::CapabilityCatalog::load(
            FLAGDNN_THEAD_ACDNN_CAPABILITY_CATALOG);
    capability_catalog.validate_versions(
        FLAGDNN_THEAD_PPU_SDK_VERSION, ACDNN_VERSION,
        static_cast<std::int64_t>(acdnnGetVersion()));
    const bool qualify_probes =
        std::getenv("FLAGDNN_THEAD_QUALIFY_PROBES") != nullptr;
    const std::string operation = operation_from_suite(suite_name);
    tvb::AcdnnProvider acdnn_provider(FLAGDNN_THEAD_BENCHMARK_CATALOG,
                                     operation, qualify_probes);
    const auto selected_cases = acdnn_provider.select_cases(cases);
    cases = selected_cases;
    acdnn_provider.require_exact_cases(cases);
    const char *filter = std::getenv("FLAGDNN_BENCHMARK_CASE");
    const bool backward_convolution =
        operation == "conv_dgrad" || operation == "conv_wgrad";
    if (backward_convolution &&
        (filter == nullptr || filter[0] == '\0') &&
        std::getenv(kIsolatedBackwardChild) == nullptr) {
      return run_isolated_backward_suite(argv, cases, suite_name,
                                         acdnn_provider);
    }

    tv::check_driver(cuInit(0), "cuInit");
    int device_count = 0;
    tv::check_driver(cuDeviceGetCount(&device_count), "cuDeviceGetCount");
    if (device_count <= 0) {
      throw std::runtime_error("PPU driver reported no devices");
    }
    CUdevice device = 0;
    tv::check_driver(cuDeviceGet(&device, 0), "cuDeviceGet");
    tv::PrimaryContext primary(device);
    tv::ScopedCurrentContext current(primary.get());
    tv::DeviceStream stream;
    BenchmarkCache cache;
    flagdnn::Handle handle("thead", 0);
    handle.set_compiler(argv[1], argv[2], cache.path().string());
    FlagdnnProvider flagdnn_provider(handle);
    flagdnn_provider.set_autotune(false);
    const std::string target(handle.target_fingerprint());

    std::size_t matched = 0;
    std::size_t comparable_executed = 0;
    std::size_t reference_skipped = 0;
    std::cout << std::setprecision(9);
    for (const BenchmarkCase &specification : cases) {
      if (filter != nullptr && filter[0] != '\0' &&
          specification.name != filter) {
        continue;
      }
      ++matched;
      const ProviderCapability capability =
          acdnn_provider.capability(specification);
      if (!capability.supported) {
        emit_skip(specification, capability.reason, target, operation);
        ++reference_skipped;
        continue;
      }
      run_case(specification, flagdnn_provider, acdnn_provider, stream);
      ++comparable_executed;
    }
    if (matched == 0) {
      throw std::invalid_argument(
          "FLAGDNN_BENCHMARK_CASE did not match any case");
    }
    if (matched != comparable_executed + reference_skipped) {
      throw std::runtime_error("benchmark suite accounting invariant failed");
    }
    const bool all_skipped = comparable_executed == 0;
    if (std::getenv(kIsolatedBackwardChild) == nullptr) {
      std::cout << suite_name << ": " << (all_skipped ? "SKIP" : "PASS")
                << " cases=" << matched
                << " comparable_executed=" << comparable_executed
                << " reference_skipped=" << reference_skipped << '\n';
    }
    return all_skipped ? kSkipReturnCode : 0;
  } catch (const std::exception &error) {
    std::cerr << suite_name << ": FAIL reason=" << error.what() << '\n';
    return 1;
  }
}

}  // namespace flagdnn::benchmarking
