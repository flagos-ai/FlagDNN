/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/runner.hpp"

#include "common/flagdnn_provider.hpp"
#include "common/reduction.hpp"
#include "runtime/sha256.hpp"
#include "validation/acl_runtime.hpp"
#include "validation/benchmark/aclnn_provider.hpp"
#include "validation/development_environment.hpp"
#include "validation/exact_reference.hpp"
#include "validation/gelu_contract.hpp"
#include "validation/reduction_validation.hpp"
#include "validation/tensor_io.hpp"

#include <flagdnn/flagdnn.hpp>

#ifndef FLAGDNN_ASCEND_VALIDATION_ENABLE_OPAPI_NN
#define FLAGDNN_ASCEND_VALIDATION_ENABLE_OPAPI_NN 0
#endif
#ifndef FLAGDNN_ASCEND_VALIDATION_ENABLE_MATMUL
#define FLAGDNN_ASCEND_VALIDATION_ENABLE_MATMUL 0
#endif
#ifndef FLAGDNN_ASCEND_VALIDATION_ENABLE_CONVOLUTION_FPROP
#define FLAGDNN_ASCEND_VALIDATION_ENABLE_CONVOLUTION_FPROP 0
#endif

#if FLAGDNN_ASCEND_VALIDATION_ENABLE_MATMUL
#include "validation/benchmark/aclnn_matmul_provider.hpp"
#endif
#if FLAGDNN_ASCEND_VALIDATION_ENABLE_CONVOLUTION_FPROP
#include "validation/benchmark/aclnn_convolution_provider.hpp"
#endif

#include <aclnn/acl_meta.h>
#include <aclnnop/aclnn_add.h>
#if FLAGDNN_ASCEND_VALIDATION_ENABLE_OPAPI_NN
#include <aclnnop/aclnn_sigmoid.h>
#include <aclnnop/aclnn_sigmoid_backward.h>
#endif

#include <dlfcn.h>
#include <link.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#ifndef FLAGDNN_ASCEND_VALIDATION_CANN_VERSION
#define FLAGDNN_ASCEND_VALIDATION_CANN_VERSION "unknown"
#endif
#ifndef FLAGDNN_ASCEND_VALIDATION_ASCENDCL_BUILD_ID
#define FLAGDNN_ASCEND_VALIDATION_ASCENDCL_BUILD_ID "unknown"
#endif
#ifndef FLAGDNN_ASCEND_VALIDATION_RUNTIME_BUILD_ID
#define FLAGDNN_ASCEND_VALIDATION_RUNTIME_BUILD_ID "unknown"
#endif
#ifndef FLAGDNN_ASCEND_VALIDATION_TRITON_JIT_SHA256
#define FLAGDNN_ASCEND_VALIDATION_TRITON_JIT_SHA256 "unknown"
#endif
#ifndef FLAGDNN_ASCEND_VALIDATION_NNOPBASE_SHA256
#define FLAGDNN_ASCEND_VALIDATION_NNOPBASE_SHA256 "unknown"
#endif
#ifndef FLAGDNN_ASCEND_VALIDATION_OPAPI_MATH_SHA256
#define FLAGDNN_ASCEND_VALIDATION_OPAPI_MATH_SHA256 "unknown"
#endif
#if FLAGDNN_ASCEND_VALIDATION_ENABLE_OPAPI_NN
#ifndef FLAGDNN_ASCEND_VALIDATION_OPAPI_NN_SHA256
#define FLAGDNN_ASCEND_VALIDATION_OPAPI_NN_SHA256 "unknown"
#endif
#endif

namespace flagdnn::benchmarking {
namespace {

namespace acl = flagdnn::validation::ascend;
namespace gelu = flagdnn::validation::ascend::gelu_contract;
namespace tensor_io = flagdnn::validation::ascend::tensor_io;

constexpr std::size_t kInputTailGuardBytes = 32;
constexpr std::uint8_t kInputAllocationGuard = 0xD7U;

flagdnn::testing::ReductionTestCase to_reduction_test_case(
    const BenchmarkCase& specification);

std::string read_text_file(const std::filesystem::path& path,
                           std::size_t maximum_size = 4U * 1024U * 1024U) {
  std::error_code size_error;
  const std::uintmax_t size = std::filesystem::file_size(path, size_error);
  if (size_error || size > maximum_size) {
    throw std::runtime_error("benchmark identity file is missing or too large: " +
                             path.string());
  }
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("cannot read benchmark identity file: " +
                             path.string());
  }
  std::string result((std::istreambuf_iterator<char>(input)),
                     std::istreambuf_iterator<char>());
  if (input.bad()) {
    throw std::runtime_error("cannot finish reading benchmark identity file: " +
                             path.string());
  }
  return result;
}

std::string extract_json_string(std::string_view document,
                                std::string_view key) {
  const std::string marker = "\"" + std::string(key) + "\"";
  std::size_t position = document.find(marker);
  if (position == std::string_view::npos) {
    throw std::runtime_error("benchmark identity is missing JSON key " +
                             std::string(key));
  }
  position = document.find(':', position + marker.size());
  if (position == std::string_view::npos) {
    throw std::runtime_error("benchmark identity JSON is malformed");
  }
  position = document.find('"', position + 1);
  if (position == std::string_view::npos) {
    throw std::runtime_error("benchmark identity JSON string is malformed");
  }
  ++position;
  std::string result;
  bool escaped = false;
  for (; position < document.size(); ++position) {
    const char character = document[position];
    if (escaped) {
      switch (character) {
        case '"':
        case '\\':
        case '/':
          result.push_back(character);
          break;
        case 'b':
          result.push_back('\b');
          break;
        case 'f':
          result.push_back('\f');
          break;
        case 'n':
          result.push_back('\n');
          break;
        case 'r':
          result.push_back('\r');
          break;
        case 't':
          result.push_back('\t');
          break;
        default:
          throw std::runtime_error(
              "benchmark identity contains unsupported JSON escape");
      }
      escaped = false;
    } else if (character == '\\') {
      escaped = true;
    } else if (character == '"') {
      if (result.empty()) {
        throw std::runtime_error("benchmark identity JSON string is empty");
      }
      return result;
    } else {
      result.push_back(character);
    }
  }
  throw std::runtime_error("benchmark identity JSON string is unterminated");
}

struct FlagdnnArtifactIdentity {
  std::string compiler_identity_sha256;
  std::string artifact_request_sha256;
  std::string launch_abi;
  std::string selected_candidate;
};

FlagdnnArtifactIdentity find_artifact_identity(
    const std::filesystem::path& cache,
    std::string_view case_name) {
  const std::string graph_marker =
      "\"graph\":{\"name\":\"" + std::string(case_name) + "\"";
  std::filesystem::path artifact_directory;
  std::error_code error;
  for (std::filesystem::recursive_directory_iterator iterator(cache, error), end;
       iterator != end && !error;
       iterator.increment(error)) {
    if (!iterator->is_regular_file() ||
        iterator->path().filename() != "request.json") {
      continue;
    }
    const std::string request = read_text_file(iterator->path());
    if (request.find(graph_marker) == std::string::npos) {
      continue;
    }
    if (!artifact_directory.empty()) {
      throw std::runtime_error(
          "multiple cached artifacts match benchmark case " +
          std::string(case_name));
    }
    artifact_directory = iterator->path().parent_path();
  }
  if (error) {
    throw std::runtime_error("cannot scan benchmark artifact cache: " +
                             error.message());
  }
  if (artifact_directory.empty()) {
    throw std::runtime_error("cannot find cached artifact for benchmark case " +
                             std::string(case_name));
  }

  const std::string manifest =
      read_text_file(artifact_directory / "manifest.json");
  FlagdnnArtifactIdentity result;
  result.compiler_identity_sha256 =
      extract_json_string(manifest, "identity_sha256");
  result.artifact_request_sha256 =
      extract_json_string(manifest, "request_sha256");
  result.launch_abi = extract_json_string(manifest, "launch_abi");
  result.selected_candidate = extract_json_string(manifest, "candidate_id");
  return result;
}

using CacheSnapshot = std::map<std::string, std::string>;

CacheSnapshot snapshot_cache(const std::filesystem::path& cache) {
  CacheSnapshot result;
  std::error_code error;
  for (std::filesystem::recursive_directory_iterator iterator(cache, error), end;
       iterator != end && !error;
       iterator.increment(error)) {
    if (!iterator->is_regular_file()) {
      continue;
    }
    const std::filesystem::path relative =
        std::filesystem::relative(iterator->path(), cache, error);
    if (error) {
      break;
    }
    result.emplace(relative.generic_string(),
                   flagdnn::native::sha256_file(iterator->path()));
  }
  if (error) {
    throw std::runtime_error("cannot snapshot benchmark cache: " +
                             error.message());
  }
  return result;
}

std::vector<float> make_input(const TensorSpec& tensor,
                              std::size_t input_index,
                              InputDomain domain) {
  std::vector<float> result(tensor_io::element_count(tensor));
  for (std::size_t index = 0; index < result.size(); ++index) {
    const int centered =
        static_cast<int>((index * 17 + input_index * 11) % 41) - 20;
    const float real_value =
        static_cast<float>(centered) / static_cast<float>(13 + input_index);
    switch (domain) {
      case InputDomain::kReal:
        result[index] = real_value;
        break;
      case InputDomain::kScaled:
        result[index] = real_value * 4.0F;
        break;
      case InputDomain::kPositive:
        result[index] = std::abs(real_value) + 0.5F;
        break;
      case InputDomain::kDivisor:
      case InputDomain::kModulo:
        result[index] = input_index == 1
                            ? std::abs(real_value) + 0.5F
                            : real_value;
        break;
      case InputDomain::kPower:
        result[index] =
            input_index == 0
                ? std::abs(real_value) + 0.5F
                : std::fmod(std::abs(real_value), 2.0F) + 0.125F;
        break;
      case InputDomain::kModuloSigned: {
        constexpr std::array<float, 6> kLeft = {
            -3.0F, -3.0F, 3.0F, 3.0F, -5.5F, 5.5F};
        constexpr std::array<float, 6> kRight = {
            2.0F, -2.0F, 2.0F, -2.0F, 2.25F, -2.25F};
        result[index] = input_index == 0
                            ? kLeft[index % kLeft.size()]
                            : kRight[index % kRight.size()];
        break;
      }
      case InputDomain::kTan:
        result[index] = static_cast<float>(centered) / 40.0F;
        break;
      case InputDomain::kLogical:
        result[index] = input_index == 0
                            ? static_cast<float>((index / 2U) % 2U)
                            : static_cast<float>(index % 2U);
        break;
      case InputDomain::kComparison: {
        constexpr std::array<float, 8> kLeft = {
            0.0F, -0.0F, -3.0F, 4.0F, 5.0F, -7.0F, 2.0F, 9.0F};
        constexpr std::array<float, 8> kRight = {
            -0.0F, 0.0F, -2.0F, 3.0F, 6.0F, -7.0F, 2.0F, 10.0F};
        result[index] = input_index == 0
                            ? kLeft[index % kLeft.size()]
                            : kRight[index % kRight.size()];
        break;
      }
      default:
        throw std::invalid_argument(
            "Ascend pointwise benchmark input domain is unsupported");
    }
  }
  return result;
}

struct InputBuffer {
  std::unique_ptr<acl::DeviceBuffer> device;
  std::vector<float> logical;
  std::vector<std::uint8_t> initial;
};

InputBuffer make_input_buffer_from_logical(
    const TensorSpec& tensor,
    std::span<const float> logical,
    acl::Stream& stream) {
  const std::vector<float> physical = tensor_io::scatter(logical, tensor);
  const std::vector<std::uint8_t> encoded =
      tensor_io::encode(physical, tensor.data_type);
  const std::size_t allocation = tensor_io::checked_add(
      tensor_io::allocation_byte_count(tensor),
      kInputTailGuardBytes,
      "benchmark guarded input allocation");
  std::vector<std::uint8_t> initial(allocation, kInputAllocationGuard);
  std::copy(encoded.begin(),
            encoded.end(),
            initial.begin() +
                static_cast<std::ptrdiff_t>(tensor.binding_byte_offset));
  auto device = std::make_unique<acl::DeviceBuffer>(allocation);
  device->copy_from_host_at(initial.data(), initial.size(), 0, stream.get());
  return {std::move(device),
          std::vector<float>(logical.begin(), logical.end()),
          std::move(initial)};
}

InputBuffer make_input_buffer(const TensorSpec& tensor,
                              std::size_t input_index,
                              InputDomain domain,
                              acl::Stream& stream) {
  const std::vector<float> generated =
      make_input(tensor, input_index, domain);
  const std::vector<float> logical =
      tensor_io::quantize(generated, tensor.data_type);
  return make_input_buffer_from_logical(tensor, logical, stream);
}

InputBuffer make_reduction_input_buffer(
    const BenchmarkCase& specification,
    acl::Stream& stream) {
  const flagdnn::testing::ReductionTestCase test_case =
      to_reduction_test_case(specification);
  const TensorSpec& tensor = specification.tensors[0];
  const std::vector<float> logical =
      flagdnn::testing::make_reduction_input(test_case);
  return make_input_buffer_from_logical(tensor, logical, stream);
}

void require_input_unchanged(std::string_view provider,
                             const InputBuffer& input,
                             acl::Stream& stream) {
  std::vector<std::uint8_t> actual(input.initial.size());
  input.device->copy_to_host_at(
      actual.data(), actual.size(), 0, stream.get());
  stream.synchronize();
  if (actual != input.initial) {
    throw std::runtime_error(std::string(provider) +
                             " modified benchmark input storage or guards");
  }
}

std::unique_ptr<acl::DeviceBuffer> make_output_buffer(
    const TensorSpec& tensor,
    acl::Stream& stream) {
  const std::vector<float> initial(tensor_io::storage_element_count(tensor),
                                   tensor_io::kPaddingSentinel);
  const std::vector<std::uint8_t> encoded =
      tensor_io::encode(initial, tensor.data_type);
  auto result =
      std::make_unique<acl::DeviceBuffer>(tensor_io::allocation_byte_count(tensor));
  result->copy_from_host_at(encoded.data(),
                            encoded.size(),
                            tensor.binding_byte_offset,
                            stream.get());
  return result;
}

std::vector<float> read_output(const acl::DeviceBuffer& buffer,
                               const TensorSpec& tensor,
                               acl::Stream& stream,
                               std::string_view provider) {
  std::vector<std::uint8_t> encoded(tensor_io::encoded_byte_count(tensor));
  buffer.copy_to_host_at(encoded.data(),
                         encoded.size(),
                         tensor.binding_byte_offset,
                         stream.get());
  stream.synchronize();
  return tensor_io::decode_storage(provider, encoded, tensor);
}

struct PointwiseReferencePlan {
  std::vector<TensorSpec> inputs;
  TensorSpec output;
};

flagdnn::testing::TestTensor to_reduction_test_tensor(
    const TensorSpec& tensor) {
  return {tensor.uid,
          tensor.data_type,
          tensor.dimensions,
          tensor.strides,
          tensor.binding_byte_offset};
}

flagdnn::testing::ReductionTestCase to_reduction_test_case(
    const BenchmarkCase& specification) {
  const AclnnReductionBenchmarkPlan plan =
      plan_aclnn_reduction(specification);
  flagdnn::testing::ReductionTestCase result;
  result.name = specification.name;
  result.input = to_reduction_test_tensor(specification.tensors[0]);
  result.output = to_reduction_test_tensor(specification.tensors[1]);
  result.mode = plan.mode;
  result.axis = specification.reduction_axis;
  result.keep_dimensions = plan.keep_dimensions;
  result.absolute_tolerance = specification.absolute_tolerance;
  result.relative_tolerance = specification.relative_tolerance;
  flagdnn::testing::validate_reduction_case(result);
  return result;
}

PointwiseReferencePlan make_reference_plan(
    const BenchmarkCase& specification) {
#if FLAGDNN_ASCEND_VALIDATION_ENABLE_CONVOLUTION_FPROP
  if (specification.operation == Operation::kConvolutionFprop) {
    const AclnnConvolutionFpropBenchmarkPlan plan =
        plan_aclnn_convolution_fprop_benchmark(specification);
    return {{plan.input, plan.filter}, plan.output};
  }
#endif
#if FLAGDNN_ASCEND_VALIDATION_ENABLE_MATMUL
  if (specification.operation == Operation::kMatmul) {
    const AclnnMatmulBenchmarkPlan plan =
        plan_aclnn_matmul_benchmark(specification);
    return {{plan.a, plan.b}, plan.output};
  }
#endif
  if (specification.operation == Operation::kReduction) {
    const AclnnReductionBenchmarkPlan plan =
        plan_aclnn_reduction(specification);
    return {{plan.input}, plan.output};
  }
  if (is_aclnn_unary_benchmark(specification)) {
    const AclnnUnaryPointwiseBenchmarkPlan plan =
        plan_aclnn_unary_pointwise(specification);
    return {{plan.input}, plan.output};
  }
  if (specification.operation == Operation::kPointwise &&
      specification.pointwise_mode == FLAGDNN_POINTWISE_BINARY_SELECT) {
    const AclnnTernaryPointwiseBenchmarkPlan plan =
        plan_aclnn_ternary_pointwise(specification);
    return {{plan.self, plan.other, plan.condition}, plan.output};
  }
  const AclnnBinaryPointwiseBenchmarkPlan plan =
      plan_aclnn_binary_pointwise(specification);
  return {{plan.left, plan.right}, plan.output};
}

bool is_neural_activation_mode(flagdnnPointwiseMode_t mode) noexcept {
  return mode == FLAGDNN_POINTWISE_SIGMOID_FWD ||
         mode == FLAGDNN_POINTWISE_ELU_FWD ||
         mode == FLAGDNN_POINTWISE_GELU_FWD ||
         mode == FLAGDNN_POINTWISE_SOFTPLUS_FWD ||
         mode == FLAGDNN_POINTWISE_SWISH_FWD ||
         mode == FLAGDNN_POINTWISE_GELU_APPROX_TANH_FWD;
}

bool is_gelu_mode(flagdnnPointwiseMode_t mode) noexcept {
  return mode == FLAGDNN_POINTWISE_GELU_FWD ||
         mode == FLAGDNN_POINTWISE_GELU_APPROX_TANH_FWD;
}

bool is_special_math_mode(flagdnnPointwiseMode_t mode) noexcept {
  return mode == FLAGDNN_POINTWISE_SIN ||
         mode == FLAGDNN_POINTWISE_COS ||
         mode == FLAGDNN_POINTWISE_TAN ||
         mode == FLAGDNN_POINTWISE_ERF;
}

bool is_logical_mode(flagdnnPointwiseMode_t mode) noexcept {
  return mode == FLAGDNN_POINTWISE_LOGICAL_NOT ||
         mode == FLAGDNN_POINTWISE_LOGICAL_AND ||
         mode == FLAGDNN_POINTWISE_LOGICAL_OR;
}

bool is_comparison_mode(flagdnnPointwiseMode_t mode) noexcept {
  return mode == FLAGDNN_POINTWISE_CMP_EQ ||
         mode == FLAGDNN_POINTWISE_CMP_NEQ ||
         mode == FLAGDNN_POINTWISE_CMP_GT ||
         mode == FLAGDNN_POINTWISE_CMP_GE ||
         mode == FLAGDNN_POINTWISE_CMP_LT ||
         mode == FLAGDNN_POINTWISE_CMP_LE;
}

bool requires_unary_benchmark_extension(
    flagdnnPointwiseMode_t mode) noexcept {
  return is_neural_activation_mode(mode) || is_special_math_mode(mode);
}

bool requires_repeatable_reference_contract(
    flagdnnPointwiseMode_t mode) noexcept {
  return is_gelu_mode(mode) || is_special_math_mode(mode) ||
         mode == FLAGDNN_POINTWISE_MOD ||
         mode == FLAGDNN_POINTWISE_POW ||
         mode == FLAGDNN_POINTWISE_SIGMOID_BWD ||
         is_comparison_mode(mode) ||
         mode == FLAGDNN_POINTWISE_LOGICAL_NOT ||
         mode == FLAGDNN_POINTWISE_LOGICAL_AND ||
         mode == FLAGDNN_POINTWISE_LOGICAL_OR ||
         mode == FLAGDNN_POINTWISE_BINARY_SELECT;
}

bool requires_binary_benchmark_extension(
    flagdnnPointwiseMode_t mode) noexcept {
  return mode == FLAGDNN_POINTWISE_MOD ||
         mode == FLAGDNN_POINTWISE_POW ||
         mode == FLAGDNN_POINTWISE_SIGMOID_BWD ||
         is_comparison_mode(mode);
}

bool requires_ternary_benchmark_extension(
    flagdnnPointwiseMode_t mode) noexcept {
  return mode == FLAGDNN_POINTWISE_BINARY_SELECT;
}

std::string unary_extension_name(flagdnnPointwiseMode_t mode) {
  switch (mode) {
    case FLAGDNN_POINTWISE_COS:
      return "cos";
    case FLAGDNN_POINTWISE_ERF:
      return "erf";
    case FLAGDNN_POINTWISE_SIN:
      return "sin";
    case FLAGDNN_POINTWISE_TAN:
      return "tan";
    case FLAGDNN_POINTWISE_SIGMOID_FWD:
      return "sigmoid";
    case FLAGDNN_POINTWISE_ELU_FWD:
      return "elu";
    case FLAGDNN_POINTWISE_GELU_FWD:
      return "gelu";
    case FLAGDNN_POINTWISE_SOFTPLUS_FWD:
      return "softplus";
    case FLAGDNN_POINTWISE_SWISH_FWD:
      return "swish";
    case FLAGDNN_POINTWISE_GELU_APPROX_TANH_FWD:
      return "gelu_approx_tanh";
    default:
      throw std::invalid_argument(
          "Ascend unary benchmark extension mode is invalid");
  }
}

std::string comparison_extension_name(flagdnnPointwiseMode_t mode) {
  switch (mode) {
    case FLAGDNN_POINTWISE_CMP_EQ:
      return "cmp_eq";
    case FLAGDNN_POINTWISE_CMP_NEQ:
      return "cmp_neq";
    case FLAGDNN_POINTWISE_CMP_GT:
      return "cmp_gt";
    case FLAGDNN_POINTWISE_CMP_GE:
      return "cmp_ge";
    case FLAGDNN_POINTWISE_CMP_LT:
      return "cmp_lt";
    case FLAGDNN_POINTWISE_CMP_LE:
      return "cmp_le";
    default:
      throw std::invalid_argument(
          "Ascend comparison benchmark extension mode is invalid");
  }
}

BenchmarkCase make_unary_benchmark_extension(
    const BenchmarkCase& prototype) {
  if (prototype.operation != Operation::kPointwise ||
      !requires_unary_benchmark_extension(prototype.pointwise_mode)) {
    throw std::invalid_argument(
        "Ascend unary benchmark extension prototype is invalid");
  }
  BenchmarkCase result;
  result.name = unary_extension_name(prototype.pointwise_mode) +
                "_ascend_strided_padding_perf_fp32";
  result.operation = Operation::kPointwise;
  result.pointwise_mode = prototype.pointwise_mode;
  result.pointwise_attributes = prototype.pointwise_attributes;
  if (result.pointwise_mode == FLAGDNN_POINTWISE_ELU_FWD) {
    result.name = "elu_ascend_nondefault_alpha_strided_padding_perf_fp32";
    result.pointwise_attributes = FLAGDNN_POINTWISE_ATTRIBUTES_INITIALIZER;
    result.pointwise_attributes.flags =
        FLAGDNN_POINTWISE_ATTRIBUTE_ELU_ALPHA;
    result.pointwise_attributes.elu_alpha = 0.375;
  } else if (result.pointwise_mode == FLAGDNN_POINTWISE_SOFTPLUS_FWD) {
    result.name =
        "softplus_ascend_nondefault_beta_strided_padding_perf_fp32";
    result.pointwise_attributes = FLAGDNN_POINTWISE_ATTRIBUTES_INITIALIZER;
    result.pointwise_attributes.flags =
        FLAGDNN_POINTWISE_ATTRIBUTE_SOFTPLUS_BETA;
    result.pointwise_attributes.softplus_beta = 1.75;
  } else if (is_gelu_mode(result.pointwise_mode)) {
    result.name = unary_extension_name(result.pointwise_mode) +
                  "_ascend_scaled_strided_padding_perf_fp32";
  }
  result.input_domain = is_gelu_mode(result.pointwise_mode)
                            ? InputDomain::kScaled
                            : prototype.input_domain;
  result.tensors = {
      strided_tensor(65000,
                     {2, 3, 4},
                     {31, 9, 1},
                     FLAGDNN_DATA_FLOAT32),
      strided_tensor(65001,
                     {2, 3, 4},
                     {37, 11, 2},
                     FLAGDNN_DATA_FLOAT32),
  };
  result.absolute_tolerance = prototype.absolute_tolerance;
  result.relative_tolerance = prototype.relative_tolerance;
  if (is_gelu_mode(result.pointwise_mode)) {
    result.absolute_tolerance = gelu::kFlagdnnAbsoluteTolerance;
    result.relative_tolerance = gelu::kFlagdnnRelativeTolerance;
  }
  result.benchmark = prototype.benchmark;
  return result;
}

BenchmarkCase make_binary_benchmark_extension(
    const BenchmarkCase& prototype) {
  if (prototype.operation != Operation::kPointwise ||
      !requires_binary_benchmark_extension(prototype.pointwise_mode)) {
    throw std::invalid_argument(
        "Ascend binary benchmark extension prototype is invalid");
  }
  BenchmarkCase result;
  result.operation = Operation::kPointwise;
  result.pointwise_mode = prototype.pointwise_mode;
  result.pointwise_attributes = prototype.pointwise_attributes;
  result.tensors = {
      strided_tensor(66000,
                     {2, 3, 4},
                     {31, 9, 1},
                     FLAGDNN_DATA_FLOAT32),
      strided_tensor(66001,
                     {1, 3, 1},
                     {7, 2, 1},
                     FLAGDNN_DATA_FLOAT32),
      strided_tensor(66002,
                     {2, 3, 4},
                     {37, 11, 2},
                     FLAGDNN_DATA_FLOAT32),
  };
  if (is_comparison_mode(result.pointwise_mode)) {
    result.name = comparison_extension_name(result.pointwise_mode) +
                  "_ascend_strided_broadcast_offsets_bool_"
                  "padding_perf_fp32";
    result.input_domain = InputDomain::kComparison;
    result.tensors[0].binding_byte_offset = 32;
    result.tensors[1].binding_byte_offset = 64;
    result.tensors[2] =
        strided_tensor(66002,
                       {2, 3, 4},
                       {37, 11, 2},
                       FLAGDNN_DATA_BOOLEAN);
    result.tensors[2].binding_byte_offset = 96;
    result.absolute_tolerance = 0.0;
    result.relative_tolerance = 0.0;
  } else if (result.pointwise_mode == FLAGDNN_POINTWISE_SIGMOID_BWD) {
    result.name =
        "sigmoid_backward_ascend_equal_shape_strided_padding_perf_fp32";
    result.tensors[1] =
        strided_tensor(66001,
                       {2, 3, 4},
                       {35, 10, 1},
                       FLAGDNN_DATA_FLOAT32);
    result.input_domains = {InputDomain::kReal, InputDomain::kReal};
  } else if (result.pointwise_mode == FLAGDNN_POINTWISE_MOD) {
    result.name =
        "mod_ascend_signed_strided_broadcast_padding_perf_fp32";
    result.input_domains = {InputDomain::kModuloSigned,
                            InputDomain::kModuloSigned};
  } else {
    result.name = "pow_ascend_safe_strided_broadcast_padding_perf_fp32";
    result.input_domains = {InputDomain::kPower, InputDomain::kPower};
  }
  result.absolute_tolerance = prototype.absolute_tolerance;
  result.relative_tolerance = prototype.relative_tolerance;
  result.benchmark = prototype.benchmark;
  return result;
}

BenchmarkCase make_binary_select_benchmark_extension(
    const BenchmarkCase& prototype) {
  if (prototype.operation != Operation::kPointwise ||
      prototype.pointwise_mode != FLAGDNN_POINTWISE_BINARY_SELECT) {
    throw std::invalid_argument(
        "Ascend binary-select benchmark extension prototype is invalid");
  }
  BenchmarkCase result;
  result.name =
      "binary_select_ascend_independent_broadcast_strided_offsets_"
      "bool_padding_perf_fp32";
  result.operation = Operation::kPointwise;
  result.pointwise_mode = FLAGDNN_POINTWISE_BINARY_SELECT;
  result.tensors = {
      strided_tensor(69000,
                     {2, 1, 8},
                     {37, 17, 2},
                     FLAGDNN_DATA_FLOAT32),
      strided_tensor(69001,
                     {1, 3, 1},
                     {19, 5, 1},
                     FLAGDNN_DATA_FLOAT32),
      strided_tensor(69002,
                     {1, 3, 8},
                     {41, 11, 1},
                     FLAGDNN_DATA_BOOLEAN),
      strided_tensor(69003,
                     {2, 3, 8},
                     {97, 29, 2},
                     FLAGDNN_DATA_FLOAT32),
  };
  result.tensors[0].binding_byte_offset = 32;
  result.tensors[1].binding_byte_offset = 64;
  result.tensors[2].binding_byte_offset = 96;
  result.tensors[3].binding_byte_offset = 128;
  result.input_domain = InputDomain::kReal;
  result.input_domains = {
      InputDomain::kReal, InputDomain::kReal, InputDomain::kLogical};
  result.absolute_tolerance = 0.0;
  result.relative_tolerance = 0.0;
  result.benchmark = prototype.benchmark;
  return result;
}

struct CaseBuffers {
  std::vector<InputBuffer> inputs;
  std::vector<InputBuffer> reference_inputs;
  std::unique_ptr<acl::DeviceBuffer> flagdnn_output;
  std::unique_ptr<acl::DeviceBuffer> aclnn_output;
  std::vector<flagdnnBinding_t> flagdnn_bindings;
  std::vector<flagdnnBinding_t> aclnn_bindings;
};

class ExecutableLifetime {
 public:
  ExecutableLifetime(acl::Stream& stream,
                     std::unique_ptr<BenchmarkExecutable>& flagdnn,
                     std::unique_ptr<BenchmarkExecutable>& aclnn)
      : stream_(stream), flagdnn_(flagdnn), aclnn_(aclnn) {}

  ~ExecutableLifetime() {
    try {
      stream_.synchronize();
    } catch (const std::exception& error) {
      std::cerr << "Ascend benchmark cleanup synchronization failed: "
                << error.what() << '\n';
    }
    aclnn_.reset();
    flagdnn_.reset();
  }

  ExecutableLifetime(const ExecutableLifetime&) = delete;
  ExecutableLifetime& operator=(const ExecutableLifetime&) = delete;

 private:
  acl::Stream& stream_;
  std::unique_ptr<BenchmarkExecutable>& flagdnn_;
  std::unique_ptr<BenchmarkExecutable>& aclnn_;
};

CaseBuffers make_buffers(const BenchmarkCase& specification,
                         const PointwiseReferencePlan& plan,
                         acl::Stream& stream) {
  CaseBuffers result;
  const std::size_t input_count = input_tensor_count(specification);
  if (input_count != plan.inputs.size() ||
      (input_count != 1 && input_count != 2 && input_count != 3)) {
    throw std::invalid_argument(
        "Ascend pointwise benchmark input count is invalid");
  }
  const auto input_domain = [&](std::size_t index) {
    if (specification.input_domains.empty()) {
      return specification.input_domain;
    }
    if (specification.input_domains.size() != input_count) {
      throw std::invalid_argument(
          "Ascend pointwise benchmark input domain count is invalid");
    }
    return specification.input_domains[index];
  };
  result.inputs.reserve(input_count);
  result.reference_inputs.reserve(input_count);
  if (specification.operation == Operation::kReduction) {
    if (input_count != 1) {
      throw std::invalid_argument(
          "Ascend reduction benchmark input count is invalid");
    }
    result.inputs.push_back(
        make_reduction_input_buffer(specification, stream));
  } else {
    for (std::size_t index = 0; index < input_count; ++index) {
      result.inputs.push_back(make_input_buffer(
          specification.tensors[index], index, input_domain(index), stream));
    }
  }
  for (std::size_t index = 0; index < input_count; ++index) {
    result.reference_inputs.push_back(make_input_buffer_from_logical(
        plan.inputs[index], result.inputs[index].logical, stream));
  }
  const TensorSpec& flagdnn_output_spec = output_tensor(specification);
  result.flagdnn_output = make_output_buffer(flagdnn_output_spec, stream);
  result.aclnn_output = make_output_buffer(plan.output, stream);
  result.flagdnn_bindings.reserve(input_count + 1);
  result.aclnn_bindings.reserve(input_count + 1);
  for (std::size_t index = 0; index < input_count; ++index) {
    result.flagdnn_bindings.push_back(
        {specification.tensors[index].uid,
         result.inputs[index].device->opaque_at(
             specification.tensors[index].binding_byte_offset)});
    result.aclnn_bindings.push_back(
        {plan.inputs[index].uid,
         result.reference_inputs[index].device->opaque_at(
             plan.inputs[index].binding_byte_offset)});
  }
  result.flagdnn_bindings.push_back(
      {flagdnn_output_spec.uid,
       result.flagdnn_output->opaque_at(
           flagdnn_output_spec.binding_byte_offset)});
  result.aclnn_bindings.push_back(
      {plan.output.uid,
       result.aclnn_output->opaque_at(plan.output.binding_byte_offset)});
  return result;
}

void require_case_inputs_unchanged(const CaseBuffers& buffers,
                                   acl::Stream& stream) {
  if (buffers.inputs.size() != buffers.reference_inputs.size()) {
    throw std::logic_error("benchmark provider input buffer count differs");
  }
  for (std::size_t index = 0; index < buffers.inputs.size(); ++index) {
    require_input_unchanged("FlagDNN", buffers.inputs[index], stream);
    require_input_unchanged("ACLNN", buffers.reference_inputs[index], stream);
  }
}

void execute(BenchmarkExecutable& executable,
             std::span<const flagdnnBinding_t> bindings,
             acl::DeviceBuffer& workspace,
             acl::Stream& stream) {
  executable.execute(bindings,
                     executable.workspace_size() == 0
                         ? nullptr
                         : workspace.opaque(),
                     executable.workspace_size(),
                     stream.opaque());
}

double percentile(std::vector<double> values, double fraction) {
  if (values.empty()) {
    throw std::invalid_argument("cannot summarize empty benchmark samples");
  }
  std::sort(values.begin(), values.end());
  const std::size_t index = static_cast<std::size_t>(
      std::ceil(fraction * static_cast<double>(values.size()))) -
      1;
  return values[std::min(index, values.size() - 1)];
}

struct MetricSamples {
  std::vector<double> stream_us;
  std::vector<double> submit_us;
  std::vector<double> end_to_end_us;

  void push(const acl::TimingSample& sample) {
    stream_us.push_back(sample.stream_us);
    submit_us.push_back(sample.submit_us);
    end_to_end_us.push_back(sample.end_to_end_us);
  }
};

std::string json_escape(std::string_view input) {
  std::string result;
  for (const char character : input) {
    switch (character) {
      case '"':
        result += "\\\"";
        break;
      case '\\':
        result += "\\\\";
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
        if (static_cast<unsigned char>(character) < 0x20U) {
          throw std::invalid_argument("benchmark JSON string has control byte");
        }
        result.push_back(character);
        break;
    }
  }
  return result;
}

void emit_metric(std::string_view name, const std::vector<double>& samples) {
  std::cout << "\"" << name << "\":{\"median\":"
            << percentile(samples, 0.5) << ",\"p90\":"
            << percentile(samples, 0.9) << ",\"samples\":[";
  for (std::size_t index = 0; index < samples.size(); ++index) {
    if (index != 0) {
      std::cout << ',';
    }
    std::cout << samples[index];
  }
  std::cout << "]}";
}

struct EnvironmentIdentity {
  std::string soc_fingerprint;
  std::string cann_package_version;
  std::string ascendcl_build_id;
  std::string runtime_build_id;
};

struct AclnnLibraryIdentity {
  std::string libnnopbase_sha256;
  std::string libopapi_math_sha256;
#if FLAGDNN_ASCEND_VALIDATION_ENABLE_OPAPI_NN
  std::string libopapi_nn_sha256;
#endif
};

template <typename Function>
std::string loaded_symbol_object_sha256(Function* symbol,
                                        std::string_view expected_name) {
  Dl_info information{};
  if (dladdr(reinterpret_cast<void*>(symbol), &information) == 0 ||
      information.dli_fname == nullptr) {
    throw std::runtime_error("cannot resolve loaded object for " +
                             std::string(expected_name));
  }
  std::error_code error;
  const std::filesystem::path path =
      std::filesystem::canonical(information.dli_fname, error);
  if (error || path.filename().string().find(expected_name) ==
                   std::string::npos) {
    throw std::runtime_error("loaded ACLNN object identity is invalid for " +
                             std::string(expected_name));
  }
  return flagdnn::native::sha256_file(path);
}

struct LoadedObjectQuery {
  std::string_view expected_name;
  std::vector<std::filesystem::path> matches;
};

int collect_loaded_object(dl_phdr_info* information,
                          std::size_t,
                          void* opaque) {
  auto& query = *static_cast<LoadedObjectQuery*>(opaque);
  if (information == nullptr || information->dlpi_name == nullptr ||
      information->dlpi_name[0] == '\0') {
    return 0;
  }
  std::error_code error;
  const std::filesystem::path path =
      std::filesystem::canonical(information->dlpi_name, error);
  if (error || path.filename().string().find(query.expected_name) ==
                   std::string::npos) {
    return 0;
  }
  if (std::find(query.matches.begin(), query.matches.end(), path) ==
      query.matches.end()) {
    query.matches.push_back(path);
  }
  return 0;
}

std::string loaded_named_object_sha256(std::string_view expected_name) {
  LoadedObjectQuery query{expected_name, {}};
  (void)dl_iterate_phdr(&collect_loaded_object, &query);
  if (query.matches.size() != 1) {
    throw std::runtime_error(
        "expected exactly one loaded object matching " +
        std::string(expected_name) + ", found " +
        std::to_string(query.matches.size()));
  }
  return flagdnn::native::sha256_file(query.matches.front());
}

AclnnLibraryIdentity aclnn_library_identity() {
  AclnnLibraryIdentity result;
  result.libnnopbase_sha256 =
      loaded_symbol_object_sha256(&aclCreateTensor, "libnnopbase.so");
  result.libopapi_math_sha256 =
      loaded_symbol_object_sha256(&aclnnAdd, "libopapi_math.so");
#if FLAGDNN_ASCEND_VALIDATION_ENABLE_OPAPI_NN
  const std::string sigmoid_object_sha256 =
      loaded_symbol_object_sha256(&aclnnSigmoid, "libopapi_nn.so");
  result.libopapi_nn_sha256 = loaded_symbol_object_sha256(
      &aclnnSigmoidBackward, "libopapi_nn.so");
  if (sigmoid_object_sha256 != result.libopapi_nn_sha256) {
    throw std::runtime_error(
        "ACLNN sigmoid forward/backward symbols resolve to different DSOs");
  }
#endif
  bool identity_mismatch =
      result.libnnopbase_sha256 !=
          FLAGDNN_ASCEND_VALIDATION_NNOPBASE_SHA256 ||
      result.libopapi_math_sha256 !=
          FLAGDNN_ASCEND_VALIDATION_OPAPI_MATH_SHA256;
#if FLAGDNN_ASCEND_VALIDATION_ENABLE_OPAPI_NN
  identity_mismatch =
      identity_mismatch ||
      result.libopapi_nn_sha256 !=
          FLAGDNN_ASCEND_VALIDATION_OPAPI_NN_SHA256;
#endif
  if (identity_mismatch) {
    throw std::runtime_error(
        "loaded ACLNN libraries differ from configured validation libraries");
  }
  return result;
}

void emit_common_prefix(std::string_view provider,
                        const BenchmarkCase& specification,
                        const EnvironmentIdentity& environment) {
  std::cout << "{\"schema_version\":2,\"kind\":\"steady_state\","
            << "\"provider\":\"" << json_escape(provider) << "\","
            << "\"case\":\"" << json_escape(specification.name) << "\","
            << "\"environment\":{"
            << "\"soc_fingerprint\":\""
            << json_escape(environment.soc_fingerprint) << "\","
            << "\"cann_package_version\":\""
            << json_escape(environment.cann_package_version) << "\","
            << "\"ascendcl_build_id\":\""
            << json_escape(environment.ascendcl_build_id) << "\","
            << "\"runtime_build_id\":\""
            << json_escape(environment.runtime_build_id) << "\"},";
}

void emit_config(const BenchmarkConfig& config) {
  std::cout << "\"benchmark_config\":{"
            << "\"warmup_iterations\":" << config.warmup_iterations << ','
            << "\"sample_count\":" << config.sample_count << ','
            << "\"iterations_per_sample\":"
            << config.iterations_per_sample << "},";
}

void emit_flagdnn_record(const BenchmarkCase& specification,
                         const EnvironmentIdentity& environment,
                         const FlagdnnArtifactIdentity& identity,
                         std::string_view libtriton_jit_sha256,
                         const MetricSamples& samples) {
  emit_common_prefix("flagdnn", specification, environment);
  std::cout << "\"provider_identity\":{"
            << "\"libtriton_jit_sha256\":\""
            << json_escape(libtriton_jit_sha256) << "\","
            << "\"compiler_identity_sha256\":\""
            << json_escape(identity.compiler_identity_sha256) << "\","
            << "\"artifact_request_sha256\":\""
            << json_escape(identity.artifact_request_sha256) << "\","
            << "\"launch_abi\":\"" << json_escape(identity.launch_abi)
            << "\",\"selected_candidate\":\""
            << json_escape(identity.selected_candidate) << "\"},";
  emit_config(specification.benchmark);
  emit_metric("stream_us", samples.stream_us);
  std::cout << ',';
  emit_metric("submit_us", samples.submit_us);
  std::cout << ',';
  emit_metric("end_to_end_us", samples.end_to_end_us);
  std::cout << "}\n";
}

void emit_aclnn_record(const BenchmarkCase& specification,
                       const EnvironmentIdentity& environment,
                       const AclnnLibraryIdentity& identity,
                       const MetricSamples& samples) {
  emit_common_prefix("aclnn", specification, environment);
  std::cout << "\"provider_identity\":{"
            << "\"libnnopbase_sha256\":\""
            << json_escape(identity.libnnopbase_sha256) << "\","
            << "\"libopapi_math_sha256\":\""
            << json_escape(identity.libopapi_math_sha256) << '"';
#if FLAGDNN_ASCEND_VALIDATION_ENABLE_OPAPI_NN
  std::cout << ",\"libopapi_nn_sha256\":\""
            << json_escape(identity.libopapi_nn_sha256) << '"';
#endif
  std::cout << "},";
  emit_config(specification.benchmark);
  emit_metric("stream_us", samples.stream_us);
  std::cout << ',';
  emit_metric("submit_us", samples.submit_us);
  std::cout << ',';
  emit_metric("end_to_end_us", samples.end_to_end_us);
  std::cout << "}\n";
}

EnvironmentIdentity environment_identity(const flagdnn::Handle& handle) {
  (void)acl::soc_name();
  const std::string runtime_version = acl::runtime_package_version();
  if (runtime_version != FLAGDNN_ASCEND_VALIDATION_CANN_VERSION) {
    throw std::runtime_error(
        "loaded CANN runtime version differs from configured package: " +
        runtime_version + " vs " FLAGDNN_ASCEND_VALIDATION_CANN_VERSION);
  }
  EnvironmentIdentity result;
  result.soc_fingerprint = std::string(handle.target_fingerprint());
  result.cann_package_version = runtime_version;
  result.ascendcl_build_id = FLAGDNN_ASCEND_VALIDATION_ASCENDCL_BUILD_ID;
  result.runtime_build_id = FLAGDNN_ASCEND_VALIDATION_RUNTIME_BUILD_ID;
  return result;
}

void validate_benchmark_config(const BenchmarkConfig& config) {
  if (config.warmup_iterations < 0 || config.sample_count <= 0 ||
      config.iterations_per_sample <= 0) {
    throw std::invalid_argument("Ascend benchmark configuration is invalid");
  }
}

std::string single_line_reason(std::string_view reason) {
  std::string result(reason);
  std::replace(result.begin(), result.end(), '\r', ' ');
  std::replace(result.begin(), result.end(), '\n', ' ');
  return result;
}

void run_case(const BenchmarkCase& specification,
              FlagdnnProvider& flagdnn_provider,
              AclnnProvider& aclnn_provider,
              acl::Stream& stream,
              const std::filesystem::path& cache,
              const EnvironmentIdentity& environment,
              std::string_view libtriton_jit_sha256,
              const AclnnLibraryIdentity& aclnn_identity) {
  validate_benchmark_config(specification.benchmark);
  const ProviderCapability capability =
      aclnn_provider.capability(specification);
  if (!capability.supported) {
    throw BenchmarkUnsupportedError(
        specification.name + ": ACLNN_UNSUPPORTED: " + capability.reason);
  }
  const PointwiseReferencePlan plan = make_reference_plan(specification);
  std::unique_ptr<BenchmarkExecutable> flagdnn =
      flagdnn_provider.build(specification);
  std::unique_ptr<BenchmarkExecutable> aclnn =
      aclnn_provider.build(specification);
  CaseBuffers buffers = make_buffers(specification, plan, stream);
  ExecutableLifetime executable_lifetime(stream, flagdnn, aclnn);

  flagdnn->prepare(buffers.flagdnn_bindings, stream.opaque());
  try {
    aclnn->prepare(buffers.aclnn_bindings, stream.opaque());
    bool require_repeatable =
        specification.operation == Operation::kReduction ||
        requires_repeatable_reference_contract(specification.pointwise_mode);
#if FLAGDNN_ASCEND_VALIDATION_ENABLE_MATMUL
    require_repeatable =
        require_repeatable || specification.operation == Operation::kMatmul;
#endif
#if FLAGDNN_ASCEND_VALIDATION_ENABLE_CONVOLUTION_FPROP
    require_repeatable = require_repeatable ||
                         specification.operation ==
                             Operation::kConvolutionFprop;
#endif
    if (require_repeatable) {
      aclnn->prepare(buffers.aclnn_bindings, stream.opaque());
    }
  } catch (const AclnnBenchmarkUnsupportedError& error) {
    aclnn.reset();
    throw BenchmarkUnsupportedError(
        specification.name + ": ACLNN_UNSUPPORTED status=" +
        std::to_string(error.status()) + " reason=" + error.what());
  }
  auto flagdnn_workspace =
      std::make_unique<acl::DeviceBuffer>(flagdnn->workspace_size());
  auto aclnn_workspace =
      std::make_unique<acl::DeviceBuffer>(aclnn->workspace_size());

  try {
    stream.synchronize();
    execute(*flagdnn,
            buffers.flagdnn_bindings,
            *flagdnn_workspace,
            stream);
    execute(*aclnn, buffers.aclnn_bindings, *aclnn_workspace, stream);
    stream.synchronize();

    const std::vector<float> flagdnn_physical =
        read_output(*buffers.flagdnn_output,
                    output_tensor(specification),
                    stream,
                    "FlagDNN");
    const std::vector<float> aclnn_physical =
        read_output(*buffers.aclnn_output, plan.output, stream, "ACLNN");
    tensor_io::require_padding_unchanged(
        "FlagDNN", flagdnn_physical, output_tensor(specification));
    tensor_io::require_padding_unchanged("ACLNN", aclnn_physical, plan.output);
    const std::vector<float> flagdnn_logical =
        tensor_io::gather(flagdnn_physical, output_tensor(specification));
    const std::vector<float> aclnn_logical =
        tensor_io::gather(aclnn_physical, plan.output);
    double absolute_tolerance = specification.absolute_tolerance;
    double relative_tolerance = specification.relative_tolerance;
    if (specification.pointwise_mode == FLAGDNN_POINTWISE_GELU_FWD) {
      absolute_tolerance = std::max(
          absolute_tolerance,
          gelu::kAclnnExactAbsoluteTolerance);
      relative_tolerance = std::max(
          relative_tolerance,
          gelu::kAclnnExactRelativeTolerance);
    }
    acl::compare_exact_reference(flagdnn_logical,
                                 aclnn_logical,
                                 absolute_tolerance,
                                 relative_tolerance,
                                 specification.name);
    std::cout << specification.name
              << ": correctness PASS exact_reference=ACLNN\n";
    require_case_inputs_unchanged(buffers, stream);

    for (int index = 0; index < specification.benchmark.warmup_iterations;
         ++index) {
      execute(*flagdnn,
              buffers.flagdnn_bindings,
              *flagdnn_workspace,
              stream);
    }
    stream.synchronize();
    for (int index = 0; index < specification.benchmark.warmup_iterations;
         ++index) {
      execute(*aclnn, buffers.aclnn_bindings, *aclnn_workspace, stream);
    }
    stream.synchronize();

    const FlagdnnArtifactIdentity artifact =
        find_artifact_identity(cache, specification.name);
    const CacheSnapshot before_samples = snapshot_cache(cache);
    MetricSamples flagdnn_samples;
    MetricSamples aclnn_samples;
    flagdnn_samples.stream_us.reserve(specification.benchmark.sample_count);
    flagdnn_samples.submit_us.reserve(specification.benchmark.sample_count);
    flagdnn_samples.end_to_end_us.reserve(
        specification.benchmark.sample_count);
    aclnn_samples.stream_us.reserve(specification.benchmark.sample_count);
    aclnn_samples.submit_us.reserve(specification.benchmark.sample_count);
    aclnn_samples.end_to_end_us.reserve(specification.benchmark.sample_count);

    acl::EventTimer timer;
    const auto measure_flagdnn = [&]() {
      flagdnn_samples.push(timer.measure(
          stream.get(),
          specification.benchmark.iterations_per_sample,
          [&]() {
            execute(*flagdnn,
                    buffers.flagdnn_bindings,
                    *flagdnn_workspace,
                    stream);
          }));
    };
    const auto measure_aclnn = [&]() {
      aclnn_samples.push(timer.measure(
          stream.get(),
          specification.benchmark.iterations_per_sample,
          [&]() {
            execute(*aclnn,
                    buffers.aclnn_bindings,
                    *aclnn_workspace,
                    stream);
          }));
    };

    for (int sample = 0; sample < specification.benchmark.sample_count;
         ++sample) {
      if (sample % 2 == 0) {
        measure_flagdnn();
        measure_aclnn();
      } else {
        measure_aclnn();
        measure_flagdnn();
      }
    }
    stream.synchronize();
    const CacheSnapshot after_samples = snapshot_cache(cache);
    if (before_samples != after_samples) {
      throw std::runtime_error(
          specification.name +
          ": Triton cache changed during benchmark sample window");
    }
    require_case_inputs_unchanged(buffers, stream);

    emit_flagdnn_record(
        specification,
        environment,
        artifact,
        libtriton_jit_sha256,
        flagdnn_samples);
    emit_aclnn_record(
        specification, environment, aclnn_identity, aclnn_samples);
    const double stream_speedup =
        percentile(aclnn_samples.stream_us, 0.5) /
        percentile(flagdnn_samples.stream_us, 0.5);
    const double end_to_end_speedup =
        percentile(aclnn_samples.end_to_end_us, 0.5) /
        percentile(flagdnn_samples.end_to_end_us, 0.5);
    std::cout << specification.name
              << ": stream_speedup=" << stream_speedup
              << " end_to_end_speedup=" << end_to_end_speedup << '\n';
  } catch (...) {
    try {
      stream.synchronize();
    } catch (...) {
    }
    aclnn.reset();
    flagdnn.reset();
    throw;
  }

  stream.synchronize();
  aclnn.reset();
  flagdnn.reset();
}

}  // namespace

int run_benchmark_suite(int argc,
                        char** argv,
                        std::span<const BenchmarkCase> cases,
                        std::string_view suite_name) {
  if (argc != 3) {
    std::cerr << "usage: " << argv[0]
              << " COMPILER_EXECUTABLE COMPILER_ENTRY\n";
    return 2;
  }
  try {
    std::cout << std::setprecision(9);
    if (cases.empty()) {
      throw std::invalid_argument("pointwise benchmark suite has no cases");
    }
    const bool reduction_suite =
        cases.front().operation == Operation::kReduction;
    const bool convolution_fprop_suite =
        cases.front().operation == Operation::kConvolutionFprop;
    const bool matmul_suite =
        cases.front().operation == Operation::kMatmul;
    std::vector<BenchmarkCase> ascend_cases;
    if (reduction_suite) {
      if (cases.size() != 9 ||
          std::any_of(cases.begin(), cases.end(), [](const auto& item) {
            return item.operation != Operation::kReduction;
          })) {
        throw std::invalid_argument(
            "Ascend reduction benchmark common catalog must contain 9 cases");
      }
      const std::size_t sum_count = static_cast<std::size_t>(std::count_if(
          cases.begin(), cases.end(), [](const auto& item) {
            return item.reduction_mode == FLAGDNN_REDUCTION_ADD;
          }));
      const std::size_t average_count = static_cast<std::size_t>(std::count_if(
          cases.begin(), cases.end(), [](const auto& item) {
            return item.reduction_mode == FLAGDNN_REDUCTION_AVG;
          }));
      const std::size_t product_count = static_cast<std::size_t>(std::count_if(
          cases.begin(), cases.end(), [](const auto& item) {
            return item.reduction_mode == FLAGDNN_REDUCTION_MUL;
          }));
      if (sum_count != 3 || average_count != 3 || product_count != 3) {
        throw std::logic_error(
            "Ascend reduction benchmark modes must split 3/3/3");
      }
      ascend_cases.assign(cases.begin(), cases.end());
    } else if (convolution_fprop_suite) {
#if FLAGDNN_ASCEND_VALIDATION_ENABLE_CONVOLUTION_FPROP
      if (cases.size() != 51U ||
          std::any_of(cases.begin(), cases.end(), [](const auto& item) {
            return item.operation != Operation::kConvolutionFprop ||
                   item.output_count != 1U || item.tensors.size() != 3U ||
                   item.convolution.mode !=
                       ConvolutionMode::kCrossCorrelation;
          })) {
        throw std::invalid_argument(
            "Ascend convolution fprop benchmark common catalog must contain "
            "51 cases");
      }
      std::array<std::size_t, 3> type_counts{};
      for (const BenchmarkCase& item : cases) {
        switch (item.tensors[0].data_type) {
          case FLAGDNN_DATA_FLOAT32:
            ++type_counts[0];
            break;
          case FLAGDNN_DATA_FLOAT16:
            ++type_counts[1];
            break;
          case FLAGDNN_DATA_BFLOAT16:
            ++type_counts[2];
            break;
          default:
            throw std::invalid_argument(
                "Ascend convolution fprop benchmark contains an invalid "
                "dtype");
        }
      }
      if (type_counts != std::array<std::size_t, 3>({17U, 17U, 17U})) {
        throw std::logic_error(
            "Ascend convolution fprop benchmark dtype counts must be "
            "17/17/17");
      }
      ascend_cases.assign(cases.begin(), cases.end());
      for (BenchmarkCase& item : ascend_cases) {
        // The common GPU catalog batches ten convolutions per sample.  The
        // Ascend direct kernel reports one complete launch per sample so the
        // 51-case matrix remains practical while retaining ten observations.
        item.benchmark.warmup_iterations = 2;
        item.benchmark.sample_count = 10;
        item.benchmark.iterations_per_sample = 1;
      }
#else
      throw std::invalid_argument(
          "Ascend convolution fprop benchmark adapter is not enabled");
#endif
    } else if (matmul_suite) {
#if FLAGDNN_ASCEND_VALIDATION_ENABLE_MATMUL
      if (cases.size() != 24 ||
          std::any_of(cases.begin(), cases.end(), [](const auto& item) {
            return item.operation != Operation::kMatmul ||
                   item.output_count != 1 || item.tensors.size() != 3;
          })) {
        throw std::invalid_argument(
            "Ascend MatMul benchmark common catalog must contain 24 cases");
      }
      std::array<std::size_t, 3> type_counts{};
      for (const BenchmarkCase& item : cases) {
        switch (item.tensors[0].data_type) {
          case FLAGDNN_DATA_FLOAT32:
            ++type_counts[0];
            break;
          case FLAGDNN_DATA_FLOAT16:
            ++type_counts[1];
            break;
          case FLAGDNN_DATA_BFLOAT16:
            ++type_counts[2];
            break;
          default:
            throw std::invalid_argument(
                "Ascend MatMul benchmark contains an invalid dtype");
        }
      }
      if (type_counts != std::array<std::size_t, 3>({8, 8, 8})) {
        throw std::logic_error(
            "Ascend MatMul benchmark dtype counts must be 8/8/8");
      }
      ascend_cases.assign(cases.begin(), cases.end());
#else
      throw std::invalid_argument(
          "Ascend MatMul benchmark adapter is not enabled");
#endif
    } else {
      if (is_logical_mode(cases.front().pointwise_mode) &&
          cases.size() != 8) {
      throw std::invalid_argument(
          "Ascend logical benchmark catalog must contain 8 cases");
      }
      if (is_comparison_mode(cases.front().pointwise_mode) &&
          cases.size() != 24) {
      throw std::invalid_argument(
          "Ascend comparison benchmark common catalog must contain 24 cases");
      }
      if (cases.front().pointwise_mode == FLAGDNN_POINTWISE_BINARY_SELECT &&
          cases.size() != 24) {
      throw std::invalid_argument(
          "Ascend binary-select benchmark common catalog must contain "
          "24 cases");
      }
      ascend_cases.assign(cases.begin(), cases.end());
    if (requires_unary_benchmark_extension(
            cases.front().pointwise_mode)) {
      if (std::any_of(cases.begin(), cases.end(), [&](const auto& item) {
            return item.operation != Operation::kPointwise ||
                   item.pointwise_mode != cases.front().pointwise_mode;
          })) {
        throw std::invalid_argument(
            "Ascend unary benchmark suite mixes operation modes");
      }
      ascend_cases.push_back(
          make_unary_benchmark_extension(cases.front()));
    } else if (requires_binary_benchmark_extension(
                   cases.front().pointwise_mode)) {
      if (std::any_of(cases.begin(), cases.end(), [&](const auto& item) {
            return item.operation != Operation::kPointwise ||
                   item.pointwise_mode != cases.front().pointwise_mode;
          })) {
        throw std::invalid_argument(
            "Ascend binary benchmark suite mixes operation modes");
      }
      ascend_cases.push_back(
          make_binary_benchmark_extension(cases.front()));
    } else if (requires_ternary_benchmark_extension(
                   cases.front().pointwise_mode)) {
      if (std::any_of(cases.begin(), cases.end(), [&](const auto& item) {
            return item.operation != Operation::kPointwise ||
                   item.pointwise_mode != cases.front().pointwise_mode ||
                   input_tensor_count(item) != 3;
          })) {
        throw std::invalid_argument(
            "Ascend ternary benchmark suite mixes operation modes or "
            "arities");
      }
      ascend_cases.push_back(
          make_binary_select_benchmark_extension(cases.front()));
    }
    if (is_logical_mode(cases.front().pointwise_mode) &&
        ascend_cases.size() != 8) {
      throw std::logic_error(
          "Ascend logical benchmark catalog must contain 8 cases");
    }
    if (is_comparison_mode(cases.front().pointwise_mode) &&
        ascend_cases.size() != 25) {
      throw std::logic_error(
          "Ascend comparison benchmark catalog must contain 25 cases");
    }
    if (cases.front().pointwise_mode == FLAGDNN_POINTWISE_BINARY_SELECT &&
        ascend_cases.size() != 25) {
      throw std::logic_error(
          "Ascend binary-select benchmark catalog must contain 25 cases");
      }
    }
    acl::DevelopmentEnvironment development(
        reduction_suite
            ? "reduction-benchmark"
            : (matmul_suite ? "matmul-benchmark" : "pointwise-benchmark"));
    acl::AclRuntime runtime;
    development.prepare_target(acl::soc_name());
    std::cout << "ASCEND_VALIDATION_DEVELOPMENT_ROOT root="
              << development.root() << '\n';
    const char* filter = std::getenv("FLAGDNN_BENCHMARK_CASE");
    std::vector<const BenchmarkCase*> selected_cases;
    selected_cases.reserve(ascend_cases.size());
    for (const BenchmarkCase& specification : ascend_cases) {
      if (filter != nullptr && filter[0] != '\0' &&
          specification.name.find(filter) == std::string::npos) {
        continue;
      }
      selected_cases.push_back(&specification);
    }
    if (selected_cases.empty()) {
      throw std::runtime_error(
          "FLAGDNN_BENCHMARK_CASE matched no benchmark cases");
    }
    acl::ExactReferenceCoverage coverage(selected_cases.size());
    {
      acl::Stream stream;
      {
        flagdnn::Handle handle("ascend", 0);
        handle.set_compiler(
            argv[1], argv[2], development.graph_cache().string());
        const EnvironmentIdentity environment = environment_identity(handle);
        const std::string libtriton_jit_sha256 =
            loaded_named_object_sha256("libtriton_jit.so");
        if (libtriton_jit_sha256 !=
            FLAGDNN_ASCEND_VALIDATION_TRITON_JIT_SHA256) {
          throw std::runtime_error(
              "loaded libtriton_jit differs from the production target");
        }
        const AclnnLibraryIdentity aclnn_identity =
            aclnn_library_identity();
        FlagdnnProvider flagdnn_provider(handle);
        // A fixed candidate gives every v2 record an exact stable identity.
        // Graph build, compilation, and candidate preparation still occur
        // before the correctness/warmup/sample timing boundaries.
        flagdnn_provider.set_autotune(false);
        AclnnProvider aclnn_provider;

        for (const BenchmarkCase* specification : selected_cases) {
          try {
            run_case(*specification,
                     flagdnn_provider,
                     aclnn_provider,
                     stream,
                     development.graph_cache(),
                     environment,
                     libtriton_jit_sha256,
                     aclnn_identity);
            coverage.record_pass(specification->name);
          } catch (const BenchmarkUnsupportedError& error) {
            coverage.record_skip(
                specification->name, single_line_reason(error.what()));
          }
        }
        stream.synchronize();
      }
    }
    runtime.finalize();
    development.cleanup();
    coverage.require_complete();
    for (const std::string& line : coverage.skip_lines()) {
      std::cout << line << '\n';
    }
    std::cout << coverage.summary() << '\n';
    std::cout << suite_name << ": PASS cases=" << coverage.passed()
              << " catalog_cases=" << ascend_cases.size()
              << " schema_v2_records=" << coverage.passed() * 2U << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << suite_name << "_FAILED: " << error.what() << '\n';
    return 1;
  }
}

}  // namespace flagdnn::benchmarking
