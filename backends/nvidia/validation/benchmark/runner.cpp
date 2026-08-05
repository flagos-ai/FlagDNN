#include "common/runner.hpp"

#include "validation/cuda_driver.hpp"
#include "validation/tensor_io.hpp"
#include "common/flagdnn_provider.hpp"
#include "validation/benchmark/cudnn_provider.hpp"
#include "validation/benchmark/cudnn_common.hpp"


#include <flagdnn/flagdnn.hpp>

#include <unistd.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

namespace tensor_io = flagdnn::validation::nvidia::tensor_io;
constexpr float kPaddingSentinel = tensor_io::kPaddingSentinel;

using flagdnn::benchmarking::BenchmarkCase;
using flagdnn::benchmarking::DeviceBuffer;
using flagdnn::benchmarking::EventTimer;
using flagdnn::benchmarking::InputDomain;
using flagdnn::benchmarking::BenchmarkExecutable;
using flagdnn::benchmarking::BenchmarkUnsupportedError;
using flagdnn::validation::nvidia::check_cuda;

class BenchmarkCache {
 public:
  BenchmarkCache() {
    const char* configured =
        std::getenv("FLAGDNN_BENCHMARK_CACHE_DIRECTORY");
    if (configured != nullptr && configured[0] != '\0') {
      path_ = configured;
    } else {
      path_ = std::filesystem::temp_directory_path() /
              ("flagdnn-benchmark-cache-" + std::to_string(getuid()));
    }
    std::error_code error;
    std::filesystem::create_directories(path_, error);
    if (error) {
      throw std::runtime_error(
          "cannot create benchmark cache directory: " + error.message());
    }
  }

  [[nodiscard]] const std::filesystem::path& path() const noexcept {
    return path_;
  }

 private:
  std::filesystem::path path_;
};

class CapturedExecutionBatch {
 public:
  template <typename Function>
  CapturedExecutionBatch(CUstream stream,
                         int execution_count,
                         Function&& execute)
      : execution_count_(execution_count) {
    if (execution_count_ <= 0) {
      throw std::invalid_argument("captured execution count must be positive");
    }

    check_cuda(cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_RELAXED),
               "cuStreamBeginCapture");
    try {
      for (int i = 0; i < execution_count_; ++i) {
        execute();
      }
    } catch (...) {
      CUgraph abandoned_graph = nullptr;
      if (cuStreamEndCapture(stream, &abandoned_graph) == CUDA_SUCCESS &&
          abandoned_graph != nullptr) {
        cuGraphDestroy(abandoned_graph);
      }
      throw;
    }

    check_cuda(cuStreamEndCapture(stream, &graph_), "cuStreamEndCapture");
    try {
      check_cuda(cuGraphInstantiate(&executable_, graph_, 0),
                 "cuGraphInstantiate");
    } catch (...) {
      cuGraphDestroy(graph_);
      graph_ = nullptr;
      throw;
    }
  }

  CapturedExecutionBatch(const CapturedExecutionBatch&) = delete;
  CapturedExecutionBatch& operator=(const CapturedExecutionBatch&) = delete;

  ~CapturedExecutionBatch() {
    if (executable_ != nullptr) {
      cuGraphExecDestroy(executable_);
    }
    if (graph_ != nullptr) {
      cuGraphDestroy(graph_);
    }
  }

  void launch(CUstream stream) const {
    check_cuda(cuGraphLaunch(executable_, stream), "cuGraphLaunch");
  }

  [[nodiscard]] int execution_count() const noexcept {
    return execution_count_;
  }

 private:
  CUgraph graph_ = nullptr;
  CUgraphExec executable_ = nullptr;
  int execution_count_ = 0;
};

std::size_t element_count(
    const flagdnn::benchmarking::TensorSpec& specification) {
  return tensor_io::element_count(specification);
}

std::size_t storage_element_count(
    const flagdnn::benchmarking::TensorSpec& specification) {
  return tensor_io::storage_element_count(specification);
}

flagdnn::benchmarking::TensorSpec cudnn_tensor_specification(
    const BenchmarkCase& specification,
    std::size_t tensor_index) {
  const auto& tensor = specification.tensors.at(tensor_index);
  if ((specification.operation ==
           flagdnn::benchmarking::Operation::kBatchnorm ||
       specification.operation ==
           flagdnn::benchmarking::Operation::kBatchnormInference) &&
      (tensor_index == 0 ||
       tensor_index == input_tensor_count(specification))) {
    return flagdnn::benchmarking::cudnn_detail::
        batchnorm_inference_nhwc_tensor(tensor);
  }
  return tensor;
}


std::vector<float> scatter_logical_values(
    std::span<const float> logical,
    const flagdnn::benchmarking::TensorSpec& specification) {
  return tensor_io::scatter(logical, specification);
}

std::vector<float> gather_logical_values(
    std::span<const float> physical,
    const flagdnn::benchmarking::TensorSpec& specification) {
  return tensor_io::gather(physical, specification);
}

void require_output_padding_unchanged(
    std::string_view provider,
    std::span<const float> physical,
    const flagdnn::benchmarking::TensorSpec& specification) {
  tensor_io::require_padding_unchanged(provider, physical, specification);
}

std::size_t data_type_size(flagdnnDataType_t data_type) {
  return tensor_io::data_type_size(data_type);
}

std::vector<std::uint8_t> encode_values(
    std::span<const float> values,
    flagdnnDataType_t data_type) {
  return tensor_io::encode(
      values, data_type, tensor_io::BooleanEncoding::kByte);
}

std::vector<float> decode_values(
    std::span<const std::uint8_t> bytes,
    flagdnnDataType_t data_type) {
  const std::size_t count = bytes.size() / data_type_size(data_type);
  return tensor_io::decode(
      bytes, data_type, count, tensor_io::BooleanEncoding::kByte);
}

std::size_t cudnn_encoded_byte_count(
    const flagdnn::benchmarking::TensorSpec& specification) {
  return tensor_io::encoded_byte_count(
      specification, tensor_io::BooleanEncoding::kBitPacked);
}

std::vector<std::uint8_t> encode_cudnn_values(
    std::span<const float> physical,
    flagdnnDataType_t data_type) {
  return tensor_io::encode(
      physical, data_type, tensor_io::BooleanEncoding::kBitPacked);
}

std::vector<float> decode_cudnn_values(
    std::span<const std::uint8_t> bytes,
    flagdnnDataType_t data_type,
    std::size_t storage_count) {
  return tensor_io::decode(
      bytes, data_type, storage_count,
      tensor_io::BooleanEncoding::kBitPacked);
}

std::vector<float> make_input(std::size_t count,
                              std::size_t tensor_index,
                              InputDomain domain) {
  std::vector<float> result(count);
  for (std::size_t index = 0; index < count; ++index) {
    const int centered =
        static_cast<int>((index * 17 + tensor_index * 11) % 41) - 20;
    const float real_value =
        static_cast<float>(centered) / static_cast<float>(13 + tensor_index);
    switch (domain) {
      case InputDomain::kReal:
        result[index] = real_value;
        break;
      case InputDomain::kPositive:
        result[index] = std::abs(real_value) + 0.5F;
        break;
      case InputDomain::kScaled:
        result[index] = real_value * 4.0F;
        break;
      case InputDomain::kTan:
        result[index] = static_cast<float>(centered) / 40.0F;
        break;
      case InputDomain::kDivisor:
      case InputDomain::kModulo:
        result[index] = tensor_index == 1
                            ? std::abs(real_value) + 0.5F
                            : real_value;
        break;
      case InputDomain::kPower:
        result[index] = tensor_index == 0
                            ? std::abs(real_value) + 0.5F
                            : std::fmod(std::abs(real_value), 2.0F) + 0.125F;
        break;
      case InputDomain::kModuloSigned: {
        constexpr std::array<float, 6> kLeft = {
            -3.0F, -3.0F, 3.0F, 3.0F, -5.5F, 5.5F};
        constexpr std::array<float, 6> kRight = {
            2.0F, -2.0F, 2.0F, -2.0F, 2.25F, -2.25F};
        result[index] = tensor_index == 0
                            ? kLeft[index % kLeft.size()]
                            : kRight[index % kRight.size()];
        break;
      }
      case InputDomain::kComparison: {
        const int base_centered = static_cast<int>((index * 17) % 41) - 20;
        const float base = static_cast<float>(base_centered) / 13.0F;
        if (tensor_index == 0 || index % 3 == 0) {
          result[index] = base;
        } else if (index % 3 == 1) {
          result[index] = base + 0.25F;
        } else {
          result[index] = base - 0.25F;
        }
        break;
      }
      case InputDomain::kLogical:
        result[index] =
            ((index * 17 + tensor_index * 11) % 3) != 0 ? 1.0F : 0.0F;
        break;
    }
  }
  return result;
}

struct Accuracy {
  double maximum_absolute = 0.0;
  double maximum_relative = 0.0;
};

Accuracy compare_outputs(std::span<const float> actual,
                         std::span<const float> reference,
                         double absolute_tolerance,
                         double relative_tolerance) {
  if (actual.size() != reference.size()) {
    throw std::runtime_error("output sizes do not match");
  }
  Accuracy result;
  for (std::size_t index = 0; index < actual.size(); ++index) {
    const double left = actual[index];
    const double right = reference[index];
    const double absolute = std::abs(left - right);
    const double relative =
        absolute / std::max({std::abs(left), std::abs(right), 1.0e-30});
    result.maximum_absolute = std::max(result.maximum_absolute, absolute);
    result.maximum_relative = std::max(result.maximum_relative, relative);
    if (!std::isfinite(absolute) ||
        (absolute > absolute_tolerance && relative > relative_tolerance)) {
      std::ostringstream message;
      message << "FlagDNN output differs from reference at element "
              << index << ": actual=" << left << ", reference=" << right
              << ", abs=" << absolute << ", rel=" << relative
              << ", atol=" << absolute_tolerance
              << ", rtol=" << relative_tolerance;
      throw std::runtime_error(message.str());
    }
  }
  return result;
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

void emit_samples(const char* provider,
                  const BenchmarkCase& specification,
                  const std::vector<double>& samples) {
  std::cout << "{\"schema_version\":1,\"kind\":\"steady_state\","
            << "\"provider\":\"" << provider << "\",\"case\":\""
            << specification.name << "\",\"unit\":\"us\","
            << "\"median\":" << percentile(samples, 0.5)
            << ",\"p90\":" << percentile(samples, 0.9)
            << ",\"samples\":[";
  for (std::size_t index = 0; index < samples.size(); ++index) {
    if (index != 0) {
      std::cout << ',';
    }
    std::cout << samples[index];
  }
  std::cout << "]}\n";
}

struct CaseBuffers {
  std::vector<std::unique_ptr<DeviceBuffer>> inputs;
  std::vector<std::vector<float>> host_inputs;
  std::vector<std::unique_ptr<DeviceBuffer>> flagdnn_outputs;
  std::vector<std::unique_ptr<DeviceBuffer>> cudnn_outputs;
  std::vector<flagdnnBinding_t> flagdnn_bindings;
  std::vector<flagdnnBinding_t> cudnn_bindings;
};

CaseBuffers make_buffers(const BenchmarkCase& specification,
                         flagdnn::benchmarking::Stream& stream) {
  if (specification.tensors.size() < 2) {
    throw std::invalid_argument("case has too few tensors");
  }
  CaseBuffers result;
  const std::size_t input_count = input_tensor_count(specification);
  result.inputs.reserve(input_count * 2);
  result.host_inputs.reserve(input_count);
  result.flagdnn_bindings.reserve(specification.tensors.size());
  result.cudnn_bindings.reserve(specification.tensors.size());
  for (std::size_t index = 0; index < input_count; ++index) {
    const auto& tensor = specification.tensors[index];
    const std::size_t count = element_count(tensor);
    const InputDomain domain =
        specification.input_domains.empty()
            ? specification.input_domain
            : specification.input_domains.at(index);
    std::vector<float> logical = make_input(count, index, domain);
    std::vector<float> physical =
        scatter_logical_values(logical, tensor);
    std::vector<std::uint8_t> encoded =
        encode_values(physical, tensor.data_type);
    result.host_inputs.push_back(gather_logical_values(
        decode_values(encoded, tensor.data_type), tensor));
    auto flagdnn_buffer = std::make_unique<DeviceBuffer>(
        tensor.binding_byte_offset + encoded.size());
    flagdnn_buffer->copy_from_host_at(encoded.data(),
                                      encoded.size(),
                                      tensor.binding_byte_offset,
                                      stream.get());
    void* flagdnn_binding =
        flagdnn_buffer->opaque_at(tensor.binding_byte_offset);
    result.flagdnn_bindings.push_back({tensor.uid, flagdnn_binding});
    const auto cudnn_tensor =
        cudnn_tensor_specification(specification, index);
    const bool separate_cudnn_buffer =
        tensor.data_type == FLAGDNN_DATA_BOOLEAN ||
        cudnn_tensor.dimensions != tensor.dimensions ||
        cudnn_tensor.strides != tensor.strides;
    if (separate_cudnn_buffer) {
      const std::vector<float> cudnn_physical =
          scatter_logical_values(logical, cudnn_tensor);
      const std::vector<std::uint8_t> cudnn_encoded =
          encode_cudnn_values(cudnn_physical, tensor.data_type);
      auto cudnn_buffer = std::make_unique<DeviceBuffer>(
          tensor.binding_byte_offset + cudnn_encoded.size());
      cudnn_buffer->copy_from_host_at(cudnn_encoded.data(),
                                      cudnn_encoded.size(),
                                      tensor.binding_byte_offset,
                                      stream.get());
      result.cudnn_bindings.push_back(
          {tensor.uid,
           cudnn_buffer->opaque_at(tensor.binding_byte_offset)});
      result.inputs.push_back(std::move(cudnn_buffer));
    } else {
      result.cudnn_bindings.push_back({tensor.uid, flagdnn_binding});
    }
    result.inputs.push_back(std::move(flagdnn_buffer));
  }

  result.flagdnn_outputs.reserve(specification.output_count);
  result.cudnn_outputs.reserve(specification.output_count);
  for (std::size_t output_index = 0;
       output_index < specification.output_count;
       ++output_index) {
    const std::size_t tensor_index = input_count + output_index;
    const auto& output = specification.tensors[tensor_index];
    const auto cudnn_output_specification =
        cudnn_tensor_specification(specification, tensor_index);
    const std::size_t output_bytes =
        storage_element_count(output) * data_type_size(output.data_type);
    auto flagdnn_output = std::make_unique<DeviceBuffer>(
        output.binding_byte_offset + output_bytes);
    const std::size_t cudnn_output_bytes =
        cudnn_encoded_byte_count(cudnn_output_specification);
    auto cudnn_output = std::make_unique<DeviceBuffer>(
        output.binding_byte_offset + cudnn_output_bytes);
    const std::vector<float> initial_output(
        storage_element_count(output), kPaddingSentinel);
    const std::vector<float> initial_cudnn_output(
        storage_element_count(cudnn_output_specification),
        kPaddingSentinel);
    const std::vector<std::uint8_t> encoded_output =
        encode_values(initial_output, output.data_type);
    const std::vector<std::uint8_t> cudnn_encoded_output =
        encode_cudnn_values(initial_cudnn_output, output.data_type);
    flagdnn_output->copy_from_host_at(
        encoded_output.data(),
        encoded_output.size(),
        output.binding_byte_offset,
        stream.get());
    cudnn_output->copy_from_host_at(
        cudnn_encoded_output.data(),
        cudnn_encoded_output.size(),
        output.binding_byte_offset,
        stream.get());
    result.flagdnn_bindings.push_back(
        {output.uid,
         flagdnn_output->opaque_at(output.binding_byte_offset)});
    result.cudnn_bindings.push_back(
        {output.uid,
         cudnn_output->opaque_at(output.binding_byte_offset)});
    result.flagdnn_outputs.push_back(std::move(flagdnn_output));
    result.cudnn_outputs.push_back(std::move(cudnn_output));
  }
  return result;
}

std::vector<float> reduction_host_reference(
    const BenchmarkCase& specification,
    std::span<const float> input) {
  if (specification.operation !=
          flagdnn::benchmarking::Operation::kReduction ||
      specification.tensors.size() != 2) {
    throw std::invalid_argument(
        "host fallback only supports one-input Reduction cases");
  }
  const auto& input_specification = specification.tensors[0];
  const auto& output_specification = specification.tensors[1];
  std::int32_t axis = specification.reduction_axis;
  const std::int32_t rank = static_cast<std::int32_t>(
      input_specification.dimensions.size());
  if (axis < 0) {
    axis += rank;
  }
  if (axis < 0 || axis >= rank) {
    throw std::invalid_argument("host Reduction axis is invalid");
  }

  std::vector<std::size_t> input_contiguous_strides(
      input_specification.dimensions.size());
  std::size_t stride = 1;
  for (std::size_t index = input_specification.dimensions.size();
       index != 0;
       --index) {
    input_contiguous_strides[index - 1] = stride;
    stride *= static_cast<std::size_t>(
        input_specification.dimensions[index - 1]);
  }

  std::vector<float> result(element_count(output_specification));
  for (std::size_t output_index = 0;
       output_index < result.size();
       ++output_index) {
    std::size_t remaining = output_index;
    std::vector<std::size_t> input_coordinates(
        input_specification.dimensions.size(), 0);
    for (std::size_t output_axis =
             output_specification.dimensions.size();
         output_axis != 0;
         --output_axis) {
      const std::size_t current = output_axis - 1;
      const std::size_t dimension = static_cast<std::size_t>(
          output_specification.dimensions[current]);
      const std::size_t coordinate = remaining % dimension;
      remaining /= dimension;
      const std::size_t input_axis =
          specification.keep_dimensions ||
                  current < static_cast<std::size_t>(axis)
              ? current
              : current + 1;
      input_coordinates[input_axis] = coordinate;
    }

    float accumulator =
        specification.reduction_mode == FLAGDNN_REDUCTION_MUL ? 1.0F
                                                               : 0.0F;
    const std::size_t extent = static_cast<std::size_t>(
        input_specification.dimensions[static_cast<std::size_t>(axis)]);
    for (std::size_t reduction_index = 0;
         reduction_index < extent;
         ++reduction_index) {
      input_coordinates[static_cast<std::size_t>(axis)] = reduction_index;
      std::size_t input_index = 0;
      for (std::size_t input_axis = 0;
           input_axis < input_coordinates.size();
           ++input_axis) {
        input_index += input_coordinates[input_axis] *
                       input_contiguous_strides[input_axis];
      }
      if (specification.reduction_mode == FLAGDNN_REDUCTION_MUL) {
        accumulator *= input[input_index];
      } else {
        accumulator += input[input_index];
      }
    }
    if (specification.reduction_mode == FLAGDNN_REDUCTION_AVG) {
      accumulator /= static_cast<float>(extent);
    }
    result[output_index] = accumulator;
  }
  return decode_values(
      encode_values(result, output_specification.data_type),
      output_specification.data_type);
}

std::vector<float> add_host_reference(
    const BenchmarkCase& specification,
    std::span<const float> left,
    std::span<const float> right) {
  if (specification.operation != flagdnn::benchmarking::Operation::kAdd ||
      specification.tensors.size() != 3) {
    throw std::invalid_argument(
        "host Add fallback requires two inputs and one output");
  }
  const auto& left_specification = specification.tensors[0];
  const auto& right_specification = specification.tensors[1];
  const auto& output_specification = specification.tensors[2];
  const std::size_t output_rank = output_specification.dimensions.size();

  const auto input_value = [&](std::span<const float> values,
                               const flagdnn::benchmarking::TensorSpec& input,
                               std::span<const std::size_t> coordinates) {
    const std::size_t leading = output_rank - input.dimensions.size();
    std::size_t input_index = 0;
    std::size_t input_stride = 1;
    for (std::size_t axis = input.dimensions.size(); axis != 0; --axis) {
      const std::size_t current = axis - 1;
      const std::size_t coordinate =
          input.dimensions[current] == 1
              ? 0
              : coordinates[leading + current];
      input_index += coordinate * input_stride;
      input_stride *=
          static_cast<std::size_t>(input.dimensions[current]);
    }
    return values[input_index];
  };

  std::vector<float> result(element_count(output_specification));
  std::vector<std::size_t> coordinates(output_rank);
  for (std::size_t output_index = 0;
       output_index < result.size();
       ++output_index) {
    std::size_t remaining = output_index;
    for (std::size_t axis = output_rank; axis != 0; --axis) {
      const std::size_t current = axis - 1;
      const std::size_t dimension = static_cast<std::size_t>(
          output_specification.dimensions[current]);
      coordinates[current] = remaining % dimension;
      remaining /= dimension;
    }
    result[output_index] =
        input_value(left, left_specification, coordinates) +
        static_cast<float>(specification.add_alpha) *
            input_value(right, right_specification, coordinates);
  }
  return decode_values(
      encode_values(result, output_specification.data_type),
      output_specification.data_type);
}

std::vector<float> pointwise_host_reference(
    const BenchmarkCase& specification,
    std::span<const float> input) {
  if (specification.operation !=
          flagdnn::benchmarking::Operation::kPointwise ||
      specification.tensors.size() != 2) {
    throw std::invalid_argument(
        "host pointwise fallback requires one input and one output");
  }
  const auto& output_specification = specification.tensors[1];
  std::vector<float> result(input.size());
  for (std::size_t index = 0; index < input.size(); ++index) {
    const float value = input[index];
    switch (specification.pointwise_mode) {
      case FLAGDNN_POINTWISE_SQRT:
        result[index] = std::sqrt(value);
        break;
      case FLAGDNN_POINTWISE_ERF:
        result[index] = std::erf(value);
        break;
      case FLAGDNN_POINTWISE_IDENTITY:
        result[index] = value;
        break;
      case FLAGDNN_POINTWISE_EXP:
        result[index] = std::exp(value);
        break;
      case FLAGDNN_POINTWISE_LOG:
        result[index] = std::log(value);
        break;
      case FLAGDNN_POINTWISE_NEG:
        result[index] = -value;
        break;
      case FLAGDNN_POINTWISE_ABS:
        result[index] = std::abs(value);
        break;
      case FLAGDNN_POINTWISE_CEIL:
        result[index] = std::ceil(value);
        break;
      case FLAGDNN_POINTWISE_COS:
        result[index] = std::cos(value);
        break;
      case FLAGDNN_POINTWISE_FLOOR:
        result[index] = std::floor(value);
        break;
      case FLAGDNN_POINTWISE_RSQRT:
        result[index] = 1.0F / std::sqrt(value);
        break;
      case FLAGDNN_POINTWISE_SIN:
        result[index] = std::sin(value);
        break;
      case FLAGDNN_POINTWISE_TAN:
        result[index] = std::tan(value);
        break;
      case FLAGDNN_POINTWISE_RECIPROCAL:
        result[index] = 1.0F / value;
        break;
      case FLAGDNN_POINTWISE_RELU_FWD:
      {
        const auto& attributes = specification.pointwise_attributes;
        const float lower =
            (attributes.flags &
             FLAGDNN_POINTWISE_ATTRIBUTE_RELU_LOWER_CLIP) != 0U
                ? static_cast<float>(attributes.relu_lower_clip)
                : 0.0F;
        const float slope =
            (attributes.flags &
             FLAGDNN_POINTWISE_ATTRIBUTE_RELU_LOWER_CLIP_SLOPE) != 0U
                ? static_cast<float>(attributes.relu_lower_clip_slope)
                : 0.0F;
        float output = value < lower
                           ? lower + slope * (value - lower)
                           : value;
        if ((attributes.flags &
             FLAGDNN_POINTWISE_ATTRIBUTE_RELU_UPPER_CLIP) != 0U) {
          output = std::min(
              output, static_cast<float>(attributes.relu_upper_clip));
        }
        result[index] = output;
        break;
      }
      case FLAGDNN_POINTWISE_LOGICAL_NOT:
        result[index] = value == 0.0F ? 1.0F : 0.0F;
        break;
      case FLAGDNN_POINTWISE_SIGMOID_FWD:
        result[index] = 1.0F / (1.0F + std::exp(-value));
        break;
      case FLAGDNN_POINTWISE_TANH_FWD:
        result[index] = std::tanh(value);
        break;
      case FLAGDNN_POINTWISE_ELU_FWD: {
        const auto& attributes = specification.pointwise_attributes;
        const float alpha =
            (attributes.flags & FLAGDNN_POINTWISE_ATTRIBUTE_ELU_ALPHA) !=
                    0U
                ? static_cast<float>(attributes.elu_alpha)
                : 1.0F;
        result[index] =
            value > 0.0F ? value : alpha * std::expm1(value);
        break;
      }
      case FLAGDNN_POINTWISE_GELU_FWD:
        result[index] = 0.5F * value *
                        (1.0F + std::erf(
                                      value * 0.7071067811865476F));
        break;
      case FLAGDNN_POINTWISE_SOFTPLUS_FWD: {
        const auto& attributes = specification.pointwise_attributes;
        const float beta =
            (attributes.flags &
             FLAGDNN_POINTWISE_ATTRIBUTE_SOFTPLUS_BETA) != 0U
                ? static_cast<float>(attributes.softplus_beta)
                : 1.0F;
        const float beta_value = beta * value;
        result[index] =
            beta_value > 20.0F
                ? value
                : (std::max(beta_value, 0.0F) +
                   std::log1p(std::exp(-std::abs(beta_value)))) /
                      beta;
        break;
      }
      case FLAGDNN_POINTWISE_SWISH_FWD: {
        const auto& attributes = specification.pointwise_attributes;
        const float beta =
            (attributes.flags & FLAGDNN_POINTWISE_ATTRIBUTE_SWISH_BETA) !=
                    0U
                ? static_cast<float>(attributes.swish_beta)
                : 1.0F;
        result[index] = value / (1.0F + std::exp(-beta * value));
        break;
      }
      case FLAGDNN_POINTWISE_GELU_APPROX_TANH_FWD: {
        const float inner = 0.7978845608028654F *
                            (value + 0.044715F * value * value * value);
        result[index] =
            0.5F * value * (1.0F + std::tanh(inner));
        break;
      }
      case FLAGDNN_POINTWISE_NOT_SET:
      case FLAGDNN_POINTWISE_BINARY_SELECT:
      case FLAGDNN_POINTWISE_SIGMOID_BWD:
      case FLAGDNN_POINTWISE_ADD:
      case FLAGDNN_POINTWISE_SUB:
      case FLAGDNN_POINTWISE_MUL:
      case FLAGDNN_POINTWISE_DIV:
      case FLAGDNN_POINTWISE_MIN:
      case FLAGDNN_POINTWISE_MAX:
      case FLAGDNN_POINTWISE_MOD:
      case FLAGDNN_POINTWISE_POW:
      case FLAGDNN_POINTWISE_CMP_EQ:
      case FLAGDNN_POINTWISE_CMP_NEQ:
      case FLAGDNN_POINTWISE_CMP_GT:
      case FLAGDNN_POINTWISE_CMP_GE:
      case FLAGDNN_POINTWISE_CMP_LT:
      case FLAGDNN_POINTWISE_CMP_LE:
      case FLAGDNN_POINTWISE_LOGICAL_AND:
      case FLAGDNN_POINTWISE_LOGICAL_OR:
        throw std::invalid_argument(
            "host fallback received a non-unary pointwise mode");
    }
  }
  return decode_values(
      encode_values(result, output_specification.data_type),
      output_specification.data_type);
}

std::vector<float> binary_pointwise_host_reference(
    const BenchmarkCase& specification,
    std::span<const float> left,
    std::span<const float> right) {
  if (specification.operation !=
          flagdnn::benchmarking::Operation::kPointwise ||
      specification.tensors.size() != 3) {
    throw std::invalid_argument(
        "host binary pointwise fallback requires two inputs and one output");
  }
  const auto& left_specification = specification.tensors[0];
  const auto& right_specification = specification.tensors[1];
  const auto& output_specification = specification.tensors[2];
  const std::size_t output_rank = output_specification.dimensions.size();

  const auto input_value = [&](std::span<const float> values,
                               const flagdnn::benchmarking::TensorSpec& input,
                               std::span<const std::size_t> coordinates) {
    const std::size_t leading = output_rank - input.dimensions.size();
    std::size_t input_index = 0;
    std::size_t input_stride = 1;
    for (std::size_t axis = input.dimensions.size(); axis != 0; --axis) {
      const std::size_t current = axis - 1;
      const std::size_t coordinate =
          input.dimensions[current] == 1
              ? 0
              : coordinates[leading + current];
      input_index += coordinate * input_stride;
      input_stride *=
          static_cast<std::size_t>(input.dimensions[current]);
    }
    return values[input_index];
  };

  std::vector<float> result(element_count(output_specification));
  std::vector<std::size_t> coordinates(output_rank);
  for (std::size_t output_index = 0;
       output_index < result.size();
       ++output_index) {
    std::size_t remaining = output_index;
    for (std::size_t axis = output_rank; axis != 0; --axis) {
      const std::size_t current = axis - 1;
      const std::size_t dimension = static_cast<std::size_t>(
          output_specification.dimensions[current]);
      coordinates[current] = remaining % dimension;
      remaining /= dimension;
    }
    const float left_value =
        input_value(left, left_specification, coordinates);
    const float right_value =
        input_value(right, right_specification, coordinates);
    switch (specification.pointwise_mode) {
      case FLAGDNN_POINTWISE_ADD:
        result[output_index] =
            left_value +
            static_cast<float>(specification.add_alpha) * right_value;
        break;
      case FLAGDNN_POINTWISE_SUB:
        result[output_index] =
            left_value -
            static_cast<float>(specification.add_alpha) * right_value;
        break;
      case FLAGDNN_POINTWISE_MUL:
        result[output_index] = left_value * right_value;
        break;
      case FLAGDNN_POINTWISE_SIGMOID_BWD: {
        const float sigmoid =
            1.0F / (1.0F + std::exp(-right_value));
        result[output_index] =
            left_value * sigmoid * (1.0F - sigmoid);
        break;
      }
      case FLAGDNN_POINTWISE_DIV:
        result[output_index] = left_value / right_value;
        break;
      case FLAGDNN_POINTWISE_MIN:
        result[output_index] = std::min(left_value, right_value);
        break;
      case FLAGDNN_POINTWISE_MAX:
        result[output_index] = std::max(left_value, right_value);
        break;
      case FLAGDNN_POINTWISE_MOD:
        result[output_index] = std::fmod(left_value, right_value);
        break;
      case FLAGDNN_POINTWISE_POW:
        result[output_index] = std::pow(left_value, right_value);
        break;
      case FLAGDNN_POINTWISE_CMP_EQ:
        result[output_index] = left_value == right_value ? 1.0F : 0.0F;
        break;
      case FLAGDNN_POINTWISE_CMP_NEQ:
        result[output_index] = left_value != right_value ? 1.0F : 0.0F;
        break;
      case FLAGDNN_POINTWISE_CMP_GT:
        result[output_index] = left_value > right_value ? 1.0F : 0.0F;
        break;
      case FLAGDNN_POINTWISE_CMP_GE:
        result[output_index] = left_value >= right_value ? 1.0F : 0.0F;
        break;
      case FLAGDNN_POINTWISE_CMP_LT:
        result[output_index] = left_value < right_value ? 1.0F : 0.0F;
        break;
      case FLAGDNN_POINTWISE_CMP_LE:
        result[output_index] = left_value <= right_value ? 1.0F : 0.0F;
        break;
      case FLAGDNN_POINTWISE_LOGICAL_AND:
        result[output_index] =
            left_value != 0.0F && right_value != 0.0F ? 1.0F : 0.0F;
        break;
      case FLAGDNN_POINTWISE_LOGICAL_OR:
        result[output_index] =
            left_value != 0.0F || right_value != 0.0F ? 1.0F : 0.0F;
        break;
      case FLAGDNN_POINTWISE_NOT_SET:
      case FLAGDNN_POINTWISE_BINARY_SELECT:
      case FLAGDNN_POINTWISE_RELU_FWD:
      case FLAGDNN_POINTWISE_SQRT:
      case FLAGDNN_POINTWISE_ERF:
      case FLAGDNN_POINTWISE_IDENTITY:
      case FLAGDNN_POINTWISE_EXP:
      case FLAGDNN_POINTWISE_LOG:
      case FLAGDNN_POINTWISE_NEG:
      case FLAGDNN_POINTWISE_ABS:
      case FLAGDNN_POINTWISE_CEIL:
      case FLAGDNN_POINTWISE_COS:
      case FLAGDNN_POINTWISE_FLOOR:
      case FLAGDNN_POINTWISE_RSQRT:
      case FLAGDNN_POINTWISE_SIN:
      case FLAGDNN_POINTWISE_TAN:
      case FLAGDNN_POINTWISE_RECIPROCAL:
      case FLAGDNN_POINTWISE_LOGICAL_NOT:
      case FLAGDNN_POINTWISE_SIGMOID_FWD:
      case FLAGDNN_POINTWISE_TANH_FWD:
      case FLAGDNN_POINTWISE_ELU_FWD:
      case FLAGDNN_POINTWISE_GELU_FWD:
      case FLAGDNN_POINTWISE_SOFTPLUS_FWD:
      case FLAGDNN_POINTWISE_SWISH_FWD:
      case FLAGDNN_POINTWISE_GELU_APPROX_TANH_FWD:
        throw std::invalid_argument(
            "host fallback received a non-binary pointwise mode");
    }
  }
  return decode_values(
      encode_values(result, output_specification.data_type),
      output_specification.data_type);
}

std::vector<float> ternary_pointwise_host_reference(
    const BenchmarkCase& specification,
    std::span<const float> a,
    std::span<const float> b,
    std::span<const float> t) {
  if (specification.operation !=
          flagdnn::benchmarking::Operation::kPointwise ||
      specification.tensors.size() != 4 ||
      specification.pointwise_mode != FLAGDNN_POINTWISE_BINARY_SELECT) {
    throw std::invalid_argument(
        "host ternary pointwise fallback requires BINARY_SELECT");
  }
  const auto& a_specification = specification.tensors[0];
  const auto& b_specification = specification.tensors[1];
  const auto& t_specification = specification.tensors[2];
  const auto& output_specification = specification.tensors[3];
  const std::size_t output_rank = output_specification.dimensions.size();

  const auto input_value = [&](std::span<const float> values,
                               const flagdnn::benchmarking::TensorSpec& input,
                               std::span<const std::size_t> coordinates) {
    const std::size_t leading = output_rank - input.dimensions.size();
    std::size_t input_index = 0;
    std::size_t input_stride = 1;
    for (std::size_t axis = input.dimensions.size(); axis != 0; --axis) {
      const std::size_t current = axis - 1;
      const std::size_t coordinate =
          input.dimensions[current] == 1
              ? 0
              : coordinates[leading + current];
      input_index += coordinate * input_stride;
      input_stride *=
          static_cast<std::size_t>(input.dimensions[current]);
    }
    return values[input_index];
  };

  std::vector<float> result(element_count(output_specification));
  std::vector<std::size_t> coordinates(output_rank);
  for (std::size_t output_index = 0;
       output_index < result.size();
       ++output_index) {
    std::size_t remaining = output_index;
    for (std::size_t axis = output_rank; axis != 0; --axis) {
      const std::size_t current = axis - 1;
      const std::size_t dimension = static_cast<std::size_t>(
          output_specification.dimensions[current]);
      coordinates[current] = remaining % dimension;
      remaining /= dimension;
    }
    const float predicate =
        input_value(t, t_specification, coordinates);
    result[output_index] =
        predicate != 0.0F
            ? input_value(a, a_specification, coordinates)
            : input_value(b, b_specification, coordinates);
  }
  return decode_values(
      encode_values(result, output_specification.data_type),
      output_specification.data_type);
}


const flagdnn::benchmarking::TensorSpec& graph_tensor_spec(
    const BenchmarkCase& specification,
    std::int64_t uid) {
  for (const auto& tensor : specification.tensors) {
    if (tensor.uid == uid) {
      return tensor;
    }
  }
  for (const auto& tensor : specification.graph.intermediates) {
    if (tensor.uid == uid) {
      return tensor;
    }
  }
  throw std::invalid_argument(
      "host graph reference found an unknown tensor UID");
}

std::vector<float> graph_host_reference(
    const BenchmarkCase& specification,
    const std::vector<std::vector<float>>& inputs) {
  if (specification.operation != flagdnn::benchmarking::Operation::kGraph ||
      inputs.size() + 1 != specification.tensors.size()) {
    throw std::invalid_argument(
        "host graph reference received invalid external inputs");
  }
  std::unordered_map<std::int64_t, std::vector<float>> values;
  for (std::size_t index = 0; index < inputs.size(); ++index) {
    if (!values.emplace(specification.tensors[index].uid, inputs[index])
             .second) {
      throw std::invalid_argument(
          "host graph reference external tensor UID is duplicate");
    }
  }
  for (const flagdnn::benchmarking::GraphNodeSpec& node :
       specification.graph.nodes) {
    if (node.operation != flagdnn::benchmarking::Operation::kPointwise ||
        (node.input_uids.size() != 1 && node.input_uids.size() != 2)) {
      throw std::invalid_argument(
          "host graph reference currently requires pointwise nodes");
    }
    BenchmarkCase node_case;
    node_case.operation = flagdnn::benchmarking::Operation::kPointwise;
    node_case.pointwise_mode = node.pointwise_mode;
    node_case.pointwise_attributes = node.pointwise_attributes;
    node_case.add_alpha = node.alpha;
    for (const std::int64_t input_uid : node.input_uids) {
      node_case.tensors.push_back(
          graph_tensor_spec(specification, input_uid));
    }
    node_case.tensors.push_back(
        graph_tensor_spec(specification, node.output_uid));

    std::vector<float> node_output;
    if (node.input_uids.size() == 1) {
      const auto found = values.find(node.input_uids[0]);
      if (found == values.end()) {
        throw std::invalid_argument(
            "host graph nodes are not in dependency order");
      }
      node_output = pointwise_host_reference(node_case, found->second);
    } else if (node.input_uids.size() == 2) {
      const auto left = values.find(node.input_uids[0]);
      const auto right = values.find(node.input_uids[1]);
      if (left == values.end() || right == values.end()) {
        throw std::invalid_argument(
            "host graph nodes are not in dependency order");
      }
      node_output = binary_pointwise_host_reference(
          node_case, left->second, right->second);
    } else if (node.input_uids.size() == 3) {
      const auto a = values.find(node.input_uids[0]);
      const auto b = values.find(node.input_uids[1]);
      const auto t = values.find(node.input_uids[2]);
      if (a == values.end() || b == values.end() || t == values.end()) {
        throw std::invalid_argument(
            "host graph nodes are not in dependency order");
      }
      node_output = ternary_pointwise_host_reference(
          node_case, a->second, b->second, t->second);
    } else {
      throw std::invalid_argument(
          "host graph pointwise node has an invalid input count");
    }
    if (!values.emplace(node.output_uid, std::move(node_output)).second) {
      throw std::invalid_argument(
          "host graph node output UID has multiple producers");
    }
  }
  const auto output = values.find(specification.tensors.back().uid);
  if (output == values.end()) {
    throw std::invalid_argument(
        "host graph reference did not produce the external output");
  }
  return output->second;
}

std::vector<float> matmul_host_reference(
    const BenchmarkCase& specification,
    std::span<const float> a,
    std::span<const float> b) {
  if (specification.operation !=
          flagdnn::benchmarking::Operation::kMatmul ||
      specification.tensors.size() != 3) {
    throw std::invalid_argument(
        "host MatMul fallback requires two inputs and one output");
  }
  const auto& a_specification = specification.tensors[0];
  const auto& b_specification = specification.tensors[1];
  const auto& output_specification = specification.tensors[2];
  const std::size_t output_rank = output_specification.dimensions.size();
  const std::size_t batch_rank = output_rank - 2;
  const std::size_t m = static_cast<std::size_t>(
      output_specification.dimensions[output_rank - 2]);
  const std::size_t n = static_cast<std::size_t>(
      output_specification.dimensions[output_rank - 1]);
  const std::size_t k = static_cast<std::size_t>(
      a_specification.dimensions.back());
  const std::size_t batch_count =
      element_count(output_specification) / (m * n);
  const auto input_batch_offset =
      [&](const flagdnn::benchmarking::TensorSpec& input,
          std::span<const std::size_t> coordinates) {
        const std::size_t input_batch_rank = input.dimensions.size() - 2;
        const std::size_t leading = batch_rank - input_batch_rank;
        std::size_t result = 0;
        for (std::size_t axis = 0; axis < input_batch_rank; ++axis) {
          const std::size_t dimension = static_cast<std::size_t>(
              input.dimensions[axis]);
          result *= dimension;
          if (dimension != 1) {
            result += coordinates[leading + axis];
          }
        }
        return result;
      };

  std::vector<float> result(element_count(output_specification));
  std::vector<std::size_t> batch_coordinates(batch_rank);
  for (std::size_t batch = 0; batch < batch_count; ++batch) {
    std::size_t remaining = batch;
    for (std::size_t axis = batch_rank; axis != 0; --axis) {
      const std::size_t current = axis - 1;
      const std::size_t dimension = static_cast<std::size_t>(
          output_specification.dimensions[current]);
      batch_coordinates[current] = remaining % dimension;
      remaining /= dimension;
    }
    const std::size_t a_base =
        input_batch_offset(a_specification, batch_coordinates) * m * k;
    const std::size_t b_base =
        input_batch_offset(b_specification, batch_coordinates) * k * n;
    const std::size_t output_base = batch * m * n;
    for (std::size_t row = 0; row < m; ++row) {
      for (std::size_t column = 0; column < n; ++column) {
        float accumulator = 0.0F;
        for (std::size_t reduction = 0; reduction < k; ++reduction) {
          accumulator += a[a_base + row * k + reduction] *
                         b[b_base + reduction * n + column];
        }
        result[output_base + row * n + column] = accumulator;
      }
    }
  }
  return decode_values(
      encode_values(result, output_specification.data_type),
      output_specification.data_type);
}

std::vector<float> layout_host_reference(
    const BenchmarkCase& specification,
    std::span<const float> input) {
  if (specification.tensors.size() != 2) {
    throw std::invalid_argument(
        "host layout fallback requires one input and one output");
  }
  const auto& input_specification = specification.tensors[0];
  const auto& output_specification = specification.tensors[1];
  std::vector<float> result(element_count(output_specification));
  const auto row_major_index = [](
      std::span<const std::size_t> coordinates,
      std::span<const std::int64_t> dimensions) {
    std::size_t index = 0;
    for (std::size_t axis = 0; axis < dimensions.size(); ++axis) {
      index = index * static_cast<std::size_t>(dimensions[axis]) +
              coordinates[axis];
    }
    return index;
  };


  if (specification.operation ==
      flagdnn::benchmarking::Operation::kReshape) {
    if (input.size() != result.size()) {
      throw std::invalid_argument(
          "host Reshape input/output element counts differ");
    }
    std::copy(input.begin(), input.end(), result.begin());
  } else if (specification.operation ==
             flagdnn::benchmarking::Operation::kTranspose) {
    const auto& permutation = specification.transpose.permutation;
    const std::size_t rank = output_specification.dimensions.size();
    if (permutation.size() != rank) {
      throw std::invalid_argument(
          "host Transpose permutation rank is invalid");
    }
    std::vector<std::size_t> output_coordinates(rank);
    std::vector<std::size_t> input_coordinates(rank);
    for (std::size_t output_index = 0;
         output_index < result.size();
         ++output_index) {
      std::size_t remaining = output_index;
      for (std::size_t axis = rank; axis != 0; --axis) {
        const std::size_t current = axis - 1;
        const std::size_t dimension = static_cast<std::size_t>(
            output_specification.dimensions[current]);
        output_coordinates[current] = remaining % dimension;
        remaining /= dimension;
      }
      for (std::size_t axis = 0; axis < rank; ++axis) {
        input_coordinates[static_cast<std::size_t>(permutation[axis])] =
            output_coordinates[axis];
      }
      result[output_index] = input[row_major_index(
          input_coordinates, input_specification.dimensions)];
    }
  } else if (specification.operation ==
             flagdnn::benchmarking::Operation::kSlice) {
    const std::size_t rank = output_specification.dimensions.size();
    if (specification.slice.slices.size() != rank) {
      throw std::invalid_argument("host Slice range rank is invalid");
    }
    std::vector<std::size_t> output_coordinates(rank);
    std::vector<std::size_t> input_coordinates(rank);
    for (std::size_t output_index = 0;
         output_index < result.size();
         ++output_index) {
      std::size_t remaining = output_index;
      for (std::size_t axis = rank; axis != 0; --axis) {
        const std::size_t current = axis - 1;
        const std::size_t dimension = static_cast<std::size_t>(
            output_specification.dimensions[current]);
        output_coordinates[current] = remaining % dimension;
        remaining /= dimension;
      }
      for (std::size_t axis = 0; axis < rank; ++axis) {
        const std::int64_t step =
            axis < specification.slice.strides.size()
                ? specification.slice.strides[axis]
                : 1;
        input_coordinates[axis] = static_cast<std::size_t>(
            specification.slice.slices[axis].first +
            static_cast<std::int64_t>(output_coordinates[axis]) * step);
      }
      result[output_index] = input[row_major_index(
          input_coordinates, input_specification.dimensions)];
    }
  } else {
    throw std::invalid_argument(
        "host layout fallback received an unsupported operation");
  }
  return decode_values(
      encode_values(result, output_specification.data_type),
      output_specification.data_type);
}

std::vector<float> relu_host_reference(
    const BenchmarkCase& specification,
    std::span<const float> input) {
  if (specification.operation != flagdnn::benchmarking::Operation::kRelu ||
      specification.tensors.size() != 2) {
    throw std::invalid_argument(
        "host ReLU fallback requires one input and one output");
  }
  BenchmarkCase pointwise = specification;
  pointwise.operation = flagdnn::benchmarking::Operation::kPointwise;
  pointwise.pointwise_mode = FLAGDNN_POINTWISE_RELU_FWD;
  return pointwise_host_reference(pointwise, input);
}

std::vector<float> convolution_backward_host_reference(
    const BenchmarkCase& specification,
    std::span<const float> loss,
    std::span<const float> other) {
  const bool data_gradient =
      specification.operation ==
      flagdnn::benchmarking::Operation::kConvolutionDgrad;
  if ((!data_gradient &&
       specification.operation !=
           flagdnn::benchmarking::Operation::kConvolutionWgrad) ||
      specification.tensors.size() != 3) {
    throw std::invalid_argument(
        "host convolution backward requires two inputs and one output");
  }
  const auto& loss_specification = specification.tensors[0];
  const auto& other_specification = specification.tensors[1];
  const auto& output_specification = specification.tensors[2];
  const auto& image_specification =
      data_gradient ? output_specification : other_specification;
  const auto& filter_specification =
      data_gradient ? other_specification : output_specification;
  const std::size_t spatial_rank =
      static_cast<std::size_t>(specification.convolution.spatial_rank);
  if (spatial_rank == 0 || spatial_rank > 3 ||
      image_specification.dimensions.size() != spatial_rank + 2 ||
      filter_specification.dimensions.size() != spatial_rank + 2 ||
      loss_specification.dimensions.size() != spatial_rank + 2) {
    throw std::invalid_argument(
        "host convolution backward tensor rank is invalid");
  }

  const auto product = [](
      std::span<const std::int64_t> dimensions) -> std::size_t {
    std::size_t result = 1;
    for (const std::int64_t dimension : dimensions) {
      result *= static_cast<std::size_t>(dimension);
    }
    return result;
  };
  const auto decode_coordinates = [](
      std::size_t flat,
      std::span<const std::int64_t> dimensions) {
    std::vector<std::size_t> result(dimensions.size());
    for (std::size_t axis = dimensions.size(); axis != 0; --axis) {
      const std::size_t current = axis - 1;
      const std::size_t dimension =
          static_cast<std::size_t>(dimensions[current]);
      result[current] = flat % dimension;
      flat /= dimension;
    }
    return result;
  };
  const auto encode_coordinates = [](
      std::span<const std::size_t> coordinates,
      std::span<const std::int64_t> dimensions) {
    std::size_t result = 0;
    for (std::size_t axis = 0; axis < dimensions.size(); ++axis) {
      result = result * static_cast<std::size_t>(dimensions[axis]) +
               coordinates[axis];
    }
    return result;
  };

  const std::span<const std::int64_t> image_spatial(
      image_specification.dimensions.data() + 2, spatial_rank);
  const std::span<const std::int64_t> filter_spatial(
      filter_specification.dimensions.data() + 2, spatial_rank);
  const std::span<const std::int64_t> loss_spatial(
      loss_specification.dimensions.data() + 2, spatial_rank);
  const std::size_t image_spatial_elements = product(image_spatial);
  const std::size_t filter_spatial_elements = product(filter_spatial);
  const std::size_t loss_spatial_elements = product(loss_spatial);
  const std::size_t batch = static_cast<std::size_t>(
      image_specification.dimensions[0]);
  const std::size_t channels = static_cast<std::size_t>(
      image_specification.dimensions[1]);
  const std::size_t output_channels = static_cast<std::size_t>(
      filter_specification.dimensions[0]);
  const std::size_t groups = static_cast<std::size_t>(
      specification.convolution.groups);
  const std::size_t channels_per_group = channels / groups;
  const std::size_t outputs_per_group = output_channels / groups;
  std::vector<float> result(element_count(output_specification), 0.0F);

  for (std::size_t n = 0; n < batch; ++n) {
    for (std::size_t k = 0; k < output_channels; ++k) {
      const std::size_t group = k / outputs_per_group;
      const std::size_t channel_base = group * channels_per_group;
      for (std::size_t loss_flat = 0;
           loss_flat < loss_spatial_elements;
           ++loss_flat) {
        const auto loss_coordinates =
            decode_coordinates(loss_flat, loss_spatial);
        const float dy = loss[
            (n * output_channels + k) * loss_spatial_elements + loss_flat];
        for (std::size_t local_channel = 0;
             local_channel < channels_per_group;
             ++local_channel) {
          const std::size_t channel = channel_base + local_channel;
          for (std::size_t filter_flat = 0;
               filter_flat < filter_spatial_elements;
               ++filter_flat) {
            const auto filter_coordinates =
                decode_coordinates(filter_flat, filter_spatial);
            std::vector<std::size_t> image_coordinates(spatial_rank);
            bool valid = true;
            for (std::size_t axis = 0; axis < spatial_rank; ++axis) {
              const std::int64_t coordinate =
                  static_cast<std::int64_t>(loss_coordinates[axis]) *
                      specification.convolution.stride[axis] -
                  specification.convolution.pre_padding[axis] +
                  static_cast<std::int64_t>(filter_coordinates[axis]) *
                      specification.convolution.dilation[axis];
              if (coordinate < 0 ||
                  coordinate >= image_spatial[axis]) {
                valid = false;
                break;
              }
              image_coordinates[axis] =
                  static_cast<std::size_t>(coordinate);
            }
            if (!valid) {
              continue;
            }
            std::vector<std::size_t> stored_filter_coordinates =
                filter_coordinates;
            if (specification.convolution.mode ==
                flagdnn::benchmarking::ConvolutionMode::kConvolution) {
              for (std::size_t axis = 0; axis < spatial_rank; ++axis) {
                stored_filter_coordinates[axis] =
                    static_cast<std::size_t>(filter_spatial[axis] - 1) -
                    filter_coordinates[axis];
              }
            }
            const std::size_t image_flat = encode_coordinates(
                image_coordinates, image_spatial);
            const std::size_t stored_filter_flat = encode_coordinates(
                stored_filter_coordinates, filter_spatial);
            const std::size_t image_index =
                (n * channels + channel) * image_spatial_elements +
                image_flat;
            const std::size_t filter_index =
                (k * channels_per_group + local_channel) *
                    filter_spatial_elements +
                stored_filter_flat;
            if (data_gradient) {
              result[image_index] += dy * other[filter_index];
            } else {
              result[filter_index] += dy * other[image_index];
            }
          }
        }
      }
    }
  }
  return decode_values(
      encode_values(result, output_specification.data_type),
      output_specification.data_type);
}

std::vector<float> batchnorm_inference_host_reference(
    const BenchmarkCase& specification,
    std::span<const float> x,
    std::span<const float> mean,
    std::span<const float> inv_variance,
    std::span<const float> scale,
    std::span<const float> bias) {
  if (specification.operation !=
          flagdnn::benchmarking::Operation::kBatchnormInference ||
      specification.tensors.size() != 6) {
    throw std::invalid_argument(
        "host BatchNorm Inference requires five inputs and one output");
  }
  const auto& x_specification = specification.tensors[0];
  const auto& output_specification = specification.tensors[5];
  const std::size_t channels = static_cast<std::size_t>(
      x_specification.dimensions[1]);
  std::size_t spatial = 1;
  for (std::size_t axis = 2;
       axis < x_specification.dimensions.size();
       ++axis) {
    spatial *= static_cast<std::size_t>(
        x_specification.dimensions[axis]);
  }
  if (mean.size() != channels || inv_variance.size() != channels ||
      scale.size() != channels || bias.size() != channels ||
      x.size() != element_count(x_specification)) {
    throw std::invalid_argument(
        "host BatchNorm Inference input sizes are invalid");
  }
  std::vector<float> result(x.size());
  for (std::size_t index = 0; index < x.size(); ++index) {
    const std::size_t channel = (index / spatial) % channels;
    result[index] = (x[index] - mean[channel]) *
                    inv_variance[channel] * scale[channel] +
                    bias[channel];
  }
  return decode_values(
      encode_values(result, output_specification.data_type),
      output_specification.data_type);
}

std::vector<std::vector<float>> batchnorm_host_reference(
    const BenchmarkCase& specification,
    const std::vector<std::vector<float>>& inputs) {
  if (specification.operation !=
          flagdnn::benchmarking::Operation::kBatchnorm ||
      specification.tensors.size() != 10 ||
      specification.output_count != 5 || inputs.size() != 5) {
    throw std::invalid_argument(
        "host BatchNorm requires five inputs and five outputs");
  }
  const auto& x_specification = specification.tensors[0];
  const std::size_t batch = static_cast<std::size_t>(
      x_specification.dimensions[0]);
  const std::size_t channels = static_cast<std::size_t>(
      x_specification.dimensions[1]);
  std::size_t spatial = 1;
  for (std::size_t axis = 2;
       axis < x_specification.dimensions.size();
       ++axis) {
    spatial *= static_cast<std::size_t>(
        x_specification.dimensions[axis]);
  }
  const std::size_t reduction_elements = batch * spatial;
  if (inputs[0].size() != batch * channels * spatial) {
    throw std::invalid_argument("host BatchNorm X size is invalid");
  }
  for (std::size_t index = 1; index < inputs.size(); ++index) {
    if (inputs[index].size() != channels) {
      throw std::invalid_argument(
          "host BatchNorm parameter size is invalid");
    }
  }

  std::vector<float> y(inputs[0].size());
  std::vector<float> mean(channels);
  std::vector<float> inv_variance(channels);
  std::vector<float> next_mean(channels);
  std::vector<float> next_variance(channels);
  const double epsilon = specification.normalization.epsilon;
  const double momentum = specification.normalization.momentum;
  for (std::size_t channel = 0; channel < channels; ++channel) {
    double sum = 0.0;
    double sum_square = 0.0;
    for (std::size_t n = 0; n < batch; ++n) {
      for (std::size_t spatial_index = 0;
           spatial_index < spatial;
           ++spatial_index) {
        const std::size_t index =
            (n * channels + channel) * spatial + spatial_index;
        const double value = inputs[0][index];
        sum += value;
        sum_square += value * value;
      }
    }
    const double channel_mean =
        sum / static_cast<double>(reduction_elements);
    const double variance = std::max(
        0.0,
        sum_square / static_cast<double>(reduction_elements) -
            channel_mean * channel_mean);
    const double channel_inv_variance =
        1.0 / std::sqrt(variance + epsilon);
    const double unbiased_variance =
        reduction_elements > 1
            ? variance * static_cast<double>(reduction_elements) /
                  static_cast<double>(reduction_elements - 1)
            : variance;
    mean[channel] = static_cast<float>(channel_mean);
    inv_variance[channel] =
        static_cast<float>(channel_inv_variance);
    next_mean[channel] = static_cast<float>(
        inputs[3][channel] * (1.0 - momentum) +
        channel_mean * momentum);
    next_variance[channel] = static_cast<float>(
        inputs[4][channel] * (1.0 - momentum) +
        unbiased_variance * momentum);
    for (std::size_t n = 0; n < batch; ++n) {
      for (std::size_t spatial_index = 0;
           spatial_index < spatial;
           ++spatial_index) {
        const std::size_t index =
            (n * channels + channel) * spatial + spatial_index;
        y[index] = static_cast<float>(
            (inputs[0][index] - channel_mean) *
                channel_inv_variance * inputs[1][channel] +
            inputs[2][channel]);
      }
    }
  }
  std::vector<std::vector<float>> result = {
      std::move(y),
      std::move(mean),
      std::move(inv_variance),
      std::move(next_mean),
      std::move(next_variance),
  };
  for (std::size_t output_index = 0;
       output_index < result.size();
       ++output_index) {
    const auto& output = output_tensor(specification, output_index);
    result[output_index] = decode_values(
        encode_values(result[output_index], output.data_type),
        output.data_type);
  }
  return result;
}

std::vector<std::vector<float>> layernorm_host_reference(
    const BenchmarkCase& specification,
    const std::vector<std::vector<float>>& inputs) {
  if (specification.operation !=
          flagdnn::benchmarking::Operation::kLayernorm ||
      specification.tensors.size() != 6 ||
      specification.output_count != 3 || inputs.size() != 3) {
    throw std::invalid_argument(
        "host LayerNorm requires three inputs and three outputs");
  }
  const std::size_t rows =
      element_count(output_tensor(specification, 1));
  const std::size_t normalized_elements = inputs[0].size() / rows;
  if (rows == 0 || rows * normalized_elements != inputs[0].size() ||
      inputs[1].size() != normalized_elements ||
      inputs[2].size() != normalized_elements) {
    throw std::invalid_argument("host LayerNorm input sizes are invalid");
  }
  std::vector<float> y(inputs[0].size());
  std::vector<float> mean(rows);
  std::vector<float> inv_variance(rows);
  for (std::size_t row = 0; row < rows; ++row) {
    double sum = 0.0;
    double sum_square = 0.0;
    for (std::size_t column = 0;
         column < normalized_elements;
         ++column) {
      const double value =
          inputs[0][row * normalized_elements + column];
      sum += value;
      sum_square += value * value;
    }
    const double row_mean = sum / normalized_elements;
    const double variance = std::max(
        0.0, sum_square / normalized_elements - row_mean * row_mean);
    const double row_inv_variance = 1.0 / std::sqrt(
        variance + specification.normalization.epsilon);
    mean[row] = static_cast<float>(row_mean);
    inv_variance[row] = static_cast<float>(row_inv_variance);
    for (std::size_t column = 0;
         column < normalized_elements;
         ++column) {
      const std::size_t index = row * normalized_elements + column;
      y[index] = static_cast<float>(
          (inputs[0][index] - row_mean) * row_inv_variance *
              inputs[1][column] +
          inputs[2][column]);
    }
  }
  std::vector<std::vector<float>> result = {
      std::move(y), std::move(mean), std::move(inv_variance)};
  for (std::size_t output_index = 0;
       output_index < result.size();
       ++output_index) {
    const auto& output = output_tensor(specification, output_index);
    result[output_index] = decode_values(
        encode_values(result[output_index], output.data_type),
        output.data_type);
  }
  return result;
}

std::vector<std::vector<float>> rmsnorm_host_reference(
    const BenchmarkCase& specification,
    const std::vector<std::vector<float>>& inputs) {
  if (specification.operation !=
          flagdnn::benchmarking::Operation::kRmsnorm ||
      specification.tensors.size() != 5 ||
      specification.output_count != 2 || inputs.size() != 3) {
    throw std::invalid_argument(
        "host RMSNorm requires three inputs and two outputs");
  }
  const std::size_t rows =
      element_count(output_tensor(specification, 1));
  const std::size_t normalized_elements = inputs[0].size() / rows;
  if (rows == 0 || rows * normalized_elements != inputs[0].size() ||
      inputs[1].size() != normalized_elements ||
      inputs[2].size() != normalized_elements) {
    throw std::invalid_argument("host RMSNorm input sizes are invalid");
  }
  std::vector<float> y(inputs[0].size());
  std::vector<float> inv_variance(rows);
  for (std::size_t row = 0; row < rows; ++row) {
    double sum_square = 0.0;
    for (std::size_t column = 0;
         column < normalized_elements;
         ++column) {
      const double value =
          inputs[0][row * normalized_elements + column];
      sum_square += value * value;
    }
    const double row_inv_variance = 1.0 / std::sqrt(
        sum_square / normalized_elements +
        specification.normalization.epsilon);
    inv_variance[row] = static_cast<float>(row_inv_variance);
    for (std::size_t column = 0;
         column < normalized_elements;
         ++column) {
      const std::size_t index = row * normalized_elements + column;
      y[index] = static_cast<float>(
          inputs[0][index] * row_inv_variance * inputs[1][column] +
          inputs[2][column]);
    }
  }
  std::vector<std::vector<float>> result = {
      std::move(y), std::move(inv_variance)};
  for (std::size_t output_index = 0;
       output_index < result.size();
       ++output_index) {
    const auto& output = output_tensor(specification, output_index);
    result[output_index] = decode_values(
        encode_values(result[output_index], output.data_type),
        output.data_type);
  }
  return result;
}

std::vector<float> host_reference(
    const BenchmarkCase& specification,
    const std::vector<std::vector<float>>& inputs) {
  if (inputs.size() != input_tensor_count(specification)) {
    throw std::invalid_argument(
        "host reference input count does not match the case");
  }
  switch (specification.operation) {
    case flagdnn::benchmarking::Operation::kRelu:
      return relu_host_reference(specification, inputs.at(0));
    case flagdnn::benchmarking::Operation::kPointwise:
      if (inputs.size() == 1) {
        return pointwise_host_reference(specification, inputs[0]);
      }
      if (inputs.size() == 2) {
        return binary_pointwise_host_reference(
            specification, inputs[0], inputs[1]);
      }
      if (inputs.size() == 3) {
        return ternary_pointwise_host_reference(
            specification, inputs[0], inputs[1], inputs[2]);
      }
      break;
    case flagdnn::benchmarking::Operation::kAdd:
      return add_host_reference(
          specification, inputs.at(0), inputs.at(1));
    case flagdnn::benchmarking::Operation::kReduction:
      return reduction_host_reference(specification, inputs.at(0));
    case flagdnn::benchmarking::Operation::kMatmul:
      return matmul_host_reference(
          specification, inputs.at(0), inputs.at(1));
    case flagdnn::benchmarking::Operation::kReshape:
    case flagdnn::benchmarking::Operation::kTranspose:
    case flagdnn::benchmarking::Operation::kSlice:
      return layout_host_reference(specification, inputs.at(0));
    case flagdnn::benchmarking::Operation::kConvolutionDgrad:
    case flagdnn::benchmarking::Operation::kConvolutionWgrad:
      return convolution_backward_host_reference(
          specification, inputs.at(0), inputs.at(1));
    case flagdnn::benchmarking::Operation::kBatchnormInference:
      return batchnorm_inference_host_reference(
          specification,
          inputs.at(0),
          inputs.at(1),
          inputs.at(2),
          inputs.at(3),
          inputs.at(4));
    case flagdnn::benchmarking::Operation::kGraph:
      return graph_host_reference(specification, inputs);
    case flagdnn::benchmarking::Operation::kConvolutionFprop:
      throw std::invalid_argument(
          "host fallback for Convolution FProp is not implemented");
    case flagdnn::benchmarking::Operation::kLayernorm:
    case flagdnn::benchmarking::Operation::kRmsnorm:
      throw std::invalid_argument(
          "normalization uses the multi-output host reference");
    case flagdnn::benchmarking::Operation::kBatchnorm:
      throw std::invalid_argument(
          "BatchNorm training uses the multi-output host reference");
  }
  throw std::invalid_argument(
      "host fallback received an unsupported input arity");
}

std::vector<std::vector<float>> host_references(
    const BenchmarkCase& specification,
    const std::vector<std::vector<float>>& inputs) {
  if (specification.operation ==
      flagdnn::benchmarking::Operation::kLayernorm) {
    return layernorm_host_reference(specification, inputs);
  }
  if (specification.operation ==
      flagdnn::benchmarking::Operation::kRmsnorm) {
    return rmsnorm_host_reference(specification, inputs);
  }
  if (specification.operation ==
      flagdnn::benchmarking::Operation::kBatchnorm) {
    return batchnorm_host_reference(specification, inputs);
  }
  return {host_reference(specification, inputs)};
}

std::unique_ptr<DeviceBuffer> make_workspace(
    const BenchmarkExecutable& executable) {
  return std::make_unique<DeviceBuffer>(executable.workspace_size());
}

void execute(
    BenchmarkExecutable& executable,
    std::span<const flagdnnBinding_t> bindings,
    DeviceBuffer& workspace,
    flagdnn::benchmarking::Stream& stream) {
  executable.execute(
      bindings,
      workspace.opaque(),
      executable.workspace_size(),
      stream.opaque());
}

Accuracy run_flagdnn_with_host_reference(
    const BenchmarkCase& specification,
    BenchmarkExecutable& executable,
    flagdnn::benchmarking::Stream& stream,
    CaseBuffers& buffers) {
  auto workspace = make_workspace(executable);
  execute(executable, buffers.flagdnn_bindings, *workspace, stream);
  stream.synchronize();

  const auto references =
      host_references(specification, buffers.host_inputs);
  if (references.size() != specification.output_count) {
    throw std::runtime_error(
        "host reference output count does not match the case");
  }

  Accuracy aggregate;
  const std::size_t input_count = input_tensor_count(specification);
  for (std::size_t output_index = 0;
       output_index < specification.output_count;
       ++output_index) {
    const auto& output =
        specification.tensors[input_count + output_index];
    const std::size_t storage_count = storage_element_count(output);
    std::vector<std::uint8_t> encoded(
        storage_count * data_type_size(output.data_type));
    buffers.flagdnn_outputs[output_index]->copy_to_host_at(
        encoded.data(),
        encoded.size(),
        output.binding_byte_offset,
        stream.get());
    stream.synchronize();
    const std::vector<float> physical =
        decode_values(encoded, output.data_type);
    require_output_padding_unchanged(
        "FlagDNN", physical, output);
    const std::vector<float> actual =
        gather_logical_values(physical, output);
    const Accuracy accuracy = compare_outputs(
        actual,
        references[output_index],
        specification.absolute_tolerance,
        specification.relative_tolerance);
    aggregate.maximum_absolute = std::max(
        aggregate.maximum_absolute, accuracy.maximum_absolute);
    aggregate.maximum_relative = std::max(
        aggregate.maximum_relative, accuracy.maximum_relative);
  }
  std::cout << specification.name
            << ": FlagDNN vs host PASS max_abs="
            << aggregate.maximum_absolute
            << " max_rel=" << aggregate.maximum_relative << std::endl;
  return aggregate;
}

Accuracy compare_provider_outputs(
    const BenchmarkCase& specification,
    flagdnn::benchmarking::Stream& stream,
    CaseBuffers& buffers) {
  const std::size_t input_count = input_tensor_count(specification);
  Accuracy aggregate;
  for (std::size_t output_index = 0;
       output_index < specification.output_count;
       ++output_index) {
    const std::size_t tensor_index = input_count + output_index;
    const auto& flagdnn_output = specification.tensors[tensor_index];
    const auto cudnn_output =
        cudnn_tensor_specification(specification, tensor_index);

    std::vector<std::uint8_t> flagdnn_encoded(
        storage_element_count(flagdnn_output) *
        data_type_size(flagdnn_output.data_type));
    std::vector<std::uint8_t> cudnn_encoded(
        cudnn_encoded_byte_count(cudnn_output));
    buffers.flagdnn_outputs[output_index]->copy_to_host_at(
        flagdnn_encoded.data(),
        flagdnn_encoded.size(),
        flagdnn_output.binding_byte_offset,
        stream.get());
    buffers.cudnn_outputs[output_index]->copy_to_host_at(
        cudnn_encoded.data(),
        cudnn_encoded.size(),
        cudnn_output.binding_byte_offset,
        stream.get());
    stream.synchronize();

    const std::vector<float> flagdnn_physical =
        decode_values(flagdnn_encoded, flagdnn_output.data_type);
    const std::vector<float> cudnn_physical =
        decode_cudnn_values(
            cudnn_encoded,
            cudnn_output.data_type,
            storage_element_count(cudnn_output));
    require_output_padding_unchanged(
        "FlagDNN", flagdnn_physical, flagdnn_output);
    if (cudnn_output.data_type != FLAGDNN_DATA_BOOLEAN) {
      require_output_padding_unchanged(
          "cuDNN", cudnn_physical, cudnn_output);
    }
    const std::vector<float> flagdnn_logical =
        gather_logical_values(flagdnn_physical, flagdnn_output);
    const std::vector<float> cudnn_logical =
        gather_logical_values(cudnn_physical, cudnn_output);
    Accuracy accuracy;
    try {
      accuracy = compare_outputs(
          flagdnn_logical,
          cudnn_logical,
          specification.absolute_tolerance,
          specification.relative_tolerance);
    } catch (const std::runtime_error& error) {
      throw std::runtime_error(
          specification.name + " output_index=" +
          std::to_string(output_index) + ": " + error.what());
    }
    aggregate.maximum_absolute = std::max(
        aggregate.maximum_absolute, accuracy.maximum_absolute);
    aggregate.maximum_relative = std::max(
        aggregate.maximum_relative, accuracy.maximum_relative);
  }
  std::cout << specification.name
            << ": FlagDNN vs cuDNN PASS max_abs="
            << aggregate.maximum_absolute
            << " max_rel=" << aggregate.maximum_relative << std::endl;
  return aggregate;
}

void warmup(
    BenchmarkExecutable& executable,
    std::span<const flagdnnBinding_t> bindings,
    DeviceBuffer& workspace,
    flagdnn::benchmarking::Stream& stream,
    int iterations) {
  for (int index = 0; index < iterations; ++index) {
    execute(executable, bindings, workspace, stream);
  }
  stream.synchronize();
}

std::vector<double> measure_provider(
    BenchmarkExecutable& executable,
    std::span<const flagdnnBinding_t> bindings,
    DeviceBuffer& workspace,
    flagdnn::benchmarking::Stream& stream,
    const flagdnn::benchmarking::BenchmarkConfig& benchmark) {
  CapturedExecutionBatch batch(
      stream.get(), benchmark.iterations_per_sample, [&] {
        execute(executable, bindings, workspace, stream);
      });
  batch.launch(stream.get());
  stream.synchronize();

  EventTimer timer;
  std::vector<double> samples;
  samples.reserve(static_cast<std::size_t>(benchmark.sample_count));
  for (int sample = 0; sample < benchmark.sample_count; ++sample) {
    const double batch_microseconds = timer.measure_microseconds(
        stream.get(), 1, [&] { batch.launch(stream.get()); });
    samples.push_back(
        batch_microseconds / static_cast<double>(batch.execution_count()));
  }
  return samples;
}

void run_case(
    const BenchmarkCase& specification,
    flagdnn::benchmarking::FlagdnnProvider& flagdnn_provider,
    flagdnn::benchmarking::CudnnProvider& cudnn_provider,
    flagdnn::benchmarking::Stream& stream) {
  std::unique_ptr<BenchmarkExecutable> flagdnn_executable =
      flagdnn_provider.build(specification);
  CaseBuffers buffers = make_buffers(specification, stream);
  stream.synchronize();

  const auto run_without_cudnn = [&](std::string_view reason) {
    run_flagdnn_with_host_reference(
        specification, *flagdnn_executable, stream, buffers);
    std::cout << specification.name
              << ": cuDNN UNSUPPORTED: " << reason << std::endl;
    auto workspace = make_workspace(*flagdnn_executable);
    warmup(
        *flagdnn_executable,
        buffers.flagdnn_bindings,
        *workspace,
        stream,
        specification.benchmark.warmup_iterations);
    const auto samples = measure_provider(
        *flagdnn_executable,
        buffers.flagdnn_bindings,
        *workspace,
        stream,
        specification.benchmark);
    emit_samples("flagdnn", specification, samples);
  };

  const auto capability = cudnn_provider.capability(specification);
  if (!capability.supported) {
    run_without_cudnn(capability.reason);
    return;
  }

  std::unique_ptr<BenchmarkExecutable> cudnn_executable;
  try {
    cudnn_executable = cudnn_provider.build(specification);
  } catch (const BenchmarkUnsupportedError& error) {
    run_without_cudnn(error.what());
    return;
  }

  auto flagdnn_workspace = make_workspace(*flagdnn_executable);
  auto cudnn_workspace = make_workspace(*cudnn_executable);
  execute(
      *flagdnn_executable,
      buffers.flagdnn_bindings,
      *flagdnn_workspace,
      stream);
  execute(
      *cudnn_executable,
      buffers.cudnn_bindings,
      *cudnn_workspace,
      stream);
  stream.synchronize();
  compare_provider_outputs(specification, stream, buffers);

  warmup(
      *flagdnn_executable,
      buffers.flagdnn_bindings,
      *flagdnn_workspace,
      stream,
      specification.benchmark.warmup_iterations);
  warmup(
      *cudnn_executable,
      buffers.cudnn_bindings,
      *cudnn_workspace,
      stream,
      specification.benchmark.warmup_iterations);

  CapturedExecutionBatch flagdnn_batch(
      stream.get(), specification.benchmark.iterations_per_sample, [&] {
        execute(
            *flagdnn_executable,
            buffers.flagdnn_bindings,
            *flagdnn_workspace,
            stream);
      });
  CapturedExecutionBatch cudnn_batch(
      stream.get(), specification.benchmark.iterations_per_sample, [&] {
        execute(
            *cudnn_executable,
            buffers.cudnn_bindings,
            *cudnn_workspace,
            stream);
      });
  flagdnn_batch.launch(stream.get());
  cudnn_batch.launch(stream.get());
  stream.synchronize();

  EventTimer flagdnn_timer;
  EventTimer cudnn_timer;
  std::vector<double> flagdnn_samples;
  std::vector<double> cudnn_samples;
  flagdnn_samples.reserve(
      static_cast<std::size_t>(specification.benchmark.sample_count));
  cudnn_samples.reserve(
      static_cast<std::size_t>(specification.benchmark.sample_count));
  for (int sample = 0;
       sample < specification.benchmark.sample_count;
       ++sample) {
    const auto measure_flagdnn = [&] {
      const double batch_microseconds = flagdnn_timer.measure_microseconds(
          stream.get(), 1, [&] { flagdnn_batch.launch(stream.get()); });
      flagdnn_samples.push_back(
          batch_microseconds /
          static_cast<double>(flagdnn_batch.execution_count()));
    };
    const auto measure_cudnn = [&] {
      const double batch_microseconds = cudnn_timer.measure_microseconds(
          stream.get(), 1, [&] { cudnn_batch.launch(stream.get()); });
      cudnn_samples.push_back(
          batch_microseconds /
          static_cast<double>(cudnn_batch.execution_count()));
    };
    if (sample % 2 == 0) {
      measure_flagdnn();
      measure_cudnn();
    } else {
      measure_cudnn();
      measure_flagdnn();
    }
  }
  emit_samples("flagdnn", specification, flagdnn_samples);
  emit_samples("cudnn", specification, cudnn_samples);
  const double flagdnn_median = percentile(flagdnn_samples, 0.5);
  const double cudnn_median = percentile(cudnn_samples, 0.5);
  std::cout << specification.name
            << ": median_us flagdnn=" << flagdnn_median
            << " cudnn=" << cudnn_median
            << " speedup=" << cudnn_median / flagdnn_median
            << std::endl;
}

}  // namespace

namespace flagdnn::benchmarking {

int run_benchmark_suite(
    int argc,
    char** argv,
    std::span<const BenchmarkCase> cases,
    std::string_view suite_name) {
  if (argc != 3) {
    std::cerr << "usage: " << suite_name
              << " COMPILER_EXECUTABLE COMPILER_ENTRY" << std::endl;
    return 2;
  }
  try {
    std::cout << std::setprecision(9);
    DriverContext driver;
    Stream stream;
    BenchmarkCache cache;
    flagdnn::Handle handle(FLAGDNN_BACKEND_NVIDIA, 0);
    handle.set_compiler(argv[1], argv[2], cache.path().string());
    FlagdnnProvider flagdnn_provider(handle);
    CudnnProvider cudnn_provider;
    const char* case_filter = std::getenv("FLAGDNN_BENCHMARK_CASE");
    std::size_t executed_cases = 0;
    for (std::size_t case_index = 0; case_index < cases.size(); ++case_index) {
      const BenchmarkCase& specification = cases[case_index];
      if (case_filter != nullptr && case_filter[0] != '\0' &&
          specification.name != case_filter) {
        continue;
      }
      flagdnn_provider.set_autotune(true);
      run_case(
          specification,
          flagdnn_provider,
          cudnn_provider,
          stream);
      ++executed_cases;
    }
    if (executed_cases == 0) {
      throw std::invalid_argument(
          "FLAGDNN_BENCHMARK_CASE did not match any case");
    }
    std::cout << suite_name << ": PASS cases=" << executed_cases
              << std::endl;
    return 0;
  } catch (const std::exception& error) {
    std::cerr << suite_name << "_FAILED: " << error.what()
              << std::endl;
    return 1;
  }
}

}  // namespace flagdnn::benchmarking
