/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/normalization.hpp"
#include "hip_driver.hpp"
#include "host_runner.hpp"
#include "normalization_reference.hpp"
#include "tensor_io.hpp"

#include <flagdnn/flagdnn.hpp>

#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace flagdnn::testing {
namespace {

namespace hv = validation::hygon;
namespace io = validation::hygon::tensor_io;

constexpr int kSkipReturnCode = 77;

class TemporaryCache {
public:
  explicit TemporaryCache(std::string_view family) {
    std::string pattern = (std::filesystem::temp_directory_path() /
                           ("flagdnn-hygon-" + std::string(family) + "-XXXXXX"))
                              .string();
    std::vector<char> writable(pattern.begin(), pattern.end());
    writable.push_back('\0');
    char *created = mkdtemp(writable.data());
    if (created == nullptr) {
      throw std::runtime_error("mkdtemp failed");
    }
    path_ = created;
  }

  ~TemporaryCache() {
    std::error_code ignored;
    std::filesystem::remove_all(path_, ignored);
  }

  [[nodiscard]] const std::filesystem::path &path() const noexcept {
    return path_;
  }

private:
  std::filesystem::path path_;
};

std::vector<TestTensor> inputs(const LayernormTestCase &test_case) {
  return {test_case.x, test_case.scale, test_case.bias};
}

std::vector<TestTensor> outputs(const LayernormTestCase &test_case) {
  return {test_case.y, test_case.mean, test_case.inv_variance};
}

std::vector<bool> positive_inputs(const LayernormTestCase &) {
  return {false, false, false};
}

std::vector<TestTensor> inputs(const RmsnormTestCase &test_case) {
  return {test_case.x, test_case.scale, test_case.bias};
}

std::vector<TestTensor> outputs(const RmsnormTestCase &test_case) {
  return {test_case.y, test_case.inv_variance};
}

std::vector<bool> positive_inputs(const RmsnormTestCase &) {
  return {false, false, false};
}

std::vector<TestTensor> inputs(const BatchnormTestCase &test_case) {
  return {test_case.x, test_case.scale, test_case.bias,
          test_case.previous_running_mean, test_case.previous_running_variance};
}

std::vector<TestTensor> outputs(const BatchnormTestCase &test_case) {
  return {test_case.y, test_case.mean, test_case.inv_variance,
          test_case.next_running_mean, test_case.next_running_variance};
}

std::vector<bool> positive_inputs(const BatchnormTestCase &) {
  return {false, false, false, false, true};
}

std::vector<TestTensor> inputs(const BatchnormInferenceTestCase &test_case) {
  return {test_case.x, test_case.mean, test_case.inv_variance, test_case.scale,
          test_case.bias};
}

std::vector<TestTensor> outputs(const BatchnormInferenceTestCase &test_case) {
  return {test_case.y};
}

std::vector<bool> positive_inputs(const BatchnormInferenceTestCase &) {
  return {false, false, true, false, false};
}

hv::HipdnnNormalizationOperation operation(const LayernormTestCase &) {
  return hv::make_hipdnn_normalization_unavailable(
      "hipDNN declares Other Normalization unsupported and exposes no "
      "LayerNorm primitive");
}

hv::HipdnnNormalizationOperation operation(const RmsnormTestCase &) {
  return hv::make_hipdnn_normalization_unavailable(
      "hipDNN declares Other Normalization unsupported and exposes no "
      "RMSNorm primitive");
}

hv::HipdnnNormalizationOperation operation(const BatchnormTestCase &test_case) {
  return hv::make_hipdnn_batchnorm_training_operation(test_case.epsilon,
                                                      test_case.momentum);
}

hv::HipdnnNormalizationOperation operation(const BatchnormInferenceTestCase &) {
  return hv::make_hipdnn_normalization_unavailable(
      "FlagDNN consumes inverse variance directly, while hipDNN inference "
      "consumes variance plus epsilon; the primitive equations are not "
      "input-equivalent");
}

std::string_view operation_name(const LayernormTestCase &) noexcept {
  return "layernorm";
}

std::string_view operation_name(const RmsnormTestCase &) noexcept {
  return "rmsnorm";
}

std::string_view operation_name(const BatchnormTestCase &) noexcept {
  return "batchnorm";
}

std::string_view operation_name(const BatchnormInferenceTestCase &) noexcept {
  return "batchnorm_inference";
}

template <typename Case>
TestTensor reference_tensor(const Case &, const TestTensor &tensor) {
  return tensor;
}

TestTensor reference_tensor(const BatchnormTestCase &test_case,
                            const TestTensor &tensor) {
  return tensor.uid == test_case.x.uid || tensor.uid == test_case.y.uid
             ? batchnorm_reference_data_tensor(tensor)
             : tensor;
}

TestTensor reference_tensor(const BatchnormInferenceTestCase &test_case,
                            const TestTensor &tensor) {
  return tensor.uid == test_case.x.uid || tensor.uid == test_case.y.uid
             ? batchnorm_reference_data_tensor(tensor)
             : tensor;
}

template <typename Case>
std::vector<TestTensor> all_specs(const Case &test_case, bool reference) {
  std::vector<TestTensor> result = inputs(test_case);
  std::vector<TestTensor> output_specs = outputs(test_case);
  result.insert(result.end(), output_specs.begin(), output_specs.end());
  if (reference) {
    for (TestTensor &tensor : result) {
      tensor = reference_tensor(test_case, tensor);
    }
  }
  return result;
}

std::vector<hv::ReferenceTensor>
as_reference_tensors(std::span<const TestTensor> tensors) {
  std::vector<hv::ReferenceTensor> result;
  result.reserve(tensors.size());
  for (const TestTensor &tensor : tensors) {
    result.push_back(hv::as_reference_tensor(tensor));
  }
  return result;
}

std::unique_ptr<NormalizationExecutable>
build_flagdnn(flagdnn::Handle &handle, const LayernormTestCase &test_case) {
  return build_flagdnn_layernorm(handle, test_case);
}

std::unique_ptr<NormalizationExecutable>
build_flagdnn(flagdnn::Handle &handle, const RmsnormTestCase &test_case) {
  return build_flagdnn_rmsnorm(handle, test_case);
}

std::unique_ptr<NormalizationExecutable>
build_flagdnn(flagdnn::Handle &handle, const BatchnormTestCase &test_case) {
  return build_flagdnn_batchnorm(handle, test_case);
}

std::unique_ptr<NormalizationExecutable>
build_flagdnn(flagdnn::Handle &handle,
              const BatchnormInferenceTestCase &test_case) {
  return build_flagdnn_batchnorm_inference(handle, test_case);
}

std::unique_ptr<NormalizationExecutable>
build_reference(const LayernormTestCase &test_case) {
  return build_layernorm_reference(test_case);
}

std::unique_ptr<NormalizationExecutable>
build_reference(const RmsnormTestCase &test_case) {
  return build_rmsnorm_reference(test_case);
}

std::unique_ptr<NormalizationExecutable>
build_reference(const BatchnormTestCase &test_case) {
  return build_batchnorm_reference(test_case);
}

std::unique_ptr<NormalizationExecutable>
build_reference(const BatchnormInferenceTestCase &test_case) {
  return build_batchnorm_inference_reference(test_case);
}

std::vector<float> make_input(std::size_t count, std::size_t tensor_index,
                              bool positive) {
  std::vector<float> result(count);
  for (std::size_t index = 0; index < count; ++index) {
    const int centered =
        static_cast<int>((index * 19 + tensor_index * 11) % 37) - 18;
    const float value =
        static_cast<float>(centered) / static_cast<float>(17 + tensor_index);
    result[index] = positive ? std::abs(value) + 0.25F : value;
  }
  return result;
}

std::vector<float> quantize_logical_values(const TestTensor &tensor,
                                           std::span<const float> logical) {
  const std::vector<std::uint8_t> bytes =
      io::encode(io::scatter(logical, tensor), tensor.data_type);
  return io::gather(
      io::decode(bytes, tensor.data_type, io::storage_element_count(tensor)),
      tensor);
}

std::vector<float> quantized_input(const TestTensor &tensor,
                                   std::size_t tensor_index, bool positive) {
  return quantize_logical_values(
      tensor, make_input(io::element_count(tensor), tensor_index, positive));
}

template <typename Case>
std::vector<std::vector<float>> logical_inputs(const Case &test_case) {
  const std::vector<TestTensor> specifications = inputs(test_case);
  const std::vector<bool> positive = positive_inputs(test_case);
  std::vector<std::vector<float>> result;
  result.reserve(specifications.size());
  for (std::size_t index = 0; index < specifications.size(); ++index) {
    result.push_back(
        quantized_input(specifications[index], index, positive[index]));
  }
  return result;
}

bool is_large_offset_batchnorm_case(const BatchnormTestCase &test_case) {
  return test_case.name.find("large_offset_small_variance") !=
         std::string::npos;
}

float large_offset_base(flagdnnDataType_t data_type) {
  switch (data_type) {
  case FLAGDNN_DATA_INT32:
    throw std::invalid_argument(
        "INT32 is not supported by this validation adapter");

  case FLAGDNN_DATA_FLOAT32:
    return 10000.0F;
  case FLAGDNN_DATA_FLOAT16:
    return 1024.0F;
  case FLAGDNN_DATA_BFLOAT16:
    return 128.0F;
  case FLAGDNN_DATA_FP8_E8M0:
  case FLAGDNN_DATA_FP8_E4M3:
  case FLAGDNN_DATA_FP8_E5M2:
  case FLAGDNN_DATA_BOOLEAN:
    break;
  }
  throw std::invalid_argument(
      "large-offset normalization case has an unsupported data type");
}

std::vector<std::vector<float>>
logical_inputs(const BatchnormTestCase &test_case) {
  if (!is_large_offset_batchnorm_case(test_case)) {
    return logical_inputs<BatchnormTestCase>(test_case);
  }
  if (test_case.x.dimensions.size() < 2) {
    throw std::invalid_argument(
        "large-offset BatchNorm case must use rank >= 2");
  }

  const std::vector<TestTensor> specifications = inputs(test_case);
  std::vector<std::vector<float>> result;
  result.reserve(specifications.size());

  const float base = large_offset_base(test_case.x.data_type);
  std::size_t spatial = 1;
  for (std::size_t axis = 2; axis < test_case.x.dimensions.size(); ++axis) {
    spatial *= static_cast<std::size_t>(test_case.x.dimensions[axis]);
  }
  const std::size_t channels =
      static_cast<std::size_t>(test_case.x.dimensions[1]);
  std::vector<float> x(io::element_count(test_case.x));
  for (std::size_t index = 0; index < x.size(); ++index) {
    const std::size_t batch = index / (channels * spatial);
    const std::size_t sample = batch * spatial + index % spatial;
    x[index] = base + (sample % 2 == 0 ? -1.0F : 1.0F);
  }
  result.push_back(quantize_logical_values(test_case.x, x));
  result.push_back(quantize_logical_values(
      test_case.scale,
      std::vector<float>(io::element_count(test_case.scale), 1.0F)));
  result.push_back(quantize_logical_values(
      test_case.bias,
      std::vector<float>(io::element_count(test_case.bias), 0.0F)));
  result.push_back(quantize_logical_values(
      test_case.previous_running_mean,
      std::vector<float>(io::element_count(test_case.previous_running_mean),
                         base - 4.0F)));
  result.push_back(quantize_logical_values(
      test_case.previous_running_variance,
      std::vector<float>(io::element_count(test_case.previous_running_variance),
                         2.0F)));
  return result;
}

std::vector<std::vector<float>>
large_offset_inputs(const LayernormTestCase &test_case) {
  const float base = large_offset_base(test_case.x.data_type);
  const std::size_t normalized = io::element_count(test_case.scale);
  if (normalized == 0 || io::element_count(test_case.x) % normalized != 0) {
    throw std::invalid_argument(
        "large-offset LayerNorm normalized extent is invalid");
  }
  std::vector<float> x(io::element_count(test_case.x));
  for (std::size_t index = 0; index < x.size(); ++index) {
    const std::size_t column = index % normalized;
    const float delta = normalized % 2 != 0 && column + 1 == normalized
                            ? 0.0F
                            : (column % 2 == 0 ? -1.0F : 1.0F);
    x[index] = base + delta;
  }
  return {
      quantize_logical_values(test_case.x, x),
      quantize_logical_values(
          test_case.scale,
          std::vector<float>(io::element_count(test_case.scale), 1.0F)),
      quantize_logical_values(
          test_case.bias,
          std::vector<float>(io::element_count(test_case.bias), 0.0F)),
  };
}

std::vector<std::vector<float>>
layernorm_cpu_outputs(const LayernormTestCase &test_case,
                      const std::vector<std::vector<float>> &logical) {
  if (logical.size() != 3) {
    throw std::invalid_argument("LayerNorm CPU oracle input arity is invalid");
  }
  const std::vector<float> &x = logical[0];
  const std::vector<float> &scale = logical[1];
  const std::vector<float> &bias = logical[2];
  const std::size_t normalized = scale.size();
  const std::size_t rows = x.size() / normalized;
  std::vector<float> y(x.size());
  std::vector<float> mean(rows);
  std::vector<float> inv_variance(rows);
  for (std::size_t row = 0; row < rows; ++row) {
    double sum = 0.0;
    for (std::size_t column = 0; column < normalized; ++column) {
      sum += static_cast<double>(x[row * normalized + column]);
    }
    const double row_mean = sum / static_cast<double>(normalized);
    double m2 = 0.0;
    for (std::size_t column = 0; column < normalized; ++column) {
      const double centered =
          static_cast<double>(x[row * normalized + column]) - row_mean;
      m2 += centered * centered;
    }
    const double inverse =
        1.0 /
        std::sqrt(m2 / static_cast<double>(normalized) + test_case.epsilon);
    mean[row] = static_cast<float>(row_mean);
    inv_variance[row] = static_cast<float>(inverse);
    for (std::size_t column = 0; column < normalized; ++column) {
      y[row * normalized + column] = static_cast<float>(
          (static_cast<double>(x[row * normalized + column]) - row_mean) *
              inverse * static_cast<double>(scale[column]) +
          static_cast<double>(bias[column]));
    }
  }
  return {
      quantize_logical_values(test_case.y, y),
      quantize_logical_values(test_case.mean, mean),
      quantize_logical_values(test_case.inv_variance, inv_variance),
  };
}

std::vector<std::vector<float>>
batchnorm_cpu_outputs(const BatchnormTestCase &test_case,
                      const std::vector<std::vector<float>> &logical) {
  if (logical.size() != 5 || test_case.x.dimensions.size() < 2) {
    throw std::invalid_argument("BatchNorm CPU oracle input arity is invalid");
  }
  const std::vector<float> &x = logical[0];
  const std::vector<float> &scale = logical[1];
  const std::vector<float> &bias = logical[2];
  const std::vector<float> &previous_mean = logical[3];
  const std::vector<float> &previous_variance = logical[4];
  const std::size_t batch = static_cast<std::size_t>(test_case.x.dimensions[0]);
  const std::size_t channels =
      static_cast<std::size_t>(test_case.x.dimensions[1]);
  std::size_t spatial = 1;
  for (std::size_t axis = 2; axis < test_case.x.dimensions.size(); ++axis) {
    spatial *= static_cast<std::size_t>(test_case.x.dimensions[axis]);
  }
  const std::size_t count = batch * spatial;
  if (count == 0 || scale.size() != channels || bias.size() != channels ||
      previous_mean.size() != channels ||
      previous_variance.size() != channels) {
    throw std::invalid_argument("BatchNorm CPU oracle metadata is invalid");
  }

  std::vector<float> y(x.size());
  std::vector<float> mean(channels);
  std::vector<float> inv_variance(channels);
  std::vector<float> next_mean(channels);
  std::vector<float> next_variance(channels);
  for (std::size_t channel = 0; channel < channels; ++channel) {
    double sum = 0.0;
    for (std::size_t n = 0; n < batch; ++n) {
      for (std::size_t s = 0; s < spatial; ++s) {
        sum += static_cast<double>(x[(n * channels + channel) * spatial + s]);
      }
    }
    const double channel_mean = sum / static_cast<double>(count);
    double m2 = 0.0;
    for (std::size_t n = 0; n < batch; ++n) {
      for (std::size_t s = 0; s < spatial; ++s) {
        const std::size_t index = (n * channels + channel) * spatial + s;
        const double centered = static_cast<double>(x[index]) - channel_mean;
        m2 += centered * centered;
      }
    }
    const double population_variance = m2 / static_cast<double>(count);
    const double inverse =
        1.0 / std::sqrt(population_variance + test_case.epsilon);
    const double unbiased =
        count > 1 ? m2 / static_cast<double>(count - 1) : population_variance;
    mean[channel] = static_cast<float>(channel_mean);
    inv_variance[channel] = static_cast<float>(inverse);
    next_mean[channel] =
        static_cast<float>(static_cast<double>(previous_mean[channel]) *
                               (1.0 - test_case.momentum) +
                           channel_mean * test_case.momentum);
    next_variance[channel] =
        static_cast<float>(static_cast<double>(previous_variance[channel]) *
                               (1.0 - test_case.momentum) +
                           unbiased * test_case.momentum);
    for (std::size_t n = 0; n < batch; ++n) {
      for (std::size_t s = 0; s < spatial; ++s) {
        const std::size_t index = (n * channels + channel) * spatial + s;
        y[index] = static_cast<float>(
            (static_cast<double>(x[index]) - channel_mean) * inverse *
                static_cast<double>(scale[channel]) +
            static_cast<double>(bias[channel]));
      }
    }
  }
  return {
      quantize_logical_values(test_case.y, y),
      quantize_logical_values(test_case.mean, mean),
      quantize_logical_values(test_case.inv_variance, inv_variance),
      quantize_logical_values(test_case.next_running_mean, next_mean),
      quantize_logical_values(test_case.next_running_variance, next_variance),
  };
}

enum class BindingAddress { kStorageBase, kTensorEntrance };

struct PreparedBuffers {
  std::vector<TestTensor> tensors;
  std::size_t input_count = 0;
  std::vector<std::unique_ptr<hv::DeviceBuffer>> buffers;
  std::vector<flagdnnBinding_t> bindings;
};

PreparedBuffers prepare_buffers(std::vector<TestTensor> tensors,
                                std::size_t input_count,
                                const std::vector<std::vector<float>> &logical,
                                BindingAddress binding_address,
                                hv::Stream &stream) {
  if (input_count > tensors.size() || logical.size() != input_count) {
    throw std::invalid_argument("normalization buffer arity is invalid");
  }
  PreparedBuffers result;
  result.tensors = std::move(tensors);
  result.input_count = input_count;
  result.buffers.reserve(result.tensors.size());
  result.bindings.reserve(result.tensors.size());
  for (std::size_t index = 0; index < result.tensors.size(); ++index) {
    const TestTensor &tensor = result.tensors[index];
    std::vector<float> physical;
    if (index < input_count) {
      physical = io::scatter(logical[index], tensor);
    } else {
      physical.assign(io::storage_element_count(tensor), io::kPaddingSentinel);
    }
    const std::vector<std::uint8_t> bytes =
        io::encode(physical, tensor.data_type);
    auto buffer = std::make_unique<hv::DeviceBuffer>(
        tensor.binding_byte_offset + bytes.size());
    buffer->copy_from_host_at(bytes.data(), bytes.size(),
                              tensor.binding_byte_offset, stream.get());
    void *pointer = binding_address == BindingAddress::kStorageBase
                        ? buffer->opaque()
                        : buffer->opaque_at(tensor.binding_byte_offset);
    result.bindings.push_back({tensor.uid, pointer});
    result.buffers.push_back(std::move(buffer));
  }
  return result;
}

std::vector<float> read_output(const PreparedBuffers &buffers,
                               std::size_t output_index, hv::Stream &stream,
                               std::string_view provider) {
  const std::size_t tensor_index = buffers.input_count + output_index;
  const TestTensor &output = buffers.tensors.at(tensor_index);
  const std::size_t storage = io::storage_element_count(output);
  std::vector<std::uint8_t> bytes(storage *
                                  io::data_type_size(output.data_type));
  buffers.buffers.at(tensor_index)
      ->copy_to_host_at(bytes.data(), bytes.size(), output.binding_byte_offset,
                        stream.get());
  stream.synchronize();
  const std::vector<float> physical =
      io::decode(bytes, output.data_type, storage);
  io::require_padding_unchanged(provider, physical, output);
  return io::gather(physical, output);
}

void execute(NormalizationExecutable &executable,
             std::span<const flagdnnBinding_t> bindings,
             hv::DeviceBuffer &workspace, hv::Stream &stream) {
  executable.execute(bindings, workspace.opaque(), executable.workspace_size(),
                     stream.opaque());
}

struct Accuracy {
  double maximum_absolute = 0.0;
  double maximum_relative = 0.0;
};

template <typename Case>
Accuracy compare(std::span<const float> actual,
                 std::span<const float> reference, const Case &test_case,
                 std::size_t output_index) {
  if (actual.size() != reference.size()) {
    throw std::runtime_error("normalization output sizes differ");
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
    if (!std::isfinite(absolute) || (absolute > test_case.absolute_tolerance &&
                                     relative > test_case.relative_tolerance)) {
      std::ostringstream message;
      message << test_case.name << " output " << output_index
              << " differs at element " << index << ": FlagDNN=" << left
              << ", hipDNN=" << right << ", abs=" << absolute
              << ", rel=" << relative
              << ", atol=" << test_case.absolute_tolerance
              << ", rtol=" << test_case.relative_tolerance;
      throw std::runtime_error(message.str());
    }
  }
  return result;
}

template <typename Case>
Accuracy compare_cpu_oracle(std::span<const float> actual,
                            std::span<const float> reference,
                            const Case &test_case, std::size_t output_index) {
  if (actual.size() != reference.size()) {
    throw std::runtime_error("normalization stability output sizes differ");
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
    if (!std::isfinite(absolute) || (absolute > test_case.absolute_tolerance &&
                                     relative > test_case.relative_tolerance)) {
      std::ostringstream message;
      message << test_case.name << " output " << output_index
              << " differs at element " << index << ": FlagDNN=" << left
              << ", stable CPU oracle=" << right << ", abs=" << absolute
              << ", rel=" << relative
              << ", atol=" << test_case.absolute_tolerance
              << ", rtol=" << test_case.relative_tolerance;
      throw std::runtime_error(message.str());
    }
  }
  return result;
}

std::vector<std::vector<float>>
stability_inputs(const LayernormTestCase &test_case) {
  return large_offset_inputs(test_case);
}

std::vector<std::vector<float>>
stability_inputs(const BatchnormTestCase &test_case) {
  return logical_inputs(test_case);
}

std::vector<std::vector<float>>
stability_outputs(const LayernormTestCase &test_case,
                  const std::vector<std::vector<float>> &logical) {
  return layernorm_cpu_outputs(test_case, logical);
}

std::vector<std::vector<float>>
stability_outputs(const BatchnormTestCase &test_case,
                  const std::vector<std::vector<float>> &logical) {
  return batchnorm_cpu_outputs(test_case, logical);
}

template <typename Case>
void run_cpu_oracle_stability_case(flagdnn::Handle &handle,
                                   const Case &test_case, hv::Stream &stream) {
  const std::vector<std::vector<float>> logical = stability_inputs(test_case);
  const std::vector<std::vector<float>> expected =
      stability_outputs(test_case, logical);
  const std::vector<TestTensor> specifications = all_specs(test_case, false);
  const std::size_t input_count = inputs(test_case).size();
  if (expected.size() != outputs(test_case).size()) {
    throw std::runtime_error(
        "normalization stability oracle output arity is invalid");
  }

  std::unique_ptr<NormalizationExecutable> executable =
      build_flagdnn(handle, test_case);
  PreparedBuffers buffers =
      prepare_buffers(specifications, input_count, logical,
                      BindingAddress::kTensorEntrance, stream);
  hv::DeviceBuffer workspace(executable->workspace_size());
  stream.synchronize();
  execute(*executable, buffers.bindings, workspace, stream);
  stream.synchronize();

  Accuracy aggregate;
  for (std::size_t index = 0; index < expected.size(); ++index) {
    const Accuracy accuracy =
        compare_cpu_oracle(read_output(buffers, index, stream, "FlagDNN"),
                           expected[index], test_case, index);
    aggregate.maximum_absolute =
        std::max(aggregate.maximum_absolute, accuracy.maximum_absolute);
    aggregate.maximum_relative =
        std::max(aggregate.maximum_relative, accuracy.maximum_relative);
  }
  std::cout << test_case.name
            << ": FlagDNN Graph vs stable CPU oracle PASS max_abs="
            << aggregate.maximum_absolute
            << " max_rel=" << aggregate.maximum_relative << std::endl;
}

double stability_tolerance(flagdnnDataType_t data_type) {
  switch (data_type) {
  case FLAGDNN_DATA_INT32:
    throw std::invalid_argument(
        "INT32 is not supported by this validation adapter");

  case FLAGDNN_DATA_FLOAT32:
    return 5.0e-3;
  case FLAGDNN_DATA_FLOAT16:
    return 3.0e-2;
  case FLAGDNN_DATA_BFLOAT16:
    return 8.0e-2;
  case FLAGDNN_DATA_FP8_E8M0:
  case FLAGDNN_DATA_FP8_E4M3:
  case FLAGDNN_DATA_FP8_E5M2:
  case FLAGDNN_DATA_BOOLEAN:
    break;
  }
  throw std::invalid_argument(
      "normalization stability tolerance has an unsupported data type");
}

std::vector<LayernormTestCase> layernorm_stability_cases() {
  std::vector<LayernormTestCase> result;
  for (const LayernormTestCase &base : make_layernorm_cases()) {
    if (base.x.dimensions != std::vector<std::int64_t>{2, 5, 17}) {
      continue;
    }
    LayernormTestCase test_case = base;
    test_case.name += "_large_offset_small_variance_hygon";
    test_case.absolute_tolerance = stability_tolerance(base.x.data_type);
    test_case.relative_tolerance = test_case.absolute_tolerance;
    test_case.autotune = false;
    result.push_back(std::move(test_case));
  }
  if (result.size() != 3) {
    throw std::runtime_error(
        "expected FP32/FP16/BF16 LayerNorm stability cases");
  }
  LayernormTestCase multi_tile = result.front();
  multi_tile.name = "layernorm_fp32_1x65537_large_offset_small_variance_hygon";
  multi_tile.x.dimensions = {1, 65537};
  multi_tile.x.strides = {65537, 1};
  multi_tile.y.dimensions = multi_tile.x.dimensions;
  multi_tile.y.strides = multi_tile.x.strides;
  multi_tile.scale.dimensions = {1, 65537};
  multi_tile.scale.strides = {65537, 1};
  multi_tile.bias.dimensions = multi_tile.scale.dimensions;
  multi_tile.bias.strides = multi_tile.scale.strides;
  multi_tile.mean.dimensions = {1, 1};
  multi_tile.mean.strides = {1, 1};
  multi_tile.inv_variance.dimensions = multi_tile.mean.dimensions;
  multi_tile.inv_variance.strides = multi_tile.mean.strides;
  result.push_back(std::move(multi_tile));
  return result;
}

std::vector<BatchnormTestCase> batchnorm_stability_cases() {
  std::vector<BatchnormTestCase> result;
  for (const BatchnormTestCase &base : make_batchnorm_cases()) {
    // This stability gate uses one fixed channel geometry and two layouts;
    // the per-operator suite covers the full, independently growing catalog.
    if (base.x.dimensions != std::vector<std::int64_t>{2, 8, 8, 8}) {
      continue;
    }
    const bool contiguous = base.name.find("_contiguous") != std::string::npos;
    const bool fp32_channels_last =
        base.x.data_type == FLAGDNN_DATA_FLOAT32 &&
        base.name.find("_channels_last") != std::string::npos;
    if (!contiguous && !fp32_channels_last) {
      continue;
    }
    BatchnormTestCase test_case = base;
    test_case.name += "_large_offset_small_variance_hygon";
    test_case.absolute_tolerance = stability_tolerance(base.x.data_type);
    test_case.relative_tolerance = test_case.absolute_tolerance;
    test_case.autotune = false;
    result.push_back(std::move(test_case));

    if (base.x.data_type == FLAGDNN_DATA_FLOAT32) {
      BatchnormTestCase multi_tile = base;
      multi_tile.name += "_large_offset_small_variance_multitile_hygon";
      multi_tile.x.dimensions = {2, 8, 16, 16};
      multi_tile.y.dimensions = multi_tile.x.dimensions;
      if (contiguous) {
        multi_tile.x.strides = {2048, 256, 16, 1};
      } else {
        multi_tile.x.strides = {2048, 1, 128, 8};
      }
      multi_tile.y.strides = multi_tile.x.strides;
      multi_tile.absolute_tolerance = stability_tolerance(base.x.data_type);
      multi_tile.relative_tolerance = multi_tile.absolute_tolerance;
      multi_tile.autotune = false;
      result.push_back(std::move(multi_tile));
    }
  }
  if (result.size() != 6) {
    throw std::runtime_error(
        "expected six BatchNorm stability cases including multi-tile paths");
  }
  return result;
}

std::vector<std::vector<float>>
cpu_outputs(const LayernormTestCase &c,
            const std::vector<std::vector<float>> &v) {
  return layernorm_cpu_outputs(c, v);
}
std::vector<std::vector<float>>
cpu_outputs(const BatchnormTestCase &c,
            const std::vector<std::vector<float>> &v) {
  return batchnorm_cpu_outputs(c, v);
}
std::vector<std::vector<float>>
cpu_outputs(const RmsnormTestCase &c,
            const std::vector<std::vector<float>> &v) {
  const auto width = v[1].size(), rows = v[0].size() / width;
  std::vector<float> y(v[0].size()), inverse(rows);
  for (std::size_t r = 0; r < rows; ++r) {
    double square = 0;
    for (std::size_t j = 0; j < width; ++j) {
      const double x = v[0][r * width + j];
      square += x * x;
    }
    const double inv = 1 / std::sqrt(square / width + c.epsilon);
    inverse[r] = static_cast<float>(inv);
    for (std::size_t j = 0; j < width; ++j)
      y[r * width + j] =
          static_cast<float>(v[0][r * width + j] * inv * v[1][j] + v[2][j]);
  }
  return {y, inverse};
}
std::vector<std::vector<float>>
cpu_outputs(const BatchnormInferenceTestCase &c,
            const std::vector<std::vector<float>> &v) {
  const auto channels = c.x.dimensions[1];
  const auto spatial = v[0].size() / c.x.dimensions[0] / channels;
  std::vector<float> y(v[0].size());
  for (std::size_t i = 0; i < y.size(); ++i) {
    const auto channel = (i / spatial) % channels;
    y[i] = static_cast<float>((double(v[0][i]) - v[1][channel]) *
                                  v[2][channel] * v[3][channel] +
                              v[4][channel]);
  }
  return {y};
}

enum class CaseResult { kExecuted, kSkipped };

template <typename Case>
CaseResult run_case(const Case &test_case,
                    const std::function<flagdnn::Handle &()> &get_handle,
                    hv::Stream &stream) {
  const hv::HipdnnNormalizationOperation reference_operation =
      operation(test_case);
  const std::vector<TestTensor> reference_specs = all_specs(test_case, true);
  const std::vector<hv::ReferenceTensor> reference_tensors =
      as_reference_tensors(reference_specs);
  const hv::HipdnnCapability capability = hv::hipdnn_normalization_capability(
      reference_operation, reference_tensors);
  hv::require_valid_hipdnn_adapter_contract(capability,
                                            operation_name(test_case));
  if (!capability.supported) {
    hygon_functional::host::run_case(
        test_case.name, inputs(test_case), outputs(test_case),
        logical_inputs(test_case), get_handle, stream,
        [&](flagdnn::Handle &h) { return build_flagdnn(h, test_case); },
        [&](const auto &v) { return cpu_outputs(test_case, v); },
        test_case.absolute_tolerance, test_case.relative_tolerance);
    return CaseResult::kExecuted;
  }

  const std::vector<TestTensor> flagdnn_specs = all_specs(test_case, false);
  const std::size_t input_count = inputs(test_case).size();
  const std::vector<std::vector<float>> logical = logical_inputs(test_case);

  std::unique_ptr<NormalizationExecutable> reference =
      build_reference(test_case);
  PreparedBuffers reference_buffers =
      prepare_buffers(reference_specs, input_count, logical,
                      BindingAddress::kStorageBase, stream);
  hv::DeviceBuffer reference_workspace(reference->workspace_size());
  stream.synchronize();
  execute(*reference, reference_buffers.bindings, reference_workspace, stream);
  stream.synchronize();

  std::unique_ptr<NormalizationExecutable> flagdnn =
      build_flagdnn(get_handle(), test_case);
  PreparedBuffers flagdnn_buffers =
      prepare_buffers(flagdnn_specs, input_count, logical,
                      BindingAddress::kTensorEntrance, stream);
  hv::DeviceBuffer flagdnn_workspace(flagdnn->workspace_size());
  stream.synchronize();
  execute(*flagdnn, flagdnn_buffers.bindings, flagdnn_workspace, stream);
  stream.synchronize();

  Accuracy aggregate;
  const std::size_t output_count = outputs(test_case).size();
  for (std::size_t index = 0; index < output_count; ++index) {
    const Accuracy accuracy =
        compare(read_output(flagdnn_buffers, index, stream, "FlagDNN"),
                read_output(reference_buffers, index, stream, "hipDNN"),
                test_case, index);
    aggregate.maximum_absolute =
        std::max(aggregate.maximum_absolute, accuracy.maximum_absolute);
    aggregate.maximum_relative =
        std::max(aggregate.maximum_relative, accuracy.maximum_relative);
  }
  std::cout << test_case.name
            << ": FlagDNN Graph vs hipDNN primitive PASS max_abs="
            << aggregate.maximum_absolute
            << " max_rel=" << aggregate.maximum_relative << std::endl;
  return CaseResult::kExecuted;
}

template <typename Case>
int run_suite(int argc, char **argv, std::span<const Case> cases,
              std::string_view family, std::string_view suite_name) {
  if (argc != 3) {
    std::cerr << "usage: " << argv[0] << " COMPILER_EXECUTABLE COMPILER_ENTRY"
              << std::endl;
    return 2;
  }
  try {
    std::cout << std::setprecision(9);
    hv::DeviceGuard device;
    hv::Stream stream;
    TemporaryCache cache(family);
    std::unique_ptr<flagdnn::Handle> handle;
    const std::function<flagdnn::Handle &()> get_handle =
        [&]() -> flagdnn::Handle & {
      if (handle == nullptr) {
        handle = std::make_unique<flagdnn::Handle>("hygon", 0);
        handle->set_compiler(argv[1], argv[2], cache.path().string());
      }
      return *handle;
    };
    const char *filter = std::getenv("FLAGDNN_NORMALIZATION_CASE");
    std::size_t matched = 0;
    std::size_t executed = 0;
    std::size_t skipped = 0;
    for (const Case &test_case : cases) {
      if (filter != nullptr &&
          test_case.name.find(filter) == std::string::npos) {
        continue;
      }
      ++matched;
      validate_normalization_case(test_case);
      const CaseResult result = run_case(test_case, get_handle, stream);
      result == CaseResult::kExecuted ? ++executed : ++skipped;
    }
    if (matched == 0) {
      throw std::runtime_error("normalization case filter matched no cases");
    }
    std::cout << suite_name << ": " << (executed == 0 ? "SKIP" : "PASS")
              << " cases=" << matched << " executed=" << executed
              << " skipped=" << skipped << std::endl;
    return executed == 0 ? kSkipReturnCode : 0;
  } catch (const std::exception &error) {
    std::cerr << suite_name << "_FAILED: " << error.what() << std::endl;
    return 1;
  }
}

} // namespace

int run_layernorm_functional_test(int argc, char **argv,
                                  std::span<const LayernormTestCase> cases) {
  return run_suite(argc, argv, cases, "layernorm",
                   "FLAGDNN_LAYERNORM_FUNCTIONAL");
}

int run_rmsnorm_functional_test(int argc, char **argv,
                                std::span<const RmsnormTestCase> cases) {
  return run_suite(argc, argv, cases, "rmsnorm", "FLAGDNN_RMSNORM_FUNCTIONAL");
}

int run_batchnorm_functional_test(int argc, char **argv,
                                  std::span<const BatchnormTestCase> cases) {
  return run_suite(argc, argv, cases, "batchnorm",
                   "FLAGDNN_BATCHNORM_FUNCTIONAL");
}

int run_batchnorm_inference_functional_test(
    int argc, char **argv, std::span<const BatchnormInferenceTestCase> cases) {
  return run_suite(argc, argv, cases, "batchnorm-inference",
                   "FLAGDNN_BATCHNORM_INFERENCE_FUNCTIONAL");
}

int run_hygon_normalization_stability_test(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "usage: " << argv[0] << " COMPILER_EXECUTABLE COMPILER_ENTRY"
              << std::endl;
    return 2;
  }
  try {
    std::cout << std::setprecision(9);
    hv::DeviceGuard device;
    hv::Stream stream;
    TemporaryCache cache("normalization-stability");
    flagdnn::Handle handle("hygon", 0);
    handle.set_compiler(argv[1], argv[2], cache.path().string());

    std::size_t executed = 0;
    for (const LayernormTestCase &test_case : layernorm_stability_cases()) {
      validate_normalization_case(test_case);
      run_cpu_oracle_stability_case(handle, test_case, stream);
      ++executed;
    }
    for (const BatchnormTestCase &test_case : batchnorm_stability_cases()) {
      validate_normalization_case(test_case);
      run_cpu_oracle_stability_case(handle, test_case, stream);
      ++executed;
    }
    std::cout << "FLAGDNN_HYGON_NORMALIZATION_STABILITY: PASS cases="
              << executed << std::endl;
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "FLAGDNN_HYGON_NORMALIZATION_STABILITY_FAILED: "
              << error.what() << std::endl;
    return 1;
  }
}

} // namespace flagdnn::testing
