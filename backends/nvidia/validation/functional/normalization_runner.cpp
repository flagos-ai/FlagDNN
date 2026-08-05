/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/normalization.hpp"
#include "validation/tensor_io.hpp"
#include "validation/cuda_driver.hpp"

#include <flagdnn/flagdnn.hpp>

#include <unistd.h>

#include <algorithm>
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
#include <utility>
#include <vector>

namespace flagdnn::testing {
namespace {


class TemporaryCache {
 public:
  TemporaryCache() {
    std::string pattern =
        (std::filesystem::temp_directory_path() /
         "flagdnn-normalization-functional-XXXXXX")
            .string();
    std::vector<char> writable(pattern.begin(), pattern.end());
    writable.push_back('\0');
    char* created = mkdtemp(writable.data());
    if (created == nullptr) {
      throw std::runtime_error("mkdtemp failed");
    }
    path_ = created;
  }

  ~TemporaryCache() {
    std::error_code ignored;
    std::filesystem::remove_all(path_, ignored);
  }

  [[nodiscard]] const std::filesystem::path& path() const noexcept {
    return path_;
  }

 private:
  std::filesystem::path path_;
};


std::vector<float> make_input(std::size_t count,
                              std::size_t tensor_index,
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

std::vector<float> quantized_input(const TestTensor& tensor,
                                   std::size_t tensor_index,
                                   bool positive) {
  const std::vector<std::uint8_t> encoded = cuda::encode(
      cuda::scatter(
          make_input(cuda::element_count(tensor), tensor_index, positive),
          tensor),
      tensor.data_type,
      cuda::BooleanEncoding::kByte);
  return cuda::gather(
      cuda::decode(encoded,
                   tensor.data_type,
                   cuda::storage_element_count(tensor),
                   cuda::BooleanEncoding::kByte),
      tensor);
}

std::vector<std::uint8_t> encode_logical(
    std::span<const float> logical,
    const TestTensor& tensor) {
  return cuda::encode(cuda::scatter(logical, tensor),
                      tensor.data_type,
                      cuda::BooleanEncoding::kByte);
}

std::vector<std::uint8_t> output_bytes(const TestTensor& tensor) {
  return cuda::encode(
      std::vector<float>(
          cuda::storage_element_count(tensor), cuda::padding_sentinel()),
      tensor.data_type,
      cuda::BooleanEncoding::kByte);
}


std::vector<TestTensor> inputs(const LayernormTestCase& test_case) {
  return {test_case.x, test_case.scale, test_case.bias};
}

std::vector<TestTensor> outputs(const LayernormTestCase& test_case) {
  return {test_case.y, test_case.mean, test_case.inv_variance};
}

std::vector<bool> positive_inputs(const LayernormTestCase&) {
  return {false, false, false};
}

std::vector<TestTensor> inputs(const RmsnormTestCase& test_case) {
  return {test_case.x, test_case.scale, test_case.bias};
}

std::vector<TestTensor> outputs(const RmsnormTestCase& test_case) {
  return {test_case.y, test_case.inv_variance};
}

std::vector<bool> positive_inputs(const RmsnormTestCase&) {
  return {false, false, false};
}

std::vector<TestTensor> inputs(const BatchnormTestCase& test_case) {
  return {test_case.x,
          test_case.scale,
          test_case.bias,
          test_case.previous_running_mean,
          test_case.previous_running_variance};
}

std::vector<TestTensor> outputs(const BatchnormTestCase& test_case) {
  return {test_case.y,
          test_case.mean,
          test_case.inv_variance,
          test_case.next_running_mean,
          test_case.next_running_variance};
}

std::vector<bool> positive_inputs(const BatchnormTestCase&) {
  return {false, false, false, false, true};
}

std::vector<TestTensor> inputs(
    const BatchnormInferenceTestCase& test_case) {
  return {test_case.x,
          test_case.mean,
          test_case.inv_variance,
          test_case.scale,
          test_case.bias};
}

std::vector<TestTensor> outputs(
    const BatchnormInferenceTestCase& test_case) {
  return {test_case.y};
}

std::vector<bool> positive_inputs(const BatchnormInferenceTestCase&) {
  return {false, false, true, false, false};
}

template <typename Case>
TestTensor reference_tensor(const Case&, const TestTensor& tensor) {
  return tensor;
}

TestTensor reference_tensor(const BatchnormTestCase& test_case,
                            const TestTensor& tensor) {
  return tensor.uid == test_case.x.uid || tensor.uid == test_case.y.uid
             ? batchnorm_reference_data_tensor(tensor)
             : tensor;
}

TestTensor reference_tensor(const BatchnormInferenceTestCase& test_case,
                            const TestTensor& tensor) {
  return tensor.uid == test_case.x.uid || tensor.uid == test_case.y.uid
             ? batchnorm_reference_data_tensor(tensor)
             : tensor;
}

std::unique_ptr<NormalizationExecutable> build_flagdnn(
    flagdnn::Handle& handle,
    const LayernormTestCase& test_case) {
  return build_flagdnn_layernorm(handle, test_case);
}

std::unique_ptr<NormalizationExecutable> build_flagdnn(
    flagdnn::Handle& handle,
    const RmsnormTestCase& test_case) {
  return build_flagdnn_rmsnorm(handle, test_case);
}

std::unique_ptr<NormalizationExecutable> build_flagdnn(
    flagdnn::Handle& handle,
    const BatchnormTestCase& test_case) {
  return build_flagdnn_batchnorm(handle, test_case);
}

std::unique_ptr<NormalizationExecutable> build_flagdnn(
    flagdnn::Handle& handle,
    const BatchnormInferenceTestCase& test_case) {
  return build_flagdnn_batchnorm_inference(handle, test_case);
}

std::unique_ptr<NormalizationExecutable> build_reference(
    const LayernormTestCase& test_case) {
  return build_layernorm_reference(test_case);
}

std::unique_ptr<NormalizationExecutable> build_reference(
    const RmsnormTestCase& test_case) {
  return build_rmsnorm_reference(test_case);
}

std::unique_ptr<NormalizationExecutable> build_reference(
    const BatchnormTestCase& test_case) {
  return build_batchnorm_reference(test_case);
}

std::unique_ptr<NormalizationExecutable> build_reference(
    const BatchnormInferenceTestCase& test_case) {
  return build_batchnorm_inference_reference(test_case);
}


struct PreparedBuffers {
  std::vector<TestTensor> inputs;
  std::vector<TestTensor> outputs;
  std::vector<std::unique_ptr<DeviceBuffer>> buffers;
  std::vector<flagdnnBinding_t> bindings;
};

PreparedBuffers prepare_buffers(
    std::vector<TestTensor> input_specs,
    std::vector<TestTensor> output_specs,
    const std::vector<std::vector<float>>& logical_inputs,
    Stream& stream) {
  if (input_specs.size() != logical_inputs.size()) {
    throw std::invalid_argument("normalization logical input count is invalid");
  }
  PreparedBuffers result;
  result.inputs = std::move(input_specs);
  result.outputs = std::move(output_specs);
  result.buffers.reserve(result.inputs.size() + result.outputs.size());
  result.bindings.reserve(result.inputs.size() + result.outputs.size());
  for (std::size_t index = 0; index < result.inputs.size(); ++index) {
    const std::vector<std::uint8_t> encoded =
        encode_logical(logical_inputs[index], result.inputs[index]);
    auto buffer = std::make_unique<DeviceBuffer>(encoded.size());
    buffer->copy_from_host(encoded.data(), encoded.size(), stream.get());
    result.bindings.push_back(
        {result.inputs[index].uid, buffer->opaque()});
    result.buffers.push_back(std::move(buffer));
  }
  for (const TestTensor& output : result.outputs) {
    const std::vector<std::uint8_t> encoded = output_bytes(output);
    auto buffer = std::make_unique<DeviceBuffer>(encoded.size());
    buffer->copy_from_host(encoded.data(), encoded.size(), stream.get());
    result.bindings.push_back({output.uid, buffer->opaque()});
    result.buffers.push_back(std::move(buffer));
  }
  return result;
}

std::vector<float> read_output(const PreparedBuffers& prepared,
                               std::size_t output_index,
                               Stream& stream,
                               std::string_view provider) {
  const TestTensor& output = prepared.outputs.at(output_index);
  const DeviceBuffer& buffer =
      *prepared.buffers.at(prepared.inputs.size() + output_index);
  std::vector<std::uint8_t> bytes(
      cuda::encoded_byte_count(output, cuda::BooleanEncoding::kByte));
  buffer.copy_to_host(bytes.data(), bytes.size(), stream.get());
  stream.synchronize();
  const std::vector<float> physical = cuda::decode(
      bytes,
      output.data_type,
      cuda::storage_element_count(output),
      cuda::BooleanEncoding::kByte);
  cuda::require_padding_unchanged(provider, physical, output);
  return cuda::gather(physical, output);
}

void execute(NormalizationExecutable& executable,
             std::span<const flagdnnBinding_t> bindings,
             DeviceBuffer& workspace,
             Stream& stream) {
  executable.execute(bindings,
                     workspace.opaque(),
                     executable.workspace_size(),
                     stream.opaque());
}

struct Accuracy {
  double maximum_absolute = 0.0;
  double maximum_relative = 0.0;
};

template <typename Case>
Accuracy compare(std::span<const float> actual,
                 std::span<const float> reference,
                 const Case& test_case,
                 std::size_t output_index,
                 std::string_view reference_name) {
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
    if (!std::isfinite(absolute) ||
        (absolute > test_case.absolute_tolerance &&
         relative > test_case.relative_tolerance)) {
      std::ostringstream message;
      message << test_case.name << " output " << output_index
              << " differs at element " << index << ": FlagDNN=" << left
              << ", " << reference_name << '=' << right
              << ", abs=" << absolute << ", rel=" << relative
              << ", atol=" << test_case.absolute_tolerance
              << ", rtol=" << test_case.relative_tolerance;
      throw std::runtime_error(message.str());
    }
  }
  return result;
}

template <typename Case>
std::vector<std::vector<float>> logical_inputs(const Case& test_case) {
  const std::vector<TestTensor> specifications = inputs(test_case);
  const std::vector<bool> positives = positive_inputs(test_case);
  std::vector<std::vector<float>> result;
  result.reserve(specifications.size());
  for (std::size_t index = 0; index < specifications.size(); ++index) {
    result.push_back(
        quantized_input(specifications[index], index, positives[index]));
  }
  return result;
}

template <typename Case>
void run_graph_case(const Case& test_case,
                    flagdnn::Handle& handle,
                    Stream& stream) {
  auto flagdnn = build_flagdnn(handle, test_case);
  auto reference = build_reference(test_case);
  const auto logical = logical_inputs(test_case);
  const std::vector<TestTensor> flagdnn_inputs = inputs(test_case);
  const std::vector<TestTensor> flagdnn_outputs = outputs(test_case);
  std::vector<TestTensor> reference_inputs;
  std::vector<TestTensor> reference_outputs;
  for (const TestTensor& tensor : flagdnn_inputs) {
    reference_inputs.push_back(reference_tensor(test_case, tensor));
  }
  for (const TestTensor& tensor : flagdnn_outputs) {
    reference_outputs.push_back(reference_tensor(test_case, tensor));
  }
  PreparedBuffers flagdnn_buffers = prepare_buffers(
      flagdnn_inputs, flagdnn_outputs, logical, stream);
  PreparedBuffers reference_buffers = prepare_buffers(
      std::move(reference_inputs), std::move(reference_outputs), logical, stream);
  DeviceBuffer flagdnn_workspace(flagdnn->workspace_size());
  DeviceBuffer reference_workspace(reference->workspace_size());
  stream.synchronize();
  execute(*flagdnn, flagdnn_buffers.bindings, flagdnn_workspace, stream);
  execute(*reference,
          reference_buffers.bindings,
          reference_workspace,
          stream);
  stream.synchronize();
  Accuracy aggregate;
  for (std::size_t index = 0; index < flagdnn_outputs.size(); ++index) {
    const Accuracy accuracy = compare(
        read_output(flagdnn_buffers, index, stream, "FlagDNN"),
        read_output(reference_buffers, index, stream, "cuDNN"),
        test_case,
        index,
        "cuDNN");
    aggregate.maximum_absolute =
        std::max(aggregate.maximum_absolute, accuracy.maximum_absolute);
    aggregate.maximum_relative =
        std::max(aggregate.maximum_relative, accuracy.maximum_relative);
  }
  std::cout << test_case.name << ": FlagDNN Graph vs cuDNN Graph PASS"
            << " max_abs=" << aggregate.maximum_absolute
            << " max_rel=" << aggregate.maximum_relative << std::endl;
}



template <typename Case>
int run_suite(int argc,
              char** argv,
              std::span<const Case> cases,
              std::string_view suite_name) {
  if (argc != 3) {
    std::cerr << "usage: " << argv[0] << " COMPILER_EXECUTABLE COMPILER_ENTRY" << std::endl;
    return 2;
  }
  try {
    std::cout << std::setprecision(9);
    DriverContext driver;
    Stream stream;
    TemporaryCache cache;
    flagdnn::Handle handle(FLAGDNN_BACKEND_NVIDIA, 0);
    handle.set_compiler(argv[1], argv[2], cache.path().string());
    const char* filter = std::getenv("FLAGDNN_NORMALIZATION_CASE");
    std::size_t executed = 0;
    for (const Case& test_case : cases) {
      if (filter != nullptr &&
          test_case.name.find(filter) == std::string::npos) {
        continue;
      }
      run_graph_case(test_case, handle, stream);
      ++executed;
    }
    if (executed == 0) {
      throw std::runtime_error(
          "normalization filter matched no test cases");
    }
    std::cout << suite_name << ": PASS cases=" << executed << std::endl;
    return 0;
  } catch (const std::exception& error) {
    std::cerr << suite_name << "_FAILED: " << error.what() << std::endl;
    return 1;
  }
}

}  // namespace

int run_layernorm_functional_test(
    int argc,
    char** argv,
    std::span<const LayernormTestCase> cases) {
  return run_suite(argc, argv, cases, "FLAGDNN_LAYERNORM_FUNCTIONAL");
}

int run_rmsnorm_functional_test(
    int argc,
    char** argv,
    std::span<const RmsnormTestCase> cases) {
  return run_suite(argc, argv, cases, "FLAGDNN_RMSNORM_FUNCTIONAL");
}

int run_batchnorm_functional_test(
    int argc,
    char** argv,
    std::span<const BatchnormTestCase> cases) {
  return run_suite(argc, argv, cases, "FLAGDNN_BATCHNORM_FUNCTIONAL");
}

int run_batchnorm_inference_functional_test(
    int argc,
    char** argv,
    std::span<const BatchnormInferenceTestCase> cases) {
  return run_suite(
      argc, argv, cases, "FLAGDNN_BATCHNORM_INFERENCE_FUNCTIONAL");
}

}  // namespace flagdnn::testing
