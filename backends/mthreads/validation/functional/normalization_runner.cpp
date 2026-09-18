/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include <musa_runtime_api.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <flagdnn/flagdnn.hpp>
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

#include "backends/mthreads/validation/functional/tensor_io_adapter.hpp"
#include "backends/mthreads/validation/musa_driver.hpp"
#include "backends/mthreads/validation/paired_timing.hpp"
#include "common/normalization.hpp"

namespace flagdnn::testing {
namespace {

namespace mv = validation::mthreads;
namespace io = validation::mthreads::tensor_io;

class TemporaryCache final {
 public:
  TemporaryCache() {
    std::string pattern =
        (std::filesystem::temp_directory_path() /
         "flagdnn-mthreads-normalization-functional-XXXXXX")
            .string();
    std::vector<char> writable(pattern.begin(), pattern.end());
    writable.push_back('\0');
    char* created = mkdtemp(writable.data());
    if (created == nullptr) {
      throw std::runtime_error(
          "mkdtemp failed for mthreads normalization cache");
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

std::vector<float> make_input(const TestTensor& tensor,
                              std::size_t input_index,
                              bool positive) {
  std::vector<float> result = io::make_input(tensor, input_index);
  if (positive) {
    for (float& value : result) {
      value = std::abs(value) + 0.25F;
    }
  }
  const std::vector<std::uint8_t> encoded =
      io::encode(io::scatter(result, tensor), tensor.data_type);
  return io::gather(io::decode(encoded, tensor.data_type), tensor);
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
  if (tensor.uid == test_case.x.uid || tensor.uid == test_case.y.uid) {
    return batchnorm_reference_data_tensor(tensor);
  }
  TestTensor result = tensor;
  if (tensor.uid == test_case.scale.uid || tensor.uid == test_case.bias.uid) {
    result.data_type = FLAGDNN_DATA_FLOAT32;
  }
  return result;
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
  std::vector<std::vector<std::uint8_t>> initial_bytes;
  std::vector<std::unique_ptr<mv::DeviceBuffer>> buffers;
  std::vector<flagdnnBinding_t> bindings;
};

PreparedBuffers prepare_buffers(
    std::vector<TestTensor> input_specs,
    std::vector<TestTensor> output_specs,
    const std::vector<std::vector<float>>& logical_inputs,
    mv::Stream& stream) {
  if (input_specs.size() != logical_inputs.size()) {
    throw std::invalid_argument(
        "normalization logical input count is invalid");
  }
  PreparedBuffers result;
  result.inputs = std::move(input_specs);
  result.outputs = std::move(output_specs);
  const std::size_t count = result.inputs.size() + result.outputs.size();
  result.initial_bytes.reserve(count);
  result.buffers.reserve(count);
  result.bindings.reserve(count);
  for (std::size_t index = 0; index < count; ++index) {
    const bool input = index < result.inputs.size();
    const TestTensor& tensor =
        input ? result.inputs[index]
              : result.outputs[index - result.inputs.size()];
    const std::vector<float> physical =
        input
            ? io::scatter(logical_inputs[index], tensor)
            : std::vector<float>(
                  io::storage_element_count(tensor), io::kPaddingSentinel);
    std::vector<std::uint8_t> encoded =
        io::encode(physical, tensor.data_type);
    auto buffer = std::make_unique<mv::DeviceBuffer>(
        tensor.binding_byte_offset + encoded.size());
    buffer->copy_from_host_at(encoded.data(),
                              encoded.size(),
                              tensor.binding_byte_offset,
                              stream.get());
    result.bindings.push_back(
        {tensor.uid, buffer->opaque_at(tensor.binding_byte_offset)});
    result.initial_bytes.push_back(std::move(encoded));
    result.buffers.push_back(std::move(buffer));
  }
  return result;
}

std::vector<std::uint8_t> read_tensor(const PreparedBuffers& prepared,
                                      std::size_t combined_index,
                                      mv::Stream& stream) {
  const bool input = combined_index < prepared.inputs.size();
  const TestTensor& tensor =
      input ? prepared.inputs.at(combined_index)
            : prepared.outputs.at(combined_index - prepared.inputs.size());
  std::vector<std::uint8_t> result =
      prepared.initial_bytes.at(combined_index);
  prepared.buffers.at(combined_index)
      ->copy_to_host_at(result.data(),
                       result.size(),
                       tensor.binding_byte_offset,
                       stream.get());
  stream.synchronize();
  return result;
}

std::vector<float> read_output(const PreparedBuffers& prepared,
                               std::size_t output_index,
                               mv::Stream& stream,
                               std::string_view provider) {
  const TestTensor& output = prepared.outputs.at(output_index);
  const std::vector<std::uint8_t> encoded = read_tensor(
      prepared, prepared.inputs.size() + output_index, stream);
  io::require_padding_unchanged(provider, encoded, output);
  return io::gather(io::decode(encoded, output.data_type), output);
}

void require_inputs_unchanged(const PreparedBuffers& prepared,
                              mv::Stream& stream,
                              std::string_view provider) {
  for (std::size_t index = 0; index < prepared.inputs.size(); ++index) {
    io::require_bytes_equal(
        std::string(provider) + " input " + std::to_string(index),
        read_tensor(prepared, index, stream),
        prepared.initial_bytes[index]);
  }
}

struct Accuracy {
  double maximum_absolute = 0.0;
  double maximum_relative = 0.0;
};

template <typename Case>
Accuracy compare(std::span<const float> actual,
                 std::span<const float> reference,
                 const Case& test_case,
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
        absolute /
        std::max({std::abs(left), std::abs(right), 1.0e-30});
    result.maximum_absolute =
        std::max(result.maximum_absolute, absolute);
    result.maximum_relative =
        std::max(result.maximum_relative, relative);
    if (!std::isfinite(absolute) ||
        (absolute > test_case.absolute_tolerance &&
         relative > test_case.relative_tolerance)) {
      std::ostringstream message;
      message << test_case.name << " output " << output_index
              << " differs at element " << index
              << ": FlagDNN=" << left << ", muDNN=" << right
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
    result.push_back(make_input(
        specifications[index], index, positives[index]));
  }
  return result;
}

template <typename Case>
Accuracy run_graph_case(const Case& test_case,
                        flagdnn::Handle& handle,
                        mv::Stream& stream) {
  validate_normalization_case(test_case);
  auto production = build_flagdnn(handle, test_case);
  auto reference = build_reference(test_case);
  const auto logical = logical_inputs(test_case);
  const std::vector<TestTensor> production_inputs = inputs(test_case);
  const std::vector<TestTensor> production_outputs = outputs(test_case);
  std::vector<TestTensor> reference_inputs;
  std::vector<TestTensor> reference_outputs;
  reference_inputs.reserve(production_inputs.size());
  reference_outputs.reserve(production_outputs.size());
  for (const TestTensor& tensor : production_inputs) {
    reference_inputs.push_back(reference_tensor(test_case, tensor));
  }
  for (const TestTensor& tensor : production_outputs) {
    reference_outputs.push_back(reference_tensor(test_case, tensor));
  }
  PreparedBuffers production_buffers = prepare_buffers(
      production_inputs, production_outputs, logical, stream);
  PreparedBuffers reference_buffers = prepare_buffers(
      std::move(reference_inputs),
      std::move(reference_outputs),
      logical,
      stream);
  mv::DeviceBuffer production_workspace(
      production->workspace_size(), 256);
  mv::DeviceBuffer reference_workspace(reference->workspace_size(), 256);
  stream.synchronize();
  production->prepare(production_buffers.bindings, stream.opaque());
  reference->prepare(reference_buffers.bindings, stream.opaque());
  production->execute(production_buffers.bindings,
                      production_workspace.opaque(),
                      production->workspace_size(),
                      stream.opaque());
  reference->execute(reference_buffers.bindings,
                     reference_workspace.opaque(),
                     reference->workspace_size(),
                     stream.opaque());
  mv::timing::paired(
      test_case.name, stream,
      [&] {
        production->execute(production_buffers.bindings,
                            production_workspace.opaque(),
                            production->workspace_size(), stream.opaque());
      },
      [&] {
        reference->execute(reference_buffers.bindings,
                           reference_workspace.opaque(),
                           reference->workspace_size(), stream.opaque());
      });
  stream.synchronize();
  require_inputs_unchanged(production_buffers, stream, "FlagDNN");
  require_inputs_unchanged(reference_buffers, stream, "muDNN");

  Accuracy aggregate;
  for (std::size_t index = 0; index < production_outputs.size(); ++index) {
    const Accuracy accuracy = compare(
        read_output(production_buffers, index, stream, "FlagDNN"),
        read_output(reference_buffers, index, stream, "muDNN"),
        test_case,
        index);
    aggregate.maximum_absolute =
        std::max(aggregate.maximum_absolute, accuracy.maximum_absolute);
    aggregate.maximum_relative =
        std::max(aggregate.maximum_relative, accuracy.maximum_relative);
  }
  return aggregate;
}

template <typename Case>
int run_suite(int argc,
              char** argv,
              std::span<const Case> cases,
              std::string_view suite_name) {
  if (argc != 3) {
    std::cerr << "usage: " << argv[0]
              << " COMPILER_EXECUTABLE COMPILER_ENTRY\n";
    return 2;
  }
  try {
    std::cout << std::setprecision(9);
    mv::check_musa(musaSetDevice(0), "musaSetDevice");
    mv::Stream stream;
    TemporaryCache cache;
    flagdnn::Handle handle("mthreads", 0);
    handle.set_compiler(argv[1], argv[2], cache.path().string());
    const char* filter = std::getenv("FLAGDNN_NORMALIZATION_CASE");
    std::size_t executed = 0;
    std::size_t skipped = 0;
    for (const Case& test_case : cases) {
      if (filter != nullptr && filter[0] != '\0' &&
          test_case.name.find(filter) == std::string::npos) {
        continue;
      }
      try {
        const Accuracy accuracy = run_graph_case(test_case, handle, stream);
        ++executed;
        std::cout << test_case.name
                  << ": FlagDNN Graph vs direct muDNN C++ PASS max_abs="
                  << accuracy.maximum_absolute
                  << " max_rel=" << accuracy.maximum_relative << '\n';
      } catch (const mv::ReferenceUnsupported& error) {
        ++skipped;
        mv::report_skip(test_case.name, error);
      }
    }
    if (executed + skipped == 0) {
      throw std::runtime_error(
          "FLAGDNN_NORMALIZATION_CASE matched no mthreads cases");
    }
    return mv::report_cases(std::string(suite_name), executed, skipped);
  } catch (const std::exception& error) {
    std::cerr << suite_name << "_FAILED: " << error.what() << '\n';
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
