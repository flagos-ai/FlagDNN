/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/composite.hpp"

#include "backends/mthreads/validation/functional/tensor_io_adapter.hpp"
#include "backends/mthreads/validation/musa_driver.hpp"

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
#include <utility>
#include <vector>

namespace flagdnn::testing {
namespace {

namespace mv = validation::mthreads;
namespace io = validation::mthreads::tensor_io;

class TemporaryCache final {
 public:
  TemporaryCache() {
    std::string pattern =
        (std::filesystem::temp_directory_path() /
         "flagdnn-mthreads-composite-functional-XXXXXX")
            .string();
    std::vector<char> writable(pattern.begin(), pattern.end());
    writable.push_back('\0');
    char* created = mkdtemp(writable.data());
    if (created == nullptr) {
      throw std::runtime_error("mkdtemp failed for composite cache");
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

struct EncodedTensor {
  std::vector<std::uint8_t> bytes;
  std::unique_ptr<mv::DeviceBuffer> device;
};

EncodedTensor make_input_tensor(const TestTensor& tensor,
                                std::size_t input_index,
                                mv::Stream& stream) {
  EncodedTensor result;
  result.bytes = io::encode(
      io::scatter(io::make_input(tensor, input_index), tensor),
      tensor.data_type);
  result.device = std::make_unique<mv::DeviceBuffer>(
      tensor.binding_byte_offset + result.bytes.size());
  result.device->copy_from_host_at(result.bytes.data(),
                                   result.bytes.size(),
                                   tensor.binding_byte_offset,
                                   stream.get());
  return result;
}

EncodedTensor make_output_tensor(const TestTensor& tensor,
                                 mv::Stream& stream) {
  EncodedTensor result;
  result.bytes = io::encode(
      std::vector<float>(io::storage_element_count(tensor),
                         io::kPaddingSentinel),
      tensor.data_type);
  result.device = std::make_unique<mv::DeviceBuffer>(
      tensor.binding_byte_offset + result.bytes.size());
  result.device->copy_from_host_at(result.bytes.data(),
                                   result.bytes.size(),
                                   tensor.binding_byte_offset,
                                   stream.get());
  return result;
}

std::vector<std::uint8_t> read_back(const EncodedTensor& tensor,
                                    const TestTensor& descriptor,
                                    mv::Stream& stream) {
  std::vector<std::uint8_t> result(tensor.bytes.size());
  tensor.device->copy_to_host_at(result.data(),
                                 result.size(),
                                 descriptor.binding_byte_offset,
                                 stream.get());
  return result;
}

struct PreparedBuffers {
  std::vector<TestTensor> input_descriptors;
  TestTensor output_descriptor;
  std::vector<EncodedTensor> inputs;
  EncodedTensor output;
  std::vector<flagdnnBinding_t> bindings;
};

PreparedBuffers prepare_buffers(std::span<const TestTensor> inputs,
                                const TestTensor& output,
                                mv::Stream& stream) {
  PreparedBuffers result;
  result.input_descriptors.assign(inputs.begin(), inputs.end());
  result.output_descriptor = output;
  result.inputs.reserve(inputs.size());
  result.bindings.reserve(inputs.size() + 1);
  for (std::size_t index = 0; index < inputs.size(); ++index) {
    const TestTensor& tensor = result.input_descriptors[index];
    result.inputs.push_back(make_input_tensor(tensor, index, stream));
    result.bindings.push_back({
        tensor.uid,
        result.inputs.back().device->opaque_at(tensor.binding_byte_offset),
    });
  }
  result.output = make_output_tensor(result.output_descriptor, stream);
  result.bindings.push_back({
      result.output_descriptor.uid,
      result.output.device->opaque_at(
          result.output_descriptor.binding_byte_offset),
  });
  return result;
}

PreparedBuffers prepare_buffers(const AddSquareTestCase& test_case,
                                mv::Stream& stream) {
  const std::array<TestTensor, 2> inputs = {
      test_case.left, test_case.right};
  return prepare_buffers(inputs, test_case.output, stream);
}

PreparedBuffers prepare_buffers(const ConvBiasReluTestCase& test_case,
                                mv::Stream& stream) {
  const std::array<TestTensor, 3> inputs = {
      test_case.x, test_case.w, test_case.bias};
  return prepare_buffers(inputs, test_case.output, stream);
}

struct Accuracy {
  double maximum_absolute = 0.0;
  double maximum_relative = 0.0;
};

Accuracy compare(std::span<const float> actual,
                 std::span<const float> reference,
                 const AddSquareTestCase& test_case) {
  if (actual.size() != reference.size()) {
    throw std::runtime_error("AddSquare output sizes differ");
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
      message << test_case.name << " differs at output element " << index
              << ": FlagDNN=" << left << ", muDNN=" << right
              << ", abs=" << absolute << ", rel=" << relative
              << ", atol=" << test_case.absolute_tolerance
              << ", rtol=" << test_case.relative_tolerance;
      throw std::runtime_error(message.str());
    }
  }
  return result;
}

Accuracy compare(std::span<const float> actual,
                 std::span<const float> reference,
                 const ConvBiasReluTestCase& test_case) {
  if (actual.size() != reference.size()) {
    throw std::runtime_error("ConvBiasRelu output sizes differ");
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
      message << test_case.name << " differs at output element " << index
              << ": FlagDNN=" << left << ", muDNN=" << right
              << ", abs=" << absolute << ", rel=" << relative
              << ", atol=" << test_case.absolute_tolerance
              << ", rtol=" << test_case.relative_tolerance;
      throw std::runtime_error(message.str());
    }
  }
  return result;
}

std::vector<float> checked_output(PreparedBuffers& buffers,
                                  mv::Stream& stream,
                                  std::string_view provider,
                                  std::string_view operation) {
  std::vector<std::vector<std::uint8_t>> observed_inputs;
  observed_inputs.reserve(buffers.inputs.size());
  for (std::size_t index = 0; index < buffers.inputs.size(); ++index) {
    observed_inputs.push_back(read_back(buffers.inputs[index],
                                        buffers.input_descriptors[index],
                                        stream));
  }
  const std::vector<std::uint8_t> output =
      read_back(buffers.output, buffers.output_descriptor, stream);
  stream.synchronize();
  for (std::size_t index = 0; index < buffers.inputs.size(); ++index) {
    io::require_bytes_equal(
        std::string(provider) + " " + std::string(operation) + " input " +
            std::to_string(index),
        observed_inputs[index],
        buffers.inputs[index].bytes);
  }
  io::require_padding_unchanged(provider, output, buffers.output_descriptor);
  return io::gather(
      io::decode(output, buffers.output_descriptor.data_type),
      buffers.output_descriptor);
}

Accuracy run_case(const AddSquareTestCase& test_case,
                  flagdnn::Handle& handle,
                  mv::Stream& stream) {
  validate_composite_case(test_case);
  auto production = build_flagdnn_add_square(handle, test_case);
  auto reference = build_add_square_reference(test_case);
  PreparedBuffers production_buffers = prepare_buffers(test_case, stream);
  PreparedBuffers reference_buffers = prepare_buffers(test_case, stream);
  mv::DeviceBuffer production_workspace(
      production->workspace_size(), 256);
  mv::DeviceBuffer reference_workspace(
      reference->workspace_size(), 256);
  stream.synchronize();
  production->execute(production_buffers.bindings,
                      production_workspace.opaque(),
                      production->workspace_size(),
                      stream.opaque());
  reference->execute(reference_buffers.bindings,
                     reference_workspace.opaque(),
                     reference->workspace_size(),
                     stream.opaque());
  const std::vector<float> production_output = checked_output(
      production_buffers, stream, "FlagDNN", "AddSquare");
  const std::vector<float> reference_output = checked_output(
      reference_buffers, stream, "muDNN", "AddSquare");
  return compare(production_output, reference_output, test_case);
}

Accuracy run_case(const ConvBiasReluTestCase& test_case,
                  flagdnn::Handle& handle,
                  mv::Stream& stream) {
  validate_composite_case(test_case);
  auto production = build_flagdnn_conv_bias_relu(handle, test_case);
  auto reference = build_conv_bias_relu_reference(test_case);
  PreparedBuffers production_buffers = prepare_buffers(test_case, stream);
  PreparedBuffers reference_buffers = prepare_buffers(test_case, stream);
  mv::DeviceBuffer production_workspace(
      production->workspace_size(), 256);
  mv::DeviceBuffer reference_workspace(
      reference->workspace_size(), 256);
  stream.synchronize();
  production->execute(production_buffers.bindings,
                      production_workspace.opaque(),
                      production->workspace_size(),
                      stream.opaque());
  reference->execute(reference_buffers.bindings,
                     reference_workspace.opaque(),
                     reference->workspace_size(),
                     stream.opaque());
  const std::vector<float> production_output = checked_output(
      production_buffers, stream, "FlagDNN", "ConvBiasRelu");
  const std::vector<float> reference_output = checked_output(
      reference_buffers, stream, "muDNN", "ConvBiasRelu");
  return compare(production_output, reference_output, test_case);
}

}  // namespace

int run_add_square_functional_test(
    int argc,
    char** argv,
    std::span<const AddSquareTestCase> cases) {
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
    const char* filter = std::getenv("FLAGDNN_COMPOSITE_CASE");
    std::size_t executed = 0;
    for (const AddSquareTestCase& test_case : cases) {
      if (filter != nullptr && filter[0] != '\0' &&
          test_case.name.find(filter) == std::string::npos) {
        continue;
      }
      const Accuracy accuracy = run_case(test_case, handle, stream);
      ++executed;
      std::cout << test_case.name
                << ": FlagDNN Graph vs direct muDNN Binary sequence PASS"
                << " max_abs=" << accuracy.maximum_absolute
                << " max_rel=" << accuracy.maximum_relative << '\n';
    }
    if (executed == 0) {
      throw std::runtime_error(
          "FLAGDNN_COMPOSITE_CASE matched no AddSquare cases");
    }
    std::cout << "FLAGDNN_ADD_SQUARE_FUNCTIONAL: PASS cases=" << executed
              << " executed=" << executed << " skipped=0\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FLAGDNN_ADD_SQUARE_FUNCTIONAL_FAILED: "
              << error.what() << '\n';
    return 1;
  }
}

int run_conv_bias_relu_functional_test(
    int argc,
    char** argv,
    std::span<const ConvBiasReluTestCase> cases) {
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
    const char* filter = std::getenv("FLAGDNN_COMPOSITE_CASE");
    std::size_t executed = 0;
    for (const ConvBiasReluTestCase& test_case : cases) {
      if (filter != nullptr && filter[0] != '\0' &&
          test_case.name.find(filter) == std::string::npos) {
        continue;
      }
      const Accuracy accuracy = run_case(test_case, handle, stream);
      ++executed;
      std::cout << test_case.name
                << ": FlagDNN Graph vs direct muDNN "
                   "Convolution/Binary/Unary sequence PASS"
                << " max_abs=" << accuracy.maximum_absolute
                << " max_rel=" << accuracy.maximum_relative << '\n';
    }
    if (executed == 0) {
      throw std::runtime_error(
          "FLAGDNN_COMPOSITE_CASE matched no ConvBiasRelu cases");
    }
    std::cout
        << "FLAGDNN_CONV_BIAS_RELU_FUNCTIONAL: PASS cases=" << executed
        << " executed=" << executed << " skipped=0\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FLAGDNN_CONV_BIAS_RELU_FUNCTIONAL_FAILED: "
              << error.what() << '\n';
    return 1;
  }
}

}  // namespace flagdnn::testing
