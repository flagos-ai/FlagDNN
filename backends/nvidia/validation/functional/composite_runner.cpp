/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/composite.hpp"
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
         "flagdnn-composite-functional-XXXXXX")
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

std::vector<TestTensor> inputs(const AddSquareTestCase& test_case) {
  return {test_case.left, test_case.right};
}

std::vector<TestTensor> inputs(const ConvBiasReluTestCase& test_case) {
  return {test_case.x, test_case.w, test_case.bias};
}

const TestTensor& output(const AddSquareTestCase& test_case) {
  return test_case.output;
}

const TestTensor& output(const ConvBiasReluTestCase& test_case) {
  return test_case.output;
}

std::unique_ptr<CompositeExecutable> build_flagdnn(
    flagdnn::Handle& handle,
    const AddSquareTestCase& test_case) {
  return build_flagdnn_add_square(handle, test_case);
}

std::unique_ptr<CompositeExecutable> build_flagdnn(
    flagdnn::Handle& handle,
    const ConvBiasReluTestCase& test_case) {
  return build_flagdnn_conv_bias_relu(handle, test_case);
}

std::unique_ptr<CompositeExecutable> build_reference(
    const AddSquareTestCase& test_case) {
  return build_add_square_reference(test_case);
}

std::unique_ptr<CompositeExecutable> build_reference(
    const ConvBiasReluTestCase& test_case) {
  return build_conv_bias_relu_reference(test_case);
}

std::vector<float> make_input(std::size_t count, std::size_t tensor_index) {
  std::vector<float> result(count);
  for (std::size_t index = 0; index < count; ++index) {
    const int centered =
        static_cast<int>((index * 17 + tensor_index * 13) % 29) - 14;
    result[index] =
        static_cast<float>(centered) / static_cast<float>(31 + tensor_index);
  }
  return result;
}

std::vector<std::uint8_t> input_bytes(const TestTensor& tensor,
                                      std::size_t tensor_index) {
  return cuda::encode(
      cuda::scatter(make_input(cuda::element_count(tensor), tensor_index),
                    tensor),
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

struct PreparedBuffers {
  std::vector<TestTensor> inputs;
  TestTensor output;
  std::vector<std::unique_ptr<DeviceBuffer>> buffers;
  std::vector<flagdnnBinding_t> bindings;
};

PreparedBuffers prepare_buffers(std::vector<TestTensor> input_specs,
                                const TestTensor& output_spec,
                                Stream& stream) {
  PreparedBuffers result;
  result.inputs = std::move(input_specs);
  result.output = output_spec;
  result.buffers.reserve(result.inputs.size() + 1);
  result.bindings.reserve(result.inputs.size() + 1);
  for (std::size_t index = 0; index < result.inputs.size(); ++index) {
    const std::vector<std::uint8_t> encoded =
        input_bytes(result.inputs[index], index);
    auto buffer = std::make_unique<DeviceBuffer>(encoded.size());
    buffer->copy_from_host(encoded.data(), encoded.size(), stream.get());
    result.bindings.push_back(
        {result.inputs[index].uid, buffer->opaque()});
    result.buffers.push_back(std::move(buffer));
  }
  const std::vector<std::uint8_t> encoded = output_bytes(result.output);
  auto buffer = std::make_unique<DeviceBuffer>(encoded.size());
  buffer->copy_from_host(encoded.data(), encoded.size(), stream.get());
  result.bindings.push_back({result.output.uid, buffer->opaque()});
  result.buffers.push_back(std::move(buffer));
  return result;
}

std::vector<float> read_output(const PreparedBuffers& prepared,
                               Stream& stream,
                               std::string_view provider) {
  const DeviceBuffer& buffer = *prepared.buffers.back();
  std::vector<std::uint8_t> bytes(
      cuda::encoded_byte_count(prepared.output,
                               cuda::BooleanEncoding::kByte));
  buffer.copy_to_host(bytes.data(), bytes.size(), stream.get());
  stream.synchronize();
  const std::vector<float> physical = cuda::decode(
      bytes,
      prepared.output.data_type,
      cuda::storage_element_count(prepared.output),
      cuda::BooleanEncoding::kByte);
  cuda::require_padding_unchanged(provider, physical, prepared.output);
  return cuda::gather(physical, prepared.output);
}

void execute(CompositeExecutable& executable,
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
                 const Case& test_case) {
  if (actual.size() != reference.size()) {
    throw std::runtime_error("composite output sizes differ");
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
              << ": FlagDNN=" << left << ", cuDNN=" << right
              << ", abs=" << absolute << ", rel=" << relative
              << ", atol=" << test_case.absolute_tolerance
              << ", rtol=" << test_case.relative_tolerance;
      throw std::runtime_error(message.str());
    }
  }
  return result;
}

template <typename Case>
void run_case(const Case& test_case,
              flagdnn::Handle& handle,
              Stream& stream) {
  validate_composite_case(test_case);
  auto flagdnn = build_flagdnn(handle, test_case);
  auto reference = build_reference(test_case);
  PreparedBuffers flagdnn_buffers =
      prepare_buffers(inputs(test_case), output(test_case), stream);
  PreparedBuffers reference_buffers =
      prepare_buffers(inputs(test_case), output(test_case), stream);
  DeviceBuffer flagdnn_workspace(flagdnn->workspace_size());
  DeviceBuffer reference_workspace(reference->workspace_size());
  stream.synchronize();
  execute(*flagdnn, flagdnn_buffers.bindings, flagdnn_workspace, stream);
  execute(*reference,
          reference_buffers.bindings,
          reference_workspace,
          stream);
  stream.synchronize();
  const Accuracy accuracy = compare(
      read_output(flagdnn_buffers, stream, "FlagDNN"),
      read_output(reference_buffers, stream, "cuDNN"),
      test_case);
  std::cout << test_case.name << ": FlagDNN Graph vs cuDNN Graph PASS"
            << " max_abs=" << accuracy.maximum_absolute
            << " max_rel=" << accuracy.maximum_relative << std::endl;
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
    const char* filter = std::getenv("FLAGDNN_COMPOSITE_CASE");
    std::size_t executed = 0;
    for (const Case& test_case : cases) {
      if (filter != nullptr &&
          test_case.name.find(filter) == std::string::npos) {
        continue;
      }
      run_case(test_case, handle, stream);
      ++executed;
    }
    if (executed == 0) {
      throw std::runtime_error("composite filter matched no test cases");
    }
    std::cout << suite_name << ": PASS cases=" << executed << std::endl;
    return 0;
  } catch (const std::exception& error) {
    std::cerr << suite_name << "_FAILED: " << error.what() << std::endl;
    return 1;
  }
}

}  // namespace

int run_add_square_functional_test(
    int argc,
    char** argv,
    std::span<const AddSquareTestCase> cases) {
  return run_suite(argc, argv, cases, "FLAGDNN_ADD_SQUARE_FUNCTIONAL");
}

int run_conv_bias_relu_functional_test(
    int argc,
    char** argv,
    std::span<const ConvBiasReluTestCase> cases) {
  return run_suite(argc, argv, cases, "FLAGDNN_CONV_BIAS_RELU_FUNCTIONAL");
}

}  // namespace flagdnn::testing
