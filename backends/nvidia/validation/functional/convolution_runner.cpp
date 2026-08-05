/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/convolution.hpp"
#include "validation/tensor_io.hpp"
#include "validation/cuda_driver.hpp"

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
#include <vector>

namespace flagdnn::testing {
namespace {

class TemporaryCache {
 public:
  TemporaryCache() {
    std::string pattern =
        (std::filesystem::temp_directory_path() /
         "flagdnn-convolution-functional-XXXXXX")
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

bool is_output(const ConvolutionTestCase& test_case,
               const TestTensor& tensor) {
  return tensor.uid == convolution_output_tensor(test_case).uid;
}

std::vector<std::uint8_t> initial_bytes(
    const ConvolutionTestCase& test_case,
    const TestTensor& tensor,
    std::size_t tensor_index) {
  return is_output(test_case, tensor) ? output_bytes(tensor)
                                     : input_bytes(tensor, tensor_index);
}

std::vector<float> read_tensor(const DeviceBuffer& buffer,
                               const TestTensor& tensor,
                               Stream& stream) {
  std::vector<std::uint8_t> bytes(
      cuda::encoded_byte_count(tensor, cuda::BooleanEncoding::kByte));
  buffer.copy_to_host(bytes.data(), bytes.size(), stream.get());
  stream.synchronize();
  return cuda::decode(bytes,
                      tensor.data_type,
                      cuda::storage_element_count(tensor),
                      cuda::BooleanEncoding::kByte);
}

struct Accuracy {
  double maximum_absolute = 0.0;
  double maximum_relative = 0.0;
};

Accuracy compare(std::span<const float> actual,
                 std::span<const float> reference,
                 const ConvolutionTestCase& test_case) {
  if (actual.size() != reference.size()) {
    throw std::runtime_error("convolution output sizes differ");
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

void execute(ConvolutionExecutable& executable,
             std::span<const flagdnnBinding_t> bindings,
             DeviceBuffer& workspace,
             Stream& stream) {
  executable.execute(bindings,
                     workspace.opaque(),
                     executable.workspace_size(),
                     stream.opaque());
}

void run_case(const ConvolutionTestCase& test_case,
              flagdnn::Handle& handle,
              Stream& stream) {
  auto flagdnn = build_flagdnn_convolution(handle, test_case);
  auto reference = build_convolution_reference(test_case);

  const std::array<const TestTensor*, 3> tensors = {
      &test_case.x, &test_case.w, &test_case.y};
  std::array<std::vector<std::uint8_t>, 3> bytes;
  for (std::size_t index = 0; index < tensors.size(); ++index) {
    bytes[index] = initial_bytes(test_case, *tensors[index], index);
  }

  std::array<std::unique_ptr<DeviceBuffer>, 3> flagdnn_buffers;
  std::array<std::unique_ptr<DeviceBuffer>, 3> reference_buffers;
  for (std::size_t index = 0; index < tensors.size(); ++index) {
    flagdnn_buffers[index] = std::make_unique<DeviceBuffer>(bytes[index].size());
    reference_buffers[index] =
        std::make_unique<DeviceBuffer>(bytes[index].size());
    flagdnn_buffers[index]->copy_from_host(
        bytes[index].data(), bytes[index].size(), stream.get());
    reference_buffers[index]->copy_from_host(
        bytes[index].data(), bytes[index].size(), stream.get());
  }

  std::array<flagdnnBinding_t, 3> flagdnn_bindings;
  std::array<flagdnnBinding_t, 3> reference_bindings;
  for (std::size_t index = 0; index < tensors.size(); ++index) {
    flagdnn_bindings[index] =
        flagdnnBinding_t{tensors[index]->uid,
                         flagdnn_buffers[index]->opaque()};
    reference_bindings[index] =
        flagdnnBinding_t{tensors[index]->uid,
                         reference_buffers[index]->opaque()};
  }

  DeviceBuffer flagdnn_workspace(flagdnn->workspace_size());
  DeviceBuffer reference_workspace(reference->workspace_size());
  stream.synchronize();
  execute(*flagdnn, flagdnn_bindings, flagdnn_workspace, stream);
  execute(*reference, reference_bindings, reference_workspace, stream);
  stream.synchronize();

  const TestTensor& output = convolution_output_tensor(test_case);
  std::size_t output_index = 0;
  while (output_index < tensors.size() &&
         tensors[output_index]->uid != output.uid) {
    ++output_index;
  }
  if (output_index == tensors.size()) {
    throw std::logic_error("convolution output tensor was not bound");
  }
  const std::vector<float> flagdnn_physical =
      read_tensor(*flagdnn_buffers[output_index], output, stream);
  const std::vector<float> reference_physical =
      read_tensor(*reference_buffers[output_index], output, stream);
  cuda::require_padding_unchanged("FlagDNN", flagdnn_physical, output);
  cuda::require_padding_unchanged("cuDNN", reference_physical, output);
  const Accuracy accuracy = compare(
      cuda::gather(flagdnn_physical, output),
      cuda::gather(reference_physical, output),
      test_case);
  std::cout << test_case.name << ": FlagDNN Graph vs cuDNN Graph PASS"
            << " max_abs=" << accuracy.maximum_absolute
            << " max_rel=" << accuracy.maximum_relative << std::endl;
}

}  // namespace

int run_convolution_functional_test(
    int argc,
    char** argv,
    std::span<const ConvolutionTestCase> cases,
    ConvolutionDirection expected_direction) {
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
    const char* filter = std::getenv("FLAGDNN_CONVOLUTION_CASE");

    std::size_t executed = 0;
    for (const ConvolutionTestCase& test_case : cases) {
      if (test_case.direction != expected_direction) {
        throw std::invalid_argument(
            "convolution suite contains the wrong direction");
      }
      if (filter != nullptr &&
          test_case.name.find(filter) == std::string::npos) {
        continue;
      }
      run_case(test_case, handle, stream);
      ++executed;
    }
    if (executed == 0) {
      throw std::runtime_error("convolution filter matched no test cases");
    }
    std::cout << "FLAGDNN_CONVOLUTION_FUNCTIONAL: PASS cases=" << executed
              << std::endl;
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FLAGDNN_CONVOLUTION_FUNCTIONAL_FAILED: " << error.what()
              << std::endl;
    return 1;
  }
}

}  // namespace flagdnn::testing
