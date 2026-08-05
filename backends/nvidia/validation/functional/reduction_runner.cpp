/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/reduction.hpp"
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
         "flagdnn-reduction-functional-XXXXXX")
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
                              flagdnnReductionMode_t mode) {
  std::vector<float> result(count);
  for (std::size_t index = 0; index < count; ++index) {
    const int centered = static_cast<int>((index * 17) % 41) - 20;
    const float value = static_cast<float>(centered) / 13.0F;
    result[index] = mode == FLAGDNN_REDUCTION_MUL
                        ? 1.0F + value * 0.125F
                        : value;
  }
  return result;
}

struct Accuracy {
  double maximum_absolute = 0.0;
  double maximum_relative = 0.0;
};

Accuracy compare(std::span<const float> actual,
                 std::span<const float> reference,
                 const ReductionTestCase& test_case) {
  if (actual.size() != reference.size()) {
    throw std::runtime_error("FlagDNN and cuDNN output sizes differ");
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

std::vector<std::uint8_t> encoded_input(
    std::span<const float> logical,
    const TestTensor& tensor) {
  return cuda::encode(cuda::scatter(logical, tensor),
                      tensor.data_type,
                      cuda::BooleanEncoding::kByte);
}

std::vector<std::uint8_t> encoded_output(const TestTensor& tensor) {
  const std::vector<float> initial(
      cuda::storage_element_count(tensor), cuda::padding_sentinel());
  return cuda::encode(
      initial, tensor.data_type, cuda::BooleanEncoding::kByte);
}

std::vector<float> read_output(const DeviceBuffer& buffer,
                               const TestTensor& tensor,
                               Stream& stream) {
  std::vector<std::uint8_t> encoded(
      cuda::encoded_byte_count(tensor, cuda::BooleanEncoding::kByte));
  buffer.copy_to_host_at(encoded.data(),
                         encoded.size(),
                         tensor.binding_byte_offset,
                         stream.get());
  stream.synchronize();
  return cuda::decode(encoded,
                      tensor.data_type,
                      cuda::storage_element_count(tensor),
                      cuda::BooleanEncoding::kByte);
}

void execute(ReductionExecutable& executable,
             std::span<const flagdnnBinding_t> bindings,
             DeviceBuffer& workspace,
             Stream& stream) {
  executable.execute(bindings,
                     workspace.opaque(),
                     executable.workspace_size(),
                     stream.opaque());
}

void run_case(const ReductionTestCase& test_case,
              flagdnn::Handle& handle,
              Stream& stream) {
  validate_reduction_case(test_case);
  auto flagdnn = build_flagdnn_reduction(handle, test_case);
  auto reference = build_reduction_reference(test_case);

  const std::vector<float> logical_input =
      make_input(cuda::element_count(test_case.input), test_case.mode);
  const TestTensor reference_input_specification =
      reduction_reference_input_tensor(test_case);
  const std::vector<std::uint8_t> flagdnn_input_bytes =
      encoded_input(logical_input, test_case.input);
  const std::vector<std::uint8_t> reference_input_bytes =
      encoded_input(logical_input, reference_input_specification);
  DeviceBuffer flagdnn_input(
      test_case.input.binding_byte_offset + flagdnn_input_bytes.size());
  DeviceBuffer reference_input(
      reference_input_specification.binding_byte_offset +
      reference_input_bytes.size());
  flagdnn_input.copy_from_host_at(flagdnn_input_bytes.data(),
                                  flagdnn_input_bytes.size(),
                                  test_case.input.binding_byte_offset,
                                  stream.get());
  reference_input.copy_from_host_at(
      reference_input_bytes.data(),
      reference_input_bytes.size(),
      reference_input_specification.binding_byte_offset,
      stream.get());

  const std::vector<std::uint8_t> output_bytes =
      encoded_output(test_case.output);
  DeviceBuffer flagdnn_output(
      test_case.output.binding_byte_offset + output_bytes.size());
  DeviceBuffer reference_output(
      test_case.output.binding_byte_offset + output_bytes.size());
  flagdnn_output.copy_from_host_at(output_bytes.data(),
                                   output_bytes.size(),
                                   test_case.output.binding_byte_offset,
                                   stream.get());
  reference_output.copy_from_host_at(output_bytes.data(),
                                     output_bytes.size(),
                                     test_case.output.binding_byte_offset,
                                     stream.get());

  const std::array<flagdnnBinding_t, 2> flagdnn_bindings = {
      flagdnnBinding_t{test_case.input.uid,
                       flagdnn_input.opaque_at(
                           test_case.input.binding_byte_offset)},
      flagdnnBinding_t{test_case.output.uid,
                       flagdnn_output.opaque_at(
                           test_case.output.binding_byte_offset)},
  };
  const std::array<flagdnnBinding_t, 2> reference_bindings = {
      flagdnnBinding_t{
          test_case.input.uid,
          reference_input.opaque_at(
              reference_input_specification.binding_byte_offset)},
      flagdnnBinding_t{test_case.output.uid,
                       reference_output.opaque_at(
                           test_case.output.binding_byte_offset)},
  };
  DeviceBuffer flagdnn_workspace(flagdnn->workspace_size());
  DeviceBuffer reference_workspace(reference->workspace_size());

  stream.synchronize();
  execute(*flagdnn, flagdnn_bindings, flagdnn_workspace, stream);
  execute(*reference, reference_bindings, reference_workspace, stream);
  stream.synchronize();

  const std::vector<float> flagdnn_physical =
      read_output(flagdnn_output, test_case.output, stream);
  const std::vector<float> reference_physical =
      read_output(reference_output, test_case.output, stream);
  cuda::require_padding_unchanged(
      "FlagDNN", flagdnn_physical, test_case.output);
  cuda::require_padding_unchanged(
      "cuDNN", reference_physical, test_case.output);
  const Accuracy accuracy = compare(
      cuda::gather(flagdnn_physical, test_case.output),
      cuda::gather(reference_physical, test_case.output),
      test_case);
  std::cout << test_case.name << ": FlagDNN Graph vs cuDNN Graph PASS"
            << " max_abs=" << accuracy.maximum_absolute
            << " max_rel=" << accuracy.maximum_relative << std::endl;
}

}  // namespace

int run_reduction_functional_test(
    int argc,
    char** argv,
    std::span<const ReductionTestCase> cases) {
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

    const char* filter = std::getenv("FLAGDNN_REDUCTION_CASE");
    std::size_t executed = 0;
    for (const ReductionTestCase& test_case : cases) {
      if (filter != nullptr &&
          test_case.name.find(filter) == std::string::npos) {
        continue;
      }
      run_case(test_case, handle, stream);
      ++executed;
    }
    if (executed == 0) {
      throw std::runtime_error("FLAGDNN_REDUCTION_CASE matched no test cases");
    }
    std::cout << "FLAGDNN_REDUCTION_FUNCTIONAL: PASS cases=" << executed
              << std::endl;
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FLAGDNN_REDUCTION_FUNCTIONAL_FAILED: " << error.what()
              << std::endl;
    return 1;
  }
}

}  // namespace flagdnn::testing
