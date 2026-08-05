/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/add.hpp"
#include "validation/cuda_driver.hpp"
#include "validation/tensor_io.hpp"


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
#include <vector>

namespace flagdnn::testing {
namespace {

constexpr float kPaddingSentinel =
    validation::nvidia::tensor_io::kPaddingSentinel;

class TemporaryCache {
 public:
  TemporaryCache() {
    std::string pattern =
        (std::filesystem::temp_directory_path() /
         "flagdnn-add-functional-XXXXXX")
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

std::size_t data_type_size(flagdnnDataType_t data_type) {
  return cuda::data_type_size(data_type);
}

std::size_t element_count(const TestTensor& tensor) {
  return cuda::element_count(tensor);
}

std::size_t storage_element_count(const TestTensor& tensor) {
  return cuda::storage_element_count(tensor);
}

std::vector<float> make_input(const TestTensor& tensor,
                              std::size_t input_index) {
  std::vector<float> result(element_count(tensor));
  for (std::size_t index = 0; index < result.size(); ++index) {
    const int centered =
        static_cast<int>((index * 17 + input_index * 11) % 41) - 20;
    result[index] =
        static_cast<float>(centered) / static_cast<float>(13 + input_index);
  }
  return result;
}

std::vector<float> scatter(std::span<const float> logical,
                           const TestTensor& tensor) {
  return cuda::scatter(logical, tensor);
}

std::vector<float> gather(std::span<const float> physical,
                          const TestTensor& tensor) {
  return cuda::gather(physical, tensor);
}

std::vector<std::uint8_t> encode(std::span<const float> values,
                                 flagdnnDataType_t data_type) {
  return cuda::encode(values, data_type, cuda::BooleanEncoding::kByte);
}

std::vector<float> decode(std::span<const std::uint8_t> bytes,
                          flagdnnDataType_t data_type) {
  const std::size_t count = bytes.size() / data_type_size(data_type);
  return cuda::decode(
      bytes, data_type, count, cuda::BooleanEncoding::kByte);
}

void require_padding_unchanged(std::string_view provider,
                               std::span<const float> physical,
                               const TestTensor& tensor) {
  cuda::require_padding_unchanged(provider, physical, tensor);
}

struct Accuracy {
  double maximum_absolute = 0.0;
  double maximum_relative = 0.0;
};

Accuracy compare(std::span<const float> actual,
                 std::span<const float> reference,
                 const AddTestCase& test_case) {
  if (actual.size() != reference.size()) {
    throw std::runtime_error("FlagDNN and reference output sizes differ");
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

std::unique_ptr<DeviceBuffer> make_input_buffer(const TestTensor& tensor,
                                                std::size_t input_index,
                                                Stream& stream) {
  const std::vector<float> logical = make_input(tensor, input_index);
  const std::vector<float> physical = scatter(logical, tensor);
  const std::vector<std::uint8_t> encoded =
      encode(physical, tensor.data_type);
  auto result = std::make_unique<DeviceBuffer>(encoded.size());
  result->copy_from_host(encoded.data(), encoded.size(), stream.get());
  return result;
}

std::unique_ptr<DeviceBuffer> make_output_buffer(const TestTensor& tensor,
                                                 Stream& stream) {
  const std::vector<float> initial(
      storage_element_count(tensor), kPaddingSentinel);
  const std::vector<std::uint8_t> encoded =
      encode(initial, tensor.data_type);
  auto result = std::make_unique<DeviceBuffer>(encoded.size());
  result->copy_from_host(encoded.data(), encoded.size(), stream.get());
  return result;
}

std::vector<float> read_output(const DeviceBuffer& buffer,
                               const TestTensor& tensor,
                               Stream& stream) {
  std::vector<std::uint8_t> encoded(
      storage_element_count(tensor) * data_type_size(tensor.data_type));
  buffer.copy_to_host(encoded.data(), encoded.size(), stream.get());
  stream.synchronize();
  return decode(encoded, tensor.data_type);
}

void execute(AddExecutable& executable,
             std::span<const flagdnnBinding_t> bindings,
             DeviceBuffer& workspace,
             Stream& stream) {
  executable.execute(bindings,
                     workspace.opaque(),
                     executable.workspace_size(),
                     stream.opaque());
}

void run_case(const AddTestCase& test_case,
              flagdnn::Handle& handle,
              Stream& stream) {
  validate_add_case(test_case);
  auto flagdnn = build_flagdnn_add(handle, test_case);
  auto reference = build_add_reference(test_case);

  auto left = make_input_buffer(test_case.left, 0, stream);
  auto right = make_input_buffer(test_case.right, 1, stream);
  auto flagdnn_output = make_output_buffer(test_case.output, stream);
  auto reference_output = make_output_buffer(test_case.output, stream);
  auto flagdnn_workspace =
      std::make_unique<DeviceBuffer>(flagdnn->workspace_size());
  auto reference_workspace =
      std::make_unique<DeviceBuffer>(reference->workspace_size());

  const std::vector<flagdnnBinding_t> flagdnn_bindings = {
      {test_case.left.uid, left->opaque()},
      {test_case.right.uid, right->opaque()},
      {test_case.output.uid, flagdnn_output->opaque()},
  };
  const std::vector<flagdnnBinding_t> reference_bindings = {
      {test_case.left.uid, left->opaque()},
      {test_case.right.uid, right->opaque()},
      {test_case.output.uid, reference_output->opaque()},
  };

  stream.synchronize();
  execute(*flagdnn, flagdnn_bindings, *flagdnn_workspace, stream);
  execute(*reference, reference_bindings, *reference_workspace, stream);
  stream.synchronize();

  const std::vector<float> flagdnn_physical =
      read_output(*flagdnn_output, test_case.output, stream);
  const std::vector<float> reference_physical =
      read_output(*reference_output, test_case.output, stream);
  require_padding_unchanged("FlagDNN", flagdnn_physical, test_case.output);
  require_padding_unchanged("cuDNN", reference_physical, test_case.output);
  const Accuracy accuracy = compare(gather(flagdnn_physical, test_case.output),
                                    gather(reference_physical, test_case.output),
                                    test_case);
  std::cout << test_case.name << ": FlagDNN Graph vs cuDNN Graph PASS"
            << " max_abs=" << accuracy.maximum_absolute
            << " max_rel=" << accuracy.maximum_relative << std::endl;
}

}  // namespace

int run_add_functional_test(int argc,
                            char** argv,
                            std::span<const AddTestCase> cases) {
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

    const char* filter = std::getenv("FLAGDNN_ADD_CASE");
    std::size_t executed = 0;
    for (const AddTestCase& test_case : cases) {
      if (filter != nullptr && test_case.name.find(filter) == std::string::npos) {
        continue;
      }
      run_case(test_case, handle, stream);
      ++executed;
    }
    if (executed == 0) {
      throw std::runtime_error("FLAGDNN_ADD_CASE matched no test cases");
    }
    std::cout << "FLAGDNN_ADD_FUNCTIONAL: PASS cases=" << executed
              << std::endl;
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FLAGDNN_ADD_FUNCTIONAL_FAILED: " << error.what()
              << std::endl;
    return 1;
  }
}

}  // namespace flagdnn::testing
