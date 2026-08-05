/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/pointwise.hpp"
#include "validation/tensor_io.hpp"
#include "validation/cuda_driver.hpp"

#include <flagdnn/flagdnn.hpp>

#include <unistd.h>

#include <algorithm>
#include <array>
#include <cctype>
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

class TemporaryCache {
 public:
  TemporaryCache() {
    std::string pattern =
        (std::filesystem::temp_directory_path() /
         "flagdnn-pointwise-functional-XXXXXX")
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
                              PointwiseInputDomain domain) {
  std::vector<float> result(count);
  for (std::size_t index = 0; index < count; ++index) {
    const int centered =
        static_cast<int>((index * 17 + tensor_index * 11) % 41) - 20;
    const float real_value =
        static_cast<float>(centered) / static_cast<float>(13 + tensor_index);
    switch (domain) {
      case PointwiseInputDomain::kReal:
        result[index] = real_value;
        break;
      case PointwiseInputDomain::kPositive:
        result[index] = std::abs(real_value) + 0.5F;
        break;
      case PointwiseInputDomain::kScaled:
        result[index] = real_value * 4.0F;
        break;
      case PointwiseInputDomain::kTan:
        result[index] = static_cast<float>(centered) / 40.0F;
        break;
      case PointwiseInputDomain::kDivisor:
      case PointwiseInputDomain::kModulo:
        result[index] = tensor_index == 1
                            ? std::abs(real_value) + 0.5F
                            : real_value;
        break;
      case PointwiseInputDomain::kPower:
        result[index] = tensor_index == 0
                            ? std::abs(real_value) + 0.5F
                            : std::fmod(std::abs(real_value), 2.0F) + 0.125F;
        break;
      case PointwiseInputDomain::kModuloSigned: {
        constexpr std::array<float, 6> kLeft = {
            -3.0F, -3.0F, 3.0F, 3.0F, -5.5F, 5.5F};
        constexpr std::array<float, 6> kRight = {
            2.0F, -2.0F, 2.0F, -2.0F, 2.25F, -2.25F};
        result[index] = tensor_index == 0
                            ? kLeft[index % kLeft.size()]
                            : kRight[index % kRight.size()];
        break;
      }
      case PointwiseInputDomain::kComparison: {
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
      case PointwiseInputDomain::kLogical:
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

Accuracy compare(std::span<const float> actual,
                 std::span<const float> reference,
                 const PointwiseTestCase& test_case,
                 std::string_view reference_name) {
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
              << ": FlagDNN=" << left << ", " << reference_name << '='
              << right
              << ", abs=" << absolute << ", rel=" << relative
              << ", atol=" << test_case.absolute_tolerance
              << ", rtol=" << test_case.relative_tolerance;
      throw std::runtime_error(message.str());
    }
  }
  return result;
}

std::unique_ptr<DeviceBuffer> make_input_buffer(
    const TestTensor& tensor,
    std::size_t input_index,
    PointwiseInputDomain domain,
    cuda::BooleanEncoding boolean_encoding,
    Stream& stream) {
  const std::vector<float> logical =
      make_input(cuda::element_count(tensor), input_index, domain);
  const std::vector<float> physical = cuda::scatter(logical, tensor);
  const std::vector<std::uint8_t> encoded =
      cuda::encode(physical, tensor.data_type, boolean_encoding);
  auto result = std::make_unique<DeviceBuffer>(encoded.size());
  result->copy_from_host(encoded.data(), encoded.size(), stream.get());
  return result;
}

std::unique_ptr<DeviceBuffer> make_output_buffer(
    const TestTensor& tensor,
    cuda::BooleanEncoding boolean_encoding,
    Stream& stream) {
  const std::vector<float> initial(
      cuda::storage_element_count(tensor), cuda::padding_sentinel());
  const std::vector<std::uint8_t> encoded =
      cuda::encode(initial, tensor.data_type, boolean_encoding);
  auto result = std::make_unique<DeviceBuffer>(encoded.size());
  result->copy_from_host(encoded.data(), encoded.size(), stream.get());
  return result;
}

std::vector<float> read_output(const DeviceBuffer& buffer,
                               const TestTensor& tensor,
                               cuda::BooleanEncoding boolean_encoding,
                               Stream& stream) {
  std::vector<std::uint8_t> encoded(
      cuda::encoded_byte_count(tensor, boolean_encoding));
  buffer.copy_to_host(encoded.data(), encoded.size(), stream.get());
  stream.synchronize();
  return cuda::decode(encoded,
                      tensor.data_type,
                      cuda::storage_element_count(tensor),
                      boolean_encoding);
}

void execute(PointwiseExecutable& executable,
             std::span<const flagdnnBinding_t> bindings,
             DeviceBuffer& workspace,
             Stream& stream) {
  executable.execute(bindings,
                     workspace.opaque(),
                     executable.workspace_size(),
                     stream.opaque());
}

void run_case(const PointwiseTestCase& test_case,
              flagdnn::Handle& handle,
              Stream& stream) {
  validate_pointwise_case(test_case);
  auto flagdnn = build_flagdnn_pointwise(handle, test_case);
  auto reference = build_pointwise_reference(test_case);

  std::vector<std::unique_ptr<DeviceBuffer>> flagdnn_inputs;
  std::vector<std::unique_ptr<DeviceBuffer>> reference_inputs;
  std::vector<flagdnnBinding_t> flagdnn_bindings;
  std::vector<flagdnnBinding_t> reference_bindings;
  flagdnn_inputs.reserve(test_case.inputs.size());
  reference_inputs.reserve(test_case.inputs.size());
  flagdnn_bindings.reserve(test_case.inputs.size() + 1);
  reference_bindings.reserve(test_case.inputs.size() + 1);
  for (std::size_t index = 0; index < test_case.inputs.size(); ++index) {
    const TestTensor& tensor = test_case.inputs[index];
    auto flagdnn_input = make_input_buffer(tensor,
                                           index,
                                           test_case.input_domains[index],
                                           cuda::BooleanEncoding::kByte,
                                           stream);
    auto reference_input = make_input_buffer(
        tensor,
        index,
        test_case.input_domains[index],
        cuda::BooleanEncoding::kBitPacked,
        stream);
    flagdnn_bindings.push_back({tensor.uid, flagdnn_input->opaque()});
    reference_bindings.push_back({tensor.uid, reference_input->opaque()});
    flagdnn_inputs.push_back(std::move(flagdnn_input));
    reference_inputs.push_back(std::move(reference_input));
  }

  auto flagdnn_output = make_output_buffer(
      test_case.output, cuda::BooleanEncoding::kByte, stream);
  auto reference_output = make_output_buffer(
      test_case.output, cuda::BooleanEncoding::kBitPacked, stream);
  flagdnn_bindings.push_back(
      {test_case.output.uid, flagdnn_output->opaque()});
  reference_bindings.push_back(
      {test_case.output.uid, reference_output->opaque()});

  DeviceBuffer flagdnn_workspace(flagdnn->workspace_size());
  DeviceBuffer reference_workspace(reference->workspace_size());
  stream.synchronize();
  execute(*flagdnn, flagdnn_bindings, flagdnn_workspace, stream);
  execute(*reference, reference_bindings, reference_workspace, stream);
  stream.synchronize();

  const std::vector<float> flagdnn_physical = read_output(
      *flagdnn_output,
      test_case.output,
      cuda::BooleanEncoding::kByte,
      stream);
  const std::vector<float> reference_physical = read_output(
      *reference_output,
      test_case.output,
      cuda::BooleanEncoding::kBitPacked,
      stream);
  cuda::require_padding_unchanged(
      "FlagDNN", flagdnn_physical, test_case.output);
  if (test_case.output.data_type != FLAGDNN_DATA_BOOLEAN) {
    cuda::require_padding_unchanged(
        "cuDNN", reference_physical, test_case.output);
  }
  const Accuracy accuracy = compare(
      cuda::gather(flagdnn_physical, test_case.output),
      cuda::gather(reference_physical, test_case.output),
      test_case,
      "cuDNN");
  std::cout << test_case.name << ": FlagDNN Graph vs cuDNN Graph PASS"
            << " max_abs=" << accuracy.maximum_absolute
            << " max_rel=" << accuracy.maximum_relative << std::endl;
}

void run_host_reference_case(const PointwiseTestCase& test_case,
                             flagdnn::Handle& handle,
                             Stream& stream) {
  validate_pointwise_case(test_case);
  if (test_case.mode != FLAGDNN_POINTWISE_NEG &&
      test_case.mode != FLAGDNN_POINTWISE_BINARY_SELECT) {
    throw std::invalid_argument(
        "pointwise host oracle does not implement this mode");
  }

  auto flagdnn = build_flagdnn_pointwise(handle, test_case);
  std::vector<std::unique_ptr<DeviceBuffer>> inputs;
  std::vector<std::vector<float>> logical_inputs;
  std::vector<flagdnnBinding_t> bindings;
  inputs.reserve(test_case.inputs.size());
  logical_inputs.reserve(test_case.inputs.size());
  bindings.reserve(test_case.inputs.size() + 1);
  for (std::size_t index = 0; index < test_case.inputs.size(); ++index) {
    const TestTensor& tensor = test_case.inputs[index];
    logical_inputs.push_back(make_input(cuda::element_count(tensor),
                                        index,
                                        test_case.input_domains[index]));
    auto input = make_input_buffer(tensor,
                                   index,
                                   test_case.input_domains[index],
                                   cuda::BooleanEncoding::kByte,
                                   stream);
    bindings.push_back({tensor.uid, input->opaque()});
    inputs.push_back(std::move(input));
  }

  auto output = make_output_buffer(
      test_case.output, cuda::BooleanEncoding::kByte, stream);
  bindings.push_back({test_case.output.uid, output->opaque()});
  DeviceBuffer workspace(flagdnn->workspace_size());
  stream.synchronize();
  execute(*flagdnn, bindings, workspace, stream);
  stream.synchronize();

  const std::vector<float> physical = read_output(
      *output, test_case.output, cuda::BooleanEncoding::kByte, stream);
  cuda::require_padding_unchanged("FlagDNN", physical, test_case.output);
  std::vector<float> expected(cuda::element_count(test_case.output));
  if (test_case.mode == FLAGDNN_POINTWISE_NEG) {
    std::transform(logical_inputs[0].begin(),
                   logical_inputs[0].end(),
                   expected.begin(),
                   [](float value) { return -value; });
  } else {
    for (std::size_t index = 0; index < expected.size(); ++index) {
      expected[index] = logical_inputs[2][index] != 0.0F
                            ? logical_inputs[0][index]
                            : logical_inputs[1][index];
    }
  }
  const Accuracy accuracy = compare(cuda::gather(physical, test_case.output),
                                    expected,
                                    test_case,
                                    "host oracle");
  std::cout << test_case.name
            << ": FlagDNN Graph vs host semantic oracle PASS"
            << " (cuDNN Graph layout unavailable)"
            << " max_abs=" << accuracy.maximum_absolute
            << " max_rel=" << accuracy.maximum_relative << std::endl;
}

void run_logical_not_case(const PointwiseTestCase& test_case,
                          flagdnn::Handle& handle,
                          Stream& stream) {
  validate_pointwise_case(test_case);
  if (test_case.mode != FLAGDNN_POINTWISE_LOGICAL_NOT ||
      test_case.inputs.size() != 1 ||
      test_case.inputs.front().data_type != FLAGDNN_DATA_BOOLEAN ||
      test_case.output.data_type != FLAGDNN_DATA_BOOLEAN) {
    throw std::invalid_argument(
        "LogicalNot host-oracle runner received a non-LogicalNot case");
  }

  auto flagdnn = build_flagdnn_pointwise(handle, test_case);
  auto input = make_input_buffer(test_case.inputs.front(),
                                 0,
                                 test_case.input_domains.front(),
                                 cuda::BooleanEncoding::kByte,
                                 stream);
  auto output = make_output_buffer(
      test_case.output, cuda::BooleanEncoding::kByte, stream);
  const std::array<flagdnnBinding_t, 2> bindings = {
      flagdnnBinding_t{test_case.inputs.front().uid, input->opaque()},
      flagdnnBinding_t{test_case.output.uid, output->opaque()},
  };
  DeviceBuffer workspace(flagdnn->workspace_size());
  stream.synchronize();
  execute(*flagdnn, bindings, workspace, stream);
  stream.synchronize();

  const std::vector<float> physical = read_output(
      *output, test_case.output, cuda::BooleanEncoding::kByte, stream);
  cuda::require_padding_unchanged("FlagDNN", physical, test_case.output);
  const std::vector<float> input_logical =
      make_input(cuda::element_count(test_case.inputs.front()),
                 0,
                 test_case.input_domains.front());
  std::vector<float> expected(input_logical.size());
  std::transform(input_logical.begin(),
                 input_logical.end(),
                 expected.begin(),
                 [](float value) { return value == 0.0F ? 1.0F : 0.0F; });
  const Accuracy accuracy = compare(cuda::gather(physical, test_case.output),
                                    expected,
                                    test_case,
                                    "host oracle");
  std::cout << test_case.name
            << ": FlagDNN Graph vs host semantic oracle PASS"
            << " (cuDNN Graph unavailable)"
            << " max_abs=" << accuracy.maximum_absolute
            << " max_rel=" << accuracy.maximum_relative << std::endl;
}

void verify_logical_not_reference_unavailable(
    const PointwiseTestCase& test_case) {
  validate_pointwise_case(test_case);
  if (test_case.mode != FLAGDNN_POINTWISE_LOGICAL_NOT) {
    throw std::invalid_argument(
        "cuDNN LogicalNot capability gate received another pointwise mode");
  }

  std::string failure;
  try {
    auto reference = build_pointwise_reference(test_case);
    (void)reference;
  } catch (const std::exception& error) {
    failure = error.what();
  }
  if (failure.empty()) {
    throw std::runtime_error(
        "cuDNN LOGICAL_NOT is now supported; replace the host-oracle "
        "correctness test with a FlagDNN Graph vs cuDNN Graph comparison");
  }

  std::string normalized = failure;
  std::transform(normalized.begin(),
                 normalized.end(),
                 normalized.begin(),
                 [](unsigned char value) {
                   return static_cast<char>(std::tolower(value));
                 });
  if (normalized.find("no valid engine configs") == std::string::npos &&
      normalized.find("not supported") == std::string::npos &&
      normalized.find("cudnn_status_not_supported") == std::string::npos) {
    throw std::runtime_error(
        "cuDNN LOGICAL_NOT failed for an unexpected reason: " + failure);
  }
  std::cout << test_case.name
            << ": cuDNN Graph unsupported capability confirmed: " << failure
            << std::endl;
}

}  // namespace

int run_pointwise_functional_test(
    int argc,
    char** argv,
    std::span<const PointwiseTestCase> cases,
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

    const char* filter = std::getenv("FLAGDNN_POINTWISE_CASE");
    const bool expect_reference_unavailable =
        std::getenv("FLAGDNN_POINTWISE_EXPECT_REFERENCE_UNAVAILABLE") !=
        nullptr;
    std::size_t executed = 0;
    for (const PointwiseTestCase& test_case : cases) {
      if (filter != nullptr &&
          test_case.name.find(filter) == std::string::npos) {
        continue;
      }
      if (expect_reference_unavailable) {
        verify_logical_not_reference_unavailable(test_case);
      } else if (test_case.use_host_reference) {
        run_host_reference_case(test_case, handle, stream);
      } else if (test_case.mode == FLAGDNN_POINTWISE_LOGICAL_NOT) {
        run_logical_not_case(test_case, handle, stream);
      } else {
        run_case(test_case, handle, stream);
      }
      ++executed;
    }
    if (executed == 0) {
      throw std::runtime_error(
          "FLAGDNN_POINTWISE_CASE matched no test cases");
    }
    std::cout << suite_name << ": PASS cases=" << executed << std::endl;
    return 0;
  } catch (const std::exception& error) {
    std::cerr << suite_name << "_FAILED: " << error.what() << std::endl;
    return 1;
  }
}

}  // namespace flagdnn::testing
