/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/pointwise.hpp"

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
         "flagdnn-mthreads-pointwise-functional-XXXXXX")
            .string();
    std::vector<char> writable(pattern.begin(), pattern.end());
    writable.push_back('\0');
    char* created = mkdtemp(writable.data());
    if (created == nullptr) {
      throw std::runtime_error("mkdtemp failed for pointwise cache");
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
                              PointwiseInputDomain domain) {
  std::vector<float> result(io::element_count(tensor));
  for (std::size_t index = 0; index < result.size(); ++index) {
    const int centered =
        static_cast<int>((index * 17U + input_index * 11U) % 41U) - 20;
    const float real_value =
        static_cast<float>(centered) /
        static_cast<float>(13U + input_index);
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
        result[index] = input_index == 1
                            ? std::abs(real_value) + 0.5F
                            : real_value;
        break;
      case PointwiseInputDomain::kPower:
        result[index] = input_index == 0
                            ? std::abs(real_value) + 0.5F
                            : std::fmod(std::abs(real_value), 2.0F) + 0.125F;
        break;
      case PointwiseInputDomain::kModuloSigned: {
        constexpr std::array<float, 6> left = {
            -3.0F, -3.0F, 3.0F, 3.0F, -5.5F, 5.5F};
        constexpr std::array<float, 6> right = {
            2.0F, -2.0F, 2.0F, -2.0F, 2.25F, -2.25F};
        result[index] = input_index == 0 ? left[index % left.size()]
                                         : right[index % right.size()];
        break;
      }
      case PointwiseInputDomain::kComparison: {
        const int base_centered =
            static_cast<int>((index * 17U) % 41U) - 20;
        const float base = static_cast<float>(base_centered) / 13.0F;
        if (input_index == 0 || index % 3U == 0U) {
          result[index] = base;
        } else if (index % 3U == 1U) {
          result[index] = base + 0.25F;
        } else {
          result[index] = base - 0.25F;
        }
        break;
      }
      case PointwiseInputDomain::kLogical:
        result[index] = input_index == 0
                            ? static_cast<float>((index / 2U) % 2U)
                            : static_cast<float>(index % 2U);
        break;
    }
  }
  return result;
}

struct EncodedTensor {
  std::vector<std::uint8_t> bytes;
  std::unique_ptr<mv::DeviceBuffer> device;
};

EncodedTensor make_input_tensor(const TestTensor& tensor,
                                std::span<const float> logical,
                                mv::Stream& stream) {
  EncodedTensor result;
  result.bytes = io::encode(io::scatter(logical, tensor), tensor.data_type);
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
  const std::vector<float> initial(
      io::storage_element_count(tensor), io::kPaddingSentinel);
  EncodedTensor result;
  result.bytes = io::encode(initial, tensor.data_type);
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

struct Accuracy {
  double maximum_absolute = 0.0;
  double maximum_relative = 0.0;
};

Accuracy compare(std::span<const float> actual,
                 std::span<const float> reference,
                 const PointwiseTestCase& test_case) {
  if (actual.size() != reference.size()) {
    throw std::runtime_error(
        "FlagDNN and muDNN pointwise output sizes differ");
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

struct PreparedExecution {
  std::vector<EncodedTensor> inputs;
  EncodedTensor output;
  std::vector<flagdnnBinding_t> bindings;
  std::unique_ptr<mv::DeviceBuffer> workspace;
};

PreparedExecution prepare_execution(
    const PointwiseTestCase& test_case,
    const std::vector<std::vector<float>>& logical_inputs,
    PointwiseExecutable& executable,
    mv::Stream& stream) {
  PreparedExecution result;
  result.inputs.reserve(test_case.inputs.size());
  result.bindings.reserve(test_case.inputs.size() + 1);
  for (std::size_t index = 0; index < test_case.inputs.size(); ++index) {
    result.inputs.push_back(make_input_tensor(
        test_case.inputs[index], logical_inputs[index], stream));
    result.bindings.push_back(
        {test_case.inputs[index].uid,
         result.inputs.back().device->opaque_at(
             test_case.inputs[index].binding_byte_offset)});
  }
  result.output = make_output_tensor(test_case.output, stream);
  result.bindings.push_back(
      {test_case.output.uid,
       result.output.device->opaque_at(
           test_case.output.binding_byte_offset)});
  result.workspace =
      std::make_unique<mv::DeviceBuffer>(executable.workspace_size(), 256);
  return result;
}

void enqueue(PointwiseExecutable& executable,
             PreparedExecution& prepared,
             mv::Stream& stream) {
  executable.prepare(prepared.bindings, stream.opaque());
  executable.execute(prepared.bindings,
                     prepared.workspace->opaque(),
                     executable.workspace_size(),
                     stream.opaque());
}

void require_identity_alignment_rejection(
    const PointwiseTestCase& test_case,
    PointwiseExecutable& executable,
    const PreparedExecution& prepared,
    mv::Stream& stream) {
  if (test_case.mode != FLAGDNN_POINTWISE_IDENTITY) {
    return;
  }

  const TestTensor& input = test_case.inputs.front();
  const EncodedTensor& encoded = prepared.inputs.front();
  mv::DeviceBuffer misaligned_storage(
      input.binding_byte_offset + encoded.bytes.size() + 1, 256);
  misaligned_storage.copy_from_host_at(
      encoded.bytes.data(),
      encoded.bytes.size(),
      input.binding_byte_offset + 1,
      stream.get());
  std::vector<flagdnnBinding_t> invalid_bindings = prepared.bindings;
  invalid_bindings.front().device_pointer = misaligned_storage.opaque_at(
      input.binding_byte_offset + 1);

  try {
    executable.execute(invalid_bindings,
                       prepared.workspace->opaque(),
                       executable.workspace_size(),
                       stream.opaque());
  } catch (const std::exception& error) {
    if (std::string_view(error.what()).find("misaligned") ==
        std::string_view::npos) {
      throw std::runtime_error(
          "identity misaligned binding failed for the wrong reason: " +
          std::string(error.what()));
    }
    return;
  }
  throw std::runtime_error(
      "identity device-copy path accepted a misaligned tensor binding");
}

Accuracy run_case(const PointwiseTestCase& test_case,
                  flagdnn::Handle& handle,
                  mv::Stream& stream) {
  validate_pointwise_case(test_case);
  auto production = build_flagdnn_pointwise(handle, test_case);
  auto reference = build_pointwise_reference(test_case);
  std::vector<std::vector<float>> logical_inputs;
  logical_inputs.reserve(test_case.inputs.size());
  for (std::size_t index = 0; index < test_case.inputs.size(); ++index) {
    logical_inputs.push_back(make_input(test_case.inputs[index],
                                        index,
                                        test_case.input_domains[index]));
  }
  PreparedExecution production_state = prepare_execution(
      test_case, logical_inputs, *production, stream);
  PreparedExecution reference_state = prepare_execution(
      test_case, logical_inputs, *reference, stream);
  stream.synchronize();
  enqueue(*production, production_state, stream);
  enqueue(*reference, reference_state, stream);
  require_identity_alignment_rejection(
      test_case, *production, production_state, stream);

  std::vector<std::vector<std::uint8_t>> production_inputs;
  std::vector<std::vector<std::uint8_t>> reference_inputs;
  for (std::size_t index = 0; index < test_case.inputs.size(); ++index) {
    production_inputs.push_back(read_back(production_state.inputs[index],
                                          test_case.inputs[index],
                                          stream));
    reference_inputs.push_back(read_back(reference_state.inputs[index],
                                         test_case.inputs[index],
                                         stream));
  }
  std::vector<std::uint8_t> production_output = read_back(
      production_state.output, test_case.output, stream);
  std::vector<std::uint8_t> reference_output = read_back(
      reference_state.output, test_case.output, stream);
  stream.synchronize();

  for (std::size_t index = 0; index < test_case.inputs.size(); ++index) {
    io::require_bytes_equal(
        "FlagDNN pointwise input " + std::to_string(index),
        production_inputs[index],
        production_state.inputs[index].bytes);
    io::require_bytes_equal(
        "muDNN pointwise input " + std::to_string(index),
        reference_inputs[index],
        reference_state.inputs[index].bytes);
  }
  io::require_padding_unchanged(
      "FlagDNN pointwise", production_output, test_case.output);
  io::require_padding_unchanged(
      "muDNN pointwise", reference_output, test_case.output);
  const std::vector<float> production_physical =
      io::decode(production_output, test_case.output.data_type);
  const std::vector<float> reference_physical =
      io::decode(reference_output, test_case.output.data_type);
  return compare(io::gather(production_physical, test_case.output),
                 io::gather(reference_physical, test_case.output),
                 test_case);
}

std::string case_filter_name(std::string_view suite_name) {
  constexpr std::string_view suffix = "_FUNCTIONAL";
  if (!suite_name.ends_with(suffix)) {
    return "FLAGDNN_POINTWISE_CASE";
  }
  return std::string(suite_name.substr(0, suite_name.size() - suffix.size())) +
         "_CASE";
}

}  // namespace

int run_pointwise_functional_test(
    int argc,
    char** argv,
    std::span<const PointwiseTestCase> cases,
    std::string_view suite_name) {
  // This adapter currently validates floating storage (and logical BOOL).
  // Other shared dtype/output combinations are enabled with their backend
  // support.
  std::vector<PointwiseTestCase> supported_cases(cases.begin(), cases.end());
  std::erase_if(supported_cases, [](const PointwiseTestCase& test_case) {
    const bool logical = test_case.mode == FLAGDNN_POINTWISE_LOGICAL_NOT ||
                         test_case.mode == FLAGDNN_POINTWISE_LOGICAL_AND ||
                         test_case.mode == FLAGDNN_POINTWISE_LOGICAL_OR ||
                         test_case.mode == FLAGDNN_POINTWISE_BINARY_SELECT;
    return std::any_of(test_case.inputs.begin(), test_case.inputs.end(),
                       [logical](const TestTensor& tensor) {
                         const auto type = tensor.data_type;
                         return type != FLAGDNN_DATA_FLOAT32 &&
                                type != FLAGDNN_DATA_FLOAT16 &&
                                type != FLAGDNN_DATA_BFLOAT16 &&
                                !(logical && type == FLAGDNN_DATA_BOOLEAN);
                       });
  });
  cases = supported_cases;

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

    const std::string filter_name = case_filter_name(suite_name);
    const char* filter = std::getenv(filter_name.c_str());
    std::size_t executed = 0;
    for (const PointwiseTestCase& test_case : cases) {
      if (filter != nullptr && filter[0] != '\0' &&
          test_case.name.find(filter) == std::string::npos) {
        continue;
      }
      const Accuracy accuracy = run_case(test_case, handle, stream);
      ++executed;
      std::cout << test_case.name
                << ": FlagDNN Graph vs muDNN tensor operator PASS max_abs="
                << accuracy.maximum_absolute
                << " max_rel=" << accuracy.maximum_relative << '\n';
    }
    if (executed == 0) {
      throw std::runtime_error(filter_name +
                               " matched no mthreads pointwise cases");
    }
    std::cout << suite_name << ": PASS cases=" << executed
              << " executed=" << executed << " skipped=0\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << suite_name << "_FAILED: " << error.what() << '\n';
    return 1;
  }
}

}  // namespace flagdnn::testing
