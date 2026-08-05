/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/layout.hpp"
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
#include <numeric>
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
         "flagdnn-layout-functional-XXXXXX")
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

std::vector<float> make_input(std::size_t count) {
  std::vector<float> result(count);
  for (std::size_t index = 0; index < count; ++index) {
    const int centered = static_cast<int>((index * 17) % 41) - 20;
    result[index] = static_cast<float>(centered) / 13.0F;
  }
  return result;
}

std::vector<std::size_t> coordinates(
    std::size_t linear,
    std::span<const std::int64_t> dimensions) {
  std::vector<std::size_t> result(dimensions.size());
  for (std::size_t axis = dimensions.size(); axis != 0; --axis) {
    const std::size_t current = axis - 1;
    const std::size_t dimension =
        static_cast<std::size_t>(dimensions[current]);
    result[current] = linear % dimension;
    linear /= dimension;
  }
  return result;
}

std::size_t contiguous_offset(
    std::span<const std::size_t> coordinate,
    std::span<const std::int64_t> dimensions) {
  std::size_t result = 0;
  for (std::size_t axis = 0; axis < dimensions.size(); ++axis) {
    result = result * static_cast<std::size_t>(dimensions[axis]) +
             coordinate[axis];
  }
  return result;
}

std::vector<float> host_reference(const LayoutTestCase& test_case,
                                  std::span<const float> input) {
  if (test_case.operation == LayoutOperation::kReshape) {
    return {input.begin(), input.end()};
  }

  std::vector<float> result(cuda::element_count(test_case.output));
  for (std::size_t output_index = 0;
       output_index < result.size();
       ++output_index) {
    const std::vector<std::size_t> output_coordinate =
        coordinates(output_index, test_case.output.dimensions);
    std::vector<std::size_t> input_coordinate(
        test_case.input.dimensions.size());
    if (test_case.operation == LayoutOperation::kTranspose) {
      for (std::size_t output_axis = 0;
           output_axis < output_coordinate.size();
           ++output_axis) {
        input_coordinate[static_cast<std::size_t>(
            test_case.permutation[output_axis])] =
            output_coordinate[output_axis];
      }
    } else {
      for (std::size_t axis = 0; axis < output_coordinate.size(); ++axis) {
        input_coordinate[axis] = static_cast<std::size_t>(
            test_case.slices[axis].first +
            static_cast<std::int64_t>(output_coordinate[axis]) *
                test_case.slice_strides[axis]);
      }
    }
    result[output_index] = input[contiguous_offset(
        input_coordinate, test_case.input.dimensions)];
  }
  return result;
}

struct Accuracy {
  double maximum_absolute = 0.0;
  double maximum_relative = 0.0;
};

Accuracy compare(std::span<const float> actual,
                 std::span<const float> reference,
                 const LayoutTestCase& test_case) {
  if (actual.size() != reference.size()) {
    throw std::runtime_error("FlagDNN and host output sizes differ");
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
    if (!std::isfinite(absolute) || absolute != 0.0) {
      std::ostringstream message;
      message << test_case.name << " differs at output element " << index
              << ": FlagDNN=" << left << ", host oracle=" << right;
      throw std::runtime_error(message.str());
    }
  }
  return result;
}

void execute(LayoutExecutable& executable,
             std::span<const flagdnnBinding_t> bindings,
             DeviceBuffer& workspace,
             Stream& stream) {
  executable.execute(bindings,
                     workspace.opaque(),
                     executable.workspace_size(),
                     stream.opaque());
}

void run_host_oracle_case(const LayoutTestCase& test_case,
                          flagdnn::Handle& handle,
                          Stream& stream) {
  validate_layout_case(test_case);
  auto flagdnn = build_flagdnn_layout(handle, test_case);

  const std::vector<float> logical_input =
      make_input(cuda::element_count(test_case.input));
  const std::vector<std::uint8_t> input_bytes = cuda::encode(
      cuda::scatter(logical_input, test_case.input),
      test_case.input.data_type,
      cuda::BooleanEncoding::kByte);
  const std::vector<float> quantized_input = cuda::gather(
      cuda::decode(input_bytes,
                   test_case.input.data_type,
                   cuda::storage_element_count(test_case.input),
                   cuda::BooleanEncoding::kByte),
      test_case.input);
  DeviceBuffer input(input_bytes.size());
  input.copy_from_host(input_bytes.data(), input_bytes.size(), stream.get());

  const std::vector<float> output_initial(
      cuda::storage_element_count(test_case.output),
      cuda::padding_sentinel());
  const std::vector<std::uint8_t> output_bytes = cuda::encode(
      output_initial,
      test_case.output.data_type,
      cuda::BooleanEncoding::kByte);
  DeviceBuffer output(output_bytes.size());
  output.copy_from_host(output_bytes.data(), output_bytes.size(), stream.get());
  const std::array<flagdnnBinding_t, 2> bindings = {
      flagdnnBinding_t{test_case.input.uid, input.opaque()},
      flagdnnBinding_t{test_case.output.uid, output.opaque()},
  };
  DeviceBuffer workspace(flagdnn->workspace_size());
  stream.synchronize();
  execute(*flagdnn, bindings, workspace, stream);
  stream.synchronize();

  std::vector<std::uint8_t> actual_bytes(output_bytes.size());
  output.copy_to_host(actual_bytes.data(), actual_bytes.size(), stream.get());
  stream.synchronize();
  const std::vector<float> physical = cuda::decode(
      actual_bytes,
      test_case.output.data_type,
      cuda::storage_element_count(test_case.output),
      cuda::BooleanEncoding::kByte);
  cuda::require_padding_unchanged("FlagDNN", physical, test_case.output);
  const Accuracy accuracy = compare(
      cuda::gather(physical, test_case.output),
      host_reference(test_case, quantized_input),
      test_case);
  std::cout << test_case.name
            << ": FlagDNN Graph vs host semantic oracle PASS"
            << " (cuDNN standalone Graph unavailable)"
            << " max_abs=" << accuracy.maximum_absolute
            << " max_rel=" << accuracy.maximum_relative << std::endl;
}

void verify_reference_unavailable(const LayoutTestCase& test_case) {
  validate_layout_case(test_case);
  std::string failure;
  try {
    auto reference = build_layout_reference(test_case);
    (void)reference;
  } catch (const std::exception& error) {
    failure = error.what();
  }
  if (failure.empty()) {
    throw std::runtime_error(
        "cuDNN standalone Layout Graph is now supported; replace the host "
        "oracle correctness test with Graph-vs-Graph");
  }
  std::string normalized = failure;
  std::transform(normalized.begin(),
                 normalized.end(),
                 normalized.begin(),
                 [](unsigned char value) {
                   return static_cast<char>(std::tolower(value));
                 });
  if (normalized.find("no standalone execution plan") == std::string::npos) {
    throw std::runtime_error(
        "cuDNN Layout Graph failed for an unexpected reason: " + failure);
  }
  std::cout << test_case.name
            << ": cuDNN Layout unsupported capability confirmed: " << failure
            << std::endl;
}

}  // namespace

int run_layout_functional_test(int argc,
                               char** argv,
                               std::span<const LayoutTestCase> cases,
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

    const char* filter = std::getenv("FLAGDNN_LAYOUT_CASE");
    const bool expect_reference_unavailable =
        std::getenv("FLAGDNN_LAYOUT_EXPECT_REFERENCE_UNAVAILABLE") != nullptr;
    std::size_t executed = 0;
    for (const LayoutTestCase& test_case : cases) {
      if (filter != nullptr &&
          test_case.name.find(filter) == std::string::npos) {
        continue;
      }
      if (expect_reference_unavailable) {
        verify_reference_unavailable(test_case);
      } else {
        run_host_oracle_case(test_case, handle, stream);
      }
      ++executed;
    }
    if (executed == 0) {
      throw std::runtime_error("FLAGDNN_LAYOUT_CASE matched no test cases");
    }
    std::cout << suite_name << ": PASS cases=" << executed << std::endl;
    return 0;
  } catch (const std::exception& error) {
    std::cerr << suite_name << "_FAILED: " << error.what() << std::endl;
    return 1;
  }
}

}  // namespace flagdnn::testing
