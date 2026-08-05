/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/matmul.hpp"
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

enum class RunMode { kGraphReference, kHostOracle, kCapability };

class TemporaryCache {
 public:
  TemporaryCache() {
    std::string pattern =
        (std::filesystem::temp_directory_path() /
         "flagdnn-matmul-functional-XXXXXX")
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

RunMode run_mode() {
  const char* value = std::getenv("FLAGDNN_MATMUL_MODE");
  if (value == nullptr || std::string_view(value) == "graph") {
    return RunMode::kGraphReference;
  }
  if (std::string_view(value) == "host") {
    return RunMode::kHostOracle;
  }
  if (std::string_view(value) == "capability") {
    return RunMode::kCapability;
  }
  throw std::invalid_argument(
      "FLAGDNN_MATMUL_MODE must be graph, host, or capability");
}

bool has_exact_graph_reference(const MatmulTestCase& test_case) {
  validate_matmul_case(test_case);
  return test_case.a.data_type != FLAGDNN_DATA_FLOAT32 &&
         test_case.output.dimensions.size() <= 3;
}

std::vector<float> make_input(std::size_t count, std::size_t tensor_index) {
  std::vector<float> result(count);
  for (std::size_t index = 0; index < count; ++index) {
    const int centered =
        static_cast<int>((index * 17 + tensor_index * 11) % 41) - 20;
    result[index] =
        static_cast<float>(centered) / static_cast<float>(13 + tensor_index);
  }
  return result;
}

struct InputData {
  std::vector<float> logical;
  std::vector<std::uint8_t> encoded;
};

InputData make_input_data(const TestTensor& tensor,
                          std::size_t tensor_index) {
  const std::vector<float> raw =
      make_input(cuda::element_count(tensor), tensor_index);
  InputData result;
  result.encoded = cuda::encode(cuda::scatter(raw, tensor),
                                tensor.data_type,
                                cuda::BooleanEncoding::kByte);
  result.logical = cuda::gather(
      cuda::decode(result.encoded,
                   tensor.data_type,
                   cuda::storage_element_count(tensor),
                   cuda::BooleanEncoding::kByte),
      tensor);
  return result;
}

std::vector<std::uint8_t> initial_output(const TestTensor& tensor) {
  const std::vector<float> values(
      cuda::storage_element_count(tensor), cuda::padding_sentinel());
  return cuda::encode(
      values, tensor.data_type, cuda::BooleanEncoding::kByte);
}

std::vector<float> read_output(const DeviceBuffer& buffer,
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

std::vector<float> host_reference(const MatmulTestCase& test_case,
                                  std::span<const float> a,
                                  std::span<const float> b) {
  const std::size_t output_rank = test_case.output.dimensions.size();
  const std::size_t batch_rank = output_rank - 2;
  const std::size_t m = static_cast<std::size_t>(
      test_case.output.dimensions[output_rank - 2]);
  const std::size_t n = static_cast<std::size_t>(
      test_case.output.dimensions[output_rank - 1]);
  const std::size_t k =
      static_cast<std::size_t>(test_case.a.dimensions.back());
  const std::size_t batch_count =
      cuda::element_count(test_case.output) / (m * n);
  const auto batch_offset =
      [batch_rank](const TestTensor& input,
                   std::span<const std::size_t> coordinates) {
        const std::size_t input_batch_rank = input.dimensions.size() - 2;
        const std::size_t leading = batch_rank - input_batch_rank;
        std::size_t result = 0;
        for (std::size_t axis = 0; axis < input_batch_rank; ++axis) {
          const std::size_t dimension =
              static_cast<std::size_t>(input.dimensions[axis]);
          result *= dimension;
          if (dimension != 1) {
            result += coordinates[leading + axis];
          }
        }
        return result;
      };

  std::vector<float> result(cuda::element_count(test_case.output));
  std::vector<std::size_t> batch_coordinates(batch_rank);
  for (std::size_t batch = 0; batch < batch_count; ++batch) {
    std::size_t remaining = batch;
    for (std::size_t axis = batch_rank; axis != 0; --axis) {
      const std::size_t current = axis - 1;
      const std::size_t dimension = static_cast<std::size_t>(
          test_case.output.dimensions[current]);
      batch_coordinates[current] = remaining % dimension;
      remaining /= dimension;
    }
    const std::size_t a_base =
        batch_offset(test_case.a, batch_coordinates) * m * k;
    const std::size_t b_base =
        batch_offset(test_case.b, batch_coordinates) * k * n;
    const std::size_t output_base = batch * m * n;
    for (std::size_t row = 0; row < m; ++row) {
      for (std::size_t column = 0; column < n; ++column) {
        float accumulator = 0.0F;
        for (std::size_t reduction = 0; reduction < k; ++reduction) {
          accumulator += a[a_base + row * k + reduction] *
                         b[b_base + reduction * n + column];
        }
        result[output_base + row * n + column] = accumulator;
      }
    }
  }
  const std::vector<std::uint8_t> encoded = cuda::encode(
      result, test_case.output.data_type, cuda::BooleanEncoding::kByte);
  return cuda::decode(encoded,
                      test_case.output.data_type,
                      result.size(),
                      cuda::BooleanEncoding::kByte);
}

struct Accuracy {
  double maximum_absolute = 0.0;
  double maximum_relative = 0.0;
};

Accuracy compare(std::span<const float> actual,
                 std::span<const float> reference,
                 const MatmulTestCase& test_case,
                 std::string_view reference_name) {
  if (actual.size() != reference.size()) {
    throw std::runtime_error("MatMul output sizes differ");
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
              << right << ", abs=" << absolute << ", rel=" << relative
              << ", atol=" << test_case.absolute_tolerance
              << ", rtol=" << test_case.relative_tolerance;
      throw std::runtime_error(message.str());
    }
  }
  return result;
}

void execute(MatmulExecutable& executable,
             std::span<const flagdnnBinding_t> bindings,
             DeviceBuffer& workspace,
             Stream& stream) {
  executable.execute(bindings,
                     workspace.opaque(),
                     executable.workspace_size(),
                     stream.opaque());
}

void run_graph_case(const MatmulTestCase& test_case,
                    flagdnn::Handle& handle,
                    Stream& stream) {
  auto flagdnn = build_flagdnn_matmul(handle, test_case);
  auto reference = build_matmul_reference(test_case);
  const InputData a_data = make_input_data(test_case.a, 0);
  const InputData b_data = make_input_data(test_case.b, 1);
  DeviceBuffer a(a_data.encoded.size());
  DeviceBuffer b(b_data.encoded.size());
  a.copy_from_host(a_data.encoded.data(), a_data.encoded.size(), stream.get());
  b.copy_from_host(b_data.encoded.data(), b_data.encoded.size(), stream.get());
  const std::vector<std::uint8_t> output_bytes =
      initial_output(test_case.output);
  DeviceBuffer flagdnn_output(output_bytes.size());
  DeviceBuffer reference_output(output_bytes.size());
  flagdnn_output.copy_from_host(
      output_bytes.data(), output_bytes.size(), stream.get());
  reference_output.copy_from_host(
      output_bytes.data(), output_bytes.size(), stream.get());
  const std::array<flagdnnBinding_t, 3> flagdnn_bindings = {
      flagdnnBinding_t{test_case.a.uid, a.opaque()},
      flagdnnBinding_t{test_case.b.uid, b.opaque()},
      flagdnnBinding_t{test_case.output.uid, flagdnn_output.opaque()},
  };
  const std::array<flagdnnBinding_t, 3> reference_bindings = {
      flagdnnBinding_t{test_case.a.uid, a.opaque()},
      flagdnnBinding_t{test_case.b.uid, b.opaque()},
      flagdnnBinding_t{test_case.output.uid, reference_output.opaque()},
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
      test_case,
      "cuDNN");
  std::cout << test_case.name << ": FlagDNN Graph vs cuDNN Graph PASS"
            << " max_abs=" << accuracy.maximum_absolute
            << " max_rel=" << accuracy.maximum_relative << std::endl;
}

void run_host_case(const MatmulTestCase& test_case,
                   flagdnn::Handle& handle,
                   Stream& stream) {
  auto flagdnn = build_flagdnn_matmul(handle, test_case);
  const InputData a_data = make_input_data(test_case.a, 0);
  const InputData b_data = make_input_data(test_case.b, 1);
  DeviceBuffer a(a_data.encoded.size());
  DeviceBuffer b(b_data.encoded.size());
  a.copy_from_host(a_data.encoded.data(), a_data.encoded.size(), stream.get());
  b.copy_from_host(b_data.encoded.data(), b_data.encoded.size(), stream.get());
  const std::vector<std::uint8_t> output_bytes =
      initial_output(test_case.output);
  DeviceBuffer output(output_bytes.size());
  output.copy_from_host(output_bytes.data(), output_bytes.size(), stream.get());
  const std::array<flagdnnBinding_t, 3> bindings = {
      flagdnnBinding_t{test_case.a.uid, a.opaque()},
      flagdnnBinding_t{test_case.b.uid, b.opaque()},
      flagdnnBinding_t{test_case.output.uid, output.opaque()},
  };
  DeviceBuffer workspace(flagdnn->workspace_size());
  stream.synchronize();
  execute(*flagdnn, bindings, workspace, stream);
  stream.synchronize();
  const std::vector<float> physical =
      read_output(output, test_case.output, stream);
  cuda::require_padding_unchanged("FlagDNN", physical, test_case.output);
  const Accuracy accuracy = compare(
      cuda::gather(physical, test_case.output),
      host_reference(test_case, a_data.logical, b_data.logical),
      test_case,
      "host oracle");
  const char* reason = test_case.a.data_type == FLAGDNN_DATA_FLOAT32
                           ? "cuDNN may use implicit TF32"
                           : "cuDNN lacks multidimensional batch broadcast";
  std::cout << test_case.name
            << ": FlagDNN Graph vs host semantic oracle PASS (" << reason
            << ") max_abs=" << accuracy.maximum_absolute
            << " max_rel=" << accuracy.maximum_relative << std::endl;
}

void verify_multidimensional_reference_unavailable(
    const MatmulTestCase& test_case) {
  if (test_case.output.dimensions.size() <= 3) {
    throw std::invalid_argument(
        "MatMul capability gate requires multidimensional batch broadcast");
  }
  std::string failure;
  try {
    auto reference = build_matmul_reference(test_case);
    (void)reference;
  } catch (const std::exception& error) {
    failure = error.what();
  }
  if (failure.empty()) {
    throw std::runtime_error(
        "cuDNN multidimensional MatMul is now supported; enable the "
        "Graph-vs-Graph correctness path");
  }
  std::string normalized = failure;
  std::transform(normalized.begin(),
                 normalized.end(),
                 normalized.begin(),
                 [](unsigned char value) {
                   return static_cast<char>(std::tolower(value));
                 });
  if (normalized.find("no valid engine configs") == std::string::npos &&
      normalized.find("not supported") == std::string::npos) {
    throw std::runtime_error(
        "cuDNN multidimensional MatMul failed unexpectedly: " + failure);
  }
  std::cout << test_case.name
            << ": cuDNN multidimensional MatMul unsupported capability "
               "confirmed: "
            << failure << std::endl;
}

}  // namespace

int run_matmul_functional_test(int argc,
                               char** argv,
                               std::span<const MatmulTestCase> cases) {
  if (argc != 3) {
    std::cerr << "usage: " << argv[0] << " COMPILER_EXECUTABLE COMPILER_ENTRY" << std::endl;
    return 2;
  }
  try {
    std::cout << std::setprecision(9);
    const RunMode mode = run_mode();
    DriverContext driver;
    Stream stream;
    TemporaryCache cache;
    flagdnn::Handle handle(FLAGDNN_BACKEND_NVIDIA, 0);
    handle.set_compiler(argv[1], argv[2], cache.path().string());
    const char* filter = std::getenv("FLAGDNN_MATMUL_CASE");

    std::size_t executed = 0;
    for (const MatmulTestCase& test_case : cases) {
      if (filter != nullptr &&
          test_case.name.find(filter) == std::string::npos) {
        continue;
      }
      const bool exact_reference = has_exact_graph_reference(test_case);
      if (mode == RunMode::kGraphReference && !exact_reference) {
        continue;
      }
      if (mode == RunMode::kHostOracle && exact_reference) {
        continue;
      }
      if (mode == RunMode::kCapability &&
          test_case.output.dimensions.size() <= 3) {
        continue;
      }
      if (mode == RunMode::kGraphReference) {
        run_graph_case(test_case, handle, stream);
      } else if (mode == RunMode::kHostOracle) {
        run_host_case(test_case, handle, stream);
      } else {
        verify_multidimensional_reference_unavailable(test_case);
      }
      ++executed;
    }
    if (executed == 0) {
      throw std::runtime_error("MatMul filters matched no test cases");
    }
    std::cout << "FLAGDNN_MATMUL_FUNCTIONAL: PASS cases=" << executed
              << std::endl;
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FLAGDNN_MATMUL_FUNCTIONAL_FAILED: " << error.what()
              << std::endl;
    return 1;
  }
}

}  // namespace flagdnn::testing
