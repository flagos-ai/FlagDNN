/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/layout.hpp"

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
         "flagdnn-mthreads-layout-functional-XXXXXX")
            .string();
    std::vector<char> writable(pattern.begin(), pattern.end());
    writable.push_back('\0');
    char* created = mkdtemp(writable.data());
    if (created == nullptr) {
      throw std::runtime_error("mkdtemp failed for layout cache");
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

struct PreparedExecution {
  EncodedTensor input;
  EncodedTensor output;
  std::array<flagdnnBinding_t, 2> bindings;
  std::unique_ptr<mv::DeviceBuffer> workspace;
};

PreparedExecution prepare_execution(const LayoutTestCase& test_case,
                                    std::span<const float> logical_input,
                                    LayoutExecutable& executable,
                                    mv::Stream& stream) {
  PreparedExecution result;
  result.input = make_input_tensor(test_case.input, logical_input, stream);
  result.output = make_output_tensor(test_case.output, stream);
  result.bindings = {
      flagdnnBinding_t{
          test_case.input.uid,
          result.input.device->opaque_at(
              test_case.input.binding_byte_offset)},
      flagdnnBinding_t{
          test_case.output.uid,
          result.output.device->opaque_at(
              test_case.output.binding_byte_offset)},
  };
  result.workspace =
      std::make_unique<mv::DeviceBuffer>(executable.workspace_size(), 256);
  return result;
}

void enqueue(LayoutExecutable& executable,
             PreparedExecution& prepared,
             mv::Stream& stream) {
  executable.prepare(prepared.bindings, stream.opaque());
  executable.execute(prepared.bindings,
                     prepared.workspace->opaque(),
                     executable.workspace_size(),
                     stream.opaque());
}

struct Accuracy {
  double maximum_absolute = 0.0;
  double maximum_relative = 0.0;
};

Accuracy compare(std::span<const float> actual,
                 std::span<const float> reference,
                 const LayoutTestCase& test_case) {
  if (actual.size() != reference.size()) {
    throw std::runtime_error(
        "FlagDNN and muDNN layout output sizes differ");
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
    if (!std::isfinite(absolute) || absolute != 0.0) {
      std::ostringstream message;
      message << test_case.name << " differs at output element " << index
              << ": FlagDNN=" << left << ", muDNN=" << right
              << ", abs=" << absolute << ", rel=" << relative;
      throw std::runtime_error(message.str());
    }
  }
  return result;
}

Accuracy run_case(const LayoutTestCase& test_case,
                  flagdnn::Handle& handle,
                  mv::Stream& stream) {
  validate_layout_case(test_case);
  auto production = build_flagdnn_layout(handle, test_case);
  auto reference = build_layout_reference(test_case);
  const std::vector<float> logical_input =
      io::make_input(test_case.input, 0);
  PreparedExecution production_state = prepare_execution(
      test_case, logical_input, *production, stream);
  PreparedExecution reference_state = prepare_execution(
      test_case, logical_input, *reference, stream);
  stream.synchronize();
  enqueue(*production, production_state, stream);
  enqueue(*reference, reference_state, stream);

  const std::vector<std::uint8_t> production_input =
      read_back(production_state.input, test_case.input, stream);
  const std::vector<std::uint8_t> reference_input =
      read_back(reference_state.input, test_case.input, stream);
  const std::vector<std::uint8_t> production_output =
      read_back(production_state.output, test_case.output, stream);
  const std::vector<std::uint8_t> reference_output =
      read_back(reference_state.output, test_case.output, stream);
  stream.synchronize();

  io::require_bytes_equal("FlagDNN layout input",
                          production_input,
                          production_state.input.bytes);
  io::require_bytes_equal("muDNN layout input",
                          reference_input,
                          reference_state.input.bytes);
  io::require_padding_unchanged(
      "FlagDNN layout", production_output, test_case.output);
  io::require_padding_unchanged(
      "muDNN layout", reference_output, test_case.output);
  const std::vector<float> production_physical =
      io::decode(production_output, test_case.output.data_type);
  const std::vector<float> reference_physical =
      io::decode(reference_output, test_case.output.data_type);
  return compare(io::gather(production_physical, test_case.output),
                 io::gather(reference_physical, test_case.output),
                 test_case);
}

}  // namespace

int run_layout_functional_test(int argc,
                               char** argv,
                               std::span<const LayoutTestCase> cases,
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

    const char* filter = std::getenv("FLAGDNN_LAYOUT_CASE");
    std::size_t executed = 0;
    for (const LayoutTestCase& test_case : cases) {
      if (filter != nullptr && filter[0] != '\0' &&
          test_case.name.find(filter) == std::string::npos) {
        continue;
      }
      const Accuracy accuracy = run_case(test_case, handle, stream);
      ++executed;
      std::cout << test_case.name
                << ": FlagDNN Graph vs direct muDNN Permute PASS max_abs="
                << accuracy.maximum_absolute
                << " max_rel=" << accuracy.maximum_relative << '\n';
    }
    if (executed == 0) {
      throw std::runtime_error(
          "FLAGDNN_LAYOUT_CASE matched no mthreads layout cases");
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
