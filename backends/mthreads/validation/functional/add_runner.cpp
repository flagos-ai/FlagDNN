/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include <unistd.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <flagdnn/flagdnn.hpp>
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

#include "backends/mthreads/validation/functional/tensor_io_adapter.hpp"
#include "backends/mthreads/validation/musa_driver.hpp"
#include "backends/mthreads/validation/paired_timing.hpp"
#include "common/add.hpp"

namespace flagdnn::testing {
namespace {

namespace mv = validation::mthreads;
namespace io = validation::mthreads::tensor_io;

class TemporaryCache final {
 public:
  TemporaryCache() {
    std::string pattern =
        (std::filesystem::temp_directory_path() /
         "flagdnn-mthreads-add-functional-XXXXXX")
            .string();
    std::vector<char> writable(pattern.begin(), pattern.end());
    writable.push_back('\0');
    char* created = mkdtemp(writable.data());
    if (created == nullptr) {
      throw std::runtime_error("mkdtemp failed for Add compiler cache");
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

EncodedTensor make_input_tensor(
    const TestTensor& tensor,
    std::span<const float> logical,
    mv::Stream& stream) {
  if (tensor.binding_byte_offset != 0) {
    throw std::invalid_argument(
        "Phase 1 mthreads Add cases require zero binding offsets");
  }
  EncodedTensor result;
  result.bytes = io::encode(io::scatter(logical, tensor), tensor.data_type);
  result.device = std::make_unique<mv::DeviceBuffer>(result.bytes.size());
  result.device->copy_from_host(
      result.bytes.data(), result.bytes.size(), stream.get());
  return result;
}

EncodedTensor make_output_tensor(
    const TestTensor& tensor, mv::Stream& stream) {
  if (tensor.binding_byte_offset != 0) {
    throw std::invalid_argument(
        "Phase 1 mthreads Add cases require zero binding offsets");
  }
  const std::vector<float> initial(
      io::storage_element_count(tensor), io::kPaddingSentinel);
  EncodedTensor result;
  result.bytes = io::encode(initial, tensor.data_type);
  result.device = std::make_unique<mv::DeviceBuffer>(result.bytes.size());
  result.device->copy_from_host(
      result.bytes.data(), result.bytes.size(), stream.get());
  return result;
}

struct Accuracy {
  double maximum_absolute = 0.0;
  double maximum_relative = 0.0;
};

Accuracy compare(
    std::span<const float> actual,
    std::span<const float> reference,
    const AddTestCase& test_case) {
  if (actual.size() != reference.size()) {
    throw std::runtime_error(
        "FlagDNN and muDNN output element counts differ");
  }
  Accuracy result;
  for (std::size_t index = 0; index < actual.size(); ++index) {
    const double left = actual[index];
    const double right = reference[index];
    const double absolute = std::abs(left - right);
    const double relative = absolute /
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

void enqueue(
    AddExecutable& executable,
    std::span<const flagdnnBinding_t> bindings,
    mv::DeviceBuffer& workspace,
    mv::Stream& stream) {
  executable.execute(
      bindings,
      workspace.opaque(),
      executable.workspace_size(),
      stream.opaque());
}

void read_back(
    const EncodedTensor& tensor,
    std::vector<std::uint8_t>& output,
    mv::Stream& stream) {
  output.resize(tensor.bytes.size());
  tensor.device->copy_to_host(
      output.data(), output.size(), stream.get());
}

Accuracy run_case(
    const AddTestCase& test_case,
    flagdnn::Handle& handle,
    mv::Stream& stream) {
  validate_add_case(test_case);
  auto production = build_flagdnn_add(handle, test_case);
  auto reference = build_add_reference(test_case);

  const std::vector<float> left_values = io::make_input(test_case.left, 0);
  const std::vector<float> right_values = io::make_input(test_case.right, 1);
  EncodedTensor production_left =
      make_input_tensor(test_case.left, left_values, stream);
  EncodedTensor production_right =
      make_input_tensor(test_case.right, right_values, stream);
  EncodedTensor reference_left =
      make_input_tensor(test_case.left, left_values, stream);
  EncodedTensor reference_right =
      make_input_tensor(test_case.right, right_values, stream);
  EncodedTensor production_output =
      make_output_tensor(test_case.output, stream);
  EncodedTensor reference_output =
      make_output_tensor(test_case.output, stream);
  mv::DeviceBuffer production_workspace(
      production->workspace_size(), 256);
  mv::DeviceBuffer reference_workspace(
      reference->workspace_size(), 256);

  const std::array<flagdnnBinding_t, 3> production_bindings = {{
      {test_case.left.uid, production_left.device->opaque()},
      {test_case.right.uid, production_right.device->opaque()},
      {test_case.output.uid, production_output.device->opaque()},
  }};
  const std::array<flagdnnBinding_t, 3> reference_bindings = {{
      {test_case.left.uid, reference_left.device->opaque()},
      {test_case.right.uid, reference_right.device->opaque()},
      {test_case.output.uid, reference_output.device->opaque()},
  }};
  enqueue(
      *production, production_bindings, production_workspace, stream);
  enqueue(*reference, reference_bindings, reference_workspace, stream);
  mv::timing::paired(
      test_case.name, stream,
      [&] {
        enqueue(*production, production_bindings, production_workspace,
                stream);
      },
      [&] {
        enqueue(*reference, reference_bindings, reference_workspace, stream);
      });

  std::vector<std::uint8_t> observed_production_left;
  std::vector<std::uint8_t> observed_production_right;
  std::vector<std::uint8_t> observed_reference_left;
  std::vector<std::uint8_t> observed_reference_right;
  std::vector<std::uint8_t> observed_production_output;
  std::vector<std::uint8_t> observed_reference_output;
  read_back(production_left, observed_production_left, stream);
  read_back(production_right, observed_production_right, stream);
  read_back(reference_left, observed_reference_left, stream);
  read_back(reference_right, observed_reference_right, stream);
  read_back(production_output, observed_production_output, stream);
  read_back(reference_output, observed_reference_output, stream);
  stream.synchronize();

  io::require_bytes_equal(
      "FlagDNN left input",
      observed_production_left,
      production_left.bytes);
  io::require_bytes_equal(
      "FlagDNN right input",
      observed_production_right,
      production_right.bytes);
  io::require_bytes_equal(
      "muDNN left input", observed_reference_left, reference_left.bytes);
  io::require_bytes_equal(
      "muDNN right input", observed_reference_right, reference_right.bytes);
  io::require_padding_unchanged(
      "FlagDNN", observed_production_output, test_case.output);
  io::require_padding_unchanged(
      "muDNN", observed_reference_output, test_case.output);

  const std::vector<float> production_physical = io::decode(
      observed_production_output, test_case.output.data_type);
  const std::vector<float> reference_physical = io::decode(
      observed_reference_output, test_case.output.data_type);
  return compare(
      io::gather(production_physical, test_case.output),
      io::gather(reference_physical, test_case.output),
      test_case);
}

}  // namespace

int run_add_functional_test(int argc, char** argv,
                            std::span<const AddTestCase> cases) {
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

    const char* filter = std::getenv("FLAGDNN_ADD_CASE");
    std::size_t executed = 0;
    std::size_t skipped = 0;
    for (const AddTestCase& test_case : cases) {
      if (filter != nullptr &&
          test_case.name.find(filter) == std::string::npos) {
        continue;
      }
      try {
        const Accuracy accuracy = run_case(test_case, handle, stream);
        ++executed;
        std::cout << test_case.name
                  << ": FlagDNN Graph vs muDNN Binary PASS max_abs="
                  << accuracy.maximum_absolute
                  << " max_rel=" << accuracy.maximum_relative << '\n';
      } catch (const mv::ReferenceUnsupported& error) {
        ++skipped;
        mv::report_skip(test_case.name, error);
      }
    }
    if (executed + skipped == 0) {
      throw std::runtime_error(
          "FLAGDNN_ADD_CASE matched no mthreads Add cases");
    }
    return mv::report_cases("FLAGDNN_ADD_FUNCTIONAL", executed, skipped);
  } catch (const std::exception& error) {
    std::cerr << "FLAGDNN_ADD_FUNCTIONAL_FAILED: "
              << error.what() << '\n';
    return 1;
  }
}

}  // namespace flagdnn::testing
