/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include <unistd.h>

#include <algorithm>
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
#include <vector>

#include "backends/mthreads/validation/functional/tensor_io_adapter.hpp"
#include "backends/mthreads/validation/musa_driver.hpp"
#include "backends/mthreads/validation/paired_timing.hpp"
#include "common/convolution.hpp"

namespace flagdnn::testing {
namespace {

namespace mv = validation::mthreads;
namespace io = validation::mthreads::tensor_io;

bool trace_enabled() noexcept {
  static const bool enabled = [] {
    const char* value = std::getenv(
        "FLAGDNN_MTHREADS_CONVOLUTION_TRACE");
    return value != nullptr && value[0] != '\0' &&
           std::string_view(value) != "0";
  }();
  return enabled;
}

void trace_case(const ConvolutionTestCase& test_case,
                std::string_view phase) {
  if (trace_enabled()) {
    std::cerr << "MTHREADS_CONVOLUTION_TRACE case="
              << test_case.name << " phase=" << phase << std::endl;
  }
}

class TemporaryCache final {
 public:
  TemporaryCache() {
    const char* configured = std::getenv(
        "FLAGDNN_MTHREADS_CONVOLUTION_CACHE_DIRECTORY");
    if (configured != nullptr && configured[0] != '\0') {
      path_ = configured;
      std::error_code error;
      std::filesystem::create_directories(path_, error);
      if (error) {
        throw std::runtime_error(
            "cannot create persistent Convolution cache: " +
            error.message());
      }
      persistent_ = true;
      return;
    }
    std::string pattern =
        (std::filesystem::temp_directory_path() /
         "flagdnn-mthreads-convolution-functional-XXXXXX")
            .string();
    std::vector<char> writable(pattern.begin(), pattern.end());
    writable.push_back('\0');
    char* created = mkdtemp(writable.data());
    if (created == nullptr) {
      throw std::runtime_error(
          "mkdtemp failed for Convolution cache");
    }
    path_ = created;
  }

  ~TemporaryCache() {
    if (!persistent_) {
      std::error_code ignored;
      std::filesystem::remove_all(path_, ignored);
    }
  }

  [[nodiscard]] const std::filesystem::path& path() const noexcept {
    return path_;
  }

 private:
  std::filesystem::path path_;
  bool persistent_ = false;
};

std::size_t output_index(const ConvolutionTestCase& test_case) {
  switch (test_case.direction) {
    case ConvolutionDirection::kFprop:
      return 2;
    case ConvolutionDirection::kDgrad:
      return 0;
    case ConvolutionDirection::kWgrad:
      return 1;
  }
  throw std::invalid_argument(
      "unsupported MThreads Convolution direction");
}

std::string_view operation_name(
    ConvolutionDirection direction) noexcept {
  switch (direction) {
    case ConvolutionDirection::kFprop:
      return "FProp";
    case ConvolutionDirection::kDgrad:
      return "Dgrad";
    case ConvolutionDirection::kWgrad:
      return "Wgrad";
  }
  return "Convolution";
}

struct EncodedTensor {
  std::vector<std::uint8_t> bytes;
  std::unique_ptr<mv::DeviceBuffer> device;
};

EncodedTensor make_tensor(const TestTensor& tensor,
                          std::size_t tensor_index,
                          bool output,
                          mv::Stream& stream) {
  std::vector<float> physical;
  if (output) {
    physical.assign(
        io::storage_element_count(tensor), io::kPaddingSentinel);
  } else {
    physical = io::scatter(
        io::make_input(tensor, tensor_index), tensor);
  }
  EncodedTensor result;
  result.bytes = io::encode(physical, tensor.data_type);
  result.device = std::make_unique<mv::DeviceBuffer>(
      tensor.binding_byte_offset + result.bytes.size());
  result.device->copy_from_host_at(
      result.bytes.data(),
      result.bytes.size(),
      tensor.binding_byte_offset,
      stream.get());
  return result;
}

struct PreparedExecution {
  std::vector<TestTensor> tensors;
  std::size_t output = 0;
  std::vector<EncodedTensor> storage;
  std::vector<flagdnnBinding_t> bindings;
  std::unique_ptr<mv::DeviceBuffer> workspace;
};

PreparedExecution prepare_execution(
    const ConvolutionTestCase& test_case,
    ConvolutionExecutable& executable,
    mv::Stream& stream) {
  PreparedExecution result;
  result.tensors = {test_case.x, test_case.w, test_case.y};
  result.output = output_index(test_case);
  result.storage.reserve(result.tensors.size());
  result.bindings.reserve(result.tensors.size());
  for (std::size_t index = 0;
       index < result.tensors.size(); ++index) {
    const TestTensor& tensor = result.tensors[index];
    result.storage.push_back(
        make_tensor(tensor, index, index == result.output, stream));
    result.bindings.push_back(
        {tensor.uid,
         result.storage.back().device->opaque_at(
             tensor.binding_byte_offset)});
  }
  result.workspace = std::make_unique<mv::DeviceBuffer>(
      executable.workspace_size(), 256);
  return result;
}

void enqueue(ConvolutionExecutable& executable,
             PreparedExecution& prepared,
             mv::Stream& stream) {
  executable.prepare(prepared.bindings, stream.opaque());
  executable.execute(
      prepared.bindings,
      prepared.workspace->opaque(),
      executable.workspace_size(),
      stream.opaque());
}

std::vector<std::uint8_t> read_back(
    const PreparedExecution& prepared,
    std::size_t index,
    mv::Stream& stream) {
  const TestTensor& tensor = prepared.tensors.at(index);
  std::vector<std::uint8_t> result =
      prepared.storage.at(index).bytes;
  prepared.storage.at(index).device->copy_to_host_at(
      result.data(),
      result.size(),
      tensor.binding_byte_offset,
      stream.get());
  stream.synchronize();
  return result;
}

std::vector<float> read_output(
    const PreparedExecution& prepared,
    mv::Stream& stream,
    std::string_view provider) {
  const TestTensor& output = prepared.tensors.at(prepared.output);
  const std::vector<std::uint8_t> encoded =
      read_back(prepared, prepared.output, stream);
  io::require_padding_unchanged(provider, encoded, output);
  return io::gather(io::decode(encoded, output.data_type), output);
}

struct Accuracy {
  double maximum_absolute = 0.0;
  double maximum_relative = 0.0;
};

Accuracy compare(std::span<const float> actual,
                 std::span<const float> reference,
                 const ConvolutionTestCase& test_case) {
  if (actual.size() != reference.size()) {
    throw std::runtime_error(
        "FlagDNN and muDNN Convolution output sizes differ");
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
      message << test_case.name << " differs at output element "
              << index << ": FlagDNN=" << left
              << ", muDNN=" << right
              << ", abs=" << absolute << ", rel=" << relative
              << ", atol=" << test_case.absolute_tolerance
              << ", rtol=" << test_case.relative_tolerance;
      throw std::runtime_error(message.str());
    }
  }
  return result;
}

Accuracy run_case(const ConvolutionTestCase& test_case,
                  flagdnn::Handle& handle,
                  mv::Stream& stream) {
  validate_convolution_case(test_case);
  trace_case(test_case, "production-build-begin");
  auto production =
      build_flagdnn_convolution(handle, test_case);
  trace_case(test_case, "production-build-end");
  trace_case(test_case, "reference-build-begin");
  auto reference = build_convolution_reference(test_case);
  trace_case(test_case, "reference-build-end");
  trace_case(test_case, "production-prepare-begin");
  PreparedExecution production_state =
      prepare_execution(test_case, *production, stream);
  trace_case(test_case, "production-prepare-end");
  trace_case(test_case, "reference-prepare-begin");
  PreparedExecution reference_state =
      prepare_execution(test_case, *reference, stream);
  trace_case(test_case, "reference-prepare-end");
  stream.synchronize();
  trace_case(test_case, "production-enqueue-begin");
  enqueue(*production, production_state, stream);
  trace_case(test_case, "production-enqueue-end");
  if (trace_enabled()) {
    trace_case(test_case, "production-synchronize-begin");
    stream.synchronize();
    trace_case(test_case, "production-synchronize-end");
  }
  trace_case(test_case, "reference-enqueue-begin");
  enqueue(*reference, reference_state, stream);
  mv::timing::paired(
      test_case.name, stream,
      [&] { enqueue(*production, production_state, stream); },
      [&] { enqueue(*reference, reference_state, stream); });
  trace_case(test_case, "reference-enqueue-end");
  if (trace_enabled()) {
    trace_case(test_case, "reference-synchronize-begin");
    stream.synchronize();
    trace_case(test_case, "reference-synchronize-end");
  }

  trace_case(test_case, "readback-begin");
  std::vector<std::vector<std::uint8_t>> production_bytes;
  std::vector<std::vector<std::uint8_t>> reference_bytes;
  production_bytes.reserve(3);
  reference_bytes.reserve(3);
  for (std::size_t index = 0; index < 3; ++index) {
    production_bytes.push_back(
        read_back(production_state, index, stream));
    reference_bytes.push_back(
        read_back(reference_state, index, stream));
  }
  stream.synchronize();
  for (std::size_t index = 0; index < 3; ++index) {
    if (index == production_state.output) {
      continue;
    }
    io::require_bytes_equal(
        "FlagDNN Convolution input " + std::to_string(index),
        production_bytes[index],
        production_state.storage[index].bytes);
    io::require_bytes_equal(
        "muDNN Convolution input " + std::to_string(index),
        reference_bytes[index],
        reference_state.storage[index].bytes);
  }
  trace_case(test_case, "compare-begin");
  return compare(
      read_output(production_state, stream, "FlagDNN Convolution"),
      read_output(reference_state, stream, "muDNN Convolution"),
      test_case);
}

}  // namespace

int run_convolution_functional_test(
    int argc,
    char** argv,
    std::span<const ConvolutionTestCase> cases,
    ConvolutionDirection expected_direction) {
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

    const char* filter = std::getenv("FLAGDNN_CONVOLUTION_CASE");
    std::size_t executed = 0;
    std::size_t skipped = 0;
    for (const ConvolutionTestCase& test_case : cases) {
      if (test_case.direction != expected_direction) {
        throw std::invalid_argument(
            "MThreads Convolution suite contains the wrong direction");
      }
      if (filter != nullptr && filter[0] != '\0' &&
          test_case.name.find(filter) == std::string::npos) {
        continue;
      }
      try {
        const Accuracy accuracy = run_case(test_case, handle, stream);
        ++executed;
        std::cout << test_case.name
                  << ": FlagDNN Graph vs direct muDNN Convolution "
                  << operation_name(test_case.direction)
                  << " PASS max_abs=" << accuracy.maximum_absolute
                  << " max_rel=" << accuracy.maximum_relative << std::endl;
      } catch (const mv::ReferenceUnsupported& error) {
        ++skipped;
        mv::report_skip(test_case.name, error);
      }
    }
    if (executed + skipped == 0) {
      throw std::runtime_error(
          "FLAGDNN_CONVOLUTION_CASE matched no MThreads cases");
    }
    return mv::report_cases("FLAGDNN_CONVOLUTION_FUNCTIONAL", executed,
                            skipped);
  } catch (const std::exception& error) {
    std::cerr << "FLAGDNN_CONVOLUTION_FUNCTIONAL_FAILED: "
              << error.what() << '\n';
    return 1;
  }
}

}  // namespace flagdnn::testing
