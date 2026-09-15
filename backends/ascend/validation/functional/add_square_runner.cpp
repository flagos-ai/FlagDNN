/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/composite.hpp"
#include "validation/acl_runtime.hpp"
#include "validation/development_environment.hpp"
#include "validation/exact_reference.hpp"
#include "validation/functional/aclnn_add_square.hpp"
#include "validation/tensor_io.hpp"

#include <flagdnn/flagdnn.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace flagdnn::testing {
namespace {

namespace acl = flagdnn::validation::ascend;
namespace tensor_io = flagdnn::validation::ascend::tensor_io;

std::vector<float> make_input(const TestTensor& tensor,
                              std::size_t input_index) {
  std::vector<float> result(tensor_io::element_count(tensor));
  for (std::size_t index = 0; index < result.size(); ++index) {
    const int centered =
        static_cast<int>((index * 17 + input_index * 11) % 41) - 20;
    result[index] =
        static_cast<float>(centered) / static_cast<float>(13 + input_index);
  }
  return result;
}

struct InputBuffer {
  std::unique_ptr<acl::DeviceBuffer> device;
};

InputBuffer make_input_buffer(const TestTensor& tensor,
                              std::size_t input_index,
                              acl::Stream& stream) {
  const std::vector<float> logical =
      tensor_io::quantize(make_input(tensor, input_index), tensor.data_type);
  const std::vector<float> physical = tensor_io::scatter(logical, tensor);
  const std::vector<std::uint8_t> encoded =
      tensor_io::encode(physical, tensor.data_type);
  auto device = std::make_unique<acl::DeviceBuffer>(
      tensor_io::allocation_byte_count(tensor));
  device->copy_from_host_at(encoded.data(),
                            encoded.size(),
                            tensor.binding_byte_offset,
                            stream.get());
  return {std::move(device)};
}

std::unique_ptr<acl::DeviceBuffer> make_output_buffer(
    const TestTensor& tensor,
    acl::Stream& stream) {
  const std::vector<float> initial(tensor_io::storage_element_count(tensor),
                                   tensor_io::kPaddingSentinel);
  const std::vector<std::uint8_t> encoded =
      tensor_io::encode(initial, tensor.data_type);
  auto result = std::make_unique<acl::DeviceBuffer>(
      tensor_io::allocation_byte_count(tensor));
  result->copy_from_host_at(encoded.data(),
                            encoded.size(),
                            tensor.binding_byte_offset,
                            stream.get());
  return result;
}

std::vector<float> read_output(const acl::DeviceBuffer& buffer,
                               const TestTensor& tensor,
                               acl::Stream& stream) {
  const std::size_t storage_count = tensor_io::storage_element_count(tensor);
  std::vector<std::uint8_t> encoded(tensor_io::encoded_byte_count(tensor));
  buffer.copy_to_host_at(encoded.data(),
                         encoded.size(),
                         tensor.binding_byte_offset,
                         stream.get());
  stream.synchronize();
  return tensor_io::decode(encoded, tensor.data_type, storage_count);
}

std::string single_line_reason(std::string_view reason) {
  std::string result(reason);
  std::replace(result.begin(), result.end(), '\r', ' ');
  std::replace(result.begin(), result.end(), '\n', ' ');
  return result;
}

void execute(CompositeExecutable& executable,
             std::span<const flagdnnBinding_t> bindings,
             acl::DeviceBuffer& workspace,
             acl::Stream& stream) {
  executable.execute(bindings,
                     executable.workspace_size() == 0
                         ? nullptr
                         : workspace.opaque(),
                     executable.workspace_size(),
                     stream.opaque());
}

std::optional<std::string> run_case(const AddSquareTestCase& test_case,
                                    flagdnn::Handle& handle,
                                    acl::Stream& stream) {
  validate_composite_case(test_case);
  const AclnnAddSquarePlan reference_plan =
      plan_aclnn_add_square(test_case);
  std::unique_ptr<CompositeExecutable> flagdnn =
      build_flagdnn_add_square(handle, test_case);
  std::unique_ptr<CompositeExecutable> reference =
      build_add_square_reference(test_case);
  if (flagdnn->workspace_size() == 0) {
    throw std::runtime_error(
        test_case.name +
        ": FlagDNN AddSquare did not allocate virtual Graph workspace");
  }

  InputBuffer left = make_input_buffer(test_case.left, 0, stream);
  InputBuffer right = make_input_buffer(test_case.right, 1, stream);
  std::unique_ptr<acl::DeviceBuffer> flagdnn_output =
      make_output_buffer(test_case.output, stream);
  std::unique_ptr<acl::DeviceBuffer> reference_output =
      make_output_buffer(reference_plan.output, stream);
  const std::vector<flagdnnBinding_t> flagdnn_bindings = {
      {test_case.left.uid,
       left.device->opaque_at(test_case.left.binding_byte_offset)},
      {test_case.right.uid,
       right.device->opaque_at(test_case.right.binding_byte_offset)},
      {test_case.output.uid,
       flagdnn_output->opaque_at(test_case.output.binding_byte_offset)},
  };
  const std::vector<flagdnnBinding_t> reference_bindings = {
      {reference_plan.left.uid,
       left.device->opaque_at(reference_plan.left.binding_byte_offset)},
      {reference_plan.right.uid,
       right.device->opaque_at(reference_plan.right.binding_byte_offset)},
      {reference_plan.output.uid,
       reference_output->opaque_at(reference_plan.output.binding_byte_offset)},
  };

  flagdnn->prepare(flagdnn_bindings, stream.opaque());
  try {
    reference->prepare(reference_bindings, stream.opaque());
  } catch (const AclnnAddSquareUnsupportedError& error) {
    try {
      stream.synchronize();
    } catch (...) {
    }
    reference.reset();
    flagdnn.reset();
    return "ACLNN status=" + std::to_string(error.status()) + " " +
           single_line_reason(error.what());
  }
  auto flagdnn_workspace =
      std::make_unique<acl::DeviceBuffer>(flagdnn->workspace_size());
  auto reference_workspace =
      std::make_unique<acl::DeviceBuffer>(reference->workspace_size());

  try {
    stream.synchronize();
    execute(*flagdnn, flagdnn_bindings, *flagdnn_workspace, stream);
    execute(*reference, reference_bindings, *reference_workspace, stream);
    stream.synchronize();

    const std::vector<float> flagdnn_physical =
        read_output(*flagdnn_output, test_case.output, stream);
    const std::vector<float> reference_physical =
        read_output(*reference_output, reference_plan.output, stream);
    tensor_io::require_padding_unchanged(
        "FlagDNN", flagdnn_physical, test_case.output);
    tensor_io::require_padding_unchanged(
        "ACLNN", reference_physical, reference_plan.output);
    const std::vector<float> flagdnn_logical =
        tensor_io::gather(flagdnn_physical, test_case.output);
    const std::vector<float> reference_logical =
        tensor_io::gather(reference_physical, reference_plan.output);
    acl::compare_exact_reference(flagdnn_logical,
                                 reference_logical,
                                 test_case.absolute_tolerance,
                                 test_case.relative_tolerance,
                                 test_case.name);
    std::cout << test_case.name
              << ": FlagDNN Graph Mul->Add/libtriton_jit vs ACLNN Mul->Add PASS"
              << " virtual_workspace_bytes=" << flagdnn->workspace_size()
              << " autotune=" << (test_case.autotune ? "true" : "false")
              << " exact_reference=ACLNN_MUL_ADD\n";
  } catch (...) {
    try {
      stream.synchronize();
    } catch (...) {
    }
    reference.reset();
    flagdnn.reset();
    throw;
  }
  stream.synchronize();
  reference.reset();
  flagdnn.reset();
  return std::nullopt;
}

}  // namespace

int run_add_square_functional_test(
    int argc,
    char** argv,
    std::span<const AddSquareTestCase> cases) {
  // This backend's reference adapter currently accepts floating storage.
  std::vector<AddSquareTestCase> supported_cases(cases.begin(), cases.end());
  std::erase_if(supported_cases, [](const AddSquareTestCase& test_case) {
    const auto type = test_case.left.data_type;
    return type != FLAGDNN_DATA_FLOAT32 && type != FLAGDNN_DATA_FLOAT16 &&
           type != FLAGDNN_DATA_BFLOAT16;
  });
  cases = supported_cases;

  constexpr std::string_view kSuite = "FLAGDNN_ADD_SQUARE_FUNCTIONAL";
  if (argc != 3) {
    std::cerr << "usage: " << argv[0]
              << " COMPILER_EXECUTABLE COMPILER_ENTRY\n";
    return 2;
  }
  try {
    std::cout << std::setprecision(9);
    if (cases.empty()) {
      throw std::logic_error(
          "Ascend AddSquare functional catalog must contain cases");
    }
    const char* filter = std::getenv("FLAGDNN_ASCEND_COMPOSITE_CASE");
    std::vector<const AddSquareTestCase*> selected;
    for (const AddSquareTestCase& test_case : cases) {
      if (filter != nullptr && filter[0] != '\0' &&
          test_case.name.find(filter) == std::string::npos) {
        continue;
      }
      selected.push_back(&test_case);
    }
    if (selected.empty()) {
      throw std::runtime_error(
          "FLAGDNN_ASCEND_COMPOSITE_CASE matched no AddSquare cases");
    }
    acl::ExactReferenceCoverage coverage(selected.size());
    acl::DevelopmentEnvironment development("add-square-functional");
    acl::AclRuntime runtime;
    const std::string soc = acl::soc_name();
    development.prepare_target(soc);
    std::cout << "ASCEND_VALIDATION_DEVELOPMENT_ROOT root="
              << development.root() << '\n';
    std::cout << "ACLNN_CAPABILITY_IDENTITY soc=" << soc
              << " cann_runtime=" << acl::runtime_package_version() << '\n';
    {
      acl::Stream stream;
      {
        flagdnn::Handle handle("ascend", 0);
        handle.set_compiler(
            argv[1], argv[2], development.graph_cache().string());
        for (const AddSquareTestCase* test_case : selected) {
          const std::optional<std::string> skip =
              run_case(*test_case, handle, stream);
          if (skip.has_value()) {
            coverage.record_skip(test_case->name, *skip);
          } else {
            coverage.record_pass(test_case->name);
          }
        }
        stream.synchronize();
      }
    }
    runtime.finalize();
    development.cleanup();
    coverage.require_complete();
    for (const std::string& line : coverage.skip_lines()) {
      std::cout << line << '\n';
    }
    std::cout << coverage.summary() << '\n';
    std::cout << kSuite << ": PASS cases=" << coverage.passed()
              << " catalog_cases=" << cases.size() << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << kSuite << "_FAILED: " << error.what() << '\n';
    return 1;
  }
}

}  // namespace flagdnn::testing
