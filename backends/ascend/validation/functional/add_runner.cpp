/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/add.hpp"
#include "validation/acl_runtime.hpp"
#include "validation/development_environment.hpp"
#include "validation/exact_reference.hpp"
#include "validation/functional/aclnn_add.hpp"
#include "validation/tensor_io.hpp"

#include <flagdnn/flagdnn.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <iterator>
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
  const std::vector<float> generated = make_input(tensor, input_index);
  const std::vector<float> logical =
      tensor_io::quantize(generated, tensor.data_type);
  const std::vector<float> physical = tensor_io::scatter(logical, tensor);
  const std::vector<std::uint8_t> encoded =
      tensor_io::encode(physical, tensor.data_type);
  auto device =
      std::make_unique<acl::DeviceBuffer>(tensor_io::allocation_byte_count(tensor));
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
  auto result =
      std::make_unique<acl::DeviceBuffer>(tensor_io::allocation_byte_count(tensor));
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

class ExecutableLifetime {
 public:
  ExecutableLifetime(acl::Stream& stream,
                     std::unique_ptr<AddExecutable>& flagdnn,
                     std::unique_ptr<AddExecutable>& reference)
      : stream_(stream), flagdnn_(flagdnn), reference_(reference) {}

  ~ExecutableLifetime() {
    try {
      stream_.synchronize();
    } catch (const std::exception& error) {
      std::cerr << "Ascend executable cleanup synchronization failed: "
                << error.what() << '\n';
    }
    reference_.reset();
    flagdnn_.reset();
  }

  ExecutableLifetime(const ExecutableLifetime&) = delete;
  ExecutableLifetime& operator=(const ExecutableLifetime&) = delete;

 private:
  acl::Stream& stream_;
  std::unique_ptr<AddExecutable>& flagdnn_;
  std::unique_ptr<AddExecutable>& reference_;
};

std::string single_line_reason(std::string_view reason) {
  std::string result(reason);
  std::replace(result.begin(), result.end(), '\r', ' ');
  std::replace(result.begin(), result.end(), '\n', ' ');
  return result;
}

void execute(AddExecutable& executable,
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

void synchronize_default_stream() {
  aclrtStream stream = nullptr;
  acl::check_acl(aclrtCtxGetCurrentDefaultStream(&stream),
                 "aclrtCtxGetCurrentDefaultStream");
  if (stream == nullptr) {
    throw std::runtime_error(
        "aclrtCtxGetCurrentDefaultStream returned a null stream");
  }
  acl::check_acl(aclrtSynchronizeStream(stream),
                 "aclrtSynchronizeStream(default)");
}

std::optional<std::string> run_case(const AddTestCase& test_case,
                                    flagdnn::Handle& handle,
                                    acl::Stream& stream) {
  validate_add_case(test_case);
  std::unique_ptr<AddExecutable> flagdnn =
      build_flagdnn_add(handle, test_case);
  std::unique_ptr<AddExecutable> reference = build_add_reference(test_case);
  const AclnnAddPlan reference_plan = plan_aclnn_add(test_case);

  InputBuffer flagdnn_left = make_input_buffer(test_case.left, 0, stream);
  InputBuffer flagdnn_right = make_input_buffer(test_case.right, 1, stream);
  InputBuffer reference_left = make_input_buffer(reference_plan.left, 0, stream);
  InputBuffer reference_right =
      make_input_buffer(reference_plan.right, 1, stream);
  std::unique_ptr<acl::DeviceBuffer> flagdnn_output =
      make_output_buffer(test_case.output, stream);
  std::unique_ptr<acl::DeviceBuffer> reference_output =
      make_output_buffer(reference_plan.output, stream);

  const std::vector<flagdnnBinding_t> flagdnn_bindings = {
      {test_case.left.uid,
       flagdnn_left.device->opaque_at(test_case.left.binding_byte_offset)},
      {test_case.right.uid,
       flagdnn_right.device->opaque_at(test_case.right.binding_byte_offset)},
      {test_case.output.uid,
       flagdnn_output->opaque_at(test_case.output.binding_byte_offset)},
  };
  const std::vector<flagdnnBinding_t> reference_bindings = {
      {reference_plan.left.uid,
       reference_left.device->opaque_at(
           reference_plan.left.binding_byte_offset)},
      {reference_plan.right.uid,
       reference_right.device->opaque_at(
           reference_plan.right.binding_byte_offset)},
      {reference_plan.output.uid,
       reference_output->opaque_at(reference_plan.output.binding_byte_offset)},
  };
  ExecutableLifetime executable_lifetime(stream, flagdnn, reference);

  flagdnn->prepare(flagdnn_bindings, stream.opaque());
  try {
    reference->prepare(reference_bindings, stream.opaque());
  } catch (const AclnnUnsupportedError& error) {
    try {
      stream.synchronize();
    } catch (...) {
    }
    return "ACLNN status=" + std::to_string(error.status()) + " " +
           single_line_reason(error.what());
  }

  auto flagdnn_workspace =
      std::make_unique<acl::DeviceBuffer>(flagdnn->workspace_size());
  auto reference_workspace =
      std::make_unique<acl::DeviceBuffer>(reference->workspace_size());

  try {
    stream.synchronize();
    if (test_case.name == "add_default_stream_fp32") {
      flagdnn->execute(flagdnn_bindings,
                       flagdnn->workspace_size() == 0
                           ? nullptr
                           : flagdnn_workspace->opaque(),
                       flagdnn->workspace_size(),
                       nullptr);
      synchronize_default_stream();
    } else {
      execute(*flagdnn, flagdnn_bindings, *flagdnn_workspace, stream);
    }
    execute(*reference, reference_bindings, *reference_workspace, stream);
    stream.synchronize();

    const std::vector<float> flagdnn_physical =
        read_output(*flagdnn_output, test_case.output, stream);
    tensor_io::require_padding_unchanged(
        "FlagDNN", flagdnn_physical, test_case.output);
    const std::vector<float> flagdnn_logical =
        tensor_io::gather(flagdnn_physical, test_case.output);

    const std::vector<float> reference_physical =
        read_output(*reference_output, reference_plan.output, stream);
    tensor_io::require_padding_unchanged(
        "ACLNN", reference_physical, reference_plan.output);
    const std::vector<float> reference_logical =
        tensor_io::gather(reference_physical, reference_plan.output);
    acl::compare_exact_reference(flagdnn_logical,
                                 reference_logical,
                                 test_case.absolute_tolerance,
                                 test_case.relative_tolerance,
                                 test_case.name);

    std::cout << test_case.name
              << ": FlagDNN Graph/libtriton_jit vs ACLNN Add PASS "
                 "exact_reference=ACLNN_ADD\n";
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

void run_unaligned_offset_rejection(const AddTestCase& prototype,
                                    acl::Stream& stream) {
  AddTestCase test_case = prototype;
  test_case.name = "add_unaligned_offset_negative_fp32";
  test_case.left.binding_byte_offset = 1;
  test_case.right.binding_byte_offset = 1;
  test_case.output.binding_byte_offset = 1;
  const AclnnAddPlan plan = plan_aclnn_add(test_case);
  auto executable = build_add_reference(test_case);
  acl::DeviceBuffer left(tensor_io::allocation_byte_count(plan.left));
  acl::DeviceBuffer right(tensor_io::allocation_byte_count(plan.right));
  acl::DeviceBuffer output(tensor_io::allocation_byte_count(plan.output));
  const std::vector<flagdnnBinding_t> bindings = {
      {plan.left.uid, left.opaque_at(plan.left.binding_byte_offset)},
      {plan.right.uid, right.opaque_at(plan.right.binding_byte_offset)},
      {plan.output.uid, output.opaque_at(plan.output.binding_byte_offset)},
  };
  try {
    executable->prepare(bindings, stream.opaque());
  } catch (const std::invalid_argument& error) {
    if (std::string_view(error.what()).find("32-byte aligned") ==
        std::string_view::npos) {
      throw;
    }
    executable.reset();
    std::cout << test_case.name << ": ACLNN_UNALIGNED_OFFSET_REJECTED PASS\n";
    return;
  }
  executable.reset();
  throw std::runtime_error(
      "ACLNN Add accepted a binding that is not 32-byte aligned");
}

AddTestCase make_ascend_extension(
    std::string name,
    TestTensor left,
    TestTensor right,
    TestTensor output,
    double alpha = 1.0) {
  AddTestCase result;
  result.name = std::move(name);
  result.left = std::move(left);
  result.right = std::move(right);
  result.output = std::move(output);
  result.alpha = alpha;
  result.absolute_tolerance =
      result.output.data_type == FLAGDNN_DATA_FLOAT32
          ? 1.0e-6
          : (result.output.data_type == FLAGDNN_DATA_BFLOAT16 ? 5.0e-2
                                                              : 2.0e-2);
  result.relative_tolerance =
      result.output.data_type == FLAGDNN_DATA_FLOAT32 ? 1.0e-6 : 1.0e-2;
  validate_add_case(result);
  return result;
}

std::vector<AddTestCase> make_ascend_semantic_extensions() {
  std::vector<AddTestCase> result;
  result.push_back(make_ascend_extension(
      "add_contiguous_basic_bf16",
      TestTensor{170, FLAGDNN_DATA_BFLOAT16, {2, 3, 5}, {15, 5, 1}, 0},
      TestTensor{171, FLAGDNN_DATA_BFLOAT16, {2, 3, 5}, {15, 5, 1}, 0},
      TestTensor{172, FLAGDNN_DATA_BFLOAT16, {2, 3, 5}, {15, 5, 1}, 0}));
  result.push_back(make_ascend_extension(
      "add_single_axis_broadcast_fp16",
      TestTensor{180, FLAGDNN_DATA_FLOAT16, {2, 3, 5}, {15, 5, 1}, 0},
      TestTensor{181, FLAGDNN_DATA_FLOAT16, {1, 3, 1}, {3, 1, 1}, 0},
      TestTensor{182, FLAGDNN_DATA_FLOAT16, {2, 3, 5}, {15, 5, 1}, 0}));
  result.push_back(make_ascend_extension(
      "add_multi_axis_broadcast_bf16",
      TestTensor{190, FLAGDNN_DATA_BFLOAT16, {2, 1, 5}, {5, 5, 1}, 0},
      TestTensor{191, FLAGDNN_DATA_BFLOAT16, {1, 3, 1}, {3, 1, 1}, 0},
      TestTensor{192, FLAGDNN_DATA_BFLOAT16, {2, 3, 5}, {15, 5, 1}, 0}));
  result.push_back(make_ascend_extension(
      "add_rank_mismatched_broadcast_fp32",
      TestTensor{200, FLAGDNN_DATA_FLOAT32, {2, 3, 5}, {15, 5, 1}, 0},
      TestTensor{201, FLAGDNN_DATA_FLOAT32, {5}, {1}, 0},
      TestTensor{202, FLAGDNN_DATA_FLOAT32, {2, 3, 5}, {15, 5, 1}, 0}));
  result.push_back(make_ascend_extension(
      "add_non_contiguous_input_fp32",
      TestTensor{210, FLAGDNN_DATA_FLOAT32, {2, 3, 5}, {48, 12, 2}, 0},
      TestTensor{211, FLAGDNN_DATA_FLOAT32, {2, 3, 5}, {15, 5, 1}, 0},
      TestTensor{212, FLAGDNN_DATA_FLOAT32, {2, 3, 5}, {15, 5, 1}, 0}));
  result.push_back(make_ascend_extension(
      "add_non_contiguous_output_padding_fp32",
      TestTensor{220, FLAGDNN_DATA_FLOAT32, {2, 3, 5}, {15, 5, 1}, 0},
      TestTensor{221, FLAGDNN_DATA_FLOAT32, {2, 3, 5}, {15, 5, 1}, 0},
      TestTensor{222, FLAGDNN_DATA_FLOAT32, {2, 3, 5}, {40, 10, 2}, 0}));
  return result;
}

AddTestCase make_aligned_offset_extension() {
  AddTestCase result;
  result.name = "add_aligned_32_byte_offset_fp32";
  result.left = TestTensor{230,
                           FLAGDNN_DATA_FLOAT32,
                           {2, 3, 5},
                           {15, 5, 1},
                           32};
  result.right = TestTensor{231,
                            FLAGDNN_DATA_FLOAT32,
                            {2, 3, 5},
                            {15, 5, 1},
                            32};
  result.output = TestTensor{232,
                             FLAGDNN_DATA_FLOAT32,
                             {2, 3, 5},
                             {15, 5, 1},
                             32};
  result.alpha = 1.0;
  result.absolute_tolerance = 1.0e-6;
  result.relative_tolerance = 1.0e-6;
  validate_add_case(result);
  return result;
}

AddTestCase make_large_strided_persistent_extension() {
  AddTestCase result;
  result.name = "add_large_strided_persistent_fp32";
  result.left = TestTensor{240,
                           FLAGDNN_DATA_FLOAT32,
                           {2, 64, 129},
                           {16640, 260, 2},
                           0};
  result.right = TestTensor{241,
                            FLAGDNN_DATA_FLOAT32,
                            {1, 64, 1},
                            {64, 1, 1},
                            0};
  result.output = TestTensor{242,
                             FLAGDNN_DATA_FLOAT32,
                             {2, 64, 129},
                             {8256, 129, 1},
                             0};
  result.alpha = -0.75;
  result.absolute_tolerance = 1.0e-6;
  result.relative_tolerance = 1.0e-6;
  validate_add_case(result);
  return result;
}

AddTestCase make_large_dense_persistent_extension() {
  AddTestCase result;
  result.name = "add_large_dense_persistent_fp32";
  result.left = TestTensor{235,
                           FLAGDNN_DATA_FLOAT32,
                           {4, 16, 64, 128},
                           {131072, 8192, 128, 1},
                           0};
  result.right = TestTensor{236,
                            FLAGDNN_DATA_FLOAT32,
                            {4, 16, 64, 128},
                            {131072, 8192, 128, 1},
                            0};
  result.output = TestTensor{237,
                             FLAGDNN_DATA_FLOAT32,
                             {4, 16, 64, 128},
                             {131072, 8192, 128, 1},
                             0};
  result.alpha = 1.0;
  result.absolute_tolerance = 1.0e-6;
  result.relative_tolerance = 1.0e-6;
  validate_add_case(result);
  return result;
}

AddTestCase make_default_stream_extension() {
  AddTestCase result;
  result.name = "add_default_stream_fp32";
  result.left = TestTensor{250,
                           FLAGDNN_DATA_FLOAT32,
                           {2, 4, 8},
                           {32, 8, 1},
                           0};
  result.right = TestTensor{251,
                            FLAGDNN_DATA_FLOAT32,
                            {2, 4, 8},
                            {32, 8, 1},
                            0};
  result.output = TestTensor{252,
                             FLAGDNN_DATA_FLOAT32,
                             {2, 4, 8},
                             {32, 8, 1},
                             0};
  result.alpha = 1.0;
  result.absolute_tolerance = 1.0e-6;
  result.relative_tolerance = 1.0e-6;
  validate_add_case(result);
  return result;
}

}  // namespace

int run_add_functional_test(int argc,
                            char** argv,
                            std::span<const AddTestCase> cases) {
  // Exact integer and raw storage cases run in functional.ascend.<op>.dtype,
  // with an independent process and the same public case catalog.
  std::vector<AddTestCase> supported_cases(cases.begin(), cases.end());
  std::erase_if(supported_cases, [](const AddTestCase& test_case) {
    const auto type = test_case.left.data_type;
    return type != FLAGDNN_DATA_FLOAT32 && type != FLAGDNN_DATA_FLOAT16 &&
           type != FLAGDNN_DATA_BFLOAT16;
  });
  cases = supported_cases;

  if (argc != 3) {
    std::cerr << "usage: " << argv[0]
              << " COMPILER_EXECUTABLE COMPILER_ENTRY\n";
    return 2;
  }
  try {
    std::cout << std::setprecision(9);
    if (cases.empty()) {
      throw std::logic_error(
          "Ascend Add functional common catalog must contain cases");
    }
    std::vector<AddTestCase> ascend_cases(cases.begin(), cases.end());
    std::vector<AddTestCase> semantic_extensions =
        make_ascend_semantic_extensions();
    ascend_cases.insert(ascend_cases.end(),
                        std::make_move_iterator(semantic_extensions.begin()),
                        std::make_move_iterator(semantic_extensions.end()));
    ascend_cases.push_back(make_aligned_offset_extension());
    ascend_cases.push_back(make_large_dense_persistent_extension());
    ascend_cases.push_back(make_large_strided_persistent_extension());
    ascend_cases.push_back(make_default_stream_extension());
    const char* filter = std::getenv("FLAGDNN_ADD_CASE");
    const bool filtered = filter != nullptr && filter[0] != '\0';
    std::vector<const AddTestCase*> selected;
    for (const AddTestCase& test_case : ascend_cases) {
      if (filtered && test_case.name.find(filter) == std::string::npos) {
        continue;
      }
      selected.push_back(&test_case);
    }
    if (selected.empty()) {
      throw std::runtime_error("FLAGDNN_ADD_CASE matched no test cases");
    }
    acl::ExactReferenceCoverage coverage(selected.size());

    acl::DevelopmentEnvironment development("add-functional");
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

        for (const AddTestCase* test_case : selected) {
          const std::optional<std::string> skip =
              run_case(*test_case, handle, stream);
          if (skip.has_value()) {
            coverage.record_skip(test_case->name, *skip);
          } else {
            coverage.record_pass(test_case->name);
          }
        }
        if (!filtered) {
          const auto prototype = std::find_if(
              ascend_cases.begin(),
              ascend_cases.end(),
              [](const AddTestCase& test_case) {
                return test_case.left.data_type == FLAGDNN_DATA_FLOAT32;
              });
          if (prototype == ascend_cases.end()) {
            throw std::runtime_error(
                "no FP32 Add case exists for the unaligned-offset test");
          }
          run_unaligned_offset_rejection(*prototype, stream);
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
    std::cout << "FLAGDNN_ADD_FUNCTIONAL: PASS cases="
              << coverage.passed()
              << " catalog_cases=" << ascend_cases.size() << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FLAGDNN_ADD_FUNCTIONAL_FAILED: " << error.what() << '\n';
    return 1;
  }
}

}  // namespace flagdnn::testing
