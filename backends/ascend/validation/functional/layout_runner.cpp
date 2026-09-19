/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/layout.hpp"

#include "validation/acl_runtime.hpp"
#include "validation/development_environment.hpp"
#include "validation/exact_reference.hpp"
#include "validation/functional/aclnn_layout.hpp"
#include "validation/tensor_io.hpp"

#include <flagdnn/flagdnn.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
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

constexpr std::size_t kTailGuardBytes = 32;
constexpr std::uint8_t kAllocationGuard = 0xD3U;

std::string data_type_name(flagdnnDataType_t data_type) {
  switch (data_type) {
    case FLAGDNN_DATA_INT32:
      throw std::invalid_argument(
          "INT32 is not supported by this validation adapter");

    case FLAGDNN_DATA_FLOAT32:
      return "fp32";
    case FLAGDNN_DATA_FLOAT16:
      return "fp16";
    case FLAGDNN_DATA_BFLOAT16:
      return "bfloat16";
    case FLAGDNN_DATA_BOOLEAN:
    case FLAGDNN_DATA_FP8_E8M0:
    case FLAGDNN_DATA_FP8_E4M3:
    case FLAGDNN_DATA_FP8_E5M2:
      break;
  }
  throw std::invalid_argument("Ascend layout data type is unsupported");
}

std::string operation_name(LayoutOperation operation) {
  switch (operation) {
    case LayoutOperation::kReshape:
      return "reshape";
    case LayoutOperation::kTranspose:
      return "transpose";
    case LayoutOperation::kSlice:
      return "slice";
  }
  throw std::invalid_argument("Ascend layout operation is invalid");
}

LayoutTestCase make_special_case(LayoutOperation operation,
                                 flagdnnDataType_t data_type,
                                 std::int64_t uid) {
  LayoutTestCase result;
  result.name = operation_name(operation) +
                "_ascend_gapped_offsets_padding_" +
                data_type_name(data_type);
  result.operation = operation;
  switch (operation) {
    case LayoutOperation::kReshape:
      result.input = {uid, data_type, {2, 3, 4}, {47, 13, 2}, 32};
      result.output = {uid + 1, data_type, {6, 4}, {11, 2}, 64};
      break;
    case LayoutOperation::kTranspose:
      result.input = {uid, data_type, {2, 3, 4}, {47, 13, 2}, 32};
      result.output =
          {uid + 1, data_type, {4, 2, 3}, {29, 11, 3}, 64};
      result.permutation = {2, 0, 1};
      break;
    case LayoutOperation::kSlice:
      result.input = {uid, data_type, {3, 5, 7}, {101, 17, 2}, 32};
      result.output =
          {uid + 1, data_type, {2, 3, 3}, {43, 11, 3}, 64};
      result.slices = {{1, 3}, {0, 5}, {1, 7}};
      result.slice_strides = {1, 2, 2};
      break;
  }
  validate_layout_case(result);
  (void)tensor_io::encoded_byte_count(result.input);
  (void)tensor_io::encoded_byte_count(result.output);
  return result;
}

std::vector<float> make_input(const TestTensor& tensor) {
  constexpr std::array<float, 16> kValues = {
      -0.0F,
      0.0F,
      std::numeric_limits<float>::infinity(),
      -std::numeric_limits<float>::infinity(),
      std::bit_cast<float>(std::uint32_t{0x7FC12345U}),
      std::bit_cast<float>(std::uint32_t{0xFFC54321U}),
      1.0F,
      -1.0F,
      0.5F,
      -2.25F,
      3.5F,
      -4.0F,
      5.25F,
      -6.5F,
      7.0F,
      -8.75F};
  std::vector<float> result(tensor_io::element_count(tensor));
  for (std::size_t index = 0; index < result.size(); ++index) {
    result[index] = kValues[index % kValues.size()];
  }
  return tensor_io::quantize(result, tensor.data_type);
}

struct GuardedBuffer {
  std::unique_ptr<acl::DeviceBuffer> device;
  std::vector<std::uint8_t> initial;
};

GuardedBuffer make_buffer(const TestTensor& tensor,
                          std::span<const float> physical,
                          acl::Stream& stream) {
  const std::vector<std::uint8_t> encoded =
      tensor_io::encode(physical, tensor.data_type);
  if (encoded.size() != tensor_io::encoded_byte_count(tensor)) {
    throw std::logic_error("layout encoded tensor byte count differs");
  }
  const std::size_t allocation = tensor_io::checked_add(
      tensor_io::allocation_byte_count(tensor),
      kTailGuardBytes,
      "layout guarded allocation");
  std::vector<std::uint8_t> initial(allocation, kAllocationGuard);
  std::copy(encoded.begin(),
            encoded.end(),
            initial.begin() +
                static_cast<std::ptrdiff_t>(tensor.binding_byte_offset));
  auto device = std::make_unique<acl::DeviceBuffer>(allocation);
  device->copy_from_host_at(initial.data(), initial.size(), 0, stream.get());
  return {std::move(device), std::move(initial)};
}

GuardedBuffer make_input_buffer(const TestTensor& tensor,
                                std::span<const float> logical,
                                acl::Stream& stream) {
  return make_buffer(tensor, tensor_io::scatter(logical, tensor), stream);
}

GuardedBuffer make_output_buffer(const TestTensor& tensor,
                                 acl::Stream& stream) {
  const std::vector<float> physical(tensor_io::storage_element_count(tensor),
                                    tensor_io::kPaddingSentinel);
  return make_buffer(tensor, physical, stream);
}

std::vector<std::uint8_t> read_all(const GuardedBuffer& buffer,
                                   acl::Stream& stream) {
  std::vector<std::uint8_t> result(buffer.initial.size());
  buffer.device->copy_to_host_at(result.data(), result.size(), 0, stream.get());
  stream.synchronize();
  return result;
}

void restore(GuardedBuffer& buffer, acl::Stream& stream) {
  buffer.device->copy_from_host_at(
      buffer.initial.data(), buffer.initial.size(), 0, stream.get());
}

void require_guards_unchanged(std::string_view provider,
                              std::span<const std::uint8_t> bytes,
                              const GuardedBuffer& buffer,
                              const TestTensor& tensor) {
  const std::size_t payload_begin = tensor.binding_byte_offset;
  const std::size_t payload_end = tensor_io::checked_add(
      payload_begin,
      tensor_io::encoded_byte_count(tensor),
      "layout guarded payload");
  if (bytes.size() != buffer.initial.size() || payload_end > bytes.size()) {
    throw std::invalid_argument("layout guarded buffer size is invalid");
  }
  if (!std::equal(bytes.begin(),
                  bytes.begin() + static_cast<std::ptrdiff_t>(payload_begin),
                  buffer.initial.begin()) ||
      !std::equal(bytes.begin() + static_cast<std::ptrdiff_t>(payload_end),
                  bytes.end(),
                  buffer.initial.begin() +
                      static_cast<std::ptrdiff_t>(payload_end))) {
    throw std::runtime_error(std::string(provider) +
                             " modified a binding offset/tail guard");
  }
}

struct OutputData {
  std::vector<std::uint8_t> payload;
  std::vector<float> logical;
};

OutputData decode_output(std::string_view provider,
                         const GuardedBuffer& buffer,
                         const TestTensor& tensor,
                         acl::Stream& stream) {
  const std::vector<std::uint8_t> all = read_all(buffer, stream);
  require_guards_unchanged(provider, all, buffer, tensor);
  const std::span<const std::uint8_t> encoded(all);
  const std::span<const std::uint8_t> payload = encoded.subspan(
      tensor.binding_byte_offset, tensor_io::encoded_byte_count(tensor));
  const std::vector<float> physical =
      tensor_io::decode_storage(provider, payload, tensor);
  tensor_io::require_padding_unchanged(provider, physical, tensor);
  return {
      std::vector<std::uint8_t>(payload.begin(), payload.end()),
      tensor_io::gather(physical, tensor),
  };
}

void require_input_unchanged(std::string_view provider,
                             const GuardedBuffer& buffer,
                             acl::Stream& stream) {
  const std::vector<std::uint8_t> actual = read_all(buffer, stream);
  if (actual != buffer.initial) {
    throw std::runtime_error(std::string(provider) +
                             " modified layout input storage or guards");
  }
}

void require_exact(std::string_view actual_name,
                   std::span<const float> actual,
                   std::string_view expected_name,
                   std::span<const float> expected,
                   flagdnnDataType_t data_type) {
  const std::vector<std::uint8_t> actual_bytes =
      tensor_io::encode(actual, data_type);
  const std::vector<std::uint8_t> expected_bytes =
      tensor_io::encode(expected, data_type);
  if (actual.size() != expected.size() ||
      actual_bytes.size() != expected_bytes.size()) {
    throw std::runtime_error(
        std::string(actual_name) + " size differs from " +
        std::string(expected_name));
  }
  if (actual_bytes == expected_bytes) {
    return;
  }
  const std::size_t element_size = tensor_io::data_type_size(data_type);
  for (std::size_t index = 0; index < actual.size(); ++index) {
    const std::size_t begin = index * element_size;
    if (!std::equal(actual_bytes.begin() +
                        static_cast<std::ptrdiff_t>(begin),
                    actual_bytes.begin() +
                        static_cast<std::ptrdiff_t>(begin + element_size),
                    expected_bytes.begin() +
                        static_cast<std::ptrdiff_t>(begin))) {
      throw std::runtime_error(
          std::string(actual_name) + " differs from " +
          std::string(expected_name) + " at logical element " +
          std::to_string(index));
    }
  }
  throw std::logic_error("layout exact comparison failed without mismatch");
}

void execute(LayoutExecutable& executable,
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

std::string single_line_reason(std::string_view reason) {
  std::string result(reason);
  std::replace(result.begin(), result.end(), '\r', ' ');
  std::replace(result.begin(), result.end(), '\n', ' ');
  return result;
}

std::optional<std::string> run_case(const LayoutTestCase& test_case,
                                    flagdnn::Handle& handle,
                                    acl::Stream& stream) {
  validate_layout_case(test_case);
  const std::vector<float> logical_input = make_input(test_case.input);
  GuardedBuffer flagdnn_input =
      make_input_buffer(test_case.input, logical_input, stream);
  GuardedBuffer aclnn_input =
      make_input_buffer(test_case.input, logical_input, stream);
  GuardedBuffer flagdnn_output = make_output_buffer(test_case.output, stream);
  GuardedBuffer aclnn_output = make_output_buffer(test_case.output, stream);
  const std::array<flagdnnBinding_t, 2> flagdnn_bindings = {
      flagdnnBinding_t{test_case.input.uid,
                       flagdnn_input.device->opaque_at(
                           test_case.input.binding_byte_offset)},
      flagdnnBinding_t{test_case.output.uid,
                       flagdnn_output.device->opaque_at(
                           test_case.output.binding_byte_offset)}};
  const std::array<flagdnnBinding_t, 2> aclnn_bindings = {
      flagdnnBinding_t{test_case.input.uid,
                       aclnn_input.device->opaque_at(
                           test_case.input.binding_byte_offset)},
      flagdnnBinding_t{test_case.output.uid,
                       aclnn_output.device->opaque_at(
                           test_case.output.binding_byte_offset)}};

  std::unique_ptr<LayoutExecutable> flagdnn =
      build_flagdnn_layout(handle, test_case);
  std::unique_ptr<LayoutExecutable> reference =
      build_layout_reference(test_case);
  try {
    flagdnn->prepare(flagdnn_bindings, stream.opaque());
    reference->prepare(aclnn_bindings, stream.opaque());
    reference->prepare(aclnn_bindings, stream.opaque());
  } catch (const AclnnLayoutUnsupportedError& error) {
    try {
      stream.synchronize();
    } catch (...) {
    }
    reference.reset();
    flagdnn.reset();
    return "ACLNN status=" + std::to_string(error.status()) + " " +
           single_line_reason(error.what());
  }
  acl::DeviceBuffer flagdnn_workspace(flagdnn->workspace_size());
  acl::DeviceBuffer aclnn_workspace(reference->workspace_size());
  std::vector<std::uint8_t> previous_flagdnn;
  std::vector<std::uint8_t> previous_aclnn;

  try {
    for (int pass = 0; pass < 2; ++pass) {
      restore(flagdnn_input, stream);
      restore(flagdnn_output, stream);
      restore(aclnn_input, stream);
      restore(aclnn_output, stream);
      execute(*flagdnn,
              flagdnn_bindings,
              flagdnn_workspace,
              stream);
      execute(*reference,
              aclnn_bindings,
              aclnn_workspace,
              stream);
      stream.synchronize();
      require_input_unchanged("FlagDNN", flagdnn_input, stream);
      require_input_unchanged("ACLNN", aclnn_input, stream);
      const OutputData flagdnn_result = decode_output(
          "FlagDNN", flagdnn_output, test_case.output, stream);
      const OutputData aclnn_result = decode_output(
          "ACLNN", aclnn_output, test_case.output, stream);
      if (flagdnn_result.payload != aclnn_result.payload) {
        const auto mismatch = std::mismatch(flagdnn_result.payload.begin(),
                                            flagdnn_result.payload.end(),
                                            aclnn_result.payload.begin(),
                                            aclnn_result.payload.end());
        throw std::runtime_error(
            test_case.name +
            " FlagDNN raw payload differs from exact ACLNN reference at "
            "physical byte " +
            std::to_string(static_cast<std::size_t>(
                mismatch.first - flagdnn_result.payload.begin())));
      }
      require_exact("FlagDNN",
                    flagdnn_result.logical,
                    "ACLNN",
                    aclnn_result.logical,
                    test_case.output.data_type);
      if (pass == 0) {
        previous_flagdnn = flagdnn_result.payload;
        previous_aclnn = aclnn_result.payload;
      } else if (flagdnn_result.payload != previous_flagdnn ||
                 aclnn_result.payload != previous_aclnn) {
        throw std::runtime_error(
            test_case.name + " layout output is not byte-repeatable");
      }
    }
    stream.synchronize();
  } catch (...) {
    try {
      stream.synchronize();
    } catch (...) {
    }
    reference.reset();
    flagdnn.reset();
    throw;
  }
  reference.reset();
  flagdnn.reset();
  std::cout << test_case.name
            << ": PASS raw_payload=PASS exact_bytes repeatable=2 "
               "input_guard=PASS "
               "output_padding=PASS exact_reference=ACLNN\n";
  return std::nullopt;
}

}  // namespace

int run_layout_functional_test(int argc,
                               char** argv,
                               std::span<const LayoutTestCase> cases,
                               std::string_view suite_name) {
  // Exact integer and raw storage cases run in functional.ascend.<op>.dtype,
  // with an independent process and the same public case catalog.
  std::vector<LayoutTestCase> supported_cases(cases.begin(), cases.end());
  std::erase_if(supported_cases, [](const LayoutTestCase& test_case) {
    const auto type = test_case.input.data_type;
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
      throw std::invalid_argument(
          "Ascend layout functional common catalog must contain cases");
    }
    const LayoutOperation operation = cases.front().operation;
    if (std::any_of(cases.begin(), cases.end(), [&](const auto& test_case) {
          return test_case.operation != operation;
        })) {
      throw std::invalid_argument(
          "Ascend layout functional suite mixes operations");
    }
    std::vector<LayoutTestCase> ascend_cases(cases.begin(), cases.end());
    constexpr std::array<flagdnnDataType_t, 3> kDataTypes = {
        FLAGDNN_DATA_FLOAT32,
        FLAGDNN_DATA_FLOAT16,
        FLAGDNN_DATA_BFLOAT16,
    };
    std::int64_t uid = 73000;
    for (const flagdnnDataType_t data_type : kDataTypes) {
      ascend_cases.push_back(make_special_case(operation, data_type, uid));
      uid += 2;
    }

    acl::DevelopmentEnvironment development(
        operation_name(operation) + "-functional");
    acl::AclRuntime runtime;
    const std::string soc = acl::soc_name();
    development.prepare_target(soc);
    std::cout << "ASCEND_VALIDATION_DEVELOPMENT_ROOT root="
              << development.root() << '\n';
    std::cout << "ACLNN_CAPABILITY_IDENTITY soc=" << soc
              << " cann_runtime=" << acl::runtime_package_version() << '\n';

    const char* filter = std::getenv("FLAGDNN_ASCEND_LAYOUT_CASE");
    std::vector<const LayoutTestCase*> selected;
    for (const LayoutTestCase& test_case : ascend_cases) {
      if (filter != nullptr && filter[0] != '\0' &&
          test_case.name.find(filter) == std::string::npos) {
        continue;
      }
      selected.push_back(&test_case);
    }
    if (selected.empty()) {
      throw std::runtime_error(
          "FLAGDNN_ASCEND_LAYOUT_CASE matched no test cases");
    }
    acl::ExactReferenceCoverage coverage(selected.size());
    {
      acl::Stream stream;
      {
        flagdnn::Handle handle("ascend", 0);
        handle.set_compiler(
            argv[1], argv[2], development.graph_cache().string());
        for (const LayoutTestCase* test_case : selected) {
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
    std::cout << suite_name << ": PASS cases=" << coverage.passed()
              << " catalog_cases=" << ascend_cases.size() << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << suite_name << "_FAILED: " << error.what() << '\n';
    return 1;
  }
}

}  // namespace flagdnn::testing
