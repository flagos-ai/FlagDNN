/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/reduction.hpp"

#include "validation/acl_runtime.hpp"
#include "validation/development_environment.hpp"
#include "validation/exact_reference.hpp"
#include "validation/functional/aclnn_reduction.hpp"
#include "validation/reduction_validation.hpp"
#include "validation/tensor_io.hpp"

#include <flagdnn/flagdnn.hpp>

#include <algorithm>
#include <array>
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

constexpr std::size_t kTailGuardBytes = 32;
constexpr std::uint8_t kAllocationGuard = 0xD3U;

struct GuardedBuffer {
  std::unique_ptr<acl::DeviceBuffer> device;
  std::vector<std::uint8_t> initial;
};

std::string mode_name(flagdnnReductionMode_t mode) {
  switch (mode) {
    case FLAGDNN_REDUCTION_ADD:
      return "sum";
    case FLAGDNN_REDUCTION_AVG:
      return "avg";
    case FLAGDNN_REDUCTION_MUL:
      return "mul";
  }
  throw std::invalid_argument("Ascend reduction mode is invalid");
}

GuardedBuffer make_buffer(const TestTensor& tensor,
                          std::span<const float> physical,
                          acl::Stream& stream) {
  const std::vector<std::uint8_t> encoded =
      tensor_io::encode(physical, tensor.data_type);
  if (encoded.size() != tensor_io::encoded_byte_count(tensor)) {
    throw std::logic_error("reduction encoded tensor byte count differs");
  }
  const std::size_t allocation = tensor_io::checked_add(
      tensor_io::allocation_byte_count(tensor),
      kTailGuardBytes,
      "reduction guarded allocation");
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

void require_input_unchanged(std::string_view provider,
                             const GuardedBuffer& buffer,
                             acl::Stream& stream) {
  if (read_all(buffer, stream) != buffer.initial) {
    throw std::runtime_error(std::string(provider) +
                             " modified reduction input storage or guards");
  }
}

std::vector<float> read_output(std::string_view provider,
                               const GuardedBuffer& buffer,
                               const TestTensor& tensor,
                               acl::Stream& stream) {
  const std::vector<std::uint8_t> all = read_all(buffer, stream);
  const std::size_t begin = tensor.binding_byte_offset;
  const std::size_t end = tensor_io::checked_add(
      begin,
      tensor_io::encoded_byte_count(tensor),
      "reduction guarded payload");
  if (all.size() != buffer.initial.size() || end > all.size() ||
      !std::equal(all.begin(),
                  all.begin() + static_cast<std::ptrdiff_t>(begin),
                  buffer.initial.begin()) ||
      !std::equal(all.begin() + static_cast<std::ptrdiff_t>(end),
                  all.end(),
                  buffer.initial.begin() + static_cast<std::ptrdiff_t>(end))) {
    throw std::runtime_error(std::string(provider) +
                             " modified a reduction binding offset/tail guard");
  }
  const std::span<const std::uint8_t> payload(all.data() + begin, end - begin);
  const std::vector<float> physical =
      tensor_io::decode_storage(provider, payload, tensor);
  tensor_io::require_padding_unchanged(provider, physical, tensor);
  return tensor_io::gather(physical, tensor);
}

std::string single_line_reason(std::string_view reason) {
  std::string result(reason);
  std::replace(result.begin(), result.end(), '\r', ' ');
  std::replace(result.begin(), result.end(), '\n', ' ');
  return result;
}

void execute(ReductionExecutable& executable,
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

void verify_pass(const ReductionTestCase& test_case,
                 GuardedBuffer& flagdnn_input,
                 GuardedBuffer& flagdnn_output,
                 GuardedBuffer& aclnn_input,
                 GuardedBuffer& aclnn_output,
                 acl::Stream& stream) {
  require_input_unchanged("FlagDNN", flagdnn_input, stream);
  require_input_unchanged("ACLNN", aclnn_input, stream);
  const std::vector<float> flagdnn =
      read_output("FlagDNN", flagdnn_output, test_case.output, stream);
  const std::vector<float> reference =
      read_output("ACLNN", aclnn_output, test_case.output, stream);
  acl::compare_exact_reference(flagdnn,
                               reference,
                               test_case.absolute_tolerance,
                               test_case.relative_tolerance,
                               test_case.name);
  std::cout << test_case.name
            << ": FlagDNN Graph/libtriton_jit vs ACLNN reduction PASS "
               "exact_reference=ACLNN_REDUCTION\n";
}

std::optional<std::string> run_case(const ReductionTestCase& test_case,
                                    flagdnn::Handle& handle,
                                    acl::Stream& stream) {
  validate_reduction_case(test_case);
  const std::vector<float> logical_input = make_reduction_input(test_case);
  const TestTensor reference_input =
      reduction_reference_input_tensor(test_case);

  GuardedBuffer flagdnn_input =
      make_input_buffer(test_case.input, logical_input, stream);
  GuardedBuffer aclnn_input =
      make_input_buffer(reference_input, logical_input, stream);
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
      flagdnnBinding_t{reference_input.uid,
                       aclnn_input.device->opaque_at(
                           reference_input.binding_byte_offset)},
      flagdnnBinding_t{test_case.output.uid,
                       aclnn_output.device->opaque_at(
                           test_case.output.binding_byte_offset)}};

  std::unique_ptr<ReductionExecutable> flagdnn =
      build_flagdnn_reduction(handle, test_case);
  std::unique_ptr<ReductionExecutable> reference =
      build_reduction_reference(test_case);
  flagdnn->prepare(flagdnn_bindings, stream.opaque());
  try {
    reference->prepare(aclnn_bindings, stream.opaque());
    reference->prepare(aclnn_bindings, stream.opaque());
  } catch (const AclnnReductionUnsupportedError& error) {
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

  try {
    for (int pass = 0; pass < 2; ++pass) {
      execute(*flagdnn,
              flagdnn_bindings,
              flagdnn_workspace,
              stream);
      execute(*reference,
              aclnn_bindings,
              aclnn_workspace,
              stream);
      stream.synchronize();
      verify_pass(test_case,
                  flagdnn_input,
                  flagdnn_output,
                  aclnn_input,
                  aclnn_output,
                  stream);
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
            << ": PASS repeatable=2 input_guard=PASS "
               "output_padding=PASS\n";
  return std::nullopt;
}

}  // namespace

int run_reduction_functional_test(
    int argc,
    char** argv,
    std::span<const ReductionTestCase> cases) {
  // Exact integer and raw storage cases run in functional.ascend.<op>.dtype,
  // with an independent process and the same public case catalog.
  std::vector<ReductionTestCase> supported_cases(cases.begin(), cases.end());
  std::erase_if(supported_cases, [](const ReductionTestCase& test_case) {
    const auto type = test_case.input.data_type;
    return (type != FLAGDNN_DATA_FLOAT32 && type != FLAGDNN_DATA_FLOAT16 &&
            type != FLAGDNN_DATA_BFLOAT16) ||
           test_case.output.data_type != type;
  });
  cases = supported_cases;

  if (argc != 3) {
    std::cerr << "usage: " << argv[0]
              << " COMPILER_EXECUTABLE COMPILER_ENTRY\n";
    return 2;
  }
  try {
    std::cout << std::setprecision(9);
    constexpr std::array<flagdnnReductionMode_t, 3> kModes = {
        FLAGDNN_REDUCTION_ADD,
        FLAGDNN_REDUCTION_AVG,
        FLAGDNN_REDUCTION_MUL};

    std::vector<ReductionTestCase> ascend_cases;
    ascend_cases.reserve(cases.size());
    for (std::size_t index = 0; index < kModes.size(); ++index) {
      std::vector<ReductionTestCase> mode_cases =
          make_ascend_reduction_cases(cases, kModes[index]);
      if (mode_cases.empty()) {
        throw std::logic_error("Ascend reduction " + mode_name(kModes[index]) +
                               " functional catalog has invalid size");
      }
      ascend_cases.insert(ascend_cases.end(),
                          std::make_move_iterator(mode_cases.begin()),
                          std::make_move_iterator(mode_cases.end()));
    }

    acl::DevelopmentEnvironment development("reduction-functional");
    acl::AclRuntime runtime;
    const std::string soc = acl::soc_name();
    development.prepare_target(soc);
    std::cout << "ASCEND_VALIDATION_DEVELOPMENT_ROOT root="
              << development.root() << '\n';
    std::cout << "ACLNN_CAPABILITY_IDENTITY soc=" << soc
              << " cann_runtime=" << acl::runtime_package_version() << '\n';

    const char* filter = std::getenv("FLAGDNN_ASCEND_REDUCTION_CASE");
    if (filter == nullptr || filter[0] == '\0') {
      filter = std::getenv("FLAGDNN_REDUCTION_CASE");
    }
    std::vector<const ReductionTestCase*> selected;
    selected.reserve(ascend_cases.size());
    for (const ReductionTestCase& test_case : ascend_cases) {
      if (filter != nullptr && filter[0] != '\0' &&
          test_case.name.find(filter) == std::string::npos) {
        continue;
      }
      selected.push_back(&test_case);
    }
    if (selected.empty()) {
      throw std::runtime_error(
          "reduction case filter matched no test cases");
    }
    acl::ExactReferenceCoverage coverage(selected.size());
    {
      acl::Stream stream;
      {
        flagdnn::Handle handle("ascend", 0);
        handle.set_compiler(
            argv[1], argv[2], development.graph_cache().string());
        for (const ReductionTestCase* test_case : selected) {
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
    std::cout << "FLAGDNN_REDUCTION_FUNCTIONAL: PASS cases="
              << coverage.passed()
              << " catalog_cases=" << ascend_cases.size() << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FLAGDNN_REDUCTION_FUNCTIONAL_FAILED: " << error.what()
              << '\n';
    return 1;
  }
}

}  // namespace flagdnn::testing
