/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <functional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "common/pointwise.hpp"
#include "host_pointwise.hpp"
#include "host_runner.hpp"
#include "reference/cpu/pointwise_int32.hpp"
#include "reference/cpu/pointwise.hpp"
#include "pointwise_runner_support.hpp"

namespace flagdnn::testing {

namespace {
void run_identity_case(
    const PointwiseTestCase &test_case,
    const std::function<flagdnn::Handle &()> &get_handle,
    validation::hygon::Stream &stream) {
  namespace host = hygon_functional::host;
  namespace io = validation::hygon::tensor_io;
  validate_pointwise_case(test_case);
  if (test_case.inputs.size() != 1 ||
      test_case.inputs.front().data_type != test_case.output.data_type ||
      io::element_count(test_case.inputs.front()) !=
          io::element_count(test_case.output))
    throw std::invalid_argument("invalid storage-preserving identity case");
  const auto &input = test_case.inputs.front();
  const auto width = io::data_type_size(input.data_type);
  const auto count = io::element_count(input);
  std::vector<host::Buffer> buffers;
  buffers.push_back(host::raw_buffer(input, stream));
  // Retain the original raw-copy corpus, including signed zero, NaN payloads,
  // integer boundary bits and every byte representation of one-byte types.
  constexpr std::uint32_t patterns[] = {0, 0x80000000U, 0x7fc12345U,
                                       0x3f800001U, 0x7fffffffU, 0xffffffffU};
  std::vector<std::uint8_t> logical(count * width);
  for (std::size_t i = 0; i < count; ++i) {
    std::uint32_t bits =
        patterns[(i + input.uid) % 6] ^
        (i % 3 == 0 ? std::uint32_t(i * 97) : 0U);
    if (width == 1)
      bits = input.data_type == FLAGDNN_DATA_BOOLEAN
                 ? (i + input.uid) % 2
                 : (i * 17 + input.uid * 23) % 256;
    std::memcpy(host::raw_element(buffers.front(), i), &bits, width);
    std::memcpy(logical.data() + i * width, &bits, width);
  }
  const auto reference = reference::cpu::evaluate_identity_bytes(logical);
  if (reference.size() != logical.size())
    throw std::logic_error("identity CPU reference byte count mismatch");
  buffers.push_back(host::raw_buffer(test_case.output, stream));
  auto expected = buffers.back().initial;
  for (std::size_t i = 0; i < count; ++i)
    std::memcpy(expected.data() + test_case.output.binding_byte_offset +
                    io::logical_offset(i, test_case.output) * width,
                reference.data() + i * width, width);
  host::execute_raw(test_case.name, buffers, std::move(expected), get_handle,
                    stream, [&](flagdnn::Handle &handle) {
                      return build_flagdnn_pointwise(handle, test_case);
                    });
}

void run_int32_sub_case(
    const PointwiseTestCase &test_case,
    const std::function<flagdnn::Handle &()> &get_handle,
    validation::hygon::Stream &stream) {
  namespace host = hygon_functional::host;
  namespace io = validation::hygon::tensor_io;
  validate_pointwise_case(test_case);
  if (test_case.inputs.size() != 2 ||
      test_case.output.data_type != FLAGDNN_DATA_INT32 ||
      std::any_of(test_case.inputs.begin(), test_case.inputs.end(),
                  [](const TestTensor &t) {
                    return t.data_type != FLAGDNN_DATA_INT32;
                  }) ||
      !std::isfinite(test_case.alpha) ||
      std::trunc(test_case.alpha) != test_case.alpha ||
      test_case.alpha < std::numeric_limits<std::int32_t>::min() ||
      test_case.alpha > std::numeric_limits<std::int32_t>::max())
    throw std::invalid_argument("invalid exact INT32 SUB test case");
  std::vector<host::Buffer> buffers;
  for (std::size_t j = 0; j < test_case.inputs.size(); ++j) {
    const auto &input = test_case.inputs[j];
    buffers.push_back(host::raw_buffer(input, stream));
    for (std::size_t i = 0; i < io::element_count(input); ++i) {
      const auto value = pointwise_integer_input(i, j, test_case.mode);
      std::memcpy(host::raw_element(buffers.back(), i), &value, sizeof(value));
    }
  }
  buffers.push_back(host::raw_buffer(test_case.output, stream));
  auto expected = buffers.back().initial;
  for (std::size_t i = 0; i < io::element_count(test_case.output); ++i) {
    std::int32_t operands[2];
    for (std::size_t j = 0; j < 2; ++j) {
      const auto index = hygon_functional::pointwise::broadcast_index(
          i, test_case.inputs[j], test_case.output);
      std::memcpy(&operands[j], host::raw_element(buffers[j], index),
                  sizeof(operands[j]));
    }
    const auto value = reference::cpu::subtract_int32(
        operands[0], operands[1], static_cast<std::int32_t>(test_case.alpha));
    std::memcpy(expected.data() + test_case.output.binding_byte_offset +
                    io::logical_offset(i, test_case.output) * sizeof(value),
                &value, sizeof(value));
  }
  host::execute_raw(test_case.name, buffers, std::move(expected), get_handle,
                    stream, [&](flagdnn::Handle &handle) {
                      return build_flagdnn_pointwise(handle, test_case);
                    });
}
} // namespace

int run_pointwise_functional_test(int argc, char **argv,
                                  std::span<const PointwiseTestCase> cases,
                                  std::string_view suite_name) {
  const auto unsupported_dtype = [](const PointwiseTestCase &test_case) {
    const bool logical = test_case.mode == FLAGDNN_POINTWISE_LOGICAL_NOT ||
                         test_case.mode == FLAGDNN_POINTWISE_LOGICAL_AND ||
                         test_case.mode == FLAGDNN_POINTWISE_LOGICAL_OR ||
                         test_case.mode == FLAGDNN_POINTWISE_BINARY_SELECT;
    const bool unsupported_input =
        std::any_of(test_case.inputs.begin(), test_case.inputs.end(),
                    [logical](const TestTensor &tensor) {
                      const auto type = tensor.data_type;
                      return type != FLAGDNN_DATA_FLOAT32 &&
                             type != FLAGDNN_DATA_FLOAT16 &&
                             type != FLAGDNN_DATA_BFLOAT16 &&
                             !(logical && type == FLAGDNN_DATA_BOOLEAN);
                    });
    const bool comparison = test_case.mode >= FLAGDNN_POINTWISE_CMP_EQ &&
                            test_case.mode <= FLAGDNN_POINTWISE_CMP_LE;
    return unsupported_input ||
           (!comparison &&
            test_case.output.data_type != test_case.inputs.front().data_type);
  };

  namespace support = hygon_functional::pointwise;
  namespace hv = validation::hygon;
  const std::string family =
      cases.empty() ? std::string("pointwise")
                    : support::case_operation_name(cases.front().name);
  return support::run_suite(
      argc, argv, family, suite_name,
      [&](const std::function<flagdnn::Handle &()> &get_handle,
          hv::Stream &stream, std::size_t &matched, std::size_t &executed,
          std::size_t &skipped) {
        const char *filter = std::getenv("FLAGDNN_POINTWISE_CASE");
        for (const PointwiseTestCase &test_case : cases) {
          if (filter != nullptr &&
              test_case.name.find(filter) == std::string::npos) {
            continue;
          }
          ++matched;
          if (test_case.mode == FLAGDNN_POINTWISE_IDENTITY) {
            run_identity_case(test_case, get_handle, stream);
            ++executed;
            continue;
          }

          if (test_case.mode == FLAGDNN_POINTWISE_SUB &&
              test_case.inputs.front().data_type == FLAGDNN_DATA_INT32) {
            run_int32_sub_case(test_case, get_handle, stream);
            ++executed;
            continue;
          }

          if (test_case.mode == FLAGDNN_POINTWISE_MOD &&
              test_case.inputs.front().data_type == FLAGDNN_DATA_INT32 &&
              test_case.output.data_type == FLAGDNN_DATA_INT32 &&
              std::all_of(test_case.inputs.begin(), test_case.inputs.end(),
                          [](const TestTensor &t) {
                            return t.data_type == FLAGDNN_DATA_INT32;
                          }) &&
              test_case.input_domains.size() == test_case.inputs.size() &&
              std::all_of(test_case.input_domains.begin(),
                          test_case.input_domains.end(),
                          [](PointwiseInputDomain domain) {
                            return domain == PointwiseInputDomain::kModuloSigned;
                          })) {
            // This catalog uses only signed integers in [-5, 5] after
            // storage conversion. Their FP32 projection and remainders are
            // exact, independently checked against NVIDIA cuDNN.
            validate_pointwise_case(test_case);
            support::run_host_pointwise_case(test_case, get_handle, stream);
            ++executed;
            continue;
          }

          if (unsupported_dtype(test_case)) {
            support::emit_skip(
                family, test_case.name,
                "dtype/output combination is not supported by the Hygon "
                "adapter",
                support::reference_tensors(test_case.inputs, test_case.output));
            ++skipped;
            continue;
          }
          validate_pointwise_case(test_case);
          const std::string &operation = family;
          support::CaseResult result;
          const auto reference_operation = hv::make_hipdnn_pointwise_operation(
              test_case.mode, test_case.attributes, test_case.alpha);
          const auto tensors =
              support::reference_tensors(test_case.inputs, test_case.output);
          const auto capability = hv::hipdnn_pointwise_capability(
              reference_operation, tensors, true);
          hv::require_valid_hipdnn_adapter_contract(capability, operation);
          if (!capability.supported) {
            result =
                support::run_host_pointwise_case(test_case, get_handle, stream);
          } else if (support::uses_hygon_cpu_reference(test_case.mode)) {
            result = support::run_cpu_case(
                operation, test_case.name, test_case.inputs, test_case.output,
                test_case.input_domains, test_case.mode,
                test_case.absolute_tolerance, test_case.relative_tolerance,
                get_handle, stream, [&](flagdnn::Handle &value) {
                  return build_flagdnn_pointwise(value, test_case);
                });
          } else {
            result = support::run_case(
                operation, test_case.name, test_case.inputs, test_case.output,
                test_case.input_domains,
                hv::make_hipdnn_pointwise_operation(
                    test_case.mode, test_case.attributes, test_case.alpha),
                test_case.absolute_tolerance, test_case.relative_tolerance,
                get_handle, stream,
                [&](flagdnn::Handle &value) {
                  return build_flagdnn_pointwise(value, test_case);
                },
                [&] { return build_pointwise_reference(test_case); });
          }
          if (result == support::CaseResult::kSkipped)
            result =
                support::run_host_pointwise_case(test_case, get_handle, stream);
          result == support::CaseResult::kExecuted ? ++executed : ++skipped;
        }
      });
}

} // namespace flagdnn::testing
