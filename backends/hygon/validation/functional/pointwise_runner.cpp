/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include <algorithm>
#include <cstdlib>
#include <functional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "common/pointwise.hpp"
#include "host_pointwise.hpp"
#include "host_runner.hpp"
#include "pointwise_runner_support.hpp"

namespace flagdnn::testing {

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

  namespace host = hygon_functional::host;
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
            host::run_copy(
                test_case.name, test_case.inputs, test_case.output, get_handle,
                stream,
                [](std::size_t i) {
                  return std::pair<std::size_t, std::size_t>{0, i};
                },
                [&](flagdnn::Handle &h) {
                  return build_flagdnn_pointwise(h, test_case);
                });
            ++executed;
            continue;
          }
          if (test_case.inputs.front().data_type == FLAGDNN_DATA_INT32) {
            host::run_integer_pointwise(test_case, get_handle, stream);
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
