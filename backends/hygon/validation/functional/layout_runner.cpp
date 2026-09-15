/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include <algorithm>
#include <cstdlib>
#include <functional>
#include <span>
#include <string_view>
#include <vector>

#include "common/layout.hpp"
#include "tensor_runner_support.hpp"

namespace flagdnn::testing {

int run_layout_functional_test(int argc, char **argv,
                               std::span<const LayoutTestCase> cases,
                               std::string_view suite_name) {
  // This adapter currently validates matching floating input/output storage.
  // Other shared dtype/output combinations are enabled with their backend
  // support.
  std::vector<LayoutTestCase> supported_cases(cases.begin(), cases.end());
  std::erase_if(supported_cases, [](const LayoutTestCase &test_case) {
    const auto type = test_case.input.data_type;
    return type != FLAGDNN_DATA_FLOAT32 && type != FLAGDNN_DATA_FLOAT16 &&
           type != FLAGDNN_DATA_BFLOAT16;
  });
  cases = supported_cases;

  namespace support = hygon_functional::tensor;
  namespace hv = validation::hygon;
  return support::run_suite(
      argc, argv, "layout", suite_name,
      [&](const std::function<flagdnn::Handle &()> &get_handle,
          hv::Stream &stream, std::size_t &matched, std::size_t &executed,
          std::size_t &skipped) {
        const char *filter = std::getenv("FLAGDNN_LAYOUT_CASE");
        for (const LayoutTestCase &test_case : cases) {
          if (filter != nullptr &&
              test_case.name.find(filter) == std::string::npos) {
            continue;
          }
          ++matched;
          validate_layout_case(test_case);
          const support::CaseResult result =
              support::run_layout_case(test_case, get_handle, stream);
          result == support::CaseResult::kExecuted ? ++executed : ++skipped;
        }
      });
}

} // namespace flagdnn::testing
