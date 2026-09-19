/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "common/statistics.hpp"
#include "validation/functional/aclnn_extended.hpp"
#include "validation/functional/paired.hpp"
namespace flagdnn::testing {
int run_statistics_functional_test(int argc, char **argv,
                                   std::span<const StatisticsTestCase> cases,
                                   bool benchmark) {
  return ascend::run_paired_cases(
      argc, argv, cases, "FLAGDNN_STATISTICS_CASE", build_flagdnn_statistics,
      [](const StatisticsTestCase &test_case) {
        std::vector<std::vector<float>> inputs;
        for (const auto &tensor : test_case.inputs) {
          std::vector<float> values(ascend::io::element_count(tensor));
          for (std::size_t index = 0; index < values.size(); ++index)
            values[index] =
                statistics_input_value(test_case, inputs.size(), index);
          inputs.push_back(std::move(values));
        }
        return inputs;
      },
      build_aclnn_statistics,
      [](const StatisticsTestCase &, std::size_t) {
        // ACLNN finalizes FP32 moments in FP32. Near zero variance its
        // inverse variance can differ from the FP64 finalization by ~1e-4.
        return ascend::PairedTolerance{2.0e-5, 2.0e-4};
      },
      benchmark);
}
} // namespace flagdnn::testing
