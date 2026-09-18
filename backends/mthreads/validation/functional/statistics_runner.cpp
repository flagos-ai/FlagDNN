/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "backends/mthreads/validation/functional/mudnn_extended.hpp"
#include "backends/mthreads/validation/functional/paired.hpp"
#include "common/statistics.hpp"
namespace flagdnn::testing {
int run_statistics_functional_test(int argc, char** argv,
                                   std::span<const StatisticsTestCase> cases,
                                   bool benchmark) {
  return mthreads::run_paired_cases(
      argc, argv, cases, "FLAGDNN_STATISTICS_CASE", cases.front().operation,
      build_flagdnn_statistics,
      [](const StatisticsTestCase& test_case) {
        std::vector<std::vector<float>> inputs;
        for (const auto& tensor : test_case.inputs) {
          std::vector<float> values(mthreads::io::element_count(tensor));
          for (std::size_t index = 0; index < values.size(); ++index)
            values[index] =
                statistics_input_value(test_case, inputs.size(), index);
          inputs.push_back(std::move(values));
        }
        return inputs;
      },
      [](const auto& c) { return mthreads::reference(c); },
      [](const StatisticsTestCase&, std::size_t) {
        // Native FP32 moment finalization can differ from FP64
        // finalization near zero variance; use the shared NVIDIA bounds.
        return mthreads::Tolerance{2.0e-5, 2.0e-4};
      },
      benchmark);
}
}  // namespace flagdnn::testing
