/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "common/normalization_extended.hpp"
#include "validation/functional/aclnn_extended.hpp"
#include "validation/functional/paired.hpp"
namespace flagdnn::testing {
int run_extended_normalization_functional_test(
    int argc, char **argv, std::span<const ExtendedNormalizationTestCase> cases,
    bool benchmark) {
  return ascend::run_paired_cases(
      argc, argv, cases, "FLAGDNN_NORMALIZATION_EXTENDED_CASE",
      build_flagdnn_extended_normalization,
      [](const ExtendedNormalizationTestCase &test_case) {
        std::vector<std::vector<float>> inputs;
        for (const auto &tensor : test_case.inputs) {
          std::vector<float> values(ascend::io::element_count(tensor));
          for (std::size_t index = 0; index < values.size(); ++index) {
            values[index] =
                static_cast<float>(
                    static_cast<int>((index * 17 + tensor.uid * 7) % 61) - 30) /
                19.0F;
            // Saved statistics are independent inputs to backward. Positive
            // inverse standard deviations exercise their supplied values.
            if (test_case.operation.ends_with("_backward") &&
                inputs.size() == test_case.inputs.size() - 1)
              values[index] = 0.5F + std::abs(values[index]);
          }
          inputs.push_back(std::move(values));
        }
        return inputs;
      },
      build_aclnn_extended_normalization,
      [](const ExtendedNormalizationTestCase &test_case, std::size_t index) {
        const auto type = test_case.outputs[index].data_type;
        const double tolerance = type == FLAGDNN_DATA_BFLOAT16  ? 8.0e-3
                                 : type == FLAGDNN_DATA_FLOAT16 ? 1.0e-3
                                                                : 2.0e-4;
        return ascend::PairedTolerance{tolerance, tolerance};
      },
      benchmark);
}
} // namespace flagdnn::testing
