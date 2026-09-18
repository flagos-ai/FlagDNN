// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#include "functional/paired.hpp"
namespace flagdnn::testing {
namespace f = flagdnn::validation::thead::functional;
int run_statistics_functional_test(int argc, char** argv,
                                   std::span<const StatisticsTestCase> cases,
                                   bool benchmark) {
  return f::run_paired_cases(
      argc, argv, cases, cases.front().operation, build_flagdnn_statistics,
      [](const StatisticsTestCase& t) {
        std::vector<std::vector<float>> values;
        for (std::size_t i = 0; i < t.inputs.size(); ++i) {
          std::vector<float> data(f::element_count(t.inputs[i]));
          for (std::size_t j = 0; j < data.size(); ++j)
            data[j] = statistics_input_value(t, i, j);
          values.push_back(std::move(data));
        }
        return values;
      },
      benchmark);
}
int run_resample_functional_test(int argc, char** argv,
                                 std::span<const ResampleTestCase> cases,
                                 bool benchmark) {
  return f::run_paired_cases(argc, argv, cases, "resample",
                             build_flagdnn_resample, resample_inputs,
                             benchmark);
}
int run_extended_normalization_functional_test(
    int argc, char** argv, std::span<const ExtendedNormalizationTestCase> cases,
    bool benchmark) {
  return f::run_paired_cases(
      argc, argv, cases, cases.front().operation,
      build_flagdnn_extended_normalization,
      [](const ExtendedNormalizationTestCase& t) {
        std::vector<std::vector<float>> values;
        for (const auto& tensor : t.inputs) {
          std::vector<float> data(f::element_count(tensor));
          for (std::size_t j = 0; j < data.size(); ++j) {
            data[j] =
                static_cast<float>(
                    static_cast<int>((j * 17 + tensor.uid * 7) % 61) - 30) /
                19.0F;
            if (values.size() == t.inputs.size() - 1)
              data[j] = 0.5F + std::abs(data[j]);
          }
          values.push_back(std::move(data));
        }
        return values;
      },
      benchmark);
}
}  // namespace flagdnn::testing
