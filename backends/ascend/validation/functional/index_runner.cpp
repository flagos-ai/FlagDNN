/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "common/index.hpp"
#include "validation/functional/aclnn_extended.hpp"
#include "validation/functional/paired.hpp"
namespace flagdnn::testing {
namespace {
struct PairedIndexCase : IndexTestCase {
  std::vector<TestTensor> outputs;
};
} // namespace
int run_index_functional_test(int argc, char **argv,
                              std::span<const IndexTestCase> cases,
                              bool benchmark) {
  std::vector<PairedIndexCase> paired;
  for (const auto &test_case : cases) {
    paired.push_back({test_case, {test_case.output}});
  }
  return ascend::run_paired_cases<PairedIndexCase>(
      argc, argv, paired, "FLAGDNN_INDEX_CASE", build_flagdnn_index,
      [](const PairedIndexCase &test_case) {
        std::vector<std::vector<float>> inputs;
        for (const auto &tensor : test_case.inputs) {
          std::vector<float> values(ascend::io::element_count(tensor));
          for (std::size_t index = 0; index < values.size(); ++index)
            values[index] =
                static_cast<float>(
                    static_cast<int>((index * 17 + inputs.size() * 7) % 61) -
                    30) /
                16.0F;
          if (tensor.data_type == FLAGDNN_DATA_BOOLEAN)
            for (std::size_t i = 0; i < values.size(); ++i)
              values[i] = i % 2;
          inputs.push_back(std::move(values));
        }
        return inputs;
      },
      [](const PairedIndexCase &c) {
        auto spec = static_cast<const IndexTestCase &>(c);
        spec.output = c.outputs[0];
        return build_aclnn_index(spec);
      },
      [](const PairedIndexCase &, std::size_t) {
        return ascend::PairedTolerance{0.0, 0.0};
      },
      benchmark);
}
} // namespace flagdnn::testing
