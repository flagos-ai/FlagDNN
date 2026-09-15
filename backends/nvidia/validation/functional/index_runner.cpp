/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "common/index.hpp"
#include "validation/functional/cudnn_extended.hpp"
#include "validation/functional/paired.hpp"
namespace flagdnn::testing {
namespace {
struct PairedIndexCase : IndexTestCase {
  std::vector<TestTensor> outputs;
};
}  // namespace
int run_index_functional_test(int argc, char** argv,
                              std::span<const IndexTestCase> cases,
                              bool benchmark) {
  std::vector<PairedIndexCase> paired;
  for (const auto& test_case : cases) {
    const auto type = test_case.output.data_type;
    if (test_case.operation == "concatenate" && type != FLAGDNN_DATA_FLOAT32 &&
        type != FLAGDNN_DATA_FLOAT16 && type != FLAGDNN_DATA_BFLOAT16)
      continue;
    paired.push_back({test_case, {test_case.output}});
  }
  return cuda::run_paired_cases<PairedIndexCase>(
      argc, argv, paired, "FLAGDNN_INDEX_CASE", build_flagdnn_index,
      [](const PairedIndexCase& test_case) {
        std::vector<std::vector<float>> inputs;
        for (const auto& tensor : test_case.inputs) {
          std::vector<float> values(cuda::element_count(tensor));
          for (std::size_t index = 0; index < values.size(); ++index)
            values[index] =
                static_cast<float>(
                    static_cast<int>((index * 17 + inputs.size() * 7) % 61) -
                    30) /
                16.0F;
          inputs.push_back(std::move(values));
        }
        return inputs;
      },
      build_cudnn_index,
      [](const PairedIndexCase&, std::size_t) {
        return cuda::PairedTolerance{0.0, 0.0};
      },
      benchmark, "cuDNN Graph / cudnnTransformTensor");
}
}  // namespace flagdnn::testing
