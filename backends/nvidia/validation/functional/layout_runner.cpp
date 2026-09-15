/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "validation/functional/layout_runner.hpp"

#include "validation/functional/paired.hpp"
namespace flagdnn::testing {
namespace {
struct PairedLayoutCase : LayoutTestCase {
  std::vector<TestTensor> inputs, outputs;
  explicit PairedLayoutCase(const LayoutTestCase& value)
      : LayoutTestCase(value), inputs{value.input}, outputs{value.output} {}
};
}  // namespace
int run_cudnn_layout_tests(int argc, char** argv,
                           std::span<const LayoutTestCase> cases,
                           bool benchmark) {
  std::vector<PairedLayoutCase> supported;
  for (const auto& test_case : cases) {
    const auto type = test_case.input.data_type;
    if (type == FLAGDNN_DATA_FLOAT32 || type == FLAGDNN_DATA_FLOAT16 ||
        type == FLAGDNN_DATA_BFLOAT16)
      supported.emplace_back(test_case);
  }
  return cuda::run_paired_cases(
      argc, argv, std::span<const PairedLayoutCase>(supported),
      "FLAGDNN_LAYOUT_CASE", build_flagdnn_layout,
      [](const PairedLayoutCase& test_case) {
        std::vector<float> values(cuda::element_count(test_case.input));
        for (std::size_t index = 0; index < values.size(); ++index)
          values[index] =
              static_cast<float>(static_cast<int>((index * 17) % 41) - 20) /
              13.0F;
        return std::vector<std::vector<float>>{std::move(values)};
      },
      build_layout_reference,
      [](const PairedLayoutCase&, std::size_t) {
        return cuda::PairedTolerance{0.0, 0.0};
      },
      benchmark, "cuDNN");
}
int run_layout_functional_test(int argc, char** argv,
                               std::span<const LayoutTestCase> cases,
                               std::string_view) {
  return run_cudnn_layout_tests(argc, argv, cases);
}
}  // namespace flagdnn::testing
