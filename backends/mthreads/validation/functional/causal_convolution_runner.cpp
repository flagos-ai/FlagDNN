/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "backends/mthreads/validation/functional/mudnn_extended.hpp"
#include "backends/mthreads/validation/functional/paired.hpp"
#include "common/causal_convolution.hpp"
namespace flagdnn::testing {
int run_causal_convolution_functional_test(
    int argc, char** argv, std::span<const CausalConvolutionTestCase> cases,
    bool benchmark) {
  return mthreads::run_paired_cases(
      argc, argv, cases, "FLAGDNN_CAUSAL_CONVOLUTION_CASE", "causal_conv1d",
      build_flagdnn_causal_convolution, causal_convolution_inputs,
      [](const auto& c) { return mthreads::reference(c); },
      [](const CausalConvolutionTestCase& test_case, std::size_t) {
        const auto type = test_case.outputs[0].data_type;
        const double tolerance = type == FLAGDNN_DATA_BFLOAT16  ? 8.0e-3
                                 : type == FLAGDNN_DATA_FLOAT16 ? 1.0e-3
                                                                : 2.0e-5;
        return mthreads::Tolerance{tolerance, tolerance};
      },
      benchmark);
}
}  // namespace flagdnn::testing
