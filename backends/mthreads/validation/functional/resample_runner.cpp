/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "backends/mthreads/validation/functional/mudnn_extended.hpp"
#include "backends/mthreads/validation/functional/paired.hpp"
#include "common/resample.hpp"
namespace flagdnn::testing {
int run_resample_functional_test(int argc, char** argv,
                                 std::span<const ResampleTestCase> cases,
                                 bool benchmark) {
  return mthreads::run_paired_cases(
      argc, argv, cases, "FLAGDNN_RESAMPLE_CASE", "resample",
      build_flagdnn_resample, resample_inputs,
      [](const auto& c) { return mthreads::reference(c); },
      [](const ResampleTestCase& test_case, std::size_t output) {
        if (output != 0) return mthreads::Tolerance{0.0, 0.0};
        const auto type = test_case.outputs[0].data_type;
        const double tolerance = type == FLAGDNN_DATA_BFLOAT16  ? 4.0e-3
                                 : type == FLAGDNN_DATA_FLOAT16 ? 5.0e-4
                                                                : 1.0e-5;
        return mthreads::Tolerance{tolerance, tolerance};
      },
      benchmark);
}
}  // namespace flagdnn::testing
