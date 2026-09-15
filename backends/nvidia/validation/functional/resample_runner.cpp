/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "common/resample.hpp"
#include "validation/functional/cudnn_extended.hpp"
#include "validation/functional/paired.hpp"
namespace flagdnn::testing {
int run_resample_functional_test(int argc, char** argv,
                                 std::span<const ResampleTestCase> cases,
                                 bool benchmark) {
  return cuda::run_paired_cases(
      argc, argv, cases, "FLAGDNN_RESAMPLE_CASE", build_flagdnn_resample,
      resample_inputs, build_cudnn_resample,
      [](const ResampleTestCase& test_case, std::size_t output) {
        if (output != 0) return cuda::PairedTolerance{0.0, 0.0};
        const auto type = test_case.outputs[0].data_type;
        const double tolerance = type == FLAGDNN_DATA_BFLOAT16  ? 4.0e-3
                                 : type == FLAGDNN_DATA_FLOAT16 ? 5.0e-4
                                                                : 1.0e-5;
        return cuda::PairedTolerance{tolerance, tolerance};
      },
      benchmark, "cuDNN Graph / cudnnSpatialTfSamplerForward");
}
}  // namespace flagdnn::testing
