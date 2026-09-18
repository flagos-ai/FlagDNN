/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "backends/mthreads/validation/functional/mudnn_extended.hpp"
#include "backends/mthreads/validation/functional/paired.hpp"
namespace flagdnn::testing {
int run_rope_functional_test(int argc, char** argv,
                             std::span<const RoPETestCase> cases,
                             bool benchmark) {
  return mthreads::run_paired_cases(
      argc, argv, cases, "FLAGDNN_ROPE_CASE", cases.front().operation,
      build_flagdnn_rope, rope_inputs,
      [](const auto& c) { return mthreads::reference(c); },
      [](const RoPETestCase& c, std::size_t) {
        const double tolerance =
            c.outputs[0].data_type == FLAGDNN_DATA_FLOAT32   ? 2e-5
            : c.outputs[0].data_type == FLAGDNN_DATA_FLOAT16 ? 1e-3
                                                             : 8e-3;
        return mthreads::Tolerance{tolerance, tolerance};
      },
      benchmark);
}
}  // namespace flagdnn::testing
