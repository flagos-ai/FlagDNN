/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "common/position_embedding.hpp"
#include "validation/functional/aclnn_extended.hpp"
#include "validation/functional/paired.hpp"
namespace flagdnn::testing {
int run_rope_functional_test(int argc, char **argv,
                             std::span<const RoPETestCase> cases,
                             bool benchmark) {
  return ascend::run_paired_cases(
      argc, argv, cases, "FLAGDNN_ROPE_CASE", build_flagdnn_rope, rope_inputs,
      build_aclnn_rope,
      [](const RoPETestCase &c, std::size_t) {
        double t = c.outputs[0].data_type == FLAGDNN_DATA_BFLOAT16  ? 8e-3
                   : c.outputs[0].data_type == FLAGDNN_DATA_FLOAT16 ? 1e-3
                                                                    : 2e-5;
        return ascend::PairedTolerance{t, t};
      },
      benchmark);
}
} // namespace flagdnn::testing
