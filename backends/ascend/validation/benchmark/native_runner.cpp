/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "common/native_runner.hpp"
#include "common/attention.hpp"
#include "common/attention_runner.hpp"
#include "common/causal_convolution.hpp"
#include "common/index.hpp"
#include "common/moe_matmul.hpp"
#include "common/normalization_extended.hpp"
#include "common/position_embedding.hpp"
#include "common/resample.hpp"
#include "common/statistics.hpp"
#include <cstdlib>
namespace flagdnn::testing {
int run_attention_benchmark_test(int argc, char **argv,
                                 AttentionBenchmarkOperation op) {
  if (setenv("FLAGDNN_ASCEND_PAIRED_BENCHMARK", "1", 1) != 0)
    return 1;
  if (op == AttentionBenchmarkOperation::kForward)
    return run_sdpa_functional_test(argc, argv, make_sdpa_benchmark_cases());
  if (op == AttentionBenchmarkOperation::kBackward)
    return run_sdpa_backward_functional_test(
        argc, argv, make_sdpa_backward_benchmark_cases());
  return 77;
}
int run_native_benchmark_test(int argc, char **argv, std::string_view op) {
  if (op == "moe_grouped_matmul" || op == "moe_grouped_matmul_bwd")
    return run_moe_matmul_functional_test(
        argc, argv, make_moe_matmul_cases(op == "moe_grouped_matmul_bwd"),
        true);
  if (op == "causal_conv1d")
    return run_causal_convolution_functional_test(
        argc, argv, make_causal_convolution_cases(), true);
  if (op == "resample")
    return run_resample_functional_test(argc, argv, make_resample_cases(),
                                        true);
  if (op == "rope" || op == "rope_backward")
    return run_rope_functional_test(
        argc, argv, make_rope_cases(op == "rope_backward"), true);
  if (op == "concatenate" || op == "gen_index")
    return run_index_functional_test(argc, argv, make_index_cases(op), true);
  if (op == "genstats")
    return run_statistics_functional_test(argc, argv, make_genstats_cases(),
                                          true);
  if (op == "bn_finalize")
    return run_statistics_functional_test(argc, argv, make_bn_finalize_cases(),
                                          true);
  return run_extended_normalization_functional_test(
      argc, argv, make_extended_normalization_cases(std::string(op)), true);
}
} // namespace flagdnn::testing
