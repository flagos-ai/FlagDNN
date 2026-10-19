// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#include "common/attention.hpp"
#include "common/attention_runner.hpp"
#include "common/causal_convolution.hpp"
#include "common/index.hpp"
#include "common/native_runner.hpp"
#include "common/normalization_extended.hpp"
#include "common/resample.hpp"
#include "common/statistics.hpp"
#include "corex_cudnn_raii.hpp"
#include "extended_reference.hpp"
#include "functional/runner_support.hpp"
#include <algorithm>
#include <cctype>
#include <iostream>
#include <stdexcept>
namespace flagdnn::testing {
int run_index_equivalent_benchmark(int argc, char **argv,
                                   std::span<const IndexTestCase> cases);
int run_native_benchmark_test(int argc, char **argv, std::string_view op) {
  if (op == "causal_conv1d")
    return run_causal_convolution_functional_test(
        argc, argv, make_causal_convolution_cases(), true);
  if (op == "concatenate" || op == "gen_index")
    return run_index_equivalent_benchmark(argc, argv, make_index_cases(op));
  if (op == "resample")
    return run_resample_functional_test(argc, argv, make_resample_cases(),
                                        true);
  if (op == "bn_finalize")
    return run_statistics_functional_test(
        argc, argv,
        make_bn_finalize_cases(),
        true);
  return run_extended_normalization_functional_test(
      argc, argv, make_extended_normalization_cases(std::string(op)), true);
}
template <class Cases> int skip_attention(const Cases &cases, std::string op) {
  const auto reason = iluvatar::validation::missing_graph_reference_reason();
  if (iluvatar::validation::CorexCudnnFlashAttentionDescriptor::
          symbols_available())
    throw std::runtime_error("CoreX Flash Attention is available; implement "
                             "its adapter before skipping");
  for (const auto &c : cases)
    iluvatar::validation::functional::emit_reference_skip(op, c.name, c.q,
                                                          reason);
  std::transform(op.begin(), op.end(), op.begin(),
                 [](unsigned char c) { return std::toupper(c); });
  std::cout << "FLAGDNN_" << op << "_BENCHMARK: SKIP cases=" << cases.size()
            << " comparable_executed=0 reference_skipped=" << cases.size()
            << '\n';
  return 77;
}
int run_attention_benchmark_test(int, char **, AttentionBenchmarkOperation op) {
  switch (op) {
  case AttentionBenchmarkOperation::kForward:
    return skip_attention(make_sdpa_benchmark_cases(), "sdpa");
  case AttentionBenchmarkOperation::kBackward:
    return skip_attention(make_sdpa_backward_benchmark_cases(),
                          "sdpa_backward");
  case AttentionBenchmarkOperation::kFp8Forward:
    return skip_attention(make_sdpa_fp8_benchmark_cases(), "sdpa_fp8");
  case AttentionBenchmarkOperation::kFp8Backward:
    return skip_attention(make_sdpa_fp8_backward_benchmark_cases(),
                          "sdpa_fp8_backward");
  }
  throw std::invalid_argument("unknown attention benchmark operation");
}
} // namespace flagdnn::testing
