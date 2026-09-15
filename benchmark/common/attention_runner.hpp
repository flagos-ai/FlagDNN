/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#ifndef FLAGDNN_BENCHMARK_COMMON_ATTENTION_RUNNER_HPP_
#define FLAGDNN_BENCHMARK_COMMON_ATTENTION_RUNNER_HPP_

namespace flagdnn::testing {
enum class AttentionBenchmarkOperation { kForward, kBackward, kFp8Forward, kFp8Backward };
// Implemented by the platform's shared Attention validation adapter.
int run_attention_benchmark_test(int argc, char** argv,
                                 AttentionBenchmarkOperation operation);
}  // namespace flagdnn::testing
#endif
