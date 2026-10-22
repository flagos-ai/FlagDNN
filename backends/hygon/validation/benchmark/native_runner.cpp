/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "common/native_runner.hpp"
#include "native.hpp"

#include <stdexcept>

#include "common/causal_convolution.hpp"
#include "common/fp8_matmul.hpp"
#include "common/index.hpp"
#include "common/moe_matmul.hpp"
#include "common/normalization_extended.hpp"
#include "common/resample.hpp"
#include "common/statistics.hpp"
namespace flagdnn::testing {
int run_native_benchmark_test(int argc, char **argv,
                              std::string_view operation) {
  if (operation == "concatenate" || operation == "gen_index")
    return run_native_index_benchmark(argc, argv, operation);
  return run_native_extended_benchmark(argc, argv, operation);
}
} // namespace flagdnn::testing
