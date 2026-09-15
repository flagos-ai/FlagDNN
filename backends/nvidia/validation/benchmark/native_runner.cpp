/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "common/native_runner.hpp"

#include <stdexcept>

#include "common/causal_convolution.hpp"
#include "common/fp8_matmul.hpp"
#include "common/index.hpp"
#include "common/moe_matmul.hpp"
#include "common/normalization_extended.hpp"
#include "common/resample.hpp"
#include "common/statistics.hpp"
namespace flagdnn::testing {
int run_native_benchmark_test(int argc, char** argv,
                              std::string_view operation) {
  if (operation == "bn_finalize")
    return run_statistics_functional_test(argc, argv, make_bn_finalize_cases(),
                                          true);
  if (operation == "resample")
    return run_resample_functional_test(argc, argv, make_resample_cases(),
                                        true);
  if (operation == "causal_conv1d")
    return run_causal_convolution_functional_test(
        argc, argv, make_causal_convolution_cases(), true);
  if (operation == "matmul_fp8")
    return run_fp8_matmul_functional_test(argc, argv, make_fp8_matmul_cases(),
                                          true);
  if (operation == "moe_grouped_matmul")
    return run_moe_matmul_functional_test(argc, argv,
                                          make_moe_matmul_cases(false), true);
  if (operation == "concatenate" || operation == "gen_index")
    return run_index_functional_test(argc, argv, make_index_cases(operation),
                                     true);
  return run_extended_normalization_functional_test(
      argc, argv, make_extended_normalization_cases(std::string(operation)),
      true);
}
}  // namespace flagdnn::testing
