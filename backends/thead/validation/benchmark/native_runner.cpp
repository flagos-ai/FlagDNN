// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#include "common/native_runner.hpp"

#include <stdexcept>

#include "common/normalization_extended.hpp"
#include "common/resample.hpp"
#include "common/statistics.hpp"
namespace flagdnn::testing {
int run_native_benchmark_test(int argc, char** argv,
                              std::string_view operation) {
  if (operation == "bn_finalize")
    return run_statistics_functional_test(argc, argv, make_bn_finalize_cases(),
                                          true);
  if (operation == "genstats")
    return run_statistics_functional_test(argc, argv, make_genstats_cases(),
                                          true);
  if (operation == "resample")
    return run_resample_functional_test(argc, argv, make_resample_cases(),
                                        true);
  if (operation == "batchnorm_backward")
    return run_extended_normalization_functional_test(
        argc, argv, make_extended_normalization_cases(std::string(operation)),
        true);
  throw std::invalid_argument("unregistered THead native benchmark operator");
}
}  // namespace flagdnn::testing
