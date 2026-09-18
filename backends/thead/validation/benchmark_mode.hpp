// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef FLAGDNN_THEAD_VALIDATION_BENCHMARK_MODE_HPP_
#define FLAGDNN_THEAD_VALIDATION_BENCHMARK_MODE_HPP_

#include <cstdlib>
#include <iostream>
#include <string_view>

namespace flagdnn::validation::thead {
// Prepare the complete benchmark graph/cache and compare both implementations
// without collecting timings. This mode is only for parallel qualification;
// tools/run_tests.py rejects its output as an incomplete performance report.
inline bool benchmark_validation_only(std::string_view name) {
  const char* mode = std::getenv("FLAGDNN_THEAD_VALIDATE_BENCHMARKS_ONLY");
  if (mode == nullptr || std::string_view(mode) != "1") return false;
  std::cout << "VALIDATION_ONLY case=" << name << " timing=not_collected\n";
  return true;
}
}  // namespace flagdnn::validation::thead
#endif
