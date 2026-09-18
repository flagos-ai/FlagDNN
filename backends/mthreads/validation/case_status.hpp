/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#ifndef FLAGDNN_MTHREADS_CASE_STATUS_HPP_
#define FLAGDNN_MTHREADS_CASE_STATUS_HPP_

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace flagdnn::validation::mthreads {

class ReferenceUnsupported : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

inline bool benchmark_enabled() {
  const char* value = std::getenv("FLAGDNN_MTHREADS_NATIVE_BENCHMARK");
  return value != nullptr && std::string_view(value) == "1";
}

inline int report_cases(std::string suite, std::size_t executed,
                        std::size_t skipped) {
  if (executed + skipped == 0) throw std::runtime_error("no cases selected");
  if (benchmark_enabled()) {
    const auto suffix = suite.rfind("_FUNCTIONAL");
    if (suffix != std::string::npos) suite.replace(suffix, 11, "_BENCHMARK");
  }
  std::cout << suite << ": " << (executed == 0 ? "SKIP" : "PASS")
            << " cases=" << executed + skipped << " executed=" << executed
            << " skipped=" << skipped << '\n';
  return executed == 0 ? 77 : 0;
}

inline void report_skip(std::string_view name,
                        const ReferenceUnsupported& error) {
  std::cout << "[skip] case=" << name
            << " provider=mudnn reason=" << error.what() << '\n';
}
}  // namespace flagdnn::validation::mthreads
#endif
