/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "backends/mthreads/validation/functional/paired.hpp"
#include "common/random.hpp"
namespace flagdnn::testing {
int run_rng_functional_test(int argc, char** argv,
                            std::span<const RngTestCase> cases,
                            bool benchmark) {
  return mthreads::run_paired_cases(
      argc, argv, cases, "FLAGDNN_RNG_CASE", "rng", build_flagdnn_rng,
      [](const RngTestCase&) { return std::vector<std::vector<float>>{}; },
      [](const RngTestCase&) -> std::unique_ptr<TestExecutable> {
        throw validation::mthreads::ReferenceUnsupported(
            "muDNN 3.1.5 has no standalone counter-based RNG operator");
      },
      [](const RngTestCase&, std::size_t) {
        return mthreads::Tolerance{0, 0};
      },
      benchmark);
}
}  // namespace flagdnn::testing
