/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "common/reduction.hpp"
#include "host_runner.hpp"
#include "reference/cpu/matrix.hpp"
namespace flagdnn::testing {
int run_reduction_functional_test(int argc, char **argv,
                                  std::span<const ReductionTestCase> cases) {
  namespace host = hygon_functional::host;
  return host::run_suite(
      argc, argv, cases, "reduction", false,
      [](const ReductionTestCase &c, const auto &handle, auto &stream) {
        validate_reduction_case(c);
        host::run_case(
            c.name, std::vector<TestTensor>{c.input},
            std::vector<TestTensor>{c.output},
            host::Values{reduction_host_input(c)}, handle, stream,
            [&](flagdnn::Handle &h) { return build_flagdnn_reduction(h, c); },
            [&](const host::Values &v) {
              return host::Values{reference::cpu::evaluate_reduction(
                  {c.input.dimensions, c.axis, c.mode}, v[0])};
            },
            c.absolute_tolerance, c.relative_tolerance);
      });
}
} // namespace flagdnn::testing
