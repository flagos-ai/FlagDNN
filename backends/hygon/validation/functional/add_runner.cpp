/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "common/add.hpp"
#include "host_runner.hpp"
#include "reference/cpu/pointwise.hpp"
namespace flagdnn::testing {
int run_add_functional_test(int argc, char **argv,
                            std::span<const AddTestCase> cases) {
  namespace host = hygon_functional::host;
  return host::run_suite(
      argc, argv, cases, "add", false,
      [](const AddTestCase &c, const auto &handle, auto &stream) {
        PointwiseTestCase pointwise;
        pointwise.name = c.name;
        pointwise.inputs = {c.left, c.right};
        pointwise.output = c.output;
        pointwise.mode = FLAGDNN_POINTWISE_ADD;
        pointwise.alpha = c.alpha;

        host::run_case(
            c.name, pointwise.inputs, std::vector<TestTensor>{c.output},
            host::default_inputs(pointwise.inputs), handle, stream,
            [&](flagdnn::Handle &h) { return build_flagdnn_add(h, c); },
            [&](const host::Values &v) {
              return host::Values{
                  reference::cpu::evaluate_binary_pointwise_with_alpha(
                      FLAGDNN_POINTWISE_ADD, v[0], c.left.dimensions,
                      v[1], c.right.dimensions, c.output.dimensions, c.alpha)};
            },
            c.absolute_tolerance, c.relative_tolerance);
      });
}
} // namespace flagdnn::testing
