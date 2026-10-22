/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/composite.hpp"
#include "convolution_runner_support.hpp"
#include "host_runner.hpp"
#include "reference/cpu/pointwise.hpp"
#include "pointwise_runner_support.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <functional>
#include <span>
#include <vector>

namespace flagdnn::testing {

int run_add_square_functional_test(int argc, char **argv,
                                   std::span<const AddSquareTestCase> cases) {
  namespace host = hygon_functional::host;
  return host::run_suite(
      argc, argv, cases, "add_square", false,
      [](const AddSquareTestCase &c, const auto &handle, auto &stream) {
        PointwiseTestCase pointwise;
        pointwise.name = c.name;
        pointwise.inputs = {c.left, c.right};
        pointwise.output = c.output;
        pointwise.mode = FLAGDNN_POINTWISE_ADD;
        pointwise.alpha = 1.0;

        host::run_case(
            c.name, pointwise.inputs, std::vector<TestTensor>{c.output},
            host::default_inputs(pointwise.inputs), handle, stream,
            [&](flagdnn::Handle &h) { return build_flagdnn_add_square(h, c); },
            [&](const host::Values &v) {
              return host::Values{reference::cpu::evaluate_add_square(
                  v[0], c.left.dimensions, v[1], c.right.dimensions,
                  c.output.dimensions, [&](std::span<const float> values) {
                    return host::io::decode(
                        host::io::encode(values, c.output.data_type),
                        c.output.data_type, values.size());
                  })};
            },
            c.absolute_tolerance, c.relative_tolerance);
      });
}

int run_conv_bias_relu_functional_test(
    int argc, char **argv, std::span<const ConvBiasReluTestCase> cases) {
  namespace support = hygon_functional::convolution;
  return support::run_suite(
      argc, argv, cases, "FLAGDNN_CONV_BIAS_RELU_FUNCTIONAL",
      "FLAGDNN_COMPOSITE_CASE", [](const ConvBiasReluTestCase &) {});
}

} // namespace flagdnn::testing
