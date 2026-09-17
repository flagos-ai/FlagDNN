/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/composite.hpp"
#include "convolution_runner_support.hpp"
#include "host_runner.hpp"
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
        if (c.left.data_type == FLAGDNN_DATA_INT32) {
          host::run_integer_case(
              pointwise, handle, stream,
              [&](flagdnn::Handle &h) {
                return build_flagdnn_add_square(h, c);
              },
              [&](auto a, auto b, bool predicate, auto alpha) {
                (void)predicate;
                return reference::cpu::pointwise_integer_reference(
                    FLAGDNN_POINTWISE_ADD, a,
                    reference::cpu::pointwise_integer_reference(
                        FLAGDNN_POINTWISE_MUL, b, b, false, 1),
                    false, alpha);
              });
          return;
        }
        host::run_case(
            c.name, pointwise.inputs, std::vector<TestTensor>{c.output},
            host::default_inputs(pointwise.inputs), handle, stream,
            [&](flagdnn::Handle &h) { return build_flagdnn_add_square(h, c); },
            [&](const host::Values &v) {
              std::vector<float> right(host::io::element_count(c.output));
              for (std::size_t i = 0; i < right.size(); ++i) {
                const float b =
                    v[1][host::pw::broadcast_index(i, c.right, c.output)];
                right[i] = b * b;
              }
              right =
                  host::io::decode(host::io::encode(right, c.output.data_type),
                                   c.output.data_type, right.size());
              for (std::size_t i = 0; i < right.size(); ++i)
                right[i] =
                    v[0][host::pw::broadcast_index(i, c.left, c.output)] +
                    right[i];
              return host::Values{right};
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
