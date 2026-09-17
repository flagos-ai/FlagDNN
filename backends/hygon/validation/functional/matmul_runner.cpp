/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "common/matmul.hpp"
#include "host_runner.hpp"
#include "precision.hpp"
#include "reference/cpu/matrix.hpp"
#include "tensor_runner_support.hpp"
namespace flagdnn::testing {
int run_matmul_functional_test(int argc, char **argv,
                               std::span<const MatmulTestCase> cases) {
  namespace host = hygon_functional::host;
  namespace hv = validation::hygon;
  namespace support = hygon_functional::tensor;
  return host::pw::run_suite(
      argc, argv, "matmul", "FLAGDNN_MATMUL_FUNCTIONAL",
      [&](const auto &handle, hv::Stream &stream, auto &matched, auto &executed,
          auto &skipped) {
        const char *filter = std::getenv("FLAGDNN_MATMUL_CASE");
        for (const auto &c : cases) {
          if (c.input_precision != hv::selected_input_precision() ||
              c.output.dimensions.size() > 3)
            continue;
          if (filter && c.name.find(filter) == std::string::npos)
            continue;
          ++matched;
          if (c.input_precision == 2) {
            support::emit_skip(
                "matmul", c.name,
                "Hygon backend does not implement TF32 arithmetic",
                support::matmul_reference_tensors(c));
            ++skipped;
            continue;
          }
          validate_matmul_case(c);
          const std::vector<TestTensor> inputs{c.a, c.b};
          host::Values values;
          for (std::size_t input = 0; input < inputs.size(); ++input) {
            std::vector<float> logical(host::io::element_count(inputs[input]));
            for (std::size_t i = 0; i < logical.size(); ++i)
              logical[i] = float(int((i * 17 + input * 11) % 41) - 20) /
                           float(13 + input);
            values.push_back(std::move(logical));
          }
          // Hygon's default FP32 matmul uses IEEE arithmetic. The shared
          // oracle's default follows NVIDIA's shape-dependent TF32 policy.
          const int reference_precision =
              c.input_precision ? c.input_precision : 1;
          host::run_case(
              c.name, inputs, std::vector<TestTensor>{c.output}, values, handle,
              stream,
              [&](flagdnn::Handle &h) { return build_flagdnn_matmul(h, c); },
              [&](const host::Values &v) {
                return host::Values{reference::cpu::evaluate_matmul(
                    {c.a.dimensions, c.b.dimensions, c.output.dimensions,
                     c.a.data_type, reference_precision},
                    v[0], v[1])};
              },
              c.absolute_tolerance, c.relative_tolerance);
          ++executed;
        }
      });
}
} // namespace flagdnn::testing
