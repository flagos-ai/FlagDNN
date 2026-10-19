/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "cpu_cudnn_benchmark.hpp"
#include "common/pointwise.hpp"
#include "common/composite.hpp"
#include "reference/cpu/benchmark.hpp"

namespace flagdnn::reference::tests {
b::BenchmarkCase formula_case(b::Operation operation,
    const std::vector<t::TestTensor>& inputs, const std::vector<t::TestTensor>& outputs) {
  b::BenchmarkCase result;
  result.operation = operation;
  result.output_count = outputs.size();
  auto tensors = inputs;
  tensors.insert(tensors.end(), outputs.begin(), outputs.end());
  for (const auto& s : tensors)
    result.tensors.push_back({s.uid, s.data_type, s.dimensions, s.strides, s.binding_byte_offset});
  return result;
}
void check_formula(const b::BenchmarkCase& specification, const Result& result,
                    double atol, double rtol) {
  const auto expected = cpu::evaluate_benchmark(specification, result.inputs, quantize);
  if (expected.size() != result.outputs.size()) throw std::runtime_error("formula output count");
  for (std::size_t i = 0; i < expected.size(); ++i)
    compare(result.outputs[i], quantize(expected[i], b::output_tensor(specification, i).data_type),
            atol, rtol);
}
void formulas(Suite& suite) {
  // This filename mirrors the CPU module. These are correctness checks only:
  // each cuDNN operation executes once, with no warmup, timer or speed result.
  for (const auto mode : {
      FLAGDNN_POINTWISE_SQRT, FLAGDNN_POINTWISE_ERF, FLAGDNN_POINTWISE_IDENTITY,
      FLAGDNN_POINTWISE_EXP, FLAGDNN_POINTWISE_LOG, FLAGDNN_POINTWISE_NEG,
      FLAGDNN_POINTWISE_ABS, FLAGDNN_POINTWISE_CEIL, FLAGDNN_POINTWISE_COS,
      FLAGDNN_POINTWISE_FLOOR, FLAGDNN_POINTWISE_RSQRT, FLAGDNN_POINTWISE_SIN,
      FLAGDNN_POINTWISE_TAN, FLAGDNN_POINTWISE_RECIPROCAL, FLAGDNN_POINTWISE_RELU_FWD,
      FLAGDNN_POINTWISE_TANH_FWD, FLAGDNN_POINTWISE_ELU_FWD,
      FLAGDNN_POINTWISE_SOFTPLUS_FWD, FLAGDNN_POINTWISE_SWISH_FWD,
      FLAGDNN_POINTWISE_LOGICAL_NOT, FLAGDNN_POINTWISE_LOGICAL_AND,
      FLAGDNN_POINTWISE_LOGICAL_OR, FLAGDNN_POINTWISE_TANH_BWD,
      FLAGDNN_POINTWISE_GELU_BWD, FLAGDNN_POINTWISE_GELU_APPROX_TANH_BWD}) {
    const bool logical = mode == FLAGDNN_POINTWISE_LOGICAL_NOT ||
                         mode == FLAGDNN_POINTWISE_LOGICAL_AND ||
                         mode == FLAGDNN_POINTWISE_LOGICAL_OR;
    const bool binary = mode == FLAGDNN_POINTWISE_LOGICAL_AND ||
        mode == FLAGDNN_POINTWISE_LOGICAL_OR || mode == FLAGDNN_POINTWISE_TANH_BWD ||
        mode == FLAGDNN_POINTWISE_GELU_BWD || mode == FLAGDNN_POINTWISE_GELU_APPROX_TANH_BWD;
    for (const auto type : {FLAGDNN_DATA_FLOAT32, FLAGDNN_DATA_FLOAT16, FLAGDNN_DATA_BFLOAT16}) {
      if (logical && type != FLAGDNN_DATA_FLOAT32) continue;
      const auto dtype = logical ? FLAGDNN_DATA_BOOLEAN : type;
      t::PointwiseTestCase c;
      c.name = "cpu_formula_" + std::to_string(mode) + "_dtype_" + std::to_string(dtype);
      c.mode = mode;
      c.inputs = {tensor(1, {1, 256, 1, 1}, dtype)};
      if (binary) c.inputs.push_back(tensor(2, {1, 256, 1, 1}, dtype));
      c.output = tensor(3, {1, 256, 1, 1}, dtype);
      if (logical && binary) {
        for (auto& input : c.inputs) input = tensor(input.uid, {2, 4, 32}, dtype);
        c.output = tensor(3, {2, 4, 32}, dtype);
      }
      c.input_domains.assign(c.inputs.size(), t::PointwiseInputDomain::kReal);
        suite.run(c.name, [&] {
        auto values = inputs_for(c.inputs);
        for (std::size_t i = 0; i < values.size(); ++i)
          for (std::size_t j = 0; j < values[i].size(); ++j) {
            if (logical) values[i][j] = float((j / (i + 1)) % 2);
            if (mode == FLAGDNN_POINTWISE_LOG || mode == FLAGDNN_POINTWISE_SQRT ||
                mode == FLAGDNN_POINTWISE_RSQRT || mode == FLAGDNN_POINTWISE_RECIPROCAL)
              values[i][j] = 0.1F + std::abs(values[i][j]);
            if (mode == FLAGDNN_POINTWISE_TANH_BWD && i == 1)
              values[i][j] = std::tanh(values[i][j]);
          }
        auto spec = formula_case(b::Operation::kPointwise, c.inputs, {c.output});
        spec.pointwise_mode = c.mode;
        const double tol = type == FLAGDNN_DATA_BFLOAT16 ? 8e-3 :
                           type == FLAGDNN_DATA_FLOAT16 ? 1e-3 : 5e-5;
        const auto r = suite.execute(c.inputs, {c.output}, t::build_pointwise_reference(c), values);
        check_formula(spec, r, tol, tol);
        if (mode == FLAGDNN_POINTWISE_RELU_FWD) {
          spec.operation = b::Operation::kRelu;
          check_formula(spec, r, tol, tol);
        }
      });
    }
  }
}
}  // namespace flagdnn::reference::tests
