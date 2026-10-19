/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "cpu_cudnn_benchmark.hpp"
#include "common/pointwise.hpp"
#include "common/composite.hpp"
#include "reference/cpu/pointwise.hpp"

#include <algorithm>

namespace flagdnn::reference::tests {
void pointwise(Suite& suite) {
  const std::vector modes = {
      FLAGDNN_POINTWISE_ADD, FLAGDNN_POINTWISE_SUB, FLAGDNN_POINTWISE_MUL,
      FLAGDNN_POINTWISE_DIV, FLAGDNN_POINTWISE_MIN, FLAGDNN_POINTWISE_MAX,
      FLAGDNN_POINTWISE_MOD, FLAGDNN_POINTWISE_POW, FLAGDNN_POINTWISE_CMP_EQ,
      FLAGDNN_POINTWISE_CMP_NEQ, FLAGDNN_POINTWISE_CMP_GT,
      FLAGDNN_POINTWISE_CMP_GE, FLAGDNN_POINTWISE_CMP_LT, FLAGDNN_POINTWISE_CMP_LE};
  for (const auto mode : modes) {
    for (const auto type : {FLAGDNN_DATA_FLOAT32, FLAGDNN_DATA_FLOAT16,
                            FLAGDNN_DATA_BFLOAT16}) {
      for (bool broadcast : {false, true}) {
        t::PointwiseTestCase c;
        c.name = "binary_" + std::to_string(mode) + "_dtype_" +
            std::to_string(type) + (broadcast ? "_broadcast" : "_same_shape");
        c.mode = mode;
        c.inputs = {tensor(1, {2, 8, 3, 4}, type),
                    tensor(2, broadcast ? std::vector<std::int64_t>{1, 8, 1, 1} :
                                          std::vector<std::int64_t>{2, 8, 3, 4}, type)};
        const bool comparison = mode >= FLAGDNN_POINTWISE_CMP_EQ &&
                                mode <= FLAGDNN_POINTWISE_CMP_LE;
        c.output = tensor(3, {2, 8, 3, 4}, comparison ? FLAGDNN_DATA_BOOLEAN : type);
        if (broadcast && (mode == FLAGDNN_POINTWISE_ADD || mode == FLAGDNN_POINTWISE_SUB))
          c.alpha = -2;
        for (auto& in : c.inputs)
          if (in.dimensions == std::vector<std::int64_t>{2, 8, 3, 4}) in.strides = {96, 1, 32, 8};
        if (c.output.dimensions == std::vector<std::int64_t>{2, 8, 3, 4})
          c.output.strides = {96, 1, 32, 8};
        c.input_domains.assign(c.inputs.size(), t::PointwiseInputDomain::kReal);
        suite.run(c.name, [&] {
          auto values = inputs_for(c.inputs);
          for (std::size_t i = 0; i < values.size(); ++i)
            for (std::size_t j = 0; j < values[i].size(); ++j) {
              if (mode == FLAGDNN_POINTWISE_DIV || mode == FLAGDNN_POINTWISE_MOD) {
                if (i == 1 && std::abs(values[i][j]) < 0.25F) values[i][j] = 2;
              } else if (mode == FLAGDNN_POINTWISE_POW)
                values[i][j] = i == 0 ? 1 + float(j % 3) : float(j % 4);
            }
          const double tol = comparison ? 0 :
                             type == FLAGDNN_DATA_BFLOAT16 ? 8e-3 :
                             type == FLAGDNN_DATA_FLOAT16 ? 1e-3 : 2e-5;
          const auto r = suite.check(c.inputs, {c.output}, t::build_pointwise_reference(c),
              [&](const Arrays& x) {

                auto out = cpu::evaluate_binary_pointwise_with_alpha(mode,
                    x[0], c.inputs[0].dimensions, x[1], c.inputs[1].dimensions,
                    c.output.dimensions, c.alpha);
                if (c.alpha == 1)
                  compare(out, cpu::evaluate_binary_pointwise(mode,
                      x[0], c.inputs[0].dimensions, x[1], c.inputs[1].dimensions,
                      c.output.dimensions), 0, 0);
                return Arrays{out};
              }, tol, tol, values);
          auto spec = formula_case(b::Operation::kPointwise, c.inputs, {c.output});
          spec.pointwise_mode = mode;
          spec.add_alpha = c.alpha;
          check_formula(spec, r, tol, tol);
          if (mode == FLAGDNN_POINTWISE_ADD) {
            spec.operation = b::Operation::kAdd;
            check_formula(spec, r, tol, tol);
          }
        });
      }
    }
  }
  for (const auto type : {FLAGDNN_DATA_FLOAT32, FLAGDNN_DATA_FLOAT16, FLAGDNN_DATA_BFLOAT16}) {
    for (const auto mode : {FLAGDNN_POINTWISE_RELU_BWD, FLAGDNN_POINTWISE_ELU_BWD,
                            FLAGDNN_POINTWISE_SOFTPLUS_BWD, FLAGDNN_POINTWISE_SWISH_BWD,
                            FLAGDNN_POINTWISE_SIGMOID_BWD, FLAGDNN_POINTWISE_SIGMOID_FWD,
                            FLAGDNN_POINTWISE_GELU_FWD, FLAGDNN_POINTWISE_GELU_APPROX_TANH_FWD,
                            FLAGDNN_POINTWISE_IDENTITY}) {
      for (bool attributes : {false, true}) {
        if (attributes && !cpu::supports_activation_backward(mode)) continue;
        t::PointwiseTestCase c;
        c.name = "activation_" + std::to_string(mode) + "_dtype_" +
                 std::to_string(type) + (attributes ? "_attributes" : "_default");
        c.mode = mode;
        c.inputs = {tensor(1, {1, 256, 1, 1}, type)};
        const bool backward = cpu::supports_activation_backward(mode) ||
                              mode == FLAGDNN_POINTWISE_SIGMOID_BWD;
        if (backward) c.inputs.push_back(tensor(2, {1, 256, 1, 1}, type));
        c.output = tensor(3, {1, 256, 1, 1}, type);
        if (attributes) {
          c.attributes.flags = mode == FLAGDNN_POINTWISE_RELU_BWD ?
              FLAGDNN_POINTWISE_ATTRIBUTE_RELU_LOWER_CLIP |
              FLAGDNN_POINTWISE_ATTRIBUTE_RELU_UPPER_CLIP |
              FLAGDNN_POINTWISE_ATTRIBUTE_RELU_LOWER_CLIP_SLOPE :
              mode == FLAGDNN_POINTWISE_ELU_BWD ? FLAGDNN_POINTWISE_ATTRIBUTE_ELU_ALPHA :
              mode == FLAGDNN_POINTWISE_SOFTPLUS_BWD ? FLAGDNN_POINTWISE_ATTRIBUTE_SOFTPLUS_BETA :
              mode == FLAGDNN_POINTWISE_SWISH_BWD ? FLAGDNN_POINTWISE_ATTRIBUTE_SWISH_BETA : 0;
          c.attributes.relu_lower_clip = 0;
          c.attributes.relu_upper_clip = 2;
          c.attributes.relu_lower_clip_slope = 0.125;
          c.attributes.elu_alpha = 1.5;
          c.attributes.softplus_beta = 0.75;
          c.attributes.swish_beta = 1.25;
        }
        c.input_domains.assign(c.inputs.size(), t::PointwiseInputDomain::kReal);
        suite.run(c.name, [&] {
          auto values = inputs_for(c.inputs);
          for (std::size_t j = 0; j < values.back().size(); ++j)
            values.back()[j] = (float(j) - 128) / 16;
          const double tol = type == FLAGDNN_DATA_BFLOAT16 ? 8e-3 :
                             type == FLAGDNN_DATA_FLOAT16 ? 1e-3 : 5e-5;
          auto r = suite.check(c.inputs, {c.output}, t::build_pointwise_reference(c),
              [&](const Arrays& x) {
                if (cpu::supports_activation_backward(mode))
                  return Arrays{cpu::evaluate_activation_backward(mode, x[0], x[1], c.attributes)};
                if (mode == FLAGDNN_POINTWISE_IDENTITY) {
                  auto bytes = cpu::evaluate_identity_bytes(io::encode(x[0], type, encoding));
                  return Arrays{io::decode(bytes, type, x[0].size(), encoding)};
                }
                Values out(x[0].size());
                for (std::size_t j = 0; j < out.size(); ++j) {
                  if (mode == FLAGDNN_POINTWISE_SIGMOID_BWD)
                    out[j] = cpu::sigmoid_backward(x[0][j], x[1][j]);
                  else if (mode == FLAGDNN_POINTWISE_SIGMOID_FWD)
                    out[j] = cpu::sigmoid(x[0][j]);
                  else if (mode == FLAGDNN_POINTWISE_GELU_FWD)
                    out[j] = cpu::gelu_exact(x[0][j]);
                  else out[j] = cpu::gelu_approximate_tanh(x[0][j]);
                }
                return Arrays{out};
              }, tol, tol, values);
          auto spec = formula_case(b::Operation::kPointwise, c.inputs, {c.output});
          spec.pointwise_mode = mode;
          spec.pointwise_attributes = c.attributes;
          check_formula(spec, r, tol, tol);
          if (mode == FLAGDNN_POINTWISE_GELU_FWD) {
            Values out = r.inputs[0];
            for (auto& v : out) v = cpu::gelu_accurate_logit(v);
            compare(r.outputs[0], quantize(out, type), tol, tol);
          }
        });
      }
    }
  }
  for (const auto type : {FLAGDNN_DATA_FLOAT32, FLAGDNN_DATA_FLOAT16,
                          FLAGDNN_DATA_BFLOAT16}) {
    suite.run("add_square_dtype_" + std::to_string(type), [&] {
      t::AddSquareTestCase c{"add_square", tensor(1, {2, 8, 3, 4}, type),
          tensor(2, {2, 8, 3, 4}, type), tensor(3, {2, 8, 3, 4}, type)};
      c.left.strides = c.right.strides = c.output.strides = {96, 1, 32, 8};
      const auto tol = type == FLAGDNN_DATA_BFLOAT16 ? 8e-3 : 1e-3;
      const auto r = suite.check({c.left, c.right}, {c.output}, t::build_add_square_reference(c),
          [&](const Arrays& x) {

            return Arrays{cpu::evaluate_add_square(x[0], c.left.dimensions,
                x[1], c.right.dimensions, c.output.dimensions,
                [&](std::span<const float> v) { return quantize(v, type); })};
          }, tol, tol);
      auto spec = formula_case(b::Operation::kGraph, {c.left, c.right}, {c.output});
      auto intermediate = spec.tensors[1]; intermediate.uid = 10;
      spec.graph.intermediates = {intermediate};
      b::GraphNodeSpec square, add;
      square.input_uids = {2, 2}; square.output_uid = 10;
      square.pointwise_mode = FLAGDNN_POINTWISE_MUL;
      add.input_uids = {1, 10}; add.output_uid = 3;
      add.pointwise_mode = FLAGDNN_POINTWISE_ADD;
      spec.graph.nodes = {square, add};
      check_formula(spec, r, tol, tol);
    });
  }
}
}  // namespace flagdnn::reference::tests
