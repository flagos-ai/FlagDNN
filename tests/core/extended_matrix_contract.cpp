/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include <flagdnn_frontend.h>
#include <iostream>
#include <stdexcept>
namespace fe = flagdnn_frontend;
using Graph = fe::graph::Graph;
namespace {
Graph::Tensor input(Graph& graph, std::int64_t uid, fe::DataType_t type,
                    std::vector<std::int64_t> shape) {
  auto strides = shape;
  std::int64_t stride = 1;
  for (std::size_t axis = shape.size(); axis > 0; --axis) {
    strides[axis - 1] = stride;
    stride *= shape[axis - 1];
  }
  return graph.tensor(fe::graph::Tensor_attributes()
                          .set_uid(uid)
                          .set_data_type(type)
                          .set_dim(shape)
                          .set_stride(strides));
}
void require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}
void output(const Graph::Tensor& value) {
  value->set_uid(100).set_output(true);
}
}  // namespace
int main() {
  try {
    for (const auto format :
         {fe::DataType_t::FP8_E4M3, fe::DataType_t::FP8_E5M2}) {
      for (int scale_mode = 0; scale_mode < 3; ++scale_mode) {
        Graph graph;
        auto attributes =
            fe::graph::Matmul_fp8_attributes().set_mxfp8(scale_mode == 2);
        if (scale_mode == 1)
          attributes.set_descale_a(input(graph, 3, fe::DataType_t::FLOAT, {1}))
              .set_descale_b(input(graph, 4, fe::DataType_t::FLOAT, {1}));
        if (scale_mode == 2)
          attributes
              .set_descale_a(
                  input(graph, 3, fe::DataType_t::FP8_E8M0, {2, 3, 2}))
              .set_descale_b(
                  input(graph, 4, fe::DataType_t::FP8_E8M0, {1, 2, 7}));
        const auto result =
            graph.matmul_fp8(input(graph, 1, format, {2, 3, 33}),
                             input(graph, 2, format, {1, 33, 7}), attributes);
        output(result);
        require(result->get_dim() == std::vector<std::int64_t>{2, 3, 7},
                "FP8 batch broadcast inference failed");
        require(result->get_data_type() == fe::DataType_t::FLOAT,
                "FP8 output default must be FP32");
        require(graph.validate().is_good(), "FP8/MXFP8 matmul graph rejected");
      }
      Graph graph;
      output(graph.matmul(input(graph, 1, format, {3, 33}),
                          input(graph, 2, format, {33, 7}),
                          fe::graph::Matmul_attributes()));
      require(graph.validate().is_good(), "plain FP8 matmul rejected");
    }
    for (const auto dtype :
         {fe::DataType_t::HALF, fe::DataType_t::BFLOAT16,
          fe::DataType_t::FP8_E4M3, fe::DataType_t::FP8_E5M2}) {
      for (int mode = 0; mode < 3; ++mode) {
        Graph graph;
        auto token = input(graph, 1, dtype, {1, mode == 1 ? 5 : 10, 17});
        auto weight = input(graph, 2, dtype, {3, 17, 7});
        auto offset = input(graph, 3, fe::DataType_t::INT32, {3, 1, 1});
        auto index =
            mode ? input(graph, 4, fe::DataType_t::INT32, {1, 10, 1}) : nullptr;
        auto ks = mode == 2 ? input(graph, 5, fe::DataType_t::INT32, {1, 10, 1})
                            : nullptr;
        output(graph.moe_grouped_matmul(
            token, weight, offset, index, ks,
            fe::graph::Moe_grouped_matmul_attributes()
                .set_mode(static_cast<fe::MoeGroupedMatmulMode_t>(mode))
                .set_top_k(2)));
        require(graph.validate().is_good(), "MoE forward mode rejected");
      }
      Graph graph;
      output(graph.moe_grouped_matmul_bwd(
          input(graph, 1, dtype, {1, 10, 7}),
          input(graph, 2, dtype, {1, 10, 17}),
          input(graph, 3, fe::DataType_t::INT32, {3, 1, 1}),
          fe::graph::Moe_grouped_matmul_bwd_attributes()));
      require(graph.validate().is_good(), "MoE weight gradient rejected");
    }
    for (const auto precision :
         {fe::InputPrecision_t::IEEE, fe::InputPrecision_t::TF32}) {
      Graph graph;
      output(graph.matmul(
          input(graph, 1, fe::DataType_t::FLOAT, {3, 17}),
          input(graph, 2, fe::DataType_t::FLOAT, {17, 7}),
          fe::graph::Matmul_attributes().set_input_precision(precision)));
      require(graph.validate().is_good(), "explicit FP32 precision rejected");
    }
    {
      Graph graph;
      output(graph.matmul(input(graph, 1, fe::DataType_t::HALF, {3, 17}),
                          input(graph, 2, fe::DataType_t::HALF, {17, 7}),
                          fe::graph::Matmul_attributes().set_input_precision(
                              fe::InputPrecision_t::TF32)));
      require(graph.validate().is_bad(),
              "TF32 precision accepted on FP16 storage");
    }
    {
      Graph graph;
      auto attrs =
          fe::graph::Matmul_fp8_attributes()
              .set_mxfp8(true)
              .set_descale_a(input(graph, 3, fe::DataType_t::FP8_E8M0, {3, 1}))
              .set_descale_b(input(graph, 4, fe::DataType_t::FP8_E8M0, {2, 7}));
      output(graph.matmul_fp8(
          input(graph, 1, fe::DataType_t::FP8_E4M3, {3, 33}),
          input(graph, 2, fe::DataType_t::FP8_E4M3, {33, 7}), attrs));
      require(graph.validate().is_bad(), "undersized MXFP8 scale accepted");
    }
    {
      Graph graph;
      output(graph.moe_grouped_matmul_bwd(
          input(graph, 1, fe::DataType_t::HALF, {1, 10, 7}),
          input(graph, 2, fe::DataType_t::HALF, {1, 9, 17}),
          input(graph, 3, fe::DataType_t::INT32, {3, 1, 1}),
          fe::graph::Moe_grouped_matmul_bwd_attributes()));
      require(graph.validate().is_bad(),
              "MoE mismatched token counts accepted");
    }
    for (const auto type : {fe::DataType_t::FLOAT, fe::DataType_t::HALF,
                            fe::DataType_t::BFLOAT16}) {
      for (bool rms : {false, true}) {
        Graph graph;
        auto x = input(graph, 1, type, {2, 3, 1});
        auto scale = input(graph, 2, type, {1});
        auto bias = input(graph, 3, type, {1});
        auto epsilon =
            graph.tensor(1.0e-5F, fe::graph::ScalarType::COMPILE_TIME_CONST);
        if (rms)
          output(graph.rmsnorm(
              x, scale,
              fe::graph::Rmsnorm_attributes()
                  .set_bias(bias)
                  .set_epsilon(epsilon)
                  .set_forward_phase(fe::NormFwdPhase_t::TRAINING))[0]);
        else
          output(graph.layernorm(
              x, scale, bias,
              fe::graph::Layernorm_attributes()
                  .set_epsilon(1.0e-5F)
                  .set_forward_phase(fe::NormFwdPhase_t::TRAINING))[0]);
        require(graph.validate().is_good(),
                "unit-width normalization rejected");
      }
      for (int variant = 0; variant < 4; ++variant) {
        Graph graph;
        auto attrs = fe::graph::Causal_conv1d_attributes().set_dilation(
            variant == 1 ? 0 : 2);
        if (variant == 2) attrs.set_bias(input(graph, 3, type, {4}));
        auto y = graph.causal_conv1d(input(graph, 1, type, {2, 3, 7}),
                                     input(graph, 2, type, {3, 5}), attrs);
        if (variant == 3) y->set_data_type(fe::DataType_t::INT32);
        output(y);
        require(graph.validate().is_good() == (variant == 0),
                "causal convolution shape/dtype validation failed");
      }
      for (int variant = 0; variant < 4; ++variant) {
        Graph graph;
        auto attrs =
            fe::graph::Resample_attributes().set_window({2, 3}).set_stride(
                {1, 2});
        if (variant == 1)
          attrs.set_resampling_mode(fe::ResampleMode_t::AVGPOOL_INCLUDE_PADDING)
              .set_generate_index(true);
        auto y = graph.resample(input(graph, 1, type, {2, 3, 7, 9}), attrs)[0];
        if (variant == 2) y->set_data_type(fe::DataType_t::INT32);
        if (variant == 3) y->set_dim({2, 3, 7, 7}).set_stride({147, 49, 7, 1});
        output(y);
        require(graph.validate().is_good() == (variant == 0),
                "resample shape/dtype validation failed");
      }
    }
    std::cout << "FP8, MXFP8, MoE and explicit precision contracts PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
