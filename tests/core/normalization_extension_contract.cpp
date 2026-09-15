/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include <flagdnn_frontend.h>

#include <array>
#include <iostream>
#include <stdexcept>
namespace fe = flagdnn_frontend;
using Graph = fe::graph::Graph;
using Tensor = Graph::Tensor;
namespace {
Tensor input(Graph& graph, std::int64_t uid, fe::DataType_t type,
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
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
void outputs(std::span<const Tensor> tensors) {
  std::int64_t uid = 100;
  for (const auto& tensor : tensors)
    if (tensor) tensor->set_uid(uid++).set_output(true);
}
}  // namespace
int main() {
  try {
    for (const auto type : {fe::DataType_t::FLOAT, fe::DataType_t::HALF,
                            fe::DataType_t::BFLOAT16}) {
      {
        Graph graph;
        const auto x = input(graph, 1, type, {2, 3, 7}),
                   scale = input(graph, 2, type, {1, 3, 1}),
                   bias = input(graph, 3, type, {1, 3, 1});
        const auto forward = graph.instancenorm(x, scale, bias);
        const auto backward = graph.instancenorm_backward(
            input(graph, 4, type, {2, 3, 7}), x, scale,
            fe::graph::Instancenorm_backward_attributes()
                .set_saved_mean_and_inv_variance(forward[1], forward[2]));
        outputs(backward);
        require(forward[1]->get_dim() == std::vector<std::int64_t>{2, 3, 1},
                "instance statistics shape invalid");
        require(graph.validate().is_good(),
                "instance forward/backward graph rejected");
      }
      {
        Graph graph;
        const auto x = input(graph, 1, type, {2, 3, 7}),
                   scale = input(graph, 2, type, {2, 1, 7}),
                   bias = input(graph, 3, type, {2, 1, 7});
        const auto forward = graph.adalayernorm(x, scale, bias);
        const auto backward = graph.adalayernorm_backward(
            input(graph, 4, type, {2, 3, 7}), x, scale,
            fe::graph::AdaLayernorm_backward_attributes()
                .set_saved_mean_and_inv_variance(forward[1], forward[2]));
        outputs(backward);
        require(forward[1]->get_dim() == std::vector<std::int64_t>{2, 3, 1},
                "adaptive statistics incorrectly reduce batch");
        require(graph.validate().is_good(),
                "adaptive forward/backward graph rejected");
      }
      {
        Graph graph;
        const auto x = input(graph, 1, type, {2, 3, 7}),
                   scale = input(graph, 2, type, {1, 1, 7});
        const auto result = graph.rmsnorm_backward(
            input(graph, 3, type, {2, 3, 7}), x, scale,
            input(graph, 4, fe::DataType_t::FLOAT, {2, 3, 1}),
            fe::graph::Rmsnorm_backward_attributes().has_dbias(false));
        outputs(result);
        require(!result[2], "RMS backward optional bias gradient exposed");
        require(result[1]->get_data_type() == fe::DataType_t::FLOAT,
                "affine gradient default is not FP32");
        require(graph.validate().is_good(),
                "RMS backward without external bias gradient rejected");
      }
      {
        Graph graph;
        const auto x = input(graph, 1, type, {2, 3, 7}),
                   scale = input(graph, 2, fe::DataType_t::FLOAT, {1, 3, 1}),
                   bias = input(graph, 3, fe::DataType_t::FLOAT, {1, 3, 1});
        const auto forward = graph.instancenorm(x, scale, bias);
        const auto backward = graph.instancenorm_backward(
            input(graph, 4, type, {2, 3, 7}), x, scale,
            fe::graph::Instancenorm_backward_attributes()
                .set_saved_mean_and_inv_variance(forward[1], forward[2]));
        outputs(backward);
        require(graph.validate().is_good(),
                "FP32 affine parameters with floating activations rejected");
      }
      {
        Graph graph;
        const auto stats = graph.genstats(input(graph, 1, type, {2, 3, 7}));
        const auto result = graph.bn_finalize(
            stats[0], stats[1], input(graph, 2, type, {1, 3, 1}),
            input(graph, 3, type, {1, 3, 1}),
            graph.tensor(1.0e-5F, fe::graph::ScalarType::COMPILE_TIME_CONST),
            graph.tensor(14.0F, fe::graph::ScalarType::COMPILE_TIME_CONST));
        outputs(result);
        require(!result[4] && !result[5],
                "BN finalize optional outputs exposed");
        require(graph.validate().is_good(),
                "genstats/bn_finalize graph rejected");
      }
      {
        Graph graph;
        const auto x = input(graph, 1, type, {2, 3, 7, 16}),
                   freqs = input(graph, 2, fe::DataType_t::FLOAT, {7, 1, 1, 8});
        const auto forward =
            graph.rope(x, freqs, fe::graph::RoPE_attributes().set_rope_dim(8));
        const auto backward = graph.rope_backward(
            forward, freqs,
            fe::graph::RoPE_backward_attributes().set_rope_dim(8));
        backward->set_uid(3).set_output(true);
        require(graph.validate().is_good(),
                "RoPE forward/backward graph rejected");
      }
      {
        Graph graph;
        auto out = graph.rng(fe::graph::Rng_attributes()
                                 .set_dim({2, 3, 17})
                                 .set_data_type(type)
                                 .set_seed(-1)
                                 .set_offset(4294967290LL));
        out->set_uid(1).set_output(true);
        require(graph.validate().is_good(), "input-free RNG graph rejected");
      }
    }
    {
      Graph graph;
      const auto result = graph.instancenorm(
          input(graph, 1, fe::DataType_t::HALF, {2, 3, 7}),
          input(graph, 2, fe::DataType_t::BFLOAT16, {1, 3, 1}),
          input(graph, 3, fe::DataType_t::BFLOAT16, {1, 3, 1}));
      outputs(result);
      require(graph.validate().is_bad(),
              "incompatible low-precision affine dtype accepted");
    }
    {
      Graph graph;
      const auto result = graph.layernorm_backward(
          input(graph, 1, fe::DataType_t::FLOAT, {2, 3}),
          input(graph, 2, fe::DataType_t::FLOAT, {2, 3}),
          input(graph, 3, fe::DataType_t::FLOAT, {1, 3}),
          fe::graph::Layernorm_backward_attributes()
              .set_saved_mean_and_inv_variance(
                  input(graph, 4, fe::DataType_t::HALF, {2, 1}),
                  input(graph, 5, fe::DataType_t::FLOAT, {2, 1})));
      outputs(result);
      require(graph.validate().is_bad(),
              "low-precision saved statistics accepted");
    }
    {
      Graph graph;
      const auto result =
          graph.rope(input(graph, 1, fe::DataType_t::FLOAT, {2, 3, 7, 16}),
                     input(graph, 2, fe::DataType_t::FLOAT, {7, 1, 1, 7}),
                     fe::graph::RoPE_attributes().set_rope_dim(7));
      result->set_uid(3).set_output(true);
      require(graph.validate().is_bad(), "odd RoPE dimension accepted");
    }
    {
      Graph graph;
      const auto result =
          graph.rng(fe::graph::Rng_attributes().set_dim({2, 3}).set_data_type(
              fe::DataType_t::INT32));
      result->set_uid(1).set_output(true);
      require(graph.validate().is_bad(), "integer RNG output accepted");
    }
    std::cout << "normalization, statistics, RoPE and RNG contracts PASS"
              << std::endl;
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << std::endl;
    return 1;
  }
}
