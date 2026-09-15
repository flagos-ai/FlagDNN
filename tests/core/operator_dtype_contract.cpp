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
             std::vector<std::int64_t> shape = {2, 3}) {
  std::vector<std::int64_t> strides(shape.size());
  std::int64_t stride = 1;
  for (std::size_t axis = shape.size(); axis != 0; --axis) {
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
bool valid_binary(fe::PointwiseMode_t mode, fe::DataType_t type,
                  double alpha = 1.0) {
  Graph graph;
  const auto x = input(graph, 1, type), y = input(graph, 2, type);
  auto z = graph.pointwise(
      x, y, fe::graph::Pointwise_attributes().set_mode(mode).set_alpha(alpha));
  z->set_uid(3).set_output(true);
  return graph.validate().is_good();
}
}  // namespace
int main() {
  try {
    for (const auto mode :
         {fe::PointwiseMode_t::ADD, fe::PointwiseMode_t::SUB,
          fe::PointwiseMode_t::MUL, fe::PointwiseMode_t::DIV,
          fe::PointwiseMode_t::MOD, fe::PointwiseMode_t::POW,
          fe::PointwiseMode_t::MIN, fe::PointwiseMode_t::MAX,
          fe::PointwiseMode_t::CMP_EQ, fe::PointwiseMode_t::CMP_NEQ,
          fe::PointwiseMode_t::CMP_GT, fe::PointwiseMode_t::CMP_GE,
          fe::PointwiseMode_t::CMP_LT, fe::PointwiseMode_t::CMP_LE}) {
      require(valid_binary(mode, fe::DataType_t::INT32),
              "numeric mode rejected INT32");
    }
    require(!valid_binary(fe::PointwiseMode_t::ADD, fe::DataType_t::INT32, 0.5),
            "fractional INT32 alpha accepted");
    for (const auto mode :
         {fe::PointwiseMode_t::RELU_BWD, fe::PointwiseMode_t::ELU_BWD,
          fe::PointwiseMode_t::GELU_BWD,
          fe::PointwiseMode_t::GELU_APPROX_TANH_BWD,
          fe::PointwiseMode_t::SIGMOID_BWD, fe::PointwiseMode_t::SWISH_BWD,
          fe::PointwiseMode_t::TANH_BWD, fe::PointwiseMode_t::SOFTPLUS_BWD}) {
      for (const auto type : {fe::DataType_t::FLOAT, fe::DataType_t::HALF,
                              fe::DataType_t::BFLOAT16})
        require(valid_binary(mode, type),
                "activation gradient rejected floating dtype");
      require(!valid_binary(mode, fe::DataType_t::INT32),
              "activation gradient accepted INT32");
      Graph graph;
      bool rejected = false;
      try {
        (void)graph.pointwise(input(graph, 1, fe::DataType_t::FLOAT),
                              input(graph, 2, fe::DataType_t::FLOAT, {1, 3}),
                              fe::graph::Pointwise_attributes().set_mode(mode));
      } catch (const std::invalid_argument&) {
        rejected = true;
      }
      require(rejected, "activation gradient accepted broadcast shapes");
    }
    for (const auto type : {fe::DataType_t::FLOAT, fe::DataType_t::HALF,
                            fe::DataType_t::BFLOAT16, fe::DataType_t::INT32}) {
      Graph graph;
      auto output = graph.reduction(input(graph, 1, type),
                                    fe::graph::Reduction_attributes()
                                        .set_mode(fe::ReductionMode_t::ADD)
                                        .set_axis(-1));
      const auto expected = type == fe::DataType_t::INT32
                                ? fe::DataType_t::FLOAT
                                : type;
      require(output->get_data_type() == expected,
              "reduction changed the existing floating output default");
      output->set_uid(2).set_output(true);
      require(graph.validate().is_good(), "default reduction graph invalid");
      output->set_data_type(fe::DataType_t::FLOAT);
      require(graph.validate().is_good(), "explicit FP32 reduction invalid");
    }
    {
      Graph graph;
      graph.set_io_data_type(fe::DataType_t::INT32);
      auto output = graph.reduction(
          input(graph, 1, fe::DataType_t::NOT_SET),
          fe::graph::Reduction_attributes()
              .set_mode(fe::ReductionMode_t::ADD)
              .set_axis(-1));
      require(output->get_data_type() == fe::DataType_t::FLOAT,
              "reduction did not resolve inherited INT32 input type");
      output->set_uid(2).set_output(true);
      require(graph.validate().is_good(), "inherited INT32 reduction invalid");
    }
    for (const auto type :
         {fe::DataType_t::FLOAT, fe::DataType_t::HALF, fe::DataType_t::BFLOAT16,
          fe::DataType_t::INT32, fe::DataType_t::BOOLEAN,
          fe::DataType_t::FP8_E4M3, fe::DataType_t::FP8_E5M2,
          fe::DataType_t::FP8_E8M0}) {
      Graph graph;
      auto output = graph.concatenate(
          {input(graph, 1, type), input(graph, 2, type, {2, 5})},
          fe::graph::Concatenate_attributes().set_axis(-1));
      require(output->get_dim() == std::vector<std::int64_t>({2, 8}),
              "concatenate inferred wrong dimensions");
      output->set_uid(3).set_output(true);
      require(graph.validate().is_good(), "concatenate graph invalid");
    }
    for (const auto type : {fe::DataType_t::INT32, fe::DataType_t::FLOAT}) {
      Graph graph;
      auto output = graph.gen_index(fe::graph::Gen_index_attributes()
                                        .set_dim({3, 7})
                                        .set_axis(-1)
                                        .set_data_type(type));
      output->set_uid(1).set_output(true);
      require(graph.validate().is_good(), "input-free gen_index graph invalid");
    }
    {
      Graph graph;
      bool rejected = false;
      try {
        (void)graph.gen_index(
            fe::graph::Gen_index_attributes().set_dim({3, 7}).set_axis(2));
      } catch (const std::invalid_argument&) {
        rejected = true;
      }
      require(rejected, "gen_index accepted an invalid axis");
    }
    std::cout << "operator dtype/shape contract PASS" << std::endl;
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << std::endl;
    return 1;
  }
}
