/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "cpu_cudnn.hpp"
#include <algorithm>
#include <array>
#include <numeric>
#include "reference/cpu/normalization.hpp"
#include "validation/functional/cudnn_graph.hpp"

namespace flagdnn::reference::tests {
namespace cfe = cudnn_frontend;

void normalization_helpers(Suite& suite) {
  for (const float epsilon : {1e-5F, 1e-3F}) {
    suite.run("variance_to_inverse_std_epsilon_" + std::to_string(epsilon), [&] {
      auto graph = std::make_shared<cfe::graph::Graph>();
      graph->set_compute_data_type(cfe::DataType_t::FLOAT)
          .set_intermediate_data_type(cfe::DataType_t::FLOAT);
      const auto variance = tensor(1, {1, 256, 1, 1});
      const auto epsilon_tensor = tensor(2, {1, 1, 1, 1});
      const auto output = tensor(3, variance.dimensions);
      auto x = t::cuda::make_cudnn_tensor(graph, variance, "variance");
      auto e = t::cuda::make_cudnn_tensor(graph, epsilon_tensor, "epsilon");
      auto sum = graph->pointwise(x, e,
          cfe::graph::Pointwise_attributes().set_mode(cfe::PointwiseMode_t::ADD));
      auto y = graph->pointwise(sum,
          cfe::graph::Pointwise_attributes().set_mode(cfe::PointwiseMode_t::RSQRT));
      y->set_uid(3).set_dim(output.dimensions).set_stride(output.strides)
          .set_data_type(cfe::DataType_t::FLOAT).set_output(true);
      Values values(256);
      for (std::size_t i = 0; i < values.size(); ++i) values[i] = float(i * i) / 256;
      suite.check({variance, epsilon_tensor}, {output}, t::cuda::build_cudnn_graph(graph),
          [&](const Arrays& inputs) {
            auto result = inputs[0];
            cpu::variance_to_inverse_standard_deviation(result, epsilon);
            return Arrays{result};
          }, 2e-5, 2e-5, {values, {epsilon}});
    });
  }
}
}  // namespace flagdnn::reference::tests
