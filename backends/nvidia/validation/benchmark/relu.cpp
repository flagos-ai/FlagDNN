/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "ops.hpp"

#include "validation/benchmark/cudnn_common.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <unordered_map>
#include <utility>

namespace flagdnn::benchmarking::cudnn_detail {
namespace {

std::int64_t element_count(const TensorSpec& specification) {
  std::int64_t result = 1;
  for (const std::int64_t dimension : specification.dimensions) {
    result *= dimension;
  }
  return result;
}

std::int64_t storage_element_count(const TensorSpec& specification) {
  std::int64_t result = 1;
  for (std::size_t axis = 0; axis < specification.dimensions.size(); ++axis) {
    result += (specification.dimensions[axis] - 1) *
              specification.strides[axis];
  }
  return result;
}

bool has_same_physical_mapping(const TensorSpec& input,
                               const TensorSpec& output) {
  if (input.dimensions != output.dimensions) {
    return false;
  }
  for (std::size_t axis = 0; axis < input.dimensions.size(); ++axis) {
    if (input.dimensions[axis] > 1 &&
        input.strides[axis] != output.strides[axis]) {
      return false;
    }
  }
  return true;
}

class ReluExecutable final : public ExecutableBase {
 public:
  explicit ReluExecutable(const BenchmarkCase& specification) {
    require_tensor_count(specification, 2);
    const std::size_t logical_rank =
        specification.tensors[1].dimensions.size();
    const bool compact_pair =
        storage_element_count(specification.tensors[0]) ==
            element_count(specification.tensors[0]) &&
        storage_element_count(specification.tensors[1]) ==
            element_count(specification.tensors[1]) &&
        has_same_physical_mapping(specification.tensors[0],
                                  specification.tensors[1]);
    const TensorSpec input_spec =
        compact_pair
            ? compact_unary_tensor(specification.tensors[0], logical_rank)
            : padded_to_rank_four(specification.tensors[0], logical_rank);
    const TensorSpec output_spec =
        compact_pair
            ? compact_unary_tensor(specification.tensors[1], logical_rank)
            : padded_to_rank_four(specification.tensors[1], logical_rank);

    graph_ = std::make_shared<fe::graph::Graph>();
    graph_->set_name(specification.name)
        .set_io_data_type(data_type(input_spec.data_type))
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);
    auto input = make_tensor(graph_, input_spec, "x", false);
    auto output = graph_->pointwise(
        input,
        fe::graph::Pointwise_attributes()
            .set_name("relu")
            .set_mode(fe::PointwiseMode_t::RELU_FWD));
    output->set_name("y")
        .set_uid(output_spec.uid)
        .set_data_type(data_type(output_spec.data_type))
        .set_dim(output_spec.dimensions)
        .set_stride(output_spec.strides)
        .set_output(true);

    check_frontend(
        graph_->build(handle(), {fe::HeurMode_t::A}), "cuDNN graph build");
    std::int64_t workspace = 0;
    check_frontend(
        graph_->get_workspace_size(workspace), "cuDNN workspace query");
    set_workspace_size(workspace);
  }

  void execute(std::span<const flagdnnBinding_t> bindings,
               void* workspace,
               std::size_t workspace_size,
               flagdnnStream_t stream) override {
    begin_execute(workspace, workspace_size, stream);
    BindingMap pointers = make_binding_map(bindings);
    check_frontend(graph_->execute(handle(), pointers, workspace),
                   "cuDNN ReLU graph execute");
  }

 private:
  std::shared_ptr<fe::graph::Graph> graph_;
};

}  // namespace

std::unique_ptr<BenchmarkExecutable> build_relu(
    const BenchmarkCase& specification) {
  return std::make_unique<ReluExecutable>(specification);
}

}  // namespace flagdnn::benchmarking::cudnn_detail
