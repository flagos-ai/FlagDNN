/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "ops.hpp"

#include "validation/benchmark/cudnn_common.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

namespace flagdnn::benchmarking::cudnn_detail {
namespace {

class ReshapeExecutable final : public ExecutableBase {
 public:
  explicit ReshapeExecutable(const BenchmarkCase& specification) {
    require_tensor_count(specification, 2);
    const TensorSpec& expected = specification.tensors[1];

    graph_ = std::make_shared<fe::graph::Graph>();
    graph_->set_name(specification.name)
        .set_io_data_type(data_type(specification.tensors[0].data_type))
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);
    const TensorSpec& original_input = specification.tensors[0];
    auto input = make_tensor(graph_, original_input, "input", false);
    auto output = graph_->reshape(
        input,
        fe::graph::Reshape_attributes()
            .set_name("reshape")
            .set_compute_data_type(fe::DataType_t::FLOAT)
            .set_dim(expected.dimensions)
            .set_stride(expected.strides)
            .set_reshape_mode(
                specification.reshape.logical
                    ? fe::ReshapeMode_t::LOGICAL
                    : fe::ReshapeMode_t::VIEW_ONLY));
    output->set_name("output")
        .set_uid(expected.uid)
        .set_data_type(data_type(expected.data_type))
        .set_dim(expected.dimensions)
        .set_stride(expected.strides)
        .set_output(true);

    build_frontend_layout_graph_or_unsupported(
        *graph_, handle(), "reshape");
    std::int64_t workspace = 0;
    check_frontend(
        graph_->get_workspace_size(workspace),
        "cuDNN reshape workspace query");
    set_workspace_size(workspace);
  }

  void execute(std::span<const flagdnnBinding_t> bindings,
               void* workspace,
               std::size_t workspace_size,
               flagdnnStream_t stream) override {
    begin_execute(workspace, workspace_size, stream);
    BindingMap pointers = make_binding_map(bindings);
    check_frontend(graph_->execute(handle(), pointers, workspace),
                   "cuDNN reshape graph execute");
  }

 private:
  std::shared_ptr<fe::graph::Graph> graph_;
};

}  // namespace

std::unique_ptr<BenchmarkExecutable> build_reshape(
    const BenchmarkCase& specification) {
  return std::make_unique<ReshapeExecutable>(specification);
}

}  // namespace flagdnn::benchmarking::cudnn_detail
