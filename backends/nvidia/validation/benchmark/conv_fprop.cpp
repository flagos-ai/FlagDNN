/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "ops.hpp"

#include "validation/benchmark/cudnn_common.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <utility>

namespace flagdnn::benchmarking::cudnn_detail {
namespace {

class ConvolutionFpropExecutable final : public ExecutableBase {
 public:
  explicit ConvolutionFpropExecutable(const BenchmarkCase& specification) {
    require_tensor_count(specification, 3);

    graph_ = std::make_shared<fe::graph::Graph>();
    graph_->set_name(specification.name)
        .set_io_data_type(data_type(specification.tensors[0].data_type))
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);
    auto input =
        make_tensor(graph_, specification.tensors[0], "input", false);
    auto filter =
        make_tensor(graph_, specification.tensors[1], "filter", false);
    auto output = graph_->conv_fprop(
        input,
        filter,
        fe::graph::Conv_fprop_attributes()
            .set_name("convolution_fprop")
            .set_pre_padding(specification.convolution.pre_padding)
            .set_post_padding(specification.convolution.post_padding)
            .set_stride(specification.convolution.stride)
            .set_dilation(specification.convolution.dilation));
    const TensorSpec& expected = specification.tensors[2];
    output->set_name("output")
        .set_uid(expected.uid)
        .set_data_type(data_type(expected.data_type))
        .set_dim(expected.dimensions)
        .set_stride(expected.strides)
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
                   "cuDNN convolution FProp graph execute");
  }

 private:
  std::shared_ptr<fe::graph::Graph> graph_;
};

}  // namespace

std::unique_ptr<BenchmarkExecutable> build_convolution_fprop(
    const BenchmarkCase& specification) {
  return std::make_unique<ConvolutionFpropExecutable>(specification);
}

}  // namespace flagdnn::benchmarking::cudnn_detail
