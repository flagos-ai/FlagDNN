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

fe::ConvolutionMode_t convolution_mode(ConvolutionMode mode) {
  switch (mode) {
    case ConvolutionMode::kCrossCorrelation:
      return fe::ConvolutionMode_t::CROSS_CORRELATION;
    case ConvolutionMode::kConvolution:
      return fe::ConvolutionMode_t::CONVOLUTION;
  }
  throw std::invalid_argument("unsupported convolution mode");
}

class ConvolutionDgradExecutable final : public ExecutableBase {
 public:
  explicit ConvolutionDgradExecutable(const BenchmarkCase& specification) {
    require_tensor_count(specification, 3);
    graph_ = std::make_shared<fe::graph::Graph>();
    graph_->set_name(specification.name)
        .set_io_data_type(data_type(specification.tensors[0].data_type))
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);
    auto loss = make_tensor(
        graph_, specification.tensors[0], "dy", false);
    auto filter = make_tensor(
        graph_, specification.tensors[1], "w", false);
    auto output = graph_->conv_dgrad(
        loss,
        filter,
        fe::graph::Conv_dgrad_attributes()
            .set_name("convolution_dgrad")
            .set_compute_data_type(fe::DataType_t::FLOAT)
            .set_pre_padding(specification.convolution.pre_padding)
            .set_post_padding(specification.convolution.post_padding)
            .set_stride(specification.convolution.stride)
            .set_dilation(specification.convolution.dilation)
            .set_convolution_mode(
                convolution_mode(specification.convolution.mode)));
    const TensorSpec& expected = specification.tensors[2];
    output->set_name("dx")
        .set_uid(expected.uid)
        .set_data_type(data_type(expected.data_type))
        .set_dim(expected.dimensions)
        .set_stride(expected.strides)
        .set_output(true);
    check_frontend(
        graph_->build(
            handle(), {fe::HeurMode_t::A, fe::HeurMode_t::FALLBACK}),
        "cuDNN Dgrad graph build");
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
                   "cuDNN convolution Dgrad graph execute");
  }

 private:
  std::shared_ptr<fe::graph::Graph> graph_;
};

class ConvolutionWgradExecutable final : public ExecutableBase {
 public:
  explicit ConvolutionWgradExecutable(const BenchmarkCase& specification) {
    require_tensor_count(specification, 3);
    graph_ = std::make_shared<fe::graph::Graph>();
    graph_->set_name(specification.name)
        .set_io_data_type(data_type(specification.tensors[0].data_type))
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);
    auto loss = make_tensor(
        graph_, specification.tensors[0], "dy", false);
    auto image = make_tensor(
        graph_, specification.tensors[1], "x", false);
    auto output = graph_->conv_wgrad(
        loss,
        image,
        fe::graph::Conv_wgrad_attributes()
            .set_name("convolution_wgrad")
            .set_compute_data_type(fe::DataType_t::FLOAT)
            .set_pre_padding(specification.convolution.pre_padding)
            .set_post_padding(specification.convolution.post_padding)
            .set_stride(specification.convolution.stride)
            .set_dilation(specification.convolution.dilation)
            .set_convolution_mode(
                convolution_mode(specification.convolution.mode)));
    const TensorSpec& expected = specification.tensors[2];
    output->set_name("dw")
        .set_uid(expected.uid)
        .set_data_type(data_type(expected.data_type))
        .set_dim(expected.dimensions)
        .set_stride(expected.strides)
        .set_output(true);
    check_frontend(
        graph_->build(
            handle(), {fe::HeurMode_t::A, fe::HeurMode_t::FALLBACK}),
        "cuDNN Wgrad graph build");
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
                   "cuDNN convolution Wgrad graph execute");
  }

 private:
  std::shared_ptr<fe::graph::Graph> graph_;
};

}  // namespace

std::unique_ptr<BenchmarkExecutable> build_convolution_dgrad(
    const BenchmarkCase& specification) {
  return std::make_unique<ConvolutionDgradExecutable>(specification);
}

std::unique_ptr<BenchmarkExecutable> build_convolution_wgrad(
    const BenchmarkCase& specification) {
  return std::make_unique<ConvolutionWgradExecutable>(specification);
}

}  // namespace flagdnn::benchmarking::cudnn_detail
