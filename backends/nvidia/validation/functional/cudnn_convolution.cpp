/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/convolution.hpp"
#include "validation/functional/cudnn_graph.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <utility>

namespace flagdnn::testing {
namespace {

cuda::cfe::ConvolutionMode_t cudnn_convolution_mode(ConvolutionMode mode) {
  switch (mode) {
    case ConvolutionMode::kCrossCorrelation:
      return cuda::cfe::ConvolutionMode_t::CROSS_CORRELATION;
    case ConvolutionMode::kConvolution:
      return cuda::cfe::ConvolutionMode_t::CONVOLUTION;
  }
  throw std::invalid_argument("unsupported cuDNN convolution mode");
}

template <typename Attributes>
Attributes apply_attributes(Attributes attributes,
                            const ConvolutionTestCase& test_case) {
  return attributes
      .set_name("convolution")
      .set_compute_data_type(cuda::cfe::DataType_t::FLOAT)
      .set_pre_padding(test_case.pre_padding)
      .set_post_padding(test_case.post_padding)
      .set_stride(test_case.stride)
      .set_dilation(test_case.dilation)
      .set_convolution_mode(cudnn_convolution_mode(test_case.mode));
}

class CudnnConvolutionExecutable final : public cuda::CudnnGraphExecutable {
 public:
  explicit CudnnConvolutionExecutable(
      const ConvolutionTestCase& test_case)
      : graph_(std::make_shared<cuda::cfe::graph::Graph>()) {
    validate_convolution_case(test_case);
    const cuda::cfe::DataType_t io_type =
        cuda::cudnn_frontend_data_type(test_case.x.data_type);
    graph_->set_name(test_case.name)
        .set_io_data_type(io_type)
        .set_intermediate_data_type(cuda::cfe::DataType_t::FLOAT)
        .set_compute_data_type(cuda::cfe::DataType_t::FLOAT);

    const auto x = test_case.direction == ConvolutionDirection::kDgrad
                       ? nullptr
                       : cuda::make_cudnn_tensor(graph_, test_case.x, "x");
    const auto w = test_case.direction == ConvolutionDirection::kWgrad
                       ? nullptr
                       : cuda::make_cudnn_tensor(graph_, test_case.w, "w");
    const auto y = test_case.direction == ConvolutionDirection::kFprop
                       ? nullptr
                       : cuda::make_cudnn_tensor(graph_, test_case.y, "y");
    std::shared_ptr<cuda::cfe::graph::Tensor_attributes> output;
    switch (test_case.direction) {
      case ConvolutionDirection::kFprop:
        output = graph_->conv_fprop(
            x,
            w,
            apply_attributes(
                cuda::cfe::graph::Conv_fprop_attributes(), test_case));
        break;
      case ConvolutionDirection::kDgrad:
        output = graph_->conv_dgrad(
            y,
            w,
            apply_attributes(
                cuda::cfe::graph::Conv_dgrad_attributes(), test_case));
        break;
      case ConvolutionDirection::kWgrad:
        output = graph_->conv_wgrad(
            y,
            x,
            apply_attributes(
                cuda::cfe::graph::Conv_wgrad_attributes(), test_case));
        break;
    }
    const TestTensor& expected = convolution_output_tensor(test_case);
    output->set_name("output")
        .set_uid(expected.uid)
        .set_data_type(cuda::cudnn_frontend_data_type(expected.data_type))
        .set_dim(expected.dimensions)
        .set_stride(expected.strides)
        .set_output(true);

    cuda::check_cudnn_frontend(
        graph_->build(handle(),
                      {cuda::cfe::HeurMode_t::A,
                       cuda::cfe::HeurMode_t::FALLBACK}),
        "cuDNN convolution graph build");
    std::int64_t workspace_size = 0;
    cuda::check_cudnn_frontend(
        graph_->get_workspace_size(workspace_size),
        "cuDNN convolution workspace query");
    set_workspace_size(workspace_size);
  }

  void execute(std::span<const flagdnnBinding_t> bindings,
               void* workspace,
               std::size_t workspace_size,
               flagdnnStream_t stream) override {
    begin_execute(workspace, workspace_size, stream);
    cuda::CudnnBindingMap pointers =
        cuda::make_cudnn_binding_map(bindings);
    cuda::check_cudnn_frontend(
        graph_->execute(handle(), pointers, workspace),
        "cuDNN convolution graph execute");
  }

 private:
  std::shared_ptr<cuda::cfe::graph::Graph> graph_;
};

}  // namespace

std::unique_ptr<ConvolutionExecutable> build_convolution_reference(
    const ConvolutionTestCase& test_case) {
  return std::make_unique<CudnnConvolutionExecutable>(test_case);
}

}  // namespace flagdnn::testing
