/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/composite.hpp"
#include "validation/functional/cudnn_graph.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace flagdnn::testing {
namespace {

namespace cfe = cuda::cfe;

void describe_tensor(
    const std::shared_ptr<cfe::graph::Tensor_attributes>& tensor_value,
    const TestTensor& specification,
    std::string_view name,
    std::int64_t uid,
    bool output) {
  tensor_value->set_name(std::string(name))
      .set_uid(uid)
      .set_data_type(cuda::cudnn_frontend_data_type(specification.data_type))
      .set_dim(specification.dimensions)
      .set_stride(specification.strides)
      .set_is_virtual(!output)
      .set_output(output);
}

class CudnnCompositeExecutable final : public cuda::CudnnGraphExecutable {
 public:
  explicit CudnnCompositeExecutable(const AddSquareTestCase& test_case)
      : graph_(std::make_shared<cfe::graph::Graph>()) {
    validate_composite_case(test_case);
    initialize_graph(test_case.name, test_case.output.data_type);
    const TestTensor left_spec =
        cuda::flatten_compact_tensor(test_case.left);
    const TestTensor right_spec =
        cuda::flatten_compact_tensor(test_case.right);
    const TestTensor output_spec =
        cuda::flatten_compact_tensor(test_case.output);
    const auto left = cuda::make_cudnn_tensor(graph_, left_spec, "left");
    const auto right = cuda::make_cudnn_tensor(graph_, right_spec, "right");
    const auto square = graph_->pointwise(
        right,
        right,
        cfe::graph::Pointwise_attributes()
            .set_name("square")
            .set_mode(cfe::PointwiseMode_t::MUL)
            .set_compute_data_type(cfe::DataType_t::FLOAT));
    describe_tensor(
        square, output_spec, "square", test_case.output.uid + 1, false);
    const auto output = graph_->pointwise(
        left,
        square,
        cfe::graph::Pointwise_attributes()
            .set_name("add_square")
            .set_mode(cfe::PointwiseMode_t::ADD)
            .set_compute_data_type(cfe::DataType_t::FLOAT));
    describe_tensor(
        output, output_spec, "output", test_case.output.uid, true);
    build("AddSquare");
  }

  explicit CudnnCompositeExecutable(const ConvBiasReluTestCase& test_case)
      : graph_(std::make_shared<cfe::graph::Graph>()) {
    validate_composite_case(test_case);
    initialize_graph(test_case.name, test_case.output.data_type);
    const auto x = cuda::make_cudnn_tensor(graph_, test_case.x, "x");
    const auto w = cuda::make_cudnn_tensor(graph_, test_case.w, "w");
    const auto bias =
        cuda::make_cudnn_tensor(graph_, test_case.bias, "bias");
    const auto convolution = graph_->conv_fprop(
        x,
        w,
        cfe::graph::Conv_fprop_attributes()
            .set_name("convolution")
            .set_compute_data_type(cfe::DataType_t::FLOAT)
            .set_pre_padding(test_case.padding)
            .set_post_padding(test_case.padding)
            .set_stride(test_case.stride)
            .set_dilation(test_case.dilation)
            .set_convolution_mode(cfe::ConvolutionMode_t::CROSS_CORRELATION));
    describe_tensor(convolution,
                    test_case.output,
                    "convolution",
                    test_case.output.uid + 1,
                    false);
    const auto biased = graph_->pointwise(
        convolution,
        bias,
        cfe::graph::Pointwise_attributes()
            .set_name("bias_add")
            .set_mode(cfe::PointwiseMode_t::ADD)
            .set_compute_data_type(cfe::DataType_t::FLOAT));
    describe_tensor(
        biased, test_case.output, "biased", test_case.output.uid + 2, false);
    const auto output = graph_->pointwise(
        biased,
        cfe::graph::Pointwise_attributes()
            .set_name("relu")
            .set_mode(cfe::PointwiseMode_t::RELU_FWD)
            .set_compute_data_type(cfe::DataType_t::FLOAT));
    describe_tensor(
        output, test_case.output, "output", test_case.output.uid, true);
    build("ConvBiasRelu");
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
        "cuDNN composite graph execute");
  }

 private:
  void initialize_graph(std::string_view name, flagdnnDataType_t data_type) {
    graph_->set_name(std::string(name) + "::cudnn")
        .set_io_data_type(cuda::cudnn_frontend_data_type(data_type))
        .set_intermediate_data_type(cfe::DataType_t::FLOAT)
        .set_compute_data_type(cfe::DataType_t::FLOAT);
  }

  void build(std::string_view operation) {
    cuda::check_cudnn_frontend(
        graph_->build(handle(), {cfe::HeurMode_t::A, cfe::HeurMode_t::FALLBACK}),
        std::string("cuDNN ") + std::string(operation) + " graph build");
    std::int64_t workspace_size = 0;
    cuda::check_cudnn_frontend(graph_->get_workspace_size(workspace_size),
                               "cuDNN composite workspace query");
    set_workspace_size(workspace_size);
  }

  std::shared_ptr<cfe::graph::Graph> graph_;
};

}  // namespace

std::unique_ptr<CompositeExecutable> build_add_square_reference(
    const AddSquareTestCase& test_case) {
  return std::make_unique<CudnnCompositeExecutable>(test_case);
}

std::unique_ptr<CompositeExecutable> build_conv_bias_relu_reference(
    const ConvBiasReluTestCase& test_case) {
  return std::make_unique<CudnnCompositeExecutable>(test_case);
}

}  // namespace flagdnn::testing
