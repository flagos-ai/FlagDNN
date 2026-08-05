/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/normalization.hpp"
#include "validation/functional/cudnn_graph.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace flagdnn::testing {
namespace {

using Graph = cuda::cfe::graph::Graph;
using Tensor = std::shared_ptr<cuda::cfe::graph::Tensor_attributes>;

std::shared_ptr<Graph> make_graph(std::string_view name,
                                  flagdnnDataType_t data_type) {
  auto graph = std::make_shared<Graph>();
  graph->set_name(std::string(name))
      .set_io_data_type(cuda::cudnn_frontend_data_type(data_type))
      .set_intermediate_data_type(cuda::cfe::DataType_t::FLOAT)
      .set_compute_data_type(cuda::cfe::DataType_t::FLOAT);
  return graph;
}

void mark_output(const Tensor& output,
                 const TestTensor& expected,
                 std::string_view name) {
  output->set_name(std::string(name))
      .set_uid(expected.uid)
      .set_data_type(cuda::cudnn_frontend_data_type(expected.data_type))
      .set_dim(expected.dimensions)
      .set_stride(expected.strides)
      .set_output(true);
}

class CudnnNormalizationExecutable final
    : public cuda::CudnnGraphExecutable {
 public:
  CudnnNormalizationExecutable(
      std::shared_ptr<Graph> graph,
      std::string operation,
      std::vector<std::pair<std::int64_t, float>> scalars = {})
      : graph_(std::move(graph)),
        operation_(std::move(operation)),
        scalars_(std::move(scalars)) {
    cuda::check_cudnn_frontend(
        graph_->build(handle(),
                      {cuda::cfe::HeurMode_t::A,
                       cuda::cfe::HeurMode_t::FALLBACK}),
        "cuDNN normalization graph build");
    std::int64_t workspace_size = 0;
    cuda::check_cudnn_frontend(
        graph_->get_workspace_size(workspace_size),
        "cuDNN normalization workspace query");
    set_workspace_size(workspace_size);
  }

  void execute(std::span<const flagdnnBinding_t> bindings,
               void* workspace,
               std::size_t workspace_size,
               flagdnnStream_t stream) override {
    begin_execute(workspace, workspace_size, stream);
    cuda::CudnnBindingMap pointers =
        cuda::make_cudnn_binding_map(bindings);
    for (auto& [uid, value] : scalars_) {
      pointers[uid] = &value;
    }
    cuda::check_cudnn_frontend(
        graph_->execute(handle(), pointers, workspace),
        std::string("cuDNN ") + operation_ + " graph execute");
  }

 private:
  std::shared_ptr<Graph> graph_;
  std::string operation_;
  std::vector<std::pair<std::int64_t, float>> scalars_;
};

}  // namespace

TestTensor batchnorm_reference_data_tensor(const TestTensor& tensor) {
  if (tensor.dimensions.size() != 4 || tensor.strides.size() != 4) {
    throw std::invalid_argument(
        "cuDNN BatchNorm reference requires rank-four X/Y tensors");
  }
  TestTensor result = tensor;
  const std::int64_t channels = tensor.dimensions[1];
  const std::int64_t height = tensor.dimensions[2];
  const std::int64_t width = tensor.dimensions[3];
  result.strides = {
      channels * height * width, 1, width * channels, channels};
  result.binding_byte_offset = 0;
  return result;
}

std::unique_ptr<NormalizationExecutable> build_layernorm_reference(
    const LayernormTestCase& test_case) {
  validate_normalization_case(test_case);
  auto graph = make_graph(test_case.name, test_case.x.data_type);
  const auto x = cuda::make_cudnn_tensor(graph, test_case.x, "x");
  const auto scale =
      cuda::make_cudnn_tensor(graph, test_case.scale, "scale");
  const auto bias =
      cuda::make_cudnn_tensor(graph, test_case.bias, "bias");
  auto attributes = cuda::cfe::graph::Layernorm_attributes()
                        .set_name("layernorm")
                        .set_compute_data_type(cuda::cfe::DataType_t::FLOAT)
                        .set_forward_phase(
                            cuda::cfe::NormFwdPhase_t::TRAINING)
                        .set_epsilon(static_cast<float>(test_case.epsilon));
  auto outputs = graph->layernorm(x, scale, bias, std::move(attributes));
  mark_output(outputs[0], test_case.y, "y");
  mark_output(outputs[1], test_case.mean, "mean");
  mark_output(outputs[2], test_case.inv_variance, "inv_variance");
  return std::make_unique<CudnnNormalizationExecutable>(
      std::move(graph), "LayerNorm");
}

std::unique_ptr<NormalizationExecutable> build_rmsnorm_reference(
    const RmsnormTestCase& test_case) {
  validate_normalization_case(test_case);
  auto graph = make_graph(test_case.name, test_case.x.data_type);
  const auto x = cuda::make_cudnn_tensor(graph, test_case.x, "x");
  const auto scale =
      cuda::make_cudnn_tensor(graph, test_case.scale, "scale");
  auto bias = cuda::make_cudnn_tensor(graph, test_case.bias, "bias");
  auto epsilon = graph->tensor(
      static_cast<float>(test_case.epsilon),
      cuda::cfe::graph::ScalarType::RUNTIME_PARAM);
  epsilon->set_name("epsilon").set_uid(900001);
  auto attributes = cuda::cfe::graph::Rmsnorm_attributes()
                        .set_name("rmsnorm")
                        .set_compute_data_type(cuda::cfe::DataType_t::FLOAT)
                        .set_forward_phase(
                            cuda::cfe::NormFwdPhase_t::TRAINING)
                        .set_bias(bias)
                        .set_epsilon(epsilon);
  auto outputs = graph->rmsnorm(x, scale, std::move(attributes));
  mark_output(outputs[0], test_case.y, "y");
  mark_output(outputs[1], test_case.inv_variance, "inv_variance");
  return std::make_unique<CudnnNormalizationExecutable>(
      std::move(graph),
      "RMSNorm",
      std::vector<std::pair<std::int64_t, float>>{
          {900001, static_cast<float>(test_case.epsilon)}});
}

std::unique_ptr<NormalizationExecutable> build_batchnorm_reference(
    const BatchnormTestCase& test_case) {
  validate_normalization_case(test_case);
  auto graph = make_graph(test_case.name, test_case.x.data_type);
  const TestTensor reference_x =
      batchnorm_reference_data_tensor(test_case.x);
  const TestTensor reference_y =
      batchnorm_reference_data_tensor(test_case.y);
  const auto x = cuda::make_cudnn_tensor(graph, reference_x, "x");
  const auto scale =
      cuda::make_cudnn_tensor(graph, test_case.scale, "scale");
  const auto bias =
      cuda::make_cudnn_tensor(graph, test_case.bias, "bias");
  auto previous_mean = cuda::make_cudnn_tensor(
      graph, test_case.previous_running_mean, "previous_running_mean");
  auto previous_variance = cuda::make_cudnn_tensor(
      graph,
      test_case.previous_running_variance,
      "previous_running_variance");
  auto epsilon = graph->tensor(
      static_cast<float>(test_case.epsilon),
      cuda::cfe::graph::ScalarType::RUNTIME_PARAM);
  auto momentum = graph->tensor(
      static_cast<float>(test_case.momentum),
      cuda::cfe::graph::ScalarType::RUNTIME_PARAM);
  epsilon->set_name("epsilon").set_uid(900002);
  momentum->set_name("momentum").set_uid(900003);
  auto attributes = cuda::cfe::graph::Batchnorm_attributes()
                        .set_name("batchnorm")
                        .set_compute_data_type(cuda::cfe::DataType_t::FLOAT)
                        .set_previous_running_stats(
                            previous_mean, previous_variance, momentum)
                        .set_epsilon(epsilon);
  auto outputs = graph->batchnorm(x, scale, bias, std::move(attributes));
  mark_output(outputs[0], reference_y, "y");
  mark_output(outputs[1], test_case.mean, "mean");
  mark_output(outputs[2], test_case.inv_variance, "inv_variance");
  mark_output(outputs[3], test_case.next_running_mean, "next_running_mean");
  mark_output(outputs[4],
              test_case.next_running_variance,
              "next_running_variance");
  return std::make_unique<CudnnNormalizationExecutable>(
      std::move(graph),
      "BatchNorm",
      std::vector<std::pair<std::int64_t, float>>{
          {900002, static_cast<float>(test_case.epsilon)},
          {900003, static_cast<float>(test_case.momentum)}});
}

std::unique_ptr<NormalizationExecutable>
build_batchnorm_inference_reference(
    const BatchnormInferenceTestCase& test_case) {
  validate_normalization_case(test_case);
  auto graph = make_graph(test_case.name, test_case.x.data_type);
  const TestTensor reference_x =
      batchnorm_reference_data_tensor(test_case.x);
  const TestTensor reference_y =
      batchnorm_reference_data_tensor(test_case.y);
  const auto x = cuda::make_cudnn_tensor(graph, reference_x, "x");
  const auto mean =
      cuda::make_cudnn_tensor(graph, test_case.mean, "mean");
  const auto inv_variance = cuda::make_cudnn_tensor(
      graph, test_case.inv_variance, "inv_variance");
  const auto scale =
      cuda::make_cudnn_tensor(graph, test_case.scale, "scale");
  const auto bias =
      cuda::make_cudnn_tensor(graph, test_case.bias, "bias");

  /*
   * cuDNN exposes Batchnorm_inference_attributes, but current backend engines
   * do not provide a standalone inference plan.  This exact pointwise graph
   * remains a real cuDNN Frontend Graph/GPU reference for the same equation.
   */
  auto centered = graph->pointwise(
      x,
      mean,
      cuda::cfe::graph::Pointwise_attributes()
          .set_name("center")
          .set_mode(cuda::cfe::PointwiseMode_t::SUB)
          .set_compute_data_type(cuda::cfe::DataType_t::FLOAT));
  centered->set_is_virtual(true);
  auto normalized = graph->pointwise(
      centered,
      inv_variance,
      cuda::cfe::graph::Pointwise_attributes()
          .set_name("normalize")
          .set_mode(cuda::cfe::PointwiseMode_t::MUL)
          .set_compute_data_type(cuda::cfe::DataType_t::FLOAT));
  normalized->set_is_virtual(true);
  auto scaled = graph->pointwise(
      normalized,
      scale,
      cuda::cfe::graph::Pointwise_attributes()
          .set_name("scale")
          .set_mode(cuda::cfe::PointwiseMode_t::MUL)
          .set_compute_data_type(cuda::cfe::DataType_t::FLOAT));
  scaled->set_is_virtual(true);
  auto output = graph->pointwise(
      scaled,
      bias,
      cuda::cfe::graph::Pointwise_attributes()
          .set_name("bias")
          .set_mode(cuda::cfe::PointwiseMode_t::ADD)
          .set_compute_data_type(cuda::cfe::DataType_t::FLOAT));
  mark_output(output, reference_y, "y");
  return std::make_unique<CudnnNormalizationExecutable>(
      std::move(graph), "BatchNorm Inference");
}

}  // namespace flagdnn::testing
