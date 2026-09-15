/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "validation/functional/cudnn_extended.hpp"

#include "validation/cuda_driver.hpp"
#include "validation/functional/cudnn_graph.hpp"
#include "validation/tensor_io.hpp"
namespace flagdnn::testing {
namespace {
namespace cfe = cuda::cfe;
using Graph = cfe::graph::Graph;
using Tensor = std::shared_ptr<cfe::graph::Tensor_attributes>;
std::shared_ptr<Graph> make_graph(const std::string& name) {
  auto graph = std::make_shared<Graph>();
  graph->set_name(name)
      .set_io_data_type(cfe::DataType_t::FLOAT)
      .set_intermediate_data_type(cfe::DataType_t::FLOAT)
      .set_compute_data_type(cfe::DataType_t::FLOAT);
  return graph;
}
void mark_output(const Tensor& output, const TestTensor& tensor) {
  output->set_uid(tensor.uid)
      .set_dim(tensor.dimensions)
      .set_stride(tensor.strides)
      .set_data_type(cuda::cudnn_frontend_data_type(tensor.data_type))
      .set_output(true);
}
}  // namespace
std::unique_ptr<TestExecutable> build_cudnn_statistics(
    const StatisticsTestCase& test_case) {
  if (test_case.operation == "bn_finalize" && test_case.inputs.size() != 6)
    throw std::invalid_argument(
        "cuDNN BN finalize requires complete running statistics");
  auto graph = make_graph(test_case.name);
  std::vector<Tensor> inputs, outputs;
  for (const auto& input : test_case.inputs)
    inputs.push_back(cuda::make_cudnn_tensor(graph, input, "input"));
  if (test_case.operation == "genstats") {
    const auto result =
        graph->genstats(inputs[0], cfe::graph::Genstats_attributes());
    outputs.assign(result.begin(), result.end());
  } else {
    auto attributes = cfe::graph::BN_finalize_attributes();
    auto momentum = graph->tensor(static_cast<float>(test_case.momentum));
    if (inputs.size() == 6)
      attributes.set_previous_running_stats(inputs[4], inputs[5], momentum);
    const auto result = graph->bn_finalize(
        inputs[0], inputs[1], inputs[2], inputs[3],
        graph->tensor(static_cast<float>(test_case.epsilon)),
        graph->tensor(static_cast<std::int64_t>(test_case.accum_count)),
        attributes);
    for (const auto& output : result)
      if (output) outputs.push_back(output);
  }
  for (std::size_t index = 0; index < outputs.size(); ++index)
    mark_output(outputs[index], test_case.outputs.at(index));
  return cuda::build_cudnn_graph(std::move(graph));
}
std::unique_ptr<TestExecutable> build_cudnn_extended_normalization(
    const ExtendedNormalizationTestCase& test_case) {
  auto graph = make_graph(test_case.name);
  std::vector<Tensor> inputs;
  for (const auto& input : test_case.inputs)
    inputs.push_back(cuda::make_cudnn_tensor(graph, input, "input"));
  auto epsilon = graph->tensor(static_cast<float>(test_case.epsilon));
  std::array<Tensor, 3> outputs;
  if (test_case.operation == "instancenorm")
    outputs = graph->instancenorm(
        inputs[0], inputs[1], inputs[2],
        cfe::graph::Instancenorm_attributes()
            .set_name(test_case.name)
            .set_compute_data_type(cfe::DataType_t::FLOAT)
            .set_forward_phase(cfe::NormFwdPhase_t::TRAINING)
            .set_epsilon(epsilon));
  else if (test_case.operation == "adalayernorm")
    outputs = graph->adalayernorm(
        inputs[0], inputs[1], inputs[2],
        cfe::graph::AdaLayernorm_attributes()
            .set_name(test_case.name)
            .set_compute_data_type(cfe::DataType_t::FLOAT)
            .set_forward_phase(cfe::NormFwdPhase_t::TRAINING)
            .set_epsilon(epsilon));
  else if (test_case.operation == "layernorm_backward")
    outputs = graph->layernorm_backward(
        inputs[0], inputs[1], inputs[2],
        cfe::graph::Layernorm_backward_attributes()
            .set_name(test_case.name)
            .set_compute_data_type(cfe::DataType_t::FLOAT)
            .set_saved_mean_and_inv_variance(inputs[3], inputs[4]));
  else if (test_case.operation == "batchnorm_backward")
    outputs = graph->batchnorm_backward(
        inputs[0], inputs[1], inputs[2],
        cfe::graph::Batchnorm_backward_attributes()
            .set_name(test_case.name)
            .set_compute_data_type(cfe::DataType_t::FLOAT)
            .set_saved_mean_and_inv_variance(inputs[3], inputs[4]));
  else if (test_case.operation == "instancenorm_backward")
    outputs = graph->instancenorm_backward(
        inputs[0], inputs[1], inputs[2],
        cfe::graph::Instancenorm_backward_attributes()
            .set_name(test_case.name)
            .set_compute_data_type(cfe::DataType_t::FLOAT)
            .set_saved_mean_and_inv_variance(inputs[3], inputs[4]));
  else if (test_case.operation == "adalayernorm_backward")
    outputs = graph->adalayernorm_backward(
        inputs[0], inputs[1], inputs[2],
        cfe::graph::AdaLayernorm_backward_attributes()
            .set_name(test_case.name)
            .set_compute_data_type(cfe::DataType_t::FLOAT)
            .set_saved_mean_and_inv_variance(inputs[3], inputs[4]));
  else if (test_case.operation == "rmsnorm_backward")
    outputs = graph->rmsnorm_backward(
        inputs[0], inputs[1], inputs[2], inputs[3],
        cfe::graph::Rmsnorm_backward_attributes()
            .has_dbias(true)
            .set_name(test_case.name)
            .set_compute_data_type(cfe::DataType_t::FLOAT));
  else
    throw std::invalid_argument("unknown normalization test operation");
  for (std::size_t index = 0; index < outputs.size(); ++index)
    mark_output(outputs[index], test_case.outputs.at(index));
  return cuda::build_cudnn_graph(std::move(graph));
}
std::unique_ptr<TestExecutable> build_cudnn_moe_matmul(
    const MoeMatmulTestCase& test_case) {
  auto graph = make_graph(test_case.name);
  std::vector<Tensor> inputs;
  for (const auto& input : test_case.inputs)
    inputs.push_back(cuda::make_cudnn_tensor(graph, input, "input"));
  Tensor output;
  if (test_case.backward) {
    output = graph->moe_grouped_matmul_bwd(
        inputs[0], inputs[1], inputs[2],
        cfe::graph::Moe_grouped_matmul_bwd_attributes());
  } else {
    const auto mode = test_case.mode == 0 ? cfe::MoeGroupedMatmulMode_t::NONE
                      : test_case.mode == 1
                          ? cfe::MoeGroupedMatmulMode_t::GATHER
                          : cfe::MoeGroupedMatmulMode_t::SCATTER;
    output = graph->moe_grouped_matmul(
        inputs[0], inputs[1], inputs[2], test_case.mode ? inputs[3] : nullptr,
        test_case.mode == 2 ? inputs[4] : nullptr,
        cfe::graph::Moe_grouped_matmul_attributes().set_mode(mode).set_top_k(
            test_case.top_k));
  }
  mark_output(output, test_case.outputs.at(0));
  return cuda::build_cudnn_graph(std::move(graph));
}
std::unique_ptr<TestExecutable> build_cudnn_resample(
    const ResampleTestCase& test_case) {
  // The Graph bilinear engine uses asymmetric coordinates. The sampler API
  // accepts a grid for the half-pixel / align-corners convention in FlagDNN.
  if (test_case.mode == 3) return build_cudnn_bilinear_resample(test_case);
  auto graph = make_graph(test_case.name);
  auto attributes =
      cfe::graph::Resample_attributes()
          .set_resampling_mode(
              test_case.mode == 1 ? cfe::ResampleMode_t::AVGPOOL_EXCLUDE_PADDING
              : test_case.mode == 2
                  ? cfe::ResampleMode_t::AVGPOOL_INCLUDE_PADDING
              : test_case.mode == 4 ? cfe::ResampleMode_t::NEAREST
                                    : cfe::ResampleMode_t::MAXPOOL)
          .set_padding_mode(
              test_case.padding == 1   ? cfe::PaddingMode_t::EDGE_VAL_PAD
              : test_case.padding == 2 ? cfe::PaddingMode_t::NEG_INF_PAD
                                       : cfe::PaddingMode_t::ZERO_PAD)
          .set_generate_index(test_case.outputs.size() == 2);
  attributes.set_window(test_case.window)
      .set_stride(test_case.stride)
      .set_pre_padding(test_case.pre)
      .set_post_padding(test_case.post);
  const auto outputs = graph->resample(
      cuda::make_cudnn_tensor(graph, test_case.inputs.at(0), "input"),
      attributes);
  for (std::size_t index = 0; index < test_case.outputs.size(); ++index)
    mark_output(outputs[index], test_case.outputs[index]);
  return cuda::build_cudnn_graph(std::move(graph));
}
std::unique_ptr<TestExecutable> build_cudnn_fp8_matmul(
    const Fp8MatmulTestCase& test_case) {
  auto graph = make_graph(test_case.name);
  std::vector<Tensor> inputs;
  for (auto input : test_case.inputs) {
    while (input.dimensions.size() < 3) {
      const auto span =
          static_cast<std::int64_t>(cuda::storage_element_count(input));
      input.dimensions.insert(input.dimensions.begin(), 1);
      input.strides.insert(input.strides.begin(), span);
    }
    inputs.push_back(cuda::make_cudnn_tensor(graph, input, "input"));
  }
  auto left = inputs[0], right = inputs[1];
  if (test_case.scale_mode == 2) {
    const auto attributes =
        cfe::graph::Block_scale_dequantize_attributes().set_block_size(
            std::vector<std::int32_t>{32});
    left = graph->block_scale_dequantize(inputs[0], inputs[2], attributes);
    right = graph->block_scale_dequantize(inputs[1], inputs[3], attributes);
  }
  auto output = graph->matmul(left, right, cfe::graph::Matmul_attributes());
  // This is the same matmul + descale graph used by cuDNN MatmulFP8Node.
  if (test_case.scale_mode == 1) {
    auto multiply =
        cfe::graph::Pointwise_attributes().set_mode(cfe::PointwiseMode_t::MUL);
    output = graph->pointwise(output, inputs[2], multiply);
    output = graph->pointwise(output, inputs[3], multiply);
  }
  mark_output(output, test_case.outputs.at(0));
  return cuda::build_cudnn_graph(std::move(graph));
}
namespace {
class CudnnIndexGraph final : public TestExecutable {
 public:
  explicit CudnnIndexGraph(const IndexTestCase& test_case) {
    auto graph = make_graph(test_case.name);
    Tensor output;
    const auto rank =
        static_cast<std::int64_t>(test_case.output.dimensions.size());
    const auto axis =
        test_case.axis < 0 ? test_case.axis + rank : test_case.axis;
    if (test_case.operation == "concatenate") {
      std::vector<Tensor> inputs;
      for (const auto& input : test_case.inputs)
        inputs.push_back(cuda::make_cudnn_tensor(graph, input, "input"));
      output = graph->concatenate(
          inputs, cfe::graph::Concatenate_attributes().set_axis(axis));
      mark_output(output, test_case.output);
    } else {
      if (rank != 1 || axis != 0)
        throw std::invalid_argument(
            "cuDNN gen_index cases require one logical axis");
      auto descriptor = cuda::flatten_compact_tensor(test_case.output);
      descriptor.data_type = FLAGDNN_DATA_FLOAT32;
      shape_input_uid_ = test_case.output.uid == 1 ? 2 : 1;
      descriptor.uid = shape_input_uid_;
      // cuDNN GEN_INDEX takes a shape-carrying input whose values are unused.
      shape_input_ = std::make_unique<DeviceBuffer>(
          cuda::storage_element_count(descriptor) * sizeof(float));
      cuda::check_cuda_runtime(
          cudaMemset(shape_input_->opaque(), 0,
                     cuda::storage_element_count(descriptor) * sizeof(float)),
          "cudaMemset(gen_index shape input)");
      const auto input =
          cuda::make_cudnn_tensor(graph, descriptor, "shape_input");
      output =
          graph->pointwise(input, cfe::graph::Pointwise_attributes()
                                      .set_mode(cfe::PointwiseMode_t::GEN_INDEX)
                                      .set_axis(1));
      mark_output(output, cuda::flatten_compact_tensor(test_case.output));
    }
    executable_ = cuda::build_cudnn_graph(std::move(graph));
  }
  std::size_t workspace_size() const noexcept override {
    return executable_->workspace_size();
  }
  void execute(std::span<const flagdnnBinding_t> bindings, void* workspace,
               std::size_t size, flagdnnStream_t stream) override {
    std::vector<flagdnnBinding_t> pointers(bindings.begin(), bindings.end());
    if (shape_input_)
      pointers.push_back({shape_input_uid_, shape_input_->opaque()});
    executable_->execute(pointers, workspace, size, stream);
  }

 private:
  std::unique_ptr<TestExecutable> executable_;
  std::unique_ptr<DeviceBuffer> shape_input_;
  std::int64_t shape_input_uid_ = 0;
};
}  // namespace
std::unique_ptr<TestExecutable> build_cudnn_index(
    const IndexTestCase& test_case) {
  try {
    return std::make_unique<CudnnIndexGraph>(test_case);
  } catch (const std::runtime_error&) {
    if (test_case.operation != "concatenate") throw;
    return build_cudnn_concatenate_transform(test_case);
  }
}
}  // namespace flagdnn::testing
