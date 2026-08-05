/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "ops.hpp"

#include "validation/benchmark/cudnn_common.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <utility>

namespace flagdnn::benchmarking::cudnn_detail {
namespace {

class BatchnormInferenceExecutable final : public ExecutableBase {
 public:
  explicit BatchnormInferenceExecutable(const BenchmarkCase& specification) {
    require_tensor_count(specification, 6);
    graph_ = std::make_shared<fe::graph::Graph>();
    graph_->set_name(specification.name)
        .set_io_data_type(data_type(specification.tensors[0].data_type))
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    const TensorSpec cudnn_x = batchnorm_inference_nhwc_tensor(
        specification.tensors[0]);
    auto x = make_tensor(graph_, cudnn_x, "x", false);
    auto mean =
        make_tensor(graph_, specification.tensors[1], "mean", false);
    auto inv_variance = make_tensor(
        graph_, specification.tensors[2], "inv_variance", false);
    auto scale =
        make_tensor(graph_, specification.tensors[3], "scale", false);
    auto bias =
        make_tensor(graph_, specification.tensors[4], "bias", false);
    // cuDNN exposes Batchnorm_inference_attributes, but the backend
    // NormalizationForward support matrix has no BatchNorm inference
    // engine. Keep the reference on the real cuDNN Frontend by spelling
    // out the exact inference equation as a fused pointwise graph.
    auto centered = graph_->pointwise(
        x,
        mean,
        fe::graph::Pointwise_attributes()
            .set_name("batchnorm_center")
            .set_mode(fe::PointwiseMode_t::SUB)
            .set_compute_data_type(fe::DataType_t::FLOAT));
    centered->set_is_virtual(true);
    auto normalized = graph_->pointwise(
        centered,
        inv_variance,
        fe::graph::Pointwise_attributes()
            .set_name("batchnorm_normalize")
            .set_mode(fe::PointwiseMode_t::MUL)
            .set_compute_data_type(fe::DataType_t::FLOAT));
    normalized->set_is_virtual(true);
    auto scaled = graph_->pointwise(
        normalized,
        scale,
        fe::graph::Pointwise_attributes()
            .set_name("batchnorm_scale")
            .set_mode(fe::PointwiseMode_t::MUL)
            .set_compute_data_type(fe::DataType_t::FLOAT));
    scaled->set_is_virtual(true);
    auto output = graph_->pointwise(
        scaled,
        bias,
        fe::graph::Pointwise_attributes()
            .set_name("batchnorm_bias")
            .set_mode(fe::PointwiseMode_t::ADD)
            .set_compute_data_type(fe::DataType_t::FLOAT));

    const TensorSpec expected = batchnorm_inference_nhwc_tensor(
        specification.tensors[5]);
    output->set_name("y")
        .set_uid(expected.uid)
        .set_data_type(data_type(expected.data_type))
        .set_dim(expected.dimensions)
        .set_stride(expected.strides)
        .set_output(true);

    check_frontend(
        graph_->build(
            handle(), {fe::HeurMode_t::A, fe::HeurMode_t::FALLBACK}),
        "cuDNN BatchNorm Inference graph build");
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
                   "cuDNN BatchNorm Inference graph execute");
  }

 private:
  std::shared_ptr<fe::graph::Graph> graph_;
};

class BatchnormExecutable final : public ExecutableBase {
 public:
  explicit BatchnormExecutable(const BenchmarkCase& specification) {
    require_tensor_count(specification, 10);
    graph_ = std::make_shared<fe::graph::Graph>();
    graph_->set_name(specification.name)
        .set_io_data_type(data_type(specification.tensors[0].data_type))
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    const TensorSpec cudnn_x = batchnorm_inference_nhwc_tensor(
        specification.tensors[0]);
    auto x = make_tensor(graph_, cudnn_x, "x", false);
    auto scale =
        make_tensor(graph_, specification.tensors[1], "scale", false);
    auto bias =
        make_tensor(graph_, specification.tensors[2], "bias", false);
    auto previous_running_mean = make_tensor(
        graph_, specification.tensors[3], "previous_running_mean", false);
    auto previous_running_variance = make_tensor(
        graph_,
        specification.tensors[4],
        "previous_running_variance",
        false);
    auto epsilon = graph_->tensor(
        static_cast<float>(specification.normalization.epsilon),
        fe::graph::ScalarType::RUNTIME_PARAM);
    auto momentum = graph_->tensor(
        static_cast<float>(specification.normalization.momentum),
        fe::graph::ScalarType::RUNTIME_PARAM);
    epsilon->set_name("epsilon").set_uid(900002);
    momentum->set_name("momentum").set_uid(900003);
    scalars_ = {
        std::pair<std::int64_t, float>{
            900002,
            static_cast<float>(specification.normalization.epsilon)},
        std::pair<std::int64_t, float>{
            900003,
            static_cast<float>(specification.normalization.momentum)},
    };
    fe::graph::Batchnorm_attributes attributes;
    attributes.set_name("batchnorm")
        .set_compute_data_type(fe::DataType_t::FLOAT)
        .set_previous_running_stats(
            previous_running_mean, previous_running_variance, momentum)
        .set_epsilon(epsilon);
    const auto outputs = graph_->batchnorm(x, scale, bias, attributes);

    for (std::size_t output_index = 0;
         output_index < outputs.size();
         ++output_index) {
      TensorSpec expected = output_tensor(specification, output_index);
      if (output_index == 0) {
        expected = batchnorm_inference_nhwc_tensor(expected);
      }
      outputs[output_index]->set_name(
              "output_" + std::to_string(output_index))
          .set_uid(expected.uid)
          .set_data_type(data_type(expected.data_type))
          .set_dim(expected.dimensions)
          .set_stride(expected.strides)
          .set_output(true);
    }

    build_frontend_layout_graph_or_unsupported(
        *graph_, handle(), "BatchNorm training");
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
    for (auto& [uid, value] : scalars_) {
      pointers[uid] = &value;
    }
    check_frontend(graph_->execute(handle(), pointers, workspace),
                   "cuDNN BatchNorm graph execute");
  }

 private:
  std::shared_ptr<fe::graph::Graph> graph_;
  std::array<std::pair<std::int64_t, float>, 2> scalars_{};
};

class LayernormExecutable final : public ExecutableBase {
 public:
  explicit LayernormExecutable(const BenchmarkCase& specification) {
    require_tensor_count(specification, 6);
    graph_ = std::make_shared<fe::graph::Graph>();
    graph_->set_name(specification.name)
        .set_io_data_type(data_type(specification.tensors[0].data_type))
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);
    auto x = make_tensor(graph_, specification.tensors[0], "x", false);
    auto scale =
        make_tensor(graph_, specification.tensors[1], "scale", false);
    auto bias =
        make_tensor(graph_, specification.tensors[2], "bias", false);
    auto epsilon = graph_->tensor(
        static_cast<float>(specification.normalization.epsilon),
        fe::graph::ScalarType::RUNTIME_PARAM);
    epsilon->set_name("epsilon").set_uid(900001);
    epsilon_ = {
        900001,
        static_cast<float>(specification.normalization.epsilon),
    };
    fe::graph::Layernorm_attributes attributes;
    attributes.set_name("layernorm")
        .set_compute_data_type(fe::DataType_t::FLOAT)
        .set_forward_phase(fe::NormFwdPhase_t::TRAINING)
        .set_epsilon(epsilon);
    const auto outputs = graph_->layernorm(x, scale, bias, attributes);
    for (std::size_t output_index = 0;
         output_index < outputs.size();
         ++output_index) {
      const TensorSpec& expected =
          output_tensor(specification, output_index);
      outputs[output_index]->set_name(
              "output_" + std::to_string(output_index))
          .set_uid(expected.uid)
          .set_data_type(data_type(expected.data_type))
          .set_dim(expected.dimensions)
          .set_stride(expected.strides)
          .set_output(true);
    }
    build_frontend_layout_graph_or_unsupported(
        *graph_, handle(), "LayerNorm training");
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
    pointers[epsilon_.first] = &epsilon_.second;
    check_frontend(graph_->execute(handle(), pointers, workspace),
                   "cuDNN LayerNorm graph execute");
  }

 private:
  std::shared_ptr<fe::graph::Graph> graph_;
  std::pair<std::int64_t, float> epsilon_{};
};

class RmsnormExecutable final : public ExecutableBase {
 public:
  explicit RmsnormExecutable(const BenchmarkCase& specification) {
    require_tensor_count(specification, 5);
    graph_ = std::make_shared<fe::graph::Graph>();
    graph_->set_name(specification.name)
        .set_io_data_type(data_type(specification.tensors[0].data_type))
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);
    auto x = make_tensor(graph_, specification.tensors[0], "x", false);
    auto scale =
        make_tensor(graph_, specification.tensors[1], "scale", false);
    auto bias =
        make_tensor(graph_, specification.tensors[2], "bias", false);
    auto epsilon = graph_->tensor(
        static_cast<float>(specification.normalization.epsilon),
        fe::graph::ScalarType::RUNTIME_PARAM);
    epsilon->set_name("epsilon").set_uid(900001);
    epsilon_ = {
        900001,
        static_cast<float>(specification.normalization.epsilon),
    };
    fe::graph::Rmsnorm_attributes attributes;
    attributes.set_name("rmsnorm")
        .set_compute_data_type(fe::DataType_t::FLOAT)
        .set_forward_phase(fe::NormFwdPhase_t::TRAINING)
        .set_bias(bias)
        .set_epsilon(epsilon);
    const auto outputs = graph_->rmsnorm(x, scale, attributes);
    for (std::size_t output_index = 0;
         output_index < outputs.size();
         ++output_index) {
      const TensorSpec& expected =
          output_tensor(specification, output_index);
      outputs[output_index]->set_name(
              "output_" + std::to_string(output_index))
          .set_uid(expected.uid)
          .set_data_type(data_type(expected.data_type))
          .set_dim(expected.dimensions)
          .set_stride(expected.strides)
          .set_output(true);
    }
    build_frontend_layout_graph_or_unsupported(
        *graph_, handle(), "RMSNorm training");
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
    pointers[epsilon_.first] = &epsilon_.second;
    check_frontend(graph_->execute(handle(), pointers, workspace),
                   "cuDNN RMSNorm graph execute");
  }

 private:
  std::shared_ptr<fe::graph::Graph> graph_;
  std::pair<std::int64_t, float> epsilon_{};
};

}  // namespace

std::unique_ptr<BenchmarkExecutable> build_layernorm(
    const BenchmarkCase& specification) {
  return std::make_unique<LayernormExecutable>(specification);
}

std::unique_ptr<BenchmarkExecutable> build_rmsnorm(
    const BenchmarkCase& specification) {
  return std::make_unique<RmsnormExecutable>(specification);
}

std::unique_ptr<BenchmarkExecutable> build_batchnorm(
    const BenchmarkCase& specification) {
  return std::make_unique<BatchnormExecutable>(specification);
}

std::unique_ptr<BenchmarkExecutable> build_batchnorm_inference(
    const BenchmarkCase& specification) {
  return std::make_unique<BatchnormInferenceExecutable>(specification);
}

}  // namespace flagdnn::benchmarking::cudnn_detail
