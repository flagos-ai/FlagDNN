/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BENCHMARK_COMMON_FLAGDNN_PROVIDER_HPP_
#define FLAGDNN_BENCHMARK_COMMON_FLAGDNN_PROVIDER_HPP_

#include "benchmark_provider.hpp"

#include <flagdnn_frontend.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace flagdnn::benchmarking {
namespace detail {

namespace fe = ::flagdnn_frontend;

inline void check_frontend(fe::error_t status,
                           std::string_view operation) {
  if (status.is_bad()) {
    throw std::runtime_error(
        std::string(operation) + " failed: " + status.get_message());
  }
}

inline fe::DataType_t frontend_data_type(flagdnnDataType_t data_type) {
  switch (data_type) {
    case FLAGDNN_DATA_FLOAT32:
      return fe::DataType_t::FLOAT;
    case FLAGDNN_DATA_FLOAT16:
      return fe::DataType_t::HALF;
    case FLAGDNN_DATA_BFLOAT16:
      return fe::DataType_t::BFLOAT16;
    case FLAGDNN_DATA_BOOLEAN:
      return fe::DataType_t::BOOLEAN;
    case FLAGDNN_DATA_FP8_E4M3:
      return fe::DataType_t::FP8_E4M3;
    case FLAGDNN_DATA_FP8_E5M2:
      return fe::DataType_t::FP8_E5M2;
  }
  throw std::invalid_argument("unsupported FlagDNN frontend data type");
}

inline fe::PointwiseMode_t frontend_pointwise_mode(
    flagdnnPointwiseMode_t mode) {
  switch (mode) {
    case FLAGDNN_POINTWISE_RELU_FWD:
      return fe::PointwiseMode_t::RELU_FWD;
    case FLAGDNN_POINTWISE_ADD:
      return fe::PointwiseMode_t::ADD;
    case FLAGDNN_POINTWISE_SQRT:
      return fe::PointwiseMode_t::SQRT;
    case FLAGDNN_POINTWISE_ERF:
      return fe::PointwiseMode_t::ERF;
    case FLAGDNN_POINTWISE_IDENTITY:
      return fe::PointwiseMode_t::IDENTITY;
    case FLAGDNN_POINTWISE_EXP:
      return fe::PointwiseMode_t::EXP;
    case FLAGDNN_POINTWISE_LOG:
      return fe::PointwiseMode_t::LOG;
    case FLAGDNN_POINTWISE_NEG:
      return fe::PointwiseMode_t::NEG;
    case FLAGDNN_POINTWISE_ABS:
      return fe::PointwiseMode_t::ABS;
    case FLAGDNN_POINTWISE_CEIL:
      return fe::PointwiseMode_t::CEIL;
    case FLAGDNN_POINTWISE_COS:
      return fe::PointwiseMode_t::COS;
    case FLAGDNN_POINTWISE_FLOOR:
      return fe::PointwiseMode_t::FLOOR;
    case FLAGDNN_POINTWISE_RSQRT:
      return fe::PointwiseMode_t::RSQRT;
    case FLAGDNN_POINTWISE_SIN:
      return fe::PointwiseMode_t::SIN;
    case FLAGDNN_POINTWISE_TAN:
      return fe::PointwiseMode_t::TAN;
    case FLAGDNN_POINTWISE_RECIPROCAL:
      return fe::PointwiseMode_t::RECIPROCAL;
    case FLAGDNN_POINTWISE_SUB:
      return fe::PointwiseMode_t::SUB;
    case FLAGDNN_POINTWISE_MUL:
      return fe::PointwiseMode_t::MUL;
    case FLAGDNN_POINTWISE_DIV:
      return fe::PointwiseMode_t::DIV;
    case FLAGDNN_POINTWISE_MIN:
      return fe::PointwiseMode_t::MIN;
    case FLAGDNN_POINTWISE_MAX:
      return fe::PointwiseMode_t::MAX;
    case FLAGDNN_POINTWISE_MOD:
      return fe::PointwiseMode_t::MOD;
    case FLAGDNN_POINTWISE_POW:
      return fe::PointwiseMode_t::POW;
    case FLAGDNN_POINTWISE_LOGICAL_NOT:
      return fe::PointwiseMode_t::LOGICAL_NOT;
    case FLAGDNN_POINTWISE_CMP_EQ:
      return fe::PointwiseMode_t::CMP_EQ;
    case FLAGDNN_POINTWISE_CMP_NEQ:
      return fe::PointwiseMode_t::CMP_NEQ;
    case FLAGDNN_POINTWISE_CMP_GT:
      return fe::PointwiseMode_t::CMP_GT;
    case FLAGDNN_POINTWISE_CMP_GE:
      return fe::PointwiseMode_t::CMP_GE;
    case FLAGDNN_POINTWISE_CMP_LT:
      return fe::PointwiseMode_t::CMP_LT;
    case FLAGDNN_POINTWISE_CMP_LE:
      return fe::PointwiseMode_t::CMP_LE;
    case FLAGDNN_POINTWISE_LOGICAL_AND:
      return fe::PointwiseMode_t::LOGICAL_AND;
    case FLAGDNN_POINTWISE_LOGICAL_OR:
      return fe::PointwiseMode_t::LOGICAL_OR;
    case FLAGDNN_POINTWISE_SIGMOID_BWD:
      return fe::PointwiseMode_t::SIGMOID_BWD;
    case FLAGDNN_POINTWISE_BINARY_SELECT:
      return fe::PointwiseMode_t::BINARY_SELECT;
    case FLAGDNN_POINTWISE_SIGMOID_FWD:
      return fe::PointwiseMode_t::SIGMOID_FWD;
    case FLAGDNN_POINTWISE_TANH_FWD:
      return fe::PointwiseMode_t::TANH_FWD;
    case FLAGDNN_POINTWISE_ELU_FWD:
      return fe::PointwiseMode_t::ELU_FWD;
    case FLAGDNN_POINTWISE_GELU_FWD:
      return fe::PointwiseMode_t::GELU_FWD;
    case FLAGDNN_POINTWISE_SOFTPLUS_FWD:
      return fe::PointwiseMode_t::SOFTPLUS_FWD;
    case FLAGDNN_POINTWISE_SWISH_FWD:
      return fe::PointwiseMode_t::SWISH_FWD;
    case FLAGDNN_POINTWISE_GELU_APPROX_TANH_FWD:
      return fe::PointwiseMode_t::GELU_APPROX_TANH_FWD;
    case FLAGDNN_POINTWISE_NOT_SET:
      break;
  }
  throw std::invalid_argument("unsupported FlagDNN pointwise mode");
}

inline void apply_pointwise_attributes(
    fe::graph::Pointwise_attributes& output,
    const flagdnnPointwiseAttributes_t& input) {
  if ((input.flags & FLAGDNN_POINTWISE_ATTRIBUTE_RELU_LOWER_CLIP) != 0U) {
    output.set_relu_lower_clip(
        static_cast<float>(input.relu_lower_clip));
  }
  if ((input.flags & FLAGDNN_POINTWISE_ATTRIBUTE_RELU_UPPER_CLIP) != 0U) {
    output.set_relu_upper_clip(
        static_cast<float>(input.relu_upper_clip));
  }
  if ((input.flags &
       FLAGDNN_POINTWISE_ATTRIBUTE_RELU_LOWER_CLIP_SLOPE) != 0U) {
    output.set_relu_lower_clip_slope(
        static_cast<float>(input.relu_lower_clip_slope));
  }
  if ((input.flags & FLAGDNN_POINTWISE_ATTRIBUTE_SWISH_BETA) != 0U) {
    output.set_swish_beta(static_cast<float>(input.swish_beta));
  }
  if ((input.flags & FLAGDNN_POINTWISE_ATTRIBUTE_ELU_ALPHA) != 0U) {
    output.set_elu_alpha(static_cast<float>(input.elu_alpha));
  }
  if ((input.flags & FLAGDNN_POINTWISE_ATTRIBUTE_SOFTPLUS_BETA) != 0U) {
    output.set_softplus_beta(static_cast<float>(input.softplus_beta));
  }
}

inline bool pointwise_uses_boolean_compute(flagdnnPointwiseMode_t mode) {
  return mode == FLAGDNN_POINTWISE_LOGICAL_NOT ||
         mode == FLAGDNN_POINTWISE_CMP_EQ ||
         mode == FLAGDNN_POINTWISE_CMP_NEQ ||
         mode == FLAGDNN_POINTWISE_CMP_GT ||
         mode == FLAGDNN_POINTWISE_CMP_GE ||
         mode == FLAGDNN_POINTWISE_CMP_LT ||
         mode == FLAGDNN_POINTWISE_CMP_LE ||
         mode == FLAGDNN_POINTWISE_LOGICAL_AND ||
         mode == FLAGDNN_POINTWISE_LOGICAL_OR;
}

inline fe::ReductionMode_t frontend_reduction_mode(
    flagdnnReductionMode_t mode) {
  switch (mode) {
    case FLAGDNN_REDUCTION_ADD:
      return fe::ReductionMode_t::ADD;
    case FLAGDNN_REDUCTION_AVG:
      return fe::ReductionMode_t::AVG;
    case FLAGDNN_REDUCTION_MUL:
      return fe::ReductionMode_t::MUL;
  }
  throw std::invalid_argument("unsupported FlagDNN reduction mode");
}

inline fe::ConvolutionMode_t frontend_convolution_mode(
    ConvolutionMode mode) {
  switch (mode) {
    case ConvolutionMode::kCrossCorrelation:
      return fe::ConvolutionMode_t::CROSS_CORRELATION;
    case ConvolutionMode::kConvolution:
      return fe::ConvolutionMode_t::CONVOLUTION;
  }
  throw std::invalid_argument("unsupported convolution mode");
}

inline std::shared_ptr<fe::graph::Tensor_attributes> make_tensor(
    const std::shared_ptr<fe::graph::Graph>& graph,
    const TensorSpec& tensor,
    std::string_view name) {
  return graph->tensor(
      fe::graph::Tensor_attributes()
          .set_name(std::string(name))
          .set_uid(tensor.uid)
          .set_data_type(frontend_data_type(tensor.data_type))
          .set_dim(tensor.dimensions)
          .set_stride(tensor.strides));
}

class FlagdnnExecutable final : public BenchmarkExecutable {
 public:
  FlagdnnExecutable(flagdnn::Handle& handle,
                    std::shared_ptr<fe::graph::Graph> graph,
                    std::size_t workspace_size)
      : handle_(handle),
        graph_(std::move(graph)),
        workspace_size_(workspace_size) {}

  [[nodiscard]] std::size_t workspace_size() const noexcept override {
    return workspace_size_;
  }

  void execute(std::span<const flagdnnBinding_t> bindings,
               void* workspace,
               std::size_t workspace_size,
               flagdnnStream_t stream) override {
    check_frontend(
        graph_->execute(
            handle_, bindings, workspace, workspace_size, stream),
        "FlagDNN frontend graph execute");
  }

 private:
  flagdnn::Handle& handle_;
  std::shared_ptr<fe::graph::Graph> graph_;
  std::size_t workspace_size_ = 0;
};

}  // namespace detail

class FlagdnnProvider final : public BenchmarkProvider {
 public:
  explicit FlagdnnProvider(flagdnn::Handle& handle) : handle_(handle) {}

  void set_autotune(bool enabled) noexcept {
    autotune_ = enabled;
  }

  [[nodiscard]] std::string_view name() const noexcept override {
    return "flagdnn";
  }

  [[nodiscard]] std::unique_ptr<BenchmarkExecutable> build(
      const BenchmarkCase& specification) override {
    if (specification.tensors.empty()) {
      throw std::invalid_argument("case has no tensors");
    }

    namespace fe = ::flagdnn_frontend;
    auto graph = std::make_shared<fe::graph::Graph>();
    graph->set_name(specification.name)
        .set_io_data_type(
            detail::frontend_data_type(specification.tensors[0].data_type))
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT)
        .set_autotune(autotune_);

    std::shared_ptr<fe::graph::Tensor_attributes> output;
    std::vector<std::shared_ptr<fe::graph::Tensor_attributes>> outputs;
    switch (specification.operation) {
      case Operation::kRelu: {
        require_tensor_count(specification, 2);
        const auto input = detail::make_tensor(
            graph, specification.tensors[0], "x");
        output = graph->pointwise(
            input,
            fe::graph::Pointwise_attributes()
                .set_name("relu")
                .set_mode(fe::PointwiseMode_t::RELU_FWD));
        break;
      }
      case Operation::kPointwise: {
        fe::graph::Pointwise_attributes attributes;
        attributes.set_name(specification.name)
            .set_mode(detail::frontend_pointwise_mode(
                specification.pointwise_mode))
            .set_compute_data_type(
                detail::pointwise_uses_boolean_compute(
                    specification.pointwise_mode)
                    ? fe::DataType_t::BOOLEAN
                    : fe::DataType_t::FLOAT)
            .set_alpha(specification.add_alpha);
        detail::apply_pointwise_attributes(
            attributes, specification.pointwise_attributes);
        if (specification.tensors.size() == 2) {
          const auto input = detail::make_tensor(
              graph, specification.tensors[0], "x");
          output = graph->pointwise(input, attributes);
        } else if (specification.tensors.size() == 3) {
          const auto left = detail::make_tensor(
              graph, specification.tensors[0], "left");
          const auto right = detail::make_tensor(
              graph, specification.tensors[1], "right");
          output = graph->pointwise(left, right, attributes);
        } else if (specification.tensors.size() == 4) {
          const auto a = detail::make_tensor(
              graph, specification.tensors[0], "a");
          const auto b = detail::make_tensor(
              graph, specification.tensors[1], "b");
          const auto t = detail::make_tensor(
              graph, specification.tensors[2], "t");
          output = graph->pointwise(a, b, t, attributes);
        } else {
          throw std::invalid_argument(
              "pointwise case must have one, two, or three inputs");
        }
        break;
      }
      case Operation::kAdd: {
        require_tensor_count(specification, 3);
        const auto left = detail::make_tensor(
            graph, specification.tensors[0], "left");
        const auto right = detail::make_tensor(
            graph, specification.tensors[1], "right");
        output = graph->pointwise(
            left,
            right,
            fe::graph::Pointwise_attributes()
                .set_name("add")
                .set_mode(fe::PointwiseMode_t::ADD)
                .set_alpha(specification.add_alpha));
        break;
      }
      case Operation::kReduction: {
        require_tensor_count(specification, 2);
        const auto input = detail::make_tensor(
            graph, specification.tensors[0], "input");
        output = graph->reduction(
            input,
            fe::graph::Reduction_attributes()
                .set_name("reduction")
                .set_mode(detail::frontend_reduction_mode(
                    specification.reduction_mode))
                .set_axis(specification.reduction_axis)
                .set_keep_dimensions(specification.keep_dimensions));
        break;
      }
      case Operation::kMatmul: {
        require_tensor_count(specification, 3);
        const auto a = detail::make_tensor(
            graph, specification.tensors[0], "a");
        const auto b = detail::make_tensor(
            graph, specification.tensors[1], "b");
        output = graph->matmul(
            a,
            b,
            fe::graph::Matmul_attributes()
                .set_name("matmul")
                .set_compute_data_type(fe::DataType_t::FLOAT));
        break;
      }
      case Operation::kReshape: {
        require_tensor_count(specification, 2);
        const auto input = detail::make_tensor(
            graph, specification.tensors[0], "input");
        output = graph->reshape(
            input,
            fe::graph::Reshape_attributes()
                .set_name("reshape")
                .set_compute_data_type(fe::DataType_t::FLOAT)
                .set_dim(specification.reshape.dimensions)
                .set_stride(specification.reshape.strides)
                .set_reshape_mode(
                    specification.reshape.logical
                        ? fe::ReshapeMode_t::LOGICAL
                        : fe::ReshapeMode_t::VIEW_ONLY));
        break;
      }
      case Operation::kTranspose: {
        require_tensor_count(specification, 2);
        const auto input = detail::make_tensor(
            graph, specification.tensors[0], "input");
        output = graph->transpose(
            input,
            fe::graph::Transpose_attributes()
                .set_name("transpose")
                .set_compute_data_type(fe::DataType_t::FLOAT)
                .set_permutation(
                    specification.transpose.permutation));
        break;
      }
      case Operation::kSlice: {
        require_tensor_count(specification, 2);
        const auto input = detail::make_tensor(
            graph, specification.tensors[0], "input");
        output = graph->slice(
            input,
            fe::graph::Slice_attributes()
                .set_name("slice")
                .set_compute_data_type(fe::DataType_t::FLOAT)
                .set_slices(specification.slice.slices)
                .set_strides(specification.slice.strides));
        break;
      }
      case Operation::kConvolutionFprop: {
        require_tensor_count(specification, 3);
        const auto input = detail::make_tensor(
            graph, specification.tensors[0], "input");
        const auto filter = detail::make_tensor(
            graph, specification.tensors[1], "filter");
        output = graph->conv_fprop(
            input,
            filter,
            fe::graph::Conv_fprop_attributes()
                .set_name("convolution_fprop")
                .set_pre_padding(
                    specification.convolution.pre_padding)
                .set_post_padding(
                    specification.convolution.post_padding)
                .set_stride(specification.convolution.stride)
                .set_dilation(specification.convolution.dilation)
                .set_convolution_mode(
                    detail::frontend_convolution_mode(
                        specification.convolution.mode))
                .set_groups(specification.convolution.groups));
        break;
      }
      case Operation::kConvolutionDgrad: {
        require_tensor_count(specification, 3);
        const auto loss = detail::make_tensor(
            graph, specification.tensors[0], "dy");
        const auto filter = detail::make_tensor(
            graph, specification.tensors[1], "w");
        output = graph->conv_dgrad(
            loss,
            filter,
            fe::graph::Conv_dgrad_attributes()
                .set_name("convolution_dgrad")
                .set_compute_data_type(fe::DataType_t::FLOAT)
                .set_pre_padding(
                    specification.convolution.pre_padding)
                .set_post_padding(
                    specification.convolution.post_padding)
                .set_stride(specification.convolution.stride)
                .set_dilation(specification.convolution.dilation)
                .set_convolution_mode(
                    detail::frontend_convolution_mode(
                        specification.convolution.mode))
                .set_groups(specification.convolution.groups));
        break;
      }
      case Operation::kConvolutionWgrad: {
        require_tensor_count(specification, 3);
        const auto loss = detail::make_tensor(
            graph, specification.tensors[0], "dy");
        const auto image = detail::make_tensor(
            graph, specification.tensors[1], "x");
        output = graph->conv_wgrad(
            loss,
            image,
            fe::graph::Conv_wgrad_attributes()
                .set_name("convolution_wgrad")
                .set_compute_data_type(fe::DataType_t::FLOAT)
                .set_pre_padding(
                    specification.convolution.pre_padding)
                .set_post_padding(
                    specification.convolution.post_padding)
                .set_stride(specification.convolution.stride)
                .set_dilation(specification.convolution.dilation)
                .set_convolution_mode(
                    detail::frontend_convolution_mode(
                        specification.convolution.mode))
                .set_groups(specification.convolution.groups));
        break;
      }
      case Operation::kLayernorm: {
        require_tensor_count(specification, 6);
        const auto x = detail::make_tensor(
            graph, specification.tensors[0], "x");
        const auto scale = detail::make_tensor(
            graph, specification.tensors[1], "scale");
        const auto bias = detail::make_tensor(
            graph, specification.tensors[2], "bias");
        auto epsilon = graph->tensor(
            static_cast<float>(specification.normalization.epsilon),
            fe::graph::ScalarType::COMPILE_TIME_CONST);
        fe::graph::Layernorm_attributes attributes;
        attributes.set_name("layernorm")
            .set_compute_data_type(fe::DataType_t::FLOAT)
            .set_forward_phase(fe::NormFwdPhase_t::TRAINING)
            .set_epsilon(epsilon);
        const auto normalization_outputs =
            graph->layernorm(x, scale, bias, attributes);
        outputs.assign(
            normalization_outputs.begin(), normalization_outputs.end());
        break;
      }
      case Operation::kRmsnorm: {
        require_tensor_count(specification, 5);
        const auto x = detail::make_tensor(
            graph, specification.tensors[0], "x");
        const auto scale = detail::make_tensor(
            graph, specification.tensors[1], "scale");
        auto bias = detail::make_tensor(
            graph, specification.tensors[2], "bias");
        auto epsilon = graph->tensor(
            static_cast<float>(specification.normalization.epsilon),
            fe::graph::ScalarType::COMPILE_TIME_CONST);
        fe::graph::Rmsnorm_attributes attributes;
        attributes.set_name("rmsnorm")
            .set_compute_data_type(fe::DataType_t::FLOAT)
            .set_forward_phase(fe::NormFwdPhase_t::TRAINING)
            .set_bias(bias)
            .set_epsilon(epsilon);
        const auto normalization_outputs =
            graph->rmsnorm(x, scale, attributes);
        outputs.assign(
            normalization_outputs.begin(), normalization_outputs.end());
        break;
      }

      case Operation::kBatchnorm: {
        require_tensor_count(specification, 10);
        const auto x = detail::make_tensor(
            graph, specification.tensors[0], "x");
        const auto scale = detail::make_tensor(
            graph, specification.tensors[1], "scale");
        const auto bias = detail::make_tensor(
            graph, specification.tensors[2], "bias");
        auto previous_running_mean = detail::make_tensor(
            graph, specification.tensors[3], "previous_running_mean");
        auto previous_running_variance = detail::make_tensor(
            graph, specification.tensors[4], "previous_running_variance");
        auto epsilon = graph->tensor(
            static_cast<float>(specification.normalization.epsilon),
            fe::graph::ScalarType::COMPILE_TIME_CONST);
        auto momentum = graph->tensor(
            static_cast<float>(specification.normalization.momentum),
            fe::graph::ScalarType::COMPILE_TIME_CONST);
        fe::graph::Batchnorm_attributes attributes;
        attributes.set_name("batchnorm")
            .set_compute_data_type(fe::DataType_t::FLOAT)
            .set_previous_running_stats(
                previous_running_mean,
                previous_running_variance,
                momentum)
            .set_epsilon(epsilon);
        const auto batchnorm_outputs =
            graph->batchnorm(x, scale, bias, attributes);
        outputs.assign(
            batchnorm_outputs.begin(), batchnorm_outputs.end());
        break;
      }

      case Operation::kBatchnormInference: {
        require_tensor_count(specification, 6);
        const auto x = detail::make_tensor(
            graph, specification.tensors[0], "x");
        const auto mean = detail::make_tensor(
            graph, specification.tensors[1], "mean");
        const auto inv_variance = detail::make_tensor(
            graph, specification.tensors[2], "inv_variance");
        const auto scale = detail::make_tensor(
            graph, specification.tensors[3], "scale");
        const auto bias = detail::make_tensor(
            graph, specification.tensors[4], "bias");
        output = graph->batchnorm_inference(
            x,
            mean,
            inv_variance,
            scale,
            bias,
            fe::graph::Batchnorm_inference_attributes()
                .set_name("batchnorm_inference")
                .set_compute_data_type(fe::DataType_t::FLOAT));
        break;
      }
      case Operation::kGraph: {
        if (specification.tensors.size() < 2 ||
            specification.graph.nodes.empty()) {
          throw std::invalid_argument(
              "graph case must have external tensors and nodes");
        }
        std::unordered_map<std::int64_t, fe::graph::Graph::Tensor> values;
        for (std::size_t index = 0;
             index < input_tensor_count(specification);
             ++index) {
          const TensorSpec& tensor = specification.tensors[index];
          if (!values.emplace(
                  tensor.uid,
                  detail::make_tensor(graph, tensor, "graph_input"))
                   .second) {
            throw std::invalid_argument(
                "graph case external tensor UID is duplicate");
          }
        }
        const auto tensor_spec = [&](std::int64_t uid) -> const TensorSpec& {
          for (const TensorSpec& tensor : specification.tensors) {
            if (tensor.uid == uid) {
              return tensor;
            }
          }
          for (const TensorSpec& tensor :
               specification.graph.intermediates) {
            if (tensor.uid == uid) {
              return tensor;
            }
          }
          throw std::invalid_argument(
              "graph node references unknown tensor UID");
        };
        const auto value = [&](std::int64_t uid) -> fe::graph::Graph::Tensor {
          const auto found = values.find(uid);
          if (found == values.end()) {
            throw std::invalid_argument(
                "graph nodes are not in dependency order");
          }
          return found->second;
        };
        for (const GraphNodeSpec& node : specification.graph.nodes) {
          fe::graph::Graph::Tensor node_output;
          if (node.operation == Operation::kPointwise) {
            if (node.input_uids.size() != 1 &&
                node.input_uids.size() != 2 &&
                node.input_uids.size() != 3) {
              throw std::invalid_argument(
                  "pointwise graph node requires one, two, or three inputs");
            }
            fe::graph::Pointwise_attributes attributes;
            attributes.set_name(node.name)
                .set_mode(detail::frontend_pointwise_mode(
                    node.pointwise_mode))
                .set_compute_data_type(
                    detail::pointwise_uses_boolean_compute(
                        node.pointwise_mode)
                        ? fe::DataType_t::BOOLEAN
                        : fe::DataType_t::FLOAT)
                .set_alpha(node.alpha);
            detail::apply_pointwise_attributes(
                attributes, node.pointwise_attributes);
            if (node.input_uids.size() == 1) {
              node_output = graph->pointwise(
                  value(node.input_uids[0]), attributes);
            } else if (node.input_uids.size() == 2) {
              node_output = graph->pointwise(
                  value(node.input_uids[0]),
                  value(node.input_uids[1]),
                  attributes);
            } else {
              node_output = graph->pointwise(
                  value(node.input_uids[0]),
                  value(node.input_uids[1]),
                  value(node.input_uids[2]),
                  attributes);
            }
          } else if (node.operation == Operation::kConvolutionFprop) {
            if (node.input_uids.size() != 2) {
              throw std::invalid_argument(
                  "convolution graph node requires input and filter");
            }
            node_output = graph->conv_fprop(
                value(node.input_uids[0]),
                value(node.input_uids[1]),
                fe::graph::Conv_fprop_attributes()
                    .set_name(node.name)
                    .set_pre_padding(node.convolution.pre_padding)
                    .set_post_padding(node.convolution.post_padding)
                    .set_stride(node.convolution.stride)
                    .set_dilation(node.convolution.dilation)
                    .set_groups(node.convolution.groups));
          } else {
            throw std::invalid_argument(
                "FlagDNN graph case node operation is not implemented");
          }
          const TensorSpec& expected_node =
              tensor_spec(node.output_uid);
          node_output->set_name(node.name + "_output")
              .set_uid(expected_node.uid)
              .set_data_type(
                  detail::frontend_data_type(expected_node.data_type))
              .set_dim(expected_node.dimensions)
              .set_stride(expected_node.strides);
          if (!values.emplace(node.output_uid, node_output).second) {
            throw std::invalid_argument(
                "graph node output UID has multiple producers");
          }
        }
        const auto found_output =
            values.find(output_tensor(specification).uid);
        if (found_output == values.end()) {
          throw std::invalid_argument(
              "graph case does not produce its external output");
        }
        output = found_output->second;
        break;
      }
    }

    if (outputs.empty()) {
      if (output == nullptr) {
        throw std::invalid_argument("operation produced no Graph output");
      }
      outputs.push_back(output);
    }
    if (outputs.size() != specification.output_count) {
      throw std::invalid_argument("operation output count is invalid");
    }
    for (std::size_t index = 0; index < outputs.size(); ++index) {
      const TensorSpec& expected = output_tensor(specification, index);
      outputs[index]->set_name(
              outputs.size() == 1
                  ? "output"
                  : "output_" + std::to_string(index))
          .set_uid(expected.uid)
          .set_data_type(detail::frontend_data_type(expected.data_type))
          .set_dim(expected.dimensions)
          .set_stride(expected.strides)
          .set_output(true);
    }
    detail::check_frontend(
        graph->build(handle_, {fe::HeurMode_t::A}),
        "FlagDNN frontend graph build");

    std::int64_t workspace_size = 0;
    detail::check_frontend(
        graph->get_workspace_size(workspace_size),
        "FlagDNN frontend workspace query");
    return std::make_unique<detail::FlagdnnExecutable>(
        handle_, graph, static_cast<std::size_t>(workspace_size));
  }

 private:
  static void require_tensor_count(const BenchmarkCase& specification,
                                   std::size_t expected) {
    if (specification.tensors.size() != expected) {
      throw std::invalid_argument("case tensor count is invalid");
    }
  }

  flagdnn::Handle& handle_;
  bool autotune_ = false;
};

}  // namespace flagdnn::benchmarking

#endif  // FLAGDNN_BENCHMARK_COMMON_FLAGDNN_PROVIDER_HPP_
