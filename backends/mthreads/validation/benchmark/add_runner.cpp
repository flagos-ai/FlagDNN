/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/runner.hpp"

#include "backends/mthreads/validation/mudnn_add.hpp"
#include "backends/mthreads/validation/mudnn_composite.hpp"
#include "backends/mthreads/validation/mudnn_convolution.hpp"
#include "backends/mthreads/validation/mudnn_layout.hpp"
#include "backends/mthreads/validation/mudnn_matmul.hpp"
#include "backends/mthreads/validation/mudnn_normalization.hpp"
#include "backends/mthreads/validation/mudnn_pointwise.hpp"
#include "backends/mthreads/validation/mudnn_reduction.hpp"
#include "backends/mthreads/validation/musa_driver.hpp"
#include "backends/mthreads/validation/tensor_io.hpp"
#include "common/flagdnn_provider.hpp"

#include <flagdnn/flagdnn.hpp>
#include <musa_runtime_api.h>

#include <unistd.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace flagdnn::benchmarking {
namespace {

namespace mv = validation::mthreads;
namespace io = validation::mthreads::tensor_io;

class BenchmarkCache final {
 public:
  BenchmarkCache() {
    const char* configured =
        std::getenv("FLAGDNN_BENCHMARK_CACHE_DIRECTORY");
    if (configured != nullptr && configured[0] != '\0') {
      path_ = configured;
    } else {
      path_ = std::filesystem::temp_directory_path() /
              ("flagdnn-mthreads-benchmark-cache-" +
               std::to_string(getuid()));
    }
    std::error_code error;
    std::filesystem::create_directories(path_, error);
    if (error) {
      throw std::runtime_error(
          "cannot create mthreads benchmark cache: " + error.message());
    }
  }

  [[nodiscard]] const std::filesystem::path& path() const noexcept {
    return path_;
  }

 private:
  std::filesystem::path path_;
};

mv::TensorDescriptor describe_tensor(const TensorSpec& tensor) {
  return {
      tensor.uid,
      tensor.data_type,
      tensor.dimensions,
      tensor.strides,
      tensor.binding_byte_offset,
  };
}

mv::MudnnAddDescriptor describe_add(
    const BenchmarkCase& specification) {
  if (specification.operation != Operation::kAdd ||
      specification.tensors.size() != 3 ||
      specification.output_count != 1) {
    throw std::invalid_argument(
        "mthreads muDNN benchmark provider requires one Add output");
  }
  return {
      describe_tensor(specification.tensors[0]),
      describe_tensor(specification.tensors[1]),
      describe_tensor(specification.tensors[2]),
      specification.add_alpha,
  };
}

bool has_default_pointwise_attributes(
    const flagdnnPointwiseAttributes_t& attributes) noexcept {
  return attributes.struct_size == sizeof(flagdnnPointwiseAttributes_t) &&
         attributes.version == FLAGDNN_POINTWISE_ATTRIBUTES_VERSION &&
         attributes.flags == 0U && attributes.relu_lower_clip == 0.0 &&
         attributes.relu_upper_clip == 0.0 &&
         attributes.relu_lower_clip_slope == 0.0 &&
         attributes.swish_beta == 1.0 && attributes.elu_alpha == 1.0 &&
         attributes.softplus_beta == 1.0;
}

bool same_tensor_layout(const TensorSpec& left,
                        const TensorSpec& right) noexcept {
  return left.data_type == right.data_type &&
         left.dimensions == right.dimensions &&
         left.strides == right.strides;
}

bool is_add_square_graph(const BenchmarkCase& specification) {
  if (specification.operation != Operation::kGraph ||
      specification.tensors.size() != 3 ||
      specification.output_count != 1 ||
      specification.graph.intermediates.size() != 1 ||
      specification.graph.nodes.size() != 2) {
    return false;
  }

  const TensorSpec& left = specification.tensors[0];
  const TensorSpec& right = specification.tensors[1];
  const TensorSpec& output = specification.tensors[2];
  const TensorSpec& square = specification.graph.intermediates[0];
  const GraphNodeSpec& square_node = specification.graph.nodes[0];
  const GraphNodeSpec& add_node = specification.graph.nodes[1];
  return left.uid > 0 && right.uid > 0 && output.uid > 0 &&
         square.uid > 0 && left.uid != right.uid &&
         left.uid != output.uid &&
         left.uid != square.uid && right.uid != output.uid &&
         right.uid != square.uid && output.uid != square.uid &&
         same_tensor_layout(square, output) &&
         square_node.operation == Operation::kPointwise &&
         square_node.pointwise_mode == FLAGDNN_POINTWISE_MUL &&
         square_node.input_uids ==
             std::vector<std::int64_t>{right.uid, right.uid} &&
         square_node.output_uid == square.uid &&
         std::isfinite(square_node.alpha) && square_node.alpha == 1.0 &&
         has_default_pointwise_attributes(
             square_node.pointwise_attributes) &&
         add_node.operation == Operation::kPointwise &&
         add_node.pointwise_mode == FLAGDNN_POINTWISE_ADD &&
         add_node.input_uids ==
             std::vector<std::int64_t>{left.uid, square.uid} &&
         add_node.output_uid == output.uid &&
         std::isfinite(add_node.alpha) && add_node.alpha == 1.0 &&
         has_default_pointwise_attributes(
             add_node.pointwise_attributes);
}

mv::MudnnAddSquareDescriptor describe_add_square(
    const BenchmarkCase& specification) {
  if (!is_add_square_graph(specification)) {
    throw std::invalid_argument(
        "mthreads muDNN benchmark requires the exact AddSquare graph");
  }
  return {
      describe_tensor(specification.tensors[0]),
      describe_tensor(specification.tensors[1]),
      describe_tensor(specification.tensors[2]),
  };
}

bool is_conv_bias_relu_graph(const BenchmarkCase& specification) {
  if (specification.operation != Operation::kGraph ||
      specification.tensors.size() != 4 ||
      specification.output_count != 1 ||
      specification.graph.intermediates.size() != 2 ||
      specification.graph.nodes.size() != 3) {
    return false;
  }
  const TensorSpec& input = specification.tensors[0];
  const TensorSpec& filter = specification.tensors[1];
  const TensorSpec& bias = specification.tensors[2];
  const TensorSpec& output = specification.tensors[3];
  const TensorSpec& convolution = specification.graph.intermediates[0];
  const TensorSpec& biased = specification.graph.intermediates[1];
  const std::array<std::int64_t, 6> uids = {
      input.uid,
      filter.uid,
      bias.uid,
      output.uid,
      convolution.uid,
      biased.uid,
  };
  for (std::size_t left = 0; left < uids.size(); ++left) {
    if (uids[left] <= 0) {
      return false;
    }
    for (std::size_t right = left + 1; right < uids.size(); ++right) {
      if (uids[left] == uids[right]) {
        return false;
      }
    }
  }
  if (input.dimensions.size() != 4 || filter.dimensions.size() != 4 ||
      bias.dimensions.size() != 4 || output.dimensions.size() != 4 ||
      input.data_type != filter.data_type ||
      input.data_type != bias.data_type ||
      input.data_type != output.data_type ||
      !same_tensor_layout(convolution, output) ||
      !same_tensor_layout(biased, output) ||
      input.dimensions[1] <= 0 || filter.dimensions[0] <= 0 ||
      bias.dimensions !=
          std::vector<std::int64_t>(
              {1, filter.dimensions[0], 1, 1})) {
    return false;
  }

  const GraphNodeSpec& convolution_node =
      specification.graph.nodes[0];
  const GraphNodeSpec& bias_node = specification.graph.nodes[1];
  const GraphNodeSpec& relu_node = specification.graph.nodes[2];
  const ConvolutionAttributes& convolution_attributes =
      convolution_node.convolution;
  if (convolution_attributes.spatial_rank != 2 ||
      convolution_attributes.pre_padding.size() != 2 ||
      convolution_attributes.post_padding.size() != 2 ||
      convolution_attributes.stride.size() != 2 ||
      convolution_attributes.dilation.size() != 2 ||
      convolution_attributes.groups <= 0 ||
      convolution_attributes.mode !=
          ConvolutionMode::kCrossCorrelation) {
    return false;
  }
  for (std::size_t axis = 0; axis < 2; ++axis) {
    if (convolution_attributes.pre_padding[axis] < 0 ||
        convolution_attributes.post_padding[axis] < 0 ||
        convolution_attributes.stride[axis] <= 0 ||
        convolution_attributes.dilation[axis] <= 0) {
      return false;
    }
  }
  return convolution_node.operation == Operation::kConvolutionFprop &&
         convolution_node.input_uids ==
             std::vector<std::int64_t>{input.uid, filter.uid} &&
         convolution_node.output_uid == convolution.uid &&
         bias_node.operation == Operation::kPointwise &&
         bias_node.pointwise_mode == FLAGDNN_POINTWISE_ADD &&
         bias_node.input_uids ==
             std::vector<std::int64_t>{convolution.uid, bias.uid} &&
         bias_node.output_uid == biased.uid &&
         std::isfinite(bias_node.alpha) && bias_node.alpha == 1.0 &&
         has_default_pointwise_attributes(
             bias_node.pointwise_attributes) &&
         relu_node.operation == Operation::kPointwise &&
         relu_node.pointwise_mode == FLAGDNN_POINTWISE_RELU_FWD &&
         relu_node.input_uids == std::vector<std::int64_t>{biased.uid} &&
         relu_node.output_uid == output.uid &&
         std::isfinite(relu_node.alpha) && relu_node.alpha == 1.0 &&
         has_default_pointwise_attributes(
             relu_node.pointwise_attributes);
}

mv::MudnnConvBiasReluDescriptor describe_conv_bias_relu(
    const BenchmarkCase& specification) {
  if (!is_conv_bias_relu_graph(specification)) {
    throw std::invalid_argument(
        "mthreads muDNN benchmark requires the exact ConvBiasRelu graph");
  }
  const ConvolutionAttributes& convolution =
      specification.graph.nodes[0].convolution;
  return {
      describe_tensor(specification.tensors[0]),
      describe_tensor(specification.tensors[1]),
      describe_tensor(specification.tensors[2]),
      describe_tensor(specification.tensors[3]),
      convolution.pre_padding,
      convolution.post_padding,
      convolution.stride,
      convolution.dilation,
      convolution.groups,
  };
}

bool is_binary_pointwise_mode(flagdnnPointwiseMode_t mode) {
  switch (mode) {
    case FLAGDNN_POINTWISE_ADD:
    case FLAGDNN_POINTWISE_SUB:
    case FLAGDNN_POINTWISE_MUL:
    case FLAGDNN_POINTWISE_DIV:
    case FLAGDNN_POINTWISE_MIN:
    case FLAGDNN_POINTWISE_MAX:
    case FLAGDNN_POINTWISE_MOD:
    case FLAGDNN_POINTWISE_POW:
    case FLAGDNN_POINTWISE_CMP_EQ:
    case FLAGDNN_POINTWISE_CMP_NEQ:
    case FLAGDNN_POINTWISE_CMP_GT:
    case FLAGDNN_POINTWISE_CMP_GE:
    case FLAGDNN_POINTWISE_CMP_LT:
    case FLAGDNN_POINTWISE_CMP_LE:
    case FLAGDNN_POINTWISE_LOGICAL_AND:
    case FLAGDNN_POINTWISE_LOGICAL_OR:
    case FLAGDNN_POINTWISE_SIGMOID_BWD:
      return true;
    default:
      return false;
  }
}

bool is_unary_pointwise_mode(flagdnnPointwiseMode_t mode) {
  switch (mode) {
    case FLAGDNN_POINTWISE_RELU_FWD:
    case FLAGDNN_POINTWISE_SQRT:
    case FLAGDNN_POINTWISE_ERF:
    case FLAGDNN_POINTWISE_IDENTITY:
    case FLAGDNN_POINTWISE_EXP:
    case FLAGDNN_POINTWISE_LOG:
    case FLAGDNN_POINTWISE_NEG:
    case FLAGDNN_POINTWISE_ABS:
    case FLAGDNN_POINTWISE_CEIL:
    case FLAGDNN_POINTWISE_COS:
    case FLAGDNN_POINTWISE_FLOOR:
    case FLAGDNN_POINTWISE_RSQRT:
    case FLAGDNN_POINTWISE_SIN:
    case FLAGDNN_POINTWISE_TAN:
    case FLAGDNN_POINTWISE_RECIPROCAL:
    case FLAGDNN_POINTWISE_LOGICAL_NOT:
    case FLAGDNN_POINTWISE_SIGMOID_FWD:
    case FLAGDNN_POINTWISE_TANH_FWD:
    case FLAGDNN_POINTWISE_ELU_FWD:
    case FLAGDNN_POINTWISE_GELU_FWD:
    case FLAGDNN_POINTWISE_SOFTPLUS_FWD:
    case FLAGDNN_POINTWISE_SWISH_FWD:
    case FLAGDNN_POINTWISE_GELU_APPROX_TANH_FWD:
      return true;
    default:
      return false;
  }
}

bool is_supported_pointwise_mode(flagdnnPointwiseMode_t mode) {
  return is_binary_pointwise_mode(mode) ||
         is_unary_pointwise_mode(mode) ||
         mode == FLAGDNN_POINTWISE_BINARY_SELECT;
}

flagdnnPointwiseMode_t benchmark_pointwise_mode(
    const BenchmarkCase& specification) {
  return specification.operation == Operation::kRelu
             ? FLAGDNN_POINTWISE_RELU_FWD
             : specification.pointwise_mode;
}

mv::MudnnPointwiseDescriptor describe_pointwise(
    const BenchmarkCase& specification) {
  const flagdnnPointwiseMode_t mode =
      benchmark_pointwise_mode(specification);
  const bool unary =
      is_unary_pointwise_mode(mode);
  const std::size_t input_count =
      mode == FLAGDNN_POINTWISE_BINARY_SELECT ? 3U : (unary ? 1U : 2U);
  if ((specification.operation != Operation::kPointwise &&
       specification.operation != Operation::kRelu) ||
      !is_supported_pointwise_mode(mode) ||
      specification.tensors.size() != input_count + 1U ||
      specification.output_count != 1) {
    throw std::invalid_argument(
        "mthreads muDNN benchmark requires one supported pointwise output");
  }
  std::vector<mv::TensorDescriptor> inputs;
  inputs.reserve(input_count);
  for (std::size_t index = 0; index < input_count; ++index) {
    inputs.push_back(describe_tensor(specification.tensors[index]));
  }
  return {
      mode,
      std::move(inputs),
      describe_tensor(specification.tensors[input_count]),
      specification.pointwise_attributes,
      specification.add_alpha,
  };
}

bool is_layout_operation(Operation operation) {
  return operation == Operation::kReshape ||
         operation == Operation::kTranspose ||
         operation == Operation::kSlice;
}

mv::MudnnLayoutMode mudnn_layout_mode(Operation operation) {
  switch (operation) {
    case Operation::kReshape:
      return mv::MudnnLayoutMode::kReshape;
    case Operation::kTranspose:
      return mv::MudnnLayoutMode::kTranspose;
    case Operation::kSlice:
      return mv::MudnnLayoutMode::kSlice;
    default:
      break;
  }
  throw std::invalid_argument(
      "mthreads muDNN benchmark layout operation is unsupported");
}

mv::MudnnLayoutDescriptor describe_layout(
    const BenchmarkCase& specification) {
  if (!is_layout_operation(specification.operation) ||
      specification.tensors.size() != 2 ||
      specification.output_count != 1) {
    throw std::invalid_argument(
        "mthreads muDNN benchmark layout requires one input and output");
  }
  const TensorSpec& output = specification.tensors[1];
  if (specification.operation == Operation::kReshape &&
      (!specification.reshape.logical ||
       specification.reshape.dimensions != output.dimensions ||
       specification.reshape.strides != output.strides)) {
    throw std::invalid_argument(
        "mthreads muDNN benchmark reshape attributes are invalid");
  }
  return {
      mudnn_layout_mode(specification.operation),
      describe_tensor(specification.tensors[0]),
      describe_tensor(output),
      specification.transpose.permutation,
      specification.slice.slices,
      specification.slice.strides,
  };
}

mv::MudnnReductionDescriptor describe_reduction(
    const BenchmarkCase& specification) {
  if (specification.operation != Operation::kReduction ||
      specification.tensors.size() != 2 ||
      specification.output_count != 1) {
    throw std::invalid_argument(
        "mthreads muDNN benchmark Reduction requires one input and output");
  }
  return {
      specification.reduction_mode,
      describe_tensor(specification.tensors[0]),
      describe_tensor(specification.tensors[1]),
      specification.reduction_axis,
      specification.keep_dimensions,
  };
}

mv::MudnnMatmulDescriptor describe_matmul(
    const BenchmarkCase& specification) {
  if (specification.operation != Operation::kMatmul ||
      specification.tensors.size() != 3 ||
      specification.output_count != 1) {
    throw std::invalid_argument(
        "mthreads muDNN benchmark Matmul requires two inputs and one output");
  }
  return {
      describe_tensor(specification.tensors[0]),
      describe_tensor(specification.tensors[1]),
      describe_tensor(specification.tensors[2]),
  };
}

bool is_convolution_operation(Operation operation) {
  return operation == Operation::kConvolutionFprop ||
         operation == Operation::kConvolutionDgrad ||
         operation == Operation::kConvolutionWgrad;
}

mv::MudnnConvolutionDirection mudnn_convolution_direction(
    Operation operation) {
  switch (operation) {
    case Operation::kConvolutionFprop:
      return mv::MudnnConvolutionDirection::kFprop;
    case Operation::kConvolutionDgrad:
      return mv::MudnnConvolutionDirection::kDgrad;
    case Operation::kConvolutionWgrad:
      return mv::MudnnConvolutionDirection::kWgrad;
    default:
      break;
  }
  throw std::invalid_argument(
      "mthreads muDNN benchmark Convolution direction is invalid");
}

mv::MudnnConvolutionMode mudnn_convolution_mode(
    ConvolutionMode mode) {
  switch (mode) {
    case ConvolutionMode::kCrossCorrelation:
      return mv::MudnnConvolutionMode::kCrossCorrelation;
    case ConvolutionMode::kConvolution:
      return mv::MudnnConvolutionMode::kConvolution;
  }
  throw std::invalid_argument(
      "mthreads muDNN benchmark Convolution mode is invalid");
}

mv::MudnnConvolutionDescriptor describe_convolution(
    const BenchmarkCase& specification) {
  if (!is_convolution_operation(specification.operation) ||
      specification.tensors.size() != 3 ||
      specification.output_count != 1) {
    throw std::invalid_argument(
        "mthreads muDNN benchmark Convolution requires two inputs "
        "and one output");
  }
  mv::TensorDescriptor image;
  mv::TensorDescriptor filter;
  mv::TensorDescriptor result;
  switch (specification.operation) {
    case Operation::kConvolutionFprop:
      image = describe_tensor(specification.tensors[0]);
      filter = describe_tensor(specification.tensors[1]);
      result = describe_tensor(specification.tensors[2]);
      break;
    case Operation::kConvolutionDgrad:
      result = describe_tensor(specification.tensors[0]);
      filter = describe_tensor(specification.tensors[1]);
      image = describe_tensor(specification.tensors[2]);
      break;
    case Operation::kConvolutionWgrad:
      result = describe_tensor(specification.tensors[0]);
      image = describe_tensor(specification.tensors[1]);
      filter = describe_tensor(specification.tensors[2]);
      break;
    default:
      throw std::invalid_argument(
          "mthreads muDNN benchmark Convolution operation is invalid");
  }
  return {
      mudnn_convolution_direction(specification.operation),
      mudnn_convolution_mode(specification.convolution.mode),
      std::move(image),
      std::move(filter),
      std::move(result),
      specification.convolution.pre_padding,
      specification.convolution.post_padding,
      specification.convolution.stride,
      specification.convolution.dilation,
      specification.convolution.groups,
  };
}

bool is_normalization_operation(Operation operation) noexcept {
  return operation == Operation::kLayernorm ||
         operation == Operation::kRmsnorm ||
         operation == Operation::kBatchnorm ||
         operation == Operation::kBatchnormInference;
}

void make_contiguous(TensorSpec& tensor) {
  tensor.strides = contiguous_strides(tensor.dimensions);
}

BenchmarkCase mudnn_reference_case(const BenchmarkCase& specification) {
  BenchmarkCase result = specification;
  if (result.operation == Operation::kBatchnorm) {
    if (result.tensors.size() != 10 || result.output_count != 5) {
      throw std::invalid_argument(
          "mthreads muDNN BatchNorm benchmark tensor arity is invalid");
    }
    make_contiguous(result.tensors[0]);
    make_contiguous(result.tensors[5]);
    result.tensors[1].data_type = FLAGDNN_DATA_FLOAT32;
    result.tensors[2].data_type = FLAGDNN_DATA_FLOAT32;
  } else if (result.operation == Operation::kBatchnormInference) {
    if (result.tensors.size() != 6 || result.output_count != 1) {
      throw std::invalid_argument(
          "mthreads muDNN BatchNorm inference benchmark tensor arity "
          "is invalid");
    }
    make_contiguous(result.tensors[0]);
    make_contiguous(result.tensors[5]);
  }
  return result;
}

mv::MudnnLayernormDescriptor describe_layernorm(
    const BenchmarkCase& specification) {
  if (specification.operation != Operation::kLayernorm ||
      specification.tensors.size() != 6 ||
      specification.output_count != 3) {
    throw std::invalid_argument(
        "mthreads muDNN LayerNorm benchmark tensor arity is invalid");
  }
  return {
      describe_tensor(specification.tensors[0]),
      describe_tensor(specification.tensors[1]),
      describe_tensor(specification.tensors[2]),
      describe_tensor(specification.tensors[3]),
      describe_tensor(specification.tensors[4]),
      describe_tensor(specification.tensors[5]),
      specification.normalization.epsilon,
  };
}

mv::MudnnRmsnormDescriptor describe_rmsnorm(
    const BenchmarkCase& specification) {
  if (specification.operation != Operation::kRmsnorm ||
      specification.tensors.size() != 5 ||
      specification.output_count != 2) {
    throw std::invalid_argument(
        "mthreads muDNN RMSNorm benchmark tensor arity is invalid");
  }
  return {
      describe_tensor(specification.tensors[0]),
      describe_tensor(specification.tensors[1]),
      describe_tensor(specification.tensors[2]),
      describe_tensor(specification.tensors[3]),
      describe_tensor(specification.tensors[4]),
      specification.normalization.epsilon,
  };
}

mv::MudnnBatchnormDescriptor describe_batchnorm(
    const BenchmarkCase& specification) {
  if (specification.operation != Operation::kBatchnorm ||
      specification.tensors.size() != 10 ||
      specification.output_count != 5) {
    throw std::invalid_argument(
        "mthreads muDNN BatchNorm benchmark tensor arity is invalid");
  }
  return {
      describe_tensor(specification.tensors[0]),
      describe_tensor(specification.tensors[1]),
      describe_tensor(specification.tensors[2]),
      describe_tensor(specification.tensors[3]),
      describe_tensor(specification.tensors[4]),
      describe_tensor(specification.tensors[5]),
      describe_tensor(specification.tensors[6]),
      describe_tensor(specification.tensors[7]),
      describe_tensor(specification.tensors[8]),
      describe_tensor(specification.tensors[9]),
      specification.normalization.epsilon,
      specification.normalization.momentum,
  };
}

mv::MudnnBatchnormInferenceDescriptor describe_batchnorm_inference(
    const BenchmarkCase& specification) {
  if (specification.operation != Operation::kBatchnormInference ||
      specification.tensors.size() != 6 ||
      specification.output_count != 1) {
    throw std::invalid_argument(
        "mthreads muDNN BatchNorm inference benchmark tensor arity is "
        "invalid");
  }
  return {
      describe_tensor(specification.tensors[0]),
      describe_tensor(specification.tensors[1]),
      describe_tensor(specification.tensors[2]),
      describe_tensor(specification.tensors[3]),
      describe_tensor(specification.tensors[4]),
      describe_tensor(specification.tensors[5]),
  };
}

class MudnnBenchmarkExecutable final : public BenchmarkExecutable {
 public:
  explicit MudnnBenchmarkExecutable(
      const BenchmarkCase& specification) {
    if (specification.operation == Operation::kAdd) {
      add_ = std::make_unique<mv::MudnnAddOperation>(
          describe_add(specification));
    } else if (is_add_square_graph(specification)) {
      add_square_ = std::make_unique<mv::MudnnAddSquareOperation>(
          describe_add_square(specification));
    } else if (is_conv_bias_relu_graph(specification)) {
      conv_bias_relu_ =
          std::make_unique<mv::MudnnConvBiasReluOperation>(
              describe_conv_bias_relu(specification));
    } else if (is_layout_operation(specification.operation)) {
      layout_ = std::make_unique<mv::MudnnLayoutOperation>(
          describe_layout(specification));
    } else if (specification.operation == Operation::kReduction) {
      reduction_ = std::make_unique<mv::MudnnReductionOperation>(
          describe_reduction(specification));
    } else if (specification.operation == Operation::kMatmul) {
      matmul_ = std::make_unique<mv::MudnnMatmulOperation>(
          describe_matmul(specification));
    } else if (is_convolution_operation(specification.operation)) {
      convolution_ =
          std::make_unique<mv::MudnnConvolutionOperation>(
              describe_convolution(specification));
    } else if (specification.operation == Operation::kLayernorm) {
      layernorm_ = std::make_unique<mv::MudnnLayernormOperation>(
          describe_layernorm(specification));
    } else if (specification.operation == Operation::kRmsnorm) {
      rmsnorm_ = std::make_unique<mv::MudnnRmsnormOperation>(
          describe_rmsnorm(specification));
    } else if (specification.operation == Operation::kBatchnorm) {
      batchnorm_ = std::make_unique<mv::MudnnBatchnormOperation>(
          describe_batchnorm(specification));
    } else if (
        specification.operation == Operation::kBatchnormInference) {
      batchnorm_inference_ =
          std::make_unique<mv::MudnnBatchnormInferenceOperation>(
              describe_batchnorm_inference(specification));
    } else {
      pointwise_ = std::make_unique<mv::MudnnPointwiseOperation>(
          describe_pointwise(specification));
    }
  }

  [[nodiscard]] std::size_t workspace_size() const noexcept override {
    if (add_ != nullptr) {
      return add_->workspace_size();
    }
    if (add_square_ != nullptr) {
      return add_square_->workspace_size();
    }
    if (conv_bias_relu_ != nullptr) {
      return conv_bias_relu_->workspace_size();
    }
    if (layout_ != nullptr) {
      return layout_->workspace_size();
    }
    if (reduction_ != nullptr) {
      return reduction_->workspace_size();
    }
    if (convolution_ != nullptr) {
      return convolution_->workspace_size();
    }
    if (layernorm_ != nullptr) {
      return layernorm_->workspace_size();
    }
    if (rmsnorm_ != nullptr) {
      return rmsnorm_->workspace_size();
    }
    if (batchnorm_ != nullptr) {
      return batchnorm_->workspace_size();
    }
    if (batchnorm_inference_ != nullptr) {
      return batchnorm_inference_->workspace_size();
    }
    return matmul_ != nullptr ? matmul_->workspace_size()
                              : pointwise_->workspace_size();
  }

  void execute(std::span<const flagdnnBinding_t> bindings,
               void* workspace,
               std::size_t workspace_size,
               flagdnnStream_t stream) override {
    if (add_ != nullptr) {
      add_->execute(bindings, workspace, workspace_size, stream);
    } else if (add_square_ != nullptr) {
      add_square_->execute(bindings, workspace, workspace_size, stream);
    } else if (conv_bias_relu_ != nullptr) {
      conv_bias_relu_->execute(
          bindings, workspace, workspace_size, stream);
    } else if (layout_ != nullptr) {
      layout_->execute(bindings, workspace, workspace_size, stream);
    } else if (reduction_ != nullptr) {
      reduction_->execute(bindings, workspace, workspace_size, stream);
    } else if (matmul_ != nullptr) {
      matmul_->execute(bindings, workspace, workspace_size, stream);
    } else if (convolution_ != nullptr) {
      convolution_->execute(
          bindings, workspace, workspace_size, stream);
    } else if (layernorm_ != nullptr) {
      layernorm_->execute(bindings, workspace, workspace_size, stream);
    } else if (rmsnorm_ != nullptr) {
      rmsnorm_->execute(bindings, workspace, workspace_size, stream);
    } else if (batchnorm_ != nullptr) {
      batchnorm_->execute(bindings, workspace, workspace_size, stream);
    } else if (batchnorm_inference_ != nullptr) {
      batchnorm_inference_->execute(
          bindings, workspace, workspace_size, stream);
    } else {
      pointwise_->execute(bindings, workspace, workspace_size, stream);
    }
  }

 private:
  std::unique_ptr<mv::MudnnAddOperation> add_;
  std::unique_ptr<mv::MudnnAddSquareOperation> add_square_;
  std::unique_ptr<mv::MudnnConvBiasReluOperation> conv_bias_relu_;
  std::unique_ptr<mv::MudnnLayoutOperation> layout_;
  std::unique_ptr<mv::MudnnPointwiseOperation> pointwise_;
  std::unique_ptr<mv::MudnnReductionOperation> reduction_;
  std::unique_ptr<mv::MudnnMatmulOperation> matmul_;
  std::unique_ptr<mv::MudnnConvolutionOperation> convolution_;
  std::unique_ptr<mv::MudnnLayernormOperation> layernorm_;
  std::unique_ptr<mv::MudnnRmsnormOperation> rmsnorm_;
  std::unique_ptr<mv::MudnnBatchnormOperation> batchnorm_;
  std::unique_ptr<mv::MudnnBatchnormInferenceOperation>
      batchnorm_inference_;
};

class MudnnProvider final : public BenchmarkProvider {
 public:
  [[nodiscard]] std::string_view name() const noexcept override {
    return "mudnn";
  }

  [[nodiscard]] ProviderCapability capability(
      const BenchmarkCase& specification) const override {
    if (is_add_square_graph(specification) ||
        is_conv_bias_relu_graph(specification)) {
      return {};
    }
    if (specification.operation != Operation::kAdd &&
        specification.operation != Operation::kRelu &&
        specification.operation != Operation::kReduction &&
        specification.operation != Operation::kMatmul &&
        !is_normalization_operation(specification.operation) &&
        !is_convolution_operation(specification.operation) &&
        !is_layout_operation(specification.operation) &&
        (specification.operation != Operation::kPointwise ||
         !is_supported_pointwise_mode(specification.pointwise_mode))) {
      return ProviderCapability::unsupported(
          "mthreads muDNN provider does not implement this operation");
    }
    return {};
  }

  [[nodiscard]] std::unique_ptr<BenchmarkExecutable> build(
      const BenchmarkCase& specification) override {
    return std::make_unique<MudnnBenchmarkExecutable>(specification);
  }
};

std::vector<float> make_benchmark_input(
    const mv::TensorDescriptor& tensor,
    std::size_t input_index,
    InputDomain domain) {
  std::vector<float> result(io::element_count(tensor));
  for (std::size_t index = 0; index < result.size(); ++index) {
    const int centered =
        static_cast<int>((index * 17U + input_index * 11U) % 41U) - 20;
    const float real_value =
        static_cast<float>(centered) /
        static_cast<float>(13U + input_index);
    switch (domain) {
      case InputDomain::kReal:
        result[index] = real_value;
        break;
      case InputDomain::kPositive:
        result[index] = std::abs(real_value) + 0.5F;
        break;
      case InputDomain::kScaled:
        result[index] = real_value * 4.0F;
        break;
      case InputDomain::kTan:
        result[index] = static_cast<float>(centered) / 40.0F;
        break;
      case InputDomain::kDivisor:
      case InputDomain::kModulo:
        result[index] = input_index == 1
                            ? std::abs(real_value) + 0.5F
                            : real_value;
        break;
      case InputDomain::kPower:
        result[index] = input_index == 0
                            ? std::abs(real_value) + 0.5F
                            : std::fmod(std::abs(real_value), 2.0F) + 0.125F;
        break;
      case InputDomain::kModuloSigned: {
        constexpr std::array<float, 6> left = {
            -3.0F, -3.0F, 3.0F, 3.0F, -5.5F, 5.5F};
        constexpr std::array<float, 6> right = {
            2.0F, -2.0F, 2.0F, -2.0F, 2.25F, -2.25F};
        result[index] = input_index == 0 ? left[index % left.size()]
                                         : right[index % right.size()];
        break;
      }
      case InputDomain::kComparison: {
        const int base_centered =
            static_cast<int>((index * 17U) % 41U) - 20;
        const float base = static_cast<float>(base_centered) / 13.0F;
        if (input_index == 0 || index % 3U == 0U) {
          result[index] = base;
        } else if (index % 3U == 1U) {
          result[index] = base + 0.25F;
        } else {
          result[index] = base - 0.25F;
        }
        break;
      }
      case InputDomain::kLogical:
        result[index] = input_index == 0
                            ? static_cast<float>((index / 2U) % 2U)
                            : static_cast<float>(index % 2U);
        break;
    }
  }
  return result;
}

std::vector<std::vector<float>> make_logical_inputs(
    const BenchmarkCase& specification) {
  const std::size_t count = input_tensor_count(specification);
  std::vector<std::vector<float>> result;
  result.reserve(count);
  for (std::size_t index = 0; index < count; ++index) {
    const mv::TensorDescriptor tensor =
        describe_tensor(specification.tensors[index]);
    const InputDomain domain =
        specification.input_domains.empty()
            ? specification.input_domain
            : specification.input_domains.at(index);
    std::vector<float> logical =
        make_benchmark_input(tensor, index, domain);
    if (specification.operation == Operation::kReduction &&
        specification.reduction_mode == FLAGDNN_REDUCTION_MUL) {
      for (float& value : logical) {
        value = 1.0F + value * 0.125F;
      }
    }
    const std::vector<float> physical = io::scatter(logical, tensor);
    const std::vector<std::uint8_t> encoded =
        io::encode(physical, tensor.data_type);
    result.push_back(io::gather(
        io::decode(encoded, tensor.data_type), tensor));
  }
  return result;
}

struct PreparedBuffers {
  std::vector<mv::TensorDescriptor> tensors;
  std::size_t input_count = 0;
  std::vector<std::vector<std::uint8_t>> initial_bytes;
  std::vector<std::unique_ptr<mv::DeviceBuffer>> buffers;
  std::vector<flagdnnBinding_t> bindings;
};

PreparedBuffers prepare_buffers(
    const BenchmarkCase& specification,
    const std::vector<std::vector<float>>& logical_inputs,
    mv::Stream& stream) {
  PreparedBuffers result;
  result.input_count = input_tensor_count(specification);
  if (logical_inputs.size() != result.input_count) {
    throw std::invalid_argument(
        "mthreads benchmark logical input arity is invalid");
  }
  result.tensors.reserve(specification.tensors.size());
  result.initial_bytes.reserve(specification.tensors.size());
  result.buffers.reserve(specification.tensors.size());
  result.bindings.reserve(specification.tensors.size());

  for (std::size_t index = 0;
       index < specification.tensors.size(); ++index) {
    mv::TensorDescriptor tensor =
        describe_tensor(specification.tensors[index]);
    std::vector<float> physical;
    if (index < result.input_count) {
      physical = io::scatter(logical_inputs[index], tensor);
    } else {
      physical.assign(
          io::storage_element_count(tensor), io::kPaddingSentinel);
    }
    std::vector<std::uint8_t> encoded =
        io::encode(physical, tensor.data_type);
    if (tensor.binding_byte_offset >
        std::numeric_limits<std::size_t>::max() - encoded.size()) {
      throw std::overflow_error(
          "mthreads benchmark allocation size overflows");
    }
    auto buffer = std::make_unique<mv::DeviceBuffer>(
        tensor.binding_byte_offset + encoded.size());
    buffer->copy_from_host_at(
        encoded.data(),
        encoded.size(),
        tensor.binding_byte_offset,
        stream.get());
    result.bindings.push_back(
        {tensor.uid, buffer->opaque_at(tensor.binding_byte_offset)});
    result.tensors.push_back(std::move(tensor));
    result.initial_bytes.push_back(std::move(encoded));
    result.buffers.push_back(std::move(buffer));
  }
  return result;
}

std::vector<std::uint8_t> read_tensor(
    const PreparedBuffers& prepared,
    std::size_t tensor_index,
    mv::Stream& stream) {
  const mv::TensorDescriptor& tensor =
      prepared.tensors.at(tensor_index);
  std::vector<std::uint8_t> result =
      prepared.initial_bytes.at(tensor_index);
  prepared.buffers.at(tensor_index)
      ->copy_to_host_at(
          result.data(),
          result.size(),
          tensor.binding_byte_offset,
          stream.get());
  stream.synchronize();
  return result;
}

std::vector<float> read_output(
    const PreparedBuffers& prepared,
    std::size_t output_index,
    mv::Stream& stream,
    std::string_view provider) {
  const std::size_t output_count =
      prepared.tensors.size() - prepared.input_count;
  if (output_index >= output_count) {
    throw std::invalid_argument(
        "mthreads benchmark output index is invalid");
  }
  const std::size_t tensor_index = prepared.input_count + output_index;
  const mv::TensorDescriptor& output =
      prepared.tensors.at(tensor_index);
  const std::vector<std::uint8_t> encoded =
      read_tensor(prepared, tensor_index, stream);
  io::require_padding_unchanged(
      std::string(provider) + " output " + std::to_string(output_index),
      encoded,
      output);
  return io::gather(io::decode(encoded, output.data_type), output);
}

std::vector<std::vector<float>> read_outputs(
    const PreparedBuffers& prepared,
    mv::Stream& stream,
    std::string_view provider) {
  const std::size_t output_count =
      prepared.tensors.size() - prepared.input_count;
  std::vector<std::vector<float>> result;
  result.reserve(output_count);
  for (std::size_t index = 0; index < output_count; ++index) {
    result.push_back(read_output(prepared, index, stream, provider));
  }
  return result;
}

void require_inputs_unchanged(
    const PreparedBuffers& prepared,
    mv::Stream& stream,
    std::string_view provider) {
  for (std::size_t index = 0; index < prepared.input_count; ++index) {
    io::require_bytes_equal(
        std::string(provider) + " input " + std::to_string(index),
        read_tensor(prepared, index, stream),
        prepared.initial_bytes[index]);
  }
}

void execute(BenchmarkExecutable& executable,
             std::span<const flagdnnBinding_t> bindings,
             mv::DeviceBuffer& workspace,
             mv::Stream& stream) {
  executable.execute(
      bindings,
      workspace.opaque(),
      executable.workspace_size(),
      stream.opaque());
}

struct Accuracy {
  double maximum_absolute = 0.0;
  double maximum_relative = 0.0;
};

Accuracy compare_outputs(
    std::span<const float> actual,
    std::span<const float> reference,
    const BenchmarkCase& specification,
    std::size_t output_index,
    std::string_view actual_provider,
    std::string_view reference_provider) {
  if (actual.size() != reference.size()) {
    throw std::runtime_error(
        "mthreads benchmark provider output sizes differ");
  }
  Accuracy result;
  for (std::size_t index = 0; index < actual.size(); ++index) {
    const double left = actual[index];
    const double right = reference[index];
    const double absolute = std::abs(left - right);
    const double relative =
        absolute /
        std::max({std::abs(left), std::abs(right), 1.0e-30});
    result.maximum_absolute =
        std::max(result.maximum_absolute, absolute);
    result.maximum_relative =
        std::max(result.maximum_relative, relative);
    if (!std::isfinite(absolute) ||
        (absolute > specification.absolute_tolerance &&
         relative > specification.relative_tolerance)) {
      std::ostringstream message;
      message << specification.name << " output " << output_index
              << " differs at element " << index << ": "
              << actual_provider << '=' << left
              << ", " << reference_provider << '=' << right
              << ", abs=" << absolute << ", rel=" << relative
              << ", atol=" << specification.absolute_tolerance
              << ", rtol=" << specification.relative_tolerance;
      throw std::runtime_error(message.str());
    }
  }
  return result;
}

Accuracy compare_output_sets(
    const std::vector<std::vector<float>>& actual,
    const std::vector<std::vector<float>>& reference,
    const BenchmarkCase& specification,
    std::string_view actual_provider,
    std::string_view reference_provider) {
  if (actual.size() != specification.output_count ||
      reference.size() != specification.output_count) {
    throw std::runtime_error(
        "mthreads benchmark provider output counts differ");
  }
  Accuracy aggregate;
  for (std::size_t index = 0; index < actual.size(); ++index) {
    const Accuracy accuracy = compare_outputs(
        actual[index],
        reference[index],
        specification,
        index,
        actual_provider,
        reference_provider);
    aggregate.maximum_absolute =
        std::max(aggregate.maximum_absolute, accuracy.maximum_absolute);
    aggregate.maximum_relative =
        std::max(aggregate.maximum_relative, accuracy.maximum_relative);
  }
  return aggregate;
}

class CapturedExecutionBatch final {
 public:
  CapturedExecutionBatch(musaStream_t stream,
                         int execution_count,
                         const std::function<void()>& enqueue)
      : execution_count_(execution_count) {
    if (stream == nullptr || execution_count_ <= 0) {
      throw std::invalid_argument(
          "MUSA Graph capture arguments are invalid");
    }
    mv::check_musa(
        musaStreamBeginCapture(
            stream, musaStreamCaptureModeThreadLocal),
        "musaStreamBeginCapture(benchmark)");
    try {
      for (int index = 0; index < execution_count_; ++index) {
        enqueue();
      }
    } catch (...) {
      musaGraph_t abandoned = nullptr;
      static_cast<void>(musaStreamEndCapture(stream, &abandoned));
      if (abandoned != nullptr) {
        static_cast<void>(musaGraphDestroy(abandoned));
      }
      throw;
    }

    musaGraph_t graph = nullptr;
    mv::check_musa(
        musaStreamEndCapture(stream, &graph),
        "musaStreamEndCapture(benchmark)");
    if (graph == nullptr) {
      throw std::runtime_error(
          "MUSA Graph capture returned a null graph");
    }
    try {
      mv::check_musa(
          musaGraphGetNodes(graph, nullptr, &node_count_),
          "musaGraphGetNodes(benchmark)");
      if (node_count_ == 0) {
        throw std::runtime_error(
            "MUSA Graph capture produced no nodes");
      }
      mv::check_musa(
          musaGraphInstantiate(&graph_exec_, graph, 0),
          "musaGraphInstantiate(benchmark)");
    } catch (...) {
      static_cast<void>(musaGraphDestroy(graph));
      throw;
    }
    mv::check_musa(
        musaGraphDestroy(graph), "musaGraphDestroy(benchmark)");
  }

  ~CapturedExecutionBatch() {
    if (graph_exec_ != nullptr) {
      static_cast<void>(musaGraphExecDestroy(graph_exec_));
    }
  }

  CapturedExecutionBatch(const CapturedExecutionBatch&) = delete;
  CapturedExecutionBatch& operator=(
      const CapturedExecutionBatch&) = delete;

  void launch(musaStream_t stream) const {
    mv::check_musa(
        musaGraphLaunch(graph_exec_, stream),
        "musaGraphLaunch(benchmark)");
  }

  [[nodiscard]] int execution_count() const noexcept {
    return execution_count_;
  }

  [[nodiscard]] std::size_t node_count() const noexcept {
    return node_count_;
  }

 private:
  musaGraphExec_t graph_exec_ = nullptr;
  int execution_count_ = 0;
  std::size_t node_count_ = 0;
};

class EventTimer final {
 public:
  EventTimer() {
    mv::check_musa(musaEventCreate(&start_), "musaEventCreate(start)");
    try {
      mv::check_musa(musaEventCreate(&finish_),
                     "musaEventCreate(finish)");
    } catch (...) {
      static_cast<void>(musaEventDestroy(start_));
      start_ = nullptr;
      throw;
    }
  }

  ~EventTimer() {
    if (finish_ != nullptr) {
      static_cast<void>(musaEventDestroy(finish_));
    }
    if (start_ != nullptr) {
      static_cast<void>(musaEventDestroy(start_));
    }
  }

  EventTimer(const EventTimer&) = delete;
  EventTimer& operator=(const EventTimer&) = delete;

  double measure_microseconds(
      musaStream_t stream,
      const CapturedExecutionBatch& batch) {
    mv::check_musa(
        musaEventRecord(start_, stream), "musaEventRecord(start)");
    batch.launch(stream);
    mv::check_musa(
        musaEventRecord(finish_, stream), "musaEventRecord(finish)");
    mv::check_musa(
        musaEventSynchronize(finish_), "musaEventSynchronize(finish)");
    float milliseconds = 0.0F;
    mv::check_musa(
        musaEventElapsedTime(&milliseconds, start_, finish_),
        "musaEventElapsedTime");
    const double per_execution =
        static_cast<double>(milliseconds) * 1000.0 /
        static_cast<double>(batch.execution_count());
    if (!std::isfinite(per_execution) || per_execution <= 0.0) {
      throw std::runtime_error(
          "MUSA Event returned a non-positive benchmark duration");
    }
    return per_execution;
  }

 private:
  musaEvent_t start_ = nullptr;
  musaEvent_t finish_ = nullptr;
};

void warmup(BenchmarkExecutable& executable,
            std::span<const flagdnnBinding_t> bindings,
            mv::DeviceBuffer& workspace,
            mv::Stream& stream,
            int iterations) {
  if (iterations < 0) {
    throw std::invalid_argument(
        "mthreads benchmark warmup iterations are negative");
  }
  for (int index = 0; index < iterations; ++index) {
    execute(executable, bindings, workspace, stream);
  }
  stream.synchronize();
}

double percentile(std::vector<double> values, double fraction) {
  if (values.empty()) {
    throw std::invalid_argument(
        "cannot summarize empty mthreads benchmark samples");
  }
  std::sort(values.begin(), values.end());
  const std::size_t index =
      static_cast<std::size_t>(
          std::ceil(fraction * static_cast<double>(values.size()))) -
      1;
  return values[std::min(index, values.size() - 1)];
}

void emit_samples(std::string_view provider,
                  const BenchmarkCase& specification,
                  const std::vector<double>& samples) {
  std::cout << "{\"schema_version\":1,\"kind\":\"steady_state\","
            << "\"provider\":\"" << provider << "\",\"case\":\""
            << specification.name << "\",\"unit\":\"us\","
            << "\"median\":" << percentile(samples, 0.5)
            << ",\"p90\":" << percentile(samples, 0.9)
            << ",\"samples\":[";
  for (std::size_t index = 0; index < samples.size(); ++index) {
    if (index != 0) {
      std::cout << ',';
    }
    std::cout << samples[index];
  }
  std::cout << "]}\n";
}

void run_case(const BenchmarkCase& specification,
              FlagdnnProvider& flagdnn_provider,
              MudnnProvider& mudnn_provider,
              mv::Stream& stream) {
  const BenchmarkConfig& config = specification.benchmark;
  if (config.warmup_iterations < 0 || config.sample_count <= 0 ||
      config.iterations_per_sample <= 0) {
    throw std::invalid_argument(
        "mthreads benchmark sample configuration is invalid");
  }
  const ProviderCapability capability =
      mudnn_provider.capability(specification);
  if (!capability.supported) {
    throw std::runtime_error(
        "muDNN capability gap is not permitted for mthreads: " +
        capability.reason);
  }

  std::unique_ptr<BenchmarkExecutable> flagdnn =
      flagdnn_provider.build(specification);
  const BenchmarkCase mudnn_specification =
      mudnn_reference_case(specification);
  std::unique_ptr<BenchmarkExecutable> mudnn =
      mudnn_provider.build(mudnn_specification);
  const std::vector<std::vector<float>> logical_inputs =
      make_logical_inputs(specification);
  PreparedBuffers flagdnn_buffers =
      prepare_buffers(specification, logical_inputs, stream);
  PreparedBuffers mudnn_buffers =
      prepare_buffers(mudnn_specification, logical_inputs, stream);
  mv::DeviceBuffer flagdnn_workspace(
      flagdnn->workspace_size(), 256);
  mv::DeviceBuffer mudnn_workspace(mudnn->workspace_size(), 256);
  stream.synchronize();

  flagdnn->prepare(flagdnn_buffers.bindings, stream.opaque());
  mudnn->prepare(mudnn_buffers.bindings, stream.opaque());
  execute(*flagdnn,
          flagdnn_buffers.bindings,
          flagdnn_workspace,
          stream);
  stream.synchronize();
  execute(*mudnn, mudnn_buffers.bindings, mudnn_workspace, stream);
  stream.synchronize();

  const std::vector<std::vector<float>> oracle =
      read_outputs(mudnn_buffers, stream, "muDNN correctness");
  const Accuracy initial_accuracy = compare_output_sets(
      read_outputs(flagdnn_buffers, stream, "FlagDNN correctness"),
      oracle,
      specification,
      "FlagDNN",
      "muDNN");
  require_inputs_unchanged(flagdnn_buffers, stream, "FlagDNN");
  require_inputs_unchanged(mudnn_buffers, stream, "muDNN");
  std::cout << "[accuracy] case=" << specification.name
            << " flagdnn_max_abs="
            << initial_accuracy.maximum_absolute
            << " flagdnn_max_rel="
            << initial_accuracy.maximum_relative << '\n';

  warmup(*flagdnn,
         flagdnn_buffers.bindings,
         flagdnn_workspace,
         stream,
         config.warmup_iterations);
  warmup(*mudnn,
         mudnn_buffers.bindings,
         mudnn_workspace,
         stream,
         config.warmup_iterations);

  CapturedExecutionBatch flagdnn_batch(
      stream.get(), config.iterations_per_sample, [&] {
        execute(*flagdnn,
                flagdnn_buffers.bindings,
                flagdnn_workspace,
                stream);
      });
  CapturedExecutionBatch mudnn_batch(
      stream.get(), config.iterations_per_sample, [&] {
        execute(*mudnn,
                mudnn_buffers.bindings,
                mudnn_workspace,
                stream);
      });
  std::cout << "[capture] case=" << specification.name
            << " provider=flagdnn nodes="
            << flagdnn_batch.node_count()
            << " executions=" << flagdnn_batch.execution_count() << '\n';
  std::cout << "[capture] case=" << specification.name
            << " provider=mudnn nodes=" << mudnn_batch.node_count()
            << " executions=" << mudnn_batch.execution_count() << '\n';

  flagdnn_batch.launch(stream.get());
  stream.synchronize();
  mudnn_batch.launch(stream.get());
  stream.synchronize();

  EventTimer flagdnn_timer;
  EventTimer mudnn_timer;
  std::vector<double> flagdnn_samples;
  std::vector<double> mudnn_samples;
  flagdnn_samples.reserve(
      static_cast<std::size_t>(config.sample_count));
  mudnn_samples.reserve(static_cast<std::size_t>(config.sample_count));
  for (int sample = 0; sample < config.sample_count; ++sample) {
    const auto measure_flagdnn = [&] {
      flagdnn_samples.push_back(
          flagdnn_timer.measure_microseconds(
              stream.get(), flagdnn_batch));
    };
    const auto measure_mudnn = [&] {
      mudnn_samples.push_back(
          mudnn_timer.measure_microseconds(stream.get(), mudnn_batch));
    };
    if (sample % 2 == 0) {
      measure_flagdnn();
      measure_mudnn();
    } else {
      measure_mudnn();
      measure_flagdnn();
    }
  }

  const Accuracy post_accuracy = compare_output_sets(
      read_outputs(flagdnn_buffers, stream, "FlagDNN post-timing"),
      oracle,
      specification,
      "FlagDNN post-timing",
      "muDNN correctness oracle");
  static_cast<void>(compare_output_sets(
      read_outputs(mudnn_buffers, stream, "muDNN post-timing"),
      oracle,
      specification,
      "muDNN post-timing",
      "muDNN correctness oracle"));
  require_inputs_unchanged(flagdnn_buffers, stream, "FlagDNN post-timing");
  require_inputs_unchanged(mudnn_buffers, stream, "muDNN post-timing");
  std::cout << "[postcheck] case=" << specification.name
            << " flagdnn_max_abs=" << post_accuracy.maximum_absolute
            << " flagdnn_max_rel=" << post_accuracy.maximum_relative
            << '\n';

  emit_samples("flagdnn", specification, flagdnn_samples);
  emit_samples("mudnn", specification, mudnn_samples);
  const double flagdnn_median = percentile(flagdnn_samples, 0.5);
  const double mudnn_median = percentile(mudnn_samples, 0.5);
  std::cout << specification.name
            << ": median_us flagdnn=" << flagdnn_median
            << " mudnn=" << mudnn_median
            << " speedup=" << mudnn_median / flagdnn_median << '\n';
}

bool benchmark_autotune_enabled() {
  const char* value = std::getenv("FLAGDNN_BENCHMARK_AUTOTUNE");
  if (value == nullptr || value[0] == '\0' ||
      std::string_view(value) == "0") {
    return false;
  }
  if (std::string_view(value) == "1") {
    return true;
  }
  throw std::invalid_argument(
      "FLAGDNN_BENCHMARK_AUTOTUNE must be 0 or 1");
}

}  // namespace

int run_benchmark_suite(int argc,
                        char** argv,
                        std::span<const BenchmarkCase> cases,
                        std::string_view suite_name) {
  if (argc != 3) {
    std::cerr << "usage: " << suite_name
              << " COMPILER_EXECUTABLE COMPILER_ENTRY\n";
    return 2;
  }
  try {
    std::cout << std::setprecision(12);
    mv::check_musa(musaSetDevice(0), "musaSetDevice");
    mv::Stream stream;
    BenchmarkCache cache;
    flagdnn::Handle handle("mthreads", 0);
    handle.set_compiler(argv[1], argv[2], cache.path().string());
    FlagdnnProvider flagdnn_provider(handle);
    flagdnn_provider.set_autotune(benchmark_autotune_enabled());
    MudnnProvider mudnn_provider;

    const char* case_filter =
        std::getenv("FLAGDNN_BENCHMARK_CASE");
    std::size_t matched = 0;
    for (const BenchmarkCase& specification : cases) {
      if (case_filter != nullptr && case_filter[0] != '\0' &&
          specification.name != case_filter) {
        continue;
      }
      ++matched;
      run_case(
          specification, flagdnn_provider, mudnn_provider, stream);
    }
    if (matched == 0) {
      throw std::invalid_argument(
          "FLAGDNN_BENCHMARK_CASE did not match any mthreads case");
    }
    std::cout << suite_name << ": PASS cases=" << matched
              << " executed=" << matched << " skipped=0\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << suite_name << "_FAILED: " << error.what() << '\n';
    return 1;
  }
}

}  // namespace flagdnn::benchmarking
