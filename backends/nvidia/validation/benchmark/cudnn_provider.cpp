/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "cudnn_provider.hpp"

#include "ops.hpp"

#include <cstddef>
#include <memory>
#include <stdexcept>

namespace flagdnn::benchmarking {
namespace {

std::size_t broadcast_axis_count(const TensorSpec& input,
                                 const TensorSpec& output) {
  if (input.dimensions.size() > output.dimensions.size()) {
    return 0;
  }
  const std::size_t leading =
      output.dimensions.size() - input.dimensions.size();
  std::size_t result = 0;
  for (std::size_t axis = 0; axis < leading; ++axis) {
    if (output.dimensions[axis] > 1) {
      ++result;
    }
  }
  for (std::size_t axis = 0; axis < input.dimensions.size(); ++axis) {
    if (input.dimensions[axis] == 1 &&
        output.dimensions[leading + axis] > 1) {
      ++result;
    }
  }
  return result;
}

std::size_t element_count(const TensorSpec& tensor) {
  std::size_t result = 1;
  for (const std::int64_t dimension : tensor.dimensions) {
    result *= static_cast<std::size_t>(dimension);
  }
  return result;
}

bool has_gapped_layout(const TensorSpec& tensor) {
  std::size_t elements = 1;
  std::size_t storage_elements = 1;
  for (std::size_t axis = 0; axis < tensor.dimensions.size(); ++axis) {
    elements *= static_cast<std::size_t>(tensor.dimensions[axis]);
    storage_elements +=
        static_cast<std::size_t>(tensor.dimensions[axis] - 1) *
        static_cast<std::size_t>(tensor.strides[axis]);
  }
  return storage_elements != elements;
}

bool has_any_gapped_layout(const BenchmarkCase& specification) {
  for (const TensorSpec& tensor : specification.tensors) {
    if (has_gapped_layout(tensor)) {
      return true;
    }
  }
  for (const TensorSpec& tensor : specification.graph.intermediates) {
    if (has_gapped_layout(tensor)) {
      return true;
    }
  }
  return false;
}

const TensorSpec* find_graph_tensor(const BenchmarkCase& specification,
                                    std::int64_t uid) {
  for (const TensorSpec& tensor : specification.tensors) {
    if (tensor.uid == uid) {
      return &tensor;
    }
  }
  for (const TensorSpec& tensor : specification.graph.intermediates) {
    if (tensor.uid == uid) {
      return &tensor;
    }
  }
  return nullptr;
}

bool has_same_physical_mapping(const TensorSpec& left,
                               const TensorSpec& right) {
  if (left.dimensions != right.dimensions ||
      left.strides.size() != right.strides.size()) {
    return false;
  }
  for (std::size_t axis = 0; axis < left.dimensions.size(); ++axis) {
    if (left.dimensions[axis] > 1 &&
        left.strides[axis] != right.strides[axis]) {
      return false;
    }
  }
  return true;
}

bool is_broadcast_compatible(const TensorSpec& input,
                             const TensorSpec& output) {
  if (input.dimensions.size() > output.dimensions.size()) {
    return false;
  }
  const std::size_t leading =
      output.dimensions.size() - input.dimensions.size();
  for (std::size_t axis = 0; axis < input.dimensions.size(); ++axis) {
    const std::int64_t input_dimension = input.dimensions[axis];
    const std::int64_t output_dimension =
        output.dimensions[leading + axis];
    if (input_dimension != 1 && input_dimension != output_dimension) {
      return false;
    }
  }
  return true;
}

ProviderCapability graph_capability(const BenchmarkCase& specification) {
  if (specification.graph.nodes.empty()) {
    return ProviderCapability::unsupported(
        "cuDNN graph reference requires at least one node");
  }

  bool has_convolution = false;
  for (const GraphNodeSpec& node : specification.graph.nodes) {
    has_convolution =
        has_convolution || node.operation == Operation::kConvolutionFprop;
  }

  for (const GraphNodeSpec& node : specification.graph.nodes) {
    const TensorSpec* output =
        find_graph_tensor(specification, node.output_uid);
    if (output == nullptr) {
      return ProviderCapability::unsupported(
          "cuDNN graph node output tensor is missing");
    }

    if (node.operation == Operation::kPointwise) {
      const bool unary_relu =
          node.input_uids.size() == 1 &&
          node.pointwise_mode == FLAGDNN_POINTWISE_RELU_FWD;
      const bool binary_arithmetic =
          node.input_uids.size() == 2 &&
          (node.pointwise_mode == FLAGDNN_POINTWISE_ADD ||
           node.pointwise_mode == FLAGDNN_POINTWISE_MUL);
      const bool ternary_select =
          node.input_uids.size() == 3 &&
          node.pointwise_mode == FLAGDNN_POINTWISE_BINARY_SELECT;
      if ((!unary_relu && !binary_arithmetic && !ternary_select) ||
          node.alpha != 1.0) {
        return ProviderCapability::unsupported(
            "cuDNN graph reference supports unary ReLU, binary "
            "ADD/MUL, and ternary BINARY_SELECT nodes");
      }
      for (const std::int64_t input_uid : node.input_uids) {
        const TensorSpec* input =
            find_graph_tensor(specification, input_uid);
        if (input == nullptr) {
          return ProviderCapability::unsupported(
              "cuDNN graph pointwise input tensor is missing");
        }
        const bool compatible =
            has_convolution
                ? is_broadcast_compatible(*input, *output)
                : has_same_physical_mapping(*input, *output);
        if (!compatible) {
          return ProviderCapability::unsupported(
              has_convolution
                  ? "cuDNN graph pointwise broadcast shapes are incompatible"
                  : "cuDNN graph reference requires equal compact "
                    "pointwise layouts");
        }
      }
      continue;
    }

    if (node.operation == Operation::kConvolutionFprop) {
      const bool attributes_valid =
          node.convolution.spatial_rank == 2 &&
          node.convolution.pre_padding.size() == 2 &&
          node.convolution.post_padding.size() == 2 &&
          node.convolution.stride.size() == 2 &&
          node.convolution.dilation.size() == 2;
      if (node.input_uids.size() != 2 ||
          node.convolution.groups != 1 || !attributes_valid) {
        return ProviderCapability::unsupported(
            "cuDNN graph reference requires rank-2 groups=1 convolution "
            "attributes");
      }
      const TensorSpec* input =
          find_graph_tensor(specification, node.input_uids[0]);
      const TensorSpec* filter =
          find_graph_tensor(specification, node.input_uids[1]);
      if (input == nullptr || filter == nullptr ||
          input->dimensions.size() != 4 ||
          filter->dimensions.size() != 4 ||
          output->dimensions.size() != 4) {
        return ProviderCapability::unsupported(
            "cuDNN graph convolution requires rank-4 tensors");
      }
      continue;
    }

    return ProviderCapability::unsupported(
        "cuDNN graph node operation is not implemented");
  }
  return {};
}

bool has_incompatible_packed_boolean_access(
    const BenchmarkCase& specification) {
  for (const TensorSpec& tensor : specification.tensors) {
    if (tensor.data_type != FLAGDNN_DATA_BOOLEAN) {
      continue;
    }
    bool has_compatible_access_width = false;
    for (std::size_t axis = 0; axis < tensor.dimensions.size(); ++axis) {
      if (tensor.strides[axis] == 1 && tensor.dimensions[axis] >= 8 &&
          tensor.dimensions[axis] % 8 == 0) {
        has_compatible_access_width = true;
        break;
      }
    }
    if (!has_compatible_access_width) {
      return true;
    }
  }
  return false;
}

}  // namespace

ProviderCapability CudnnProvider::capability(
    const BenchmarkCase& specification) const {
  if (specification.operation == Operation::kLayernorm &&
      (specification.tensors.size() != 6 ||
       specification.output_count != 3)) {
    return ProviderCapability::unsupported(
        "the cuDNN LayerNorm reference requires three inputs and outputs");
  }
  if (specification.operation == Operation::kRmsnorm &&
      (specification.tensors.size() != 5 ||
       specification.output_count != 2)) {
    return ProviderCapability::unsupported(
        "the cuDNN RMSNorm reference requires three inputs and two outputs");
  }
  if (specification.operation == Operation::kBatchnorm &&
      (specification.tensors.size() != 10 ||
       specification.output_count != 5 ||
       specification.tensors.front().dimensions.size() != 4 ||
       output_tensor(specification).dimensions.size() != 4)) {
    return ProviderCapability::unsupported(
        "the cuDNN BatchNorm training reference adapter currently requires "
        "rank-four logical tensors and five outputs");
  }
  if (specification.operation == Operation::kBatchnormInference &&
      (specification.tensors.size() != 6 ||
       specification.tensors.front().dimensions.size() != 4 ||
       specification.tensors.back().dimensions.size() != 4)) {
    return ProviderCapability::unsupported(
        "the cuDNN BatchNorm Inference reference adapter currently requires "
        "rank-four logical tensors");
  }
  if (specification.operation == Operation::kMatmul &&
      specification.tensors.back().dimensions.size() > 3) {
    return ProviderCapability::unsupported(
        "cuDNN Frontend MatMul does not support this multidimensional "
        "batch broadcast without materialization");
  }
  if (specification.operation == Operation::kGraph) {
    if (has_any_gapped_layout(specification)) {
      return ProviderCapability::unsupported(
          "cuDNN Frontend has no multi-node pointwise engine for gapped layouts");
    }
    return graph_capability(specification);
  }
  if (specification.operation == Operation::kPointwise &&
      specification.pointwise_mode == FLAGDNN_POINTWISE_LOGICAL_NOT) {
    return ProviderCapability::unsupported(
        "cuDNN Frontend has no standalone unary packed-BOOLEAN "
        "LOGICAL_NOT engine on this backend");
  }
  if (specification.operation == Operation::kPointwise &&
      specification.pointwise_mode == FLAGDNN_POINTWISE_IDENTITY &&
      specification.tensors.size() == 2 &&
      element_count(specification.tensors[0]) == 1) {
    return ProviderCapability::unsupported(
        "cuDNN Frontend has no unary Identity engine when C equals one");
  }
  if ((specification.operation == Operation::kRelu ||
       specification.operation == Operation::kPointwise) &&
      (specification.tensors.size() == 2 ||
       specification.tensors.size() == 3) &&
      has_any_gapped_layout(specification)) {
    return ProviderCapability::unsupported(
        "cuDNN Frontend has no pointwise engine for gapped tensor "
        "layouts");
  }
  if (specification.operation == Operation::kPointwise &&
      has_incompatible_packed_boolean_access(specification)) {
    return ProviderCapability::unsupported(
        "cuDNN packed BOOLEAN pointwise tensors require a contiguous "
        "logical extent that is a multiple of eight");
  }
  if (specification.operation == Operation::kPointwise &&
      specification.tensors.size() == 3) {
    if ((specification.pointwise_mode == FLAGDNN_POINTWISE_ADD ||
         specification.pointwise_mode == FLAGDNN_POINTWISE_SUB) &&
        specification.add_alpha != 1.0) {
      return ProviderCapability::unsupported(
          "cuDNN Frontend binary pointwise graph has no right-operand "
          "alpha attribute");
    }
    const TensorSpec& output = specification.tensors[2];
    if (broadcast_axis_count(specification.tensors[0], output) > 0 ||
        broadcast_axis_count(specification.tensors[1], output) > 0) {
      return ProviderCapability::unsupported(
          "cuDNN Frontend has no standalone binary pointwise broadcast "
          "engine for this graph");
    }
    return {};
  }
  if (specification.operation == Operation::kAdd &&
      specification.tensors.size() == 3) {
    const TensorSpec& output = specification.tensors[2];
    if (broadcast_axis_count(specification.tensors[0], output) > 0 ||
        broadcast_axis_count(specification.tensors[1], output) > 0) {
      return ProviderCapability::unsupported(
          "cuDNN Frontend has no standalone binary pointwise broadcast "
          "engine for this graph");
    }
    return {};
  }
  if (specification.operation != Operation::kReduction ||
      specification.tensors.size() != 2) {
    return {};
  }
  const TensorSpec& input = specification.tensors[0];
  if (input.binding_byte_offset % 16 != 0) {
    return ProviderCapability::unsupported(
        "cuDNN Reduction requires a 16-byte-aligned input entrance");
  }
  if (specification.reduction_mode != FLAGDNN_REDUCTION_MUL ||
      input.data_type != FLAGDNN_DATA_BFLOAT16) {
    return {};
  }
  std::int32_t axis = specification.reduction_axis;
  const std::int32_t rank =
      static_cast<std::int32_t>(input.dimensions.size());
  if (axis < 0) {
    axis += rank;
  }
  if (axis >= 0 && axis < rank &&
      input.dimensions[static_cast<std::size_t>(axis)] > 1 &&
      (input.strides[static_cast<std::size_t>(axis)] * 2) % 16 != 0) {
    return ProviderCapability::unsupported(
        "cuDNN has no exact unaligned BF16 MUL reduction fallback");
  }
  return {};
}

std::unique_ptr<BenchmarkExecutable> CudnnProvider::build(
    const BenchmarkCase& specification) {
  switch (specification.operation) {
    case Operation::kRelu:
      return cudnn_detail::build_relu(specification);
    case Operation::kPointwise:
      return cudnn_detail::build_pointwise(specification);
    case Operation::kAdd: {
      BenchmarkCase pointwise = specification;
      pointwise.operation = Operation::kPointwise;
      pointwise.pointwise_mode = FLAGDNN_POINTWISE_ADD;
      pointwise.pointwise_attributes = default_pointwise_attributes();
      return cudnn_detail::build_pointwise(pointwise);
    }
    case Operation::kReduction:
      return cudnn_detail::build_reduction(specification);
    case Operation::kMatmul:
      return cudnn_detail::build_matmul(specification);
    case Operation::kReshape:
      return cudnn_detail::build_reshape(specification);
    case Operation::kTranspose:
      return cudnn_detail::build_transpose(specification);
    case Operation::kSlice:
      return cudnn_detail::build_slice(specification);
    case Operation::kConvolutionFprop:
      return cudnn_detail::build_convolution_fprop(specification);
    case Operation::kConvolutionDgrad:
      return cudnn_detail::build_convolution_dgrad(specification);
    case Operation::kConvolutionWgrad:
      return cudnn_detail::build_convolution_wgrad(specification);
    case Operation::kLayernorm:
      return cudnn_detail::build_layernorm(specification);
    case Operation::kRmsnorm:
      return cudnn_detail::build_rmsnorm(specification);
    case Operation::kBatchnorm:
      return cudnn_detail::build_batchnorm(specification);
    case Operation::kBatchnormInference:
      return cudnn_detail::build_batchnorm_inference(specification);
    case Operation::kGraph:
      return cudnn_detail::build_graph(specification);
  }
  throw std::invalid_argument("unsupported cuDNN reference operation");
}

}  // namespace flagdnn::benchmarking
