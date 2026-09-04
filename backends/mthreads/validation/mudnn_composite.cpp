/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "backends/mthreads/validation/mudnn_composite.hpp"

#include "backends/mthreads/validation/mudnn_convolution.hpp"
#include "backends/mthreads/validation/mudnn_pointwise.hpp"
#include "backends/mthreads/validation/tensor_io.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace flagdnn::validation::mthreads {
namespace {

constexpr std::size_t kCompositeWorkspaceAlignment = 256;

void validate_descriptor(const MudnnAddSquareDescriptor& descriptor) {
  const std::array<const TensorDescriptor*, 3> tensors = {
      &descriptor.left, &descriptor.right, &descriptor.output};
  for (const TensorDescriptor* tensor : tensors) {
    if (tensor->uid <= 0 || tensor->dimensions.empty() ||
        tensor->dimensions.size() != tensor->strides.size()) {
      throw std::invalid_argument(
          "muDNN AddSquare tensor descriptor is invalid");
    }
    static_cast<void>(tensor_io::storage_element_count(*tensor));
    static_cast<void>(tensor_io::data_type_size(tensor->data_type));
  }
  if (descriptor.left.uid == descriptor.right.uid ||
      descriptor.left.uid == descriptor.output.uid ||
      descriptor.right.uid == descriptor.output.uid ||
      descriptor.left.data_type != descriptor.right.data_type ||
      descriptor.left.data_type != descriptor.output.data_type ||
      descriptor.left.dimensions != descriptor.right.dimensions ||
      descriptor.left.dimensions != descriptor.output.dimensions ||
      descriptor.left.strides != descriptor.right.strides ||
      descriptor.left.strides != descriptor.output.strides) {
    throw std::invalid_argument(
        "muDNN AddSquare tensor roles/type/shape/layout are invalid");
  }
  const std::int64_t maximum_uid = std::max(
      {descriptor.left.uid, descriptor.right.uid, descriptor.output.uid});
  if (maximum_uid > std::numeric_limits<std::int64_t>::max() - 2) {
    throw std::overflow_error("muDNN AddSquare internal UID overflows");
  }
}

std::size_t tensor_bytes(const TensorDescriptor& tensor) {
  const std::size_t elements = tensor_io::storage_element_count(tensor);
  const std::size_t width = tensor_io::data_type_size(tensor.data_type);
  if (elements > std::numeric_limits<std::size_t>::max() / width) {
    throw std::overflow_error("muDNN AddSquare workspace size overflows");
  }
  return elements * width;
}

std::size_t checked_add(std::size_t left,
                        std::size_t right,
                        const char* description) {
  if (left > std::numeric_limits<std::size_t>::max() - right) {
    throw std::overflow_error(description);
  }
  return left + right;
}

std::size_t align_up(std::size_t value) {
  const std::size_t remainder = value % kCompositeWorkspaceAlignment;
  return remainder == 0
             ? value
             : checked_add(
                   value,
                   kCompositeWorkspaceAlignment - remainder,
                   "muDNN composite workspace alignment overflows");
}

std::size_t conv_bias_relu_tensor_bytes(const TensorDescriptor& tensor) {
  const std::size_t elements = tensor_io::storage_element_count(tensor);
  const std::size_t width = tensor_io::data_type_size(tensor.data_type);
  if (elements > std::numeric_limits<std::size_t>::max() / width) {
    throw std::overflow_error(
        "muDNN ConvBiasRelu workspace size overflows");
  }
  return elements * width;
}

void validate_conv_bias_relu_descriptor(
    const MudnnConvBiasReluDescriptor& descriptor) {
  const std::array<const TensorDescriptor*, 4> tensors = {
      &descriptor.input,
      &descriptor.filter,
      &descriptor.bias,
      &descriptor.output,
  };
  for (const TensorDescriptor* tensor : tensors) {
    if (tensor->uid <= 0 || tensor->dimensions.size() != 4 ||
        tensor->dimensions.size() != tensor->strides.size()) {
      throw std::invalid_argument(
          "muDNN ConvBiasRelu tensor descriptor is invalid");
    }
    for (std::size_t axis = 0; axis < tensor->dimensions.size(); ++axis) {
      if (tensor->dimensions[axis] <= 0 || tensor->strides[axis] <= 0) {
        throw std::invalid_argument(
            "muDNN ConvBiasRelu tensor layout is invalid");
      }
    }
    static_cast<void>(conv_bias_relu_tensor_bytes(*tensor));
  }
  if (descriptor.input.uid == descriptor.filter.uid ||
      descriptor.input.uid == descriptor.bias.uid ||
      descriptor.input.uid == descriptor.output.uid ||
      descriptor.filter.uid == descriptor.bias.uid ||
      descriptor.filter.uid == descriptor.output.uid ||
      descriptor.bias.uid == descriptor.output.uid ||
      descriptor.input.data_type != descriptor.filter.data_type ||
      descriptor.input.data_type != descriptor.bias.data_type ||
      descriptor.input.data_type != descriptor.output.data_type ||
      descriptor.input.dimensions[1] != descriptor.filter.dimensions[1] ||
      descriptor.output.dimensions[0] != descriptor.input.dimensions[0] ||
      descriptor.output.dimensions[1] != descriptor.filter.dimensions[0] ||
      descriptor.bias.dimensions !=
          std::vector<std::int64_t>(
              {1, descriptor.output.dimensions[1], 1, 1})) {
    throw std::invalid_argument(
        "muDNN ConvBiasRelu tensor roles/type/geometry are invalid");
  }
  const std::int64_t maximum_uid = std::max(
      {descriptor.input.uid,
       descriptor.filter.uid,
       descriptor.bias.uid,
       descriptor.output.uid});
  if (maximum_uid > std::numeric_limits<std::int64_t>::max() - 2) {
    throw std::overflow_error("muDNN ConvBiasRelu internal UID overflows");
  }
}

}  // namespace

struct MudnnAddSquareOperation::Impl {
  explicit Impl(MudnnAddSquareDescriptor value)
      : descriptor(std::move(value)),
        square_descriptor(descriptor.output),
        right_alias(descriptor.right),
        square(make_square_operation()),
        add(make_add_operation()),
        workspace_bytes(tensor_bytes(square_descriptor)) {}

  MudnnPointwiseOperation make_square_operation() {
    validate_descriptor(descriptor);
    const std::int64_t maximum_uid = std::max(
        {descriptor.left.uid, descriptor.right.uid, descriptor.output.uid});
    square_descriptor.uid = maximum_uid + 1;
    square_descriptor.binding_byte_offset = 0;
    right_alias.uid = maximum_uid + 2;
    right_alias.binding_byte_offset = 0;
    return MudnnPointwiseOperation({
        FLAGDNN_POINTWISE_MUL,
        {descriptor.right, right_alias},
        square_descriptor,
        FLAGDNN_POINTWISE_ATTRIBUTES_INITIALIZER,
        1.0,
    });
  }

  MudnnPointwiseOperation make_add_operation() {
    return MudnnPointwiseOperation({
        FLAGDNN_POINTWISE_ADD,
        {descriptor.left, square_descriptor},
        descriptor.output,
        FLAGDNN_POINTWISE_ATTRIBUTES_INITIALIZER,
        1.0,
    });
  }

  MudnnAddSquareDescriptor descriptor;
  TensorDescriptor square_descriptor;
  TensorDescriptor right_alias;
  MudnnPointwiseOperation square;
  MudnnPointwiseOperation add;
  std::size_t workspace_bytes = 0;
};

MudnnAddSquareOperation::MudnnAddSquareOperation(
    MudnnAddSquareDescriptor descriptor)
    : implementation_(std::make_unique<Impl>(std::move(descriptor))) {}

MudnnAddSquareOperation::~MudnnAddSquareOperation() = default;

std::size_t MudnnAddSquareOperation::workspace_size() const noexcept {
  return implementation_->workspace_bytes;
}

void MudnnAddSquareOperation::execute(
    std::span<const flagdnnBinding_t> raw_bindings,
    void* workspace,
    std::size_t workspace_size,
    flagdnnStream_t stream) {
  Impl& state = *implementation_;
  if (raw_bindings.size() != 3 || stream == nullptr ||
      workspace == nullptr || workspace_size != state.workspace_bytes) {
    throw std::invalid_argument("muDNN AddSquare execute arguments are invalid");
  }
  std::unordered_map<std::int64_t, void*> bindings;
  bindings.reserve(raw_bindings.size());
  for (const flagdnnBinding_t& binding : raw_bindings) {
    if (binding.device_pointer == nullptr ||
        !bindings.emplace(binding.uid, binding.device_pointer).second) {
      throw std::invalid_argument("muDNN AddSquare binding is invalid");
    }
  }
  const auto pointer = [&](std::int64_t uid) -> void* {
    const auto found = bindings.find(uid);
    if (found == bindings.end()) {
      throw std::invalid_argument("muDNN AddSquare binding UID is missing");
    }
    return found->second;
  };
  const std::array<flagdnnBinding_t, 3> square_bindings = {{
      {state.descriptor.right.uid, pointer(state.descriptor.right.uid)},
      {state.right_alias.uid, pointer(state.descriptor.right.uid)},
      {state.square_descriptor.uid, workspace},
  }};
  state.square.execute(square_bindings, nullptr, 0, stream);
  const std::array<flagdnnBinding_t, 3> add_bindings = {{
      {state.descriptor.left.uid, pointer(state.descriptor.left.uid)},
      {state.square_descriptor.uid, workspace},
      {state.descriptor.output.uid, pointer(state.descriptor.output.uid)},
  }};
  state.add.execute(add_bindings, nullptr, 0, stream);
}

struct MudnnConvBiasReluOperation::Impl {
  explicit Impl(MudnnConvBiasReluDescriptor value)
      : descriptor(std::move(value)) {
    validate_conv_bias_relu_descriptor(descriptor);
    const std::int64_t maximum_uid = std::max(
        {descriptor.input.uid,
         descriptor.filter.uid,
         descriptor.bias.uid,
         descriptor.output.uid});
    convolution_result = descriptor.output;
    convolution_result.uid = maximum_uid + 1;
    convolution_result.binding_byte_offset = 0;
    biased_result = descriptor.output;
    biased_result.uid = maximum_uid + 2;
    biased_result.binding_byte_offset = 0;

    MudnnConvolutionDescriptor convolution_descriptor;
    convolution_descriptor.direction = MudnnConvolutionDirection::kFprop;
    convolution_descriptor.mode =
        MudnnConvolutionMode::kCrossCorrelation;
    convolution_descriptor.image = descriptor.input;
    convolution_descriptor.filter = descriptor.filter;
    convolution_descriptor.result = convolution_result;
    convolution_descriptor.pre_padding = descriptor.pre_padding;
    convolution_descriptor.post_padding = descriptor.post_padding;
    convolution_descriptor.stride = descriptor.stride;
    convolution_descriptor.dilation = descriptor.dilation;
    convolution_descriptor.groups = descriptor.groups;
    convolution = std::make_unique<MudnnConvolutionOperation>(
        std::move(convolution_descriptor));
    bias_add = std::make_unique<MudnnPointwiseOperation>(
        MudnnPointwiseDescriptor{
            FLAGDNN_POINTWISE_ADD,
            {convolution_result, descriptor.bias},
            biased_result,
            FLAGDNN_POINTWISE_ATTRIBUTES_INITIALIZER,
            1.0,
        });
    relu = std::make_unique<MudnnPointwiseOperation>(
        MudnnPointwiseDescriptor{
            FLAGDNN_POINTWISE_RELU_FWD,
            {biased_result},
            descriptor.output,
            FLAGDNN_POINTWISE_ATTRIBUTES_INITIALIZER,
            1.0,
        });
    if (bias_add->workspace_size() != 0 || relu->workspace_size() != 0) {
      throw std::logic_error(
          "muDNN ConvBiasRelu pointwise workspace must be empty");
    }

    const std::size_t intermediate_bytes =
        conv_bias_relu_tensor_bytes(convolution_result);
    convolution_result_offset = 0;
    biased_result_offset = align_up(intermediate_bytes);
    convolution_workspace_offset = align_up(checked_add(
        biased_result_offset,
        intermediate_bytes,
        "muDNN ConvBiasRelu intermediate workspace overflows"));
    convolution_workspace_bytes = convolution->workspace_size();
    workspace_bytes = checked_add(
        convolution_workspace_offset,
        convolution_workspace_bytes,
        "muDNN ConvBiasRelu total workspace overflows");
  }

  MudnnConvBiasReluDescriptor descriptor;
  TensorDescriptor convolution_result;
  TensorDescriptor biased_result;
  std::unique_ptr<MudnnConvolutionOperation> convolution;
  std::unique_ptr<MudnnPointwiseOperation> bias_add;
  std::unique_ptr<MudnnPointwiseOperation> relu;
  std::size_t convolution_result_offset = 0;
  std::size_t biased_result_offset = 0;
  std::size_t convolution_workspace_offset = 0;
  std::size_t convolution_workspace_bytes = 0;
  std::size_t workspace_bytes = 0;
};

MudnnConvBiasReluOperation::MudnnConvBiasReluOperation(
    MudnnConvBiasReluDescriptor descriptor)
    : implementation_(std::make_unique<Impl>(std::move(descriptor))) {}

MudnnConvBiasReluOperation::~MudnnConvBiasReluOperation() = default;

std::size_t MudnnConvBiasReluOperation::workspace_size() const noexcept {
  return implementation_->workspace_bytes;
}

void MudnnConvBiasReluOperation::execute(
    std::span<const flagdnnBinding_t> raw_bindings,
    void* workspace,
    std::size_t workspace_size,
    flagdnnStream_t stream) {
  Impl& state = *implementation_;
  if (raw_bindings.size() != 4 || stream == nullptr ||
      workspace == nullptr || workspace_size < state.workspace_bytes) {
    throw std::invalid_argument(
        "muDNN ConvBiasRelu execute arguments are invalid");
  }
  std::unordered_map<std::int64_t, void*> bindings;
  bindings.reserve(raw_bindings.size());
  for (const flagdnnBinding_t& binding : raw_bindings) {
    if (binding.device_pointer == nullptr ||
        !bindings.emplace(binding.uid, binding.device_pointer).second) {
      throw std::invalid_argument(
          "muDNN ConvBiasRelu binding is invalid");
    }
  }
  const auto pointer = [&](std::int64_t uid) -> void* {
    const auto found = bindings.find(uid);
    if (found == bindings.end()) {
      throw std::invalid_argument(
          "muDNN ConvBiasRelu binding UID is missing");
    }
    return found->second;
  };
  auto* const workspace_bytes = static_cast<std::byte*>(workspace);
  void* const convolution_result =
      workspace_bytes + state.convolution_result_offset;
  void* const biased_result =
      workspace_bytes + state.biased_result_offset;
  void* const convolution_workspace =
      state.convolution_workspace_bytes == 0
          ? nullptr
          : workspace_bytes + state.convolution_workspace_offset;

  const std::array<flagdnnBinding_t, 3> convolution_bindings = {{
      {state.descriptor.input.uid, pointer(state.descriptor.input.uid)},
      {state.descriptor.filter.uid, pointer(state.descriptor.filter.uid)},
      {state.convolution_result.uid, convolution_result},
  }};
  state.convolution->execute(
      convolution_bindings,
      convolution_workspace,
      state.convolution_workspace_bytes,
      stream);
  const std::array<flagdnnBinding_t, 3> bias_bindings = {{
      {state.convolution_result.uid, convolution_result},
      {state.descriptor.bias.uid, pointer(state.descriptor.bias.uid)},
      {state.biased_result.uid, biased_result},
  }};
  state.bias_add->execute(bias_bindings, nullptr, 0, stream);
  const std::array<flagdnnBinding_t, 2> relu_bindings = {{
      {state.biased_result.uid, biased_result},
      {state.descriptor.output.uid, pointer(state.descriptor.output.uid)},
  }};
  state.relu->execute(relu_bindings, nullptr, 0, stream);
}

}  // namespace flagdnn::validation::mthreads
