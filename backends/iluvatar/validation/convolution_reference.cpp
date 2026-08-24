// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "convolution_reference.hpp"

#include "corex_cudnn_raii.hpp"
#include "corex_cudnn_status.hpp"
#include "reference_tensor.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace flagdnn::iluvatar::validation {
namespace {

constexpr std::size_t kWorkspaceAlignment = 256;

void *binding(std::span<const flagdnnBinding_t> bindings, std::int64_t uid) {
  const auto iterator = std::find_if(
      bindings.begin(), bindings.end(),
      [uid](const flagdnnBinding_t &value) { return value.uid == uid; });
  if (iterator == bindings.end() || iterator->device_pointer == nullptr) {
    throw std::invalid_argument("CoreX cuDNN convolution binding is missing");
  }
  return iterator->device_pointer;
}

void check_cuda(cudaError_t status, const char *operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) +
                             " failed: " + cudaGetErrorString(status));
  }
}

std::size_t checked_add(std::size_t left, std::size_t right,
                        const char *description) {
  if (left > std::numeric_limits<std::size_t>::max() - right) {
    throw std::invalid_argument(std::string(description) + " is too large");
  }
  return left + right;
}

std::size_t checked_multiply(std::size_t left, std::size_t right,
                             const char *description) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
    throw std::invalid_argument(std::string(description) + " is too large");
  }
  return left * right;
}

std::size_t align_up(std::size_t value, std::size_t alignment) {
  const std::size_t remainder = value % alignment;
  return remainder == 0 ? value
                        : checked_add(value, alignment - remainder,
                                      "convolution workspace alignment");
}

std::size_t element_count(const testing::TestTensor &tensor) {
  std::size_t result = 1;
  for (const std::int64_t dimension : tensor.dimensions) {
    if (dimension <= 0) {
      throw std::invalid_argument(
          "CoreX cuDNN convolution dimension is invalid");
    }
    result = checked_multiply(result, static_cast<std::size_t>(dimension),
                              "convolution element count");
  }
  return result;
}

int positive_int(std::int64_t value, const char *description) {
  if (value <= 0 || value > std::numeric_limits<int>::max()) {
    throw std::invalid_argument(std::string(description) +
                                " does not fit the cuDNN int ABI");
  }
  return static_cast<int>(value);
}

int nonnegative_int(std::int64_t value, const char *description) {
  if (value < 0 || value > std::numeric_limits<int>::max()) {
    throw std::invalid_argument(std::string(description) +
                                " does not fit the cuDNN int ABI");
  }
  return static_cast<int>(value);
}

std::vector<std::int64_t>
contiguous_strides(const std::vector<std::int64_t> &dimensions) {
  std::vector<std::int64_t> result(dimensions.size());
  std::int64_t stride = 1;
  for (std::size_t axis = dimensions.size(); axis != 0; --axis) {
    result[axis - 1] = stride;
    if (dimensions[axis - 1] <= 0 ||
        stride >
            std::numeric_limits<std::int64_t>::max() / dimensions[axis - 1]) {
      throw std::invalid_argument("convolution shape is too large");
    }
    stride *= dimensions[axis - 1];
  }
  return result;
}

std::vector<std::int64_t>
channels_last_strides(const std::vector<std::int64_t> &dimensions) {
  if (dimensions.size() < 3 || dimensions.size() > 5) {
    throw std::invalid_argument("convolution channels-last rank is invalid");
  }
  std::vector<std::int64_t> result(dimensions.size());
  result[1] = 1;
  std::int64_t stride = dimensions[1];
  for (std::size_t axis = dimensions.size(); axis != 2; --axis) {
    const std::size_t current = axis - 1;
    result[current] = stride;
    if (dimensions[current] <= 0 ||
        stride >
            std::numeric_limits<std::int64_t>::max() / dimensions[current]) {
      throw std::invalid_argument("convolution shape is too large");
    }
    stride *= dimensions[current];
  }
  result[0] = stride;
  return result;
}

bool is_channels_last(const testing::TestTensor &tensor) {
  return tensor.strides == channels_last_strides(tensor.dimensions);
}

bool is_contiguous(const testing::TestTensor &tensor) {
  return tensor.strides == contiguous_strides(tensor.dimensions);
}

testing::TestTensor contiguous_tensor(const testing::TestTensor &tensor) {
  testing::TestTensor result = tensor;
  result.strides = contiguous_strides(result.dimensions);
  result.binding_byte_offset = 0;
  return result;
}

ReferenceTensor convolution_data_tensor(const testing::TestTensor &tensor) {
  if (tensor.dimensions.size() != 3) {
    return make_reference_tensor(tensor);
  }
  if (!is_contiguous(tensor) && !is_channels_last(tensor)) {
    throw std::invalid_argument("1D convolution tensor layout is unsupported");
  }
  ReferenceTensor result;
  result.uid = tensor.uid;
  result.data_type = corex_cudnn_data_type(tensor.data_type);
  result.byte_offset = tensor.binding_byte_offset;
  result.dimensions = {
      positive_int(tensor.dimensions[0], "convolution batch"),
      positive_int(tensor.dimensions[1], "convolution channels"),
      1,
      positive_int(tensor.dimensions[2], "convolution width"),
  };
  if (is_channels_last(tensor)) {
    result.strides = {
        positive_int(tensor.strides[0], "convolution batch stride"),
        positive_int(tensor.strides[1], "convolution channel stride"),
        positive_int(tensor.strides[0], "convolution height stride"),
        positive_int(tensor.strides[2], "convolution width stride"),
    };
  } else {
    result.strides = {
        positive_int(tensor.strides[0], "convolution batch stride"),
        positive_int(tensor.strides[1], "convolution channel stride"),
        positive_int(tensor.strides[1], "convolution height stride"),
        positive_int(tensor.strides[2], "convolution width stride"),
    };
  }
  return result;
}

std::vector<int> filter_dimensions(const testing::TestTensor &filter) {
  std::vector<int> result;
  result.reserve(std::max<std::size_t>(4, filter.dimensions.size()));
  result.push_back(positive_int(filter.dimensions[0], "filter K"));
  result.push_back(positive_int(filter.dimensions[1], "filter C"));
  if (filter.dimensions.size() == 3) {
    result.push_back(1);
  }
  for (std::size_t axis = 2; axis < filter.dimensions.size(); ++axis) {
    result.push_back(
        positive_int(filter.dimensions[axis], "filter spatial dimension"));
  }
  return result;
}

cudnnTensorFormat_t filter_format(const testing::TestTensor &filter) {
  if (is_channels_last(filter)) {
    return CUDNN_TENSOR_NHWC;
  }
  if (is_contiguous(filter)) {
    return CUDNN_TENSOR_NCHW;
  }
  throw std::invalid_argument("convolution filter layout is unsupported");
}

void set_filter_descriptor(CorexCudnnFilterDescriptor &descriptor,
                           const testing::TestTensor &filter) {
  const std::vector<int> dimensions = filter_dimensions(filter);
  check_cudnn(cudnnSetFilterNdDescriptor(
                  descriptor.get(), corex_cudnn_data_type(filter.data_type),
                  filter_format(filter), static_cast<int>(dimensions.size()),
                  dimensions.data()),
              "cudnnSetFilterNdDescriptor(convolution)");
}

bool has_asymmetric_padding(std::span<const std::int64_t> pre_padding,
                            std::span<const std::int64_t> post_padding) {
  return !std::equal(pre_padding.begin(), pre_padding.end(),
                     post_padding.begin(), post_padding.end());
}

testing::TestTensor padded_input(const testing::TestTensor &input,
                                 std::span<const std::int64_t> pre_padding,
                                 std::span<const std::int64_t> post_padding) {
  testing::TestTensor result = input;
  for (std::size_t axis = 0; axis < pre_padding.size(); ++axis) {
    const std::size_t tensor_axis = axis + 2;
    result.dimensions[tensor_axis] = checked_add(
        checked_add(static_cast<std::size_t>(input.dimensions[tensor_axis]),
                    static_cast<std::size_t>(pre_padding[axis]),
                    "padded convolution dimension"),
        static_cast<std::size_t>(post_padding[axis]),
        "padded convolution dimension");
  }
  result.strides = is_channels_last(input)
                       ? channels_last_strides(result.dimensions)
                       : contiguous_strides(result.dimensions);
  result.binding_byte_offset = 0;
  return result;
}

std::vector<int> spatial_values(std::span<const std::int64_t> values,
                                int leading_value, bool nonnegative) {
  std::vector<int> result;
  result.reserve(std::max<std::size_t>(2, values.size()));
  if (values.size() == 1) {
    result.push_back(leading_value);
  }
  for (const std::int64_t value : values) {
    result.push_back(nonnegative
                         ? nonnegative_int(value, "convolution padding")
                         : positive_int(value, "convolution attribute"));
  }
  return result;
}

void set_convolution_descriptor(CorexCudnnConvolutionDescriptor &descriptor,
                                std::span<const std::int64_t> padding,
                                std::span<const std::int64_t> stride,
                                std::span<const std::int64_t> dilation,
                                std::int64_t groups,
                                testing::ConvolutionMode mode) {
  const std::vector<int> pads = spatial_values(padding, 0, true);
  const std::vector<int> strides = spatial_values(stride, 1, false);
  const std::vector<int> dilations = spatial_values(dilation, 1, false);
  const cudnnConvolutionMode_t cudnn_mode =
      mode == testing::ConvolutionMode::kCrossCorrelation
          ? CUDNN_CROSS_CORRELATION
          : CUDNN_CONVOLUTION;
  check_cudnn(cudnnSetConvolutionNdDescriptor(
                  descriptor.get(), static_cast<int>(pads.size()), pads.data(),
                  strides.data(), dilations.data(), cudnn_mode,
                  CUDNN_DATA_FLOAT),
              "cudnnSetConvolutionNdDescriptor");
  check_cudnn(cudnnSetConvolutionGroupCount(
                  descriptor.get(), positive_int(groups, "group count")),
              "cudnnSetConvolutionGroupCount");
}

std::size_t outer_spatial_count(const testing::TestTensor &tensor) {
  std::size_t result = 1;
  for (std::size_t axis = 2; axis + 1 < tensor.dimensions.size(); ++axis) {
    result = checked_multiply(result,
                              static_cast<std::size_t>(tensor.dimensions[axis]),
                              "convolution outer spatial count");
  }
  return result;
}

std::size_t row_offset(const testing::TestTensor &tensor,
                       const testing::TestTensor &logical_tensor,
                       std::size_t batch, std::size_t channel,
                       std::size_t outer_index,
                       std::span<const std::int64_t> spatial_offsets) {
  std::size_t result =
      checked_multiply(batch, static_cast<std::size_t>(tensor.strides[0]),
                       "convolution row offset");
  result = checked_add(
      result,
      checked_multiply(channel, static_cast<std::size_t>(tensor.strides[1]),
                       "convolution row offset"),
      "convolution row offset");
  for (std::size_t axis = tensor.dimensions.size() - 1; axis != 2; --axis) {
    const std::size_t tensor_axis = axis - 1;
    const std::size_t dimension =
        static_cast<std::size_t>(logical_tensor.dimensions[tensor_axis]);
    const std::size_t coordinate = outer_index % dimension;
    outer_index /= dimension;
    result = checked_add(
        result,
        checked_multiply(coordinate + static_cast<std::size_t>(
                                          spatial_offsets[tensor_axis - 2]),
                         static_cast<std::size_t>(tensor.strides[tensor_axis]),
                         "convolution row offset"),
        "convolution row offset");
  }
  result = checked_add(
      result,
      checked_multiply(static_cast<std::size_t>(spatial_offsets.back()),
                       static_cast<std::size_t>(tensor.strides.back()),
                       "convolution row offset"),
      "convolution row offset");
  return result;
}

void copy_spatial_rows(const void *source,
                       const testing::TestTensor &source_tensor,
                       std::span<const std::int64_t> source_offsets,
                       void *destination,
                       const testing::TestTensor &destination_tensor,
                       std::span<const std::int64_t> destination_offsets,
                       const testing::TestTensor &logical_tensor,
                       flagdnnStream_t stream) {
  if (source_tensor.data_type != destination_tensor.data_type ||
      source_tensor.data_type != logical_tensor.data_type ||
      source_tensor.dimensions.size() != logical_tensor.dimensions.size() ||
      destination_tensor.dimensions.size() !=
          logical_tensor.dimensions.size() ||
      source_offsets.size() + 2 != logical_tensor.dimensions.size() ||
      destination_offsets.size() != source_offsets.size() ||
      is_channels_last(source_tensor) != is_channels_last(destination_tensor)) {
    throw std::invalid_argument(
        "convolution padded copy metadata is inconsistent");
  }
  const bool channels_last = is_channels_last(logical_tensor);
  const std::size_t element_size =
      flagdnn_data_type_size(logical_tensor.data_type);
  const std::size_t batches =
      static_cast<std::size_t>(logical_tensor.dimensions[0]);
  const std::size_t channels =
      static_cast<std::size_t>(logical_tensor.dimensions[1]);
  const std::size_t outer = outer_spatial_count(logical_tensor);
  const std::size_t width =
      static_cast<std::size_t>(logical_tensor.dimensions.back());
  const std::size_t row_elements = checked_multiply(
      width, channels_last ? channels : 1, "convolution row elements");
  const std::size_t row_bytes =
      checked_multiply(row_elements, element_size, "convolution row bytes");
  const auto *source_bytes = static_cast<const std::byte *>(source);
  auto *destination_bytes = static_cast<std::byte *>(destination);
  const std::size_t channel_iterations = channels_last ? 1 : channels;
  for (std::size_t batch = 0; batch < batches; ++batch) {
    for (std::size_t channel = 0; channel < channel_iterations; ++channel) {
      for (std::size_t row = 0; row < outer; ++row) {
        const std::size_t source_offset = row_offset(
            source_tensor, logical_tensor, batch, channel, row, source_offsets);
        const std::size_t destination_offset =
            row_offset(destination_tensor, logical_tensor, batch, channel, row,
                       destination_offsets);
        check_cuda(cudaMemcpyAsync(destination_bytes +
                                       destination_offset * element_size,
                                   source_bytes + source_offset * element_size,
                                   row_bytes, cudaMemcpyDeviceToDevice,
                                   reinterpret_cast<cudaStream_t>(stream)),
                   "cudaMemcpyAsync(convolution padded rows)");
      }
    }
  }
}

class ClassicConvolutionExecutable final
    : public testing::ConvolutionExecutable {
public:
  explicit ClassicConvolutionExecutable(
      const testing::ConvolutionTestCase &test_case)
      : test_case_(test_case),
        explicit_padding_(has_asymmetric_padding(test_case.pre_padding,
                                                 test_case.post_padding)),
        effective_x_(explicit_padding_
                         ? padded_input(test_case.x, test_case.pre_padding,
                                        test_case.post_padding)
                         : test_case.x),
        effective_y_(test_case.direction ==
                                 testing::ConvolutionDirection::kFprop &&
                             test_case.y.dimensions.size() == 3 &&
                             is_channels_last(test_case.y)
                         ? contiguous_tensor(test_case.y)
                         : test_case.y),
        transform_y_(effective_y_.strides != test_case_.y.strides),
        x_descriptor_(convolution_data_tensor(effective_x_)),
        y_descriptor_(convolution_data_tensor(effective_y_)),
        requested_y_descriptor_(convolution_data_tensor(test_case.y)) {
    set_filter_descriptor(w_descriptor_, test_case_.w);
    const std::vector<std::int64_t> convolution_padding(
        test_case_.pre_padding.size(), 0);
    set_convolution_descriptor(
        convolution_descriptor_,
        explicit_padding_
            ? std::span<const std::int64_t>(convolution_padding)
            : std::span<const std::int64_t>(test_case_.pre_padding),
        test_case_.stride, test_case_.dilation, test_case_.groups,
        test_case_.mode);
    switch (test_case_.direction) {
    case testing::ConvolutionDirection::kFprop:
      check_cudnn(cudnnGetConvolutionForwardWorkspaceSize(
                      handle_.get(), x_descriptor_.get(), w_descriptor_.get(),
                      convolution_descriptor_.get(), y_descriptor_.get(),
                      forward_algorithm_, &cudnn_workspace_size_),
                  "cudnnGetConvolutionForwardWorkspaceSize");
      break;
    case testing::ConvolutionDirection::kDgrad:
      check_cudnn(cudnnGetConvolutionBackwardDataWorkspaceSize(
                      handle_.get(), w_descriptor_.get(), y_descriptor_.get(),
                      convolution_descriptor_.get(), x_descriptor_.get(),
                      backward_data_algorithm_, &cudnn_workspace_size_),
                  "cudnnGetConvolutionBackwardDataWorkspaceSize");
      break;
    case testing::ConvolutionDirection::kWgrad:
      check_cudnn(cudnnGetConvolutionBackwardFilterWorkspaceSize(
                      handle_.get(), x_descriptor_.get(), y_descriptor_.get(),
                      convolution_descriptor_.get(), w_descriptor_.get(),
                      backward_filter_algorithm_, &cudnn_workspace_size_),
                  "cudnnGetConvolutionBackwardFilterWorkspaceSize");
      break;
    }
    workspace_size_ = cudnn_workspace_size_;
    if (explicit_padding_) {
      padded_x_offset_ = align_up(workspace_size_, kWorkspaceAlignment);
      const std::size_t padded_bytes =
          checked_multiply(element_count(effective_x_),
                           flagdnn_data_type_size(effective_x_.data_type),
                           "padded convolution bytes");
      workspace_size_ =
          checked_add(padded_x_offset_, padded_bytes, "convolution workspace");
    }
    if (transform_y_) {
      transformed_y_offset_ = align_up(workspace_size_, kWorkspaceAlignment);
      const std::size_t transformed_y_bytes =
          checked_multiply(element_count(effective_y_),
                           flagdnn_data_type_size(effective_y_.data_type),
                           "transformed convolution output bytes");
      workspace_size_ = checked_add(transformed_y_offset_, transformed_y_bytes,
                                    "convolution workspace");
    }
  }

  [[nodiscard]] std::size_t workspace_size() const noexcept override {
    return workspace_size_;
  }

  void execute(std::span<const flagdnnBinding_t> bindings, void *workspace,
               std::size_t workspace_size, flagdnnStream_t stream) override {
    if (workspace_size < workspace_size_ ||
        (workspace_size_ != 0 && workspace == nullptr)) {
      throw std::invalid_argument(
          "CoreX cuDNN convolution workspace is too small");
    }
    handle_.bind_stream(stream);
    auto *scratch = static_cast<std::byte *>(workspace);
    void *x = binding(bindings, test_case_.x.uid);
    void *padded_x = explicit_padding_ ? scratch + padded_x_offset_ : x;
    void *y = binding(bindings, test_case_.y.uid);
    void *effective_y = transform_y_ ? scratch + transformed_y_offset_ : y;
    const std::vector<std::int64_t> zeros(test_case_.pre_padding.size(), 0);
    if (explicit_padding_ &&
        test_case_.direction != testing::ConvolutionDirection::kDgrad) {
      const std::size_t padded_bytes =
          checked_multiply(element_count(effective_x_),
                           flagdnn_data_type_size(effective_x_.data_type),
                           "padded convolution bytes");
      check_cuda(cudaMemsetAsync(padded_x, 0, padded_bytes,
                                 reinterpret_cast<cudaStream_t>(stream)),
                 "cudaMemsetAsync(convolution padded input)");
      copy_spatial_rows(x, test_case_.x, zeros, padded_x, effective_x_,
                        test_case_.pre_padding, test_case_.x, stream);
    }

    const float one = 1.0F;
    const float zero = 0.0F;
    switch (test_case_.direction) {
    case testing::ConvolutionDirection::kFprop:
      check_cudnn(cudnnConvolutionForward(
                      handle_.get(), &one, x_descriptor_.get(), padded_x,
                      w_descriptor_.get(), binding(bindings, test_case_.w.uid),
                      convolution_descriptor_.get(), forward_algorithm_,
                      workspace, cudnn_workspace_size_, &zero,
                      y_descriptor_.get(), effective_y),
                  "cudnnConvolutionForward");
      if (transform_y_) {
        check_cudnn(cudnnTransformTensor(
                        handle_.get(), &one, y_descriptor_.get(), effective_y,
                        &zero, requested_y_descriptor_.get(), y),
                    "cudnnTransformTensor(convolution output)");
      }
      break;
    case testing::ConvolutionDirection::kDgrad:
      check_cudnn(cudnnConvolutionBackwardData(
                      handle_.get(), &one, w_descriptor_.get(),
                      binding(bindings, test_case_.w.uid), y_descriptor_.get(),
                      binding(bindings, test_case_.y.uid),
                      convolution_descriptor_.get(), backward_data_algorithm_,
                      workspace, cudnn_workspace_size_, &zero,
                      x_descriptor_.get(), padded_x),
                  "cudnnConvolutionBackwardData");
      if (explicit_padding_) {
        copy_spatial_rows(padded_x, effective_x_, test_case_.pre_padding, x,
                          test_case_.x, zeros, test_case_.x, stream);
      }
      break;
    case testing::ConvolutionDirection::kWgrad:
      check_cudnn(cudnnConvolutionBackwardFilter(
                      handle_.get(), &one, x_descriptor_.get(), padded_x,
                      y_descriptor_.get(), binding(bindings, test_case_.y.uid),
                      convolution_descriptor_.get(), backward_filter_algorithm_,
                      workspace, cudnn_workspace_size_, &zero,
                      w_descriptor_.get(), binding(bindings, test_case_.w.uid)),
                  "cudnnConvolutionBackwardFilter");
      break;
    }
  }

private:
  testing::ConvolutionTestCase test_case_;
  bool explicit_padding_ = false;
  testing::TestTensor effective_x_;
  testing::TestTensor effective_y_;
  bool transform_y_ = false;
  CorexCudnnHandle handle_;
  CorexCudnnTensorDescriptor x_descriptor_;
  CorexCudnnTensorDescriptor y_descriptor_;
  CorexCudnnTensorDescriptor requested_y_descriptor_;
  CorexCudnnFilterDescriptor w_descriptor_;
  CorexCudnnConvolutionDescriptor convolution_descriptor_;
  cudnnConvolutionFwdAlgo_t forward_algorithm_ =
      CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
  cudnnConvolutionBwdDataAlgo_t backward_data_algorithm_ =
      CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
  // CoreX 4.4 cuDNN ALGO_1 is the qualified classic-API baseline for
  // backward-filter, including cancellation-heavy FP16 reductions.
  cudnnConvolutionBwdFilterAlgo_t backward_filter_algorithm_ =
      CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
  std::size_t cudnn_workspace_size_ = 0;
  std::size_t padded_x_offset_ = 0;
  std::size_t transformed_y_offset_ = 0;
  std::size_t workspace_size_ = 0;
};

class ClassicConvBiasReluExecutable final
    : public testing::CompositeExecutable {
public:
  explicit ClassicConvBiasReluExecutable(
      const testing::ConvBiasReluTestCase &test_case)
      : test_case_(test_case),
        x_descriptor_(convolution_data_tensor(test_case.x)),
        output_descriptor_(convolution_data_tensor(test_case.output)),
        bias_descriptor_(make_reference_tensor(test_case.bias)) {
    set_filter_descriptor(w_descriptor_, test_case_.w);
    set_convolution_descriptor(convolution_descriptor_, test_case_.padding,
                               test_case_.stride, test_case_.dilation, 1,
                               testing::ConvolutionMode::kCrossCorrelation);
    check_cudnn(
        cudnnSetOpTensorDescriptor(add_descriptor_.get(), CUDNN_OP_TENSOR_ADD,
                                   CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN),
        "cudnnSetOpTensorDescriptor(ConvBiasRelu ADD)");
    check_cudnn(cudnnSetActivationDescriptor(relu_descriptor_.get(),
                                             CUDNN_ACTIVATION_RELU,
                                             CUDNN_PROPAGATE_NAN, 0.0),
                "cudnnSetActivationDescriptor(ConvBiasRelu RELU)");
    check_cudnn(cudnnGetConvolutionForwardWorkspaceSize(
                    handle_.get(), x_descriptor_.get(), w_descriptor_.get(),
                    convolution_descriptor_.get(), output_descriptor_.get(),
                    forward_algorithm_, &cudnn_workspace_size_),
                "cudnnGetConvolutionForwardWorkspaceSize(ConvBiasRelu)");
    const std::size_t output_bytes =
        checked_multiply(element_count(test_case_.output),
                         flagdnn_data_type_size(test_case_.output.data_type),
                         "ConvBiasRelu intermediate bytes");
    convolution_output_offset_ =
        align_up(cudnn_workspace_size_, kWorkspaceAlignment);
    biased_output_offset_ =
        align_up(checked_add(convolution_output_offset_, output_bytes,
                             "ConvBiasRelu workspace"),
                 kWorkspaceAlignment);
    workspace_size_ = checked_add(biased_output_offset_, output_bytes,
                                  "ConvBiasRelu workspace");
  }

  [[nodiscard]] std::size_t workspace_size() const noexcept override {
    return workspace_size_;
  }

  void execute(std::span<const flagdnnBinding_t> bindings, void *workspace,
               std::size_t workspace_size, flagdnnStream_t stream) override {
    if (workspace == nullptr || workspace_size < workspace_size_) {
      throw std::invalid_argument(
          "CoreX cuDNN ConvBiasRelu workspace is too small");
    }
    handle_.bind_stream(stream);
    auto *scratch = static_cast<std::byte *>(workspace);
    void *convolution_output = scratch + convolution_output_offset_;
    void *biased_output = scratch + biased_output_offset_;
    void *output = binding(bindings, test_case_.output.uid);
    const float one = 1.0F;
    const float zero = 0.0F;
    check_cudnn(cudnnConvolutionForward(
                    handle_.get(), &one, x_descriptor_.get(),
                    binding(bindings, test_case_.x.uid), w_descriptor_.get(),
                    binding(bindings, test_case_.w.uid),
                    convolution_descriptor_.get(), forward_algorithm_,
                    workspace, cudnn_workspace_size_, &zero,
                    output_descriptor_.get(), convolution_output),
                "cudnnConvolutionForward(ConvBiasRelu)");
    check_cudnn(cudnnOpTensor(handle_.get(), add_descriptor_.get(), &one,
                              output_descriptor_.get(), convolution_output,
                              &one, bias_descriptor_.get(),
                              binding(bindings, test_case_.bias.uid), &zero,
                              output_descriptor_.get(), biased_output),
                "cudnnOpTensor(ConvBiasRelu bias)");
    check_cudnn(cudnnActivationForward(handle_.get(), relu_descriptor_.get(),
                                       &one, output_descriptor_.get(),
                                       biased_output, &zero,
                                       output_descriptor_.get(), output),
                "cudnnActivationForward(ConvBiasRelu RELU)");
  }

private:
  testing::ConvBiasReluTestCase test_case_;
  CorexCudnnHandle handle_;
  CorexCudnnTensorDescriptor x_descriptor_;
  CorexCudnnTensorDescriptor output_descriptor_;
  CorexCudnnTensorDescriptor bias_descriptor_;
  CorexCudnnFilterDescriptor w_descriptor_;
  CorexCudnnConvolutionDescriptor convolution_descriptor_;
  CorexCudnnOpTensorDescriptor add_descriptor_;
  CorexCudnnActivationDescriptor relu_descriptor_;
  cudnnConvolutionFwdAlgo_t forward_algorithm_ =
      CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
  std::size_t cudnn_workspace_size_ = 0;
  std::size_t convolution_output_offset_ = 0;
  std::size_t biased_output_offset_ = 0;
  std::size_t workspace_size_ = 0;
};

} // namespace

std::unique_ptr<testing::ConvolutionExecutable>
make_classic_convolution_reference(
    const testing::ConvolutionTestCase &test_case) {
  return std::make_unique<ClassicConvolutionExecutable>(test_case);
}

std::unique_ptr<testing::CompositeExecutable>
make_classic_conv_bias_relu_reference(
    const testing::ConvBiasReluTestCase &test_case) {
  return std::make_unique<ClassicConvBiasReluExecutable>(test_case);
}

} // namespace flagdnn::iluvatar::validation
