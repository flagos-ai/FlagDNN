// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "common/normalization.hpp"

#include "corex_cudnn_raii.hpp"
#include "corex_cudnn_status.hpp"
#include "reference_tensor.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>

namespace flagdnn::testing {
namespace {

using iluvatar::validation::check_cudnn;
using iluvatar::validation::CorexCudnnHandle;
using iluvatar::validation::CorexCudnnOpTensorDescriptor;
using iluvatar::validation::CorexCudnnTensorDescriptor;
using iluvatar::validation::flagdnn_data_type_size;
using iluvatar::validation::make_reference_tensor;

constexpr std::size_t kWorkspaceAlignment = 256;

void *binding(std::span<const flagdnnBinding_t> bindings, std::int64_t uid) {
  const auto iterator = std::find_if(
      bindings.begin(), bindings.end(),
      [uid](const flagdnnBinding_t &value) { return value.uid == uid; });
  if (iterator == bindings.end() || iterator->device_pointer == nullptr) {
    throw std::invalid_argument("CoreX cuDNN normalization binding is missing");
  }
  return iterator->device_pointer;
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
                                      "normalization workspace alignment");
}

std::size_t element_count(const TestTensor &tensor) {
  std::size_t result = 1;
  for (const std::int64_t dimension : tensor.dimensions) {
    if (dimension <= 0) {
      throw std::invalid_argument(
          "CoreX cuDNN normalization dimension is invalid");
    }
    result = checked_multiply(result, static_cast<std::size_t>(dimension),
                              "normalization element count");
  }
  return result;
}

std::size_t logical_offset(std::size_t logical_index,
                           const TestTensor &tensor) {
  std::size_t result = 0;
  for (std::size_t axis = tensor.dimensions.size(); axis != 0; --axis) {
    const std::size_t current = axis - 1;
    const std::size_t dimension =
        static_cast<std::size_t>(tensor.dimensions[current]);
    const std::size_t coordinate = logical_index % dimension;
    logical_index /= dimension;
    const std::size_t term = checked_multiply(
        coordinate, static_cast<std::size_t>(tensor.strides[current]),
        "normalization logical offset");
    result = checked_add(result, term, "normalization logical offset");
  }
  return result;
}

TestTensor contiguous_like(const TestTensor &source,
                           flagdnnDataType_t data_type) {
  TestTensor result = source;
  result.data_type = data_type;
  result.binding_byte_offset = 0;
  result.strides.resize(result.dimensions.size());
  std::int64_t stride = 1;
  for (std::size_t axis = result.dimensions.size(); axis != 0; --axis) {
    result.strides[axis - 1] = stride;
    if (result.dimensions[axis - 1] <= 0 ||
        stride > std::numeric_limits<std::int64_t>::max() /
                     result.dimensions[axis - 1]) {
      throw std::invalid_argument(
          "CoreX cuDNN normalization shape is too large");
    }
    stride *= result.dimensions[axis - 1];
  }
  return result;
}

TestTensor scalar_like(const TestTensor &source, flagdnnDataType_t data_type) {
  TestTensor result = source;
  result.data_type = data_type;
  result.dimensions.clear();
  result.strides.clear();
  result.binding_byte_offset = 0;
  return result;
}

bool is_contiguous(const TestTensor &tensor) {
  std::int64_t stride = 1;
  for (std::size_t axis = tensor.dimensions.size(); axis != 0; --axis) {
    if (tensor.strides[axis - 1] != stride) {
      return false;
    }
    stride *= tensor.dimensions[axis - 1];
  }
  return true;
}

std::size_t reserve_workspace(std::size_t &cursor, std::size_t bytes) {
  const std::size_t offset = align_up(cursor, kWorkspaceAlignment);
  cursor = checked_add(offset, bytes, "normalization workspace");
  return offset;
}

void transform_logical(CorexCudnnHandle &handle,
                       CorexCudnnTensorDescriptor &source_element,
                       const void *source, const TestTensor &source_tensor,
                       CorexCudnnTensorDescriptor &output_element, void *output,
                       const TestTensor &output_tensor) {
  if (source_tensor.dimensions != output_tensor.dimensions) {
    throw std::invalid_argument(
        "CoreX cuDNN normalization transform shapes differ");
  }
  const std::size_t elements = element_count(source_tensor);
  const std::size_t source_size =
      flagdnn_data_type_size(source_tensor.data_type);
  const std::size_t output_size =
      flagdnn_data_type_size(output_tensor.data_type);
  const auto *source_bytes = static_cast<const std::byte *>(source);
  auto *output_bytes = static_cast<std::byte *>(output);
  const float one = 1.0F;
  const float zero = 0.0F;
  for (std::size_t index = 0; index < elements; ++index) {
    check_cudnn(
        cudnnTransformTensor(
            handle.get(), &one, source_element.get(),
            source_bytes + logical_offset(index, source_tensor) * source_size,
            &zero, output_element.get(),
            output_bytes + logical_offset(index, output_tensor) * output_size),
        "cudnnTransformTensor(normalization scalar sequence)");
  }
}

class BatchnormTrainingExecutable final : public NormalizationExecutable {
public:
  explicit BatchnormTrainingExecutable(const BatchnormTestCase &test_case)
      : test_case_(test_case),
        contiguous_x_(contiguous_like(test_case.x, test_case.x.data_type)),
        contiguous_y_(contiguous_like(test_case.y, test_case.y.data_type)),
        float_parameter_(contiguous_like(test_case.previous_running_mean,
                                         FLAGDNN_DATA_FLOAT32)),
        scalar_x_(scalar_like(test_case.x, test_case.x.data_type)),
        scalar_y_(scalar_like(test_case.y, test_case.y.data_type)),
        x_descriptor_(make_reference_tensor(test_case.x)),
        y_descriptor_(make_reference_tensor(test_case.y)),
        contiguous_x_descriptor_(make_reference_tensor(contiguous_x_)),
        contiguous_y_descriptor_(make_reference_tensor(contiguous_y_)),
        scale_descriptor_(make_reference_tensor(test_case.scale)),
        bias_descriptor_(make_reference_tensor(test_case.bias)),
        parameter_descriptor_(make_reference_tensor(float_parameter_)),
        scalar_x_descriptor_(make_reference_tensor(scalar_x_)),
        scalar_y_descriptor_(make_reference_tensor(scalar_y_)),
        reorder_data_(!is_contiguous(test_case.x) ||
                      !is_contiguous(test_case.y)),
        convert_parameters_(test_case.scale.data_type != FLAGDNN_DATA_FLOAT32) {
    std::size_t cursor = 0;
    const std::size_t parameter_bytes =
        checked_multiply(element_count(float_parameter_), sizeof(float),
                         "BatchNorm parameter bytes");
    if (convert_parameters_) {
      scale_offset_ = reserve_workspace(cursor, parameter_bytes);
      bias_offset_ = reserve_workspace(cursor, parameter_bytes);
    }
    if (reorder_data_) {
      const std::size_t data_bytes =
          checked_multiply(element_count(test_case.x),
                           flagdnn_data_type_size(test_case.x.data_type),
                           "BatchNorm data bytes");
      x_offset_ = reserve_workspace(cursor, data_bytes);
      y_offset_ = reserve_workspace(cursor, data_bytes);
    }
    workspace_size_ = cursor;
  }

  [[nodiscard]] std::size_t workspace_size() const noexcept override {
    return workspace_size_;
  }

  void execute(std::span<const flagdnnBinding_t> bindings, void *workspace,
               std::size_t workspace_size, flagdnnStream_t stream) override {
    if (workspace_size < workspace_size_ ||
        (workspace_size_ != 0 && workspace == nullptr)) {
      throw std::invalid_argument(
          "CoreX cuDNN BatchNorm workspace is too small");
    }
    handle_.bind_stream(stream);
    auto *scratch = static_cast<std::byte *>(workspace);
    void *scale = binding(bindings, test_case_.scale.uid);
    void *bias = binding(bindings, test_case_.bias.uid);
    const float one = 1.0F;
    const float zero = 0.0F;
    if (convert_parameters_) {
      scale = scratch + scale_offset_;
      bias = scratch + bias_offset_;
      check_cudnn(
          cudnnTransformTensor(handle_.get(), &one, scale_descriptor_.get(),
                               binding(bindings, test_case_.scale.uid), &zero,
                               parameter_descriptor_.get(), scale),
          "cudnnTransformTensor(BatchNorm scale)");
      check_cudnn(
          cudnnTransformTensor(handle_.get(), &one, bias_descriptor_.get(),
                               binding(bindings, test_case_.bias.uid), &zero,
                               parameter_descriptor_.get(), bias),
          "cudnnTransformTensor(BatchNorm bias)");
    }

    void *next_mean = binding(bindings, test_case_.next_running_mean.uid);
    void *next_variance =
        binding(bindings, test_case_.next_running_variance.uid);
    check_cudnn(cudnnTransformTensor(
                    handle_.get(), &one, parameter_descriptor_.get(),
                    binding(bindings, test_case_.previous_running_mean.uid),
                    &zero, parameter_descriptor_.get(), next_mean),
                "cudnnTransformTensor(BatchNorm running mean)");
    check_cudnn(cudnnTransformTensor(
                    handle_.get(), &one, parameter_descriptor_.get(),
                    binding(bindings, test_case_.previous_running_variance.uid),
                    &zero, parameter_descriptor_.get(), next_variance),
                "cudnnTransformTensor(BatchNorm running variance)");

    void *x = binding(bindings, test_case_.x.uid);
    void *y = binding(bindings, test_case_.y.uid);
    cudnnTensorDescriptor_t x_descriptor = x_descriptor_.get();
    cudnnTensorDescriptor_t y_descriptor = y_descriptor_.get();
    if (reorder_data_) {
      x = scratch + x_offset_;
      y = scratch + y_offset_;
      transform_logical(handle_, scalar_x_descriptor_,
                        binding(bindings, test_case_.x.uid), test_case_.x,
                        scalar_x_descriptor_, x, contiguous_x_);
      x_descriptor = contiguous_x_descriptor_.get();
      y_descriptor = contiguous_y_descriptor_.get();
    }

    check_cudnn(cudnnBatchNormalizationForwardTraining(
                    handle_.get(), CUDNN_BATCHNORM_SPATIAL, &one, &zero,
                    x_descriptor, x, y_descriptor, y,
                    parameter_descriptor_.get(), scale, bias,
                    test_case_.momentum, next_mean, next_variance,
                    test_case_.epsilon, binding(bindings, test_case_.mean.uid),
                    binding(bindings, test_case_.inv_variance.uid)),
                "cudnnBatchNormalizationForwardTraining");

    if (reorder_data_) {
      transform_logical(handle_, scalar_y_descriptor_, y, contiguous_y_,
                        scalar_y_descriptor_,
                        binding(bindings, test_case_.y.uid), test_case_.y);
    }
  }

private:
  BatchnormTestCase test_case_;
  TestTensor contiguous_x_;
  TestTensor contiguous_y_;
  TestTensor float_parameter_;
  TestTensor scalar_x_;
  TestTensor scalar_y_;
  CorexCudnnHandle handle_;
  CorexCudnnTensorDescriptor x_descriptor_;
  CorexCudnnTensorDescriptor y_descriptor_;
  CorexCudnnTensorDescriptor contiguous_x_descriptor_;
  CorexCudnnTensorDescriptor contiguous_y_descriptor_;
  CorexCudnnTensorDescriptor scale_descriptor_;
  CorexCudnnTensorDescriptor bias_descriptor_;
  CorexCudnnTensorDescriptor parameter_descriptor_;
  CorexCudnnTensorDescriptor scalar_x_descriptor_;
  CorexCudnnTensorDescriptor scalar_y_descriptor_;
  bool reorder_data_ = false;
  bool convert_parameters_ = false;
  std::size_t scale_offset_ = 0;
  std::size_t bias_offset_ = 0;
  std::size_t x_offset_ = 0;
  std::size_t y_offset_ = 0;
  std::size_t workspace_size_ = 0;
};

class BatchnormInferenceExecutable final : public NormalizationExecutable {
public:
  explicit BatchnormInferenceExecutable(
      const BatchnormInferenceTestCase &test_case)
      : test_case_(test_case),
        contiguous_x_(contiguous_like(test_case.x, test_case.x.data_type)),
        contiguous_y_(contiguous_like(test_case.y, test_case.y.data_type)),
        float_parameter_(contiguous_like(test_case.mean, FLAGDNN_DATA_FLOAT32)),
        scalar_input_(scalar_like(test_case.x, test_case.x.data_type)),
        scalar_output_(scalar_like(test_case.y, test_case.y.data_type)),
        scalar_parameter_(scalar_like(test_case.mean, FLAGDNN_DATA_FLOAT32)),
        input_descriptor_(make_reference_tensor(test_case.x)),
        output_descriptor_(make_reference_tensor(test_case.y)),
        contiguous_x_descriptor_(make_reference_tensor(contiguous_x_)),
        contiguous_y_descriptor_(make_reference_tensor(contiguous_y_)),
        parameter_descriptor_(make_reference_tensor(float_parameter_)),
        scalar_input_descriptor_(make_reference_tensor(scalar_input_)),
        scalar_output_descriptor_(make_reference_tensor(scalar_output_)),
        scalar_parameter_descriptor_(make_reference_tensor(scalar_parameter_)),
        reorder_data_(!is_contiguous(test_case.x) ||
                      !is_contiguous(test_case.y)) {
    check_cudnn(cudnnSetOpTensorDescriptor(multiply_.get(), CUDNN_OP_TENSOR_MUL,
                                           CUDNN_DATA_FLOAT,
                                           CUDNN_PROPAGATE_NAN),
                "cudnnSetOpTensorDescriptor(BatchNorm inference MUL)");
    std::size_t cursor = 0;
    const std::size_t parameter_bytes =
        checked_multiply(element_count(float_parameter_), sizeof(float),
                         "BatchNorm inference parameter bytes");
    effective_scale_offset_ = reserve_workspace(cursor, parameter_bytes);
    variance_offset_ = reserve_workspace(cursor, parameter_bytes);
    if (reorder_data_) {
      x_offset_ = reserve_workspace(
          cursor,
          checked_multiply(element_count(contiguous_x_),
                           flagdnn_data_type_size(contiguous_x_.data_type),
                           "BatchNorm inference input bytes"));
      y_offset_ = reserve_workspace(
          cursor,
          checked_multiply(element_count(contiguous_y_),
                           flagdnn_data_type_size(contiguous_y_.data_type),
                           "BatchNorm inference output bytes"));
    }
    workspace_size_ = cursor;
  }

  [[nodiscard]] std::size_t workspace_size() const noexcept override {
    return workspace_size_;
  }

  void execute(std::span<const flagdnnBinding_t> bindings, void *workspace,
               std::size_t workspace_size, flagdnnStream_t stream) override {
    if (workspace == nullptr || workspace_size < workspace_size_) {
      throw std::invalid_argument(
          "CoreX cuDNN BatchNorm inference workspace is too small");
    }
    handle_.bind_stream(stream);
    auto *scratch = static_cast<std::byte *>(workspace);
    void *effective_scale = scratch + effective_scale_offset_;
    void *variance = scratch + variance_offset_;
    const float one = 1.0F;
    const float zero = 0.0F;
    const auto *inv_variance = static_cast<const std::byte *>(
        binding(bindings, test_case_.inv_variance.uid));
    const auto *scale =
        static_cast<const std::byte *>(binding(bindings, test_case_.scale.uid));
    auto *effective_scale_bytes = static_cast<std::byte *>(effective_scale);
    const std::size_t channels = element_count(float_parameter_);
    for (std::size_t channel = 0; channel < channels; ++channel) {
      check_cudnn(
          cudnnOpTensor(handle_.get(), multiply_.get(), &one,
                        scalar_parameter_descriptor_.get(),
                        scale + channel * sizeof(float), &one,
                        scalar_parameter_descriptor_.get(),
                        inv_variance + channel * sizeof(float), &zero,
                        scalar_parameter_descriptor_.get(),
                        effective_scale_bytes + channel * sizeof(float)),
          "cudnnOpTensor(BatchNorm inference effective scale)");
    }

    check_cudnn(cudnnTransformTensor(handle_.get(), &zero,
                                     parameter_descriptor_.get(), scale, &zero,
                                     parameter_descriptor_.get(), variance),
                "cudnnTransformTensor(BatchNorm inference neutral variance)");

    void *x = binding(bindings, test_case_.x.uid);
    void *y = binding(bindings, test_case_.y.uid);
    cudnnTensorDescriptor_t x_descriptor = input_descriptor_.get();
    cudnnTensorDescriptor_t y_descriptor = output_descriptor_.get();
    if (reorder_data_) {
      x = scratch + x_offset_;
      y = scratch + y_offset_;
      transform_logical(handle_, scalar_input_descriptor_,
                        binding(bindings, test_case_.x.uid), test_case_.x,
                        scalar_input_descriptor_, x, contiguous_x_);
      x_descriptor = contiguous_x_descriptor_.get();
      y_descriptor = contiguous_y_descriptor_.get();
    }

    constexpr double kNeutralVarianceEpsilon = 1.0;
    check_cudnn(cudnnBatchNormalizationForwardInference(
                    handle_.get(), CUDNN_BATCHNORM_SPATIAL, &one, &zero,
                    x_descriptor, x, y_descriptor, y,
                    parameter_descriptor_.get(), effective_scale,
                    binding(bindings, test_case_.bias.uid),
                    binding(bindings, test_case_.mean.uid), variance,
                    kNeutralVarianceEpsilon),
                "cudnnBatchNormalizationForwardInference");

    if (reorder_data_) {
      transform_logical(handle_, scalar_output_descriptor_, y, contiguous_y_,
                        scalar_output_descriptor_,
                        binding(bindings, test_case_.y.uid), test_case_.y);
    }
  }

private:
  BatchnormInferenceTestCase test_case_;
  TestTensor contiguous_x_;
  TestTensor contiguous_y_;
  TestTensor float_parameter_;
  TestTensor scalar_input_;
  TestTensor scalar_output_;
  TestTensor scalar_parameter_;
  CorexCudnnHandle handle_;
  CorexCudnnTensorDescriptor input_descriptor_;
  CorexCudnnTensorDescriptor output_descriptor_;
  CorexCudnnTensorDescriptor contiguous_x_descriptor_;
  CorexCudnnTensorDescriptor contiguous_y_descriptor_;
  CorexCudnnTensorDescriptor parameter_descriptor_;
  CorexCudnnTensorDescriptor scalar_input_descriptor_;
  CorexCudnnTensorDescriptor scalar_output_descriptor_;
  CorexCudnnTensorDescriptor scalar_parameter_descriptor_;
  CorexCudnnOpTensorDescriptor multiply_;
  bool reorder_data_ = false;
  std::size_t effective_scale_offset_ = 0;
  std::size_t variance_offset_ = 0;
  std::size_t x_offset_ = 0;
  std::size_t y_offset_ = 0;
  std::size_t workspace_size_ = 0;
};

class RmsnormExecutable final : public NormalizationExecutable {
  RmsnormTestCase c_;
  CorexCudnnHandle handle_;
  CorexCudnnTensorDescriptor x_, y_, gamma_, bias_, inverse_;
  CorexCudnnOpTensorDescriptor add_;
  static TestTensor flat(const TestTensor &source, std::size_t rows,
                         std::size_t width) {
    if (!is_contiguous(source))
      throw std::invalid_argument(
          "RMS reference requires contiguous shared tensors");
    auto result = source;
    result.dimensions = {1, 1, static_cast<std::int64_t>(rows),
                         static_cast<std::int64_t>(width)};
    result.strides = {static_cast<std::int64_t>(rows * width),
                      static_cast<std::int64_t>(rows * width),
                      static_cast<std::int64_t>(width), 1};
    return result;
  }

public:
  explicit RmsnormExecutable(const RmsnormTestCase &c)
      : c_(c), x_(make_reference_tensor(
                   flat(c.x, element_count(c.x) / element_count(c.scale),
                        element_count(c.scale)))),
        y_(make_reference_tensor(
            flat(c.y, element_count(c.x) / element_count(c.scale),
                 element_count(c.scale)))),
        gamma_(make_reference_tensor(flat(c.scale, 1, element_count(c.scale)))),
        bias_(make_reference_tensor(flat(c.bias, 1, element_count(c.bias)))),
        inverse_(make_reference_tensor(
            flat(c.inv_variance, element_count(c.inv_variance), 1))) {
    check_cudnn(cudnnSetOpTensorDescriptor(add_.get(), CUDNN_OP_TENSOR_ADD,
                                           CUDNN_DATA_FLOAT,
                                           CUDNN_PROPAGATE_NAN),
                "cudnnSetOpTensorDescriptor(RMS bias)");
  }
  std::size_t workspace_size() const noexcept override { return 0; }
  void execute(std::span<const flagdnnBinding_t> b, void *, std::size_t,
               flagdnnStream_t stream) override {
    handle_.bind_stream(stream);
    check_cudnn(cudnnRmsNormalizationForward(
                    handle_.get(), x_.get(), binding(b, c_.x.uid), gamma_.get(),
                    binding(b, c_.scale.uid), static_cast<float>(c_.epsilon),
                    y_.get(), binding(b, c_.y.uid), inverse_.get(),
                    binding(b, c_.inv_variance.uid)),
                "cudnnRmsNormalizationForward");
    const float one = 1, zero = 0;
    check_cudnn(cudnnOpTensor(handle_.get(), add_.get(), &one, y_.get(),
                              binding(b, c_.y.uid), &one, bias_.get(),
                              binding(b, c_.bias.uid), &zero, y_.get(),
                              binding(b, c_.y.uid)),
                "cudnnOpTensor(RMS bias)");
  }
};

} // namespace

TestTensor batchnorm_reference_data_tensor(const TestTensor &tensor) {
  return tensor;
}

std::unique_ptr<NormalizationExecutable>
build_layernorm_reference(const LayernormTestCase &) {
  throw std::logic_error(
      "LayerNorm has no exact CoreX cuDNN 7.6.5 classic primitive");
}

std::unique_ptr<NormalizationExecutable>
build_rmsnorm_reference(const RmsnormTestCase &c) {
  return std::make_unique<RmsnormExecutable>(c);
}

std::unique_ptr<NormalizationExecutable>
build_batchnorm_reference(const BatchnormTestCase &test_case) {
  return std::make_unique<BatchnormTrainingExecutable>(test_case);
}

std::unique_ptr<NormalizationExecutable> build_batchnorm_inference_reference(
    const BatchnormInferenceTestCase &test_case) {
  return std::make_unique<BatchnormInferenceExecutable>(test_case);
}

} // namespace flagdnn::testing
