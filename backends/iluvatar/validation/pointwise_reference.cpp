// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "pointwise_reference.hpp"

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

namespace flagdnn::iluvatar::validation {
namespace {

void *find_binding(std::span<const flagdnnBinding_t> bindings,
                   std::int64_t uid) {
  const auto iterator = std::find_if(
      bindings.begin(), bindings.end(),
      [uid](const flagdnnBinding_t &binding) { return binding.uid == uid; });
  if (iterator == bindings.end() || iterator->device_pointer == nullptr) {
    throw std::invalid_argument("CoreX cuDNN pointwise binding is missing");
  }
  return iterator->device_pointer;
}

std::size_t storage_elements(const flagdnn::testing::TestTensor &tensor) {
  if (tensor.dimensions.empty() ||
      tensor.dimensions.size() != tensor.strides.size()) {
    throw std::invalid_argument("pointwise reference tensor is invalid");
  }
  std::size_t maximum_offset = 0;
  for (std::size_t axis = 0; axis < tensor.dimensions.size(); ++axis) {
    if (tensor.dimensions[axis] <= 0 || tensor.strides[axis] <= 0) {
      throw std::invalid_argument("pointwise reference stride is invalid");
    }
    const std::size_t dimension =
        static_cast<std::size_t>(tensor.dimensions[axis] - 1);
    const std::size_t stride = static_cast<std::size_t>(tensor.strides[axis]);
    if (dimension != 0 &&
        stride > (std::numeric_limits<std::size_t>::max() - maximum_offset) /
                     dimension) {
      throw std::invalid_argument("pointwise reference tensor is too large");
    }
    maximum_offset += dimension * stride;
  }
  return maximum_offset + 1;
}

cudnnDataType_t compute_type(flagdnnPointwiseMode_t mode) {
  return mode == FLAGDNN_POINTWISE_LOGICAL_NOT ? CUDNN_DATA_INT8
                                               : CUDNN_DATA_FLOAT;
}

cudnnOpTensorOp_t op_tensor_mode(flagdnnPointwiseMode_t mode) {
  switch (mode) {
  case FLAGDNN_POINTWISE_ADD:
  case FLAGDNN_POINTWISE_SUB:
    return CUDNN_OP_TENSOR_ADD;
  case FLAGDNN_POINTWISE_MUL:
    return CUDNN_OP_TENSOR_MUL;
  case FLAGDNN_POINTWISE_MIN:
    return CUDNN_OP_TENSOR_MIN;
  case FLAGDNN_POINTWISE_MAX:
    return CUDNN_OP_TENSOR_MAX;
  case FLAGDNN_POINTWISE_SQRT:
    return CUDNN_OP_TENSOR_SQRT;
  case FLAGDNN_POINTWISE_LOGICAL_NOT:
    return CUDNN_OP_TENSOR_NOT;
  default:
    throw std::invalid_argument("pointwise mode has no OpTensor mapping");
  }
}

bool is_op_tensor(flagdnnPointwiseMode_t mode) {
  switch (mode) {
  case FLAGDNN_POINTWISE_ADD:
  case FLAGDNN_POINTWISE_SUB:
  case FLAGDNN_POINTWISE_MUL:
  case FLAGDNN_POINTWISE_MIN:
  case FLAGDNN_POINTWISE_MAX:
  case FLAGDNN_POINTWISE_SQRT:
  case FLAGDNN_POINTWISE_LOGICAL_NOT:
    return true;
  default:
    return false;
  }
}

cudnnActivationMode_t activation_mode(flagdnnPointwiseMode_t mode) {
  switch (mode) {
  case FLAGDNN_POINTWISE_SIGMOID_FWD:
  case FLAGDNN_POINTWISE_SIGMOID_BWD:
    return CUDNN_ACTIVATION_SIGMOID;
  case FLAGDNN_POINTWISE_RELU_FWD:
    return CUDNN_ACTIVATION_RELU;
  case FLAGDNN_POINTWISE_TANH_FWD:
    return CUDNN_ACTIVATION_TANH;
  case FLAGDNN_POINTWISE_ELU_FWD:
    return CUDNN_ACTIVATION_ELU;
  case FLAGDNN_POINTWISE_IDENTITY:
    return CUDNN_ACTIVATION_IDENTITY;
  case FLAGDNN_POINTWISE_SWISH_FWD:
    return CUDNN_ACTIVATION_SWISH;
  case FLAGDNN_POINTWISE_GELU_FWD:
    return CUDNN_ACTIVATION_GELU;
  case FLAGDNN_POINTWISE_GELU_APPROX_TANH_FWD:
    return CUDNN_ACTIVATION_GELU_TAHN;
  default:
    throw std::invalid_argument("pointwise mode has no activation mapping");
  }
}

double activation_coefficient(const ClassicPointwiseReferenceSpec &spec) {
  if (spec.mode == FLAGDNN_POINTWISE_ELU_FWD) {
    return spec.attributes.elu_alpha;
  }
  if (spec.mode == FLAGDNN_POINTWISE_SWISH_FWD) {
    return spec.attributes.swish_beta;
  }
  return 0.0;
}

class ClassicPointwiseExecutable final
    : public flagdnn::testing::TestExecutable {
public:
  explicit ClassicPointwiseExecutable(ClassicPointwiseReferenceSpec spec)
      : spec_(std::move(spec)), output_(make_reference_tensor(spec_.output)) {
    if (spec_.inputs.empty() || spec_.inputs.size() > 2) {
      throw std::invalid_argument("CoreX cuDNN pointwise arity is invalid");
    }
    inputs_.reserve(spec_.inputs.size());
    input_descriptors_.reserve(spec_.inputs.size());
    for (const auto &input : spec_.inputs) {
      inputs_.push_back(make_reference_tensor(input));
      input_descriptors_.emplace_back(inputs_.back());
    }
    output_descriptor_.set(output_);

    if (spec_.mode == FLAGDNN_POINTWISE_NEG ||
        spec_.mode == FLAGDNN_POINTWISE_IDENTITY) {
      transform_alpha_ = spec_.mode == FLAGDNN_POINTWISE_NEG ? -1.0F : 1.0F;
    } else if (is_op_tensor(spec_.mode)) {
      check_cudnn(cudnnSetOpTensorDescriptor(
                      op_.get(), op_tensor_mode(spec_.mode),
                      compute_type(spec_.mode), CUDNN_PROPAGATE_NAN),
                  "cudnnSetOpTensorDescriptor");
    } else {
      check_cudnn(cudnnSetActivationDescriptor(
                      activation_.get(), activation_mode(spec_.mode),
                      CUDNN_PROPAGATE_NAN, activation_coefficient(spec_)),
                  "cudnnSetActivationDescriptor");
      if (spec_.mode == FLAGDNN_POINTWISE_SIGMOID_BWD) {
        workspace_size_ = storage_elements(spec_.output) *
                          flagdnn_data_type_size(spec_.output.data_type);
      }
    }
  }

  [[nodiscard]] std::size_t workspace_size() const noexcept override {
    return workspace_size_;
  }

  void execute(std::span<const flagdnnBinding_t> bindings, void *workspace,
               std::size_t workspace_size, flagdnnStream_t stream) override {
    if (workspace_size < workspace_size_ ||
        (workspace_size_ != 0 && workspace == nullptr)) {
      throw std::invalid_argument("CoreX cuDNN pointwise workspace is small");
    }
    handle_.bind_stream(stream);
    void *output = find_binding(bindings, spec_.output.uid);
    void *first = find_binding(bindings, spec_.inputs.front().uid);
    const float zero = 0.0F;

    if (spec_.mode == FLAGDNN_POINTWISE_NEG ||
        spec_.mode == FLAGDNN_POINTWISE_IDENTITY) {
      check_cudnn(cudnnTransformTensor(handle_.get(), &transform_alpha_,
                                       input_descriptors_.front().get(), first,
                                       &zero, output_descriptor_.get(), output),
                  "cudnnTransformTensor(pointwise unary transform)");
      return;
    }
    if (is_op_tensor(spec_.mode)) {
      const float alpha1 = 1.0F;
      float alpha2 = static_cast<float>(spec_.alpha);
      if (spec_.mode == FLAGDNN_POINTWISE_SUB) {
        alpha2 = -alpha2;
      }
      void *second = spec_.inputs.size() == 2
                         ? find_binding(bindings, spec_.inputs[1].uid)
                         : first;
      const auto second_descriptor = spec_.inputs.size() == 2
                                         ? input_descriptors_[1].get()
                                         : input_descriptors_.front().get();
      check_cudnn(cudnnOpTensor(handle_.get(), op_.get(), &alpha1,
                                input_descriptors_.front().get(), first,
                                &alpha2, second_descriptor, second, &zero,
                                output_descriptor_.get(), output),
                  "cudnnOpTensor");
      return;
    }

    const float one = 1.0F;
    if (spec_.mode != FLAGDNN_POINTWISE_SIGMOID_BWD) {
      check_cudnn(cudnnActivationForward(handle_.get(), activation_.get(), &one,
                                         input_descriptors_.front().get(),
                                         first, &zero, output_descriptor_.get(),
                                         output),
                  "cudnnActivationForward");
      return;
    }

    void *x = find_binding(bindings, spec_.inputs[1].uid);
    check_cudnn(cudnnActivationForward(handle_.get(), activation_.get(), &one,
                                       input_descriptors_[1].get(), x, &zero,
                                       output_descriptor_.get(), workspace),
                "cudnnActivationForward(SIGMOID_BWD)");
    check_cudnn(cudnnActivationBackward(handle_.get(), activation_.get(), &one,
                                        output_descriptor_.get(), workspace,
                                        input_descriptors_.front().get(), first,
                                        input_descriptors_[1].get(), x, &zero,
                                        output_descriptor_.get(), output),
                "cudnnActivationBackward(SIGMOID_BWD)");
  }

private:
  ClassicPointwiseReferenceSpec spec_;
  std::vector<ReferenceTensor> inputs_;
  ReferenceTensor output_;
  CorexCudnnHandle handle_;
  std::vector<CorexCudnnTensorDescriptor> input_descriptors_;
  CorexCudnnTensorDescriptor output_descriptor_;
  CorexCudnnOpTensorDescriptor op_;
  CorexCudnnActivationDescriptor activation_;
  std::size_t workspace_size_ = 0;
  float transform_alpha_ = 1.0F;
};

class AddSquareExecutable final : public flagdnn::testing::TestExecutable {
public:
  AddSquareExecutable(flagdnn::testing::TestTensor left,
                      flagdnn::testing::TestTensor right,
                      flagdnn::testing::TestTensor output)
      : left_(std::move(left)), right_(std::move(right)),
        output_(std::move(output)),
        left_descriptor_(make_reference_tensor(left_)),
        right_descriptor_(make_reference_tensor(right_)),
        output_descriptor_(make_reference_tensor(output_)),
        workspace_size_(storage_elements(output_) *
                        flagdnn_data_type_size(output_.data_type)) {
    check_cudnn(cudnnSetOpTensorDescriptor(add_.get(), CUDNN_OP_TENSOR_ADD,
                                           CUDNN_DATA_FLOAT,
                                           CUDNN_PROPAGATE_NAN),
                "cudnnSetOpTensorDescriptor(ADD_SQUARE add)");
    check_cudnn(cudnnSetOpTensorDescriptor(multiply_.get(), CUDNN_OP_TENSOR_MUL,
                                           CUDNN_DATA_FLOAT,
                                           CUDNN_PROPAGATE_NAN),
                "cudnnSetOpTensorDescriptor(ADD_SQUARE mul)");
  }

  [[nodiscard]] std::size_t workspace_size() const noexcept override {
    return workspace_size_;
  }

  void execute(std::span<const flagdnnBinding_t> bindings, void *workspace,
               std::size_t workspace_size, flagdnnStream_t stream) override {
    if (workspace == nullptr || workspace_size < workspace_size_) {
      throw std::invalid_argument("CoreX cuDNN add-square workspace is small");
    }
    handle_.bind_stream(stream);
    void *left = find_binding(bindings, left_.uid);
    void *right = find_binding(bindings, right_.uid);
    void *output = find_binding(bindings, output_.uid);
    const float one = 1.0F;
    const float zero = 0.0F;
    check_cudnn(cudnnOpTensor(handle_.get(), multiply_.get(), &one,
                              right_descriptor_.get(), right, &one,
                              right_descriptor_.get(), right, &zero,
                              output_descriptor_.get(), workspace),
                "cudnnOpTensor(ADD_SQUARE mul)");
    check_cudnn(cudnnOpTensor(handle_.get(), add_.get(), &one,
                              left_descriptor_.get(), left, &one,
                              output_descriptor_.get(), workspace, &zero,
                              output_descriptor_.get(), output),
                "cudnnOpTensor(ADD_SQUARE add)");
  }

private:
  flagdnn::testing::TestTensor left_;
  flagdnn::testing::TestTensor right_;
  flagdnn::testing::TestTensor output_;
  CorexCudnnHandle handle_;
  CorexCudnnTensorDescriptor left_descriptor_;
  CorexCudnnTensorDescriptor right_descriptor_;
  CorexCudnnTensorDescriptor output_descriptor_;
  CorexCudnnOpTensorDescriptor add_;
  CorexCudnnOpTensorDescriptor multiply_;
  std::size_t workspace_size_ = 0;
};

} // namespace

std::unique_ptr<flagdnn::testing::TestExecutable>
make_classic_pointwise_reference(ClassicPointwiseReferenceSpec specification) {
  return std::make_unique<ClassicPointwiseExecutable>(std::move(specification));
}

std::unique_ptr<flagdnn::testing::TestExecutable>
make_classic_add_square_reference(const flagdnn::testing::TestTensor &left,
                                  const flagdnn::testing::TestTensor &right,
                                  const flagdnn::testing::TestTensor &output) {
  return std::make_unique<AddSquareExecutable>(left, right, output);
}

} // namespace flagdnn::iluvatar::validation
