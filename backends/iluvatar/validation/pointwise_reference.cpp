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
  (void)mode;
  return CUDNN_DATA_FLOAT;
}

cudnnOpTensorOp_t op_tensor_mode(flagdnnPointwiseMode_t mode) {
  switch (mode) {
  case FLAGDNN_POINTWISE_ADD:
  case FLAGDNN_POINTWISE_SUB:
  case FLAGDNN_POINTWISE_LOGICAL_NOT:
    return CUDNN_OP_TENSOR_ADD;
  case FLAGDNN_POINTWISE_MUL:
  case FLAGDNN_POINTWISE_LOGICAL_AND:
    return CUDNN_OP_TENSOR_MUL;
  case FLAGDNN_POINTWISE_MIN:
    return CUDNN_OP_TENSOR_MIN;
  case FLAGDNN_POINTWISE_MAX:
  case FLAGDNN_POINTWISE_ABS:
  case FLAGDNN_POINTWISE_LOGICAL_OR:
    return CUDNN_OP_TENSOR_MAX;
  case FLAGDNN_POINTWISE_SQRT:
    return CUDNN_OP_TENSOR_SQRT;
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
  case FLAGDNN_POINTWISE_LOGICAL_AND:
  case FLAGDNN_POINTWISE_LOGICAL_OR:
  case FLAGDNN_POINTWISE_ABS:
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
  case FLAGDNN_POINTWISE_RELU_BWD:
    return CUDNN_ACTIVATION_RELU;
  case FLAGDNN_POINTWISE_TANH_FWD:
  case FLAGDNN_POINTWISE_TANH_BWD:
    return CUDNN_ACTIVATION_TANH;
  case FLAGDNN_POINTWISE_ELU_FWD:
  case FLAGDNN_POINTWISE_ELU_BWD:
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

bool is_activation_backward(flagdnnPointwiseMode_t mode) {
  return mode == FLAGDNN_POINTWISE_SIGMOID_BWD ||
         mode == FLAGDNN_POINTWISE_RELU_BWD ||
         mode == FLAGDNN_POINTWISE_TANH_BWD ||
         mode == FLAGDNN_POINTWISE_ELU_BWD;
}

double activation_coefficient(const ClassicPointwiseReferenceSpec &spec) {
  if (spec.mode == FLAGDNN_POINTWISE_ELU_FWD ||
      spec.mode == FLAGDNN_POINTWISE_ELU_BWD) {
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
    if (spec_.mode == FLAGDNN_POINTWISE_LOGICAL_NOT) {
      auto scalar = spec_.output;
      scalar.dimensions = {1};
      scalar.strides = {1};
      scalar_descriptor_.set(make_reference_tensor(scalar));
      scalar_one_ = std::make_unique<CorexDeviceWorkspace>(1);
      const unsigned char one = 1;
      const auto status =
          cudaMemcpy(scalar_one_->data(), &one, 1, cudaMemcpyHostToDevice);
      if (status != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(status));
    }

    leaky_forward_ = spec_.mode == FLAGDNN_POINTWISE_RELU_FWD &&
                     spec_.attributes.flags ==
                         FLAGDNN_POINTWISE_ATTRIBUTE_RELU_LOWER_CLIP_SLOPE;
    if (leaky_forward_) {
      check_cudnn(
          cudnnSetOpTensorDescriptor(op_.get(),
                                     spec_.attributes.relu_lower_clip_slope <= 1
                                         ? CUDNN_OP_TENSOR_MAX
                                         : CUDNN_OP_TENSOR_MIN,
                                     CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN),
          "cudnnSetOpTensorDescriptor(leaky ReLU)");
    } else if (spec_.mode == FLAGDNN_POINTWISE_NEG ||
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
      if (spec_.mode == FLAGDNN_POINTWISE_RELU_BWD &&
          spec_.attributes.flags ==
              FLAGDNN_POINTWISE_ATTRIBUTE_RELU_LOWER_CLIP_SLOPE) {
        leaky_backward_ = true;
        check_cudnn(cudnnSetOpTensorDescriptor(op_.get(), CUDNN_OP_TENSOR_ADD,
                                               CUDNN_DATA_FLOAT,
                                               CUDNN_PROPAGATE_NAN),
                    "cudnnSetOpTensorDescriptor(leaky gradient)");
      }
      if (is_activation_backward(spec_.mode)) {
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
    if (leaky_forward_) {
      const float one = 1, slope = static_cast<float>(
                               spec_.attributes.relu_lower_clip_slope);
      check_cudnn(cudnnOpTensor(handle_.get(), op_.get(), &one,
                                input_descriptors_.front().get(), first, &slope,
                                input_descriptors_.front().get(), first, &zero,
                                output_descriptor_.get(), output),
                  "cudnnOpTensor(leaky ReLU)");
      return;
    }
    if (spec_.mode == FLAGDNN_POINTWISE_NEG ||
        spec_.mode == FLAGDNN_POINTWISE_IDENTITY) {
      check_cudnn(cudnnTransformTensor(handle_.get(), &transform_alpha_,
                                       input_descriptors_.front().get(), first,
                                       &zero, output_descriptor_.get(), output),
                  "cudnnTransformTensor(pointwise unary transform)");
      return;
    }
    if (is_op_tensor(spec_.mode)) {
      const float alpha1 =
          spec_.mode == FLAGDNN_POINTWISE_LOGICAL_NOT ? -1.0F : 1.0F;
      float alpha2 = static_cast<float>(spec_.alpha);
      if (spec_.mode == FLAGDNN_POINTWISE_SUB ||
          spec_.mode == FLAGDNN_POINTWISE_ABS) {
        alpha2 = -alpha2;
      }
      void *second = spec_.inputs.size() == 2
                         ? find_binding(bindings, spec_.inputs[1].uid)
                         : first;
      auto second_descriptor = spec_.inputs.size() == 2
                                   ? input_descriptors_[1].get()
                                   : input_descriptors_.front().get();
      if (spec_.mode == FLAGDNN_POINTWISE_LOGICAL_NOT) {
        second = scalar_one_->data();
        second_descriptor = scalar_descriptor_.get();
      }
      check_cudnn(cudnnOpTensor(handle_.get(), op_.get(), &alpha1,
                                input_descriptors_.front().get(), first,
                                &alpha2, second_descriptor, second, &zero,
                                output_descriptor_.get(), output),
                  "cudnnOpTensor");
      return;
    }

    const float one = 1.0F;
    if (!is_activation_backward(spec_.mode)) {
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
                "cudnnActivationForward(backward saved output)");
    check_cudnn(cudnnActivationBackward(handle_.get(), activation_.get(), &one,
                                        output_descriptor_.get(), workspace,
                                        input_descriptors_.front().get(), first,
                                        input_descriptors_[1].get(), x, &zero,
                                        output_descriptor_.get(), output),
                "cudnnActivationBackward");
    if (leaky_backward_) {
      const float slope = static_cast<float>(
                      spec_.attributes.relu_lower_clip_slope),
                  positive = 1 - slope;
      check_cudnn(cudnnOpTensor(handle_.get(), op_.get(), &positive,
                                output_descriptor_.get(), output, &slope,
                                input_descriptors_.front().get(), first, &zero,
                                output_descriptor_.get(), output),
                  "cudnnOpTensor(leaky gradient)");
    }
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
  bool leaky_forward_ = false, leaky_backward_ = false;
  CorexCudnnTensorDescriptor scalar_descriptor_;
  std::unique_ptr<CorexDeviceWorkspace> scalar_one_;
};

// Use available classic primitives when the SDK's SWISH enum is a stub.
class SigmoidProductExecutable final : public flagdnn::testing::TestExecutable {
  ClassicPointwiseReferenceSpec spec_;
  CorexCudnnHandle handle_;
  CorexCudnnTensorDescriptor x_, dy_, output_;
  CorexCudnnOpTensorDescriptor add_, mul_;
  CorexCudnnActivationDescriptor sigmoid_;
  std::size_t span_;
  float beta_;

public:
  explicit SigmoidProductExecutable(ClassicPointwiseReferenceSpec spec)
      : spec_(std::move(spec)), x_(make_reference_tensor(spec_.inputs.back())),
        dy_(make_reference_tensor(spec_.inputs.front())),
        output_(make_reference_tensor(spec_.output)),
        span_((storage_elements(spec_.output) *
                   flagdnn_data_type_size(spec_.output.data_type) +
               255) /
              256 * 256),
        beta_(static_cast<float>(spec_.mode == FLAGDNN_POINTWISE_SOFTPLUS_BWD
                                     ? spec_.attributes.softplus_beta
                                     : spec_.attributes.swish_beta)) {
    check_cudnn(cudnnSetOpTensorDescriptor(add_.get(), CUDNN_OP_TENSOR_ADD,
                                           CUDNN_DATA_FLOAT,
                                           CUDNN_PROPAGATE_NAN),
                "cudnnSetOpTensorDescriptor(sigmoid composition add)");
    check_cudnn(cudnnSetOpTensorDescriptor(mul_.get(), CUDNN_OP_TENSOR_MUL,
                                           CUDNN_DATA_FLOAT,
                                           CUDNN_PROPAGATE_NAN),
                "cudnnSetOpTensorDescriptor(sigmoid composition mul)");
    check_cudnn(cudnnSetActivationDescriptor(sigmoid_.get(),
                                             CUDNN_ACTIVATION_SIGMOID,
                                             CUDNN_PROPAGATE_NAN, 0),
                "cudnnSetActivationDescriptor(sigmoid composition)");
  }
  std::size_t workspace_size() const noexcept override { return 4 * span_; }
  void execute(std::span<const flagdnnBinding_t> b, void *workspace,
               std::size_t bytes, flagdnnStream_t stream) override {
    if (!workspace || bytes < workspace_size())
      throw std::invalid_argument("sigmoid composition workspace is small");
    handle_.bind_stream(stream);
    const float one = 1, zero = 0;
    auto *z = static_cast<std::byte *>(workspace);
    void *s = z + span_, *product = z + 2 * span_, *term = z + 3 * span_;
    void *x = find_binding(b, spec_.inputs.back().uid),
         *dy = find_binding(b, spec_.inputs.front().uid),
         *out = find_binding(b, spec_.output.uid);
    check_cudnn(cudnnOpTensor(handle_.get(), add_.get(), &beta_, x_.get(), x,
                              &zero, x_.get(), x, &zero, output_.get(), z),
                "cudnnOpTensor(sigmoid input scaling)");
    check_cudnn(cudnnActivationForward(handle_.get(), sigmoid_.get(), &one,
                                       output_.get(), z, &zero, output_.get(),
                                       s),
                "cudnnActivationForward(sigmoid composition)");
    if (spec_.mode == FLAGDNN_POINTWISE_SWISH_BWD) {
      check_cudnn(cudnnOpTensor(handle_.get(), mul_.get(), &beta_, x_.get(), x,
                                &one, dy_.get(), dy, &zero, output_.get(),
                                product),
                  "cudnnOpTensor(swish upstream product)");
      check_cudnn(cudnnActivationBackward(handle_.get(), sigmoid_.get(), &one,
                                          output_.get(), s, output_.get(),
                                          product, output_.get(), z, &zero,
                                          output_.get(), term),
                  "cudnnActivationBackward(swish chain rule)");
    }
    // Forward: x * sigmoid(beta*x). Backward softplus: dy * sigmoid(beta*x).
    check_cudnn(cudnnOpTensor(handle_.get(), mul_.get(), &one, output_.get(), s,
                              &one, dy_.get(), dy, &zero, output_.get(), out),
                "cudnnOpTensor(sigmoid product)");
    if (spec_.mode == FLAGDNN_POINTWISE_SWISH_BWD)
      check_cudnn(cudnnOpTensor(handle_.get(), add_.get(), &one, output_.get(),
                                out, &one, output_.get(), term, &zero,
                                output_.get(), out),
                  "cudnnOpTensor(swish derivative sum)");
  }
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
  if (specification.mode == FLAGDNN_POINTWISE_SWISH_FWD ||
      specification.mode == FLAGDNN_POINTWISE_SWISH_BWD ||
      specification.mode == FLAGDNN_POINTWISE_SOFTPLUS_BWD)
    return std::make_unique<SigmoidProductExecutable>(std::move(specification));
  return std::make_unique<ClassicPointwiseExecutable>(std::move(specification));
}

std::unique_ptr<flagdnn::testing::TestExecutable>
make_classic_add_square_reference(const flagdnn::testing::TestTensor &left,
                                  const flagdnn::testing::TestTensor &right,
                                  const flagdnn::testing::TestTensor &output) {
  return std::make_unique<AddSquareExecutable>(left, right, output);
}

} // namespace flagdnn::iluvatar::validation
