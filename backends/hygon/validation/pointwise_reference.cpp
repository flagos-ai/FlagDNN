/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "pointwise_reference.hpp"

#include "activation_layout.hpp"
#include "tensor_io.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace flagdnn::validation::hygon {
namespace {

hipdnnDataType_t hipdnn_data_type(flagdnnDataType_t data_type) {
  switch (data_type) {
    case FLAGDNN_DATA_INT32:
      throw std::invalid_argument(
          "INT32 is not supported by this validation adapter");

    case FLAGDNN_DATA_FP8_E8M0:
      break;
    case FLAGDNN_DATA_FLOAT32:
      return HIPDNN_DATA_FLOAT;
    case FLAGDNN_DATA_FLOAT16:
      return HIPDNN_DATA_HALF;
    case FLAGDNN_DATA_BFLOAT16:
      return HIPDNN_DATA_BFLOAT16;
    case FLAGDNN_DATA_BOOLEAN:
      return HIPDNN_DATA_BOOLEAN;
    case FLAGDNN_DATA_FP8_E4M3:
      return HIPDNN_DATA_FP8_E4M3;
    case FLAGDNN_DATA_FP8_E5M2:
      return HIPDNN_DATA_FP8_E5M2;
  }
  throw std::invalid_argument("unsupported hipDNN tensor data type");
}

bool is_binary(HipdnnPointwiseKind kind) noexcept {
  switch (kind) {
  case HipdnnPointwiseKind::kAdd:
  case HipdnnPointwiseKind::kSub:
  case HipdnnPointwiseKind::kMul:
  case HipdnnPointwiseKind::kMin:
  case HipdnnPointwiseKind::kMax:
  case HipdnnPointwiseKind::kAddSquare:
    return true;
  default:
    return false;
  }
}

bool is_activation_backward(HipdnnPointwiseKind kind) noexcept {
  switch (kind) {
  case HipdnnPointwiseKind::kSigmoidBackward:
  case HipdnnPointwiseKind::kReluBackward:
  case HipdnnPointwiseKind::kTanhBackward:
  case HipdnnPointwiseKind::kEluBackward:
  case HipdnnPointwiseKind::kSoftplusBackward:
  case HipdnnPointwiseKind::kSwishBackward:
    return true;
  default:
    return false;
  }
}

bool is_activation_forward(HipdnnPointwiseKind kind) noexcept {
  switch (kind) {
  case HipdnnPointwiseKind::kSigmoid:
  case HipdnnPointwiseKind::kRelu:
  case HipdnnPointwiseKind::kTanh:
  case HipdnnPointwiseKind::kElu:
  case HipdnnPointwiseKind::kSoftplus:
  case HipdnnPointwiseKind::kAbs:
  case HipdnnPointwiseKind::kIdentity:
  case HipdnnPointwiseKind::kSwish:
    return true;
  default:
    return false;
  }
}

hipdnnOpTensorOp_t op_tensor_mode(HipdnnPointwiseKind kind) {
  switch (kind) {
  case HipdnnPointwiseKind::kAdd:
  case HipdnnPointwiseKind::kSub:
    return HIPDNN_OP_TENSOR_ADD;
  case HipdnnPointwiseKind::kMul:
  case HipdnnPointwiseKind::kAddSquare:
    return HIPDNN_OP_TENSOR_MUL;
  case HipdnnPointwiseKind::kMin:
    return HIPDNN_OP_TENSOR_MIN;
  case HipdnnPointwiseKind::kMax:
    return HIPDNN_OP_TENSOR_MAX;
  default:
    break;
  }
  throw std::invalid_argument("pointwise kind is not a hipDNN OpTensor mode");
}

hipdnnActivationMode_t activation_mode(HipdnnPointwiseKind kind) {
  switch (kind) {
  case HipdnnPointwiseKind::kSigmoid:
  case HipdnnPointwiseKind::kSigmoidBackward:
    return HIPDNN_ACTIVATION_SIGMOID;
  case HipdnnPointwiseKind::kRelu:
  case HipdnnPointwiseKind::kReluBackward:
    return HIPDNN_ACTIVATION_RELU;
  case HipdnnPointwiseKind::kTanh:
  case HipdnnPointwiseKind::kTanhBackward:
    return HIPDNN_ACTIVATION_TANH;
  case HipdnnPointwiseKind::kElu:
  case HipdnnPointwiseKind::kEluBackward:
    return HIPDNN_ACTIVATION_ELU;
  case HipdnnPointwiseKind::kSoftplus:
  case HipdnnPointwiseKind::kSoftplusBackward:
    return HIPDNN_ACTIVATION_SOFTRELU;
  case HipdnnPointwiseKind::kAbs:
    return HIPDNN_ACTIVATION_ABS;
  case HipdnnPointwiseKind::kIdentity:
    return HIPDNN_ACTIVATION_IDENTITY;
  case HipdnnPointwiseKind::kSwish:
  case HipdnnPointwiseKind::kSwishBackward:
    return HIPDNN_ACTIVATION_SWISH;
  default:
    break;
  }
  throw std::invalid_argument("pointwise kind is not a hipDNN activation");
}

std::vector<int> checked_ints(std::span<const std::int64_t> values,
                              std::string_view role) {
  std::vector<int> result;
  result.reserve(values.size());
  for (std::int64_t value : values) {
    if (value <= 0 || value > std::numeric_limits<int>::max()) {
      throw std::invalid_argument(std::string(role) +
                                  " does not fit hipDNN int metadata");
    }
    result.push_back(static_cast<int>(value));
  }
  return result;
}

ReferenceTensor normalize_rank(const ReferenceTensor &tensor,
                               std::size_t rank) {
  if (tensor.dimensions.size() > rank) {
    throw std::invalid_argument("hipDNN input rank exceeds output rank");
  }
  ReferenceTensor result = tensor;
  const std::size_t padding = rank - result.dimensions.size();
  if (padding != 0) {
    const std::int64_t leading_stride =
        static_cast<std::int64_t>(tensor_io::storage_element_count(tensor));
    result.dimensions.insert(result.dimensions.begin(), padding, 1);
    result.strides.insert(result.strides.begin(), padding, leading_stride);
  }
  return result;
}

void set_tensor_descriptor(hipdnnTensorDescriptor_t descriptor,
                           const ReferenceTensor &tensor) {
  const std::vector<int> dimensions =
      checked_ints(tensor.dimensions, "tensor dimensions");
  const std::vector<int> strides =
      checked_ints(tensor.strides, "tensor strides");
  check_hipdnn_status(hipdnnSetTensorNdDescriptor(
                          descriptor, hipdnn_data_type(tensor.data_type),
                          static_cast<int>(dimensions.size()),
                          dimensions.data(), strides.data()),
                      "hipdnnSetTensorNdDescriptor");
}

void *binding_pointer(std::span<const flagdnnBinding_t> bindings,
                      const ReferenceTensor &tensor) {
  const auto found = std::find_if(bindings.begin(), bindings.end(),
                                  [&](const flagdnnBinding_t &binding) {
                                    return binding.uid == tensor.uid;
                                  });
  if (found == bindings.end() || found->device_pointer == nullptr) {
    throw std::invalid_argument("hipDNN pointwise binding is missing");
  }
  return static_cast<void *>(
      static_cast<std::uint8_t *>(found->device_pointer) +
      tensor.binding_byte_offset);
}

HipdnnPointwiseOperation unavailable(std::string reason) {
  HipdnnPointwiseOperation result;
  result.unavailable_reason = std::move(reason);
  return result;
}

bool has_only_flags(const flagdnnPointwiseAttributes_t &attributes,
                    std::uint64_t allowed) noexcept {
  return (attributes.flags & ~allowed) == 0U;
}

} // namespace

HipdnnPointwiseOperation
make_hipdnn_pointwise_operation(flagdnnPointwiseMode_t mode,
                                const flagdnnPointwiseAttributes_t &attributes,
                                double alpha) {
  HipdnnPointwiseOperation result;
  result.alpha = alpha;
  result.unavailable_reason.clear();
  switch (mode) {
  case FLAGDNN_POINTWISE_ADD:
    result.kind = HipdnnPointwiseKind::kAdd;
    return result;
  case FLAGDNN_POINTWISE_SUB:
    result.kind = HipdnnPointwiseKind::kSub;
    return result;
  case FLAGDNN_POINTWISE_MUL:
    result.kind = HipdnnPointwiseKind::kMul;
    return result;
  case FLAGDNN_POINTWISE_MIN:
    result.kind = HipdnnPointwiseKind::kMin;
    return result;
  case FLAGDNN_POINTWISE_MAX:
    result.kind = HipdnnPointwiseKind::kMax;
    return result;
  case FLAGDNN_POINTWISE_SIGMOID_FWD:
    if (attributes.flags == 0U) {
      result.kind = HipdnnPointwiseKind::kSigmoid;
      return result;
    }
    break;
  case FLAGDNN_POINTWISE_TANH_FWD:
    if (attributes.flags == 0U) {
      result.kind = HipdnnPointwiseKind::kTanh;
      return result;
    }
    break;
  case FLAGDNN_POINTWISE_ABS:
    if (attributes.flags == 0U) {
      result.kind = HipdnnPointwiseKind::kAbs;
      return result;
    }
    break;
  case FLAGDNN_POINTWISE_IDENTITY:
    if (attributes.flags == 0U) {
      result.kind = HipdnnPointwiseKind::kIdentity;
      return result;
    }
    break;
  case FLAGDNN_POINTWISE_NEG:
    if (attributes.flags == 0U) {
      result.kind = HipdnnPointwiseKind::kIdentity;
      result.alpha = -1.0;
      return result;
    }
    break;
  case FLAGDNN_POINTWISE_RELU_FWD:
    if (attributes.flags == 0U) {
      result.kind = HipdnnPointwiseKind::kRelu;
      return result;
    }
    if (attributes.flags == FLAGDNN_POINTWISE_ATTRIBUTE_RELU_LOWER_CLIP_SLOPE) {
      return unavailable("hipDNN LEAKYRELU has no working exact slope setter "
                         "on the validated DTK stack");
    }
    return unavailable(
        "ReLU attributes have no exact validated hipDNN activation mapping");
  case FLAGDNN_POINTWISE_ELU_FWD:
    if (has_only_flags(attributes, FLAGDNN_POINTWISE_ATTRIBUTE_ELU_ALPHA)) {
      result.kind = HipdnnPointwiseKind::kElu;
      result.activation_coefficient = attributes.elu_alpha;
      return result;
    }
    return unavailable(
        "ELU attributes have no exact validated hipDNN activation mapping");
  case FLAGDNN_POINTWISE_SOFTPLUS_FWD:
    if (has_only_flags(attributes, FLAGDNN_POINTWISE_ATTRIBUTE_SOFTPLUS_BETA) &&
        attributes.softplus_beta == 1.0) {
      result.kind = HipdnnPointwiseKind::kSoftplus;
      result.activation_coefficient = 1.0;
      return result;
    }
    return unavailable(
        "hipDNN SOFTRELU exactly matches FlagDNN softplus only for beta=1");
  case FLAGDNN_POINTWISE_SWISH_FWD:
    if (has_only_flags(attributes, FLAGDNN_POINTWISE_ATTRIBUTE_SWISH_BETA)) {
      result.kind = HipdnnPointwiseKind::kSwish;
      result.swish_beta = attributes.swish_beta;
      return result;
    }
    return unavailable(
        "SWISH attributes have no exact validated hipDNN activation mapping");
  case FLAGDNN_POINTWISE_RELU_BWD:
    result = make_hipdnn_pointwise_operation(
        FLAGDNN_POINTWISE_RELU_FWD, attributes, alpha);
    if (result.kind == HipdnnPointwiseKind::kRelu) {
      result.kind = HipdnnPointwiseKind::kReluBackward;
    }
    return result;
  case FLAGDNN_POINTWISE_TANH_BWD:
    result = make_hipdnn_pointwise_operation(
        FLAGDNN_POINTWISE_TANH_FWD, attributes, alpha);
    if (result.kind == HipdnnPointwiseKind::kTanh) {
      result.kind = HipdnnPointwiseKind::kTanhBackward;
    }
    return result;
  case FLAGDNN_POINTWISE_ELU_BWD:
    result = make_hipdnn_pointwise_operation(
        FLAGDNN_POINTWISE_ELU_FWD, attributes, alpha);
    if (result.kind == HipdnnPointwiseKind::kElu) {
      result.kind = HipdnnPointwiseKind::kEluBackward;
    }
    return result;
  case FLAGDNN_POINTWISE_SOFTPLUS_BWD:
    result = make_hipdnn_pointwise_operation(
        FLAGDNN_POINTWISE_SOFTPLUS_FWD, attributes, alpha);
    if (result.kind == HipdnnPointwiseKind::kSoftplus) {
      result.kind = HipdnnPointwiseKind::kSoftplusBackward;
    }
    return result;
  case FLAGDNN_POINTWISE_SWISH_BWD:
    result = make_hipdnn_pointwise_operation(
        FLAGDNN_POINTWISE_SWISH_FWD, attributes, alpha);
    if (result.kind == HipdnnPointwiseKind::kSwish) {
      result.kind = HipdnnPointwiseKind::kSwishBackward;
    }
    return result;
  case FLAGDNN_POINTWISE_SIGMOID_BWD:
    if (attributes.flags == 0U) {
      result.kind = HipdnnPointwiseKind::kSigmoidBackward;
      return result;
    }
    break;
  case FLAGDNN_POINTWISE_SQRT:
    return unavailable(
        "hipDNN OpTensor SQRT is unavailable on the validated DTK stack");
  case FLAGDNN_POINTWISE_LOGICAL_NOT:
    return unavailable("hipDNN BOOLEAN tensor/OpTensor NOT is unavailable on "
                       "the validated DTK stack");
  case FLAGDNN_POINTWISE_GELU_BWD:
  case FLAGDNN_POINTWISE_GELU_APPROX_TANH_BWD:
  case FLAGDNN_POINTWISE_NOT_SET:
  case FLAGDNN_POINTWISE_ERF:
  case FLAGDNN_POINTWISE_EXP:
  case FLAGDNN_POINTWISE_LOG:
  case FLAGDNN_POINTWISE_CEIL:
  case FLAGDNN_POINTWISE_COS:
  case FLAGDNN_POINTWISE_FLOOR:
  case FLAGDNN_POINTWISE_RSQRT:
  case FLAGDNN_POINTWISE_SIN:
  case FLAGDNN_POINTWISE_TAN:
  case FLAGDNN_POINTWISE_RECIPROCAL:
  case FLAGDNN_POINTWISE_DIV:
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
  case FLAGDNN_POINTWISE_GELU_FWD:
  case FLAGDNN_POINTWISE_GELU_APPROX_TANH_FWD:
  case FLAGDNN_POINTWISE_BINARY_SELECT:
    return unavailable("no exact hipDNN primitive for this pointwise mode");
  }
  return unavailable(
      "pointwise attributes have no exact validated hipDNN mapping");
}

HipdnnCapability
hipdnn_pointwise_capability(const HipdnnPointwiseOperation &operation,
                            std::span<const ReferenceTensor> tensors,
                            bool inputs_are_finite) {
  if (operation.kind == HipdnnPointwiseKind::kUnavailable) {
    return HipdnnCapability::vendor_unsupported(
        operation.unavailable_reason.empty()
            ? "no exact hipDNN pointwise primitive"
            : operation.unavailable_reason);
  }
  const std::size_t expected_tensors =
      is_activation_forward(operation.kind)
          ? 2
          : (is_binary(operation.kind) ||
                     is_activation_backward(operation.kind)
                 ? 3
                 : 0);
  if (tensors.size() != expected_tensors) {
    return HipdnnCapability::invalid_adapter_contract(
        "hipDNN pointwise primitive arity does not match tensors");
  }
  const ReferenceTensor &output = tensors.back();
  if (output.data_type == FLAGDNN_DATA_BFLOAT16) {
    return HipdnnCapability::vendor_unsupported(
        "BF16 is outside the validated hipDNN pointwise capability");
  }
  if (output.data_type != FLAGDNN_DATA_FLOAT32 &&
      output.data_type != FLAGDNN_DATA_FLOAT16) {
    return HipdnnCapability::vendor_unsupported(
        "dtype has no validated hipDNN pointwise mapping");
  }
  (void)inputs_are_finite;
  if (output.dimensions.empty() ||
      output.dimensions.size() != output.strides.size()) {
    return HipdnnCapability::invalid_adapter_contract(
        "pointwise output dimensions/strides metadata is invalid");
  }
  if (output.dimensions.size() != 3 && output.dimensions.size() != 4) {
    return HipdnnCapability::vendor_unsupported(
        "only rank-3/rank-4 hipDNN pointwise is validated");
  }
  const std::size_t element_size = tensor_io::data_type_size(output.data_type);
  for (std::size_t index = 0; index < tensors.size(); ++index) {
    const ReferenceTensor &tensor = tensors[index];
    if (tensor.data_type != output.data_type) {
      return HipdnnCapability::invalid_adapter_contract(
          "pointwise tensor data types must match");
    }
    if (tensor.dimensions.empty() ||
        tensor.dimensions.size() != tensor.strides.size()) {
      return HipdnnCapability::invalid_adapter_contract(
          "pointwise tensor dimensions/strides metadata is invalid");
    }
    if (tensor.dimensions.size() > output.dimensions.size()) {
      return HipdnnCapability::vendor_unsupported(
          "tensor rank cannot be represented by the hipDNN pointwise plan");
    }
    if (tensor.binding_byte_offset % element_size != 0) {
      return HipdnnCapability::invalid_adapter_contract(
          "binding byte offset is not element aligned");
    }
    for (std::size_t axis = 0; axis < tensor.dimensions.size(); ++axis) {
      if (tensor.dimensions[axis] <= 0 || tensor.strides[axis] <= 0) {
        return HipdnnCapability::invalid_adapter_contract(
            "pointwise tensor dimensions/strides must be positive");
      }
      if (tensor.dimensions[axis] > std::numeric_limits<int>::max() ||
          tensor.strides[axis] > std::numeric_limits<int>::max()) {
        return HipdnnCapability::vendor_unsupported(
            "tensor dimensions/strides do not fit hipDNN metadata");
      }
    }
  }

  if (is_activation_forward(operation.kind)) {
    if (tensors.front().dimensions != output.dimensions) {
      return HipdnnCapability::invalid_adapter_contract(
          "activation input/output dimensions must be equal");
    }
    return {};
  }
  if (is_activation_backward(operation.kind)) {
    if (tensors[0].dimensions != output.dimensions ||
        tensors[1].dimensions != output.dimensions) {
      return HipdnnCapability::invalid_adapter_contract(
          "activation backward dy/x/dx dimensions must be equal");
    }
    return {};
  }

  if (tensors.front().dimensions != output.dimensions) {
    return HipdnnCapability::vendor_unsupported(
        "hipDNN OpTensor requires left/A dimensions equal output/C");
  }
  const ReferenceTensor &right = tensors[1];
  if (right.dimensions != output.dimensions) {
    return HipdnnCapability::vendor_unsupported(
        "hipDNN OpTensor broadcast B is outside the validated capability");
  }
  if (tensors.front().strides != output.strides ||
      right.strides != output.strides) {
    return HipdnnCapability::vendor_unsupported(
        "validated hipDNN OpTensor requires identical A/B/C strides");
  }
  return {};
}

HipdnnCapability
hipdnn_pointwise_capture_capability(const HipdnnPointwiseOperation &operation) {
  if (operation.kind == HipdnnPointwiseKind::kIdentity &&
      operation.alpha == -1.0) {
    return HipdnnCapability::vendor_unsupported(
        "provider=hipdnn phase=capture-replay "
        "status=HIPDNN_CAPTURE_REPLAY_UNSAFE: IDENTITY alpha=-1 (NEG) is not "
        "safe for repeated HIP Graph timing on the validated DTK stack");
  }
  return {};
}

std::vector<ReferenceTensor>
hipdnn_pointwise_descriptor_tensors(const HipdnnPointwiseOperation &operation,
                                    std::span<const ReferenceTensor> tensors) {
  const bool activation_forward =
      is_activation_forward(operation.kind) && tensors.size() == 2;
  const bool activation_backward_sequence =
      is_activation_backward(operation.kind) &&
      tensors.size() == 3;
  if (!activation_forward && !activation_backward_sequence) {
    return std::vector<ReferenceTensor>(tensors.begin(), tensors.end());
  }
  return hipdnn_activation_descriptor_tensors(tensors);
}

bool hipdnn_pointwise_uses_sequence(HipdnnPointwiseKind kind) noexcept {
  return kind == HipdnnPointwiseKind::kAddSquare ||
         is_activation_backward(kind);
}

class HipdnnPointwisePlan::Impl final {
public:
  Impl(HipdnnPointwiseOperation operation, std::vector<ReferenceTensor> tensors)
      : operation_(std::move(operation)), tensors_(std::move(tensors)) {
    const HipdnnCapability capability =
        hipdnn_pointwise_capability(operation_, tensors_, true);
    if (!capability.supported) {
      throw std::invalid_argument("invalid hipDNN pointwise plan: " +
                                  capability.reason);
    }
    tensors_ = hipdnn_pointwise_descriptor_tensors(operation_, tensors_);
    const std::size_t rank = tensors_.back().dimensions.size();
    for (ReferenceTensor &tensor : tensors_) {
      tensor = normalize_rank(tensor, rank);
    }

    check_hipdnn_status(hipdnnCreate(&handle_), "hipdnnCreate");
    try {
      descriptors_.resize(tensors_.size(), nullptr);
      for (std::size_t index = 0; index < tensors_.size(); ++index) {
        check_hipdnn_status(hipdnnCreateTensorDescriptor(&descriptors_[index]),
                            "hipdnnCreateTensorDescriptor");
        set_tensor_descriptor(descriptors_[index], tensors_[index]);
      }
      if (is_binary(operation_.kind)) {
        check_hipdnn_status(hipdnnCreateOpTensorDescriptor(&first_operation_),
                            "hipdnnCreateOpTensorDescriptor");
        check_hipdnn_status(hipdnnSetOpTensorDescriptor(
                                first_operation_,
                                op_tensor_mode(operation_.kind),
                                HIPDNN_DATA_FLOAT, HIPDNN_PROPAGATE_NAN),
                            "hipdnnSetOpTensorDescriptor");
        if (operation_.kind == HipdnnPointwiseKind::kAddSquare) {
          check_hipdnn_status(
              hipdnnCreateOpTensorDescriptor(&second_operation_),
              "hipdnnCreateOpTensorDescriptor(add_square add)");
          check_hipdnn_status(hipdnnSetOpTensorDescriptor(
                                  second_operation_, HIPDNN_OP_TENSOR_ADD,
                                  HIPDNN_DATA_FLOAT, HIPDNN_PROPAGATE_NAN),
                              "hipdnnSetOpTensorDescriptor(add_square add)");
        }
      } else {
        check_hipdnn_status(hipdnnCreateActivationDescriptor(&activation_),
                            "hipdnnCreateActivationDescriptor");
        check_hipdnn_status(hipdnnSetActivationDescriptor(
                                activation_, activation_mode(operation_.kind),
                                HIPDNN_PROPAGATE_NAN,
                                operation_.activation_coefficient),
                            "hipdnnSetActivationDescriptor");
        if (operation_.kind == HipdnnPointwiseKind::kSwish ||
            operation_.kind == HipdnnPointwiseKind::kSwishBackward) {
          check_hipdnn_status(hipdnnSetActivationDescriptorSwishBeta(
                                  activation_, operation_.swish_beta),
                              "hipdnnSetActivationDescriptorSwishBeta");
        }
      }
      if (hipdnn_pointwise_uses_sequence(operation_.kind)) {
        workspace_size_ = tensor_io::storage_element_count(tensors_.back()) *
                          tensor_io::data_type_size(tensors_.back().data_type);
      }
    } catch (...) {
      cleanup();
      throw;
    }
  }

  ~Impl() { cleanup(); }

  [[nodiscard]] std::size_t workspace_size() const noexcept {
    return workspace_size_;
  }

  void execute(std::span<const flagdnnBinding_t> bindings, void *workspace,
               std::size_t workspace_size, flagdnnStream_t stream) {
    check_hipdnn_status(
        hipdnnSetStream(handle_, reinterpret_cast<hipdnnStream_t>(stream)),
        "hipdnnSetStream");
    constexpr float one = 1.0F;
    constexpr float zero = 0.0F;

    if (is_activation_forward(operation_.kind)) {
      void *input = binding_pointer(bindings, tensors_[0]);
      void *output = binding_pointer(bindings, tensors_[1]);
      const float alpha = static_cast<float>(operation_.alpha);
      check_hipdnn_status(hipdnnActivationForward(handle_, activation_, &alpha,
                                                  descriptors_[0], input, &zero,
                                                  descriptors_[1], output),
                          "hipdnnActivationForward");
      return;
    }

    if (is_activation_backward(operation_.kind)) {
      require_workspace(workspace, workspace_size, "activation_backward");
      void *dy = binding_pointer(bindings, tensors_[0]);
      void *x = binding_pointer(bindings, tensors_[1]);
      void *dx = binding_pointer(bindings, tensors_[2]);
      check_hipdnn_status(hipdnnActivationForward(handle_, activation_, &one,
                                                  descriptors_[1], x, &zero,
                                                  descriptors_[2], workspace),
                          "hipdnnActivationForward(activation_backward y)");
      check_hipdnn_status(hipdnnActivationBackward(
                              handle_, activation_, &one, descriptors_[2],
                              workspace, descriptors_[0], dy, descriptors_[1],
                              x, &zero, descriptors_[2], dx),
                          "hipdnnActivationBackward(activation_backward dx)");
      return;
    }

    void *left = binding_pointer(bindings, tensors_[0]);
    void *right = binding_pointer(bindings, tensors_[1]);
    void *output = binding_pointer(bindings, tensors_[2]);
    float right_alpha = 1.0F;
    if (operation_.kind == HipdnnPointwiseKind::kAdd) {
      right_alpha = static_cast<float>(operation_.alpha);
    } else if (operation_.kind == HipdnnPointwiseKind::kSub) {
      right_alpha = static_cast<float>(-operation_.alpha);
    }
    if (operation_.kind != HipdnnPointwiseKind::kAddSquare) {
      check_hipdnn_status(hipdnnOpTensor(handle_, first_operation_, &one,
                                         descriptors_[0], left, &right_alpha,
                                         descriptors_[1], right, &zero,
                                         descriptors_[2], output),
                          "hipdnnOpTensor");
      return;
    }

    require_workspace(workspace, workspace_size, "add_square");
    check_hipdnn_status(hipdnnOpTensor(handle_, first_operation_, &one,
                                       descriptors_[1], right, &one,
                                       descriptors_[1], right, &zero,
                                       descriptors_[2], workspace),
                        "hipdnnOpTensor(add_square mul)");
    check_hipdnn_status(hipdnnOpTensor(handle_, second_operation_, &one,
                                       descriptors_[0], left, &one,
                                       descriptors_[2], workspace, &zero,
                                       descriptors_[2], output),
                        "hipdnnOpTensor(add_square add)");
  }

private:
  void require_workspace(void *workspace, std::size_t workspace_size,
                         std::string_view operation) const {
    if (workspace == nullptr || workspace_size < workspace_size_) {
      throw std::invalid_argument("hipDNN " + std::string(operation) +
                                  " workspace is too small");
    }
  }

  void cleanup() noexcept {
    if (activation_ != nullptr) {
      (void)hipdnnDestroyActivationDescriptor(activation_);
      activation_ = nullptr;
    }
    if (second_operation_ != nullptr) {
      (void)hipdnnDestroyOpTensorDescriptor(second_operation_);
      second_operation_ = nullptr;
    }
    if (first_operation_ != nullptr) {
      (void)hipdnnDestroyOpTensorDescriptor(first_operation_);
      first_operation_ = nullptr;
    }
    for (hipdnnTensorDescriptor_t &descriptor : descriptors_) {
      if (descriptor != nullptr) {
        (void)hipdnnDestroyTensorDescriptor(descriptor);
        descriptor = nullptr;
      }
    }
    if (handle_ != nullptr) {
      (void)hipdnnDestroy(handle_);
      handle_ = nullptr;
    }
  }

  HipdnnPointwiseOperation operation_;
  std::vector<ReferenceTensor> tensors_;
  hipdnnHandle_t handle_ = nullptr;
  std::vector<hipdnnTensorDescriptor_t> descriptors_;
  hipdnnOpTensorDescriptor_t first_operation_ = nullptr;
  hipdnnOpTensorDescriptor_t second_operation_ = nullptr;
  hipdnnActivationDescriptor_t activation_ = nullptr;
  std::size_t workspace_size_ = 0;
};

HipdnnPointwisePlan::HipdnnPointwisePlan(HipdnnPointwiseOperation operation,
                                         std::vector<ReferenceTensor> tensors)
    : impl_(std::make_unique<Impl>(std::move(operation), std::move(tensors))) {}

HipdnnPointwisePlan::~HipdnnPointwisePlan() = default;
HipdnnPointwisePlan::HipdnnPointwisePlan(HipdnnPointwisePlan &&) noexcept =
    default;
HipdnnPointwisePlan &
HipdnnPointwisePlan::operator=(HipdnnPointwisePlan &&) noexcept = default;

std::size_t HipdnnPointwisePlan::workspace_size() const noexcept {
  return impl_->workspace_size();
}

void HipdnnPointwisePlan::execute(std::span<const flagdnnBinding_t> bindings,
                                  void *workspace, std::size_t workspace_size,
                                  flagdnnStream_t stream) {
  impl_->execute(bindings, workspace, workspace_size, stream);
}

} // namespace flagdnn::validation::hygon
