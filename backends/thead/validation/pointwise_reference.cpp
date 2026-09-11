// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "pointwise_reference.hpp"

#include "acdnn_reference.hpp"
#include "acdnn_pointwise_dag.hpp"
#include "backend_pointwise_reference.hpp"
#include "numeric_types.hpp"

#include <acdnn.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace flagdnn::validation::thead {
namespace {

acdnnDataType_t acdnn_data_type(flagdnnDataType_t data_type) {
  switch (data_type) {
    case FLAGDNN_DATA_FLOAT32:
      return ACDNN_DATA_FLOAT;
    case FLAGDNN_DATA_FLOAT16:
      return ACDNN_DATA_HALF;
    case FLAGDNN_DATA_BFLOAT16:
      return ACDNN_DATA_BF16;
    case FLAGDNN_DATA_BOOLEAN:
    case FLAGDNN_DATA_FP8_E4M3:
    case FLAGDNN_DATA_FP8_E5M2:
      break;
  }
  throw std::invalid_argument(
      "qualified THead acDNN pointwise path requires a floating type");
}

bool is_floating_data_type(flagdnnDataType_t data_type) {
  return data_type == FLAGDNN_DATA_FLOAT32 ||
         data_type == FLAGDNN_DATA_FLOAT16 ||
         data_type == FLAGDNN_DATA_BFLOAT16;
}

std::vector<int> checked_ints(std::span<const std::int64_t> values,
                              std::string_view description) {
  std::vector<int> result;
  result.reserve(values.size());
  for (const std::int64_t value : values) {
    if (value <= 0 || value > std::numeric_limits<int>::max()) {
      throw std::invalid_argument(std::string(description) +
                                  " is outside positive int32 range");
    }
    result.push_back(static_cast<int>(value));
  }
  return result;
}

void require_same_tensor_geometry(
    const flagdnn::testing::TestTensor &left,
    const flagdnn::testing::TestTensor &right) {
  if (left.data_type != right.data_type ||
      left.dimensions != right.dimensions || left.strides != right.strides) {
    throw std::invalid_argument(
        "qualified THead acDNN pointwise operation requires identical "
        "tensor geometry");
  }
}

struct AcdnnBinaryOperation {
  acdnnOpTensorOp_t operation;
  std::string primitive;
  std::string name;
  float right_coefficient;
  bool unary;
};

struct AcdnnActivationOperation {
  acdnnActivationMode_t operation;
  std::string primitive;
  std::string name;
  double coefficient;
  std::uint64_t allowed_attribute_flags;
  double expected_relu_lower_clip_slope;
  bool uses_transform;
};

struct AcdnnBackendUnaryOperation {
  acdnnPointwiseMode_t operation;
  std::string primitive;
  std::string name;
  std::uint64_t allowed_attribute_flags;
  double expected_softplus_beta;
  double expected_swish_beta;
};

constexpr std::array<std::string_view, 2> kConvertInputFp32 = {
    "acdnnBackendExecute(POINTWISE_IDENTITY_FWD,convert-input0-fp32)",
    "acdnnBackendExecute(POINTWISE_IDENTITY_FWD,convert-input1-fp32)",
};
constexpr std::string_view kConvertOutputBfloat16 =
    "acdnnBackendExecute(POINTWISE_IDENTITY_FWD,convert-output-bfloat16)";
constexpr std::string_view kLeakyConvertInput =
    "acdnnBackendExecute(POINTWISE_IDENTITY_FWD,leaky-input-fp32)";
constexpr std::string_view kLeakyNeg =
    "acdnnTransformTensor(alpha=-1,beta=0)";
constexpr std::string_view kLeakyPositiveRelu =
    "acdnnActivationForward(RELU,NOT_PROPAGATE_NAN,positive)";
constexpr std::string_view kLeakyNegativeRelu =
    "acdnnActivationForward(RELU,NOT_PROPAGATE_NAN,negative)";
constexpr std::string_view kLeakyCombine =
    "acdnnOpTensor(ADD,alpha_right=-0.2)";
constexpr std::string_view kLeakyConvertOutput =
    "acdnnBackendExecute(POINTWISE_IDENTITY_FWD,leaky-output-data-type)";

AcdnnBackendUnaryOperation backend_unary_operation(
    flagdnnPointwiseMode_t mode) {
  switch (mode) {
    case FLAGDNN_POINTWISE_ADD:
      return {ACDNN_POINTWISE_ADD,
              "acdnnBackendExecute(POINTWISE_ADD)", "Add", 0U, 1.0,
              1.0};
    case FLAGDNN_POINTWISE_SUB:
      return {ACDNN_POINTWISE_SUB,
              "acdnnBackendExecute(POINTWISE_SUB)", "Sub", 0U, 1.0,
              1.0};
    case FLAGDNN_POINTWISE_MUL:
      return {ACDNN_POINTWISE_MUL,
              "acdnnBackendExecute(POINTWISE_MUL)", "Mul", 0U, 1.0,
              1.0};
    case FLAGDNN_POINTWISE_MIN:
      return {ACDNN_POINTWISE_MIN,
              "acdnnBackendExecute(POINTWISE_MIN)", "Min", 0U, 1.0,
              1.0};
    case FLAGDNN_POINTWISE_MAX:
      return {ACDNN_POINTWISE_MAX,
              "acdnnBackendExecute(POINTWISE_MAX)", "Max", 0U, 1.0,
              1.0};
    case FLAGDNN_POINTWISE_SQRT:
      return {ACDNN_POINTWISE_SQRT,
              "acdnnBackendExecute(POINTWISE_SQRT)", "Sqrt", 0U, 1.0,
              1.0};
    case FLAGDNN_POINTWISE_RELU_FWD:
      return {ACDNN_POINTWISE_RELU_FWD,
              "acdnnBackendExecute(POINTWISE_RELU_FWD)", "ReLU", 0U,
              1.0, 1.0};
    case FLAGDNN_POINTWISE_SIGMOID_FWD:
      return {ACDNN_POINTWISE_SIGMOID_FWD,
              "acdnnBackendExecute(POINTWISE_SIGMOID_FWD)", "Sigmoid",
              0U, 1.0, 1.0};
    case FLAGDNN_POINTWISE_TANH_FWD:
      return {ACDNN_POINTWISE_TANH_FWD,
              "acdnnBackendExecute(POINTWISE_TANH_FWD)", "Tanh", 0U,
              1.0, 1.0};
    case FLAGDNN_POINTWISE_ELU_FWD:
      return {ACDNN_POINTWISE_ELU_FWD,
              "acdnnBackendExecute(POINTWISE_ELU_FWD,alpha=1)", "Elu",
              FLAGDNN_POINTWISE_ATTRIBUTE_ELU_ALPHA, 1.0, 1.0};
    case FLAGDNN_POINTWISE_GELU_FWD:
      return {ACDNN_POINTWISE_GELU_FWD,
              "acdnnBackendExecute(POINTWISE_GELU_FWD)", "Gelu", 0U,
              1.0, 1.0};
    case FLAGDNN_POINTWISE_IDENTITY:
      return {ACDNN_POINTWISE_IDENTITY_FWD,
              "acdnnBackendExecute(POINTWISE_IDENTITY_FWD)", "Identity",
              0U, 1.0, 1.0};
    case FLAGDNN_POINTWISE_NEG:
      return {ACDNN_POINTWISE_NEG,
              "acdnnBackendExecute(POINTWISE_NEG)", "Neg", 0U, 1.0,
              1.0};
    case FLAGDNN_POINTWISE_ABS:
      return {ACDNN_POINTWISE_ABS,
              "acdnnBackendExecute(POINTWISE_ABS)", "Abs", 0U, 1.0,
              1.0};
    case FLAGDNN_POINTWISE_CEIL:
      return {ACDNN_POINTWISE_CEIL,
              "acdnnBackendExecute(POINTWISE_CEIL)", "Ceil", 0U, 1.0,
              1.0};
    case FLAGDNN_POINTWISE_FLOOR:
      return {ACDNN_POINTWISE_FLOOR,
              "acdnnBackendExecute(POINTWISE_FLOOR)", "Floor", 0U, 1.0,
              1.0};
    case FLAGDNN_POINTWISE_EXP:
      return {ACDNN_POINTWISE_EXP,
              "acdnnBackendExecute(POINTWISE_EXP)", "Exp", 0U, 1.0,
              1.0};
    case FLAGDNN_POINTWISE_LOG:
      return {ACDNN_POINTWISE_LOG,
              "acdnnBackendExecute(POINTWISE_LOG)", "Log", 0U, 1.0,
              1.0};
    case FLAGDNN_POINTWISE_COS:
      return {ACDNN_POINTWISE_COS,
              "acdnnBackendExecute(POINTWISE_COS)", "Cos", 0U, 1.0,
              1.0};
    case FLAGDNN_POINTWISE_RSQRT:
      return {ACDNN_POINTWISE_RSQRT,
              "acdnnBackendExecute(POINTWISE_RSQRT)", "Rsqrt", 0U, 1.0,
              1.0};
    case FLAGDNN_POINTWISE_SIN:
      return {ACDNN_POINTWISE_SIN,
              "acdnnBackendExecute(POINTWISE_SIN)", "Sin", 0U, 1.0,
              1.0};
    case FLAGDNN_POINTWISE_TAN:
      return {ACDNN_POINTWISE_TAN,
              "acdnnBackendExecute(POINTWISE_TAN)", "Tan", 0U, 1.0,
              1.0};
    case FLAGDNN_POINTWISE_SOFTPLUS_FWD:
      return {ACDNN_POINTWISE_SOFTPLUS_FWD,
              "acdnnBackendExecute(POINTWISE_SOFTPLUS_FWD,beta=1)",
              "Softplus", FLAGDNN_POINTWISE_ATTRIBUTE_SOFTPLUS_BETA, 1.0,
              1.0};
    case FLAGDNN_POINTWISE_SWISH_FWD:
      return {ACDNN_POINTWISE_SWISH_FWD,
              "acdnnBackendExecute(POINTWISE_SWISH_FWD,beta=1.25)",
              "Swish", FLAGDNN_POINTWISE_ATTRIBUTE_SWISH_BETA, 1.0,
              1.25};
    case FLAGDNN_POINTWISE_GELU_APPROX_TANH_FWD:
      return {ACDNN_POINTWISE_GELU_APPROX_TANH_FWD,
              "acdnnBackendExecute(POINTWISE_GELU_APPROX_TANH_FWD)",
              "GeluApproxTanh", 0U, 1.0, 1.0};
    case FLAGDNN_POINTWISE_DIV:
      return {ACDNN_POINTWISE_DIV,
              "acdnnBackendExecute(POINTWISE_DIV)", "Div", 0U, 1.0,
              1.0};
    case FLAGDNN_POINTWISE_POW:
      return {ACDNN_POINTWISE_POW,
              "acdnnBackendExecute(POINTWISE_POW)", "Pow", 0U, 1.0,
              1.0};
    case FLAGDNN_POINTWISE_MOD:
      return {ACDNN_POINTWISE_MOD,
              "acdnnBackendExecute(POINTWISE_MOD)", "Mod", 0U, 1.0,
              1.0};
    case FLAGDNN_POINTWISE_SIGMOID_BWD:
      return {ACDNN_POINTWISE_SIGMOID_BWD,
              "acdnnBackendExecute(POINTWISE_SIGMOID_BWD)",
              "SigmoidBackward", 0U, 1.0, 1.0};
    case FLAGDNN_POINTWISE_RECIPROCAL:
      return {ACDNN_POINTWISE_DIV,
              "acdnnBackendExecute(POINTWISE_DIV,numerator=1)",
              "Reciprocal", 0U, 1.0, 1.0};
    case FLAGDNN_POINTWISE_CMP_EQ:
      return {ACDNN_POINTWISE_CMP_EQ,
              "acdnnBackendExecute(POINTWISE_CMP_EQ)", "CmpEq", 0U,
              1.0, 1.0};
    case FLAGDNN_POINTWISE_CMP_NEQ:
      return {ACDNN_POINTWISE_CMP_NEQ,
              "acdnnBackendExecute(POINTWISE_CMP_NEQ)", "CmpNeq", 0U,
              1.0, 1.0};
    case FLAGDNN_POINTWISE_CMP_GT:
      return {ACDNN_POINTWISE_CMP_GT,
              "acdnnBackendExecute(POINTWISE_CMP_GT)", "CmpGt", 0U,
              1.0, 1.0};
    case FLAGDNN_POINTWISE_CMP_GE:
      return {ACDNN_POINTWISE_CMP_GE,
              "acdnnBackendExecute(POINTWISE_CMP_GE)", "CmpGe", 0U,
              1.0, 1.0};
    case FLAGDNN_POINTWISE_CMP_LT:
      return {ACDNN_POINTWISE_CMP_LT,
              "acdnnBackendExecute(POINTWISE_CMP_LT)", "CmpLt", 0U,
              1.0, 1.0};
    case FLAGDNN_POINTWISE_CMP_LE:
      return {ACDNN_POINTWISE_CMP_LE,
              "acdnnBackendExecute(POINTWISE_CMP_LE)", "CmpLe", 0U,
              1.0, 1.0};
    default:
      throw std::invalid_argument(
          "THead acDNN backend unary reference mode is not qualified");
  }
}

void require_activation_attributes(
    const flagdnnPointwiseAttributes_t &attributes,
    std::string_view operation_name,
    std::uint64_t allowed_attribute_flags,
    double expected_relu_lower_clip_slope,
    double expected_softplus_beta = 1.0,
    double expected_swish_beta = 1.0) {
  if (attributes.struct_size < sizeof(flagdnnPointwiseAttributes_t) ||
      attributes.version != FLAGDNN_POINTWISE_ATTRIBUTES_VERSION ||
      (attributes.flags & ~allowed_attribute_flags) != 0U ||
      attributes.relu_lower_clip != 0.0 ||
      attributes.relu_upper_clip != 0.0 ||
      attributes.relu_lower_clip_slope != expected_relu_lower_clip_slope ||
      attributes.swish_beta != expected_swish_beta ||
      attributes.elu_alpha != 1.0 ||
      attributes.softplus_beta != expected_softplus_beta) {
    throw std::invalid_argument(
        "qualified THead acDNN " + std::string(operation_name) +
        " requires default attributes");
  }
}

AcdnnBinaryOperation binary_operation(flagdnnPointwiseMode_t mode) {
  switch (mode) {
    case FLAGDNN_POINTWISE_ADD:
      return {ACDNN_OP_TENSOR_ADD, "acdnnOpTensor(ADD)", "Add", 1.0F,
              false};
    case FLAGDNN_POINTWISE_SUB:
      return {ACDNN_OP_TENSOR_ADD,
              "acdnnOpTensor(ADD,alpha_right=-alpha)", "Sub", -1.0F,
              false};
    case FLAGDNN_POINTWISE_MUL:
      return {ACDNN_OP_TENSOR_MUL, "acdnnOpTensor(MUL)", "Mul", 1.0F,
              false};
    case FLAGDNN_POINTWISE_MIN:
      return {ACDNN_OP_TENSOR_MIN, "acdnnOpTensor(MIN)", "Min", 1.0F,
              false};
    case FLAGDNN_POINTWISE_MAX:
      return {ACDNN_OP_TENSOR_MAX, "acdnnOpTensor(MAX)", "Max", 1.0F,
              false};
    case FLAGDNN_POINTWISE_SQRT:
      return {ACDNN_OP_TENSOR_SQRT, "acdnnOpTensor(SQRT)", "Sqrt", 1.0F,
              true};
    default:
      throw std::invalid_argument(
          "THead acDNN pointwise reference supports only qualified "
          "Add/Sub/Mul/Min/Max/Sqrt");
  }
}

AcdnnActivationOperation activation_operation(flagdnnPointwiseMode_t mode) {
  switch (mode) {
    case FLAGDNN_POINTWISE_RELU_FWD:
      return {ACDNN_ACTIVATION_RELU,
              "acdnnActivationForward(RELU,NOT_PROPAGATE_NAN)", "ReLU",
              0.0, 0U, 0.0, false};
    case FLAGDNN_POINTWISE_SIGMOID_FWD:
      return {ACDNN_ACTIVATION_SIGMOID,
              "acdnnActivationForward(SIGMOID,NOT_PROPAGATE_NAN)",
              "Sigmoid", 0.0, 0U, 0.0, false};
    case FLAGDNN_POINTWISE_TANH_FWD:
      return {ACDNN_ACTIVATION_TANH,
              "acdnnActivationForward(TANH,NOT_PROPAGATE_NAN)", "Tanh",
              0.0, 0U, 0.0, false};
    case FLAGDNN_POINTWISE_ELU_FWD:
      return {ACDNN_ACTIVATION_ELU,
              "acdnnActivationForward(ELU,NOT_PROPAGATE_NAN,alpha=1)",
              "Elu", 1.0, FLAGDNN_POINTWISE_ATTRIBUTE_ELU_ALPHA, 0.0,
              false};
    case FLAGDNN_POINTWISE_IDENTITY:
      return {ACDNN_ACTIVATION_IDENTITY,
              "acdnnTransformTensor(alpha=1,beta=0)", "Identity", 0.0,
              0U, 0.0, true};
    case FLAGDNN_POINTWISE_GELU_FWD:
      return {ACDNN_ACTIVATION_GELU,
              "acdnnActivationForward(GELU,NOT_PROPAGATE_NAN)", "Gelu",
              0.0, 0U, 0.0, false};
    case FLAGDNN_POINTWISE_NEG:
      return {ACDNN_ACTIVATION_IDENTITY,
              "acdnnTransformTensor(alpha=-1,beta=0)", "Neg", 0.0,
              0U, 0.0, true};
    default:
      throw std::invalid_argument(
          "THead acDNN activation reference supports only qualified "
          "ReLU/Sigmoid/Tanh/Elu/Identity/Gelu/Neg");
  }
}

class AcdnnPointwiseBinary final
    : public flagdnn::testing::TestExecutable {
 public:
  AcdnnPointwiseBinary(PointwiseReferenceSpecification specification,
                       CapabilityRecord capability)
      : specification_(std::move(specification)),
        capability_(std::move(capability)),
        binary_operation_(binary_operation(specification_.mode)) {
    const std::size_t expected_inputs = binary_operation_.unary ? 1U : 2U;
    if (specification_.inputs.size() != expected_inputs) {
      throw std::invalid_argument(
          "THead acDNN OpTensor reference has wrong input arity");
    }
    const auto selection = select_reference(capability_);
    if (!std::holds_alternative<ReferencePlan>(selection)) {
      throw std::invalid_argument(
          "unsupported capability reached acDNN pointwise construction");
    }
    const ReferencePlan &plan = std::get<ReferencePlan>(selection);
    if (plan.path != ReferencePath::kStablePrimitive ||
        plan.primitives !=
            std::vector<std::string>{binary_operation_.primitive}) {
      throw std::invalid_argument(
          "THead acDNN pointwise reference plan mismatch");
    }
    const bool scaled_binary =
        specification_.mode == FLAGDNN_POINTWISE_ADD ||
        specification_.mode == FLAGDNN_POINTWISE_SUB;
    if (specification_.alpha != 1.0 && !scaled_binary) {
      throw std::invalid_argument(
          "qualified THead acDNN pointwise operation does not support alpha");
    }
    if (!binary_operation_.unary) {
      require_same_tensor_geometry(specification_.inputs[0],
                                   specification_.inputs[1]);
    } else {
      require_activation_attributes(specification_.attributes,
                                    binary_operation_.name, 0U, 0.0);
    }
    require_same_tensor_geometry(specification_.inputs[0],
                                 specification_.output);
    if (specification_.output.dimensions.empty()) {
      throw std::invalid_argument(
          "THead acDNN pointwise operation requires rank at least one");
    }

    const acdnnDataType_t data_type =
        acdnn_data_type(specification_.output.data_type);
    const std::vector<int> dimensions = checked_ints(
        specification_.output.dimensions,
        "acDNN " + binary_operation_.name + " dimension");
    const std::vector<int> strides =
        checked_ints(specification_.output.strides,
                     "acDNN " + binary_operation_.name + " stride");
    left_.set(data_type, dimensions, strides);
    right_.set(data_type, dimensions, strides);
    output_.set(data_type, dimensions, strides);
    operation_.set(binary_operation_.operation, ACDNN_DATA_FLOAT,
                   ACDNN_PROPAGATE_NAN);
  }

  [[nodiscard]] std::size_t workspace_size() const noexcept override {
    return 0;
  }

  void execute(std::span<const flagdnnBinding_t> bindings, void *workspace,
               std::size_t workspace_size,
               flagdnnStream_t stream) override {
    if (workspace != nullptr || workspace_size != 0) {
      throw std::invalid_argument(
          "acDNN pointwise stable primitive requires zero workspace");
    }
    if (stream == nullptr) {
      throw std::invalid_argument(
          "acDNN pointwise operation requires a non-default stream");
    }

    std::map<std::int64_t, void *> pointers;
    for (const flagdnnBinding_t &binding : bindings) {
      if (binding.device_pointer == nullptr ||
          !pointers.emplace(binding.uid, binding.device_pointer).second) {
        throw std::invalid_argument(
            "acDNN pointwise bindings contain null or duplicate entries");
      }
    }
    const auto require_pointer = [this, &pointers](std::int64_t uid) {
      const auto iterator = pointers.find(uid);
      if (iterator == pointers.end()) {
        throw std::invalid_argument("acDNN pointwise binding is missing");
      }
      if (reinterpret_cast<std::uintptr_t>(iterator->second) %
              element_size(specification_.output.data_type) !=
          0) {
        throw std::invalid_argument("acDNN pointwise binding is misaligned");
      }
      return iterator->second;
    };
    const std::size_t expected_bindings = binary_operation_.unary ? 2U : 3U;
    if (pointers.size() != expected_bindings) {
      throw std::invalid_argument(
          "acDNN pointwise operation has wrong binding count");
    }
    void *left_pointer = require_pointer(specification_.inputs[0].uid);
    void *right_pointer =
        binary_operation_.unary
            ? left_pointer
            : require_pointer(specification_.inputs[1].uid);
    void *output_pointer = require_pointer(specification_.output.uid);

    handle_.set_stream(reinterpret_cast<hggcStream_t>(stream));
    constexpr float alpha_left = 1.0F;
    const float alpha_right = binary_operation_.right_coefficient *
                              static_cast<float>(specification_.alpha);
    constexpr float beta = 0.0F;
    require_reference_status(
        capability_,
        acdnnOpTensor(handle_.get(), operation_.get(), &alpha_left,
                      left_.get(), left_pointer, &alpha_right, right_.get(),
                      right_pointer, &beta, output_.get(), output_pointer),
        binary_operation_.primitive);
  }

 private:
  PointwiseReferenceSpecification specification_;
  CapabilityRecord capability_;
  AcdnnBinaryOperation binary_operation_;
  AcdnnHandle handle_;
  AcdnnTensorDescriptor left_;
  AcdnnTensorDescriptor right_;
  AcdnnTensorDescriptor output_;
  AcdnnOpTensorDescriptor operation_;
};

class AcdnnPointwiseActivation final
    : public flagdnn::testing::TestExecutable {
 public:
  AcdnnPointwiseActivation(PointwiseReferenceSpecification specification,
                           CapabilityRecord capability)
      : specification_(std::move(specification)),
        capability_(std::move(capability)),
        activation_operation_(activation_operation(specification_.mode)) {
    if (specification_.inputs.size() != 1) {
      throw std::invalid_argument(
          "THead acDNN " + activation_operation_.name +
          " reference requires one unary input");
    }
    const auto selection = select_reference(capability_);
    if (!std::holds_alternative<ReferencePlan>(selection)) {
      throw std::invalid_argument(
          "unsupported capability reached acDNN " +
          activation_operation_.name + " construction");
    }
    const ReferencePlan &plan = std::get<ReferencePlan>(selection);
    if (plan.path != ReferencePath::kStablePrimitive ||
        plan.primitives !=
            std::vector<std::string>{activation_operation_.primitive}) {
      throw std::invalid_argument("THead acDNN " +
                                  activation_operation_.name +
                                  " reference plan mismatch");
    }
    if (specification_.alpha != 1.0) {
      throw std::invalid_argument(
          "qualified THead acDNN " + activation_operation_.name +
          " requires alpha equal to one");
    }
    require_activation_attributes(
        specification_.attributes, activation_operation_.name,
        activation_operation_.allowed_attribute_flags,
        activation_operation_.expected_relu_lower_clip_slope);
    require_same_tensor_geometry(specification_.inputs[0],
                                 specification_.output);
    if (specification_.output.dimensions.empty()) {
      throw std::invalid_argument(
          "THead acDNN " + activation_operation_.name +
          " requires rank at least one");
    }

    const acdnnDataType_t data_type =
        acdnn_data_type(specification_.output.data_type);
    const std::vector<int> dimensions = checked_ints(
        specification_.output.dimensions,
        "acDNN " + activation_operation_.name + " dimension");
    const std::vector<int> strides = checked_ints(
        specification_.output.strides,
        "acDNN " + activation_operation_.name + " stride");
    input_.set(data_type, dimensions, strides);
    output_.set(data_type, dimensions, strides);
    if (!activation_operation_.uses_transform) {
      activation_.set(activation_operation_.operation,
                      ACDNN_NOT_PROPAGATE_NAN,
                      activation_operation_.coefficient);
    }
  }

  [[nodiscard]] std::size_t workspace_size() const noexcept override {
    return 0;
  }

  void execute(std::span<const flagdnnBinding_t> bindings, void *workspace,
               std::size_t workspace_size,
               flagdnnStream_t stream) override {
    if (workspace != nullptr || workspace_size != 0) {
      throw std::invalid_argument(
          "acDNN " + activation_operation_.name +
          " stable primitive requires zero workspace");
    }
    if (stream == nullptr) {
      throw std::invalid_argument(
          "acDNN " + activation_operation_.name +
          " requires a non-default stream");
    }

    std::map<std::int64_t, void *> pointers;
    for (const flagdnnBinding_t &binding : bindings) {
      if (binding.device_pointer == nullptr ||
          !pointers.emplace(binding.uid, binding.device_pointer).second) {
        throw std::invalid_argument(
            "acDNN " + activation_operation_.name +
            " bindings contain null or duplicate entries");
      }
    }
    const auto require_pointer = [this, &pointers](std::int64_t uid) {
      const auto iterator = pointers.find(uid);
      if (iterator == pointers.end()) {
        throw std::invalid_argument("acDNN " + activation_operation_.name +
                                    " binding is missing");
      }
      if (reinterpret_cast<std::uintptr_t>(iterator->second) %
              element_size(specification_.output.data_type) !=
          0) {
        throw std::invalid_argument("acDNN " + activation_operation_.name +
                                    " binding is misaligned");
      }
      return iterator->second;
    };
    if (pointers.size() != 2) {
      throw std::invalid_argument(
          "acDNN " + activation_operation_.name +
          " requires exactly two bindings");
    }
    void *input_pointer = require_pointer(specification_.inputs[0].uid);
    void *output_pointer = require_pointer(specification_.output.uid);

    handle_.set_stream(reinterpret_cast<hggcStream_t>(stream));
    const float alpha = specification_.mode == FLAGDNN_POINTWISE_NEG
                            ? -1.0F
                            : 1.0F;
    constexpr float beta = 0.0F;
    const acdnnStatus_t status = activation_operation_.uses_transform
                                    ? acdnnTransformTensor(
                                          handle_.get(), &alpha, input_.get(),
                                          input_pointer, &beta, output_.get(),
                                          output_pointer)
                                    : acdnnActivationForward(
                                          handle_.get(), activation_.get(),
                                          &alpha, input_.get(), input_pointer,
                                          &beta, output_.get(), output_pointer);
    require_reference_status(capability_, status,
                             activation_operation_.primitive);
  }

 private:
  PointwiseReferenceSpecification specification_;
  CapabilityRecord capability_;
  AcdnnActivationOperation activation_operation_;
  AcdnnHandle handle_;
  AcdnnTensorDescriptor input_;
  AcdnnTensorDescriptor output_;
  AcdnnActivationDescriptor activation_;
};

std::size_t tensor_storage_bytes(
    const flagdnn::testing::TestTensor &tensor) {
  std::size_t maximum_offset = 0;
  if (tensor.dimensions.empty() ||
      tensor.dimensions.size() != tensor.strides.size()) {
    throw std::invalid_argument(
        "converted acDNN pointwise tensor geometry is invalid");
  }
  for (std::size_t axis = 0; axis < tensor.dimensions.size(); ++axis) {
    if (tensor.dimensions[axis] <= 0 || tensor.strides[axis] <= 0) {
      throw std::invalid_argument(
          "converted acDNN pointwise tensor geometry is invalid");
    }
    const std::size_t extent =
        static_cast<std::size_t>(tensor.dimensions[axis] - 1);
    const std::size_t stride =
        static_cast<std::size_t>(tensor.strides[axis]);
    if (stride != 0 &&
        extent > std::numeric_limits<std::size_t>::max() / stride) {
      throw std::overflow_error(
          "converted acDNN pointwise tensor storage overflows");
    }
    const std::size_t contribution = extent * stride;
    if (maximum_offset >
        std::numeric_limits<std::size_t>::max() - contribution) {
      throw std::overflow_error(
          "converted acDNN pointwise tensor storage overflows");
    }
    maximum_offset += contribution;
  }
  if (maximum_offset == std::numeric_limits<std::size_t>::max() ||
      maximum_offset + 1 >
          std::numeric_limits<std::size_t>::max() / sizeof(float)) {
    throw std::overflow_error(
        "converted acDNN pointwise tensor storage is too large");
  }
  return (maximum_offset + 1) * sizeof(float);
}

CapabilityRecord single_operation_capability(
    ReferencePath path, std::string primitive,
    const CapabilityRecord &source) {
  return {
      .status = source.status,
      .path = path,
      .reference_plan = {std::move(primitive)},
      .constraints = source.constraints,
      .reason_code = source.status == CapabilityStatus::kProbeRequired
                         ? "real_device_qualification_pending"
                         : "",
      .detail = "acDNN primitive in converted BF16 pointwise DAG",
  };
}

class AcdnnConvertedBfloat16Pointwise final
    : public flagdnn::testing::TestExecutable {
 public:
  AcdnnConvertedBfloat16Pointwise(
      PointwiseReferenceSpecification specification,
      const CapabilityRecord &capability)
      : specification_(std::move(specification)) {
    const bool binary =
        specification_.mode == FLAGDNN_POINTWISE_MIN ||
        specification_.mode == FLAGDNN_POINTWISE_MAX;
    if ((!binary &&
         specification_.mode != FLAGDNN_POINTWISE_RELU_FWD) ||
        specification_.inputs.size() != (binary ? 2U : 1U) ||
        specification_.output.data_type != FLAGDNN_DATA_BFLOAT16 ||
        std::ranges::any_of(
            specification_.inputs,
            [](const flagdnn::testing::TestTensor &input) {
              return input.data_type != FLAGDNN_DATA_BFLOAT16;
            })) {
      throw std::invalid_argument(
          "converted acDNN pointwise DAG supports BF16 Min/Max/ReLU only");
    }
    const std::string operation_primitive =
        binary ? binary_operation(specification_.mode).primitive
               : activation_operation(specification_.mode).primitive;
    std::vector<std::string> expected_plan;
    expected_plan.reserve(specification_.inputs.size() + 2);
    for (std::size_t index = 0; index < specification_.inputs.size(); ++index) {
      expected_plan.emplace_back(kConvertInputFp32.at(index));
    }
    expected_plan.push_back(operation_primitive);
    expected_plan.push_back(std::string(kConvertOutputBfloat16));
    const ReferenceSelection selection = select_reference(capability);
    if (!std::holds_alternative<ReferencePlan>(selection)) {
      throw std::invalid_argument(
          "unsupported converted BF16 pointwise case reached construction");
    }
    const ReferencePlan &plan = std::get<ReferencePlan>(selection);
    if (plan.path != ReferencePath::kBackendDescriptor ||
        plan.primitives != expected_plan) {
      throw std::invalid_argument(
          "converted BF16 pointwise acDNN DAG plan mismatch");
    }

    std::int64_t maximum_uid = specification_.output.uid;
    for (const flagdnn::testing::TestTensor &input : specification_.inputs) {
      maximum_uid = std::max(maximum_uid, input.uid);
    }
    const std::size_t internal_count = specification_.inputs.size() + 1;
    if (maximum_uid >
        std::numeric_limits<std::int64_t>::max() -
            static_cast<std::int64_t>(internal_count)) {
      throw std::overflow_error(
          "converted BF16 pointwise internal UID overflows");
    }

    fp32_inputs_.reserve(specification_.inputs.size());
    input_buffers_.reserve(specification_.inputs.size());
    input_conversions_.reserve(specification_.inputs.size());
    for (const flagdnn::testing::TestTensor &input : specification_.inputs) {
      flagdnn::testing::TestTensor fp32 = input;
      fp32.uid = ++maximum_uid;
      fp32.data_type = FLAGDNN_DATA_FLOAT32;
      fp32.binding_byte_offset = 0;
      input_buffers_.emplace_back(tensor_storage_bytes(fp32));
      input_conversions_.push_back(
          make_acdnn_backend_pointwise_reference(
              {.mode = ACDNN_POINTWISE_IDENTITY_FWD,
               .inputs = {input},
               .output = fp32,
               .primitive = std::string(
                   kConvertInputFp32.at(input_conversions_.size()))}));
      workspace_size_ = std::max(
          workspace_size_, input_conversions_.back()->workspace_size());
      fp32_inputs_.push_back(std::move(fp32));
    }
    fp32_output_ = specification_.output;
    fp32_output_.uid = ++maximum_uid;
    fp32_output_.data_type = FLAGDNN_DATA_FLOAT32;
    fp32_output_.binding_byte_offset = 0;
    output_buffer_ = DeviceBuffer(tensor_storage_bytes(fp32_output_));

    operation_ = make_acdnn_pointwise_reference(
        {.mode = specification_.mode,
         .inputs = fp32_inputs_,
         .output = fp32_output_,
         .alpha = specification_.alpha,
         .attributes = specification_.attributes},
        single_operation_capability(
            ReferencePath::kStablePrimitive, operation_primitive,
            capability));
    output_conversion_ = make_acdnn_backend_pointwise_reference(
        {.mode = ACDNN_POINTWISE_IDENTITY_FWD,
         .inputs = {fp32_output_},
         .output = specification_.output,
         .primitive = std::string(kConvertOutputBfloat16)});
    workspace_size_ =
        std::max({workspace_size_, operation_->workspace_size(),
                  output_conversion_->workspace_size()});
  }

  void prepare(std::span<const flagdnnBinding_t> bindings,
               flagdnnStream_t stream) override {
    const auto pointers = binding_map(bindings);
    for (std::size_t index = 0; index < input_conversions_.size(); ++index) {
      const std::array<flagdnnBinding_t, 2> conversion_bindings = {{
          {specification_.inputs[index].uid,
           pointers.at(specification_.inputs[index].uid)},
          {fp32_inputs_[index].uid, input_buffers_[index].data()},
      }};
      input_conversions_[index]->prepare(conversion_bindings, stream);
    }
    const std::vector<flagdnnBinding_t> operation_bindings =
        internal_operation_bindings();
    operation_->prepare(operation_bindings, stream);
    const std::array<flagdnnBinding_t, 2> output_bindings = {{
        {fp32_output_.uid, output_buffer_.data()},
        {specification_.output.uid,
         pointers.at(specification_.output.uid)},
    }};
    output_conversion_->prepare(output_bindings, stream);
  }

  [[nodiscard]] std::size_t workspace_size() const noexcept override {
    return workspace_size_;
  }

  void execute(std::span<const flagdnnBinding_t> bindings, void *workspace,
               std::size_t workspace_size,
               flagdnnStream_t stream) override {
    if (workspace_size != workspace_size_ ||
        (workspace_size != 0 && workspace == nullptr)) {
      throw std::invalid_argument(
          "converted BF16 pointwise workspace does not match plan");
    }
    const auto pointers = binding_map(bindings);
    for (std::size_t index = 0; index < input_conversions_.size(); ++index) {
      const std::array<flagdnnBinding_t, 2> conversion_bindings = {{
          {specification_.inputs[index].uid,
           pointers.at(specification_.inputs[index].uid)},
          {fp32_inputs_[index].uid, input_buffers_[index].data()},
      }};
      execute_inner(*input_conversions_[index], conversion_bindings,
                    workspace, stream);
    }
    const std::vector<flagdnnBinding_t> operation_bindings =
        internal_operation_bindings();
    execute_inner(*operation_, operation_bindings, workspace, stream);
    const std::array<flagdnnBinding_t, 2> output_bindings = {{
        {fp32_output_.uid, output_buffer_.data()},
        {specification_.output.uid,
         pointers.at(specification_.output.uid)},
    }};
    execute_inner(*output_conversion_, output_bindings, workspace, stream);
  }

 private:
  [[nodiscard]] std::map<std::int64_t, void *> binding_map(
      std::span<const flagdnnBinding_t> bindings) const {
    std::map<std::int64_t, void *> result;
    for (const flagdnnBinding_t &binding : bindings) {
      if (binding.device_pointer == nullptr ||
          !result.emplace(binding.uid, binding.device_pointer).second) {
        throw std::invalid_argument(
            "converted BF16 pointwise binding is null or duplicate");
      }
    }
    if (result.size() != specification_.inputs.size() + 1 ||
        !result.contains(specification_.output.uid) ||
        std::ranges::any_of(
            specification_.inputs,
            [&result](const flagdnn::testing::TestTensor &input) {
              return !result.contains(input.uid);
            })) {
      throw std::invalid_argument(
          "converted BF16 pointwise bindings do not match graph");
    }
    return result;
  }

  [[nodiscard]] std::vector<flagdnnBinding_t>
  internal_operation_bindings() const {
    std::vector<flagdnnBinding_t> result;
    result.reserve(fp32_inputs_.size() + 1);
    for (std::size_t index = 0; index < fp32_inputs_.size(); ++index) {
      result.push_back(
          {fp32_inputs_[index].uid, input_buffers_[index].data()});
    }
    result.push_back({fp32_output_.uid, output_buffer_.data()});
    return result;
  }

  void execute_inner(flagdnn::testing::TestExecutable &executable,
                     std::span<const flagdnnBinding_t> bindings,
                     void *workspace, flagdnnStream_t stream) const {
    executable.execute(bindings,
                       executable.workspace_size() == 0 ? nullptr
                                                        : workspace,
                       executable.workspace_size(), stream);
  }

  PointwiseReferenceSpecification specification_;
  std::vector<flagdnn::testing::TestTensor> fp32_inputs_;
  flagdnn::testing::TestTensor fp32_output_;
  std::vector<DeviceBuffer> input_buffers_;
  DeviceBuffer output_buffer_;
  std::vector<std::unique_ptr<flagdnn::testing::TestExecutable>>
      input_conversions_;
  std::unique_ptr<flagdnn::testing::TestExecutable> operation_;
  std::unique_ptr<flagdnn::testing::TestExecutable> output_conversion_;
  std::size_t workspace_size_ = 0;
};

class AcdnnLeakyReluDag final
    : public flagdnn::testing::TestExecutable {
 public:
  AcdnnLeakyReluDag(PointwiseReferenceSpecification specification,
                    const CapabilityRecord &capability)
      : specification_(std::move(specification)) {
    const ReferenceSelection selection = select_reference(capability);
    const std::vector<std::string> expected_plan = {
        std::string(kLeakyConvertInput), std::string(kLeakyNeg),
        std::string(kLeakyPositiveRelu),
        std::string(kLeakyNegativeRelu), std::string(kLeakyCombine),
        std::string(kLeakyConvertOutput)};
    if (specification_.mode != FLAGDNN_POINTWISE_RELU_FWD ||
        specification_.inputs.size() != 1 || specification_.alpha != 1.0 ||
        specification_.inputs[0].data_type !=
            specification_.output.data_type ||
        !is_floating_data_type(specification_.output.data_type) ||
        specification_.attributes.flags !=
            FLAGDNN_POINTWISE_ATTRIBUTE_RELU_LOWER_CLIP_SLOPE ||
        specification_.attributes.relu_lower_clip != 0.0 ||
        specification_.attributes.relu_upper_clip != 0.0 ||
        specification_.attributes.relu_lower_clip_slope != 0.2 ||
        specification_.attributes.swish_beta != 1.0 ||
        specification_.attributes.elu_alpha != 1.0 ||
        specification_.attributes.softplus_beta != 1.0 ||
        !std::holds_alternative<ReferencePlan>(selection) ||
        std::get<ReferencePlan>(selection).path !=
            ReferencePath::kBackendDescriptor ||
        std::get<ReferencePlan>(selection).primitives != expected_plan) {
      throw std::invalid_argument("LeakyReLU acDNN DAG contract mismatch");
    }
    require_same_tensor_geometry(specification_.inputs[0],
                                 specification_.output);
    const std::int64_t maximum_uid =
        std::max(specification_.inputs[0].uid, specification_.output.uid);
    if (maximum_uid > std::numeric_limits<std::int64_t>::max() - 5) {
      throw std::overflow_error("LeakyReLU acDNN DAG UID overflows");
    }
    x_fp32_ = fp32_tensor(specification_.inputs[0], maximum_uid + 1);
    negative_ = fp32_tensor(specification_.output, maximum_uid + 2);
    positive_ = fp32_tensor(specification_.output, maximum_uid + 3);
    negative_positive_ =
        fp32_tensor(specification_.output, maximum_uid + 4);
    result_ = fp32_tensor(specification_.output, maximum_uid + 5);
    x_buffer_ = DeviceBuffer(tensor_storage_bytes(x_fp32_));
    negative_buffer_ = DeviceBuffer(tensor_storage_bytes(negative_));
    positive_buffer_ = DeviceBuffer(tensor_storage_bytes(positive_));
    negative_positive_buffer_ =
        DeviceBuffer(tensor_storage_bytes(negative_positive_));
    result_buffer_ = DeviceBuffer(tensor_storage_bytes(result_));

    input_conversion_ = make_acdnn_backend_pointwise_reference(
        {.mode = ACDNN_POINTWISE_IDENTITY_FWD,
         .inputs = {specification_.inputs[0]},
         .output = x_fp32_,
         .primitive = std::string(kLeakyConvertInput)});
    flagdnnPointwiseAttributes_t defaults =
        FLAGDNN_POINTWISE_ATTRIBUTES_INITIALIZER;
    negative_operation_ = make_acdnn_pointwise_reference(
        {.mode = FLAGDNN_POINTWISE_NEG,
         .inputs = {x_fp32_},
         .output = negative_,
         .alpha = 1.0,
         .attributes = defaults},
        single_operation_capability(ReferencePath::kStablePrimitive,
                                    std::string(kLeakyNeg), capability));
    positive_operation_ = make_acdnn_pointwise_reference(
        {.mode = FLAGDNN_POINTWISE_RELU_FWD,
         .inputs = {x_fp32_},
         .output = positive_,
         .alpha = 1.0,
         .attributes = defaults},
        single_operation_capability(
            ReferencePath::kStablePrimitive,
            "acdnnActivationForward(RELU,NOT_PROPAGATE_NAN)", capability));
    negative_positive_operation_ = make_acdnn_pointwise_reference(
        {.mode = FLAGDNN_POINTWISE_RELU_FWD,
         .inputs = {negative_},
         .output = negative_positive_,
         .alpha = 1.0,
         .attributes = defaults},
        single_operation_capability(
            ReferencePath::kStablePrimitive,
            "acdnnActivationForward(RELU,NOT_PROPAGATE_NAN)", capability));
    combine_operation_ = make_acdnn_pointwise_reference(
        {.mode = FLAGDNN_POINTWISE_SUB,
         .inputs = {positive_, negative_positive_},
         .output = result_,
         .alpha = 0.2,
         .attributes = defaults},
        single_operation_capability(
            ReferencePath::kStablePrimitive,
            "acdnnOpTensor(ADD,alpha_right=-alpha)", capability));
    output_conversion_ = make_acdnn_backend_pointwise_reference(
        {.mode = ACDNN_POINTWISE_IDENTITY_FWD,
         .inputs = {result_},
         .output = specification_.output,
         .primitive = std::string(kLeakyConvertOutput)});
    workspace_size_ = std::max(
        {input_conversion_->workspace_size(),
         negative_operation_->workspace_size(),
         positive_operation_->workspace_size(),
         negative_positive_operation_->workspace_size(),
         combine_operation_->workspace_size(),
         output_conversion_->workspace_size()});
  }

  void prepare(std::span<const flagdnnBinding_t> bindings,
               flagdnnStream_t stream) override {
    const auto pointers = binding_map(bindings);
    for_each_stage(
        pointers, stream,
        [](flagdnn::testing::TestExecutable &executable,
           std::span<const flagdnnBinding_t> stage_bindings,
           flagdnnStream_t stage_stream) {
          executable.prepare(stage_bindings, stage_stream);
        });
  }

  [[nodiscard]] std::size_t workspace_size() const noexcept override {
    return workspace_size_;
  }

  void execute(std::span<const flagdnnBinding_t> bindings, void *workspace,
               std::size_t workspace_size,
               flagdnnStream_t stream) override {
    if (workspace_size != workspace_size_ ||
        (workspace_size != 0 && workspace == nullptr)) {
      throw std::invalid_argument(
          "LeakyReLU acDNN DAG workspace does not match plan");
    }
    const auto pointers = binding_map(bindings);
    for_each_stage(
        pointers, stream,
        [workspace](flagdnn::testing::TestExecutable &executable,
                    std::span<const flagdnnBinding_t> stage_bindings,
                    flagdnnStream_t stage_stream) {
          executable.execute(
              stage_bindings,
              executable.workspace_size() == 0 ? nullptr : workspace,
              executable.workspace_size(), stage_stream);
        });
  }

 private:
  static flagdnn::testing::TestTensor fp32_tensor(
      const flagdnn::testing::TestTensor &source, std::int64_t uid) {
    flagdnn::testing::TestTensor result = source;
    result.uid = uid;
    result.data_type = FLAGDNN_DATA_FLOAT32;
    result.binding_byte_offset = 0;
    return result;
  }

  [[nodiscard]] std::map<std::int64_t, void *> binding_map(
      std::span<const flagdnnBinding_t> bindings) const {
    std::map<std::int64_t, void *> result;
    for (const flagdnnBinding_t &binding : bindings) {
      if (binding.device_pointer == nullptr ||
          !result.emplace(binding.uid, binding.device_pointer).second) {
        throw std::invalid_argument(
            "LeakyReLU acDNN DAG binding is null or duplicate");
      }
    }
    if (result.size() != 2 ||
        !result.contains(specification_.inputs[0].uid) ||
        !result.contains(specification_.output.uid)) {
      throw std::invalid_argument(
          "LeakyReLU acDNN DAG bindings do not match graph");
    }
    return result;
  }

  template <typename Invoke>
  void for_each_stage(const std::map<std::int64_t, void *> &pointers,
                      flagdnnStream_t stream, Invoke invoke) {
    const std::array<flagdnnBinding_t, 2> input_bindings = {{
        {specification_.inputs[0].uid,
         pointers.at(specification_.inputs[0].uid)},
        {x_fp32_.uid, x_buffer_.data()},
    }};
    const std::array<flagdnnBinding_t, 2> negative_bindings = {{
        {x_fp32_.uid, x_buffer_.data()},
        {negative_.uid, negative_buffer_.data()},
    }};
    const std::array<flagdnnBinding_t, 2> positive_bindings = {{
        {x_fp32_.uid, x_buffer_.data()},
        {positive_.uid, positive_buffer_.data()},
    }};
    const std::array<flagdnnBinding_t, 2> negative_positive_bindings = {{
        {negative_.uid, negative_buffer_.data()},
        {negative_positive_.uid, negative_positive_buffer_.data()},
    }};
    const std::array<flagdnnBinding_t, 3> combine_bindings = {{
        {positive_.uid, positive_buffer_.data()},
        {negative_positive_.uid, negative_positive_buffer_.data()},
        {result_.uid, result_buffer_.data()},
    }};
    const std::array<flagdnnBinding_t, 2> output_bindings = {{
        {result_.uid, result_buffer_.data()},
        {specification_.output.uid,
         pointers.at(specification_.output.uid)},
    }};
    invoke(*input_conversion_, input_bindings, stream);
    invoke(*negative_operation_, negative_bindings, stream);
    invoke(*positive_operation_, positive_bindings, stream);
    invoke(*negative_positive_operation_, negative_positive_bindings, stream);
    invoke(*combine_operation_, combine_bindings, stream);
    invoke(*output_conversion_, output_bindings, stream);
  }

  PointwiseReferenceSpecification specification_;
  flagdnn::testing::TestTensor x_fp32_;
  flagdnn::testing::TestTensor negative_;
  flagdnn::testing::TestTensor positive_;
  flagdnn::testing::TestTensor negative_positive_;
  flagdnn::testing::TestTensor result_;
  DeviceBuffer x_buffer_;
  DeviceBuffer negative_buffer_;
  DeviceBuffer positive_buffer_;
  DeviceBuffer negative_positive_buffer_;
  DeviceBuffer result_buffer_;
  std::unique_ptr<flagdnn::testing::TestExecutable> input_conversion_;
  std::unique_ptr<flagdnn::testing::TestExecutable> negative_operation_;
  std::unique_ptr<flagdnn::testing::TestExecutable> positive_operation_;
  std::unique_ptr<flagdnn::testing::TestExecutable>
      negative_positive_operation_;
  std::unique_ptr<flagdnn::testing::TestExecutable> combine_operation_;
  std::unique_ptr<flagdnn::testing::TestExecutable> output_conversion_;
  std::size_t workspace_size_ = 0;
};

}  // namespace

std::unique_ptr<flagdnn::testing::TestExecutable>
make_acdnn_pointwise_reference(
    const PointwiseReferenceSpecification &specification,
    const CapabilityRecord &capability) {
  if (!acdnn_pointwise_dag_plan(specification.mode).empty() &&
      capability.reference_plan.size() > 1) {
    return make_acdnn_pointwise_dag(specification, capability);
  }
  const bool leaky_relu =
      specification.mode == FLAGDNN_POINTWISE_RELU_FWD &&
      specification.attributes.flags ==
          FLAGDNN_POINTWISE_ATTRIBUTE_RELU_LOWER_CLIP_SLOPE;
  if (leaky_relu && capability.path == ReferencePath::kBackendDescriptor &&
      capability.reference_plan.size() > 1) {
    return std::make_unique<AcdnnLeakyReluDag>(specification, capability);
  }
  if (capability.path == ReferencePath::kBackendDescriptor &&
      capability.reference_plan.size() > 1) {
    return std::make_unique<AcdnnConvertedBfloat16Pointwise>(
        specification, capability);
  }
  if (capability.path == ReferencePath::kBackendDescriptor) {
    const auto selection = select_reference(capability);
    const bool leaky_relu =
        specification.mode == FLAGDNN_POINTWISE_RELU_FWD &&
        specification.attributes.flags ==
            FLAGDNN_POINTWISE_ATTRIBUTE_RELU_LOWER_CLIP_SLOPE;
    if (!leaky_relu) {
      const AcdnnBackendUnaryOperation operation =
          backend_unary_operation(specification.mode);
      const bool binary = specification.mode == FLAGDNN_POINTWISE_ADD ||
                          specification.mode == FLAGDNN_POINTWISE_SUB ||
                          specification.mode == FLAGDNN_POINTWISE_MUL ||
                          specification.mode == FLAGDNN_POINTWISE_MIN ||
                          specification.mode == FLAGDNN_POINTWISE_MAX ||
                          specification.mode == FLAGDNN_POINTWISE_DIV ||
                          specification.mode == FLAGDNN_POINTWISE_POW ||
                          specification.mode == FLAGDNN_POINTWISE_MOD ||
                          specification.mode ==
                              FLAGDNN_POINTWISE_SIGMOID_BWD ||
                          specification.mode == FLAGDNN_POINTWISE_CMP_EQ ||
                          specification.mode == FLAGDNN_POINTWISE_CMP_NEQ ||
                          specification.mode == FLAGDNN_POINTWISE_CMP_GT ||
                          specification.mode == FLAGDNN_POINTWISE_CMP_GE ||
                          specification.mode == FLAGDNN_POINTWISE_CMP_LT ||
                          specification.mode == FLAGDNN_POINTWISE_CMP_LE;
      const bool scaled_binary =
          specification.mode == FLAGDNN_POINTWISE_ADD ||
          specification.mode == FLAGDNN_POINTWISE_SUB;
      if (!std::holds_alternative<ReferencePlan>(selection) ||
          specification.inputs.size() != (binary ? 2U : 1U) ||
          (specification.alpha != 1.0 && !scaled_binary)) {
        throw std::invalid_argument(
            "qualified THead acDNN backend " + operation.name +
            " contract mismatch");
      }
      const ReferencePlan &plan = std::get<ReferencePlan>(selection);
      if (plan.path != ReferencePath::kBackendDescriptor ||
          plan.primitives !=
              std::vector<std::string>{operation.primitive}) {
        throw std::invalid_argument(
            "THead acDNN backend " + operation.name +
            " reference plan mismatch");
      }
      require_activation_attributes(specification.attributes,
                                    operation.name,
                                    operation.allowed_attribute_flags, 0.0,
                                    operation.expected_softplus_beta,
                                    operation.expected_swish_beta);
      return make_acdnn_backend_pointwise_reference(
          {.mode = operation.operation,
           .inputs = specification.inputs,
           .output = specification.output,
           .primitive = operation.primitive,
           .relu_lower_clip = specification.attributes.relu_lower_clip,
           .relu_upper_clip = specification.attributes.relu_upper_clip,
           .relu_lower_clip_slope =
               specification.attributes.relu_lower_clip_slope,
           .elu_alpha = specification.attributes.elu_alpha,
           .softplus_beta = specification.attributes.softplus_beta,
           .swish_beta = specification.attributes.swish_beta,
           .alpha2 = static_cast<float>(specification.alpha),
           .constant_one_numerator =
               specification.mode == FLAGDNN_POINTWISE_RECIPROCAL});
    }
    if (!std::holds_alternative<ReferencePlan>(selection) ||
        specification.mode != FLAGDNN_POINTWISE_RELU_FWD ||
        specification.inputs.size() != 1 ||
        specification.alpha != 1.0 ||
        specification.attributes.flags !=
            FLAGDNN_POINTWISE_ATTRIBUTE_RELU_LOWER_CLIP_SLOPE ||
        specification.attributes.relu_lower_clip != 0.0 ||
        specification.attributes.relu_upper_clip != 0.0 ||
        specification.attributes.relu_lower_clip_slope != 0.2 ||
        specification.attributes.swish_beta != 1.0 ||
        specification.attributes.elu_alpha != 1.0 ||
        specification.attributes.softplus_beta != 1.0) {
      throw std::invalid_argument(
          "qualified THead acDNN backend LeakyReLU contract mismatch");
    }
    const ReferencePlan &plan = std::get<ReferencePlan>(selection);
    const std::string primitive =
        "acdnnBackendExecute(POINTWISE_RELU_FWD,negative_slope=0.2)";
    if (plan.path != ReferencePath::kBackendDescriptor ||
        plan.primitives != std::vector<std::string>{primitive}) {
      throw std::invalid_argument(
          "THead acDNN backend LeakyReLU reference plan mismatch");
    }
    return make_acdnn_backend_pointwise_reference(
        {.mode = ACDNN_POINTWISE_RELU_FWD,
         .inputs = specification.inputs,
         .output = specification.output,
         .primitive = primitive,
         .relu_lower_clip = 0.0,
         .relu_upper_clip = 0.0,
         .relu_lower_clip_slope = 0.2});
  }
  if (specification.mode == FLAGDNN_POINTWISE_RELU_FWD ||
      specification.mode == FLAGDNN_POINTWISE_SIGMOID_FWD ||
      specification.mode == FLAGDNN_POINTWISE_TANH_FWD ||
      specification.mode == FLAGDNN_POINTWISE_ELU_FWD ||
      specification.mode == FLAGDNN_POINTWISE_IDENTITY ||
      specification.mode == FLAGDNN_POINTWISE_GELU_FWD ||
      specification.mode == FLAGDNN_POINTWISE_NEG) {
    return std::make_unique<AcdnnPointwiseActivation>(specification,
                                                      capability);
  }
  return std::make_unique<AcdnnPointwiseBinary>(specification, capability);
}

}  // namespace flagdnn::validation::thead
