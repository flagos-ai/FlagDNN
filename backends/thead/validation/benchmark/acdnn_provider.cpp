// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "benchmark/acdnn_provider.hpp"

#include "acdnn_convolution_reference.hpp"
#include "acdnn_layout_reference.hpp"
#include "acdnn_matmul_reference.hpp"
#include "acdnn_normalization_reference.hpp"
#include "acdnn_reduction_reference.hpp"
#include "common/convolution.hpp"
#include "common/composite.hpp"
#include "common/layout.hpp"
#include "common/matmul.hpp"
#include "common/normalization.hpp"
#include "common/reduction.hpp"
#include "runtime/json.hpp"

#include <fstream>
#include <iterator>
#include <limits>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

namespace flagdnn::validation::thead::benchmark {
namespace {

using JsonObject = flagdnn::native::json::Value::Object;

void require_keys(const JsonObject &object,
                  std::initializer_list<std::string_view> expected,
                  std::string_view context) {
  for (const std::string_view key : expected) {
    if (!object.contains(key)) {
      throw std::runtime_error(std::string(context) +
                               " is missing required key " +
                               std::string(key));
    }
  }
  for (const auto &[key, value] : object) {
    (void)value;
    bool known = false;
    for (const std::string_view candidate : expected) {
      known = known || key == candidate;
    }
    if (!known) {
      throw std::runtime_error(std::string(context) + " has unknown key " +
                               key);
    }
  }
}

ComparableStatus parse_status(std::string_view value) {
  if (value == "comparable") {
    return ComparableStatus::kComparable;
  }
  if (value == "unsupported") {
    return ComparableStatus::kUnsupported;
  }
  if (value == "probe_required") {
    return ComparableStatus::kProbeRequired;
  }
  throw std::runtime_error("unknown THead benchmark capability status");
}

flagdnnPointwiseMode_t pointwise_mode(std::string_view operation) {
  if (operation == "erf") return FLAGDNN_POINTWISE_ERF;
  if (operation == "binary_select") return FLAGDNN_POINTWISE_BINARY_SELECT;
  if (operation == "logical_not") return FLAGDNN_POINTWISE_LOGICAL_NOT;
  if (operation == "logical_and") return FLAGDNN_POINTWISE_LOGICAL_AND;
  if (operation == "logical_or") return FLAGDNN_POINTWISE_LOGICAL_OR;
  if (operation == "sub") return FLAGDNN_POINTWISE_SUB;
  if (operation == "mul" || operation == "scale") {
    return FLAGDNN_POINTWISE_MUL;
  }
  if (operation == "min") return FLAGDNN_POINTWISE_MIN;
  if (operation == "max") return FLAGDNN_POINTWISE_MAX;
  if (operation == "leaky_relu") return FLAGDNN_POINTWISE_RELU_FWD;
  if (operation == "sigmoid") return FLAGDNN_POINTWISE_SIGMOID_FWD;
  if (operation == "tanh") return FLAGDNN_POINTWISE_TANH_FWD;
  if (operation == "elu") return FLAGDNN_POINTWISE_ELU_FWD;
  if (operation == "identity") return FLAGDNN_POINTWISE_IDENTITY;
  if (operation == "gelu") return FLAGDNN_POINTWISE_GELU_FWD;
  if (operation == "sqrt") return FLAGDNN_POINTWISE_SQRT;
  if (operation == "neg") return FLAGDNN_POINTWISE_NEG;
  if (operation == "abs") return FLAGDNN_POINTWISE_ABS;
  if (operation == "ceil") return FLAGDNN_POINTWISE_CEIL;
  if (operation == "floor") return FLAGDNN_POINTWISE_FLOOR;
  if (operation == "exp") return FLAGDNN_POINTWISE_EXP;
  if (operation == "log") return FLAGDNN_POINTWISE_LOG;
  if (operation == "cos") return FLAGDNN_POINTWISE_COS;
  if (operation == "rsqrt") return FLAGDNN_POINTWISE_RSQRT;
  if (operation == "sin") return FLAGDNN_POINTWISE_SIN;
  if (operation == "tan") return FLAGDNN_POINTWISE_TAN;
  if (operation == "softplus") return FLAGDNN_POINTWISE_SOFTPLUS_FWD;
  if (operation == "swish") return FLAGDNN_POINTWISE_SWISH_FWD;
  if (operation == "gelu_approx_tanh") {
    return FLAGDNN_POINTWISE_GELU_APPROX_TANH_FWD;
  }
  if (operation == "div") return FLAGDNN_POINTWISE_DIV;
  if (operation == "pow") return FLAGDNN_POINTWISE_POW;
  if (operation == "mod") return FLAGDNN_POINTWISE_MOD;
  if (operation == "sigmoid_backward") {
    return FLAGDNN_POINTWISE_SIGMOID_BWD;
  }
  if (operation == "reciprocal") return FLAGDNN_POINTWISE_RECIPROCAL;
  if (operation == "cmp_eq") return FLAGDNN_POINTWISE_CMP_EQ;
  if (operation == "cmp_neq") return FLAGDNN_POINTWISE_CMP_NEQ;
  if (operation == "cmp_gt") return FLAGDNN_POINTWISE_CMP_GT;
  if (operation == "cmp_ge") return FLAGDNN_POINTWISE_CMP_GE;
  if (operation == "cmp_lt") return FLAGDNN_POINTWISE_CMP_LT;
  if (operation == "cmp_le") return FLAGDNN_POINTWISE_CMP_LE;
  return FLAGDNN_POINTWISE_NOT_SET;
}

flagdnn::testing::LayoutOperation layout_operation(
    std::string_view operation) {
  if (operation == "reshape") {
    return flagdnn::testing::LayoutOperation::kReshape;
  }
  if (operation == "transpose") {
    return flagdnn::testing::LayoutOperation::kTranspose;
  }
  if (operation == "slice") {
    return flagdnn::testing::LayoutOperation::kSlice;
  }
  throw std::invalid_argument("unknown THead benchmark Layout operation");
}

bool is_layout_operation(std::string_view operation) {
  return operation == "reshape" || operation == "transpose" ||
         operation == "slice";
}

bool is_convolution_operation(std::string_view operation) {
  return operation == "conv_fprop" || operation == "conv_dgrad" ||
         operation == "conv_wgrad";
}

bool owns_case(std::string_view operation, std::string_view case_name) {
  if (!is_convolution_operation(operation)) {
    return case_name.starts_with(std::string(operation) + "_");
  }
  const std::string direction(operation.substr(5));
  return case_name.starts_with("conv1d_" + direction + "_") ||
         case_name.starts_with("conv2d_" + direction + "_") ||
         case_name.starts_with("conv3d_" + direction + "_");
}

flagdnn::testing::TestTensor test_tensor(
    const flagdnn::benchmarking::TensorSpec &tensor) {
  return {tensor.uid, tensor.data_type, tensor.dimensions, tensor.strides,
          tensor.binding_byte_offset};
}

flagdnn::testing::ReductionTestCase reduction_case(
    const flagdnn::benchmarking::BenchmarkCase &specification) {
  if (specification.operation !=
          flagdnn::benchmarking::Operation::kReduction ||
      specification.tensors.size() != 2 || specification.output_count != 1) {
    throw std::invalid_argument(
        "THead Reduction benchmark requires input/output tensors");
  }
  flagdnn::testing::ReductionTestCase result;
  result.name = specification.name;
  result.input = test_tensor(specification.tensors[0]);
  result.output = test_tensor(specification.tensors[1]);
  result.mode = specification.reduction_mode;
  result.axis = specification.reduction_axis;
  result.keep_dimensions = specification.keep_dimensions;
  result.absolute_tolerance = specification.absolute_tolerance;
  result.relative_tolerance = specification.relative_tolerance;
  return result;
}

flagdnn::testing::BatchnormTestCase batchnorm_case(
    const flagdnn::benchmarking::BenchmarkCase &specification) {
  if (specification.operation !=
          flagdnn::benchmarking::Operation::kBatchnorm ||
      specification.tensors.size() != 10 || specification.output_count != 5) {
    throw std::invalid_argument(
        "THead BatchNorm benchmark requires five inputs and five outputs");
  }
  flagdnn::testing::BatchnormTestCase result;
  result.name = specification.name;
  result.x = test_tensor(specification.tensors[0]);
  result.scale = test_tensor(specification.tensors[1]);
  result.bias = test_tensor(specification.tensors[2]);
  result.previous_running_mean = test_tensor(specification.tensors[3]);
  result.previous_running_variance = test_tensor(specification.tensors[4]);
  result.y = test_tensor(specification.tensors[5]);
  result.mean = test_tensor(specification.tensors[6]);
  result.inv_variance = test_tensor(specification.tensors[7]);
  result.next_running_mean = test_tensor(specification.tensors[8]);
  result.next_running_variance = test_tensor(specification.tensors[9]);
  result.epsilon = specification.normalization.epsilon;
  result.momentum = specification.normalization.momentum;
  result.absolute_tolerance = specification.absolute_tolerance;
  result.relative_tolerance = specification.relative_tolerance;
  return result;
}

flagdnn::testing::BatchnormInferenceTestCase batchnorm_inference_case(
    const flagdnn::benchmarking::BenchmarkCase &specification) {
  if (specification.operation !=
          flagdnn::benchmarking::Operation::kBatchnormInference ||
      specification.tensors.size() != 6 || specification.output_count != 1) {
    throw std::invalid_argument(
        "THead BatchNorm-inference benchmark requires five inputs and one output");
  }
  flagdnn::testing::BatchnormInferenceTestCase result;
  result.name = specification.name;
  result.x = test_tensor(specification.tensors[0]);
  result.mean = test_tensor(specification.tensors[1]);
  result.inv_variance = test_tensor(specification.tensors[2]);
  result.scale = test_tensor(specification.tensors[3]);
  result.bias = test_tensor(specification.tensors[4]);
  result.y = test_tensor(specification.tensors[5]);
  result.absolute_tolerance = specification.absolute_tolerance;
  result.relative_tolerance = specification.relative_tolerance;
  return result;
}

flagdnn::testing::LayernormTestCase layernorm_case(
    const flagdnn::benchmarking::BenchmarkCase &specification) {
  if (specification.operation !=
          flagdnn::benchmarking::Operation::kLayernorm ||
      specification.tensors.size() != 6 || specification.output_count != 3) {
    throw std::invalid_argument(
        "THead LayerNorm benchmark requires three inputs and three outputs");
  }
  flagdnn::testing::LayernormTestCase result;
  result.name = specification.name;
  result.x = test_tensor(specification.tensors[0]);
  result.scale = test_tensor(specification.tensors[1]);
  result.bias = test_tensor(specification.tensors[2]);
  result.y = test_tensor(specification.tensors[3]);
  result.mean = test_tensor(specification.tensors[4]);
  result.inv_variance = test_tensor(specification.tensors[5]);
  result.epsilon = specification.normalization.epsilon;
  result.absolute_tolerance = specification.absolute_tolerance;
  result.relative_tolerance = specification.relative_tolerance;
  return result;
}

flagdnn::testing::RmsnormTestCase rmsnorm_case(
    const flagdnn::benchmarking::BenchmarkCase &specification) {
  if (specification.operation !=
          flagdnn::benchmarking::Operation::kRmsnorm ||
      specification.tensors.size() != 5 || specification.output_count != 2) {
    throw std::invalid_argument(
        "THead RMSNorm benchmark requires three inputs and two outputs");
  }
  flagdnn::testing::RmsnormTestCase result;
  result.name = specification.name;
  result.x = test_tensor(specification.tensors[0]);
  result.scale = test_tensor(specification.tensors[1]);
  result.bias = test_tensor(specification.tensors[2]);
  result.y = test_tensor(specification.tensors[3]);
  result.inv_variance = test_tensor(specification.tensors[4]);
  result.epsilon = specification.normalization.epsilon;
  result.absolute_tolerance = specification.absolute_tolerance;
  result.relative_tolerance = specification.relative_tolerance;
  return result;
}

CapabilityRecord normalization_capability(
    std::string_view operation, ComparableStatus qualification,
    flagdnnDataType_t data_type) {
  CapabilityRecord result;
  result.status = qualification == ComparableStatus::kProbeRequired
                      ? CapabilityStatus::kProbeRequired
                      : CapabilityStatus::kSupported;
  result.path = ReferencePath::kBackendDescriptor;
  if (operation == "layernorm") {
    result.reference_plan = {
        "acdnnReduceTensor(AVG,mean)",
        "acdnnOpTensor(ADD[-mean])",
        "acdnnOpTensor(MUL,square)",
        "acdnnReduceTensor(AVG,variance)",
        "acdnnOpTensor(ADD,epsilon)",
        "acdnnBackendExecute(POINTWISE_RSQRT)",
        "acdnnOpTensor(MUL,normalize)",
        "acdnnOpTensor(MUL,scale)",
        "acdnnOpTensor(ADD,bias)",
    };
  } else if (operation == "rmsnorm") {
    result.reference_plan = {
        "acdnnOpTensor(MUL,square)",
        "acdnnReduceTensor(AVG,mean_square)",
        "acdnnOpTensor(ADD,epsilon)",
        "acdnnBackendExecute(POINTWISE_RSQRT)",
        "acdnnOpTensor(MUL,normalize)",
        "acdnnOpTensor(MUL,scale)",
        "acdnnOpTensor(ADD,bias)",
    };
  } else {
    throw std::invalid_argument("unknown THead normalization benchmark");
  }
  if (data_type != FLAGDNN_DATA_FLOAT32) {
    result.reference_plan.insert(
        result.reference_plan.begin(),
        {"acdnnBackendExecute(POINTWISE_IDENTITY_FWD,convert-x-fp32)",
         "acdnnBackendExecute(POINTWISE_IDENTITY_FWD,convert-scale-fp32)",
         "acdnnBackendExecute(POINTWISE_IDENTITY_FWD,convert-bias-fp32)"});
    result.reference_plan.push_back(
        "acdnnBackendExecute(POINTWISE_IDENTITY_FWD,convert-y-data-type)");
  }
  result.constraints = CapabilityConstraints{};
  return result;
}

CapabilityRecord batchnorm_capability(bool inference,
                                      ComparableStatus qualification,
                                      flagdnnDataType_t data_type) {
  CapabilityRecord result;
  result.status = qualification == ComparableStatus::kProbeRequired
                      ? CapabilityStatus::kProbeRequired
                      : CapabilityStatus::kSupported;
  result.path = ReferencePath::kStablePrimitive;
  result.reference_plan = inference
      ? std::vector<std::string>{
            "acdnnOpTensor(ADD[-mean])->MUL(inv_variance)->MUL(scale)->ADD(bias)"}
      : std::vector<std::string>{
            "acdnnTransformTensor(previous_running_mean)",
            "acdnnTransformTensor(previous_running_variance)",
            "acdnnBatchNormalizationForwardTraining(SPATIAL)"};
  if (data_type != FLAGDNN_DATA_FLOAT32) {
    result.path = ReferencePath::kBackendDescriptor;
    if (inference) {
      result.reference_plan.insert(
          result.reference_plan.begin(),
          "acdnnBackendExecute(POINTWISE_IDENTITY_FWD,convert-x-fp32)");
    } else {
      result.reference_plan.insert(
          result.reference_plan.begin(),
          {"acdnnBackendExecute(POINTWISE_IDENTITY_FWD,convert-x-fp32)",
           "acdnnBackendExecute(POINTWISE_IDENTITY_FWD,convert-scale-fp32)",
           "acdnnBackendExecute(POINTWISE_IDENTITY_FWD,convert-bias-fp32)"});
    }
    result.reference_plan.push_back(
        "acdnnBackendExecute(POINTWISE_IDENTITY_FWD,convert-y-data-type)");
  }
  result.constraints = CapabilityConstraints{};
  return result;
}

CapabilityRecord reduction_capability(
    const flagdnn::testing::ReductionTestCase &test_case,
    ComparableStatus qualification) {
  CapabilityRecord result;
  result.status = qualification == ComparableStatus::kProbeRequired
                      ? CapabilityStatus::kProbeRequired
                      : CapabilityStatus::kSupported;
  result.path = ReferencePath::kStablePrimitive;
  if (test_case.mode == FLAGDNN_REDUCTION_ADD) {
    result.reference_plan = {"acdnnReduceTensor(ADD,alpha=1,beta=0)"};
  } else if (test_case.mode == FLAGDNN_REDUCTION_AVG) {
    result.reference_plan = {"acdnnReduceTensor(AVG,alpha=1,beta=0)"};
  } else if (test_case.mode == FLAGDNN_REDUCTION_MUL) {
    result.reference_plan = {"acdnnReduceTensor(MUL,alpha=1,beta=0)"};
  } else {
    throw std::invalid_argument("unknown THead benchmark Reduction mode");
  }
  std::int64_t dense_stride = 1;
  bool dense_input =
      test_case.input.dimensions.size() == test_case.input.strides.size();
  for (std::size_t axis = test_case.input.dimensions.size();
       dense_input && axis != 0; --axis) {
    const std::int64_t dimension = test_case.input.dimensions[axis - 1];
    dense_input = dimension > 0 &&
                  test_case.input.strides[axis - 1] == dense_stride &&
                  dense_stride <=
                      std::numeric_limits<std::int64_t>::max() / dimension;
    if (dense_input) {
      dense_stride *= dimension;
    }
  }
  if (test_case.input.data_type == FLAGDNN_DATA_BFLOAT16) {
    result.path = ReferencePath::kBackendDescriptor;
    result.reference_plan.insert(
        result.reference_plan.begin(),
        "acdnnBackendExecute(POINTWISE_IDENTITY_FWD,convert-pack-fp32)");
    result.reference_plan.push_back(
        "acdnnBackendExecute(POINTWISE_IDENTITY_FWD,convert-bfloat16)");
  } else if (!dense_input) {
    result.path = ReferencePath::kBackendDescriptor;
    result.reference_plan.insert(
        result.reference_plan.begin(),
        "acdnnBackendExecute(POINTWISE_IDENTITY_FWD,pack-segments)");
  }
  result.constraints = CapabilityConstraints{};
  return result;
}

flagdnn::testing::MatmulTestCase matmul_case(
    const flagdnn::benchmarking::BenchmarkCase &specification) {
  if (specification.operation !=
          flagdnn::benchmarking::Operation::kMatmul ||
      specification.tensors.size() != 3 || specification.output_count != 1) {
    throw std::invalid_argument(
        "THead MatMul benchmark requires two inputs and one output");
  }
  flagdnn::testing::MatmulTestCase result;
  result.name = specification.name;
  result.a = test_tensor(specification.tensors[0]);
  result.b = test_tensor(specification.tensors[1]);
  result.output = test_tensor(specification.tensors[2]);
  result.absolute_tolerance = specification.absolute_tolerance;
  result.relative_tolerance = specification.relative_tolerance;
  return result;
}

CapabilityRecord matmul_capability(ComparableStatus qualification) {
  CapabilityRecord result;
  result.status = qualification == ComparableStatus::kProbeRequired
                      ? CapabilityStatus::kProbeRequired
                      : CapabilityStatus::kSupported;
  result.path = ReferencePath::kBackendDescriptor;
  result.reference_plan = {"acdnnBackendExecute(MATMUL)"};
  result.constraints = CapabilityConstraints{};
  return result;
}

flagdnn::testing::ConvolutionDirection convolution_direction(
    std::string_view operation) {
  if (operation == "conv_fprop") {
    return flagdnn::testing::ConvolutionDirection::kFprop;
  }
  if (operation == "conv_dgrad") {
    return flagdnn::testing::ConvolutionDirection::kDgrad;
  }
  if (operation == "conv_wgrad") {
    return flagdnn::testing::ConvolutionDirection::kWgrad;
  }
  throw std::invalid_argument("unknown THead benchmark convolution operation");
}

flagdnn::testing::ConvolutionTestCase convolution_case(
    const flagdnn::benchmarking::BenchmarkCase &specification,
    std::string_view operation) {
  const flagdnn::testing::ConvolutionDirection direction =
      convolution_direction(operation);
  const bool operation_matches =
      (direction == flagdnn::testing::ConvolutionDirection::kFprop &&
       specification.operation ==
           flagdnn::benchmarking::Operation::kConvolutionFprop) ||
      (direction == flagdnn::testing::ConvolutionDirection::kDgrad &&
       specification.operation ==
           flagdnn::benchmarking::Operation::kConvolutionDgrad) ||
      (direction == flagdnn::testing::ConvolutionDirection::kWgrad &&
       specification.operation ==
           flagdnn::benchmarking::Operation::kConvolutionWgrad);
  if (!operation_matches || specification.tensors.size() != 3 ||
      specification.output_count != 1) {
    throw std::invalid_argument(
        "THead convolution benchmark requires two inputs and one output");
  }

  flagdnn::testing::ConvolutionTestCase result;
  result.name = specification.name;
  result.direction = direction;
  if (direction == flagdnn::testing::ConvolutionDirection::kFprop) {
    result.x = test_tensor(specification.tensors[0]);
    result.w = test_tensor(specification.tensors[1]);
    result.y = test_tensor(specification.tensors[2]);
  } else if (direction ==
             flagdnn::testing::ConvolutionDirection::kDgrad) {
    result.y = test_tensor(specification.tensors[0]);
    result.w = test_tensor(specification.tensors[1]);
    result.x = test_tensor(specification.tensors[2]);
  } else {
    result.y = test_tensor(specification.tensors[0]);
    result.x = test_tensor(specification.tensors[1]);
    result.w = test_tensor(specification.tensors[2]);
  }
  result.pre_padding = specification.convolution.pre_padding;
  result.post_padding = specification.convolution.post_padding;
  result.stride = specification.convolution.stride;
  result.dilation = specification.convolution.dilation;
  result.groups = specification.convolution.groups;
  result.mode =
      specification.convolution.mode ==
              flagdnn::benchmarking::ConvolutionMode::kCrossCorrelation
          ? flagdnn::testing::ConvolutionMode::kCrossCorrelation
          : flagdnn::testing::ConvolutionMode::kConvolution;
  result.absolute_tolerance = specification.absolute_tolerance;
  result.relative_tolerance = specification.relative_tolerance;
  return result;
}

CapabilityRecord convolution_capability(
    const flagdnn::testing::ConvolutionTestCase &test_case,
    ComparableStatus qualification) {
  CapabilityRecord result;
  result.status = qualification == ComparableStatus::kProbeRequired
                      ? CapabilityStatus::kProbeRequired
                      : CapabilityStatus::kSupported;
  result.path = ReferencePath::kStablePrimitive;
  if (test_case.pre_padding != test_case.post_padding) {
    const bool backend_copy =
        test_case.x.data_type == FLAGDNN_DATA_BFLOAT16;
    if (test_case.direction ==
        flagdnn::testing::ConvolutionDirection::kFprop) {
      result.reference_plan = {
          "acdnnConvolutionForward(IMPLICIT_GEMM,symmetric-superset)",
          backend_copy
              ? "acdnnBackendExecute(POINTWISE_IDENTITY_FWD,asymmetric-output-slice)"
              : "acdnnTransformTensor(alpha=1,beta=0,asymmetric-output-slice)"};
    } else {
      result.reference_plan = {
          backend_copy
              ? "acdnnBackendExecute(POINTWISE_IDENTITY_FWD,asymmetric-gradient-zero)"
              : "acdnnTransformTensor(alpha=1,beta=0,asymmetric-gradient-zero)",
          backend_copy
              ? "acdnnBackendExecute(POINTWISE_IDENTITY_FWD,asymmetric-gradient-pad)"
              : "acdnnTransformTensor(alpha=1,beta=0,asymmetric-gradient-pad)",
          test_case.direction ==
                  flagdnn::testing::ConvolutionDirection::kDgrad
              ? "acdnnConvolutionBackwardData(ALGO_0,symmetric-superset)"
              : "acdnnConvolutionBackwardFilter(ALGO_0,symmetric-superset)"};
    }
    result.constraints = CapabilityConstraints{};
    return result;
  }
  switch (test_case.direction) {
    case flagdnn::testing::ConvolutionDirection::kFprop:
      result.reference_plan = {"acdnnConvolutionForward(IMPLICIT_GEMM)"};
      break;
    case flagdnn::testing::ConvolutionDirection::kDgrad:
      result.reference_plan = {"acdnnConvolutionBackwardData(ALGO_0)"};
      break;
    case flagdnn::testing::ConvolutionDirection::kWgrad:
      result.reference_plan = {"acdnnConvolutionBackwardFilter(ALGO_0)"};
      break;
  }
  result.constraints = CapabilityConstraints{};
  return result;
}

flagdnn::testing::ConvBiasReluTestCase conv_bias_relu_case(
    const flagdnn::benchmarking::BenchmarkCase &specification) {
  if (specification.operation != flagdnn::benchmarking::Operation::kGraph ||
      specification.tensors.size() != 4 || specification.output_count != 1 ||
      specification.graph.nodes.size() != 3 ||
      specification.graph.nodes[0].operation !=
          flagdnn::benchmarking::Operation::kConvolutionFprop ||
      specification.graph.nodes[1].pointwise_mode != FLAGDNN_POINTWISE_ADD ||
      specification.graph.nodes[2].pointwise_mode !=
          FLAGDNN_POINTWISE_RELU_FWD) {
    throw std::invalid_argument(
        "THead ConvBiasRelu benchmark requires a canonical fused graph");
  }
  const auto &convolution = specification.graph.nodes[0].convolution;
  if (convolution.pre_padding != convolution.post_padding ||
      convolution.groups != 1 ||
      convolution.mode !=
          flagdnn::benchmarking::ConvolutionMode::kCrossCorrelation) {
    throw std::invalid_argument(
        "THead ConvBiasRelu benchmark convolution attributes are invalid");
  }
  flagdnn::testing::ConvBiasReluTestCase result;
  result.name = specification.name;
  result.x = test_tensor(specification.tensors[0]);
  result.w = test_tensor(specification.tensors[1]);
  result.bias = test_tensor(specification.tensors[2]);
  result.output = test_tensor(specification.tensors[3]);
  result.padding = convolution.pre_padding;
  result.stride = convolution.stride;
  result.dilation = convolution.dilation;
  result.absolute_tolerance = specification.absolute_tolerance;
  result.relative_tolerance = specification.relative_tolerance;
  return result;
}

CapabilityRecord conv_bias_relu_capability(ComparableStatus qualification) {
  CapabilityRecord result;
  result.status = qualification == ComparableStatus::kProbeRequired
                      ? CapabilityStatus::kProbeRequired
                      : CapabilityStatus::kSupported;
  result.path = ReferencePath::kStablePrimitive;
  result.reference_plan = {
      "acdnnConvolutionBiasActivationForward(RELU)"};
  result.constraints = CapabilityConstraints{};
  return result;
}

flagdnn::testing::LayoutTestCase layout_case(
    const flagdnn::benchmarking::BenchmarkCase &specification,
    std::string_view operation) {
  if (specification.tensors.size() != 2 ||
      specification.output_count != 1) {
    throw std::invalid_argument("THead Layout benchmark requires two tensors");
  }
  flagdnn::testing::LayoutTestCase result;
  result.name = specification.name;
  result.operation = layout_operation(operation);
  result.input = test_tensor(specification.tensors[0]);
  result.output = test_tensor(specification.tensors[1]);
  result.permutation = specification.transpose.permutation;
  result.slices = specification.slice.slices;
  result.slice_strides = specification.slice.strides;
  return result;
}

CapabilityRecord layout_capability(std::string_view operation,
                                   ComparableStatus qualification,
                                   const flagdnn::testing::LayoutTestCase
                                       &test_case) {
  CapabilityRecord result;
  result.status = qualification == ComparableStatus::kProbeRequired
                      ? CapabilityStatus::kProbeRequired
                      : CapabilityStatus::kSupported;
  const bool transpose = operation == "transpose";
  if (transpose && !legacy_acdnn_transpose_descriptor_compatible(
                       test_case.input.dimensions,
                       test_case.permutation)) {
    throw std::invalid_argument(
        "THead Transpose benchmark has no certified acDNN descriptor map");
  }
  const bool converted_bfloat16_transpose =
      test_case.input.data_type == FLAGDNN_DATA_BFLOAT16 &&
      transpose;
  const bool backend = test_case.input.data_type != FLAGDNN_DATA_FLOAT32 &&
                       !transpose;
  result.path = (backend || converted_bfloat16_transpose)
                    ? ReferencePath::kBackendDescriptor
                    : ReferencePath::kStablePrimitive;
  if (converted_bfloat16_transpose) {
    result.reference_plan = {
        "acdnnBackendExecute(POINTWISE_IDENTITY_FWD,convert-input0-fp32)",
        "acdnnTransformTensor(permuted-stride,alpha=1,beta=0)",
        "acdnnBackendExecute(POINTWISE_IDENTITY_FWD,convert-output-bfloat16)"};
    result.constraints = CapabilityConstraints{};
    return result;
  }
  if (backend) {
    result.reference_plan = {
        "acdnnBackendExecute(POINTWISE_IDENTITY_FWD,layout-map)"};
    result.constraints = CapabilityConstraints{};
    return result;
  }
  if (operation == "reshape") {
    result.reference_plan = {
        "acdnnTransformTensor(flattened,alpha=1,beta=0)"};
  } else if (operation == "transpose") {
    result.reference_plan = {
        "acdnnTransformTensor(permuted-stride,alpha=1,beta=0)"};
  } else {
    result.reference_plan = {
        "acdnnTransformTensor(slice-segments,alpha=1,beta=0)"};
  }
  result.constraints = CapabilityConstraints{};
  return result;
}

class LayoutBenchmarkExecutable final
    : public flagdnn::benchmarking::BenchmarkExecutable {
 public:
  explicit LayoutBenchmarkExecutable(
      std::unique_ptr<flagdnn::testing::LayoutExecutable> delegate)
      : delegate_(std::move(delegate)) {
    if (delegate_ == nullptr) {
      throw std::invalid_argument("THead Layout benchmark delegate is null");
    }
  }

  void prepare(std::span<const flagdnnBinding_t> bindings,
               flagdnnStream_t stream) override {
    delegate_->prepare(bindings, stream);
  }

  [[nodiscard]] std::size_t workspace_size() const noexcept override {
    return delegate_->workspace_size();
  }

  void execute(std::span<const flagdnnBinding_t> bindings, void *workspace,
               std::size_t workspace_size,
               flagdnnStream_t stream) override {
    delegate_->execute(bindings, workspace, workspace_size, stream);
  }

 private:
  std::unique_ptr<flagdnn::testing::LayoutExecutable> delegate_;
};

class ReductionBenchmarkExecutable final
    : public flagdnn::benchmarking::BenchmarkExecutable {
 public:
  explicit ReductionBenchmarkExecutable(
      std::unique_ptr<flagdnn::testing::ReductionExecutable> delegate)
      : delegate_(std::move(delegate)) {
    if (delegate_ == nullptr) {
      throw std::invalid_argument("THead Reduction benchmark delegate is null");
    }
  }

  void prepare(std::span<const flagdnnBinding_t> bindings,
               flagdnnStream_t stream) override {
    delegate_->prepare(bindings, stream);
  }

  [[nodiscard]] std::size_t workspace_size() const noexcept override {
    return delegate_->workspace_size();
  }

  void execute(std::span<const flagdnnBinding_t> bindings, void *workspace,
               std::size_t workspace_size,
               flagdnnStream_t stream) override {
    delegate_->execute(bindings, workspace, workspace_size, stream);
  }

 private:
  std::unique_ptr<flagdnn::testing::ReductionExecutable> delegate_;
};

class NormalizationBenchmarkExecutable final
    : public flagdnn::benchmarking::BenchmarkExecutable {
 public:
  explicit NormalizationBenchmarkExecutable(
      std::unique_ptr<flagdnn::testing::NormalizationExecutable> delegate)
      : delegate_(std::move(delegate)) {
    if (delegate_ == nullptr) {
      throw std::invalid_argument(
          "THead normalization benchmark delegate is null");
    }
  }

  void prepare(std::span<const flagdnnBinding_t> bindings,
               flagdnnStream_t stream) override {
    delegate_->prepare(bindings, stream);
  }

  [[nodiscard]] std::size_t workspace_size() const noexcept override {
    return delegate_->workspace_size();
  }

  void execute(std::span<const flagdnnBinding_t> bindings, void *workspace,
               std::size_t workspace_size,
               flagdnnStream_t stream) override {
    delegate_->execute(bindings, workspace, workspace_size, stream);
  }

 private:
  std::unique_ptr<flagdnn::testing::NormalizationExecutable> delegate_;
};

class MatmulBenchmarkExecutable final
    : public flagdnn::benchmarking::BenchmarkExecutable {
 public:
  explicit MatmulBenchmarkExecutable(
      std::unique_ptr<flagdnn::testing::MatmulExecutable> delegate)
      : delegate_(std::move(delegate)) {
    if (delegate_ == nullptr) {
      throw std::invalid_argument("THead MatMul benchmark delegate is null");
    }
  }

  void prepare(std::span<const flagdnnBinding_t> bindings,
               flagdnnStream_t stream) override {
    delegate_->prepare(bindings, stream);
  }

  [[nodiscard]] std::size_t workspace_size() const noexcept override {
    return delegate_->workspace_size();
  }

  void execute(std::span<const flagdnnBinding_t> bindings, void *workspace,
               std::size_t workspace_size,
               flagdnnStream_t stream) override {
    delegate_->execute(bindings, workspace, workspace_size, stream);
  }

 private:
  std::unique_ptr<flagdnn::testing::MatmulExecutable> delegate_;
};

class ConvolutionBenchmarkExecutable final
    : public flagdnn::benchmarking::BenchmarkExecutable {
 public:
  explicit ConvolutionBenchmarkExecutable(
      std::unique_ptr<flagdnn::testing::ConvolutionExecutable> delegate)
      : delegate_(std::move(delegate)) {
    if (delegate_ == nullptr) {
      throw std::invalid_argument(
          "THead convolution benchmark delegate is null");
    }
  }

  void prepare(std::span<const flagdnnBinding_t> bindings,
               flagdnnStream_t stream) override {
    delegate_->prepare(bindings, stream);
  }

  [[nodiscard]] std::size_t workspace_size() const noexcept override {
    return delegate_->workspace_size();
  }

  void execute(std::span<const flagdnnBinding_t> bindings, void *workspace,
               std::size_t workspace_size,
               flagdnnStream_t stream) override {
    delegate_->execute(bindings, workspace, workspace_size, stream);
  }

 private:
  std::unique_ptr<flagdnn::testing::ConvolutionExecutable> delegate_;
};

}  // namespace

AcdnnProvider::AcdnnProvider(const std::string &catalog_path,
                             std::string operation, bool qualify_probes)
    : operation_(std::move(operation)), qualify_probes_(qualify_probes) {
  if (operation_ != "add" && operation_ != "sub" && operation_ != "mul" &&
      operation_ != "min" && operation_ != "max" && operation_ != "scale" &&
      operation_ != "relu" && operation_ != "sigmoid" &&
      operation_ != "tanh" && operation_ != "elu" &&
      operation_ != "identity" && operation_ != "gelu" &&
      operation_ != "sqrt" && operation_ != "neg" &&
      operation_ != "abs" && operation_ != "ceil" &&
      operation_ != "floor" && operation_ != "exp" &&
      operation_ != "add_square" && operation_ != "reduction" &&
      operation_ != "batchnorm" &&
      operation_ != "batchnorm_inference" &&
      operation_ != "layernorm" && operation_ != "rmsnorm" &&
      operation_ != "matmul" &&
      operation_ != "conv_bias_relu" &&
      !is_convolution_operation(operation_) &&
      !is_layout_operation(operation_) &&
      pointwise_mode(operation_) == FLAGDNN_POINTWISE_NOT_SET) {
    throw std::invalid_argument(
        "THead acDNN benchmark provider supports only "
        "a catalog-qualified Add, Relu, or pointwise operation");
  }
  std::ifstream input(catalog_path);
  if (!input) {
    throw std::runtime_error("cannot open THead comparable-case catalog: " +
                             catalog_path);
  }
  const auto root_value = flagdnn::native::json::parse(
      std::string(std::istreambuf_iterator<char>(input),
                  std::istreambuf_iterator<char>()));
  const JsonObject &root = root_value.as_object();
  require_keys(root,
               {"schema_version", "platform", "reference_provider",
                "operators"},
               "comparable catalog root");
  if (root.at("schema_version").as_int() != 2 ||
      root.at("platform").as_string() != "thead" ||
      root.at("reference_provider").as_string() != "acdnn") {
    throw std::runtime_error("THead comparable-case catalog identity mismatch");
  }
  const JsonObject &operators = root.at("operators").as_object();
  require_keys(operators,
               {"add", "sub", "mul", "min", "max", "scale", "relu",
                "leaky_relu", "sigmoid", "tanh", "elu", "identity", "gelu",
                "sqrt",
                "neg", "abs", "ceil", "floor", "exp", "log", "cos",
                "rsqrt", "sin", "tan", "softplus", "swish",
                "gelu_approx_tanh", "div", "pow", "mod",
                "sigmoid_backward", "reciprocal", "add_square",
                "cmp_eq", "cmp_neq", "cmp_gt", "cmp_ge", "cmp_lt",
                "cmp_le", "reshape", "transpose", "slice",
                "reduction", "batchnorm", "batchnorm_inference",
                "layernorm", "rmsnorm",
                "matmul", "conv_fprop", "conv_dgrad", "conv_wgrad",
                "conv_bias_relu", "erf", "binary_select",
                "logical_not", "logical_and", "logical_or"},
               "comparable catalog operators");
  const JsonObject &cases = operators.at(operation_).as_object();
  if (cases.empty()) {
    throw std::runtime_error(
        "THead comparable-case catalog operation has no cases");
  }
  for (const auto &[case_name, value] : cases) {
    if (case_name.empty() || !owns_case(operation_, case_name)) {
      throw std::runtime_error(
          "THead comparable catalog has an invalid case owner");
    }
    const JsonObject &object = value.as_object();
    require_keys(object, {"status", "reason_code", "detail"},
                 "benchmark case " + case_name);
    ComparableRecord record;
    record.status = parse_status(object.at("status").as_string());
    record.reason_code = object.at("reason_code").as_string();
    record.detail = object.at("detail").as_string();
    if (record.detail.empty()) {
      throw std::runtime_error("benchmark capability detail is empty");
    }
    if (record.status == ComparableStatus::kComparable) {
      if (!record.reason_code.empty()) {
        throw std::runtime_error(
            "comparable benchmark case has a skip reason");
      }
    } else if (record.status == ComparableStatus::kProbeRequired) {
      if (record.reason_code != "real_device_qualification_pending") {
        throw std::runtime_error(
            "benchmark probe has no qualification reason");
      }
    } else if (record.reason_code != "dtype_unsupported" &&
               record.reason_code != "acdnn_status_not_supported" &&
               record.reason_code != "acdnn_semantic_mismatch" &&
               record.reason_code != "layout_unsupported" &&
               record.reason_code != "no_certified_acdnn_primitive" &&
               record.reason_code != "production_kernel_unavailable" &&
               record.reason_code != "attribute_unsupported" &&
               record.reason_code != "shape_unsupported") {
      throw std::runtime_error(
          "benchmark case has an unknown unsupported reason");
    }
    if (!records_.emplace(case_name, std::move(record)).second) {
      throw std::runtime_error(
          "THead comparable catalog contains a duplicate case");
    }
  }
}

flagdnn::benchmarking::ProviderCapability AcdnnProvider::capability(
    const flagdnn::benchmarking::BenchmarkCase &specification) const {
  const flagdnnPointwiseMode_t expected_mode = pointwise_mode(operation_);
  const bool operation_matches =
      (operation_ == "add" &&
       specification.operation == flagdnn::benchmarking::Operation::kAdd) ||
      (operation_ == "relu" &&
       specification.operation == flagdnn::benchmarking::Operation::kRelu) ||
      (operation_ == "add_square" &&
       specification.operation ==
           flagdnn::benchmarking::Operation::kGraph) ||
      (operation_ == "conv_bias_relu" &&
       specification.operation == flagdnn::benchmarking::Operation::kGraph &&
       specification.name.starts_with("conv_bias_relu_perf_")) ||
      (operation_ == "reshape" &&
       specification.operation ==
           flagdnn::benchmarking::Operation::kReshape) ||
      (operation_ == "transpose" &&
       specification.operation ==
           flagdnn::benchmarking::Operation::kTranspose) ||
      (operation_ == "slice" &&
       specification.operation ==
           flagdnn::benchmarking::Operation::kSlice) ||
      (operation_ == "reduction" &&
       specification.operation ==
           flagdnn::benchmarking::Operation::kReduction) ||
      (operation_ == "batchnorm" &&
       specification.operation ==
           flagdnn::benchmarking::Operation::kBatchnorm) ||
      (operation_ == "batchnorm_inference" &&
       specification.operation ==
           flagdnn::benchmarking::Operation::kBatchnormInference) ||
      (operation_ == "layernorm" &&
       specification.operation ==
           flagdnn::benchmarking::Operation::kLayernorm) ||
      (operation_ == "rmsnorm" &&
       specification.operation ==
           flagdnn::benchmarking::Operation::kRmsnorm) ||
      (operation_ == "matmul" &&
       specification.operation ==
           flagdnn::benchmarking::Operation::kMatmul) ||
      (operation_ == "conv_fprop" &&
       specification.operation ==
           flagdnn::benchmarking::Operation::kConvolutionFprop) ||
      (operation_ == "conv_dgrad" &&
       specification.operation ==
           flagdnn::benchmarking::Operation::kConvolutionDgrad) ||
      (operation_ == "conv_wgrad" &&
       specification.operation ==
           flagdnn::benchmarking::Operation::kConvolutionWgrad) ||
      (expected_mode != FLAGDNN_POINTWISE_NOT_SET &&
       specification.operation ==
           flagdnn::benchmarking::Operation::kPointwise &&
       specification.pointwise_mode == expected_mode);
  if (!operation_matches) {
    throw std::invalid_argument(
        "THead benchmark specification does not match provider operation");
  }
  const ComparableRecord &record = lookup(specification.name);
  if (record.status == ComparableStatus::kComparable ||
      (record.status == ComparableStatus::kProbeRequired &&
       qualify_probes_)) {
    return {};
  }
  return flagdnn::benchmarking::ProviderCapability::unsupported(
      record.reason_code);
}

std::unique_ptr<flagdnn::benchmarking::BenchmarkExecutable>
AcdnnProvider::build(
    const flagdnn::benchmarking::BenchmarkCase &specification) {
  const ComparableRecord &record = lookup(specification.name);
  if (record.status == ComparableStatus::kUnsupported ||
      (record.status == ComparableStatus::kProbeRequired &&
       !qualify_probes_)) {
    throw flagdnn::benchmarking::BenchmarkUnsupportedError(
        record.reason_code);
  }
  if (is_layout_operation(operation_)) {
    const flagdnn::testing::LayoutTestCase test_case =
        layout_case(specification, operation_);
    return std::make_unique<LayoutBenchmarkExecutable>(
        make_acdnn_layout_reference(
            test_case,
            layout_capability(operation_, record.status, test_case)));
  }
  if (operation_ == "reduction") {
    const flagdnn::testing::ReductionTestCase test_case =
        reduction_case(specification);
    return std::make_unique<ReductionBenchmarkExecutable>(
        make_acdnn_reduction_reference(
            test_case, reduction_capability(test_case, record.status)));
  }
  if (operation_ == "batchnorm") {
    return std::make_unique<NormalizationBenchmarkExecutable>(
        make_acdnn_batchnorm_reference(
            batchnorm_case(specification),
            batchnorm_capability(false, record.status,
                                 specification.tensors.front().data_type)));
  }
  if (operation_ == "batchnorm_inference") {
    return std::make_unique<NormalizationBenchmarkExecutable>(
        make_acdnn_batchnorm_inference_reference(
            batchnorm_inference_case(specification),
            batchnorm_capability(true, record.status,
                                 specification.tensors.front().data_type)));
  }
  if (operation_ == "layernorm") {
    return std::make_unique<NormalizationBenchmarkExecutable>(
        make_acdnn_layernorm_reference(
            layernorm_case(specification),
            normalization_capability(
                operation_, record.status,
                specification.tensors.front().data_type)));
  }
  if (operation_ == "rmsnorm") {
    return std::make_unique<NormalizationBenchmarkExecutable>(
        make_acdnn_rmsnorm_reference(
            rmsnorm_case(specification),
            normalization_capability(
                operation_, record.status,
                specification.tensors.front().data_type)));
  }
  if (operation_ == "matmul") {
    return std::make_unique<MatmulBenchmarkExecutable>(
        make_acdnn_matmul_reference(
            matmul_case(specification), matmul_capability(record.status)));
  }
  if (is_convolution_operation(operation_)) {
    flagdnn::testing::ConvolutionTestCase test_case =
        convolution_case(specification, operation_);
    return std::make_unique<ConvolutionBenchmarkExecutable>(
        make_acdnn_convolution_reference(
            test_case, convolution_capability(test_case, record.status)));
  }
  if (operation_ == "conv_bias_relu") {
    return std::make_unique<ConvolutionBenchmarkExecutable>(
        make_acdnn_conv_bias_relu_reference(
            conv_bias_relu_case(specification),
            conv_bias_relu_capability(record.status)));
  }
  return build_acdnn_pointwise_benchmark(specification, record.status);
}

void AcdnnProvider::require_exact_cases(
    std::span<const flagdnn::benchmarking::BenchmarkCase> cases) const {
  if (records_.size() != cases.size()) {
    throw std::runtime_error("THead benchmark capability case count mismatch");
  }
  std::set<std::string, std::less<>> expected;
  for (const auto &specification : cases) {
    if (!expected.insert(specification.name).second) {
      throw std::runtime_error("benchmark workload contains a duplicate case");
    }
    if (!records_.contains(specification.name)) {
      throw std::runtime_error("THead benchmark capability is missing case " +
                               specification.name);
    }
  }
}

const ComparableRecord &AcdnnProvider::lookup(
    std::string_view case_name) const {
  const auto iterator = records_.find(case_name);
  if (iterator == records_.end()) {
    throw std::runtime_error("unknown THead benchmark capability case");
  }
  return iterator->second;
}

}  // namespace flagdnn::validation::thead::benchmark
