// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "benchmark/acdnn_provider.hpp"

#include "acdnn_composite_reference.hpp"
#include "acdnn_reference.hpp"
#include "backend_pointwise_reference.hpp"
#include "capability.hpp"
#include "pointwise_reference.hpp"
#include "acdnn_pointwise_dag.hpp"

#include "common/benchmark_provider.hpp"
#include "common/case.hpp"

#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace flagdnn::validation::thead::benchmark {
namespace {

std::string backend_primitive(flagdnnPointwiseMode_t mode) {
  switch (mode) {
    case FLAGDNN_POINTWISE_ADD:
      return "acdnnBackendExecute(POINTWISE_ADD)";
    case FLAGDNN_POINTWISE_SUB:
      return "acdnnBackendExecute(POINTWISE_SUB)";
    case FLAGDNN_POINTWISE_MUL:
      return "acdnnBackendExecute(POINTWISE_MUL)";
    case FLAGDNN_POINTWISE_MIN:
      return "acdnnBackendExecute(POINTWISE_MIN)";
    case FLAGDNN_POINTWISE_MAX:
      return "acdnnBackendExecute(POINTWISE_MAX)";
    case FLAGDNN_POINTWISE_SQRT:
      return "acdnnBackendExecute(POINTWISE_SQRT)";
    case FLAGDNN_POINTWISE_RELU_FWD:
      return "acdnnBackendExecute(POINTWISE_RELU_FWD)";
    case FLAGDNN_POINTWISE_SIGMOID_FWD:
      return "acdnnBackendExecute(POINTWISE_SIGMOID_FWD)";
    case FLAGDNN_POINTWISE_TANH_FWD:
      return "acdnnBackendExecute(POINTWISE_TANH_FWD)";
    case FLAGDNN_POINTWISE_ELU_FWD:
      return "acdnnBackendExecute(POINTWISE_ELU_FWD,alpha=1)";
    case FLAGDNN_POINTWISE_GELU_FWD:
      return "acdnnBackendExecute(POINTWISE_GELU_FWD)";
    case FLAGDNN_POINTWISE_IDENTITY:
      return "acdnnBackendExecute(POINTWISE_IDENTITY_FWD)";
    case FLAGDNN_POINTWISE_NEG:
      return "acdnnBackendExecute(POINTWISE_NEG)";
    case FLAGDNN_POINTWISE_ABS:
      return "acdnnBackendExecute(POINTWISE_ABS)";
    case FLAGDNN_POINTWISE_CEIL:
      return "acdnnBackendExecute(POINTWISE_CEIL)";
    case FLAGDNN_POINTWISE_FLOOR:
      return "acdnnBackendExecute(POINTWISE_FLOOR)";
    case FLAGDNN_POINTWISE_EXP:
      return "acdnnBackendExecute(POINTWISE_EXP)";
    case FLAGDNN_POINTWISE_LOG:
      return "acdnnBackendExecute(POINTWISE_LOG)";
    case FLAGDNN_POINTWISE_COS:
      return "acdnnBackendExecute(POINTWISE_COS)";
    case FLAGDNN_POINTWISE_RSQRT:
      return "acdnnBackendExecute(POINTWISE_RSQRT)";
    case FLAGDNN_POINTWISE_SIN:
      return "acdnnBackendExecute(POINTWISE_SIN)";
    case FLAGDNN_POINTWISE_TAN:
      return "acdnnBackendExecute(POINTWISE_TAN)";
    case FLAGDNN_POINTWISE_SOFTPLUS_FWD:
      return "acdnnBackendExecute(POINTWISE_SOFTPLUS_FWD,beta=1)";
    case FLAGDNN_POINTWISE_SWISH_FWD:
      return "acdnnBackendExecute(POINTWISE_SWISH_FWD,beta=1.25)";
    case FLAGDNN_POINTWISE_GELU_APPROX_TANH_FWD:
      return "acdnnBackendExecute(POINTWISE_GELU_APPROX_TANH_FWD)";
    case FLAGDNN_POINTWISE_DIV:
      return "acdnnBackendExecute(POINTWISE_DIV)";
    case FLAGDNN_POINTWISE_POW:
      return "acdnnBackendExecute(POINTWISE_POW)";
    case FLAGDNN_POINTWISE_MOD:
      return "acdnnBackendExecute(POINTWISE_MOD)";
    case FLAGDNN_POINTWISE_SIGMOID_BWD:
      return "acdnnBackendExecute(POINTWISE_SIGMOID_BWD)";
    case FLAGDNN_POINTWISE_RECIPROCAL:
      return "acdnnBackendExecute(POINTWISE_DIV,numerator=1)";
    case FLAGDNN_POINTWISE_CMP_EQ:
      return "acdnnBackendExecute(POINTWISE_CMP_EQ)";
    case FLAGDNN_POINTWISE_CMP_NEQ:
      return "acdnnBackendExecute(POINTWISE_CMP_NEQ)";
    case FLAGDNN_POINTWISE_CMP_GT:
      return "acdnnBackendExecute(POINTWISE_CMP_GT)";
    case FLAGDNN_POINTWISE_CMP_GE:
      return "acdnnBackendExecute(POINTWISE_CMP_GE)";
    case FLAGDNN_POINTWISE_CMP_LT:
      return "acdnnBackendExecute(POINTWISE_CMP_LT)";
    case FLAGDNN_POINTWISE_CMP_LE:
      return "acdnnBackendExecute(POINTWISE_CMP_LE)";
    case FLAGDNN_POINTWISE_NOT_SET:
      break;
    default:
      break;
  }
  throw std::invalid_argument("unsupported typed pointwise benchmark mode");
}

std::string capability_dtype(flagdnnDataType_t data_type) {
  switch (data_type) {
    case FLAGDNN_DATA_INT32:
    case FLAGDNN_DATA_FP8_E8M0:
      throw std::invalid_argument(
          "THead validation does not support INT32 or E8M0 here");
    case FLAGDNN_DATA_FLOAT32:
      return "fp32";
    case FLAGDNN_DATA_FLOAT16:
      return "fp16";
    case FLAGDNN_DATA_BFLOAT16:
      return "bf16";
    case FLAGDNN_DATA_BOOLEAN:
      return "bool";
    case FLAGDNN_DATA_FP8_E4M3:
    case FLAGDNN_DATA_FP8_E5M2:
      break;
  }
  throw std::invalid_argument("unsupported pointwise benchmark data type");
}

flagdnn::testing::TestTensor
to_test_tensor(const flagdnn::benchmarking::TensorSpec &tensor) {
  return {tensor.uid, tensor.data_type, tensor.dimensions, tensor.strides,
          tensor.binding_byte_offset};
}

class ExecutableAdapter final
    : public flagdnn::benchmarking::BenchmarkExecutable {
 public:
  explicit ExecutableAdapter(
      std::unique_ptr<flagdnn::testing::TestExecutable> executable)
      : executable_(std::move(executable)) {
    if (executable_ == nullptr) {
      throw std::invalid_argument("acDNN benchmark executable is null");
    }
  }

  void prepare(std::span<const flagdnnBinding_t> bindings,
               flagdnnStream_t stream) override {
    executable_->prepare(bindings, stream);
  }

  [[nodiscard]] std::size_t workspace_size() const noexcept override {
    return executable_->workspace_size();
  }

  void execute(std::span<const flagdnnBinding_t> bindings, void *workspace,
               std::size_t workspace_size,
               flagdnnStream_t stream) override {
    executable_->execute(bindings, workspace, workspace_size, stream);
  }

 private:
  std::unique_ptr<flagdnn::testing::TestExecutable> executable_;
};

}  // namespace

std::unique_ptr<flagdnn::benchmarking::BenchmarkExecutable>
build_acdnn_pointwise_benchmark(
    const flagdnn::benchmarking::BenchmarkCase &specification,
    ComparableStatus qualification) {
  using flagdnn::benchmarking::Operation;
  if (specification.pointwise_mode >= FLAGDNN_POINTWISE_RELU_BWD &&
      specification.pointwise_mode <= FLAGDNN_POINTWISE_GELU_APPROX_TANH_BWD) {
    if (specification.tensors.size() != 3) {
      throw std::invalid_argument(
          "activation backward benchmark arity mismatch");
    }
    CapabilityRecord capability;
    capability.status = CapabilityStatus::kSupported;
    capability.path = ReferencePath::kBackendDescriptor;
    capability.constraints = CapabilityConstraints{};
    if (specification.pointwise_mode == FLAGDNN_POINTWISE_RELU_BWD &&
        specification.pointwise_attributes.flags != 0) {
      capability.reference_plan =
          acdnn_pointwise_dag_plan(specification.pointwise_mode);
    }
    return std::make_unique<ExecutableAdapter>(make_acdnn_pointwise_reference(
        {.mode = specification.pointwise_mode,
         .inputs = {to_test_tensor(specification.tensors[0]),
                    to_test_tensor(specification.tensors[1])},
         .output = to_test_tensor(specification.tensors[2]),
         .attributes = specification.pointwise_attributes},
        capability));
  }
  const auto dag_plan = acdnn_pointwise_dag_plan(specification.pointwise_mode);
  if (specification.operation == Operation::kPointwise && !dag_plan.empty()) {
    const auto mode = specification.pointwise_mode;
    const bool logical = mode == FLAGDNN_POINTWISE_LOGICAL_NOT ||
                         mode == FLAGDNN_POINTWISE_LOGICAL_AND ||
                         mode == FLAGDNN_POINTWISE_LOGICAL_OR;
    const std::size_t inputs_count = mode == FLAGDNN_POINTWISE_BINARY_SELECT ? 3 :
        (mode == FLAGDNN_POINTWISE_MOD || mode == FLAGDNN_POINTWISE_LOGICAL_AND ||
         mode == FLAGDNN_POINTWISE_LOGICAL_OR) ? 2 : 1;
    if (specification.tensors.size() != inputs_count + 1 ||
        specification.output_count != 1) {
      throw std::invalid_argument("acDNN pointwise DAG benchmark arity mismatch");
    }
    CapabilityRecord capability;
    capability.status = qualification == ComparableStatus::kComparable
                            ? CapabilityStatus::kSupported
                            : CapabilityStatus::kProbeRequired;
    capability.path = ReferencePath::kBackendDescriptor;
    capability.reference_plan = dag_plan;
    capability.constraints = CapabilityConstraints{
        .dtypes = {logical ? "bool" : capability_dtype(specification.tensors[0].data_type)},
        .compute_type = logical ? "bool" : "fp32",
        .rank = {.minimum = 1, .maximum = 8},
        .shape = {"same_shape", "positive_extents"},
        .layouts = {"contiguous"},
        .stride_policy = "dense_contiguous",
        .broadcast = "none",
        .attributes = {{"autotune", {"false"}},
                       {"reference_input_domain", {logical ? "canonical_boolean" :
                           mode == FLAGDNN_POINTWISE_MOD ? "finite_nonzero_divisor" : "finite"}}},
    };
    capability.reason_code = capability.status == CapabilityStatus::kProbeRequired
                                 ? "real_device_qualification_pending" : "";
    capability.detail = "Paired acDNN-only pointwise composite reference";
    std::vector<flagdnn::testing::TestTensor> inputs;
    for (std::size_t index = 0; index < inputs_count; ++index) {
      inputs.push_back(to_test_tensor(specification.tensors[index]));
    }
    return std::make_unique<ExecutableAdapter>(make_acdnn_pointwise_dag(
        {.mode = mode, .inputs = std::move(inputs),
         .output = to_test_tensor(specification.tensors.back()),
         .attributes = specification.pointwise_attributes}, capability));
  }
  const bool is_add_square =
      specification.operation == Operation::kGraph &&
      specification.name.starts_with("add_square_perf_");
  const bool is_add = specification.operation == Operation::kAdd;
  const bool is_sub =
      specification.operation == Operation::kPointwise &&
      specification.pointwise_mode == FLAGDNN_POINTWISE_SUB;
  const bool is_mul =
      specification.operation == Operation::kPointwise &&
      specification.pointwise_mode == FLAGDNN_POINTWISE_MUL;
  const bool is_min =
      specification.operation == Operation::kPointwise &&
      specification.pointwise_mode == FLAGDNN_POINTWISE_MIN;
  const bool is_max =
      specification.operation == Operation::kPointwise &&
      specification.pointwise_mode == FLAGDNN_POINTWISE_MAX;
  const bool is_div =
      specification.operation == Operation::kPointwise &&
      specification.pointwise_mode == FLAGDNN_POINTWISE_DIV;
  const bool is_pow =
      specification.operation == Operation::kPointwise &&
      specification.pointwise_mode == FLAGDNN_POINTWISE_POW;
  const bool is_mod =
      specification.operation == Operation::kPointwise &&
      specification.pointwise_mode == FLAGDNN_POINTWISE_MOD;
  const bool is_relu = specification.operation == Operation::kRelu;
  const bool is_leaky_relu =
      specification.operation == Operation::kPointwise &&
      specification.pointwise_mode == FLAGDNN_POINTWISE_RELU_FWD &&
      specification.pointwise_attributes.flags ==
          FLAGDNN_POINTWISE_ATTRIBUTE_RELU_LOWER_CLIP_SLOPE &&
      specification.pointwise_attributes.relu_lower_clip_slope == 0.2;
  const bool is_sigmoid =
      specification.operation == Operation::kPointwise &&
      specification.pointwise_mode == FLAGDNN_POINTWISE_SIGMOID_FWD;
  const bool is_tanh =
      specification.operation == Operation::kPointwise &&
      specification.pointwise_mode == FLAGDNN_POINTWISE_TANH_FWD;
  const bool is_elu =
      specification.operation == Operation::kPointwise &&
      specification.pointwise_mode == FLAGDNN_POINTWISE_ELU_FWD;
  const bool is_identity =
      specification.operation == Operation::kPointwise &&
      specification.pointwise_mode == FLAGDNN_POINTWISE_IDENTITY;
  const bool is_gelu =
      specification.operation == Operation::kPointwise &&
      specification.pointwise_mode == FLAGDNN_POINTWISE_GELU_FWD;
  const bool is_sqrt =
      specification.operation == Operation::kPointwise &&
      specification.pointwise_mode == FLAGDNN_POINTWISE_SQRT;
  const bool is_neg =
      specification.operation == Operation::kPointwise &&
      specification.pointwise_mode == FLAGDNN_POINTWISE_NEG;
  const bool is_abs =
      specification.operation == Operation::kPointwise &&
      specification.pointwise_mode == FLAGDNN_POINTWISE_ABS;
  const bool is_ceil =
      specification.operation == Operation::kPointwise &&
      specification.pointwise_mode == FLAGDNN_POINTWISE_CEIL;
  const bool is_floor =
      specification.operation == Operation::kPointwise &&
      specification.pointwise_mode == FLAGDNN_POINTWISE_FLOOR;
  const bool is_exp =
      specification.operation == Operation::kPointwise &&
      specification.pointwise_mode == FLAGDNN_POINTWISE_EXP;
  const bool is_log =
      specification.operation == Operation::kPointwise &&
      specification.pointwise_mode == FLAGDNN_POINTWISE_LOG;
  const bool is_cos =
      specification.operation == Operation::kPointwise &&
      specification.pointwise_mode == FLAGDNN_POINTWISE_COS;
  const bool is_rsqrt =
      specification.operation == Operation::kPointwise &&
      specification.pointwise_mode == FLAGDNN_POINTWISE_RSQRT;
  const bool is_sin =
      specification.operation == Operation::kPointwise &&
      specification.pointwise_mode == FLAGDNN_POINTWISE_SIN;
  const bool is_tan =
      specification.operation == Operation::kPointwise &&
      specification.pointwise_mode == FLAGDNN_POINTWISE_TAN;
  const bool is_softplus =
      specification.operation == Operation::kPointwise &&
      specification.pointwise_mode == FLAGDNN_POINTWISE_SOFTPLUS_FWD;
  const bool is_swish =
      specification.operation == Operation::kPointwise &&
      specification.pointwise_mode == FLAGDNN_POINTWISE_SWISH_FWD;
  const bool is_gelu_approx_tanh =
      specification.operation == Operation::kPointwise &&
      specification.pointwise_mode ==
          FLAGDNN_POINTWISE_GELU_APPROX_TANH_FWD;
  const bool is_sigmoid_backward =
      specification.operation == Operation::kPointwise &&
      specification.pointwise_mode == FLAGDNN_POINTWISE_SIGMOID_BWD;
  const bool is_reciprocal =
      specification.operation == Operation::kPointwise &&
      specification.pointwise_mode == FLAGDNN_POINTWISE_RECIPROCAL;
  const bool is_comparison =
      specification.operation == Operation::kPointwise &&
      (specification.pointwise_mode == FLAGDNN_POINTWISE_CMP_EQ ||
       specification.pointwise_mode == FLAGDNN_POINTWISE_CMP_NEQ ||
       specification.pointwise_mode == FLAGDNN_POINTWISE_CMP_GT ||
       specification.pointwise_mode == FLAGDNN_POINTWISE_CMP_GE ||
       specification.pointwise_mode == FLAGDNN_POINTWISE_CMP_LT ||
       specification.pointwise_mode == FLAGDNN_POINTWISE_CMP_LE);
  const bool is_activation = is_relu || is_leaky_relu || is_sigmoid ||
                             is_tanh || is_elu || is_identity || is_gelu;
  const bool is_unary =
      is_activation || is_sqrt || is_neg || is_abs || is_ceil || is_floor ||
      is_exp || is_log || is_cos || is_rsqrt || is_sin || is_tan ||
      is_softplus || is_swish || is_gelu_approx_tanh || is_reciprocal;
  const std::size_t expected_tensor_count = is_unary ? 2 : 3;
  if ((!is_add && !is_sub && !is_mul && !is_min && !is_max && !is_div &&
       !is_pow && !is_mod && !is_sigmoid_backward && !is_add_square &&
       !is_comparison && !is_unary) ||
      specification.tensors.size() != expected_tensor_count ||
      specification.output_count != 1) {
    throw std::invalid_argument(
        "THead acDNN benchmark provider requires a qualified "
        "catalog-qualified pointwise case");
  }
  flagdnnPointwiseMode_t mode = FLAGDNN_POINTWISE_MAX;
  std::string primitive = "acdnnOpTensor(MAX)";
  std::string detail = "THead paired Max benchmark";
  if (is_add) {
    mode = FLAGDNN_POINTWISE_ADD;
    primitive = "acdnnOpTensor(ADD)";
    detail = "THead paired Add benchmark";
  } else if (is_add_square) {
    if (specification.graph.intermediates.size() != 1 ||
        specification.graph.nodes.size() != 2 ||
        specification.graph.nodes[0].operation != Operation::kPointwise ||
        specification.graph.nodes[0].pointwise_mode !=
            FLAGDNN_POINTWISE_MUL ||
        specification.graph.nodes[0].input_uids.size() != 2 ||
        specification.graph.nodes[0].input_uids[0] !=
            specification.graph.nodes[0].input_uids[1] ||
        specification.graph.nodes[0].output_uid !=
            specification.graph.intermediates[0].uid ||
        specification.graph.nodes[1].operation != Operation::kPointwise ||
        specification.graph.nodes[1].pointwise_mode !=
            FLAGDNN_POINTWISE_ADD ||
        specification.graph.nodes[1].input_uids !=
            std::vector<std::int64_t>{specification.tensors[0].uid,
                                      specification.graph.intermediates[0].uid} ||
        specification.graph.nodes[1].output_uid !=
            specification.tensors.back().uid) {
      throw std::invalid_argument(
          "THead AddSquare benchmark graph is not canonical");
    }
    mode = FLAGDNN_POINTWISE_NOT_SET;
    primitive = "acdnnOpTensor(MUL)->acdnnOpTensor(ADD)";
    detail = "THead paired AddSquare acDNN stable-primitive DAG benchmark";
  } else if (is_sub) {
    mode = FLAGDNN_POINTWISE_SUB;
    primitive = "acdnnOpTensor(ADD,alpha_right=-alpha)";
    detail = "THead paired Sub benchmark";
  } else if (is_mul) {
    mode = FLAGDNN_POINTWISE_MUL;
    primitive = "acdnnOpTensor(MUL)";
    detail = specification.name.starts_with("scale_")
                 ? "THead paired Scale-as-Mul benchmark"
                 : "THead paired Mul benchmark";
  } else if (is_min) {
    mode = FLAGDNN_POINTWISE_MIN;
    primitive = "acdnnOpTensor(MIN)";
    detail = "THead paired Min benchmark";
  } else if (is_div) {
    mode = FLAGDNN_POINTWISE_DIV;
    primitive = "acdnnBackendExecute(POINTWISE_DIV)";
    detail = "THead paired positive-input backend descriptor Div benchmark";
  } else if (is_pow) {
    mode = FLAGDNN_POINTWISE_POW;
    primitive = "acdnnBackendExecute(POINTWISE_POW)";
    detail = "THead paired positive-input backend descriptor Pow benchmark";
  } else if (is_mod) {
    mode = FLAGDNN_POINTWISE_MOD;
    primitive = "acdnnBackendExecute(POINTWISE_MOD)";
    detail = "THead paired positive-input backend descriptor Mod benchmark";
  } else if (is_sigmoid_backward) {
    mode = FLAGDNN_POINTWISE_SIGMOID_BWD;
    primitive = "acdnnBackendExecute(POINTWISE_SIGMOID_BWD)";
    detail = "THead paired backend descriptor SigmoidBackward benchmark";
  } else if (is_reciprocal) {
    mode = FLAGDNN_POINTWISE_RECIPROCAL;
    primitive = "acdnnBackendExecute(POINTWISE_DIV,numerator=1)";
    detail = "THead paired positive-input acDNN Div reciprocal benchmark";
  } else if (is_comparison) {
    mode = specification.pointwise_mode;
    switch (mode) {
      case FLAGDNN_POINTWISE_CMP_EQ:
        primitive = "acdnnBackendExecute(POINTWISE_CMP_EQ)";
        break;
      case FLAGDNN_POINTWISE_CMP_NEQ:
        primitive = "acdnnBackendExecute(POINTWISE_CMP_NEQ)";
        break;
      case FLAGDNN_POINTWISE_CMP_GT:
        primitive = "acdnnBackendExecute(POINTWISE_CMP_GT)";
        break;
      case FLAGDNN_POINTWISE_CMP_GE:
        primitive = "acdnnBackendExecute(POINTWISE_CMP_GE)";
        break;
      case FLAGDNN_POINTWISE_CMP_LT:
        primitive = "acdnnBackendExecute(POINTWISE_CMP_LT)";
        break;
      case FLAGDNN_POINTWISE_CMP_LE:
        primitive = "acdnnBackendExecute(POINTWISE_CMP_LE)";
        break;
      default:
        throw std::logic_error("invalid comparison benchmark mode");
    }
    detail = "THead paired fp32-to-byte-boolean acDNN comparison benchmark";
  } else if (is_relu) {
    mode = FLAGDNN_POINTWISE_RELU_FWD;
    primitive = "acdnnActivationForward(RELU,NOT_PROPAGATE_NAN)";
    detail = "THead paired finite-input Relu benchmark";
  } else if (is_leaky_relu) {
    mode = FLAGDNN_POINTWISE_RELU_FWD;
    primitive = "acdnnOpTensor(LeakyReLU-DAG)";
    detail = "THead paired finite-input LeakyReLU acDNN DAG benchmark";
  } else if (is_sigmoid) {
    mode = FLAGDNN_POINTWISE_SIGMOID_FWD;
    primitive = "acdnnActivationForward(SIGMOID,NOT_PROPAGATE_NAN)";
    detail = "THead paired finite-input Sigmoid benchmark";
  } else if (is_tanh) {
    mode = FLAGDNN_POINTWISE_TANH_FWD;
    primitive = "acdnnActivationForward(TANH,NOT_PROPAGATE_NAN)";
    detail = "THead paired finite-input Tanh benchmark";
  } else if (is_elu) {
    mode = FLAGDNN_POINTWISE_ELU_FWD;
    primitive =
        "acdnnActivationForward(ELU,NOT_PROPAGATE_NAN,alpha=1)";
    detail = "THead paired finite-input Elu alpha=1 benchmark";
  } else if (is_identity) {
    mode = FLAGDNN_POINTWISE_IDENTITY;
    primitive = "acdnnTransformTensor(alpha=1,beta=0)";
    detail = "THead paired exact Identity benchmark";
  } else if (is_gelu) {
    mode = FLAGDNN_POINTWISE_GELU_FWD;
    primitive = "acdnnActivationForward(GELU,NOT_PROPAGATE_NAN)";
    detail = "THead paired finite-input exact Gelu benchmark";
  } else if (is_sqrt) {
    mode = FLAGDNN_POINTWISE_SQRT;
    primitive = "acdnnOpTensor(SQRT)";
    detail = "THead paired positive-input exact Sqrt benchmark";
  } else if (is_neg) {
    mode = FLAGDNN_POINTWISE_NEG;
    primitive = "acdnnTransformTensor(alpha=-1,beta=0)";
    detail = "THead paired exact Neg benchmark";
  } else if (is_abs) {
    mode = FLAGDNN_POINTWISE_ABS;
    primitive = "acdnnBackendExecute(POINTWISE_ABS)";
    detail = "THead paired exact backend descriptor Abs benchmark";
  } else if (is_ceil) {
    mode = FLAGDNN_POINTWISE_CEIL;
    primitive = "acdnnBackendExecute(POINTWISE_CEIL)";
    detail = "THead paired exact backend descriptor Ceil benchmark";
  } else if (is_floor) {
    mode = FLAGDNN_POINTWISE_FLOOR;
    primitive = "acdnnBackendExecute(POINTWISE_FLOOR)";
    detail = "THead paired exact backend descriptor Floor benchmark";
  } else if (is_exp) {
    mode = FLAGDNN_POINTWISE_EXP;
    primitive = "acdnnBackendExecute(POINTWISE_EXP)";
    detail = "THead paired exact backend descriptor Exp benchmark";
  } else if (is_log) {
    mode = FLAGDNN_POINTWISE_LOG;
    primitive = "acdnnBackendExecute(POINTWISE_LOG)";
    detail = "THead paired positive-input backend descriptor Log benchmark";
  } else if (is_cos) {
    mode = FLAGDNN_POINTWISE_COS;
    primitive = "acdnnBackendExecute(POINTWISE_COS)";
    detail = "THead paired exact backend descriptor Cos benchmark";
  } else if (is_rsqrt) {
    mode = FLAGDNN_POINTWISE_RSQRT;
    primitive = "acdnnBackendExecute(POINTWISE_RSQRT)";
    detail =
        "THead paired positive-input backend descriptor Rsqrt benchmark";
  } else if (is_sin) {
    mode = FLAGDNN_POINTWISE_SIN;
    primitive = "acdnnBackendExecute(POINTWISE_SIN)";
    detail = "THead paired exact backend descriptor Sin benchmark";
  } else if (is_tan) {
    mode = FLAGDNN_POINTWISE_TAN;
    primitive = "acdnnBackendExecute(POINTWISE_TAN)";
    detail = "THead paired bounded-input backend descriptor Tan benchmark";
  } else if (is_softplus) {
    mode = FLAGDNN_POINTWISE_SOFTPLUS_FWD;
    primitive = "acdnnBackendExecute(POINTWISE_SOFTPLUS_FWD,beta=1)";
    detail = "THead paired backend descriptor Softplus beta=1 benchmark";
  } else if (is_swish) {
    mode = FLAGDNN_POINTWISE_SWISH_FWD;
    primitive = "acdnnBackendExecute(POINTWISE_SWISH_FWD,beta=1.25)";
    detail = "THead paired backend descriptor Swish beta=1.25 benchmark";
  } else if (is_gelu_approx_tanh) {
    mode = FLAGDNN_POINTWISE_GELU_APPROX_TANH_FWD;
    primitive =
        "acdnnBackendExecute(POINTWISE_GELU_APPROX_TANH_FWD)";
    detail = "THead paired backend descriptor GeluApproxTanh benchmark";
  }

  const flagdnnDataType_t input_data_type =
      specification.tensors.front().data_type;
  const bool converted_bfloat16 =
      input_data_type == FLAGDNN_DATA_BFLOAT16 &&
      (is_min || is_max || is_relu);
  const bool stable_fp16 =
      input_data_type == FLAGDNN_DATA_FLOAT16 &&
      (is_sub || is_mul || is_min || is_max || is_relu || is_sigmoid ||
       is_tanh || is_elu || is_identity || is_gelu || is_sqrt || is_neg);
  const bool typed_backend = input_data_type != FLAGDNN_DATA_FLOAT32 &&
                             !is_add_square && !converted_bfloat16 &&
                             !is_leaky_relu &&
                             !stable_fp16;
  const bool backend_add_square =
      is_add_square && input_data_type == FLAGDNN_DATA_BFLOAT16;
  if (typed_backend) {
    primitive = backend_primitive(mode);
  }

  CapabilityRecord capability;
  capability.status = qualification == ComparableStatus::kComparable
                          ? CapabilityStatus::kSupported
                          : CapabilityStatus::kProbeRequired;
  capability.path = (typed_backend || converted_bfloat16 || is_leaky_relu ||
                     backend_add_square || is_abs ||
                     is_ceil || is_floor || is_exp || is_log || is_cos ||
                     is_rsqrt || is_sin || is_tan || is_softplus ||
                     is_swish || is_gelu_approx_tanh || is_div || is_pow ||
                     is_mod || is_sigmoid_backward || is_reciprocal ||
                     is_comparison)
                        ? ReferencePath::kBackendDescriptor
                        : ReferencePath::kStablePrimitive;
  if (is_leaky_relu) {
    capability.reference_plan = {
        "acdnnBackendExecute(POINTWISE_IDENTITY_FWD,leaky-input-fp32)",
        "acdnnTransformTensor(alpha=-1,beta=0)",
        "acdnnActivationForward(RELU,NOT_PROPAGATE_NAN,positive)",
        "acdnnActivationForward(RELU,NOT_PROPAGATE_NAN,negative)",
        "acdnnOpTensor(ADD,alpha_right=-0.2)",
        "acdnnBackendExecute(POINTWISE_IDENTITY_FWD,leaky-output-data-type)"};
  } else if (converted_bfloat16) {
    capability.reference_plan = {
        "acdnnBackendExecute(POINTWISE_IDENTITY_FWD,convert-input0-fp32)"};
    if (is_min || is_max) {
      capability.reference_plan.push_back(
          "acdnnBackendExecute(POINTWISE_IDENTITY_FWD,convert-input1-fp32)");
    }
    capability.reference_plan.push_back(primitive);
    capability.reference_plan.push_back(
        "acdnnBackendExecute(POINTWISE_IDENTITY_FWD,convert-output-bfloat16)");
  } else if (backend_add_square) {
    capability.reference_plan = {
        "acdnnBackendExecute(POINTWISE_MUL)",
        "acdnnBackendExecute(POINTWISE_ADD)"};
  } else if (is_add_square) {
    capability.reference_plan = {"acdnnOpTensor(MUL)",
                                 "acdnnOpTensor(ADD)"};
  } else {
    capability.reference_plan = {primitive};
  }
  std::map<std::string, std::vector<std::string>, std::less<>> attributes =
      {{"alpha", {"1"}}, {"autotune", {"false"}}};
  if (is_activation) {
    attributes = {
        {"autotune", {"false"}},
        {"reference_nan_policy", {"not_propagate_finite_inputs_only"}},
    };
    if (is_relu || is_leaky_relu) {
      attributes.insert({"relu_lower_clip", {"0"}});
      attributes.insert(
          {"relu_lower_clip_slope", {is_leaky_relu ? "0.2" : "0"}});
      attributes.insert({"relu_upper_clip", {"0"}});
      attributes.insert({"relu_upper_clip_set", {"false"}});
    } else if (is_elu) {
      attributes.insert({"elu_alpha", {"1"}});
    }
  } else if (is_softplus) {
    attributes = {{"autotune", {"false"}},
                  {"softplus_beta", {"1"}}};
  } else if (is_swish) {
    attributes = {{"autotune", {"false"}},
                  {"swish_beta", {"1.25"}}};
  }
  capability.constraints = CapabilityConstraints{
      .dtypes = {capability_dtype(input_data_type)},
      .compute_type = is_comparison ? "bool" : "fp32",
      .rank = {.minimum = 1, .maximum = 8},
      .shape = {"same_shape", "positive_extents"},
      .layouts = {"contiguous"},
      .stride_policy = "dense_contiguous",
      .broadcast = "none",
      .attributes = std::move(attributes),
  };
  capability.reason_code =
      capability.status == CapabilityStatus::kProbeRequired
          ? "real_device_qualification_pending"
          : "";
  capability.detail = std::move(detail);

  std::vector<flagdnn::testing::TestTensor> inputs;
  inputs.push_back(to_test_tensor(specification.tensors[0]));
  if (!is_unary) {
    inputs.push_back(to_test_tensor(specification.tensors[1]));
  }
  std::unique_ptr<flagdnn::testing::TestExecutable> reference;
  if (is_add_square) {
    reference = make_acdnn_add_square_reference(
        to_test_tensor(specification.tensors[0]),
        to_test_tensor(specification.tensors[1]),
        to_test_tensor(specification.tensors.back()), capability);
  } else {
    reference = make_acdnn_pointwise_reference(
        {.mode = mode,
         .inputs = std::move(inputs),
         .output = to_test_tensor(specification.tensors.back()),
         .alpha = is_add ? specification.add_alpha : 1.0,
         .attributes = specification.pointwise_attributes},
        capability);
  }
  return std::make_unique<ExecutableAdapter>(std::move(reference));
}

}  // namespace flagdnn::validation::thead::benchmark
