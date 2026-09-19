/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/pointwise.hpp"
#include "validation/acl_runtime.hpp"
#include "validation/development_environment.hpp"
#include "validation/exact_reference.hpp"
#include "validation/functional/aclnn_binary_pointwise.hpp"
#include "validation/functional/aclnn_ternary_pointwise.hpp"
#include "validation/functional/aclnn_unary_pointwise.hpp"
#include "validation/gelu_contract.hpp"
#include "validation/mod_contract.hpp"
#include "validation/tensor_io.hpp"

#include <flagdnn/flagdnn.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace flagdnn::testing {
namespace {

namespace acl = flagdnn::validation::ascend;
namespace gelu = flagdnn::validation::ascend::gelu_contract;
namespace mod = flagdnn::validation::ascend::mod_contract;
namespace tensor_io = flagdnn::validation::ascend::tensor_io;

struct AscendPointwiseCase {
  PointwiseTestCase specification;
  std::vector<std::vector<float>> explicit_inputs;
  bool strict_zero_signbit = false;
};

std::string operation_name(flagdnnPointwiseMode_t mode) {
  switch (mode) {
    case FLAGDNN_POINTWISE_RELU_FWD:
      return "relu";
    case FLAGDNN_POINTWISE_IDENTITY:
      return "identity";
    case FLAGDNN_POINTWISE_NEG:
      return "neg";
    case FLAGDNN_POINTWISE_ABS:
      return "abs";
    case FLAGDNN_POINTWISE_CEIL:
      return "ceil";
    case FLAGDNN_POINTWISE_COS:
      return "cos";
    case FLAGDNN_POINTWISE_ERF:
      return "erf";
    case FLAGDNN_POINTWISE_FLOOR:
      return "floor";
    case FLAGDNN_POINTWISE_RECIPROCAL:
      return "reciprocal";
    case FLAGDNN_POINTWISE_SQRT:
      return "sqrt";
    case FLAGDNN_POINTWISE_RSQRT:
      return "rsqrt";
    case FLAGDNN_POINTWISE_SIN:
      return "sin";
    case FLAGDNN_POINTWISE_TAN:
      return "tan";
    case FLAGDNN_POINTWISE_EXP:
      return "exp";
    case FLAGDNN_POINTWISE_LOG:
      return "log";
    case FLAGDNN_POINTWISE_TANH_FWD:
      return "tanh";
    case FLAGDNN_POINTWISE_SIGMOID_FWD:
      return "sigmoid";
    case FLAGDNN_POINTWISE_ELU_FWD:
      return "elu";
    case FLAGDNN_POINTWISE_GELU_FWD:
      return "gelu";
    case FLAGDNN_POINTWISE_SOFTPLUS_FWD:
      return "softplus";
    case FLAGDNN_POINTWISE_SWISH_FWD:
      return "swish";
    case FLAGDNN_POINTWISE_GELU_APPROX_TANH_FWD:
      return "gelu_approx_tanh";
    case FLAGDNN_POINTWISE_LOGICAL_NOT:
      return "logical_not";
    case FLAGDNN_POINTWISE_CMP_EQ:
      return "cmp_eq";
    case FLAGDNN_POINTWISE_CMP_NEQ:
      return "cmp_neq";
    case FLAGDNN_POINTWISE_CMP_GT:
      return "cmp_gt";
    case FLAGDNN_POINTWISE_CMP_GE:
      return "cmp_ge";
    case FLAGDNN_POINTWISE_CMP_LT:
      return "cmp_lt";
    case FLAGDNN_POINTWISE_CMP_LE:
      return "cmp_le";
    case FLAGDNN_POINTWISE_SUB:
      return "sub";
    case FLAGDNN_POINTWISE_MUL:
      return "mul";
    case FLAGDNN_POINTWISE_DIV:
      return "div";
    case FLAGDNN_POINTWISE_MOD:
      return "mod";
    case FLAGDNN_POINTWISE_POW:
      return "pow";
    case FLAGDNN_POINTWISE_SIGMOID_BWD:
      return "sigmoid_backward";
    case FLAGDNN_POINTWISE_MIN:
      return "min";
    case FLAGDNN_POINTWISE_MAX:
      return "max";
    case FLAGDNN_POINTWISE_LOGICAL_AND:
      return "logical_and";
    case FLAGDNN_POINTWISE_LOGICAL_OR:
      return "logical_or";
    case FLAGDNN_POINTWISE_BINARY_SELECT:
      return "binary_select";
    default:
      throw std::invalid_argument(
          "Ascend pointwise runner received an unsupported mode");
  }
}

bool is_gelu_mode(flagdnnPointwiseMode_t mode) noexcept;
bool requires_repeatable_reference_contract(
    flagdnnPointwiseMode_t mode) noexcept;

bool is_comparison_mode(flagdnnPointwiseMode_t mode) noexcept {
  return mode == FLAGDNN_POINTWISE_CMP_EQ ||
         mode == FLAGDNN_POINTWISE_CMP_NEQ ||
         mode == FLAGDNN_POINTWISE_CMP_GT ||
         mode == FLAGDNN_POINTWISE_CMP_GE ||
         mode == FLAGDNN_POINTWISE_CMP_LT ||
         mode == FLAGDNN_POINTWISE_CMP_LE;
}

bool is_logical_mode(flagdnnPointwiseMode_t mode) noexcept {
  return mode == FLAGDNN_POINTWISE_LOGICAL_NOT ||
         mode == FLAGDNN_POINTWISE_LOGICAL_AND ||
         mode == FLAGDNN_POINTWISE_LOGICAL_OR;
}

void require_logical_extension_binding_offsets(
    const PointwiseTestCase& test_case) {
  const auto require_offset = [](const TestTensor& tensor) {
    if (tensor.binding_byte_offset == 0 ||
        tensor.binding_byte_offset % 32U != 0U) {
      throw std::logic_error(
          "Ascend logical extension binding offsets must be nonzero and "
          "32-byte aligned");
    }
  };
  for (const TestTensor& input : test_case.inputs) {
    require_offset(input);
  }
  require_offset(test_case.output);
}

std::vector<float> make_input(const TestTensor& tensor,
                              std::size_t input_index,
                              PointwiseInputDomain domain) {
  std::vector<float> result(tensor_io::element_count(tensor));
  for (std::size_t index = 0; index < result.size(); ++index) {
    const int centered =
        static_cast<int>((index * 17 + input_index * 11) % 41) - 20;
    const float real_value =
        static_cast<float>(centered) / static_cast<float>(13 + input_index);
    switch (domain) {
      case PointwiseInputDomain::kReal:
        result[index] = real_value;
        break;
      case PointwiseInputDomain::kScaled:
        result[index] = real_value * 4.0F;
        break;
      case PointwiseInputDomain::kPositive:
        result[index] = std::abs(real_value) + 0.5F;
        break;
      case PointwiseInputDomain::kDivisor:
      case PointwiseInputDomain::kModulo:
        result[index] = input_index == 1
                            ? std::abs(real_value) + 0.5F
                            : real_value;
        break;
      case PointwiseInputDomain::kPower:
        result[index] =
            input_index == 0
                ? std::abs(real_value) + 0.5F
                : std::fmod(std::abs(real_value), 2.0F) + 0.125F;
        break;
      case PointwiseInputDomain::kModuloSigned: {
        constexpr std::array<float, 6> kLeft = {
            -3.0F, -3.0F, 3.0F, 3.0F, -5.5F, 5.5F};
        constexpr std::array<float, 6> kRight = {
            2.0F, -2.0F, 2.0F, -2.0F, 2.25F, -2.25F};
        result[index] = input_index == 0
                            ? kLeft[index % kLeft.size()]
                            : kRight[index % kRight.size()];
        break;
      }
      case PointwiseInputDomain::kTan:
        result[index] = static_cast<float>(centered) / 40.0F;
        break;
      case PointwiseInputDomain::kLogical:
        // Across the first four elements, the two input streams form the
        // complete 00/01/10/11 truth table.  Unary LogicalNot also sees both
        // false and true values.
        result[index] = input_index == 0
                            ? static_cast<float>((index / 2U) % 2U)
                            : static_cast<float>(index % 2U);
        break;
      case PointwiseInputDomain::kComparison: {
        constexpr std::array<float, 8> kLeft = {
            0.0F, -0.0F, -3.0F, 4.0F, 5.0F, -7.0F, 2.0F, 9.0F};
        constexpr std::array<float, 8> kRight = {
            -0.0F, 0.0F, -2.0F, 3.0F, 6.0F, -7.0F, 2.0F, 10.0F};
        result[index] = input_index == 0
                            ? kLeft[index % kLeft.size()]
                            : kRight[index % kRight.size()];
        break;
      }
      default:
        throw std::invalid_argument(
            "Ascend pointwise validation received an unsupported "
            "input domain");
    }
  }
  return result;
}

struct PointwiseReferencePlan {
  std::vector<TestTensor> inputs;
  TestTensor output;
};

PointwiseReferencePlan make_reference_plan(
    const PointwiseTestCase& test_case) {
  if (test_case.inputs.size() == 1) {
    const AclnnUnaryPointwisePlan plan =
        plan_aclnn_unary_pointwise(test_case);
    return {{plan.input}, plan.output};
  }
  if (test_case.inputs.size() == 3) {
    const AclnnTernaryPointwisePlan plan =
        plan_aclnn_ternary_pointwise(test_case);
    return {{plan.self, plan.other, plan.condition}, plan.output};
  }
  const AclnnBinaryPointwisePlan plan =
      plan_aclnn_binary_pointwise(test_case);
  return {{plan.left, plan.right}, plan.output};
}

struct InputBuffer {
  std::unique_ptr<acl::DeviceBuffer> device;
};

InputBuffer make_input_buffer(const TestTensor& tensor,
                              std::size_t input_index,
                              PointwiseInputDomain domain,
                              acl::Stream& stream,
                              std::span<const float> explicit_input) {
  std::vector<float> generated;
  if (explicit_input.empty()) {
    generated = make_input(tensor, input_index, domain);
  } else {
    if (explicit_input.size() != tensor_io::element_count(tensor)) {
      throw std::invalid_argument(
          "Ascend explicit pointwise input size is invalid");
    }
    generated.assign(explicit_input.begin(), explicit_input.end());
  }
  const std::vector<float> logical =
      tensor_io::quantize(generated, tensor.data_type);
  const std::vector<float> physical = tensor_io::scatter(logical, tensor);
  const std::vector<std::uint8_t> encoded =
      tensor_io::encode(physical, tensor.data_type);
  auto device =
      std::make_unique<acl::DeviceBuffer>(tensor_io::allocation_byte_count(tensor));
  device->copy_from_host_at(encoded.data(),
                            encoded.size(),
                            tensor.binding_byte_offset,
                            stream.get());
  return {std::move(device)};
}

std::unique_ptr<acl::DeviceBuffer> make_output_buffer(
    const TestTensor& tensor,
    acl::Stream& stream) {
  const std::vector<float> initial(tensor_io::storage_element_count(tensor),
                                   tensor_io::kPaddingSentinel);
  const std::vector<std::uint8_t> encoded =
      tensor_io::encode(initial, tensor.data_type);
  auto result =
      std::make_unique<acl::DeviceBuffer>(tensor_io::allocation_byte_count(tensor));
  result->copy_from_host_at(encoded.data(),
                            encoded.size(),
                            tensor.binding_byte_offset,
                            stream.get());
  return result;
}

std::vector<float> read_output(const acl::DeviceBuffer& buffer,
                               const TestTensor& tensor,
                               acl::Stream& stream,
                               std::string_view provider) {
  std::vector<std::uint8_t> encoded(tensor_io::encoded_byte_count(tensor));
  buffer.copy_to_host_at(encoded.data(),
                         encoded.size(),
                         tensor.binding_byte_offset,
                         stream.get());
  stream.synchronize();
  return tensor_io::decode_storage(provider, encoded, tensor);
}

std::string single_line_reason(std::string_view reason) {
  std::string result(reason);
  std::replace(result.begin(), result.end(), '\r', ' ');
  std::replace(result.begin(), result.end(), '\n', ' ');
  return result;
}

void compare_provider_outputs(std::span<const float> actual,
                              std::span<const float> reference,
                              const PointwiseTestCase& test_case,
                              bool strict_zero_signbit) {
  double absolute_tolerance = test_case.absolute_tolerance;
  double relative_tolerance = test_case.relative_tolerance;
  if (test_case.mode == FLAGDNN_POINTWISE_GELU_FWD) {
    absolute_tolerance =
        std::max(absolute_tolerance, gelu::kAclnnExactAbsoluteTolerance);
    relative_tolerance =
        std::max(relative_tolerance, gelu::kAclnnExactRelativeTolerance);
  }
  acl::compare_exact_reference(actual,
                               reference,
                               absolute_tolerance,
                               relative_tolerance,
                               test_case.name);
  if (!strict_zero_signbit) {
    return;
  }
  for (std::size_t index = 0; index < actual.size(); ++index) {
    if (!mod::strict_ieee_matches(actual[index], reference[index])) {
      throw std::runtime_error(
          test_case.name +
          " exact-reference signed-zero mismatch at index " +
          std::to_string(index));
    }
  }
}

void execute(PointwiseExecutable& executable,
             std::span<const flagdnnBinding_t> bindings,
             acl::DeviceBuffer& workspace,
             acl::Stream& stream) {
  executable.execute(bindings,
                     executable.workspace_size() == 0
                         ? nullptr
                         : workspace.opaque(),
                     executable.workspace_size(),
                     stream.opaque());
}

std::optional<std::string> run_case(const AscendPointwiseCase& ascend_case,
                                    flagdnn::Handle& handle,
                                    acl::Stream& stream) {
  const PointwiseTestCase& test_case = ascend_case.specification;
  validate_pointwise_case(test_case);
  (void)operation_name(test_case.mode);
  const bool strict_mod = test_case.mode == FLAGDNN_POINTWISE_MOD &&
                          ascend_case.strict_zero_signbit &&
                          !ascend_case.explicit_inputs.empty();
  const PointwiseReferencePlan reference_plan =
      make_reference_plan(test_case);
  std::unique_ptr<PointwiseExecutable> flagdnn =
      build_flagdnn_pointwise(handle, test_case);
  std::unique_ptr<PointwiseExecutable> reference =
      build_pointwise_reference(test_case);

  std::vector<InputBuffer> flagdnn_inputs;
  std::vector<InputBuffer> reference_inputs;
  flagdnn_inputs.reserve(test_case.inputs.size());
  reference_inputs.reserve(reference_plan.inputs.size());
  if (!ascend_case.explicit_inputs.empty() &&
      ascend_case.explicit_inputs.size() != test_case.inputs.size()) {
    throw std::invalid_argument(
        "Ascend explicit pointwise input count is invalid");
  }
  if (reference_plan.inputs.size() != test_case.inputs.size()) {
    throw std::logic_error(
        "ACLNN pointwise reference input count is invalid");
  }
  for (std::size_t index = 0; index < test_case.inputs.size(); ++index) {
    const std::span<const float> explicit_input =
        ascend_case.explicit_inputs.empty()
            ? std::span<const float>{}
            : std::span<const float>(ascend_case.explicit_inputs[index]);
    flagdnn_inputs.push_back(make_input_buffer(test_case.inputs[index],
                                               index,
                                               test_case.input_domains[index],
                                               stream,
                                               explicit_input));
    reference_inputs.push_back(make_input_buffer(reference_plan.inputs[index],
                                                  index,
                                                  test_case.input_domains[index],
                                                  stream,
                                                  explicit_input));
  }
  std::unique_ptr<acl::DeviceBuffer> flagdnn_output =
      make_output_buffer(test_case.output, stream);
  std::unique_ptr<acl::DeviceBuffer> reference_output =
      make_output_buffer(reference_plan.output, stream);
  std::vector<flagdnnBinding_t> flagdnn_bindings;
  std::vector<flagdnnBinding_t> reference_bindings;
  flagdnn_bindings.reserve(test_case.inputs.size() + 1);
  reference_bindings.reserve(reference_plan.inputs.size() + 1);
  for (std::size_t index = 0; index < test_case.inputs.size(); ++index) {
    flagdnn_bindings.push_back(
        {test_case.inputs[index].uid,
         flagdnn_inputs[index].device->opaque_at(
             test_case.inputs[index].binding_byte_offset)});
    reference_bindings.push_back(
        {reference_plan.inputs[index].uid,
         reference_inputs[index].device->opaque_at(
             reference_plan.inputs[index].binding_byte_offset)});
  }
  flagdnn_bindings.push_back(
      {test_case.output.uid,
       flagdnn_output->opaque_at(test_case.output.binding_byte_offset)});
  reference_bindings.push_back(
      {reference_plan.output.uid,
       reference_output->opaque_at(reference_plan.output.binding_byte_offset)});

  flagdnn->prepare(flagdnn_bindings, stream.opaque());
  try {
    reference->prepare(reference_bindings, stream.opaque());
    if (requires_repeatable_reference_contract(test_case.mode)) {
      // Rebuild the direct ACLNN executor before using it.  Subsequent paired
      // executions below also verify the repeatable-executor contract.
      reference->prepare(reference_bindings, stream.opaque());
    }
  } catch (const AclnnPointwiseUnsupportedError& error) {
    try {
      stream.synchronize();
    } catch (...) {
    }
    reference.reset();
    flagdnn.reset();
    return "ACLNN status=" + std::to_string(error.status()) + " " +
           single_line_reason(error.what());
  }
  auto flagdnn_workspace =
      std::make_unique<acl::DeviceBuffer>(flagdnn->workspace_size());
  auto reference_workspace =
      std::make_unique<acl::DeviceBuffer>(reference->workspace_size());

  try {
    stream.synchronize();
    execute(*flagdnn, flagdnn_bindings, *flagdnn_workspace, stream);
    execute(*reference, reference_bindings, *reference_workspace, stream);
    if (requires_repeatable_reference_contract(test_case.mode)) {
      execute(*flagdnn, flagdnn_bindings, *flagdnn_workspace, stream);
      execute(*reference, reference_bindings, *reference_workspace, stream);
    }
    stream.synchronize();

    const std::vector<float> flagdnn_physical =
        read_output(*flagdnn_output, test_case.output, stream, "FlagDNN");
    const std::vector<float> reference_physical =
        read_output(*reference_output,
                    reference_plan.output,
                    stream,
                    "ACLNN");
    tensor_io::require_padding_unchanged(
        "FlagDNN", flagdnn_physical, test_case.output);
    tensor_io::require_padding_unchanged(
        "ACLNN", reference_physical, reference_plan.output);
    const std::vector<float> flagdnn_logical =
        tensor_io::gather(flagdnn_physical, test_case.output);
    const std::vector<float> reference_logical =
        tensor_io::gather(reference_physical, reference_plan.output);
    if (strict_mod) {
      const mod::SemanticInputs semantic_inputs{
          ascend_case.explicit_inputs[0], ascend_case.explicit_inputs[1]};
      const std::vector<float> host_reference = mod::host_reference(
          semantic_inputs, test_case.inputs.front().data_type);
      compare_provider_outputs(
          flagdnn_logical, host_reference, test_case, true);
      const mod::AclnnReferenceObservation observation =
          mod::classify_aclnn_reference(reference_logical, host_reference);
      std::cout << test_case.name
                << ": FlagDNN Graph/libtriton_jit vs std::fmod PASS "
                << "exact_reference=HOST_FMOD aclnn_matched="
                << observation.matched << " aclnn_differed="
                << observation.differed << " aclnn_nonfinite="
                << observation.nonfinite << '\n';
    } else {
      compare_provider_outputs(flagdnn_logical,
                               reference_logical,
                               test_case,
                               ascend_case.strict_zero_signbit);
      std::cout << test_case.name
                << ": FlagDNN Graph/libtriton_jit vs ACLNN "
                << operation_name(test_case.mode)
                << " PASS exact_reference=ACLNN_POINTWISE\n";
    }
  } catch (...) {
    try {
      stream.synchronize();
    } catch (...) {
    }
    reference.reset();
    flagdnn.reset();
    throw;
  }
  stream.synchronize();
  reference.reset();
  flagdnn.reset();
  return std::nullopt;
}

PointwiseTestCase make_binary_strided_broadcast_extension(
    flagdnnPointwiseMode_t mode) {
  PointwiseTestCase result;
  result.name = operation_name(mode) +
                "_ascend_strided_broadcast_alpha_fp32";
  result.mode = mode;
  result.inputs = {
      TestTensor{61000, FLAGDNN_DATA_FLOAT32, {2, 3, 4}, {31, 9, 1}},
      TestTensor{61001, FLAGDNN_DATA_FLOAT32, {1, 3, 1}, {7, 2, 1}},
  };
  result.output =
      TestTensor{61002, FLAGDNN_DATA_FLOAT32, {2, 3, 4}, {37, 11, 1}};
  if (mode == FLAGDNN_POINTWISE_LOGICAL_AND ||
      mode == FLAGDNN_POINTWISE_LOGICAL_OR) {
    const std::string logical_name =
        mode == FLAGDNN_POINTWISE_LOGICAL_AND ? "logical_and" : "logical_or";
    result.name = logical_name +
                  "_ascend_strided_broadcast_padding_bool";
    result.inputs = {
        TestTensor{
            61000, FLAGDNN_DATA_BOOLEAN, {2, 3, 4}, {31, 9, 2}, 32},
        TestTensor{
            61001, FLAGDNN_DATA_BOOLEAN, {1, 3, 1}, {7, 2, 1}, 64},
    };
    result.output = TestTensor{
        61002, FLAGDNN_DATA_BOOLEAN, {2, 3, 4}, {37, 11, 2}, 96};
    result.input_domains = {PointwiseInputDomain::kLogical,
                            PointwiseInputDomain::kLogical};
  } else if (mode == FLAGDNN_POINTWISE_SIGMOID_BWD) {
    result.name =
        "sigmoid_backward_ascend_equal_shape_strided_padding_fp32";
    result.inputs[1] =
        TestTensor{61001, FLAGDNN_DATA_FLOAT32, {2, 3, 4}, {35, 10, 1}};
    result.input_domains = {PointwiseInputDomain::kReal,
                            PointwiseInputDomain::kReal};
  } else if (mode == FLAGDNN_POINTWISE_DIV) {
    result.input_domains = {PointwiseInputDomain::kDivisor,
                            PointwiseInputDomain::kDivisor};
  } else if (mode == FLAGDNN_POINTWISE_MOD) {
    result.name =
        "mod_ascend_signed_strided_broadcast_padding_fp32";
    result.input_domains = {PointwiseInputDomain::kModuloSigned,
                            PointwiseInputDomain::kModuloSigned};
  } else if (mode == FLAGDNN_POINTWISE_POW) {
    result.name = "pow_ascend_safe_strided_broadcast_padding_fp32";
    result.input_domains = {PointwiseInputDomain::kPower,
                            PointwiseInputDomain::kPower};
  } else {
    result.input_domains = {PointwiseInputDomain::kReal,
                            PointwiseInputDomain::kReal};
  }
  result.alpha = mode == FLAGDNN_POINTWISE_SUB ? -0.75 : 1.0;
  result.absolute_tolerance =
      mode == FLAGDNN_POINTWISE_LOGICAL_AND ||
              mode == FLAGDNN_POINTWISE_LOGICAL_OR
          ? 0.0
          : (mode == FLAGDNN_POINTWISE_POW ? 2.0e-5 : 1.0e-6);
  result.relative_tolerance =
      result.absolute_tolerance;
  result.autotune = true;
  if (is_logical_mode(mode)) {
    require_logical_extension_binding_offsets(result);
  }
  validate_pointwise_case(result);
  return result;
}

std::string comparison_data_type_name(flagdnnDataType_t data_type) {
  switch (data_type) {
    case FLAGDNN_DATA_INT32:
      throw std::invalid_argument(
          "INT32 is not supported by this validation adapter");

    case FLAGDNN_DATA_FLOAT32:
      return "fp32";
    case FLAGDNN_DATA_FLOAT16:
      return "fp16";
    case FLAGDNN_DATA_BFLOAT16:
      return "bf16";
    case FLAGDNN_DATA_BOOLEAN:
    case FLAGDNN_DATA_FP8_E8M0:
    case FLAGDNN_DATA_FP8_E4M3:
    case FLAGDNN_DATA_FP8_E5M2:
      break;
  }
  throw std::invalid_argument(
      "Ascend comparison extension data type is unsupported");
}

AscendPointwiseCase make_comparison_semantic_extension(
    flagdnnPointwiseMode_t mode,
    flagdnnDataType_t data_type,
    std::int64_t uid) {
  if (!is_comparison_mode(mode)) {
    throw std::invalid_argument(
        "Ascend comparison extension received a different mode");
  }
  PointwiseTestCase result;
  result.name = operation_name(mode) + "_ascend_" +
                comparison_data_type_name(data_type) +
                "_special_strided_broadcast_offsets_bool_padding";
  result.mode = mode;
  result.inputs = {
      TestTensor{uid, data_type, {2, 3, 8}, {37, 11, 1}, 32},
      TestTensor{uid + 1, data_type, {1, 3, 8}, {29, 9, 1}, 64},
  };
  result.output = TestTensor{
      uid + 2, FLAGDNN_DATA_BOOLEAN, {2, 3, 8}, {43, 13, 1}, 96};
  result.input_domains = {PointwiseInputDomain::kComparison,
                          PointwiseInputDomain::kComparison};
  result.absolute_tolerance = 0.0;
  result.relative_tolerance = 0.0;
  result.autotune = true;
  validate_pointwise_case(result);

  const float infinity = std::numeric_limits<float>::infinity();
  const float quiet_nan = std::numeric_limits<float>::quiet_NaN();
  const std::array<float, 24> right = {
      -0.0F, 0.0F, infinity, -infinity, quiet_nan, 2.0F, 3.0F, -5.0F,
      1.0F, 4.0F, -2.0F, 8.0F, -9.0F, 0.5F, -0.5F, quiet_nan,
      infinity, -infinity, 7.0F, 7.0F, 6.0F, -6.0F, 11.0F, -11.0F,
  };
  const std::array<float, 24> first = {
      0.0F, -0.0F, infinity, -infinity, quiet_nan, 1.0F, 4.0F, -4.0F,
      1.0F, 3.0F, -3.0F, 9.0F, -10.0F, 0.5F, -0.25F, 1.0F,
      infinity, -infinity, 6.0F, 8.0F, 6.0F, -7.0F, 12.0F, -12.0F,
  };
  const std::array<float, 24> second = {
      -0.0F, 0.0F, -infinity, infinity, 0.0F, 3.0F, 2.0F, -6.0F,
      0.0F, 5.0F, -1.0F, 7.0F, -8.0F, 0.25F, -0.75F, quiet_nan,
      -infinity, infinity, 8.0F, 6.0F, 5.0F, -5.0F, 10.0F, -10.0F,
  };
  std::vector<float> left;
  left.reserve(first.size() + second.size());
  left.insert(left.end(), first.begin(), first.end());
  left.insert(left.end(), second.begin(), second.end());
  return {std::move(result),
          {std::move(left), std::vector<float>(right.begin(), right.end())},
          false};
}

std::string binary_select_data_type_name(flagdnnDataType_t data_type) {
  switch (data_type) {
    case FLAGDNN_DATA_INT32:
      throw std::invalid_argument(
          "INT32 is not supported by this validation adapter");

    case FLAGDNN_DATA_FLOAT32:
      return "fp32";
    case FLAGDNN_DATA_FLOAT16:
      return "fp16";
    case FLAGDNN_DATA_BFLOAT16:
      return "bf16";
    case FLAGDNN_DATA_BOOLEAN:
    case FLAGDNN_DATA_FP8_E8M0:
    case FLAGDNN_DATA_FP8_E4M3:
    case FLAGDNN_DATA_FP8_E5M2:
      break;
  }
  throw std::invalid_argument(
      "Ascend binary-select extension data type is unsupported");
}

AscendPointwiseCase make_binary_select_semantic_extension(
    flagdnnDataType_t data_type,
    std::int64_t uid) {
  PointwiseTestCase result;
  result.name = "binary_select_ascend_" +
                binary_select_data_type_name(data_type) +
                "_independent_broadcast_strided_offsets_bool_padding";
  result.mode = FLAGDNN_POINTWISE_BINARY_SELECT;
  result.inputs = {
      TestTensor{uid, data_type, {2, 1, 8}, {37, 17, 2}, 32},
      TestTensor{uid + 1, data_type, {1, 3, 1}, {19, 5, 1}, 64},
      TestTensor{
          uid + 2, FLAGDNN_DATA_BOOLEAN, {1, 3, 8}, {41, 11, 1}, 96},
  };
  result.output =
      TestTensor{uid + 3, data_type, {2, 3, 8}, {97, 29, 2}, 128};
  result.input_domains = {PointwiseInputDomain::kReal,
                          PointwiseInputDomain::kReal,
                          PointwiseInputDomain::kLogical};
  result.absolute_tolerance = 0.0;
  result.relative_tolerance = 0.0;
  result.autotune = true;
  require_logical_extension_binding_offsets(result);
  validate_pointwise_case(result);

  std::vector<float> self = {
      -0.0F, 1.0F,  -2.0F, 3.5F,  -4.0F, 5.0F,  -6.5F, 7.0F,
      8.0F,  -9.0F, 10.0F, -11.5F, 12.0F, -13.0F, 14.5F, -15.0F,
  };
  std::vector<float> other = {0.0F, -0.0F, 19.0F};
  std::vector<float> condition = {
      0.0F, 1.0F, 0.0F, 1.0F, 1.0F, 0.0F, 1.0F, 0.0F,
      1.0F, 0.0F, 1.0F, 0.0F, 0.0F, 1.0F, 0.0F, 1.0F,
      0.0F, 0.0F, 1.0F, 1.0F, 0.0F, 1.0F, 1.0F, 0.0F,
  };
  return {std::move(result),
          {std::move(self), std::move(other), std::move(condition)},
          true};
}

std::string mod_semantic_data_type_name(flagdnnDataType_t data_type) {
  switch (data_type) {
    case FLAGDNN_DATA_INT32:
      throw std::invalid_argument(
          "INT32 is not supported by this validation adapter");

    case FLAGDNN_DATA_FLOAT32:
      return "fp32";
    case FLAGDNN_DATA_FLOAT16:
      return "fp16";
    case FLAGDNN_DATA_BFLOAT16:
      return "bf16";
    case FLAGDNN_DATA_BOOLEAN:
    case FLAGDNN_DATA_FP8_E8M0:
    case FLAGDNN_DATA_FP8_E4M3:
    case FLAGDNN_DATA_FP8_E5M2:
      break;
  }
  throw std::invalid_argument(
      "Ascend Mod semantic case data type is unsupported");
}

AscendPointwiseCase make_mod_semantic_extension(
    flagdnnDataType_t data_type,
    std::int64_t uid) {
  static_assert(mod::kSentinelCount == 14);
  const mod::SemanticInputs inputs = mod::semantic_inputs(data_type);
  PointwiseTestCase result;
  result.name = "mod_ascend_semantic_strided_padding_" +
                mod_semantic_data_type_name(data_type);
  result.mode = FLAGDNN_POINTWISE_MOD;
  result.inputs = {
      TestTensor{uid, data_type, {2, 7}, {17, 2}},
      TestTensor{uid + 1, data_type, {2, 7}, {19, 2}},
  };
  result.output =
      TestTensor{uid + 2, data_type, {2, 7}, {23, 2}};
  // The values are supplied explicitly by the Ascend-only wrapper.  Keep the
  // public domain truthful for plan validation without coupling dispatch to a
  // case name or UID.
  result.input_domains = {PointwiseInputDomain::kModulo,
                          PointwiseInputDomain::kModulo};
  result.absolute_tolerance = 0.0;
  result.relative_tolerance = 0.0;
  result.autotune = true;
  validate_pointwise_case(result);
  return {std::move(result), {inputs.left, inputs.right}, true};
}

std::string suite_operation_name(const PointwiseTestCase& test_case) {
  if (test_case.mode == FLAGDNN_POINTWISE_RELU_FWD &&
      test_case.name.rfind("leaky_relu_", 0) == 0) {
    return "leaky_relu";
  }
  return operation_name(test_case.mode);
}

bool is_gelu_mode(flagdnnPointwiseMode_t mode) noexcept {
  return mode == FLAGDNN_POINTWISE_GELU_FWD ||
         mode == FLAGDNN_POINTWISE_GELU_APPROX_TANH_FWD;
}

bool is_special_math_mode(flagdnnPointwiseMode_t mode) noexcept {
  return mode == FLAGDNN_POINTWISE_SIN ||
         mode == FLAGDNN_POINTWISE_COS ||
         mode == FLAGDNN_POINTWISE_TAN ||
         mode == FLAGDNN_POINTWISE_ERF;
}

bool requires_repeatable_reference_contract(
    flagdnnPointwiseMode_t mode) noexcept {
  return is_gelu_mode(mode) || is_special_math_mode(mode) ||
         mode == FLAGDNN_POINTWISE_MOD ||
         mode == FLAGDNN_POINTWISE_POW ||
         mode == FLAGDNN_POINTWISE_SIGMOID_BWD ||
         is_comparison_mode(mode) ||
         mode == FLAGDNN_POINTWISE_LOGICAL_NOT ||
         mode == FLAGDNN_POINTWISE_LOGICAL_AND ||
         mode == FLAGDNN_POINTWISE_LOGICAL_OR ||
         mode == FLAGDNN_POINTWISE_BINARY_SELECT;
}

bool has_padded_unary_case(std::span<const PointwiseTestCase> cases) {
  return std::any_of(cases.begin(), cases.end(), [](const auto& test_case) {
    return test_case.inputs.size() == 1 &&
           (tensor_io::storage_element_count(test_case.inputs.front()) >
                tensor_io::element_count(test_case.inputs.front()) ||
            tensor_io::storage_element_count(test_case.output) >
                tensor_io::element_count(test_case.output));
  });
}

PointwiseTestCase make_unary_strided_extension(
    const PointwiseTestCase& prototype) {
  PointwiseTestCase result;
  result.name = suite_operation_name(prototype) +
                "_ascend_strided_padding_fp32";
  result.mode = prototype.mode;
  result.inputs = {
      TestTensor{62000, FLAGDNN_DATA_FLOAT32, {2, 3, 4}, {31, 9, 1}},
  };
  result.output =
      TestTensor{62001, FLAGDNN_DATA_FLOAT32, {2, 3, 4}, {37, 11, 1}};
  result.input_domains = {prototype.input_domains.front()};
  result.attributes = prototype.attributes;
  result.alpha = prototype.alpha;
  result.absolute_tolerance = prototype.absolute_tolerance;
  result.relative_tolerance = prototype.relative_tolerance;
  result.autotune = true;
  if (result.mode == FLAGDNN_POINTWISE_LOGICAL_NOT) {
    result.name = "logical_not_ascend_strided_padding_bool";
    result.inputs = {
        TestTensor{
            62000, FLAGDNN_DATA_BOOLEAN, {2, 3, 4}, {31, 9, 2}, 32},
    };
    result.output = TestTensor{
        62001, FLAGDNN_DATA_BOOLEAN, {2, 3, 4}, {37, 11, 2}, 64};
    result.input_domains = {PointwiseInputDomain::kLogical};
    result.absolute_tolerance = 0.0;
    result.relative_tolerance = 0.0;
    require_logical_extension_binding_offsets(result);
  } else if (is_gelu_mode(result.mode)) {
    // The exact and tanh GELU curves are intentionally close on the common
    // real-domain cases.  Exercise a wider domain with a strict FP32 oracle so
    // an accidental aclnnGelu/aclnnGeluV2 swap cannot pass validation.
    result.name = suite_operation_name(prototype) +
                  "_ascend_scaled_strided_padding_fp32";
    result.input_domains = {PointwiseInputDomain::kScaled};
    result.absolute_tolerance = gelu::kFlagdnnAbsoluteTolerance;
    result.relative_tolerance = gelu::kFlagdnnRelativeTolerance;
  }
  validate_pointwise_case(result);
  return result;
}

PointwiseTestCase make_full_relu_attribute_extension(
    const PointwiseTestCase& prototype) {
  PointwiseTestCase result;
  result.name = suite_operation_name(prototype) +
                "_ascend_full_attrs_strided_padding_fp32";
  result.mode = FLAGDNN_POINTWISE_RELU_FWD;
  result.inputs = {
      TestTensor{63000, FLAGDNN_DATA_FLOAT32, {2, 3, 4}, {31, 9, 1}},
  };
  result.output =
      TestTensor{63001, FLAGDNN_DATA_FLOAT32, {2, 3, 4}, {37, 11, 2}};
  result.input_domains = {PointwiseInputDomain::kReal};
  result.attributes = FLAGDNN_POINTWISE_ATTRIBUTES_INITIALIZER;
  result.attributes.flags =
      FLAGDNN_POINTWISE_ATTRIBUTE_RELU_LOWER_CLIP |
      FLAGDNN_POINTWISE_ATTRIBUTE_RELU_UPPER_CLIP |
      FLAGDNN_POINTWISE_ATTRIBUTE_RELU_LOWER_CLIP_SLOPE;
  result.attributes.relu_lower_clip = -0.25;
  result.attributes.relu_lower_clip_slope = 0.375;
  result.attributes.relu_upper_clip = 0.875;
  result.absolute_tolerance = 2.0e-5;
  result.relative_tolerance = 2.0e-5;
  result.autotune = true;
  validate_pointwise_case(result);
  return result;
}

PointwiseTestCase make_activation_attribute_extension(
    const PointwiseTestCase& prototype) {
  PointwiseTestCase result = make_unary_strided_extension(prototype);
  result.name = suite_operation_name(prototype) +
                "_ascend_nondefault_attrs_strided_padding_fp32";
  result.attributes = FLAGDNN_POINTWISE_ATTRIBUTES_INITIALIZER;
  if (prototype.mode == FLAGDNN_POINTWISE_ELU_FWD) {
    result.attributes.flags = FLAGDNN_POINTWISE_ATTRIBUTE_ELU_ALPHA;
    result.attributes.elu_alpha = 0.375;
  } else if (prototype.mode == FLAGDNN_POINTWISE_SOFTPLUS_FWD) {
    result.attributes.flags = FLAGDNN_POINTWISE_ATTRIBUTE_SOFTPLUS_BETA;
    result.attributes.softplus_beta = 1.75;
  } else {
    throw std::invalid_argument(
        "Ascend activation attribute extension mode is invalid");
  }
  validate_pointwise_case(result);
  return result;
}

}  // namespace

int run_pointwise_functional_test(
    int argc,
    char** argv,
    std::span<const PointwiseTestCase> cases,
    std::string_view suite_name) {
  // Exact integer and raw storage cases run in functional.ascend.<op>.dtype,
  // with an independent process and the same public case catalog.
  std::vector<PointwiseTestCase> supported_cases(cases.begin(), cases.end());
  std::erase_if(supported_cases, [](const PointwiseTestCase& test_case) {
    const bool logical = test_case.mode == FLAGDNN_POINTWISE_LOGICAL_NOT ||
                         test_case.mode == FLAGDNN_POINTWISE_LOGICAL_AND ||
                         test_case.mode == FLAGDNN_POINTWISE_LOGICAL_OR ||
                         test_case.mode == FLAGDNN_POINTWISE_BINARY_SELECT;
    return std::any_of(test_case.inputs.begin(), test_case.inputs.end(),
                       [logical](const TestTensor& tensor) {
                         const auto type = tensor.data_type;
                         return type != FLAGDNN_DATA_FLOAT32 &&
                                type != FLAGDNN_DATA_FLOAT16 &&
                                type != FLAGDNN_DATA_BFLOAT16 &&
                                !(logical && type == FLAGDNN_DATA_BOOLEAN);
                       });
  });
  cases = supported_cases;

  if (argc != 3) {
    std::cerr << "usage: " << argv[0]
              << " COMPILER_EXECUTABLE COMPILER_ENTRY\n";
    return 2;
  }
  try {
    std::cout << std::setprecision(9);
    if (cases.empty()) {
      throw std::invalid_argument("pointwise suite has no cases");
    }
    const flagdnnPointwiseMode_t mode = cases.front().mode;
    const std::size_t input_count = cases.front().inputs.size();
    (void)operation_name(mode);

    if (std::any_of(cases.begin(), cases.end(), [&](const auto& test_case) {
          return test_case.mode != mode ||
                 test_case.inputs.size() != input_count;
        })) {
      throw std::invalid_argument(
          "pointwise suite mixes operation modes or arities");
    }
    if (input_count != 1 && input_count != 2 && input_count != 3) {
      throw std::invalid_argument("pointwise suite arity is unsupported");
    }
    std::vector<AscendPointwiseCase> ascend_cases;
    ascend_cases.reserve(cases.size() + 4);
    for (const PointwiseTestCase& test_case : cases) {
      ascend_cases.push_back({test_case, {}, false});
    }
    if (input_count == 1) {
      if (mode == FLAGDNN_POINTWISE_RELU_FWD) {
        ascend_cases.push_back(
            {make_full_relu_attribute_extension(cases.front()), {}, false});
      } else if (mode == FLAGDNN_POINTWISE_ELU_FWD ||
                 mode == FLAGDNN_POINTWISE_SOFTPLUS_FWD) {
        ascend_cases.push_back(
            {make_activation_attribute_extension(cases.front()), {}, false});
      } else if (!has_padded_unary_case(cases)) {
        ascend_cases.push_back(
            {make_unary_strided_extension(cases.front()), {}, false});
      }
    } else if (mode == FLAGDNN_POINTWISE_BINARY_SELECT) {
      constexpr std::array<flagdnnDataType_t, 3> kDataTypes = {
          FLAGDNN_DATA_FLOAT32,
          FLAGDNN_DATA_FLOAT16,
          FLAGDNN_DATA_BFLOAT16,
      };
      std::int64_t uid = 69000;
      for (const flagdnnDataType_t data_type : kDataTypes) {
        ascend_cases.push_back(
            make_binary_select_semantic_extension(data_type, uid));
        uid += 4;
      }
    } else if (is_comparison_mode(mode)) {
      constexpr std::array<flagdnnDataType_t, 3> kDataTypes = {
          FLAGDNN_DATA_FLOAT32,
          FLAGDNN_DATA_FLOAT16,
          FLAGDNN_DATA_BFLOAT16,
      };
      std::int64_t uid = 68000;
      for (const flagdnnDataType_t data_type : kDataTypes) {
        ascend_cases.push_back(
            make_comparison_semantic_extension(mode, data_type, uid));
        uid += 3;
      }
    } else {
      ascend_cases.push_back(
          {make_binary_strided_broadcast_extension(mode), {}, false});
      if (mode == FLAGDNN_POINTWISE_MOD) {
        constexpr std::array<flagdnnDataType_t, 3> kDataTypes = {
            FLAGDNN_DATA_FLOAT32,
            FLAGDNN_DATA_FLOAT16,
            FLAGDNN_DATA_BFLOAT16,
        };
        std::int64_t uid = 67000;
        for (const flagdnnDataType_t data_type : kDataTypes) {
          ascend_cases.push_back(
              make_mod_semantic_extension(data_type, uid));
          uid += 3;
        }
      }
    }
    const char* filter = std::getenv("FLAGDNN_ASCEND_POINTWISE_CASE");
    const bool filtered = filter != nullptr && filter[0] != '\0';
    std::vector<const AscendPointwiseCase*> selected;
    selected.reserve(ascend_cases.size());
    for (const AscendPointwiseCase& ascend_case : ascend_cases) {
      if (filtered &&
          ascend_case.specification.name.find(filter) == std::string::npos) {
        continue;
      }
      selected.push_back(&ascend_case);
    }
    if (selected.empty()) {
      throw std::runtime_error(
          "FLAGDNN_ASCEND_POINTWISE_CASE matched no test cases");
    }
    acl::ExactReferenceCoverage coverage(selected.size());

    acl::DevelopmentEnvironment development(
        suite_operation_name(cases.front()) + "-functional");
    acl::AclRuntime runtime;
    const std::string soc = acl::soc_name();
    development.prepare_target(soc);
    std::cout << "ASCEND_VALIDATION_DEVELOPMENT_ROOT root="
              << development.root() << '\n';
    std::cout << "ACLNN_CAPABILITY_IDENTITY soc=" << soc
              << " cann_runtime=" << acl::runtime_package_version() << '\n';
    {
      acl::Stream stream;
      {
        flagdnn::Handle handle("ascend", 0);
        handle.set_compiler(
            argv[1], argv[2], development.graph_cache().string());
        for (const AscendPointwiseCase* ascend_case : selected) {
          const std::optional<std::string> skip =
              run_case(*ascend_case, handle, stream);
          const std::string& case_name = ascend_case->specification.name;
          if (skip.has_value()) {
            coverage.record_skip(case_name, *skip);
          } else {
            coverage.record_pass(case_name);
          }
        }
        stream.synchronize();
      }
    }
    runtime.finalize();
    development.cleanup();

    coverage.require_complete();
    for (const std::string& line : coverage.skip_lines()) {
      std::cout << line << '\n';
    }
    std::cout << coverage.summary() << '\n';
    std::cout << suite_name << ": PASS cases=" << coverage.passed()
              << " catalog_cases=" << ascend_cases.size() << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << suite_name << "_FAILED: " << error.what() << '\n';
    return 1;
  }
}

}  // namespace flagdnn::testing
