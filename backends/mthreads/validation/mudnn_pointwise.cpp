/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "backends/mthreads/validation/mudnn_pointwise.hpp"

#include "backends/mthreads/validation/musa_driver.hpp"
#include "backends/mthreads/validation/tensor_io.hpp"

#include <mudnn.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace flagdnn::validation::mthreads {
namespace {

musa::dnn::Tensor::Type mudnn_data_type(flagdnnDataType_t data_type) {
  switch (data_type) {
    case FLAGDNN_DATA_INT32:
      throw std::invalid_argument(
          "INT32 is not supported by this validation adapter");

    case FLAGDNN_DATA_FLOAT32:
      return musa::dnn::Tensor::Type::FLOAT;
    case FLAGDNN_DATA_FLOAT16:
      return musa::dnn::Tensor::Type::HALF;
    case FLAGDNN_DATA_BFLOAT16:
      return musa::dnn::Tensor::Type::BFLOAT16;
    case FLAGDNN_DATA_BOOLEAN:
      return musa::dnn::Tensor::Type::BOOL;
    case FLAGDNN_DATA_FP8_E8M0:
    case FLAGDNN_DATA_FP8_E4M3:
    case FLAGDNN_DATA_FP8_E5M2:
      break;
  }
  throw std::invalid_argument("muDNN pointwise tensor type is unsupported");
}

bool is_comparison_mode(flagdnnPointwiseMode_t mode) {
  return mode == FLAGDNN_POINTWISE_CMP_EQ ||
         mode == FLAGDNN_POINTWISE_CMP_NEQ ||
         mode == FLAGDNN_POINTWISE_CMP_GT ||
         mode == FLAGDNN_POINTWISE_CMP_GE ||
         mode == FLAGDNN_POINTWISE_CMP_LT ||
         mode == FLAGDNN_POINTWISE_CMP_LE;
}

bool is_logical_binary_mode(flagdnnPointwiseMode_t mode) {
  return mode == FLAGDNN_POINTWISE_LOGICAL_AND ||
         mode == FLAGDNN_POINTWISE_LOGICAL_OR;
}

bool is_binary_mode(flagdnnPointwiseMode_t mode) {
  switch (mode) {
    case FLAGDNN_POINTWISE_ADD:
    case FLAGDNN_POINTWISE_SUB:
    case FLAGDNN_POINTWISE_MUL:
    case FLAGDNN_POINTWISE_DIV:
    case FLAGDNN_POINTWISE_MIN:
    case FLAGDNN_POINTWISE_MAX:
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
    case FLAGDNN_POINTWISE_SIGMOID_BWD:
      return true;
    default:
      return false;
  }
}

bool is_unary_mode(flagdnnPointwiseMode_t mode) {
  switch (mode) {
    case FLAGDNN_POINTWISE_RELU_FWD:
    case FLAGDNN_POINTWISE_SQRT:
    case FLAGDNN_POINTWISE_ERF:
    case FLAGDNN_POINTWISE_IDENTITY:
    case FLAGDNN_POINTWISE_EXP:
    case FLAGDNN_POINTWISE_LOG:
    case FLAGDNN_POINTWISE_NEG:
    case FLAGDNN_POINTWISE_ABS:
    case FLAGDNN_POINTWISE_CEIL:
    case FLAGDNN_POINTWISE_COS:
    case FLAGDNN_POINTWISE_FLOOR:
    case FLAGDNN_POINTWISE_RSQRT:
    case FLAGDNN_POINTWISE_SIN:
    case FLAGDNN_POINTWISE_TAN:
    case FLAGDNN_POINTWISE_RECIPROCAL:
    case FLAGDNN_POINTWISE_LOGICAL_NOT:
    case FLAGDNN_POINTWISE_SIGMOID_FWD:
    case FLAGDNN_POINTWISE_TANH_FWD:
    case FLAGDNN_POINTWISE_ELU_FWD:
    case FLAGDNN_POINTWISE_GELU_FWD:
    case FLAGDNN_POINTWISE_SOFTPLUS_FWD:
    case FLAGDNN_POINTWISE_SWISH_FWD:
    case FLAGDNN_POINTWISE_GELU_APPROX_TANH_FWD:
      return true;
    default:
      return false;
  }
}

bool is_ternary_mode(flagdnnPointwiseMode_t mode) {
  return mode == FLAGDNN_POINTWISE_BINARY_SELECT;
}

musa::dnn::Unary::Mode mudnn_unary_mode(
    flagdnnPointwiseMode_t mode,
    const flagdnnPointwiseAttributes_t& attributes) {
  switch (mode) {
    case FLAGDNN_POINTWISE_RELU_FWD:
      if ((attributes.flags &
           FLAGDNN_POINTWISE_ATTRIBUTE_RELU_LOWER_CLIP_SLOPE) != 0U &&
          attributes.relu_lower_clip_slope != 0.0) {
        return musa::dnn::Unary::Mode::LEAKY_RELU;
      }
      if ((attributes.flags &
           (FLAGDNN_POINTWISE_ATTRIBUTE_RELU_LOWER_CLIP |
            FLAGDNN_POINTWISE_ATTRIBUTE_RELU_UPPER_CLIP)) != 0U) {
        return musa::dnn::Unary::Mode::CLIP;
      }
      return musa::dnn::Unary::Mode::RELU;
    case FLAGDNN_POINTWISE_SQRT:
      return musa::dnn::Unary::Mode::SQRT;
    case FLAGDNN_POINTWISE_ERF:
      return musa::dnn::Unary::Mode::ERF;
    case FLAGDNN_POINTWISE_IDENTITY:
      return musa::dnn::Unary::Mode::IDENTITY;
    case FLAGDNN_POINTWISE_EXP:
      return musa::dnn::Unary::Mode::EXP;
    case FLAGDNN_POINTWISE_LOG:
      return musa::dnn::Unary::Mode::LOG;
    case FLAGDNN_POINTWISE_NEG:
      return musa::dnn::Unary::Mode::MUL;
    case FLAGDNN_POINTWISE_ABS:
      return musa::dnn::Unary::Mode::ABS;
    case FLAGDNN_POINTWISE_CEIL:
      return musa::dnn::Unary::Mode::CEIL;
    case FLAGDNN_POINTWISE_COS:
      return musa::dnn::Unary::Mode::COS;
    case FLAGDNN_POINTWISE_FLOOR:
      return musa::dnn::Unary::Mode::FLOOR;
    case FLAGDNN_POINTWISE_RSQRT:
      return musa::dnn::Unary::Mode::RSQRT;
    case FLAGDNN_POINTWISE_SIN:
      return musa::dnn::Unary::Mode::SIN;
    case FLAGDNN_POINTWISE_TAN:
      return musa::dnn::Unary::Mode::TAN;
    case FLAGDNN_POINTWISE_RECIPROCAL:
      return musa::dnn::Unary::Mode::RECIPROCAL;
    case FLAGDNN_POINTWISE_LOGICAL_NOT:
      return musa::dnn::Unary::Mode::EQ;
    case FLAGDNN_POINTWISE_SIGMOID_FWD:
      return musa::dnn::Unary::Mode::SIGMOID;
    case FLAGDNN_POINTWISE_TANH_FWD:
      return musa::dnn::Unary::Mode::TANH;
    case FLAGDNN_POINTWISE_ELU_FWD:
      return musa::dnn::Unary::Mode::ELU;
    case FLAGDNN_POINTWISE_GELU_FWD:
      return musa::dnn::Unary::Mode::GELU;
    case FLAGDNN_POINTWISE_SOFTPLUS_FWD:
      return musa::dnn::Unary::Mode::SOFTPLUS;
    case FLAGDNN_POINTWISE_SWISH_FWD:
      return musa::dnn::Unary::Mode::SWISH;
    case FLAGDNN_POINTWISE_GELU_APPROX_TANH_FWD:
      return musa::dnn::Unary::Mode::GELU_TANH;
    default:
      throw std::invalid_argument(
          "muDNN unary pointwise mode is unsupported");
  }
}

musa::dnn::Binary::Mode mudnn_binary_mode(
    flagdnnPointwiseMode_t mode, bool scaled) {
  switch (mode) {
    case FLAGDNN_POINTWISE_ADD:
      return scaled ? musa::dnn::Binary::Mode::ADD_ALPHA
                    : musa::dnn::Binary::Mode::ADD;
    case FLAGDNN_POINTWISE_SUB:
      return scaled ? musa::dnn::Binary::Mode::SUB_ALPHA
                    : musa::dnn::Binary::Mode::SUB;
    case FLAGDNN_POINTWISE_MUL:
      return musa::dnn::Binary::Mode::MUL;
    case FLAGDNN_POINTWISE_DIV:
      return musa::dnn::Binary::Mode::DIV;
    case FLAGDNN_POINTWISE_MIN:
      return musa::dnn::Binary::Mode::MIN;
    case FLAGDNN_POINTWISE_MAX:
      return musa::dnn::Binary::Mode::MAX;
    case FLAGDNN_POINTWISE_MOD:
      return musa::dnn::Binary::Mode::TRUNCATEMOD;
    case FLAGDNN_POINTWISE_POW:
      return musa::dnn::Binary::Mode::POW;
    case FLAGDNN_POINTWISE_CMP_EQ:
      return musa::dnn::Binary::Mode::EQ;
    case FLAGDNN_POINTWISE_CMP_NEQ:
      return musa::dnn::Binary::Mode::NE;
    case FLAGDNN_POINTWISE_CMP_GT:
      return musa::dnn::Binary::Mode::GT;
    case FLAGDNN_POINTWISE_CMP_GE:
      return musa::dnn::Binary::Mode::GE;
    case FLAGDNN_POINTWISE_CMP_LT:
      return musa::dnn::Binary::Mode::LT;
    case FLAGDNN_POINTWISE_CMP_LE:
      return musa::dnn::Binary::Mode::LE;
    case FLAGDNN_POINTWISE_LOGICAL_AND:
      return musa::dnn::Binary::Mode::LOGICAL_AND;
    case FLAGDNN_POINTWISE_LOGICAL_OR:
      return musa::dnn::Binary::Mode::LOGICAL_OR;
    case FLAGDNN_POINTWISE_SIGMOID_BWD:
      return musa::dnn::Binary::Mode::SIGMOID_BW;
    default:
      throw std::invalid_argument(
          "muDNN binary pointwise mode is unsupported");
  }
}

void validate_tensor(const TensorDescriptor& tensor,
                     std::string_view name) {
  if (tensor.uid <= 0 || tensor.dimensions.empty() ||
      tensor.dimensions.size() != tensor.strides.size()) {
    throw std::invalid_argument(
        std::string(name) + " muDNN tensor descriptor is invalid");
  }
  for (std::size_t axis = 0; axis < tensor.dimensions.size(); ++axis) {
    if (tensor.dimensions[axis] <= 0 || tensor.strides[axis] <= 0) {
      throw std::invalid_argument(
          std::string(name) + " muDNN tensor layout is invalid");
    }
  }
  static_cast<void>(mudnn_data_type(tensor.data_type));
}

bool broadcasts_to(const TensorDescriptor& input,
                   const TensorDescriptor& output) {
  if (input.dimensions.size() > output.dimensions.size()) {
    return false;
  }
  const std::size_t leading =
      output.dimensions.size() - input.dimensions.size();
  for (std::size_t axis = 0; axis < input.dimensions.size(); ++axis) {
    const std::int64_t input_dimension = input.dimensions[axis];
    const std::int64_t output_dimension =
        output.dimensions[leading + axis];
    if (input_dimension != 1 && input_dimension != output_dimension) {
      return false;
    }
  }
  return true;
}

std::vector<std::int64_t> broadcast_shape(
    std::span<const TensorDescriptor> inputs) {
  std::size_t rank = 0;
  for (const TensorDescriptor& input : inputs) {
    rank = std::max(rank, input.dimensions.size());
  }
  std::vector<std::int64_t> result(rank, 1);
  for (const TensorDescriptor& input : inputs) {
    const std::size_t leading = rank - input.dimensions.size();
    for (std::size_t axis = 0; axis < input.dimensions.size(); ++axis) {
      const std::size_t output_axis = leading + axis;
      const std::int64_t dimension = input.dimensions[axis];
      if (dimension != result[output_axis] && dimension != 1 &&
          result[output_axis] != 1) {
        throw std::invalid_argument(
            "muDNN pointwise inputs are not broadcast-compatible");
      }
      result[output_axis] = std::max(result[output_axis], dimension);
    }
  }
  return result;
}

void validate_descriptor(const MudnnPointwiseDescriptor& descriptor) {
  const bool binary = is_binary_mode(descriptor.mode);
  const bool unary = is_unary_mode(descriptor.mode);
  const bool ternary = is_ternary_mode(descriptor.mode);
  const std::size_t expected_inputs = ternary ? 3U : (binary ? 2U : 1U);
  if ((!binary && !unary && !ternary) ||
      descriptor.inputs.size() != expected_inputs) {
    throw std::invalid_argument(
        "muDNN pointwise descriptor arity or mode is invalid");
  }
  validate_tensor(
      descriptor.inputs[0], unary ? "input" : (ternary ? "a" : "left"));
  if (binary || ternary) {
    validate_tensor(descriptor.inputs[1], "right");
  }
  if (ternary) {
    validate_tensor(descriptor.inputs[2], "predicate");
  }
  validate_tensor(descriptor.output, "output");
  std::unordered_set<std::int64_t> uids;
  uids.insert(descriptor.output.uid);
  for (const TensorDescriptor& input : descriptor.inputs) {
    if (!uids.insert(input.uid).second ||
        !broadcasts_to(input, descriptor.output)) {
      throw std::invalid_argument(
          "muDNN pointwise tensor UIDs or broadcast shapes are invalid");
    }
  }
  if ((binary || ternary) &&
      descriptor.inputs[0].data_type != descriptor.inputs[1].data_type) {
    throw std::invalid_argument(
        "muDNN pointwise input data types must match");
  }
  const flagdnnDataType_t input_type = descriptor.inputs[0].data_type;
  if (unary) {
    if (descriptor.inputs[0].dimensions != descriptor.output.dimensions ||
        (descriptor.mode == FLAGDNN_POINTWISE_LOGICAL_NOT
             ? input_type != FLAGDNN_DATA_BOOLEAN ||
                   descriptor.output.data_type != FLAGDNN_DATA_BOOLEAN
             : input_type == FLAGDNN_DATA_BOOLEAN ||
                   descriptor.output.data_type != input_type)) {
      throw std::invalid_argument(
          "muDNN unary pointwise tensor types or shapes are invalid");
    }
  } else if (ternary) {
    if (input_type == FLAGDNN_DATA_BOOLEAN ||
        descriptor.inputs[2].data_type != FLAGDNN_DATA_BOOLEAN ||
        descriptor.output.data_type != input_type ||
        broadcast_shape(descriptor.inputs) != descriptor.output.dimensions) {
      throw std::invalid_argument(
          "muDNN binary_select tensor types or shapes are invalid");
    }
  } else if (is_comparison_mode(descriptor.mode)) {
    if (input_type == FLAGDNN_DATA_BOOLEAN ||
        descriptor.output.data_type != FLAGDNN_DATA_BOOLEAN) {
      throw std::invalid_argument(
          "muDNN comparison tensor types are invalid");
    }
  } else if (is_logical_binary_mode(descriptor.mode)) {
    if (input_type != FLAGDNN_DATA_BOOLEAN ||
        descriptor.output.data_type != FLAGDNN_DATA_BOOLEAN) {
      throw std::invalid_argument(
          "muDNN logical tensor types are invalid");
    }
  } else if (input_type == FLAGDNN_DATA_BOOLEAN ||
             descriptor.output.data_type != input_type) {
    throw std::invalid_argument(
        "muDNN numeric pointwise tensor types are invalid");
  }
  if (descriptor.mode == FLAGDNN_POINTWISE_SIGMOID_BWD &&
      (descriptor.inputs[0].dimensions != descriptor.inputs[1].dimensions ||
       descriptor.output.dimensions != descriptor.inputs[0].dimensions)) {
    throw std::invalid_argument(
        "muDNN sigmoid backward tensors must have equal shapes");
  }
  if (!std::isfinite(descriptor.alpha) ||
      (descriptor.mode != FLAGDNN_POINTWISE_ADD &&
       descriptor.mode != FLAGDNN_POINTWISE_SUB &&
       descriptor.alpha != 1.0)) {
    throw std::invalid_argument("muDNN pointwise alpha is invalid");
  }
  const flagdnnPointwiseAttributes_t& attributes = descriptor.attributes;
  if (attributes.struct_size != sizeof(flagdnnPointwiseAttributes_t) ||
      attributes.version != FLAGDNN_POINTWISE_ATTRIBUTES_VERSION ||
      (attributes.flags & ~FLAGDNN_POINTWISE_ATTRIBUTE_FLAGS_ALL) != 0U ||
      !std::isfinite(attributes.relu_lower_clip) ||
      !std::isfinite(attributes.relu_upper_clip) ||
      !std::isfinite(attributes.relu_lower_clip_slope) ||
      !std::isfinite(attributes.swish_beta) ||
      !std::isfinite(attributes.elu_alpha) ||
      !std::isfinite(attributes.softplus_beta) ||
      attributes.softplus_beta <= 0.0) {
    throw std::invalid_argument(
        "muDNN pointwise attribute ABI or values are invalid");
  }
  std::uint64_t allowed_flags = 0;
  if (descriptor.mode == FLAGDNN_POINTWISE_RELU_FWD) {
    allowed_flags =
        FLAGDNN_POINTWISE_ATTRIBUTE_RELU_LOWER_CLIP |
        FLAGDNN_POINTWISE_ATTRIBUTE_RELU_UPPER_CLIP |
        FLAGDNN_POINTWISE_ATTRIBUTE_RELU_LOWER_CLIP_SLOPE;
  } else if (descriptor.mode == FLAGDNN_POINTWISE_SWISH_FWD) {
    allowed_flags = FLAGDNN_POINTWISE_ATTRIBUTE_SWISH_BETA;
  } else if (descriptor.mode == FLAGDNN_POINTWISE_ELU_FWD) {
    allowed_flags = FLAGDNN_POINTWISE_ATTRIBUTE_ELU_ALPHA;
  } else if (descriptor.mode == FLAGDNN_POINTWISE_SOFTPLUS_FWD) {
    allowed_flags = FLAGDNN_POINTWISE_ATTRIBUTE_SOFTPLUS_BETA;
  }
  if ((attributes.flags & ~allowed_flags) != 0U ||
      ((attributes.flags & FLAGDNN_POINTWISE_ATTRIBUTE_RELU_UPPER_CLIP) !=
           0U &&
       attributes.relu_upper_clip < attributes.relu_lower_clip)) {
    throw std::invalid_argument(
        "muDNN pointwise attributes do not match the selected mode");
  }
}

void* binding_pointer(
    const TensorDescriptor& tensor,
    const std::unordered_map<std::int64_t, void*>& bindings) {
  const auto found = bindings.find(tensor.uid);
  if (found == bindings.end()) {
    throw std::invalid_argument("muDNN pointwise binding UID is missing");
  }
  return found->second;
}

void configure_tensor(musa::dnn::Tensor& tensor,
                      const TensorDescriptor& descriptor,
                      void* pointer) {
  check_mudnn(tensor.SetAddr(pointer), "muDNN Tensor::SetAddr");
  check_mudnn(
      tensor.SetType(mudnn_data_type(descriptor.data_type)),
      "muDNN Tensor::SetType");
  check_mudnn(
      tensor.SetNdInfo(
          static_cast<std::int64_t>(descriptor.dimensions.size()),
          descriptor.dimensions.data(),
          descriptor.strides.data()),
      "muDNN Tensor::SetNdInfo");
}

}  // namespace

struct MudnnPointwiseOperation::Impl {
  explicit Impl(MudnnPointwiseDescriptor value)
      : descriptor(std::move(value)),
        handle(0) {
    validate_descriptor(descriptor);
    if (is_binary_mode(descriptor.mode)) {
      binary = std::make_unique<musa::dnn::Binary>();
      const bool scaled = descriptor.alpha != 1.0;
      check_mudnn(
          binary->SetMode(mudnn_binary_mode(descriptor.mode, scaled)),
          "muDNN Binary::SetMode(pointwise)");
      if (scaled) {
        check_mudnn(
            binary->SetAlpha(descriptor.alpha),
            "muDNN Binary::SetAlpha(pointwise)");
      }
    } else if (is_unary_mode(descriptor.mode)) {
      unary = std::make_unique<musa::dnn::Unary>();
      check_mudnn(
          unary->SetMode(
              mudnn_unary_mode(descriptor.mode, descriptor.attributes)),
          "muDNN Unary::SetMode(pointwise)");
      const flagdnnPointwiseAttributes_t& attributes =
          descriptor.attributes;
      if (descriptor.mode == FLAGDNN_POINTWISE_NEG) {
        check_mudnn(
            unary->SetAlpha(-1.0), "muDNN Unary::SetAlpha(neg)");
      } else if (descriptor.mode == FLAGDNN_POINTWISE_LOGICAL_NOT) {
        check_mudnn(
            unary->SetAlpha(std::int64_t{0}),
            "muDNN Unary::SetAlpha(logical_not)");
      } else if (descriptor.mode == FLAGDNN_POINTWISE_RELU_FWD) {
        if ((attributes.flags &
             FLAGDNN_POINTWISE_ATTRIBUTE_RELU_LOWER_CLIP_SLOPE) != 0U &&
            attributes.relu_lower_clip_slope != 0.0) {
          check_mudnn(
              unary->SetAlpha(attributes.relu_lower_clip_slope),
              "muDNN Unary::SetAlpha(leaky_relu)");
        } else if ((attributes.flags &
                    (FLAGDNN_POINTWISE_ATTRIBUTE_RELU_LOWER_CLIP |
                     FLAGDNN_POINTWISE_ATTRIBUTE_RELU_UPPER_CLIP)) != 0U) {
          const double lower =
              (attributes.flags &
               FLAGDNN_POINTWISE_ATTRIBUTE_RELU_LOWER_CLIP) != 0U
                  ? attributes.relu_lower_clip
                  : 0.0;
          const double upper =
              (attributes.flags &
               FLAGDNN_POINTWISE_ATTRIBUTE_RELU_UPPER_CLIP) != 0U
                  ? attributes.relu_upper_clip
                  : static_cast<double>(
                        std::numeric_limits<float>::max());
          check_mudnn(
              unary->SetAlpha(lower), "muDNN Unary::SetAlpha(clip)");
          check_mudnn(
              unary->SetBeta(upper), "muDNN Unary::SetBeta(clip)");
        }
      } else if (descriptor.mode == FLAGDNN_POINTWISE_ELU_FWD) {
        check_mudnn(
            unary->SetAlpha(attributes.elu_alpha),
            "muDNN Unary::SetAlpha(elu)");
      } else if (descriptor.mode == FLAGDNN_POINTWISE_SOFTPLUS_FWD) {
        check_mudnn(
            unary->SetAlpha(attributes.softplus_beta),
            "muDNN Unary::SetAlpha(softplus)");
        check_mudnn(
            unary->SetBeta(20.0),
            "muDNN Unary::SetBeta(softplus threshold)");
      } else if (descriptor.mode == FLAGDNN_POINTWISE_SWISH_FWD) {
        check_mudnn(
          unary->SetAlpha(attributes.swish_beta),
          "muDNN Unary::SetAlpha(swish)");
      }
    } else {
      ternary = std::make_unique<musa::dnn::Ternary>();
      check_mudnn(
          ternary->SetMode(musa::dnn::Ternary::Mode::SELECT),
          "muDNN Ternary::SetMode(binary_select)");
    }
    if (descriptor.mode == FLAGDNN_POINTWISE_SIGMOID_BWD) {
      sigmoid_forward = std::make_unique<musa::dnn::Unary>();
      check_mudnn(
          sigmoid_forward->SetMode(musa::dnn::Unary::Mode::SIGMOID),
          "muDNN Unary::SetMode(sigmoid_backward bridge)");
      sigmoid_tensor = descriptor.inputs[1];
      sigmoid_tensor.binding_byte_offset = 0;
      const std::size_t storage =
          tensor_io::storage_element_count(sigmoid_tensor);
      const std::size_t width =
          tensor_io::data_type_size(sigmoid_tensor.data_type);
      if (storage > std::numeric_limits<std::size_t>::max() / width) {
        throw std::overflow_error(
            "muDNN sigmoid backward workspace size overflows");
      }
      workspace_bytes = storage * width;
    }
  }

  MudnnPointwiseDescriptor descriptor;
  musa::dnn::Handle handle;
  std::unique_ptr<musa::dnn::Binary> binary;
  std::unique_ptr<musa::dnn::Unary> unary;
  std::unique_ptr<musa::dnn::Ternary> ternary;
  std::unique_ptr<musa::dnn::Unary> sigmoid_forward;
  TensorDescriptor sigmoid_tensor;
  std::size_t workspace_bytes = 0;
};

MudnnPointwiseOperation::MudnnPointwiseOperation(
    MudnnPointwiseDescriptor descriptor)
    : implementation_(std::make_unique<Impl>(std::move(descriptor))) {}

MudnnPointwiseOperation::~MudnnPointwiseOperation() = default;

std::size_t MudnnPointwiseOperation::workspace_size() const noexcept {
  return implementation_->workspace_bytes;
}

void MudnnPointwiseOperation::execute(
    std::span<const flagdnnBinding_t> raw_bindings,
    void* workspace,
    std::size_t workspace_size,
    flagdnnStream_t stream) {
  Impl& state = *implementation_;
  if (workspace_size != state.workspace_bytes ||
      (state.workspace_bytes != 0 && workspace == nullptr) ||
      stream == nullptr ||
      raw_bindings.size() != state.descriptor.inputs.size() + 1) {
    throw std::invalid_argument(
        "muDNN pointwise execute arguments are invalid");
  }
  std::unordered_map<std::int64_t, void*> bindings;
  bindings.reserve(raw_bindings.size());
  for (const flagdnnBinding_t& binding : raw_bindings) {
    if (binding.device_pointer == nullptr ||
        !bindings.emplace(binding.uid, binding.device_pointer).second) {
      throw std::invalid_argument("muDNN pointwise binding is invalid");
    }
  }

  const musaStream_t musa_stream = reinterpret_cast<musaStream_t>(stream);
  check_mudnn(
      state.handle.SetStream(musa_stream), "muDNN Handle::SetStream");
  if (state.handle.GetStream() != musa_stream) {
    throw std::runtime_error("muDNN did not retain the caller stream");
  }

  musa::dnn::Tensor left;
  musa::dnn::Tensor output;
  configure_tensor(
      left,
      state.descriptor.inputs[0],
      binding_pointer(state.descriptor.inputs[0], bindings));
  configure_tensor(
      output,
      state.descriptor.output,
      binding_pointer(state.descriptor.output, bindings));
  if (is_unary_mode(state.descriptor.mode)) {
    check_mudnn(
        state.unary->Run(state.handle, output, left),
        "muDNN Unary::Run(pointwise)");
    return;
  }

  musa::dnn::Tensor right;
  configure_tensor(
      right,
      state.descriptor.inputs[1],
      binding_pointer(state.descriptor.inputs[1], bindings));
  if (is_ternary_mode(state.descriptor.mode)) {
    musa::dnn::Tensor predicate;
    configure_tensor(
        predicate,
        state.descriptor.inputs[2],
        binding_pointer(state.descriptor.inputs[2], bindings));
    check_mudnn(
        state.ternary->Run(
            state.handle, output, predicate, left, right),
        "muDNN Ternary::Run(binary_select)");
    return;
  }
  if (state.descriptor.mode == FLAGDNN_POINTWISE_SIGMOID_BWD) {
    musa::dnn::Tensor sigmoid;
    configure_tensor(sigmoid, state.sigmoid_tensor, workspace);
    check_mudnn(
        state.sigmoid_forward->Run(state.handle, sigmoid, right),
        "muDNN Unary::Run(sigmoid_backward bridge)");
    check_mudnn(
        state.binary->Run(state.handle, output, left, sigmoid),
        "muDNN Binary::Run(sigmoid_backward bridge)");
    return;
  }
  check_mudnn(
      state.binary->Run(state.handle, output, left, right),
      "muDNN Binary::Run(pointwise)");
}

}  // namespace flagdnn::validation::mthreads
