/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "validation/aclnn_unary_runtime.hpp"

#include "validation/acl_runtime.hpp"
#include "validation/tensor_io.hpp"

#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include <aclnnop/aclnn_abs.h>
#include <aclnnop/aclnn_add.h>
#include <aclnnop/aclnn_ceil.h>
#include <aclnnop/aclnn_clamp.h>
#include <aclnnop/aclnn_cos.h>
#include <aclnnop/aclnn_erf.h>
#include <aclnnop/aclnn_exp.h>
#include <aclnnop/aclnn_floor.h>
#include <aclnnop/aclnn_log.h>
#include <aclnnop/aclnn_logical_not.h>
#include <aclnnop/aclnn_mul.h>
#include <aclnnop/aclnn_neg.h>
#include <aclnnop/aclnn_reciprocal.h>
#include <aclnnop/aclnn_rsqrt.h>
#include <aclnnop/aclnn_sin.h>
#include <aclnnop/aclnn_sqrt.h>
#include <aclnnop/aclnn_sub.h>
#include <aclnnop/aclnn_tan.h>
#include <aclnnop/aclnn_tanh.h>

#ifndef FLAGDNN_ASCEND_VALIDATION_ENABLE_OPAPI_NN
#define FLAGDNN_ASCEND_VALIDATION_ENABLE_OPAPI_NN 0
#endif

#if FLAGDNN_ASCEND_VALIDATION_ENABLE_OPAPI_NN
#include <aclnnop/aclnn_elu.h>
#include <aclnnop/aclnn_gelu.h>
#include <aclnnop/aclnn_gelu_v2.h>
#include <aclnnop/aclnn_sigmoid.h>
#include <aclnnop/aclnn_softplus.h>
#include <aclnnop/aclnn_swish.h>
#endif

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace flagdnn::validation::ascend {
namespace {

namespace tensor_io = flagdnn::validation::ascend::tensor_io;

constexpr std::uint64_t kReluAttributeFlags =
    FLAGDNN_POINTWISE_ATTRIBUTE_RELU_LOWER_CLIP |
    FLAGDNN_POINTWISE_ATTRIBUTE_RELU_UPPER_CLIP |
    FLAGDNN_POINTWISE_ATTRIBUTE_RELU_LOWER_CLIP_SLOPE;

enum class StageKind {
  kNeg,
  kAbs,
  kCeil,
  kCos,
  kErf,
  kFloor,
  kReciprocal,
  kSqrt,
  kRsqrt,
  kSin,
  kTan,
  kExp,
  kLog,
  kTanh,
  kLogicalNot,
#if FLAGDNN_ASCEND_VALIDATION_ENABLE_OPAPI_NN
  kSigmoid,
  kElu,
  kGelu,
  kGeluV2,
  kSoftplus,
  kSwish,
#endif
  kClampMin,
  kClampMax,
  kMuls,
  kAdd,
  kAdds,
  kSubs,
};

std::string stage_name(StageKind kind) {
  switch (kind) {
    case StageKind::kNeg:
      return "aclnnNeg";
    case StageKind::kAbs:
      return "aclnnAbs";
    case StageKind::kCeil:
      return "aclnnCeil";
    case StageKind::kCos:
      return "aclnnCos";
    case StageKind::kErf:
      return "aclnnErf";
    case StageKind::kFloor:
      return "aclnnFloor";
    case StageKind::kReciprocal:
      return "aclnnReciprocal";
    case StageKind::kSqrt:
      return "aclnnSqrt";
    case StageKind::kRsqrt:
      return "aclnnRsqrt";
    case StageKind::kSin:
      return "aclnnSin";
    case StageKind::kTan:
      return "aclnnTan";
    case StageKind::kExp:
      return "aclnnExp";
    case StageKind::kLog:
      return "aclnnLog";
    case StageKind::kTanh:
      return "aclnnTanh";
    case StageKind::kLogicalNot:
      return "aclnnLogicalNot";
#if FLAGDNN_ASCEND_VALIDATION_ENABLE_OPAPI_NN
    case StageKind::kSigmoid:
      return "aclnnSigmoid";
    case StageKind::kElu:
      return "aclnnElu";
    case StageKind::kGelu:
      return "aclnnGelu";
    case StageKind::kGeluV2:
      return "aclnnGeluV2";
    case StageKind::kSoftplus:
      return "aclnnSoftplus";
    case StageKind::kSwish:
      return "aclnnSwish";
#endif
    case StageKind::kClampMin:
      return "aclnnClampMin";
    case StageKind::kClampMax:
      return "aclnnClampMax";
    case StageKind::kMuls:
      return "aclnnMuls";
    case StageKind::kAdd:
      return "aclnnAdd";
    case StageKind::kAdds:
      return "aclnnAdds";
    case StageKind::kSubs:
      return "aclnnSubs";
  }
  throw std::logic_error("ACLNN unary stage is invalid");
}

std::string aclnn_error_message(aclnnStatus status,
                                std::string_view operation) {
  std::string result(operation);
  result += " failed with ACLNN status ";
  result += std::to_string(status);
  const char* recent = aclGetRecentErrMsg();
  if (recent != nullptr && recent[0] != '\0') {
    result += ": ";
    result += recent;
  }
  return result;
}

bool explicitly_unsupported(aclnnStatus status, std::string_view message) {
  if (status == ACL_ERROR_UNSUPPORTED_DATA_TYPE ||
      status == ACL_ERROR_OP_UNSUPPORTED_DYNAMIC ||
      status == ACL_ERROR_API_NOT_SUPPORT ||
      status == ACL_ERROR_FEATURE_UNSUPPORTED) {
    return true;
  }
  std::string lowercase(message);
  std::transform(lowercase.begin(),
                 lowercase.end(),
                 lowercase.begin(),
                 [](unsigned char character) {
                   return static_cast<char>(std::tolower(character));
                 });
  return lowercase.find("not support") != std::string::npos ||
         lowercase.find("unsupported") != std::string::npos;
}

void log_destroy_status(aclnnStatus status, std::string_view operation) {
  if (status != 0) {
    std::cerr << "Ascend ACLNN unary cleanup: "
              << aclnn_error_message(status, operation) << '\n';
  }
}

aclDataType acl_data_type(flagdnnDataType_t data_type) {
  switch (data_type) {
    case FLAGDNN_DATA_INT32:
      throw std::invalid_argument(
          "INT32 is not supported by this validation adapter");

    case FLAGDNN_DATA_FLOAT32:
      return ACL_FLOAT;
    case FLAGDNN_DATA_FLOAT16:
      return ACL_FLOAT16;
    case FLAGDNN_DATA_BFLOAT16:
      return ACL_BF16;
    case FLAGDNN_DATA_BOOLEAN:
      return ACL_BOOL;
    case FLAGDNN_DATA_FP8_E8M0:
    case FLAGDNN_DATA_FP8_E4M3:
    case FLAGDNN_DATA_FP8_E5M2:
      break;
  }
  throw std::invalid_argument(
      "ACLNN unary supports FP32, FP16, BF16, and BOOLEAN only");
}

std::vector<std::int64_t> contiguous_strides(
    const std::vector<std::int64_t>& dimensions) {
  std::vector<std::int64_t> result(dimensions.size());
  std::int64_t stride = 1;
  for (std::size_t axis = dimensions.size(); axis != 0; --axis) {
    const std::int64_t dimension = dimensions[axis - 1];
    if (dimension <= 0 ||
        stride > std::numeric_limits<std::int64_t>::max() / dimension) {
      throw std::overflow_error("ACLNN unary contiguous stride overflows");
    }
    result[axis - 1] = stride;
    stride *= dimension;
  }
  return result;
}

std::size_t storage_element_count(const AclnnUnaryTensor& tensor) {
  if (tensor.dimensions.empty() ||
      tensor.dimensions.size() != tensor.strides.size()) {
    throw std::invalid_argument("ACLNN unary tensor metadata is invalid");
  }
  std::size_t maximum_offset = 0;
  for (std::size_t axis = 0; axis < tensor.dimensions.size(); ++axis) {
    if (tensor.dimensions[axis] <= 0 || tensor.strides[axis] <= 0) {
      throw std::invalid_argument(
          "ACLNN unary tensor dimensions and strides must be positive");
    }
    const std::size_t extent =
        static_cast<std::size_t>(tensor.dimensions[axis] - 1);
    const std::size_t stride =
        static_cast<std::size_t>(tensor.strides[axis]);
    if (extent != 0 &&
        stride > (std::numeric_limits<std::size_t>::max() - maximum_offset) /
                     extent) {
      throw std::overflow_error("ACLNN unary tensor storage overflows");
    }
    maximum_offset += extent * stride;
  }
  return maximum_offset + 1;
}

std::size_t encoded_byte_count(const AclnnUnaryTensor& tensor) {
  return tensor_io::checked_multiply(storage_element_count(tensor),
                                     tensor_io::data_type_size(tensor.data_type),
                                     "ACLNN unary tensor byte count");
}

void validate_tensor(const AclnnUnaryTensor& tensor) {
  (void)acl_data_type(tensor.data_type);
  (void)encoded_byte_count(tensor);
}

aclTensor* create_tensor(const AclnnUnaryTensor& tensor, void* pointer) {
  const std::size_t storage_count = storage_element_count(tensor);
  if (storage_count >
      static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
    throw std::overflow_error("ACLNN unary storage shape overflows int64");
  }
  const std::int64_t storage_dimension =
      static_cast<std::int64_t>(storage_count);
  aclTensor* result = aclCreateTensor(tensor.dimensions.data(),
                                      tensor.dimensions.size(),
                                      acl_data_type(tensor.data_type),
                                      tensor.strides.data(),
                                      0,
                                      ACL_FORMAT_ND,
                                      &storage_dimension,
                                      1,
                                      pointer);
  if (result == nullptr) {
    std::string message = "aclCreateTensor returned null for unary reference";
    const char* recent = aclGetRecentErrMsg();
    if (recent != nullptr && recent[0] != '\0') {
      message += ": ";
      message += recent;
    }
    throw std::runtime_error(std::move(message));
  }
  return result;
}

void validate_attributes(flagdnnPointwiseMode_t mode,
                         const flagdnnPointwiseAttributes_t& attributes) {
  if (attributes.struct_size != sizeof(flagdnnPointwiseAttributes_t) ||
      attributes.version != FLAGDNN_POINTWISE_ATTRIBUTES_VERSION ||
      (attributes.flags & ~FLAGDNN_POINTWISE_ATTRIBUTE_FLAGS_ALL) != 0U) {
    throw std::invalid_argument("ACLNN unary attributes ABI is invalid");
  }
  if (mode == FLAGDNN_POINTWISE_RELU_FWD) {
    if ((attributes.flags & ~kReluAttributeFlags) != 0U) {
      throw std::invalid_argument(
          "ACLNN ReLU reference received unrelated attributes");
    }
    const double lower =
        (attributes.flags & FLAGDNN_POINTWISE_ATTRIBUTE_RELU_LOWER_CLIP) != 0U
            ? attributes.relu_lower_clip
            : 0.0;
    const double slope =
        (attributes.flags &
         FLAGDNN_POINTWISE_ATTRIBUTE_RELU_LOWER_CLIP_SLOPE) != 0U
            ? attributes.relu_lower_clip_slope
            : 0.0;
    if (!std::isfinite(lower) || !std::isfinite(slope) ||
        ((attributes.flags &
          FLAGDNN_POINTWISE_ATTRIBUTE_RELU_UPPER_CLIP) != 0U &&
         !std::isfinite(attributes.relu_upper_clip))) {
      throw std::invalid_argument("ACLNN ReLU attributes must be finite");
    }
#if FLAGDNN_ASCEND_VALIDATION_ENABLE_OPAPI_NN
  } else if (mode == FLAGDNN_POINTWISE_ELU_FWD) {
    if ((attributes.flags & ~FLAGDNN_POINTWISE_ATTRIBUTE_ELU_ALPHA) != 0U ||
        !std::isfinite(attributes.elu_alpha)) {
      throw std::invalid_argument(
          "ACLNN ELU reference attributes are invalid");
    }
  } else if (mode == FLAGDNN_POINTWISE_SOFTPLUS_FWD) {
    if ((attributes.flags & ~FLAGDNN_POINTWISE_ATTRIBUTE_SOFTPLUS_BETA) != 0U ||
        !std::isfinite(attributes.softplus_beta) ||
        attributes.softplus_beta <= 0.0) {
      throw std::invalid_argument(
          "ACLNN Softplus reference beta must be finite and positive");
    }
  } else if (mode == FLAGDNN_POINTWISE_SWISH_FWD) {
    if ((attributes.flags & ~FLAGDNN_POINTWISE_ATTRIBUTE_SWISH_BETA) != 0U ||
        !std::isfinite(attributes.swish_beta)) {
      throw std::invalid_argument(
          "ACLNN Swish reference beta is invalid");
    }
#endif
  } else if (attributes.flags != 0U) {
    throw std::invalid_argument(
        "ACLNN non-ReLU unary operation does not accept attributes");
  }
}

}  // namespace

struct AclnnUnaryRuntime::State {
  struct Stage {
    StageKind kind = StageKind::kNeg;
    aclOpExecutor* executor = nullptr;
    std::size_t workspace_size = 0;
  };

  ~State() {
    for (auto stage = stages.rbegin(); stage != stages.rend(); ++stage) {
      if (stage->executor != nullptr) {
        log_destroy_status(aclDestroyAclOpExecutor(stage->executor),
                           "aclDestroyAclOpExecutor(" +
                               stage_name(stage->kind) + ")");
      }
    }
    for (auto tensor = tensors.rbegin(); tensor != tensors.rend(); ++tensor) {
      if (*tensor != nullptr) {
        log_destroy_status(aclDestroyTensor(*tensor), "aclDestroyTensor");
      }
    }
    for (auto scalar = scalars.rbegin(); scalar != scalars.rend(); ++scalar) {
      if (*scalar != nullptr) {
        log_destroy_status(aclDestroyScalar(*scalar), "aclDestroyScalar");
      }
    }
  }

  std::unique_ptr<DeviceBuffer> shifted_buffer;
  std::unique_ptr<DeviceBuffer> positive_buffer;
  std::unique_ptr<DeviceBuffer> negative_buffer;
  std::unique_ptr<DeviceBuffer> scaled_buffer;
  std::unique_ptr<DeviceBuffer> combined_buffer;
  std::vector<aclTensor*> tensors;
  std::vector<aclScalar*> scalars;
  std::vector<Stage> stages;
  aclTensor* input = nullptr;
  aclTensor* output = nullptr;
  aclTensor* shifted = nullptr;
  aclTensor* positive = nullptr;
  aclTensor* negative = nullptr;
  aclTensor* scaled = nullptr;
  aclTensor* combined = nullptr;
  void* input_pointer = nullptr;
  void* output_pointer = nullptr;
  aclrtStream stream = nullptr;
  std::size_t workspace_size = 0;
  double lower = 0.0;
  double slope = 0.0;
  double zero = 0.0;
  double one = 1.0;
  double upper = 0.0;
#if FLAGDNN_ASCEND_VALIDATION_ENABLE_OPAPI_NN
  double elu_alpha = 1.0;
  double softplus_beta = 1.0;
  double softplus_threshold = 20.0;
  double swish_beta = 1.0;
#endif
};

bool is_aclnn_unary_mode(flagdnnPointwiseMode_t mode) noexcept {
  return mode == FLAGDNN_POINTWISE_IDENTITY ||
         mode == FLAGDNN_POINTWISE_NEG ||
         mode == FLAGDNN_POINTWISE_ABS ||
         mode == FLAGDNN_POINTWISE_CEIL ||
         mode == FLAGDNN_POINTWISE_COS ||
         mode == FLAGDNN_POINTWISE_ERF ||
         mode == FLAGDNN_POINTWISE_FLOOR ||
         mode == FLAGDNN_POINTWISE_RECIPROCAL ||
         mode == FLAGDNN_POINTWISE_SQRT ||
         mode == FLAGDNN_POINTWISE_RSQRT ||
         mode == FLAGDNN_POINTWISE_SIN ||
         mode == FLAGDNN_POINTWISE_TAN ||
         mode == FLAGDNN_POINTWISE_EXP ||
         mode == FLAGDNN_POINTWISE_LOG ||
         mode == FLAGDNN_POINTWISE_TANH_FWD ||
         mode == FLAGDNN_POINTWISE_LOGICAL_NOT ||
#if FLAGDNN_ASCEND_VALIDATION_ENABLE_OPAPI_NN
         mode == FLAGDNN_POINTWISE_SIGMOID_FWD ||
         mode == FLAGDNN_POINTWISE_ELU_FWD ||
         mode == FLAGDNN_POINTWISE_GELU_FWD ||
         mode == FLAGDNN_POINTWISE_SOFTPLUS_FWD ||
         mode == FLAGDNN_POINTWISE_SWISH_FWD ||
         mode == FLAGDNN_POINTWISE_GELU_APPROX_TANH_FWD ||
#endif
         mode == FLAGDNN_POINTWISE_RELU_FWD;
}

AclnnUnaryRuntime::AclnnUnaryRuntime(
    flagdnnPointwiseMode_t mode,
    flagdnnPointwiseAttributes_t attributes)
    : mode_(mode), attributes_(attributes) {
  if (!is_aclnn_unary_mode(mode_)) {
    throw std::invalid_argument("ACLNN unary runtime mode is unsupported");
  }
  validate_attributes(mode_, attributes_);
}

AclnnUnaryRuntime::~AclnnUnaryRuntime() = default;

void AclnnUnaryRuntime::prepare(const AclnnUnaryTensor& input,
                                void* input_pointer,
                                const AclnnUnaryTensor& output,
                                void* output_pointer,
                                flagdnnStream_t stream) {
  if (input_pointer == nullptr || output_pointer == nullptr || stream == nullptr) {
    throw std::invalid_argument("ACLNN unary prepare received a null argument");
  }
  validate_tensor(input);
  validate_tensor(output);
  if (input.data_type != output.data_type ||
      input.dimensions != output.dimensions) {
    throw std::invalid_argument(
        "ACLNN unary input and output metadata do not match");
  }
  if (state_ != nullptr) {
    check_acl(aclrtSynchronizeStream(state_->stream),
              "aclrtSynchronizeStream(before ACLNN unary reprepare)");
  }

  auto candidate = std::make_unique<State>();
  // Reserve before creating any ACLNN-owned handle so every subsequent
  // insertion is non-allocating and the candidate state remains transactional
  // even if host allocation fails.
  candidate->tensors.reserve(7);
  candidate->scalars.reserve(5);
  candidate->stages.reserve(7);
  candidate->input_pointer = input_pointer;
  candidate->output_pointer = output_pointer;
  candidate->stream = reinterpret_cast<aclrtStream>(stream);
  candidate->input = create_tensor(input, input_pointer);
  candidate->tensors.push_back(candidate->input);
  candidate->output = create_tensor(output, output_pointer);
  candidate->tensors.push_back(candidate->output);

  const auto create_scalar = [&](double* value) {
    aclScalar* scalar = aclCreateScalar(value, ACL_DOUBLE);
    if (scalar == nullptr) {
      throw std::runtime_error("aclCreateScalar returned null for unary reference");
    }
    candidate->scalars.push_back(scalar);
    return scalar;
  };

  const auto add_stage = [&](StageKind kind, auto&& query) {
    candidate->stages.push_back({kind, nullptr, 0});
    State::Stage& stage = candidate->stages.back();
    std::uint64_t workspace_size = 0;
    const aclnnStatus status = query(&workspace_size, &stage.executor);
    if (status != 0) {
      const std::string api = stage_name(kind) + "GetWorkspaceSize";
      const std::string message = aclnn_error_message(status, api);
      if (explicitly_unsupported(status, message)) {
        throw AclnnUnaryUnsupportedError(status, message);
      }
      throw std::runtime_error(message);
    }
    if (stage.executor == nullptr) {
      throw std::runtime_error(stage_name(kind) +
                               " returned a null ACLNN executor");
    }
    const aclnnStatus repeatable =
        aclSetAclOpExecutorRepeatable(stage.executor);
    if (repeatable != 0) {
      throw std::runtime_error(aclnn_error_message(
          repeatable, "aclSetAclOpExecutorRepeatable(" + stage_name(kind) + ")"));
    }
    if (workspace_size > std::numeric_limits<std::size_t>::max()) {
      throw std::overflow_error("ACLNN unary workspace size overflows size_t");
    }
    stage.workspace_size = static_cast<std::size_t>(workspace_size);
    candidate->workspace_size =
        std::max(candidate->workspace_size, stage.workspace_size);
  };

  if (mode_ == FLAGDNN_POINTWISE_IDENTITY) {
    const aclScalar* zero = create_scalar(&candidate->zero);
    const aclScalar* one = create_scalar(&candidate->one);
    add_stage(StageKind::kAdds,
              [&](std::uint64_t* size, aclOpExecutor** executor) {
                return aclnnAddsGetWorkspaceSize(candidate->input,
                                                  zero,
                                                  one,
                                                  candidate->output,
                                                  size,
                                                  executor);
              });
  } else if (mode_ == FLAGDNN_POINTWISE_NEG) {
    add_stage(StageKind::kNeg,
              [&](std::uint64_t* size, aclOpExecutor** executor) {
                return aclnnNegGetWorkspaceSize(
                    candidate->input, candidate->output, size, executor);
              });
  } else if (mode_ == FLAGDNN_POINTWISE_ABS) {
    add_stage(StageKind::kAbs,
              [&](std::uint64_t* size, aclOpExecutor** executor) {
                return aclnnAbsGetWorkspaceSize(
                    candidate->input, candidate->output, size, executor);
              });
  } else if (mode_ == FLAGDNN_POINTWISE_CEIL) {
    add_stage(StageKind::kCeil,
              [&](std::uint64_t* size, aclOpExecutor** executor) {
                return aclnnCeilGetWorkspaceSize(
                    candidate->input, candidate->output, size, executor);
              });
  } else if (mode_ == FLAGDNN_POINTWISE_COS) {
    add_stage(StageKind::kCos,
              [&](std::uint64_t* size, aclOpExecutor** executor) {
                return aclnnCosGetWorkspaceSize(
                    candidate->input, candidate->output, size, executor);
              });
  } else if (mode_ == FLAGDNN_POINTWISE_ERF) {
    add_stage(StageKind::kErf,
              [&](std::uint64_t* size, aclOpExecutor** executor) {
                return aclnnErfGetWorkspaceSize(
                    candidate->input, candidate->output, size, executor);
              });
  } else if (mode_ == FLAGDNN_POINTWISE_FLOOR) {
    add_stage(StageKind::kFloor,
              [&](std::uint64_t* size, aclOpExecutor** executor) {
                return aclnnFloorGetWorkspaceSize(
                    candidate->input, candidate->output, size, executor);
              });
  } else if (mode_ == FLAGDNN_POINTWISE_RECIPROCAL) {
    add_stage(StageKind::kReciprocal,
              [&](std::uint64_t* size, aclOpExecutor** executor) {
                return aclnnReciprocalGetWorkspaceSize(
                    candidate->input, candidate->output, size, executor);
              });
  } else if (mode_ == FLAGDNN_POINTWISE_SQRT) {
    add_stage(StageKind::kSqrt,
              [&](std::uint64_t* size, aclOpExecutor** executor) {
                return aclnnSqrtGetWorkspaceSize(
                    candidate->input, candidate->output, size, executor);
              });
  } else if (mode_ == FLAGDNN_POINTWISE_RSQRT) {
    add_stage(StageKind::kRsqrt,
              [&](std::uint64_t* size, aclOpExecutor** executor) {
                return aclnnRsqrtGetWorkspaceSize(
                    candidate->input, candidate->output, size, executor);
              });
  } else if (mode_ == FLAGDNN_POINTWISE_SIN) {
    add_stage(StageKind::kSin,
              [&](std::uint64_t* size, aclOpExecutor** executor) {
                return aclnnSinGetWorkspaceSize(
                    candidate->input, candidate->output, size, executor);
              });
  } else if (mode_ == FLAGDNN_POINTWISE_TAN) {
    add_stage(StageKind::kTan,
              [&](std::uint64_t* size, aclOpExecutor** executor) {
                return aclnnTanGetWorkspaceSize(
                    candidate->input, candidate->output, size, executor);
              });
  } else if (mode_ == FLAGDNN_POINTWISE_EXP) {
    add_stage(StageKind::kExp,
              [&](std::uint64_t* size, aclOpExecutor** executor) {
                return aclnnExpGetWorkspaceSize(
                    candidate->input, candidate->output, size, executor);
              });
  } else if (mode_ == FLAGDNN_POINTWISE_LOG) {
    add_stage(StageKind::kLog,
              [&](std::uint64_t* size, aclOpExecutor** executor) {
                return aclnnLogGetWorkspaceSize(
                    candidate->input, candidate->output, size, executor);
              });
  } else if (mode_ == FLAGDNN_POINTWISE_TANH_FWD) {
    add_stage(StageKind::kTanh,
              [&](std::uint64_t* size, aclOpExecutor** executor) {
                return aclnnTanhGetWorkspaceSize(
                    candidate->input, candidate->output, size, executor);
              });
  } else if (mode_ == FLAGDNN_POINTWISE_LOGICAL_NOT) {
    add_stage(StageKind::kLogicalNot,
              [&](std::uint64_t* size, aclOpExecutor** executor) {
                return aclnnLogicalNotGetWorkspaceSize(
                    candidate->input, candidate->output, size, executor);
              });
#if FLAGDNN_ASCEND_VALIDATION_ENABLE_OPAPI_NN
  } else if (mode_ == FLAGDNN_POINTWISE_SIGMOID_FWD) {
    add_stage(StageKind::kSigmoid,
              [&](std::uint64_t* size, aclOpExecutor** executor) {
                return aclnnSigmoidGetWorkspaceSize(
                    candidate->input, candidate->output, size, executor);
              });
  } else if (mode_ == FLAGDNN_POINTWISE_ELU_FWD) {
    candidate->elu_alpha =
        (attributes_.flags & FLAGDNN_POINTWISE_ATTRIBUTE_ELU_ALPHA) != 0U
            ? attributes_.elu_alpha
            : 1.0;
    const aclScalar* alpha = create_scalar(&candidate->elu_alpha);
    const aclScalar* one = create_scalar(&candidate->one);
    add_stage(StageKind::kElu,
              [&](std::uint64_t* size, aclOpExecutor** executor) {
                return aclnnEluGetWorkspaceSize(candidate->input,
                                                 alpha,
                                                 one,
                                                 one,
                                                 candidate->output,
                                                 size,
                                                 executor);
              });
  } else if (mode_ == FLAGDNN_POINTWISE_GELU_FWD) {
    add_stage(StageKind::kGelu,
              [&](std::uint64_t* size, aclOpExecutor** executor) {
                return aclnnGeluGetWorkspaceSize(
                    candidate->input, candidate->output, size, executor);
              });
  } else if (mode_ == FLAGDNN_POINTWISE_SOFTPLUS_FWD) {
    candidate->softplus_beta =
        (attributes_.flags & FLAGDNN_POINTWISE_ATTRIBUTE_SOFTPLUS_BETA) != 0U
            ? attributes_.softplus_beta
            : 1.0;
    const aclScalar* beta = create_scalar(&candidate->softplus_beta);
    const aclScalar* threshold =
        create_scalar(&candidate->softplus_threshold);
    add_stage(StageKind::kSoftplus,
              [&](std::uint64_t* size, aclOpExecutor** executor) {
                return aclnnSoftplusGetWorkspaceSize(candidate->input,
                                                      beta,
                                                      threshold,
                                                      candidate->output,
                                                      size,
                                                      executor);
              });
  } else if (mode_ == FLAGDNN_POINTWISE_SWISH_FWD) {
    const aclScalar* beta = nullptr;
    if ((attributes_.flags & FLAGDNN_POINTWISE_ATTRIBUTE_SWISH_BETA) != 0U) {
      candidate->swish_beta = attributes_.swish_beta;
      beta = create_scalar(&candidate->swish_beta);
    }
    add_stage(StageKind::kSwish,
              [&](std::uint64_t* size, aclOpExecutor** executor) {
                return aclnnSwishGetWorkspaceSize(candidate->input,
                                                   beta,
                                                   candidate->output,
                                                   size,
                                                   executor);
              });
  } else if (mode_ == FLAGDNN_POINTWISE_GELU_APPROX_TANH_FWD) {
    add_stage(StageKind::kGeluV2,
              [&](std::uint64_t* size, aclOpExecutor** executor) {
                constexpr std::int64_t kApproximateTanh = 1;
                return aclnnGeluV2GetWorkspaceSize(candidate->input,
                                                   kApproximateTanh,
                                                   candidate->output,
                                                   size,
                                                   executor);
              });
#endif
  } else {
    candidate->lower =
        (attributes_.flags & FLAGDNN_POINTWISE_ATTRIBUTE_RELU_LOWER_CLIP) != 0U
            ? attributes_.relu_lower_clip
            : 0.0;
    candidate->slope =
        (attributes_.flags &
         FLAGDNN_POINTWISE_ATTRIBUTE_RELU_LOWER_CLIP_SLOPE) != 0U
            ? attributes_.relu_lower_clip_slope
            : 0.0;
    const bool has_upper =
        (attributes_.flags & FLAGDNN_POINTWISE_ATTRIBUTE_RELU_UPPER_CLIP) != 0U;
    candidate->upper = attributes_.relu_upper_clip;

    const aclScalar* lower = create_scalar(&candidate->lower);
    const aclScalar* zero = create_scalar(&candidate->zero);
    const aclScalar* one = create_scalar(&candidate->one);
    const aclScalar* upper = has_upper ? create_scalar(&candidate->upper) : nullptr;
    const AclnnUnaryTensor intermediate{
        input.data_type, input.dimensions, contiguous_strides(input.dimensions)};
    const std::size_t intermediate_bytes = encoded_byte_count(intermediate);

    if (candidate->slope == 0.0) {
      candidate->positive_buffer =
          std::make_unique<DeviceBuffer>(intermediate_bytes);
      candidate->positive =
          create_tensor(intermediate, candidate->positive_buffer->opaque());
      candidate->tensors.push_back(candidate->positive);
      add_stage(StageKind::kClampMin,
                [&](std::uint64_t* size, aclOpExecutor** executor) {
                  return aclnnClampMinGetWorkspaceSize(candidate->input,
                                                       lower,
                                                       candidate->positive,
                                                       size,
                                                       executor);
                });
      if (has_upper) {
        add_stage(StageKind::kClampMax,
                  [&](std::uint64_t* size, aclOpExecutor** executor) {
                    return aclnnClampMaxGetWorkspaceSize(candidate->positive,
                                                         upper,
                                                         candidate->output,
                                                         size,
                                                         executor);
                  });
      } else {
        add_stage(StageKind::kAdds,
                  [&](std::uint64_t* size, aclOpExecutor** executor) {
                    return aclnnAddsGetWorkspaceSize(candidate->positive,
                                                     zero,
                                                     one,
                                                     candidate->output,
                                                     size,
                                                     executor);
                  });
      }
    } else {
      const aclScalar* slope = create_scalar(&candidate->slope);
      candidate->shifted_buffer =
          std::make_unique<DeviceBuffer>(intermediate_bytes);
      candidate->shifted =
          create_tensor(intermediate, candidate->shifted_buffer->opaque());
      candidate->tensors.push_back(candidate->shifted);
      add_stage(StageKind::kSubs,
                [&](std::uint64_t* size, aclOpExecutor** executor) {
                  return aclnnSubsGetWorkspaceSize(candidate->input,
                                                   lower,
                                                   one,
                                                   candidate->shifted,
                                                   size,
                                                   executor);
                });

      candidate->positive_buffer =
          std::make_unique<DeviceBuffer>(intermediate_bytes);
      candidate->positive =
          create_tensor(intermediate, candidate->positive_buffer->opaque());
      candidate->tensors.push_back(candidate->positive);
      add_stage(StageKind::kClampMin,
                [&](std::uint64_t* size, aclOpExecutor** executor) {
                  return aclnnClampMinGetWorkspaceSize(candidate->shifted,
                                                       zero,
                                                       candidate->positive,
                                                       size,
                                                       executor);
                });

      candidate->negative_buffer =
          std::make_unique<DeviceBuffer>(intermediate_bytes);
      candidate->negative =
          create_tensor(intermediate, candidate->negative_buffer->opaque());
      candidate->tensors.push_back(candidate->negative);
      add_stage(StageKind::kClampMax,
                [&](std::uint64_t* size, aclOpExecutor** executor) {
                  return aclnnClampMaxGetWorkspaceSize(candidate->shifted,
                                                       zero,
                                                       candidate->negative,
                                                       size,
                                                       executor);
                });

      candidate->scaled_buffer =
          std::make_unique<DeviceBuffer>(intermediate_bytes);
      candidate->scaled =
          create_tensor(intermediate, candidate->scaled_buffer->opaque());
      candidate->tensors.push_back(candidate->scaled);
      add_stage(StageKind::kMuls,
                [&](std::uint64_t* size, aclOpExecutor** executor) {
                  return aclnnMulsGetWorkspaceSize(candidate->negative,
                                                   slope,
                                                   candidate->scaled,
                                                   size,
                                                   executor);
                });

      const bool needs_lower_add = candidate->lower != 0.0;
      aclTensor* combined_output = candidate->output;
      if (needs_lower_add || has_upper) {
        candidate->combined_buffer =
            std::make_unique<DeviceBuffer>(intermediate_bytes);
        candidate->combined =
            create_tensor(intermediate, candidate->combined_buffer->opaque());
        candidate->tensors.push_back(candidate->combined);
        combined_output = candidate->combined;
      }
      add_stage(StageKind::kAdd,
                [&](std::uint64_t* size, aclOpExecutor** executor) {
                  return aclnnAddGetWorkspaceSize(candidate->scaled,
                                                  candidate->positive,
                                                  one,
                                                  combined_output,
                                                  size,
                                                  executor);
                });
      aclTensor* pre_upper = combined_output;
      if (needs_lower_add) {
        aclTensor* lower_output = candidate->output;
        if (has_upper) {
          lower_output = candidate->shifted;
          pre_upper = candidate->shifted;
        }
        add_stage(StageKind::kAdds,
                  [&](std::uint64_t* size, aclOpExecutor** executor) {
                    return aclnnAddsGetWorkspaceSize(candidate->combined,
                                                     lower,
                                                     one,
                                                     lower_output,
                                                     size,
                                                     executor);
                  });
      }
      if (has_upper) {
        add_stage(StageKind::kClampMax,
                  [&](std::uint64_t* size, aclOpExecutor** executor) {
                    return aclnnClampMaxGetWorkspaceSize(pre_upper,
                                                         upper,
                                                         candidate->output,
                                                         size,
                                                         executor);
                  });
      }
    }
  }

  std::unique_ptr<State> old = std::move(state_);
  state_ = std::move(candidate);
  old.reset();
}

std::size_t AclnnUnaryRuntime::workspace_size() const noexcept {
  return state_ == nullptr ? 0 : state_->workspace_size;
}

void AclnnUnaryRuntime::execute(void* input_pointer,
                                void* output_pointer,
                                void* workspace,
                                std::size_t workspace_size,
                                flagdnnStream_t stream) {
  if (state_ == nullptr) {
    throw std::logic_error("ACLNN unary execute called before prepare");
  }
  if (input_pointer != state_->input_pointer ||
      output_pointer != state_->output_pointer ||
      reinterpret_cast<aclrtStream>(stream) != state_->stream) {
    throw std::invalid_argument(
        "ACLNN unary repeatable executor binding or stream changed");
  }
  if (workspace_size < state_->workspace_size ||
      (state_->workspace_size != 0 && workspace == nullptr)) {
    throw std::invalid_argument("ACLNN unary workspace is too small");
  }

  for (const State::Stage& stage : state_->stages) {
    void* effective_workspace = stage.workspace_size == 0 ? nullptr : workspace;
    aclnnStatus status = 0;
    switch (stage.kind) {
      case StageKind::kNeg:
        status = aclnnNeg(effective_workspace,
                          stage.workspace_size,
                          stage.executor,
                          state_->stream);
        break;
      case StageKind::kAbs:
        status = aclnnAbs(effective_workspace,
                          stage.workspace_size,
                          stage.executor,
                          state_->stream);
        break;
      case StageKind::kCeil:
        status = aclnnCeil(effective_workspace,
                           stage.workspace_size,
                           stage.executor,
                           state_->stream);
        break;
      case StageKind::kCos:
        status = aclnnCos(effective_workspace,
                          stage.workspace_size,
                          stage.executor,
                          state_->stream);
        break;
      case StageKind::kErf:
        status = aclnnErf(effective_workspace,
                          stage.workspace_size,
                          stage.executor,
                          state_->stream);
        break;
      case StageKind::kFloor:
        status = aclnnFloor(effective_workspace,
                            stage.workspace_size,
                            stage.executor,
                            state_->stream);
        break;
      case StageKind::kReciprocal:
        status = aclnnReciprocal(effective_workspace,
                                 stage.workspace_size,
                                 stage.executor,
                                 state_->stream);
        break;
      case StageKind::kSqrt:
        status = aclnnSqrt(effective_workspace,
                           stage.workspace_size,
                           stage.executor,
                           state_->stream);
        break;
      case StageKind::kRsqrt:
        status = aclnnRsqrt(effective_workspace,
                            stage.workspace_size,
                            stage.executor,
                            state_->stream);
        break;
      case StageKind::kSin:
        status = aclnnSin(effective_workspace,
                          stage.workspace_size,
                          stage.executor,
                          state_->stream);
        break;
      case StageKind::kTan:
        status = aclnnTan(effective_workspace,
                          stage.workspace_size,
                          stage.executor,
                          state_->stream);
        break;
      case StageKind::kExp:
        status = aclnnExp(effective_workspace,
                          stage.workspace_size,
                          stage.executor,
                          state_->stream);
        break;
      case StageKind::kLog:
        status = aclnnLog(effective_workspace,
                          stage.workspace_size,
                          stage.executor,
                          state_->stream);
        break;
      case StageKind::kTanh:
        status = aclnnTanh(effective_workspace,
                           stage.workspace_size,
                           stage.executor,
                           state_->stream);
        break;
      case StageKind::kLogicalNot:
        status = aclnnLogicalNot(effective_workspace,
                                 stage.workspace_size,
                                 stage.executor,
                                 state_->stream);
        break;
#if FLAGDNN_ASCEND_VALIDATION_ENABLE_OPAPI_NN
      case StageKind::kSigmoid:
        status = aclnnSigmoid(effective_workspace,
                              stage.workspace_size,
                              stage.executor,
                              state_->stream);
        break;
      case StageKind::kElu:
        status = aclnnElu(effective_workspace,
                          stage.workspace_size,
                          stage.executor,
                          state_->stream);
        break;
      case StageKind::kGelu:
        status = aclnnGelu(effective_workspace,
                           stage.workspace_size,
                           stage.executor,
                           state_->stream);
        break;
      case StageKind::kGeluV2:
        status = aclnnGeluV2(effective_workspace,
                             stage.workspace_size,
                             stage.executor,
                             state_->stream);
        break;
      case StageKind::kSoftplus:
        status = aclnnSoftplus(effective_workspace,
                               stage.workspace_size,
                               stage.executor,
                               state_->stream);
        break;
      case StageKind::kSwish:
        status = aclnnSwish(effective_workspace,
                            stage.workspace_size,
                            stage.executor,
                            state_->stream);
        break;
#endif
      case StageKind::kClampMin:
        status = aclnnClampMin(effective_workspace,
                               stage.workspace_size,
                               stage.executor,
                               state_->stream);
        break;
      case StageKind::kClampMax:
        status = aclnnClampMax(effective_workspace,
                               stage.workspace_size,
                               stage.executor,
                               state_->stream);
        break;
      case StageKind::kMuls:
        status = aclnnMuls(effective_workspace,
                           stage.workspace_size,
                           stage.executor,
                           state_->stream);
        break;
      case StageKind::kAdd:
        status = aclnnAdd(effective_workspace,
                          stage.workspace_size,
                          stage.executor,
                          state_->stream);
        break;
      case StageKind::kAdds:
        status = aclnnAdds(effective_workspace,
                           stage.workspace_size,
                           stage.executor,
                           state_->stream);
        break;
      case StageKind::kSubs:
        status = aclnnSubs(effective_workspace,
                           stage.workspace_size,
                           stage.executor,
                           state_->stream);
        break;
    }
    if (status != 0) {
      throw std::runtime_error(aclnn_error_message(status,
                                                   stage_name(stage.kind)));
    }
  }
}

}  // namespace flagdnn::validation::ascend
