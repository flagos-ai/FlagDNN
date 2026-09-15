/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "validation/benchmark/aclnn_provider.hpp"

#ifndef FLAGDNN_ASCEND_VALIDATION_ENABLE_BATCHNORM_INFERENCE
#define FLAGDNN_ASCEND_VALIDATION_ENABLE_BATCHNORM_INFERENCE 0
#endif
#ifndef FLAGDNN_ASCEND_VALIDATION_ENABLE_MATMUL
#define FLAGDNN_ASCEND_VALIDATION_ENABLE_MATMUL 0
#endif
#ifndef FLAGDNN_ASCEND_VALIDATION_ENABLE_CONVOLUTION_FPROP
#define FLAGDNN_ASCEND_VALIDATION_ENABLE_CONVOLUTION_FPROP 0
#endif

#if FLAGDNN_ASCEND_VALIDATION_ENABLE_BATCHNORM_INFERENCE
#include "validation/benchmark/aclnn_batchnorm_inference_provider.hpp"
#include "validation/batchnorm_inference_validation.hpp"
#endif

#if FLAGDNN_ASCEND_VALIDATION_ENABLE_MATMUL
#include "validation/benchmark/aclnn_matmul_provider.hpp"
#endif

#if FLAGDNN_ASCEND_VALIDATION_ENABLE_CONVOLUTION_FPROP
#include "validation/benchmark/aclnn_convolution_provider.hpp"
#endif

#include "validation/acl_runtime.hpp"
#include "validation/tensor_io.hpp"

#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include <aclnnop/aclnn_add.h>
#include <aclnnop/aclnn_div.h>
#include <aclnnop/aclnn_eq_tensor.h>
#include <aclnnop/aclnn_fmod_tensor.h>
#include <aclnnop/aclnn_ge_tensor.h>
#include <aclnnop/aclnn_gt_tensor.h>
#include <aclnnop/aclnn_logical_and.h>
#include <aclnnop/aclnn_logical_or.h>
#include <aclnnop/aclnn_maximum.h>
#include <aclnnop/aclnn_minimum.h>
#include <aclnnop/aclnn_mul.h>
#include <aclnnop/aclnn_ne_tensor.h>
#include <aclnnop/aclnn_pow_tensor_tensor.h>
#include <aclnnop/aclnn_s_where.h>
#include <aclnnop/aclnn_sub.h>

#ifndef FLAGDNN_ASCEND_VALIDATION_ENABLE_OPAPI_NN
#define FLAGDNN_ASCEND_VALIDATION_ENABLE_OPAPI_NN 0
#endif

#if FLAGDNN_ASCEND_VALIDATION_ENABLE_OPAPI_NN
#include <aclnnop/aclnn_sigmoid.h>
#include <aclnnop/aclnn_sigmoid_backward.h>
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace flagdnn::benchmarking {
namespace {

namespace tensor_io = flagdnn::validation::ascend::tensor_io;

std::vector<std::int64_t> checked_contiguous_strides(
    const std::vector<std::int64_t>& dimensions) {
  std::vector<std::int64_t> result(dimensions.size());
  std::int64_t stride = 1;
  for (std::size_t axis = dimensions.size(); axis != 0; --axis) {
    result[axis - 1] = stride;
    if (dimensions[axis - 1] >
        std::numeric_limits<std::int64_t>::max() / stride) {
      throw std::overflow_error("ACLNN contiguous stride overflows int64");
    }
    stride *= dimensions[axis - 1];
  }
  return result;
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
      "ACLNN binary pointwise benchmark data type is unsupported");
}

flagdnnPointwiseMode_t benchmark_mode(
    const BenchmarkCase& specification) {
  if (specification.operation == Operation::kAdd) {
    return FLAGDNN_POINTWISE_ADD;
  }
  if (specification.operation == Operation::kPointwise &&
      (specification.pointwise_mode == FLAGDNN_POINTWISE_SUB ||
       specification.pointwise_mode == FLAGDNN_POINTWISE_MUL ||
       specification.pointwise_mode == FLAGDNN_POINTWISE_DIV ||
       specification.pointwise_mode == FLAGDNN_POINTWISE_MOD ||
       specification.pointwise_mode == FLAGDNN_POINTWISE_POW ||
       specification.pointwise_mode == FLAGDNN_POINTWISE_SIGMOID_BWD ||
       specification.pointwise_mode == FLAGDNN_POINTWISE_MIN ||
       specification.pointwise_mode == FLAGDNN_POINTWISE_MAX ||
       specification.pointwise_mode == FLAGDNN_POINTWISE_CMP_EQ ||
       specification.pointwise_mode == FLAGDNN_POINTWISE_CMP_NEQ ||
       specification.pointwise_mode == FLAGDNN_POINTWISE_CMP_GT ||
       specification.pointwise_mode == FLAGDNN_POINTWISE_CMP_GE ||
       specification.pointwise_mode == FLAGDNN_POINTWISE_CMP_LT ||
       specification.pointwise_mode == FLAGDNN_POINTWISE_CMP_LE ||
       specification.pointwise_mode == FLAGDNN_POINTWISE_LOGICAL_AND ||
       specification.pointwise_mode == FLAGDNN_POINTWISE_LOGICAL_OR)) {
    return specification.pointwise_mode;
  }
  throw std::invalid_argument(
      "ACLNN provider requires a supported binary pointwise mode");
}

bool has_aclnn_binary_benchmark_mapping(
    const BenchmarkCase& specification) noexcept {
  if (specification.operation == Operation::kAdd) {
    return true;
  }
  if (specification.operation != Operation::kPointwise) {
    return false;
  }
  switch (specification.pointwise_mode) {
    case FLAGDNN_POINTWISE_SUB:
    case FLAGDNN_POINTWISE_MUL:
    case FLAGDNN_POINTWISE_DIV:
    case FLAGDNN_POINTWISE_MOD:
    case FLAGDNN_POINTWISE_POW:
    case FLAGDNN_POINTWISE_SIGMOID_BWD:
    case FLAGDNN_POINTWISE_MIN:
    case FLAGDNN_POINTWISE_MAX:
    case FLAGDNN_POINTWISE_CMP_EQ:
    case FLAGDNN_POINTWISE_CMP_NEQ:
    case FLAGDNN_POINTWISE_CMP_GT:
    case FLAGDNN_POINTWISE_CMP_GE:
    case FLAGDNN_POINTWISE_CMP_LT:
    case FLAGDNN_POINTWISE_CMP_LE:
    case FLAGDNN_POINTWISE_LOGICAL_AND:
    case FLAGDNN_POINTWISE_LOGICAL_OR:
      return true;
    default:
      return false;
  }
}

std::string operation_name(flagdnnPointwiseMode_t mode) {
  switch (mode) {
    case FLAGDNN_POINTWISE_ADD:
      return "Add";
    case FLAGDNN_POINTWISE_SUB:
      return "Sub";
    case FLAGDNN_POINTWISE_MUL:
      return "Mul";
    case FLAGDNN_POINTWISE_DIV:
      return "Div";
    case FLAGDNN_POINTWISE_MOD:
      return "FmodTensor";
    case FLAGDNN_POINTWISE_POW:
      return "PowTensorTensor";
    case FLAGDNN_POINTWISE_SIGMOID_BWD:
      return "SigmoidBackward";
    case FLAGDNN_POINTWISE_MIN:
      return "Minimum";
    case FLAGDNN_POINTWISE_MAX:
      return "Maximum";
    case FLAGDNN_POINTWISE_CMP_EQ:
      return "EqTensor";
    case FLAGDNN_POINTWISE_CMP_NEQ:
      return "NeTensor";
    case FLAGDNN_POINTWISE_CMP_GT:
    case FLAGDNN_POINTWISE_CMP_LT:
      return "GtTensor";
    case FLAGDNN_POINTWISE_CMP_GE:
    case FLAGDNN_POINTWISE_CMP_LE:
      return "GeTensor";
    case FLAGDNN_POINTWISE_LOGICAL_AND:
      return "LogicalAnd";
    case FLAGDNN_POINTWISE_LOGICAL_OR:
      return "LogicalOr";
    case FLAGDNN_POINTWISE_BINARY_SELECT:
      return "SWhere";
    default:
      throw std::invalid_argument(
          "ACLNN benchmark pointwise mode is invalid");
  }
}

bool broadcasts_to(const TensorSpec& input, const TensorSpec& output) {
  if (input.dimensions.size() > output.dimensions.size()) {
    return false;
  }
  const std::size_t leading =
      output.dimensions.size() - input.dimensions.size();
  for (std::size_t axis = 0; axis < input.dimensions.size(); ++axis) {
    const std::int64_t input_dimension = input.dimensions[axis];
    const std::int64_t output_dimension = output.dimensions[leading + axis];
    if (input_dimension != 1 && input_dimension != output_dimension) {
      return false;
    }
  }
  return true;
}

bool has_exact_broadcast_output(const TensorSpec& self,
                                const TensorSpec& other,
                                const TensorSpec& condition,
                                const TensorSpec& output) {
  const std::array<const TensorSpec*, 3> inputs = {
      &self, &other, &condition};
  for (std::size_t axis = 0; axis < output.dimensions.size(); ++axis) {
    std::int64_t expected = 1;
    for (const TensorSpec* input : inputs) {
      if (input->dimensions.size() > output.dimensions.size()) {
        return false;
      }
      const std::size_t leading =
          output.dimensions.size() - input->dimensions.size();
      const std::int64_t dimension =
          axis < leading ? 1 : input->dimensions[axis - leading];
      if (dimension != 1) {
        if (expected != 1 && expected != dimension) {
          return false;
        }
        expected = dimension;
      }
    }
    if (output.dimensions[axis] != expected) {
      return false;
    }
  }
  return true;
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
    std::cerr << "Ascend ACLNN benchmark cleanup: "
              << aclnn_error_message(status, operation) << '\n';
  }
}

const flagdnnBinding_t& find_binding(
    std::span<const flagdnnBinding_t> bindings,
    std::int64_t uid) {
  const flagdnnBinding_t* result = nullptr;
  for (const flagdnnBinding_t& binding : bindings) {
    if (binding.uid != uid) {
      continue;
    }
    if (result != nullptr) {
      throw std::invalid_argument("ACLNN benchmark binding UID is duplicated");
    }
    result = &binding;
  }
  if (result == nullptr || result->device_pointer == nullptr) {
    throw std::invalid_argument("ACLNN benchmark binding is missing or null");
  }
  return *result;
}

void require_aligned(void* pointer, std::int64_t uid) {
  if (reinterpret_cast<std::uintptr_t>(pointer) % 32U != 0U) {
    throw std::invalid_argument(
        "ACLNN benchmark binding UID " + std::to_string(uid) +
        " is not 32-byte aligned");
  }
}

aclTensor* create_tensor(const TensorSpec& tensor, void* pointer) {
  const std::size_t storage_count = tensor_io::storage_element_count(tensor);
  if (storage_count >
      static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
    throw std::overflow_error("ACLNN tensor storage shape overflows int64");
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
    throw std::runtime_error("aclCreateTensor returned null for benchmark");
  }
  return result;
}

#if FLAGDNN_ASCEND_VALIDATION_ENABLE_OPAPI_NN
class AclnnSigmoidBackwardBenchmarkExecutable final
    : public BenchmarkExecutable {
 public:
  AclnnSigmoidBackwardBenchmarkExecutable(
      BenchmarkCase specification,
      AclnnBinaryPointwiseBenchmarkPlan plan)
      : specification_(std::move(specification)), plan_(std::move(plan)) {
    if (plan_.mode != FLAGDNN_POINTWISE_SIGMOID_BWD) {
      throw std::invalid_argument(
          "ACLNN sigmoid backward benchmark received a different mode");
    }
    sigmoid_output_ = plan_.right;
    sigmoid_output_.uid = 0;
    sigmoid_output_.strides =
        checked_contiguous_strides(sigmoid_output_.dimensions);
    sigmoid_output_.binding_byte_offset = 0;
    (void)tensor_io::encoded_byte_count(sigmoid_output_);
  }

  void prepare(std::span<const flagdnnBinding_t> bindings,
               flagdnnStream_t stream) override {
    if (stream == nullptr) {
      throw std::invalid_argument(
          "ACLNN sigmoid backward benchmark prepare stream is null");
    }
    if (state_ != nullptr) {
      flagdnn::validation::ascend::check_acl(
          aclrtSynchronizeStream(state_->stream),
          "aclrtSynchronizeStream(before ACLNN sigmoid backward benchmark "
          "reprepare)");
    }
    const flagdnnBinding_t& gradient =
        find_binding(bindings, plan_.left.uid);
    const flagdnnBinding_t& logit =
        find_binding(bindings, plan_.right.uid);
    const flagdnnBinding_t& output =
        find_binding(bindings, plan_.output.uid);
    require_aligned(gradient.device_pointer, gradient.uid);
    require_aligned(logit.device_pointer, logit.uid);
    require_aligned(output.device_pointer, output.uid);

    auto candidate = std::make_unique<State>();
    candidate->gradient_pointer = gradient.device_pointer;
    candidate->logit_pointer = logit.device_pointer;
    candidate->output_pointer = output.device_pointer;
    candidate->stream = reinterpret_cast<aclrtStream>(stream);
    candidate->sigmoid_storage =
        std::make_unique<flagdnn::validation::ascend::DeviceBuffer>(
            tensor_io::encoded_byte_count(sigmoid_output_));
    candidate->gradient =
        create_tensor(plan_.left, gradient.device_pointer);
    candidate->logit = create_tensor(plan_.right, logit.device_pointer);
    candidate->sigmoid_output = create_tensor(
        sigmoid_output_, candidate->sigmoid_storage->opaque());
    candidate->output = create_tensor(plan_.output, output.device_pointer);

    std::uint64_t sigmoid_workspace_size = 0;
    aclnnStatus status = aclnnSigmoidGetWorkspaceSize(
        candidate->logit,
        candidate->sigmoid_output,
        &sigmoid_workspace_size,
        &candidate->sigmoid_executor);
    require_query_success(status, "aclnnSigmoidGetWorkspaceSize");
    require_repeatable(candidate->sigmoid_executor, "aclnnSigmoid");

    std::uint64_t backward_workspace_size = 0;
    status = aclnnSigmoidBackwardGetWorkspaceSize(
        candidate->gradient,
        candidate->sigmoid_output,
        candidate->output,
        &backward_workspace_size,
        &candidate->backward_executor);
    require_query_success(
        status, "aclnnSigmoidBackwardGetWorkspaceSize");
    require_repeatable(
        candidate->backward_executor, "aclnnSigmoidBackward");

    if (sigmoid_workspace_size > std::numeric_limits<std::size_t>::max() ||
        backward_workspace_size > std::numeric_limits<std::size_t>::max()) {
      throw std::overflow_error(
          "ACLNN sigmoid backward benchmark workspace overflows size_t");
    }
    candidate->sigmoid_workspace_size =
        static_cast<std::size_t>(sigmoid_workspace_size);
    candidate->backward_workspace_size =
        static_cast<std::size_t>(backward_workspace_size);
    candidate->workspace_size = std::max(candidate->sigmoid_workspace_size,
                                         candidate->backward_workspace_size);
    std::unique_ptr<State> old = std::move(state_);
    state_ = std::move(candidate);
    old.reset();
  }

  [[nodiscard]] std::size_t workspace_size() const noexcept override {
    return state_ == nullptr ? 0 : state_->workspace_size;
  }

  void execute(std::span<const flagdnnBinding_t> bindings,
               void* workspace,
               std::size_t workspace_size,
               flagdnnStream_t stream) override {
    if (state_ == nullptr) {
      throw std::logic_error(
          "ACLNN sigmoid backward benchmark execute called before prepare");
    }
    const flagdnnBinding_t& gradient =
        find_binding(bindings, plan_.left.uid);
    const flagdnnBinding_t& logit =
        find_binding(bindings, plan_.right.uid);
    const flagdnnBinding_t& output =
        find_binding(bindings, plan_.output.uid);
    if (gradient.device_pointer != state_->gradient_pointer ||
        logit.device_pointer != state_->logit_pointer ||
        output.device_pointer != state_->output_pointer ||
        reinterpret_cast<aclrtStream>(stream) != state_->stream) {
      throw std::invalid_argument(
          "ACLNN sigmoid backward benchmark binding or stream changed");
    }
    if (workspace_size < state_->workspace_size ||
        (state_->workspace_size != 0 && workspace == nullptr)) {
      throw std::invalid_argument(
          "ACLNN sigmoid backward benchmark workspace is too small");
    }

    aclnnStatus status = aclnnSigmoid(
        state_->sigmoid_workspace_size == 0 ? nullptr : workspace,
        state_->sigmoid_workspace_size,
        state_->sigmoid_executor,
        state_->stream);
    if (status != 0) {
      throw std::runtime_error(
          aclnn_error_message(status, "aclnnSigmoid"));
    }
    status = aclnnSigmoidBackward(
        state_->backward_workspace_size == 0 ? nullptr : workspace,
        state_->backward_workspace_size,
        state_->backward_executor,
        state_->stream);
    if (status != 0) {
      throw std::runtime_error(
          aclnn_error_message(status, "aclnnSigmoidBackward"));
    }
  }

 private:
  static void require_query_success(aclnnStatus status,
                                    std::string_view operation) {
    if (status == 0) {
      return;
    }
    const std::string message = aclnn_error_message(status, operation);
    if (explicitly_unsupported(status, message)) {
      throw AclnnBenchmarkUnsupportedError(status, message);
    }
    throw std::runtime_error(message);
  }

  static void require_repeatable(aclOpExecutor* executor,
                                 std::string_view operation) {
    const aclnnStatus status = aclSetAclOpExecutorRepeatable(executor);
    if (status != 0) {
      throw std::runtime_error(aclnn_error_message(
          status, std::string(operation) + " repeatable executor"));
    }
  }

  struct State {
    ~State() {
      if (backward_executor != nullptr) {
        log_destroy_status(aclDestroyAclOpExecutor(backward_executor),
                           "aclDestroyAclOpExecutor(sigmoid backward)");
      }
      if (sigmoid_executor != nullptr) {
        log_destroy_status(aclDestroyAclOpExecutor(sigmoid_executor),
                           "aclDestroyAclOpExecutor(sigmoid)");
      }
      if (output != nullptr) {
        log_destroy_status(aclDestroyTensor(output),
                           "aclDestroyTensor(sigmoid backward output)");
      }
      if (sigmoid_output != nullptr) {
        log_destroy_status(aclDestroyTensor(sigmoid_output),
                           "aclDestroyTensor(sigmoid output)");
      }
      if (logit != nullptr) {
        log_destroy_status(aclDestroyTensor(logit),
                           "aclDestroyTensor(sigmoid backward logit)");
      }
      if (gradient != nullptr) {
        log_destroy_status(aclDestroyTensor(gradient),
                           "aclDestroyTensor(sigmoid backward gradient)");
      }
    }

    std::unique_ptr<flagdnn::validation::ascend::DeviceBuffer>
        sigmoid_storage;
    aclTensor* gradient = nullptr;
    aclTensor* logit = nullptr;
    aclTensor* sigmoid_output = nullptr;
    aclTensor* output = nullptr;
    aclOpExecutor* sigmoid_executor = nullptr;
    aclOpExecutor* backward_executor = nullptr;
    void* gradient_pointer = nullptr;
    void* logit_pointer = nullptr;
    void* output_pointer = nullptr;
    aclrtStream stream = nullptr;
    std::size_t sigmoid_workspace_size = 0;
    std::size_t backward_workspace_size = 0;
    std::size_t workspace_size = 0;
  };

  BenchmarkCase specification_;
  AclnnBinaryPointwiseBenchmarkPlan plan_;
  TensorSpec sigmoid_output_;
  std::unique_ptr<State> state_;
};
#endif

class AclnnBinaryPointwiseBenchmarkExecutable final
    : public BenchmarkExecutable {
 public:
  AclnnBinaryPointwiseBenchmarkExecutable(
      BenchmarkCase specification,
      AclnnBinaryPointwiseBenchmarkPlan plan)
      : specification_(std::move(specification)),
        plan_(std::move(plan)),
        alpha_(plan_.alpha) {}

  void prepare(std::span<const flagdnnBinding_t> bindings,
               flagdnnStream_t stream) override {
    if (stream == nullptr) {
      throw std::invalid_argument("ACLNN benchmark prepare stream is null");
    }
    if (state_ != nullptr) {
      flagdnn::validation::ascend::check_acl(
          aclrtSynchronizeStream(state_->stream),
          "aclrtSynchronizeStream(before ACLNN benchmark reprepare)");
    }
    const flagdnnBinding_t& left = find_binding(bindings, plan_.left.uid);
    const flagdnnBinding_t& right = find_binding(bindings, plan_.right.uid);
    const flagdnnBinding_t& output = find_binding(bindings, plan_.output.uid);
    require_aligned(left.device_pointer, left.uid);
    require_aligned(right.device_pointer, right.uid);
    require_aligned(output.device_pointer, output.uid);

    auto candidate = std::make_unique<State>();
    candidate->left_pointer = left.device_pointer;
    candidate->right_pointer = right.device_pointer;
    candidate->output_pointer = output.device_pointer;
    candidate->stream = reinterpret_cast<aclrtStream>(stream);
    candidate->left = create_tensor(plan_.left, left.device_pointer);
    candidate->right = create_tensor(plan_.right, right.device_pointer);
    candidate->output = create_tensor(plan_.output, output.device_pointer);
    std::uint64_t workspace_size = 0;
    aclnnStatus query_status = 0;
    if (plan_.mode == FLAGDNN_POINTWISE_ADD ||
        plan_.mode == FLAGDNN_POINTWISE_SUB) {
      candidate->alpha = aclCreateScalar(&alpha_, ACL_DOUBLE);
      if (candidate->alpha == nullptr) {
        throw std::runtime_error("aclCreateScalar(alpha) returned null");
      }
    }
    if (plan_.mode == FLAGDNN_POINTWISE_ADD) {
      query_status = aclnnAddGetWorkspaceSize(candidate->left,
                                              candidate->right,
                                              candidate->alpha,
                                              candidate->output,
                                              &workspace_size,
                                              &candidate->executor);
    } else if (plan_.mode == FLAGDNN_POINTWISE_SUB) {
      query_status = aclnnSubGetWorkspaceSize(candidate->left,
                                              candidate->right,
                                              candidate->alpha,
                                              candidate->output,
                                              &workspace_size,
                                              &candidate->executor);
    } else if (plan_.mode == FLAGDNN_POINTWISE_MUL) {
      query_status = aclnnMulGetWorkspaceSize(candidate->left,
                                              candidate->right,
                                              candidate->output,
                                              &workspace_size,
                                              &candidate->executor);
    } else if (plan_.mode == FLAGDNN_POINTWISE_DIV) {
      query_status = aclnnDivGetWorkspaceSize(candidate->left,
                                              candidate->right,
                                              candidate->output,
                                              &workspace_size,
                                              &candidate->executor);
    } else if (plan_.mode == FLAGDNN_POINTWISE_MOD) {
      query_status = aclnnFmodTensorGetWorkspaceSize(candidate->left,
                                                     candidate->right,
                                                     candidate->output,
                                                     &workspace_size,
                                                     &candidate->executor);
    } else if (plan_.mode == FLAGDNN_POINTWISE_POW) {
      query_status = aclnnPowTensorTensorGetWorkspaceSize(
          candidate->left,
          candidate->right,
          candidate->output,
          &workspace_size,
          &candidate->executor);
    } else if (plan_.mode == FLAGDNN_POINTWISE_MIN) {
      query_status = aclnnMinimumGetWorkspaceSize(candidate->left,
                                                  candidate->right,
                                                  candidate->output,
                                                  &workspace_size,
                                                  &candidate->executor);
    } else if (plan_.mode == FLAGDNN_POINTWISE_MAX) {
      query_status = aclnnMaximumGetWorkspaceSize(candidate->left,
                                                  candidate->right,
                                                  candidate->output,
                                                  &workspace_size,
                                                  &candidate->executor);
    } else if (plan_.mode == FLAGDNN_POINTWISE_CMP_EQ) {
      query_status = aclnnEqTensorGetWorkspaceSize(candidate->left,
                                                   candidate->right,
                                                   candidate->output,
                                                   &workspace_size,
                                                   &candidate->executor);
    } else if (plan_.mode == FLAGDNN_POINTWISE_CMP_NEQ) {
      query_status = aclnnNeTensorGetWorkspaceSize(candidate->left,
                                                   candidate->right,
                                                   candidate->output,
                                                   &workspace_size,
                                                   &candidate->executor);
    } else if (plan_.mode == FLAGDNN_POINTWISE_CMP_GT) {
      query_status = aclnnGtTensorGetWorkspaceSize(candidate->left,
                                                   candidate->right,
                                                   candidate->output,
                                                   &workspace_size,
                                                   &candidate->executor);
    } else if (plan_.mode == FLAGDNN_POINTWISE_CMP_GE) {
      query_status = aclnnGeTensorGetWorkspaceSize(candidate->left,
                                                   candidate->right,
                                                   candidate->output,
                                                   &workspace_size,
                                                   &candidate->executor);
    } else if (plan_.mode == FLAGDNN_POINTWISE_CMP_LT) {
      // CANN's Lt header omits BF16; swapped Gt is semantically identical.
      query_status = aclnnGtTensorGetWorkspaceSize(candidate->right,
                                                   candidate->left,
                                                   candidate->output,
                                                   &workspace_size,
                                                   &candidate->executor);
    } else if (plan_.mode == FLAGDNN_POINTWISE_CMP_LE) {
      // Preserve the same BF16 reference path with swapped Ge.
      query_status = aclnnGeTensorGetWorkspaceSize(candidate->right,
                                                   candidate->left,
                                                   candidate->output,
                                                   &workspace_size,
                                                   &candidate->executor);
    } else if (plan_.mode == FLAGDNN_POINTWISE_LOGICAL_AND) {
      query_status = aclnnLogicalAndGetWorkspaceSize(candidate->left,
                                                     candidate->right,
                                                     candidate->output,
                                                     &workspace_size,
                                                     &candidate->executor);
    } else if (plan_.mode == FLAGDNN_POINTWISE_LOGICAL_OR) {
      query_status = aclnnLogicalOrGetWorkspaceSize(candidate->left,
                                                    candidate->right,
                                                    candidate->output,
                                                    &workspace_size,
                                                    &candidate->executor);
    } else {
      throw std::invalid_argument(
          "ACLNN benchmark prepare received an unknown mode");
    }
    if (query_status != 0) {
      const std::string message = aclnn_error_message(
          query_status,
          "aclnn" + operation_name(plan_.mode) + "GetWorkspaceSize");
      if (explicitly_unsupported(query_status, message)) {
        throw AclnnBenchmarkUnsupportedError(query_status, message);
      }
      throw std::runtime_error(message);
    }
    const aclnnStatus repeatable_status =
        aclSetAclOpExecutorRepeatable(candidate->executor);
    if (repeatable_status != 0) {
      throw std::runtime_error(aclnn_error_message(
          repeatable_status, "aclSetAclOpExecutorRepeatable"));
    }
    if (workspace_size > std::numeric_limits<std::size_t>::max()) {
      throw std::overflow_error("ACLNN workspace size overflows size_t");
    }
    candidate->workspace_size = static_cast<std::size_t>(workspace_size);
    std::unique_ptr<State> old = std::move(state_);
    state_ = std::move(candidate);
    old.reset();
  }

  [[nodiscard]] std::size_t workspace_size() const noexcept override {
    return state_ == nullptr ? 0 : state_->workspace_size;
  }

  void execute(std::span<const flagdnnBinding_t> bindings,
               void* workspace,
               std::size_t workspace_size,
               flagdnnStream_t stream) override {
    if (state_ == nullptr) {
      throw std::logic_error("ACLNN benchmark execute called before prepare");
    }
    const flagdnnBinding_t& left = find_binding(bindings, plan_.left.uid);
    const flagdnnBinding_t& right = find_binding(bindings, plan_.right.uid);
    const flagdnnBinding_t& output = find_binding(bindings, plan_.output.uid);
    if (left.device_pointer != state_->left_pointer ||
        right.device_pointer != state_->right_pointer ||
        output.device_pointer != state_->output_pointer ||
        reinterpret_cast<aclrtStream>(stream) != state_->stream) {
      throw std::invalid_argument(
          "ACLNN repeatable benchmark binding or stream changed");
    }
    if (workspace_size < state_->workspace_size ||
        (state_->workspace_size != 0 && workspace == nullptr)) {
      throw std::invalid_argument("ACLNN benchmark workspace is too small");
    }
    void* effective_workspace =
        state_->workspace_size == 0 ? nullptr : workspace;
    aclnnStatus status = 0;
    if (plan_.mode == FLAGDNN_POINTWISE_ADD) {
      status = aclnnAdd(effective_workspace,
                        state_->workspace_size,
                        state_->executor,
                        state_->stream);
    } else if (plan_.mode == FLAGDNN_POINTWISE_SUB) {
      status = aclnnSub(effective_workspace,
                        state_->workspace_size,
                        state_->executor,
                        state_->stream);
    } else if (plan_.mode == FLAGDNN_POINTWISE_MUL) {
      status = aclnnMul(effective_workspace,
                        state_->workspace_size,
                        state_->executor,
                        state_->stream);
    } else if (plan_.mode == FLAGDNN_POINTWISE_DIV) {
      status = aclnnDiv(effective_workspace,
                        state_->workspace_size,
                        state_->executor,
                        state_->stream);
    } else if (plan_.mode == FLAGDNN_POINTWISE_MOD) {
      status = aclnnFmodTensor(effective_workspace,
                               state_->workspace_size,
                               state_->executor,
                               state_->stream);
    } else if (plan_.mode == FLAGDNN_POINTWISE_POW) {
      status = aclnnPowTensorTensor(effective_workspace,
                                    state_->workspace_size,
                                    state_->executor,
                                    state_->stream);
    } else if (plan_.mode == FLAGDNN_POINTWISE_MIN) {
      status = aclnnMinimum(effective_workspace,
                            state_->workspace_size,
                            state_->executor,
                            state_->stream);
    } else if (plan_.mode == FLAGDNN_POINTWISE_MAX) {
      status = aclnnMaximum(effective_workspace,
                            state_->workspace_size,
                            state_->executor,
                            state_->stream);
    } else if (plan_.mode == FLAGDNN_POINTWISE_CMP_EQ) {
      status = aclnnEqTensor(effective_workspace,
                             state_->workspace_size,
                             state_->executor,
                             state_->stream);
    } else if (plan_.mode == FLAGDNN_POINTWISE_CMP_NEQ) {
      status = aclnnNeTensor(effective_workspace,
                             state_->workspace_size,
                             state_->executor,
                             state_->stream);
    } else if (plan_.mode == FLAGDNN_POINTWISE_CMP_GT ||
               plan_.mode == FLAGDNN_POINTWISE_CMP_LT) {
      status = aclnnGtTensor(effective_workspace,
                             state_->workspace_size,
                             state_->executor,
                             state_->stream);
    } else if (plan_.mode == FLAGDNN_POINTWISE_CMP_GE ||
               plan_.mode == FLAGDNN_POINTWISE_CMP_LE) {
      status = aclnnGeTensor(effective_workspace,
                             state_->workspace_size,
                             state_->executor,
                             state_->stream);
    } else if (plan_.mode == FLAGDNN_POINTWISE_LOGICAL_AND) {
      status = aclnnLogicalAnd(effective_workspace,
                               state_->workspace_size,
                               state_->executor,
                               state_->stream);
    } else if (plan_.mode == FLAGDNN_POINTWISE_LOGICAL_OR) {
      status = aclnnLogicalOr(effective_workspace,
                              state_->workspace_size,
                              state_->executor,
                              state_->stream);
    } else {
      throw std::invalid_argument(
          "ACLNN benchmark execute received an unknown mode");
    }
    if (status != 0) {
      throw std::runtime_error(aclnn_error_message(
          status, "aclnn" + operation_name(plan_.mode)));
    }
  }

 private:
  struct State {
    ~State() {
      if (executor != nullptr) {
        log_destroy_status(aclDestroyAclOpExecutor(executor),
                           "aclDestroyAclOpExecutor");
      }
      if (output != nullptr) {
        log_destroy_status(aclDestroyTensor(output),
                           "aclDestroyTensor(output)");
      }
      if (right != nullptr) {
        log_destroy_status(aclDestroyTensor(right),
                           "aclDestroyTensor(right)");
      }
      if (left != nullptr) {
        log_destroy_status(aclDestroyTensor(left),
                           "aclDestroyTensor(left)");
      }
      if (alpha != nullptr) {
        log_destroy_status(aclDestroyScalar(alpha),
                           "aclDestroyScalar(alpha)");
      }
    }

    aclTensor* left = nullptr;
    aclTensor* right = nullptr;
    aclTensor* output = nullptr;
    aclScalar* alpha = nullptr;
    aclOpExecutor* executor = nullptr;
    void* left_pointer = nullptr;
    void* right_pointer = nullptr;
    void* output_pointer = nullptr;
    aclrtStream stream = nullptr;
    std::size_t workspace_size = 0;
  };

  BenchmarkCase specification_;
  AclnnBinaryPointwiseBenchmarkPlan plan_;
  double alpha_ = 1.0;
  std::unique_ptr<State> state_;
};

class AclnnTernaryPointwiseBenchmarkExecutable final
    : public BenchmarkExecutable {
 public:
  explicit AclnnTernaryPointwiseBenchmarkExecutable(
      AclnnTernaryPointwiseBenchmarkPlan plan)
      : plan_(std::move(plan)) {
    if (plan_.mode != FLAGDNN_POINTWISE_BINARY_SELECT) {
      throw std::invalid_argument(
          "ACLNN ternary benchmark requires BINARY_SELECT");
    }
  }

  void prepare(std::span<const flagdnnBinding_t> bindings,
               flagdnnStream_t stream) override {
    if (stream == nullptr) {
      throw std::invalid_argument(
          "ACLNN ternary benchmark prepare stream is null");
    }
    if (state_ != nullptr) {
      flagdnn::validation::ascend::check_acl(
          aclrtSynchronizeStream(state_->stream),
          "aclrtSynchronizeStream(before ACLNN ternary benchmark "
          "reprepare)");
    }
    const flagdnnBinding_t& self = find_binding(bindings, plan_.self.uid);
    const flagdnnBinding_t& other = find_binding(bindings, plan_.other.uid);
    const flagdnnBinding_t& condition =
        find_binding(bindings, plan_.condition.uid);
    const flagdnnBinding_t& output = find_binding(bindings, plan_.output.uid);
    require_aligned(self.device_pointer, self.uid);
    require_aligned(other.device_pointer, other.uid);
    require_aligned(condition.device_pointer, condition.uid);
    require_aligned(output.device_pointer, output.uid);

    auto candidate = std::make_unique<State>();
    candidate->self_pointer = self.device_pointer;
    candidate->other_pointer = other.device_pointer;
    candidate->condition_pointer = condition.device_pointer;
    candidate->output_pointer = output.device_pointer;
    candidate->stream = reinterpret_cast<aclrtStream>(stream);
    candidate->self = create_tensor(plan_.self, self.device_pointer);
    candidate->other = create_tensor(plan_.other, other.device_pointer);
    candidate->condition =
        create_tensor(plan_.condition, condition.device_pointer);
    candidate->output = create_tensor(plan_.output, output.device_pointer);

    std::uint64_t workspace_size = 0;
    const aclnnStatus query_status = aclnnSWhereGetWorkspaceSize(
        candidate->condition,
        candidate->self,
        candidate->other,
        candidate->output,
        &workspace_size,
        &candidate->executor);
    if (query_status != 0) {
      const std::string message = aclnn_error_message(
          query_status, "aclnnSWhereGetWorkspaceSize");
      if (explicitly_unsupported(query_status, message)) {
        throw AclnnBenchmarkUnsupportedError(query_status, message);
      }
      throw std::runtime_error(message);
    }
    const aclnnStatus repeatable_status =
        aclSetAclOpExecutorRepeatable(candidate->executor);
    if (repeatable_status != 0) {
      throw std::runtime_error(aclnn_error_message(
          repeatable_status, "aclSetAclOpExecutorRepeatable"));
    }
    if (workspace_size > std::numeric_limits<std::size_t>::max()) {
      throw std::overflow_error(
          "ACLNN ternary benchmark workspace size overflows size_t");
    }
    candidate->workspace_size = static_cast<std::size_t>(workspace_size);

    std::unique_ptr<State> old = std::move(state_);
    state_ = std::move(candidate);
    old.reset();
  }

  [[nodiscard]] std::size_t workspace_size() const noexcept override {
    return state_ == nullptr ? 0 : state_->workspace_size;
  }

  void execute(std::span<const flagdnnBinding_t> bindings,
               void* workspace,
               std::size_t workspace_size,
               flagdnnStream_t stream) override {
    if (state_ == nullptr) {
      throw std::logic_error(
          "ACLNN ternary benchmark execute called before prepare");
    }
    const flagdnnBinding_t& self = find_binding(bindings, plan_.self.uid);
    const flagdnnBinding_t& other = find_binding(bindings, plan_.other.uid);
    const flagdnnBinding_t& condition =
        find_binding(bindings, plan_.condition.uid);
    const flagdnnBinding_t& output = find_binding(bindings, plan_.output.uid);
    if (self.device_pointer != state_->self_pointer ||
        other.device_pointer != state_->other_pointer ||
        condition.device_pointer != state_->condition_pointer ||
        output.device_pointer != state_->output_pointer) {
      throw std::invalid_argument(
          "ACLNN ternary benchmark repeatable binding address changed");
    }
    if (reinterpret_cast<aclrtStream>(stream) != state_->stream) {
      throw std::invalid_argument(
          "ACLNN ternary benchmark repeatable stream changed");
    }
    if (workspace_size < state_->workspace_size ||
        (state_->workspace_size != 0 && workspace == nullptr)) {
      throw std::invalid_argument(
          "ACLNN ternary benchmark workspace is too small");
    }
    void* effective_workspace =
        state_->workspace_size == 0 ? nullptr : workspace;
    const aclnnStatus status = aclnnSWhere(effective_workspace,
                                           state_->workspace_size,
                                           state_->executor,
                                           state_->stream);
    if (status != 0) {
      throw std::runtime_error(aclnn_error_message(status, "aclnnSWhere"));
    }
  }

 private:
  struct State {
    ~State() {
      if (executor != nullptr) {
        log_destroy_status(aclDestroyAclOpExecutor(executor),
                           "aclDestroyAclOpExecutor");
      }
      if (output != nullptr) {
        log_destroy_status(aclDestroyTensor(output),
                           "aclDestroyTensor(output)");
      }
      if (condition != nullptr) {
        log_destroy_status(aclDestroyTensor(condition),
                           "aclDestroyTensor(condition)");
      }
      if (other != nullptr) {
        log_destroy_status(aclDestroyTensor(other),
                           "aclDestroyTensor(other)");
      }
      if (self != nullptr) {
        log_destroy_status(aclDestroyTensor(self),
                           "aclDestroyTensor(self)");
      }
    }

    aclTensor* self = nullptr;
    aclTensor* other = nullptr;
    aclTensor* condition = nullptr;
    aclTensor* output = nullptr;
    aclOpExecutor* executor = nullptr;
    void* self_pointer = nullptr;
    void* other_pointer = nullptr;
    void* condition_pointer = nullptr;
    void* output_pointer = nullptr;
    aclrtStream stream = nullptr;
    std::size_t workspace_size = 0;
  };

  AclnnTernaryPointwiseBenchmarkPlan plan_;
  std::unique_ptr<State> state_;
};

}  // namespace

AclnnBinaryPointwiseBenchmarkPlan plan_aclnn_binary_pointwise(
    const BenchmarkCase& specification) {
  const flagdnnPointwiseMode_t mode = benchmark_mode(specification);
  if (specification.tensors.size() != 3 ||
      specification.output_count != 1) {
    throw std::invalid_argument(
        "ACLNN provider requires two inputs and one output");
  }
  const double alpha =
      (mode == FLAGDNN_POINTWISE_ADD || mode == FLAGDNN_POINTWISE_SUB)
          ? specification.add_alpha
          : 1.0;
  AclnnBinaryPointwiseBenchmarkPlan result{specification.tensors[0],
                                           specification.tensors[1],
                                           specification.tensors[2],
                                           mode,
                                           alpha,
                                           false};
  const bool comparison = mode == FLAGDNN_POINTWISE_CMP_EQ ||
                          mode == FLAGDNN_POINTWISE_CMP_NEQ ||
                          mode == FLAGDNN_POINTWISE_CMP_GT ||
                          mode == FLAGDNN_POINTWISE_CMP_GE ||
                          mode == FLAGDNN_POINTWISE_CMP_LT ||
                          mode == FLAGDNN_POINTWISE_CMP_LE;
  if (result.left.data_type != result.right.data_type ||
      (comparison ? result.output.data_type != FLAGDNN_DATA_BOOLEAN
                  : result.left.data_type != result.output.data_type)) {
    throw std::invalid_argument(
        "ACLNN binary pointwise benchmark dtypes must match");
  }
  if (mode == FLAGDNN_POINTWISE_SIGMOID_BWD &&
      (result.left.dimensions != result.right.dimensions ||
       result.left.dimensions != result.output.dimensions)) {
    throw std::invalid_argument(
        "ACLNN sigmoid backward benchmark tensors must have equal shapes");
  }
  (void)acl_data_type(result.left.data_type);
  if (!broadcasts_to(result.left, result.output) ||
      !broadcasts_to(result.right, result.output)) {
    throw std::invalid_argument(
        "ACLNN binary pointwise benchmark shapes do not broadcast");
  }
  if (!std::isfinite(alpha)) {
    throw std::invalid_argument(
        "ACLNN binary pointwise benchmark alpha must be finite");
  }
  const std::vector<std::int64_t> contiguous =
      checked_contiguous_strides(result.output.dimensions);
  if (result.output.strides != contiguous) {
    result.output.strides = contiguous;
    result.uses_contiguous_reference_output = true;
  }
  (void)tensor_io::encoded_byte_count(result.left);
  (void)tensor_io::encoded_byte_count(result.right);
  (void)tensor_io::encoded_byte_count(result.output);
  return result;
}

AclnnTernaryPointwiseBenchmarkPlan plan_aclnn_ternary_pointwise(
    const BenchmarkCase& specification) {
  if (specification.operation != Operation::kPointwise ||
      specification.pointwise_mode != FLAGDNN_POINTWISE_BINARY_SELECT ||
      specification.tensors.size() != 4 ||
      specification.output_count != 1) {
    throw std::invalid_argument(
        "ACLNN ternary provider requires BINARY_SELECT, three inputs, and "
        "one output");
  }
  AclnnTernaryPointwiseBenchmarkPlan result{
      specification.tensors[0],
      specification.tensors[1],
      specification.tensors[2],
      specification.tensors[3],
      specification.pointwise_mode};
  if (result.self.data_type != result.other.data_type ||
      result.self.data_type != result.output.data_type ||
      result.condition.data_type != FLAGDNN_DATA_BOOLEAN ||
      result.self.data_type == FLAGDNN_DATA_BOOLEAN) {
    throw std::invalid_argument(
        "ACLNN SWhere benchmark dtypes are invalid");
  }
  (void)acl_data_type(result.self.data_type);
  (void)acl_data_type(result.condition.data_type);
  if (result.output.dimensions.size() > 8 ||
      !has_exact_broadcast_output(result.self,
                                  result.other,
                                  result.condition,
                                  result.output)) {
    throw std::invalid_argument(
        "ACLNN SWhere benchmark output is not the exact broadcast shape");
  }
  (void)tensor_io::encoded_byte_count(result.self);
  (void)tensor_io::encoded_byte_count(result.other);
  (void)tensor_io::encoded_byte_count(result.condition);
  (void)tensor_io::encoded_byte_count(result.output);
  return result;
}

ProviderCapability AclnnProvider::capability(
    const BenchmarkCase& specification) const {
#if FLAGDNN_ASCEND_VALIDATION_ENABLE_CONVOLUTION_FPROP
  if (specification.operation == Operation::kConvolutionFprop) {
    (void)plan_aclnn_convolution_fprop_benchmark(specification);
    return {};
  }
#endif
#if FLAGDNN_ASCEND_VALIDATION_ENABLE_MATMUL
  if (specification.operation == Operation::kMatmul) {
    (void)plan_aclnn_matmul_benchmark(specification);
    return {};
  }
#endif
#if FLAGDNN_ASCEND_VALIDATION_ENABLE_BATCHNORM_INFERENCE
  if (specification.operation == Operation::kBatchnormInference) {
    (void)flagdnn::validation::ascend::plan_batchnorm_inference(
        specification);
    return {};
  }
#endif
  if (specification.operation == Operation::kReduction) {
    (void)plan_aclnn_reduction(specification);
    return {};
  }
  if (is_aclnn_unary_benchmark(specification)) {
    (void)plan_aclnn_unary_pointwise(specification);
    return {};
  }
  if (specification.operation == Operation::kPointwise &&
      specification.pointwise_mode == FLAGDNN_POINTWISE_BINARY_SELECT) {
    (void)plan_aclnn_ternary_pointwise(specification);
    return {};
  }
  if (has_aclnn_binary_benchmark_mapping(specification)) {
    (void)plan_aclnn_binary_pointwise(specification);
    return {};
  }
  return ProviderCapability::unsupported(
      "no exact ACLNN benchmark mapping for this operation");
}

std::unique_ptr<BenchmarkExecutable> AclnnProvider::build(
    const BenchmarkCase& specification) {
#if FLAGDNN_ASCEND_VALIDATION_ENABLE_CONVOLUTION_FPROP
  if (specification.operation == Operation::kConvolutionFprop) {
    return build_aclnn_convolution_fprop(specification);
  }
#endif
#if FLAGDNN_ASCEND_VALIDATION_ENABLE_MATMUL
  if (specification.operation == Operation::kMatmul) {
    return build_aclnn_matmul(specification);
  }
#endif
#if FLAGDNN_ASCEND_VALIDATION_ENABLE_BATCHNORM_INFERENCE
  if (specification.operation == Operation::kBatchnormInference) {
    return build_aclnn_batchnorm_inference(specification);
  }
#endif
  if (specification.operation == Operation::kReduction) {
    return build_aclnn_reduction(specification);
  }
  if (is_aclnn_unary_benchmark(specification)) {
    return build_aclnn_unary_pointwise(specification);
  }
  if (specification.operation == Operation::kPointwise &&
      specification.pointwise_mode == FLAGDNN_POINTWISE_BINARY_SELECT) {
    return std::make_unique<AclnnTernaryPointwiseBenchmarkExecutable>(
        plan_aclnn_ternary_pointwise(specification));
  }
#if FLAGDNN_ASCEND_VALIDATION_ENABLE_OPAPI_NN
  if (specification.operation == Operation::kPointwise &&
      specification.pointwise_mode == FLAGDNN_POINTWISE_SIGMOID_BWD) {
    return std::make_unique<AclnnSigmoidBackwardBenchmarkExecutable>(
        specification, plan_aclnn_binary_pointwise(specification));
  }
#else
  if (specification.operation == Operation::kPointwise &&
      specification.pointwise_mode == FLAGDNN_POINTWISE_SIGMOID_BWD) {
    throw std::invalid_argument(
        "ACLNN sigmoid backward benchmark requires libopapi_nn");
  }
#endif
  return std::make_unique<AclnnBinaryPointwiseBenchmarkExecutable>(
      specification, plan_aclnn_binary_pointwise(specification));
}

}  // namespace flagdnn::benchmarking
