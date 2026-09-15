/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "validation/functional/aclnn_binary_pointwise.hpp"

#include "validation/aclnn_unary_runtime.hpp"
#include "validation/functional/aclnn_ternary_pointwise.hpp"
#include "validation/functional/aclnn_unary_pointwise.hpp"

#include "validation/acl_runtime.hpp"
#include "validation/tensor_io.hpp"

#include <acl/acl.h>
#include <aclnn/acl_meta.h>
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
#include <aclnnop/aclnn_sub.h>

#ifndef FLAGDNN_ASCEND_VALIDATION_ENABLE_OPAPI_NN
#define FLAGDNN_ASCEND_VALIDATION_ENABLE_OPAPI_NN 0
#endif

#if FLAGDNN_ASCEND_VALIDATION_ENABLE_OPAPI_NN
#include <aclnnop/aclnn_sigmoid.h>
#include <aclnnop/aclnn_sigmoid_backward.h>
#endif

#include <algorithm>
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

namespace flagdnn::testing {
namespace {

namespace tensor_io = flagdnn::validation::ascend::tensor_io;

std::vector<std::int64_t> contiguous_strides(
    const std::vector<std::int64_t>& dimensions) {
  std::vector<std::int64_t> result(dimensions.size());
  std::int64_t stride = 1;
  for (std::size_t axis = dimensions.size(); axis != 0; --axis) {
    result[axis - 1] = stride;
    const std::int64_t dimension = dimensions[axis - 1];
    if (dimension > std::numeric_limits<std::int64_t>::max() / stride) {
      throw std::overflow_error(
          "ACLNN binary pointwise contiguous stride overflows int64");
    }
    stride *= dimension;
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
      "ACLNN binary pointwise supports FP32, FP16, BF16, and BOOLEAN only");
}

std::string operation_name(flagdnnPointwiseMode_t mode) {
  switch (mode) {
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
    default:
      throw std::invalid_argument(
          "ACLNN binary pointwise mode is unsupported");
  }
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
    std::cerr << "Ascend ACLNN pointwise validation cleanup: "
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
      throw std::invalid_argument(
          "ACLNN pointwise binding UID is duplicated");
    }
    result = &binding;
  }
  if (result == nullptr || result->device_pointer == nullptr) {
    throw std::invalid_argument(
        "ACLNN pointwise binding is missing or null");
  }
  return *result;
}

void require_aligned(void* pointer, std::int64_t uid) {
  if (reinterpret_cast<std::uintptr_t>(pointer) % 32U != 0U) {
    throw std::invalid_argument(
        "ACLNN pointwise binding UID " + std::to_string(uid) +
        " is not 32-byte aligned");
  }
}

aclTensor* create_tensor(const TestTensor& tensor, void* pointer) {
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
    std::string message = "aclCreateTensor returned null";
    const char* recent = aclGetRecentErrMsg();
    if (recent != nullptr && recent[0] != '\0') {
      message += ": ";
      message += recent;
    }
    throw std::runtime_error(std::move(message));
  }
  return result;
}

#if FLAGDNN_ASCEND_VALIDATION_ENABLE_OPAPI_NN
class AclnnSigmoidBackwardExecutable final : public PointwiseExecutable {
 public:
  AclnnSigmoidBackwardExecutable(PointwiseTestCase test_case,
                                 AclnnBinaryPointwisePlan plan)
      : test_case_(std::move(test_case)), plan_(std::move(plan)) {
    validate_pointwise_case(test_case_);
    if (test_case_.mode != FLAGDNN_POINTWISE_SIGMOID_BWD) {
      throw std::invalid_argument(
          "ACLNN sigmoid backward executable received a different mode");
    }
    sigmoid_output_ = TestTensor{0,
                                 plan_.right.data_type,
                                 plan_.right.dimensions,
                                 contiguous_strides(plan_.right.dimensions)};
    (void)tensor_io::encoded_byte_count(sigmoid_output_);
  }

  void prepare(std::span<const flagdnnBinding_t> bindings,
               flagdnnStream_t stream) override {
    if (stream == nullptr) {
      throw std::invalid_argument(
          "ACLNN sigmoid backward prepare stream is null");
    }
    if (state_ != nullptr) {
      flagdnn::validation::ascend::check_acl(
          aclrtSynchronizeStream(state_->stream),
          "aclrtSynchronizeStream(before ACLNN sigmoid backward reprepare)");
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
    require_query_success(
        status, "aclnnSigmoidGetWorkspaceSize");
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
          "ACLNN sigmoid backward workspace size overflows size_t");
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
          "ACLNN sigmoid backward execute called before prepare");
    }
    const flagdnnBinding_t& gradient =
        find_binding(bindings, plan_.left.uid);
    const flagdnnBinding_t& logit =
        find_binding(bindings, plan_.right.uid);
    const flagdnnBinding_t& output =
        find_binding(bindings, plan_.output.uid);
    if (gradient.device_pointer != state_->gradient_pointer ||
        logit.device_pointer != state_->logit_pointer ||
        output.device_pointer != state_->output_pointer) {
      throw std::invalid_argument(
          "ACLNN sigmoid backward repeatable executor binding changed");
    }
    if (reinterpret_cast<aclrtStream>(stream) != state_->stream) {
      throw std::invalid_argument(
          "ACLNN sigmoid backward repeatable executor stream changed");
    }
    if (workspace_size < state_->workspace_size ||
        (state_->workspace_size != 0 && workspace == nullptr)) {
      throw std::invalid_argument(
          "ACLNN sigmoid backward workspace is too small");
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
      throw AclnnPointwiseUnsupportedError(status, message);
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

  PointwiseTestCase test_case_;
  AclnnBinaryPointwisePlan plan_;
  TestTensor sigmoid_output_;
  std::unique_ptr<State> state_;
};
#endif

class AclnnBinaryPointwiseExecutable final : public PointwiseExecutable {
 public:
  AclnnBinaryPointwiseExecutable(PointwiseTestCase test_case,
                                 AclnnBinaryPointwisePlan plan)
      : test_case_(std::move(test_case)),
        plan_(std::move(plan)),
        alpha_(test_case_.alpha) {
    validate_pointwise_case(test_case_);
    (void)operation_name(test_case_.mode);
  }

  void prepare(std::span<const flagdnnBinding_t> bindings,
               flagdnnStream_t stream) override {
    if (stream == nullptr) {
      throw std::invalid_argument("ACLNN pointwise prepare stream is null");
    }
    if (state_ != nullptr) {
      flagdnn::validation::ascend::check_acl(
          aclrtSynchronizeStream(state_->stream),
          "aclrtSynchronizeStream(before ACLNN pointwise reprepare)");
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
    switch (test_case_.mode) {
      case FLAGDNN_POINTWISE_SUB:
        candidate->alpha = aclCreateScalar(&alpha_, ACL_DOUBLE);
        if (candidate->alpha == nullptr) {
          throw std::runtime_error("aclCreateScalar(alpha) returned null");
        }
        query_status = aclnnSubGetWorkspaceSize(candidate->left,
                                                candidate->right,
                                                candidate->alpha,
                                                candidate->output,
                                                &workspace_size,
                                                &candidate->executor);
        break;
      case FLAGDNN_POINTWISE_MUL:
        query_status = aclnnMulGetWorkspaceSize(candidate->left,
                                                candidate->right,
                                                candidate->output,
                                                &workspace_size,
                                                &candidate->executor);
        break;
      case FLAGDNN_POINTWISE_DIV:
        query_status = aclnnDivGetWorkspaceSize(candidate->left,
                                                candidate->right,
                                                candidate->output,
                                                &workspace_size,
                                                &candidate->executor);
        break;
      case FLAGDNN_POINTWISE_MOD:
        query_status = aclnnFmodTensorGetWorkspaceSize(candidate->left,
                                                       candidate->right,
                                                       candidate->output,
                                                       &workspace_size,
                                                       &candidate->executor);
        break;
      case FLAGDNN_POINTWISE_POW:
        query_status = aclnnPowTensorTensorGetWorkspaceSize(
            candidate->left,
            candidate->right,
            candidate->output,
            &workspace_size,
            &candidate->executor);
        break;
      case FLAGDNN_POINTWISE_MIN:
        query_status = aclnnMinimumGetWorkspaceSize(candidate->left,
                                                    candidate->right,
                                                    candidate->output,
                                                    &workspace_size,
                                                    &candidate->executor);
        break;
      case FLAGDNN_POINTWISE_MAX:
        query_status = aclnnMaximumGetWorkspaceSize(candidate->left,
                                                    candidate->right,
                                                    candidate->output,
                                                    &workspace_size,
                                                    &candidate->executor);
        break;
      case FLAGDNN_POINTWISE_CMP_EQ:
        query_status = aclnnEqTensorGetWorkspaceSize(candidate->left,
                                                     candidate->right,
                                                     candidate->output,
                                                     &workspace_size,
                                                     &candidate->executor);
        break;
      case FLAGDNN_POINTWISE_CMP_NEQ:
        query_status = aclnnNeTensorGetWorkspaceSize(candidate->left,
                                                     candidate->right,
                                                     candidate->output,
                                                     &workspace_size,
                                                     &candidate->executor);
        break;
      case FLAGDNN_POINTWISE_CMP_GT:
        query_status = aclnnGtTensorGetWorkspaceSize(candidate->left,
                                                     candidate->right,
                                                     candidate->output,
                                                     &workspace_size,
                                                     &candidate->executor);
        break;
      case FLAGDNN_POINTWISE_CMP_GE:
        query_status = aclnnGeTensorGetWorkspaceSize(candidate->left,
                                                     candidate->right,
                                                     candidate->output,
                                                     &workspace_size,
                                                     &candidate->executor);
        break;
      case FLAGDNN_POINTWISE_CMP_LT:
        // CANN's Lt header does not advertise BF16.  Swapping Gt preserves
        // IEEE comparison semantics, including NaN and signed zero.
        query_status = aclnnGtTensorGetWorkspaceSize(candidate->right,
                                                     candidate->left,
                                                     candidate->output,
                                                     &workspace_size,
                                                     &candidate->executor);
        break;
      case FLAGDNN_POINTWISE_CMP_LE:
        // Use the BF16-capable Ge primitive for Le by swapping operands.
        query_status = aclnnGeTensorGetWorkspaceSize(candidate->right,
                                                     candidate->left,
                                                     candidate->output,
                                                     &workspace_size,
                                                     &candidate->executor);
        break;
      case FLAGDNN_POINTWISE_LOGICAL_AND:
        query_status = aclnnLogicalAndGetWorkspaceSize(candidate->left,
                                                       candidate->right,
                                                       candidate->output,
                                                       &workspace_size,
                                                       &candidate->executor);
        break;
      case FLAGDNN_POINTWISE_LOGICAL_OR:
        query_status = aclnnLogicalOrGetWorkspaceSize(candidate->left,
                                                      candidate->right,
                                                      candidate->output,
                                                      &workspace_size,
                                                      &candidate->executor);
        break;
      default:
        throw std::invalid_argument(
            "ACLNN binary pointwise prepare received an unknown mode");
    }
    if (query_status != 0) {
      const std::string api =
          "aclnn" + operation_name(test_case_.mode) + "GetWorkspaceSize";
      const std::string message = aclnn_error_message(query_status, api);
      if (explicitly_unsupported(query_status, message)) {
        throw AclnnPointwiseUnsupportedError(query_status, message);
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
          "ACLNN pointwise workspace size overflows size_t");
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
          "ACLNN pointwise execute called before prepare");
    }
    const flagdnnBinding_t& left = find_binding(bindings, plan_.left.uid);
    const flagdnnBinding_t& right = find_binding(bindings, plan_.right.uid);
    const flagdnnBinding_t& output = find_binding(bindings, plan_.output.uid);
    if (left.device_pointer != state_->left_pointer ||
        right.device_pointer != state_->right_pointer ||
        output.device_pointer != state_->output_pointer) {
      throw std::invalid_argument(
          "ACLNN pointwise repeatable executor binding address changed");
    }
    if (reinterpret_cast<aclrtStream>(stream) != state_->stream) {
      throw std::invalid_argument(
          "ACLNN pointwise repeatable executor stream changed");
    }
    if (workspace_size < state_->workspace_size ||
        (state_->workspace_size != 0 && workspace == nullptr)) {
      throw std::invalid_argument("ACLNN pointwise workspace is too small");
    }
    void* effective_workspace =
        state_->workspace_size == 0 ? nullptr : workspace;
    aclnnStatus status = 0;
    switch (test_case_.mode) {
      case FLAGDNN_POINTWISE_SUB:
        status = aclnnSub(effective_workspace,
                          state_->workspace_size,
                          state_->executor,
                          state_->stream);
        break;
      case FLAGDNN_POINTWISE_MUL:
        status = aclnnMul(effective_workspace,
                          state_->workspace_size,
                          state_->executor,
                          state_->stream);
        break;
      case FLAGDNN_POINTWISE_DIV:
        status = aclnnDiv(effective_workspace,
                          state_->workspace_size,
                          state_->executor,
                          state_->stream);
        break;
      case FLAGDNN_POINTWISE_MOD:
        status = aclnnFmodTensor(effective_workspace,
                                 state_->workspace_size,
                                 state_->executor,
                                 state_->stream);
        break;
      case FLAGDNN_POINTWISE_POW:
        status = aclnnPowTensorTensor(effective_workspace,
                                      state_->workspace_size,
                                      state_->executor,
                                      state_->stream);
        break;
      case FLAGDNN_POINTWISE_MIN:
        status = aclnnMinimum(effective_workspace,
                              state_->workspace_size,
                              state_->executor,
                              state_->stream);
        break;
      case FLAGDNN_POINTWISE_MAX:
        status = aclnnMaximum(effective_workspace,
                              state_->workspace_size,
                              state_->executor,
                              state_->stream);
        break;
      case FLAGDNN_POINTWISE_CMP_EQ:
        status = aclnnEqTensor(effective_workspace,
                               state_->workspace_size,
                               state_->executor,
                               state_->stream);
        break;
      case FLAGDNN_POINTWISE_CMP_NEQ:
        status = aclnnNeTensor(effective_workspace,
                               state_->workspace_size,
                               state_->executor,
                               state_->stream);
        break;
      case FLAGDNN_POINTWISE_CMP_GT:
      case FLAGDNN_POINTWISE_CMP_LT:
        status = aclnnGtTensor(effective_workspace,
                               state_->workspace_size,
                               state_->executor,
                               state_->stream);
        break;
      case FLAGDNN_POINTWISE_CMP_GE:
      case FLAGDNN_POINTWISE_CMP_LE:
        status = aclnnGeTensor(effective_workspace,
                               state_->workspace_size,
                               state_->executor,
                               state_->stream);
        break;
      case FLAGDNN_POINTWISE_LOGICAL_AND:
        status = aclnnLogicalAnd(effective_workspace,
                                 state_->workspace_size,
                                 state_->executor,
                                 state_->stream);
        break;
      case FLAGDNN_POINTWISE_LOGICAL_OR:
        status = aclnnLogicalOr(effective_workspace,
                                state_->workspace_size,
                                state_->executor,
                                state_->stream);
        break;
      default:
        throw std::invalid_argument(
            "ACLNN binary pointwise execute received an unknown mode");
    }
    if (status != 0) {
      throw std::runtime_error(aclnn_error_message(
          status, "aclnn" + operation_name(test_case_.mode)));
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

  PointwiseTestCase test_case_;
  AclnnBinaryPointwisePlan plan_;
  double alpha_ = 1.0;
  std::unique_ptr<State> state_;
};

}  // namespace

AclnnBinaryPointwisePlan plan_aclnn_binary_pointwise(
    const PointwiseTestCase& test_case) {
  validate_pointwise_case(test_case);
  if ((test_case.mode != FLAGDNN_POINTWISE_SUB &&
       test_case.mode != FLAGDNN_POINTWISE_MUL &&
       test_case.mode != FLAGDNN_POINTWISE_DIV &&
       test_case.mode != FLAGDNN_POINTWISE_MOD &&
       test_case.mode != FLAGDNN_POINTWISE_POW &&
       test_case.mode != FLAGDNN_POINTWISE_MIN &&
       test_case.mode != FLAGDNN_POINTWISE_MAX &&
       test_case.mode != FLAGDNN_POINTWISE_CMP_EQ &&
       test_case.mode != FLAGDNN_POINTWISE_CMP_NEQ &&
       test_case.mode != FLAGDNN_POINTWISE_CMP_GT &&
       test_case.mode != FLAGDNN_POINTWISE_CMP_GE &&
       test_case.mode != FLAGDNN_POINTWISE_CMP_LT &&
       test_case.mode != FLAGDNN_POINTWISE_CMP_LE &&
       test_case.mode != FLAGDNN_POINTWISE_LOGICAL_AND &&
       test_case.mode != FLAGDNN_POINTWISE_LOGICAL_OR &&
       test_case.mode != FLAGDNN_POINTWISE_SIGMOID_BWD) ||
      test_case.inputs.size() != 2) {
    throw std::invalid_argument(
        "ACLNN binary pointwise requires a supported mode with two inputs");
  }
  AclnnBinaryPointwisePlan result{test_case.inputs[0],
                                  test_case.inputs[1],
                                  test_case.output,
                                  false};
  if (test_case.mode == FLAGDNN_POINTWISE_SIGMOID_BWD &&
      (result.left.dimensions != result.right.dimensions ||
       result.left.dimensions != result.output.dimensions)) {
    throw std::invalid_argument(
        "ACLNN sigmoid backward tensors must have equal shapes");
  }
  (void)acl_data_type(result.left.data_type);
  const std::vector<std::int64_t> contiguous =
      contiguous_strides(result.output.dimensions);
  if (result.output.strides != contiguous) {
    result.output.strides = contiguous;
    result.uses_contiguous_reference_output = true;
  }
  (void)tensor_io::encoded_byte_count(result.left);
  (void)tensor_io::encoded_byte_count(result.right);
  (void)tensor_io::encoded_byte_count(result.output);
  return result;
}

std::unique_ptr<PointwiseExecutable> build_pointwise_reference(
    const PointwiseTestCase& test_case) {
  if (flagdnn::validation::ascend::is_aclnn_unary_mode(test_case.mode)) {
    return build_aclnn_unary_pointwise_reference(test_case);
  }
  if (test_case.mode == FLAGDNN_POINTWISE_BINARY_SELECT) {
    return build_aclnn_ternary_pointwise_reference(test_case);
  }
#if FLAGDNN_ASCEND_VALIDATION_ENABLE_OPAPI_NN
  if (test_case.mode == FLAGDNN_POINTWISE_SIGMOID_BWD) {
    return std::make_unique<AclnnSigmoidBackwardExecutable>(
        test_case, plan_aclnn_binary_pointwise(test_case));
  }
#else
  if (test_case.mode == FLAGDNN_POINTWISE_SIGMOID_BWD) {
    throw std::invalid_argument(
        "ACLNN sigmoid backward reference requires libopapi_nn");
  }
#endif
  return std::make_unique<AclnnBinaryPointwiseExecutable>(
      test_case, plan_aclnn_binary_pointwise(test_case));
}

}  // namespace flagdnn::testing
