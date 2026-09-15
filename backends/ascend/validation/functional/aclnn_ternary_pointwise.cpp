/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "validation/functional/aclnn_ternary_pointwise.hpp"

#include "validation/acl_runtime.hpp"
#include "validation/functional/aclnn_binary_pointwise.hpp"
#include "validation/tensor_io.hpp"

#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include <aclnnop/aclnn_s_where.h>

#include <algorithm>
#include <array>
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

namespace flagdnn::testing {
namespace {

namespace tensor_io = flagdnn::validation::ascend::tensor_io;

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
      "ACLNN ternary pointwise supports FP32, FP16, BF16, and BOOLEAN only");
}

bool has_exact_broadcast_output(const TestTensor& self,
                                const TestTensor& other,
                                const TestTensor& condition,
                                const TestTensor& output) {
  const std::array<const TestTensor*, 3> inputs = {
      &self, &other, &condition};
  for (std::size_t axis = 0; axis < output.dimensions.size(); ++axis) {
    std::int64_t expected = 1;
    for (const TestTensor* input : inputs) {
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
    std::cerr << "Ascend ACLNN ternary validation cleanup: "
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
          "ACLNN ternary pointwise binding UID is duplicated");
    }
    result = &binding;
  }
  if (result == nullptr || result->device_pointer == nullptr) {
    throw std::invalid_argument(
        "ACLNN ternary pointwise binding is missing or null");
  }
  return *result;
}

void require_aligned(void* pointer, std::int64_t uid) {
  if (reinterpret_cast<std::uintptr_t>(pointer) % 32U != 0U) {
    throw std::invalid_argument(
        "ACLNN ternary pointwise binding UID " + std::to_string(uid) +
        " is not 32-byte aligned");
  }
}

aclTensor* create_tensor(const TestTensor& tensor, void* pointer) {
  const std::size_t storage_count = tensor_io::storage_element_count(tensor);
  if (storage_count >
      static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
    throw std::overflow_error(
        "ACLNN ternary pointwise storage shape overflows int64");
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
    std::string message =
        "aclCreateTensor returned null for ternary pointwise";
    const char* recent = aclGetRecentErrMsg();
    if (recent != nullptr && recent[0] != '\0') {
      message += ": ";
      message += recent;
    }
    throw std::runtime_error(std::move(message));
  }
  return result;
}

class AclnnTernaryPointwiseExecutable final : public PointwiseExecutable {
 public:
  AclnnTernaryPointwiseExecutable(PointwiseTestCase test_case,
                                  AclnnTernaryPointwisePlan plan)
      : test_case_(std::move(test_case)), plan_(std::move(plan)) {
    validate_pointwise_case(test_case_);
    if (test_case_.mode != FLAGDNN_POINTWISE_BINARY_SELECT) {
      throw std::invalid_argument(
          "ACLNN ternary executable requires BINARY_SELECT");
    }
  }

  void prepare(std::span<const flagdnnBinding_t> bindings,
               flagdnnStream_t stream) override {
    if (stream == nullptr) {
      throw std::invalid_argument(
          "ACLNN ternary pointwise prepare stream is null");
    }
    if (state_ != nullptr) {
      flagdnn::validation::ascend::check_acl(
          aclrtSynchronizeStream(state_->stream),
          "aclrtSynchronizeStream(before ACLNN ternary reprepare)");
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
          "ACLNN ternary pointwise workspace size overflows size_t");
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
          "ACLNN ternary pointwise execute called before prepare");
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
          "ACLNN ternary repeatable executor binding address changed");
    }
    if (reinterpret_cast<aclrtStream>(stream) != state_->stream) {
      throw std::invalid_argument(
          "ACLNN ternary repeatable executor stream changed");
    }
    if (workspace_size < state_->workspace_size ||
        (state_->workspace_size != 0 && workspace == nullptr)) {
      throw std::invalid_argument(
          "ACLNN ternary pointwise workspace is too small");
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

  PointwiseTestCase test_case_;
  AclnnTernaryPointwisePlan plan_;
  std::unique_ptr<State> state_;
};

}  // namespace

AclnnTernaryPointwisePlan plan_aclnn_ternary_pointwise(
    const PointwiseTestCase& test_case) {
  validate_pointwise_case(test_case);
  if (test_case.mode != FLAGDNN_POINTWISE_BINARY_SELECT ||
      test_case.inputs.size() != 3) {
    throw std::invalid_argument(
        "ACLNN ternary pointwise requires BINARY_SELECT with three inputs");
  }
  AclnnTernaryPointwisePlan result{test_case.inputs[0],
                                   test_case.inputs[1],
                                   test_case.inputs[2],
                                   test_case.output};
  if (result.self.data_type != result.other.data_type ||
      result.self.data_type != result.output.data_type ||
      result.condition.data_type != FLAGDNN_DATA_BOOLEAN ||
      result.self.data_type == FLAGDNN_DATA_BOOLEAN) {
    throw std::invalid_argument(
        "ACLNN SWhere validation dtypes are invalid");
  }
  if (result.output.dimensions.size() > 8 ||
      !has_exact_broadcast_output(result.self,
                                  result.other,
                                  result.condition,
                                  result.output)) {
    throw std::invalid_argument(
        "ACLNN SWhere validation output is not the exact broadcast shape");
  }
  (void)acl_data_type(result.self.data_type);
  (void)acl_data_type(result.condition.data_type);
  (void)tensor_io::encoded_byte_count(result.self);
  (void)tensor_io::encoded_byte_count(result.other);
  (void)tensor_io::encoded_byte_count(result.condition);
  (void)tensor_io::encoded_byte_count(result.output);
  return result;
}

std::unique_ptr<PointwiseExecutable>
build_aclnn_ternary_pointwise_reference(
    const PointwiseTestCase& test_case) {
  return std::make_unique<AclnnTernaryPointwiseExecutable>(
      test_case, plan_aclnn_ternary_pointwise(test_case));
}

}  // namespace flagdnn::testing
