/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "validation/functional/aclnn_add.hpp"

#include "validation/acl_runtime.hpp"
#include "validation/tensor_io.hpp"

#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include <aclnnop/aclnn_add.h>

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
      throw std::overflow_error("ACLNN contiguous stride overflows int64");
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
    case FLAGDNN_DATA_FP8_E8M0:
    case FLAGDNN_DATA_FP8_E4M3:
    case FLAGDNN_DATA_FP8_E5M2:
      break;
  }
  throw std::invalid_argument("ACLNN Add supports FP32, FP16, and BF16 only");
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
    std::cerr << "Ascend ACLNN validation cleanup: "
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
      throw std::invalid_argument("ACLNN Add binding UID is duplicated");
    }
    result = &binding;
  }
  if (result == nullptr || result->device_pointer == nullptr) {
    throw std::invalid_argument("ACLNN Add binding is missing or null");
  }
  return *result;
}

void require_aligned(void* pointer, std::int64_t uid) {
  if (reinterpret_cast<std::uintptr_t>(pointer) % 32U != 0U) {
    throw std::invalid_argument(
        "ACLNN Add binding UID " + std::to_string(uid) +
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

class AclnnAddExecutable final : public AddExecutable {
 public:
  AclnnAddExecutable(const AddTestCase& test_case, AclnnAddPlan plan)
      : test_case_(test_case), plan_(std::move(plan)), alpha_(test_case.alpha) {
    validate_add_case(test_case_);
  }

  ~AclnnAddExecutable() override = default;

  void prepare(std::span<const flagdnnBinding_t> bindings,
               flagdnnStream_t stream) override {
    if (stream == nullptr) {
      throw std::invalid_argument("ACLNN Add prepare stream is null");
    }
    if (state_ != nullptr) {
      flagdnn::validation::ascend::check_acl(
          aclrtSynchronizeStream(state_->stream),
          "aclrtSynchronizeStream(before ACLNN Add reprepare)");
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
    candidate->alpha = aclCreateScalar(&alpha_, ACL_DOUBLE);
    if (candidate->alpha == nullptr) {
      throw std::runtime_error("aclCreateScalar(alpha) returned null");
    }

    std::uint64_t workspace_size = 0;
    const aclnnStatus query_status = aclnnAddGetWorkspaceSize(
        candidate->left,
        candidate->right,
        candidate->alpha,
        candidate->output,
        &workspace_size,
        &candidate->executor);
    if (query_status != 0) {
      const std::string message =
          aclnn_error_message(query_status, "aclnnAddGetWorkspaceSize");
      if (explicitly_unsupported(query_status, message)) {
        throw AclnnUnsupportedError(query_status, message);
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
      throw std::overflow_error("ACLNN Add workspace size overflows size_t");
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
      throw std::logic_error("ACLNN Add execute called before prepare");
    }
    const flagdnnBinding_t& left = find_binding(bindings, plan_.left.uid);
    const flagdnnBinding_t& right = find_binding(bindings, plan_.right.uid);
    const flagdnnBinding_t& output = find_binding(bindings, plan_.output.uid);
    if (left.device_pointer != state_->left_pointer ||
        right.device_pointer != state_->right_pointer ||
        output.device_pointer != state_->output_pointer) {
      throw std::invalid_argument(
          "ACLNN Add repeatable executor binding address changed");
    }
    if (reinterpret_cast<aclrtStream>(stream) != state_->stream) {
      throw std::invalid_argument(
          "ACLNN Add repeatable executor stream changed");
    }
    if (workspace_size < state_->workspace_size ||
        (state_->workspace_size != 0 && workspace == nullptr)) {
      throw std::invalid_argument("ACLNN Add workspace is too small");
    }
    void* effective_workspace =
        state_->workspace_size == 0 ? nullptr : workspace;
    const aclnnStatus status = aclnnAdd(effective_workspace,
                                        state_->workspace_size,
                                        state_->executor,
                                        state_->stream);
    if (status != 0) {
      throw std::runtime_error(aclnn_error_message(status, "aclnnAdd"));
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

  AddTestCase test_case_;
  AclnnAddPlan plan_;
  double alpha_ = 1.0;
  std::unique_ptr<State> state_;
};

}  // namespace

AclnnAddPlan plan_aclnn_add(const AddTestCase& test_case) {
  validate_add_case(test_case);
  AclnnAddPlan result{test_case.left, test_case.right, test_case.output, false};
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

std::unique_ptr<AddExecutable> build_add_reference(
    const AddTestCase& test_case) {
  return std::make_unique<AclnnAddExecutable>(
      test_case, plan_aclnn_add(test_case));
}

}  // namespace flagdnn::testing
