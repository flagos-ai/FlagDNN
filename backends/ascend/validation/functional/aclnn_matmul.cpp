/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "validation/functional/aclnn_matmul.hpp"

#include "validation/acl_runtime.hpp"
#include "validation/tensor_io.hpp"

#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include <aclnnop/aclnn_matmul.h>

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

namespace flagdnn::testing {
namespace {

namespace acl = flagdnn::validation::ascend;
namespace tensor_io = flagdnn::validation::ascend::tensor_io;

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

void check_aclnn(aclnnStatus status, std::string_view operation) {
  if (status != 0) {
    throw std::runtime_error(aclnn_error_message(status, operation));
  }
}

void check_aclnn_query(aclnnStatus status, std::string_view operation) {
  if (status == 0) {
    return;
  }
  const std::string message = aclnn_error_message(status, operation);
  if (aclnn_matmul_status_is_unsupported(status, message)) {
    throw AclnnMatmulUnsupportedError(status, message);
  }
  throw std::runtime_error(message);
}

void log_destroy_status(aclnnStatus status, std::string_view operation) {
  if (status != 0) {
    std::cerr << "Ascend ACLNN MatMul validation cleanup: "
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
    case FLAGDNN_DATA_FP8_E8M0:
    case FLAGDNN_DATA_FP8_E4M3:
    case FLAGDNN_DATA_FP8_E5M2:
      break;
  }
  throw std::invalid_argument(
      "ACLNN MatMul supports FP32, FP16, and BF16 only");
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
      throw std::invalid_argument("ACLNN MatMul binding UID is duplicated");
    }
    result = &binding;
  }
  if (result == nullptr || result->device_pointer == nullptr) {
    throw std::invalid_argument("ACLNN MatMul binding is missing or null");
  }
  return *result;
}

void require_aligned(void* pointer, std::int64_t uid) {
  if (reinterpret_cast<std::uintptr_t>(pointer) % 32U != 0U) {
    throw std::invalid_argument(
        "ACLNN MatMul binding UID " + std::to_string(uid) +
        " is not 32-byte aligned");
  }
}

aclTensor* create_tensor(const TestTensor& tensor, void* pointer) {
  const std::size_t storage_count = tensor_io::storage_element_count(tensor);
  if (storage_count >
      static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
    throw std::overflow_error("ACLNN MatMul storage shape overflows int64");
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
    std::string message = "aclCreateTensor(MatMul) returned null";
    const char* recent = aclGetRecentErrMsg();
    if (recent != nullptr && recent[0] != '\0') {
      message += ": ";
      message += recent;
    }
    throw std::runtime_error(std::move(message));
  }
  return result;
}

class AclnnMatmulExecutable final : public MatmulExecutable {
 public:
  explicit AclnnMatmulExecutable(const MatmulTestCase& test_case)
      : plan_(plan_aclnn_matmul(test_case)) {}

  void prepare(std::span<const flagdnnBinding_t> bindings,
               flagdnnStream_t stream) override {
    if (stream == nullptr) {
      throw std::invalid_argument("ACLNN MatMul prepare stream is null");
    }
    if (state_ != nullptr) {
      acl::check_acl(aclrtSynchronizeStream(state_->stream),
                     "aclrtSynchronizeStream(before ACLNN MatMul reprepare)");
    }
    const flagdnnBinding_t& a = find_binding(bindings, plan_.a.uid);
    const flagdnnBinding_t& b = find_binding(bindings, plan_.b.uid);
    const flagdnnBinding_t& output =
        find_binding(bindings, plan_.output.uid);
    require_aligned(a.device_pointer, a.uid);
    require_aligned(b.device_pointer, b.uid);
    require_aligned(output.device_pointer, output.uid);

    auto candidate = std::make_unique<State>();
    candidate->a_pointer = a.device_pointer;
    candidate->b_pointer = b.device_pointer;
    candidate->output_pointer = output.device_pointer;
    candidate->stream = reinterpret_cast<aclrtStream>(stream);
    candidate->a = create_tensor(plan_.a, a.device_pointer);
    candidate->b = create_tensor(plan_.b, b.device_pointer);
    candidate->output = create_tensor(plan_.output, output.device_pointer);
    std::uint64_t workspace_size = 0;
    check_aclnn_query(aclnnMatmulGetWorkspaceSize(candidate->a,
                                                   candidate->b,
                                                   candidate->output,
                                                   plan_.cube_math_type,
                                                   &workspace_size,
                                                   &candidate->executor),
                      "aclnnMatmulGetWorkspaceSize");
    if (candidate->executor == nullptr ||
        workspace_size > std::numeric_limits<std::size_t>::max()) {
      throw std::runtime_error("ACLNN MatMul returned an invalid plan");
    }
    candidate->workspace_size = static_cast<std::size_t>(workspace_size);
    check_aclnn(aclSetAclOpExecutorRepeatable(candidate->executor),
                "aclSetAclOpExecutorRepeatable(MatMul)");

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
      throw std::logic_error("ACLNN MatMul execute called before prepare");
    }
    const flagdnnBinding_t& a = find_binding(bindings, plan_.a.uid);
    const flagdnnBinding_t& b = find_binding(bindings, plan_.b.uid);
    const flagdnnBinding_t& output =
        find_binding(bindings, plan_.output.uid);
    if (a.device_pointer != state_->a_pointer ||
        b.device_pointer != state_->b_pointer ||
        output.device_pointer != state_->output_pointer) {
      throw std::invalid_argument(
          "ACLNN MatMul repeatable executor binding address changed");
    }
    if (reinterpret_cast<aclrtStream>(stream) != state_->stream) {
      throw std::invalid_argument(
          "ACLNN MatMul repeatable executor stream changed");
    }
    if (workspace_size < state_->workspace_size ||
        (state_->workspace_size != 0 && workspace == nullptr)) {
      throw std::invalid_argument("ACLNN MatMul workspace is too small");
    }
    void* effective_workspace =
        state_->workspace_size == 0 ? nullptr : workspace;
    check_aclnn(aclnnMatmul(effective_workspace,
                            state_->workspace_size,
                            state_->executor,
                            state_->stream),
                "aclnnMatmul");
  }

 private:
  struct State {
    ~State() {
      if (executor != nullptr) {
        log_destroy_status(aclDestroyAclOpExecutor(executor),
                           "aclDestroyAclOpExecutor(MatMul)");
      }
      if (output != nullptr) {
        log_destroy_status(aclDestroyTensor(output),
                           "aclDestroyTensor(MatMul output)");
      }
      if (b != nullptr) {
        log_destroy_status(aclDestroyTensor(b),
                           "aclDestroyTensor(MatMul B)");
      }
      if (a != nullptr) {
        log_destroy_status(aclDestroyTensor(a),
                           "aclDestroyTensor(MatMul A)");
      }
    }

    aclTensor* a = nullptr;
    aclTensor* b = nullptr;
    aclTensor* output = nullptr;
    aclOpExecutor* executor = nullptr;
    void* a_pointer = nullptr;
    void* b_pointer = nullptr;
    void* output_pointer = nullptr;
    aclrtStream stream = nullptr;
    std::size_t workspace_size = 0;
  };

  AclnnMatmulPlan plan_;
  std::unique_ptr<State> state_;
};

}  // namespace

bool aclnn_matmul_status_is_unsupported(
    std::int32_t status, std::string_view message) {
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

std::unique_ptr<MatmulExecutable> build_matmul_reference(
    const MatmulTestCase& test_case) {
  return std::make_unique<AclnnMatmulExecutable>(test_case);
}

}  // namespace flagdnn::testing
