/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "validation/functional/aclnn_rmsnorm.hpp"

#include "validation/acl_runtime.hpp"
#include "validation/tensor_io.hpp"

#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include <aclnnop/aclnn_add.h>
#include <aclnnop/aclnn_rms_norm.h>

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
  if (aclnn_rmsnorm_status_is_unsupported(status, message)) {
    throw AclnnRmsnormUnsupportedError(status, message);
  }
  throw std::runtime_error(message);
}

void log_destroy(aclnnStatus status, std::string_view operation) {
  if (status != 0) {
    std::cerr << "Ascend ACLNN RMSNorm cleanup: "
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
      "ACLNN RMSNorm supports FP32, FP16, and BF16 only");
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
      throw std::invalid_argument("ACLNN RMSNorm binding UID is duplicated");
    }
    result = &binding;
  }
  if (result == nullptr || result->device_pointer == nullptr) {
    throw std::invalid_argument("ACLNN RMSNorm binding is missing or null");
  }
  return *result;
}

void require_aligned(void* pointer, std::int64_t uid) {
  if (reinterpret_cast<std::uintptr_t>(pointer) % 32U != 0U) {
    throw std::invalid_argument(
        "ACLNN RMSNorm binding UID " + std::to_string(uid) +
        " is not 32-byte aligned");
  }
}

aclTensor* create_tensor(const TestTensor& tensor, void* pointer) {
  const std::size_t storage_count = tensor_io::storage_element_count(tensor);
  if (storage_count >
      static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
    throw std::overflow_error("ACLNN RMSNorm storage shape overflows int64");
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
    throw std::runtime_error("aclCreateTensor(RMSNorm) returned null");
  }
  return result;
}

class AclnnRmsnormExecutable final : public NormalizationExecutable {
 public:
  explicit AclnnRmsnormExecutable(const RmsnormTestCase& test_case)
      : plan_(plan_aclnn_rmsnorm(test_case)) {}

  void prepare(std::span<const flagdnnBinding_t> bindings,
               flagdnnStream_t stream) override {
    if (stream == nullptr) {
      throw std::invalid_argument("ACLNN RMSNorm prepare stream is null");
    }
    if (state_ != nullptr) {
      acl::check_acl(
          aclrtSynchronizeStream(state_->stream),
          "aclrtSynchronizeStream(before ACLNN RMSNorm reprepare)");
    }

    const auto& operation = plan_.operation;
    const std::array<const flagdnnBinding_t*, 5> binding = {
        &find_binding(bindings, operation.input.uid),
        &find_binding(bindings, operation.scale.uid),
        &find_binding(bindings, operation.bias.uid),
        &find_binding(bindings, operation.output.uid),
        &find_binding(bindings, operation.inverse_variance.uid),
    };
    for (const flagdnnBinding_t* item : binding) {
      require_aligned(item->device_pointer, item->uid);
    }

    auto candidate = std::make_unique<State>();
    candidate->stream = reinterpret_cast<aclrtStream>(stream);
    for (std::size_t index = 0; index < binding.size(); ++index) {
      candidate->pointers[index] = binding[index]->device_pointer;
    }
    candidate->normalized_buffer =
        std::make_unique<acl::DeviceBuffer>(plan_.normalized_output_bytes);

    candidate->rms_input =
        create_tensor(operation.input, binding[0]->device_pointer);
    candidate->rms_gamma =
        create_tensor(plan_.gamma, binding[1]->device_pointer);
    candidate->rms_output = create_tensor(
        plan_.normalized_output, candidate->normalized_buffer->opaque());
    candidate->rms_inverse_variance = create_tensor(
        operation.inverse_variance, binding[4]->device_pointer);

    std::uint64_t rms_workspace = 0;
    check_aclnn_query(
        aclnnRmsNormGetWorkspaceSize(candidate->rms_input,
                                     candidate->rms_gamma,
                                     operation.epsilon,
                                     candidate->rms_output,
                                     candidate->rms_inverse_variance,
                                     &rms_workspace,
                                     &candidate->rms_executor),
        "aclnnRmsNormGetWorkspaceSize");
    if (candidate->rms_executor == nullptr) {
      throw std::runtime_error("ACLNN RMSNorm returned a null executor");
    }
    check_aclnn(aclSetAclOpExecutorRepeatable(candidate->rms_executor),
                "aclSetAclOpExecutorRepeatable(RMSNorm)");

    candidate->add_left = create_tensor(
        plan_.normalized_output, candidate->normalized_buffer->opaque());
    candidate->add_right =
        create_tensor(operation.bias, binding[2]->device_pointer);
    candidate->add_output =
        create_tensor(operation.output, binding[3]->device_pointer);
    candidate->add_alpha_value = 1.0;
    candidate->add_alpha =
        aclCreateScalar(&candidate->add_alpha_value, ACL_DOUBLE);
    if (candidate->add_alpha == nullptr) {
      throw std::runtime_error("aclCreateScalar(RMSNorm bias alpha) returned null");
    }

    std::uint64_t add_workspace = 0;
    check_aclnn_query(aclnnAddGetWorkspaceSize(candidate->add_left,
                                               candidate->add_right,
                                               candidate->add_alpha,
                                               candidate->add_output,
                                               &add_workspace,
                                               &candidate->add_executor),
                      "aclnnAddGetWorkspaceSize(RMSNorm bias)");
    if (candidate->add_executor == nullptr) {
      throw std::runtime_error("ACLNN RMSNorm bias Add returned a null executor");
    }
    check_aclnn(aclSetAclOpExecutorRepeatable(candidate->add_executor),
                "aclSetAclOpExecutorRepeatable(RMSNorm bias Add)");

    const std::uint64_t required_workspace =
        std::max(rms_workspace, add_workspace);
    if (required_workspace > std::numeric_limits<std::size_t>::max()) {
      throw std::overflow_error("ACLNN RMSNorm workspace overflows size_t");
    }
    candidate->rms_workspace_size = static_cast<std::size_t>(rms_workspace);
    candidate->add_workspace_size = static_cast<std::size_t>(add_workspace);
    candidate->workspace_size =
        static_cast<std::size_t>(required_workspace);

    std::unique_ptr<State> old = std::move(state_);
    state_ = std::move(candidate);
    old.reset();
  }

  [[nodiscard]] std::size_t workspace_size() const noexcept override {
    return state_ == nullptr ? 0U : state_->workspace_size;
  }

  void execute(std::span<const flagdnnBinding_t> bindings,
               void* workspace,
               std::size_t workspace_size,
               flagdnnStream_t stream) override {
    if (state_ == nullptr) {
      throw std::logic_error("ACLNN RMSNorm execute called before prepare");
    }
    const auto& operation = plan_.operation;
    const std::array<std::int64_t, 5> uids = {
        operation.input.uid,
        operation.scale.uid,
        operation.bias.uid,
        operation.output.uid,
        operation.inverse_variance.uid,
    };
    for (std::size_t index = 0; index < uids.size(); ++index) {
      if (find_binding(bindings, uids[index]).device_pointer !=
          state_->pointers[index]) {
        throw std::invalid_argument(
            "ACLNN RMSNorm repeatable binding address changed");
      }
    }
    if (stream == nullptr || reinterpret_cast<aclrtStream>(stream) !=
                                 state_->stream) {
      throw std::invalid_argument("ACLNN RMSNorm repeatable stream changed");
    }
    if (workspace_size < state_->workspace_size ||
        (state_->workspace_size != 0 && workspace == nullptr)) {
      throw std::invalid_argument("ACLNN RMSNorm workspace is too small");
    }
    if (state_->workspace_size != 0 &&
        reinterpret_cast<std::uintptr_t>(workspace) % 32U != 0U) {
      throw std::invalid_argument("ACLNN RMSNorm workspace is not 32-byte aligned");
    }

    void* rms_workspace =
        state_->rms_workspace_size == 0 ? nullptr : workspace;
    check_aclnn(aclnnRmsNorm(rms_workspace,
                             state_->rms_workspace_size,
                             state_->rms_executor,
                             state_->stream),
                "aclnnRmsNorm");
    void* add_workspace =
        state_->add_workspace_size == 0 ? nullptr : workspace;
    check_aclnn(aclnnAdd(add_workspace,
                         state_->add_workspace_size,
                         state_->add_executor,
                         state_->stream),
                "aclnnAdd(RMSNorm bias)");
  }

 private:
  struct State {
    ~State() {
      for (aclOpExecutor* executor : {add_executor, rms_executor}) {
        if (executor != nullptr) {
          log_destroy(aclDestroyAclOpExecutor(executor),
                      "aclDestroyAclOpExecutor(RMSNorm)");
        }
      }
      for (aclTensor* tensor : {add_output,
                                add_right,
                                add_left,
                                rms_inverse_variance,
                                rms_output,
                                rms_gamma,
                                rms_input}) {
        if (tensor != nullptr) {
          log_destroy(aclDestroyTensor(tensor),
                      "aclDestroyTensor(RMSNorm)");
        }
      }
      if (add_alpha != nullptr) {
        log_destroy(aclDestroyScalar(add_alpha),
                    "aclDestroyScalar(RMSNorm bias alpha)");
      }
    }

    std::unique_ptr<acl::DeviceBuffer> normalized_buffer;
    aclTensor* rms_input = nullptr;
    aclTensor* rms_gamma = nullptr;
    aclTensor* rms_output = nullptr;
    aclTensor* rms_inverse_variance = nullptr;
    aclTensor* add_left = nullptr;
    aclTensor* add_right = nullptr;
    aclTensor* add_output = nullptr;
    aclScalar* add_alpha = nullptr;
    aclOpExecutor* rms_executor = nullptr;
    aclOpExecutor* add_executor = nullptr;
    std::array<void*, 5> pointers = {};
    aclrtStream stream = nullptr;
    double add_alpha_value = 1.0;
    std::size_t rms_workspace_size = 0;
    std::size_t add_workspace_size = 0;
    std::size_t workspace_size = 0;
  };

  AclnnRmsnormPlan plan_;
  std::unique_ptr<State> state_;
};

}  // namespace

bool aclnn_rmsnorm_status_is_unsupported(
    std::int32_t status, std::string_view message) {
  if (status == ACL_ERROR_UNSUPPORTED_DATA_TYPE ||
      status == ACL_ERROR_OP_UNSUPPORTED_DYNAMIC ||
      status == ACL_ERROR_API_NOT_SUPPORT ||
      status == ACL_ERROR_FEATURE_UNSUPPORTED) {
    return true;
  }
  std::string lowercase(message);
  std::transform(lowercase.begin(), lowercase.end(), lowercase.begin(),
                 [](unsigned char character) {
                   return static_cast<char>(std::tolower(character));
                 });
  return lowercase.find("not support") != std::string::npos ||
         lowercase.find("unsupported") != std::string::npos;
}

std::unique_ptr<NormalizationExecutable> build_rmsnorm_reference(
    const RmsnormTestCase& test_case) {
  return std::make_unique<AclnnRmsnormExecutable>(test_case);
}

}  // namespace flagdnn::testing
