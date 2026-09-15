/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "validation/functional/aclnn_batchnorm_inference.hpp"

#include "validation/acl_runtime.hpp"
#include "validation/batchnorm_inference_validation.hpp"
#include "validation/tensor_io.hpp"

#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include <aclnnop/aclnn_batch_norm_elemt.h>

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

namespace acl = flagdnn::validation::ascend;
namespace tensor_io = flagdnn::validation::ascend::tensor_io;
using acl::BatchnormInferencePlan;

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
  if (aclnn_batchnorm_inference_status_is_unsupported(status, message)) {
    throw AclnnBatchnormInferenceUnsupportedError(status, message);
  }
  throw std::runtime_error(message);
}

void log_destroy(aclnnStatus status, std::string_view operation) {
  if (status != 0) {
    std::cerr << "Ascend ACLNN BatchNorm inference cleanup: "
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
      "ACLNN BatchNorm inference supports FP32, FP16, and BF16 only");
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
          "ACLNN BatchNorm inference binding UID is duplicated");
    }
    result = &binding;
  }
  if (result == nullptr || result->device_pointer == nullptr) {
    throw std::invalid_argument(
        "ACLNN BatchNorm inference binding is missing or null");
  }
  return *result;
}

void require_aligned(void* pointer, std::int64_t uid) {
  if (reinterpret_cast<std::uintptr_t>(pointer) % 32U != 0U) {
    throw std::invalid_argument(
        "ACLNN BatchNorm inference binding UID " + std::to_string(uid) +
        " is not 32-byte aligned");
  }
}

aclTensor* create_data_tensor(const TestTensor& tensor, void* pointer) {
  const std::size_t storage_count =
      tensor_io::storage_element_count(tensor);
  if (storage_count >
      static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
    throw std::overflow_error(
        "ACLNN BatchNorm inference storage shape overflows int64");
  }
  const std::int64_t storage_dimension =
      static_cast<std::int64_t>(storage_count);
  aclTensor* result = aclCreateTensor(tensor.dimensions.data(),
                                      tensor.dimensions.size(),
                                      acl_data_type(tensor.data_type),
                                      tensor.strides.data(),
                                      0,
                                      ACL_FORMAT_NCHW,
                                      &storage_dimension,
                                      1,
                                      pointer);
  if (result == nullptr) {
    throw std::runtime_error(
        "aclCreateTensor(BatchNorm inference data) returned null");
  }
  return result;
}

aclTensor* create_parameter_tensor(std::size_t channels, void* pointer) {
  if (channels >
      static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
    throw std::overflow_error(
        "ACLNN BatchNorm inference channel count overflows int64");
  }
  const std::int64_t dimension = static_cast<std::int64_t>(channels);
  constexpr std::int64_t stride = 1;
  aclTensor* result = aclCreateTensor(&dimension,
                                      1,
                                      ACL_FLOAT,
                                      &stride,
                                      0,
                                      ACL_FORMAT_NCHW,
                                      &dimension,
                                      1,
                                      pointer);
  if (result == nullptr) {
    throw std::runtime_error(
        "aclCreateTensor(BatchNorm inference parameter) returned null");
  }
  return result;
}

class AclnnBatchnormInferenceExecutable final
    : public NormalizationExecutable {
 public:
  explicit AclnnBatchnormInferenceExecutable(
      const BatchnormInferenceTestCase& test_case)
      : plan_(acl::plan_batchnorm_inference(test_case)) {}

  void prepare(std::span<const flagdnnBinding_t> bindings,
               flagdnnStream_t stream) override {
    if (stream == nullptr) {
      throw std::invalid_argument(
          "ACLNN BatchNorm inference prepare stream is null");
    }
    if (state_ != nullptr) {
      acl::check_acl(
          aclrtSynchronizeStream(state_->stream),
          "aclrtSynchronizeStream(before ACLNN BatchNorm inference reprepare)");
    }
    const std::array<const flagdnnBinding_t*, 6> binding = {
        &find_binding(bindings, plan_.input.uid),
        &find_binding(bindings, plan_.weight.uid),
        &find_binding(bindings, plan_.bias.uid),
        &find_binding(bindings, plan_.mean.uid),
        &find_binding(bindings, plan_.inverse_standard_deviation.uid),
        &find_binding(bindings, plan_.output.uid),
    };
    for (const flagdnnBinding_t* item : binding) {
      require_aligned(item->device_pointer, item->uid);
    }

    auto candidate = std::make_unique<State>();
    for (std::size_t index = 0; index < binding.size(); ++index) {
      candidate->pointers[index] = binding[index]->device_pointer;
    }
    candidate->stream = reinterpret_cast<aclrtStream>(stream);
    candidate->input = create_data_tensor(plan_.input, binding[0]->device_pointer);
    const std::size_t channels =
        static_cast<std::size_t>(plan_.input.dimensions[1]);
    candidate->weight =
        create_parameter_tensor(channels, binding[1]->device_pointer);
    candidate->bias =
        create_parameter_tensor(channels, binding[2]->device_pointer);
    candidate->mean =
        create_parameter_tensor(channels, binding[3]->device_pointer);
    candidate->invstd =
        create_parameter_tensor(channels, binding[4]->device_pointer);
    candidate->output =
        create_data_tensor(plan_.output, binding[5]->device_pointer);

    std::uint64_t workspace_size = 0;
    check_aclnn_query(
        aclnnBatchNormElemtGetWorkspaceSize(candidate->input,
                                            candidate->weight,
                                            candidate->bias,
                                            candidate->mean,
                                            candidate->invstd,
                                            0.0,
                                            candidate->output,
                                            &workspace_size,
                                            &candidate->executor),
        "aclnnBatchNormElemtGetWorkspaceSize");
    if (candidate->executor == nullptr ||
        workspace_size > std::numeric_limits<std::size_t>::max()) {
      throw std::runtime_error(
          "ACLNN BatchNorm inference returned an invalid plan");
    }
    candidate->workspace_size = static_cast<std::size_t>(workspace_size);
    check_aclnn(aclSetAclOpExecutorRepeatable(candidate->executor),
                "aclSetAclOpExecutorRepeatable(BatchNorm inference)");

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
          "ACLNN BatchNorm inference execute called before prepare");
    }
    const std::array<std::int64_t, 6> uids = {
        plan_.input.uid,
        plan_.weight.uid,
        plan_.bias.uid,
        plan_.mean.uid,
        plan_.inverse_standard_deviation.uid,
        plan_.output.uid,
    };
    for (std::size_t index = 0; index < uids.size(); ++index) {
      if (find_binding(bindings, uids[index]).device_pointer !=
          state_->pointers[index]) {
        throw std::invalid_argument(
            "ACLNN BatchNorm inference repeatable binding address changed");
      }
    }
    if (stream == nullptr || reinterpret_cast<aclrtStream>(stream) !=
                                 state_->stream) {
      throw std::invalid_argument(
          "ACLNN BatchNorm inference repeatable stream changed");
    }
    if (workspace_size < state_->workspace_size ||
        (state_->workspace_size != 0 && workspace == nullptr)) {
      throw std::invalid_argument(
          "ACLNN BatchNorm inference workspace is too small");
    }
    if (state_->workspace_size != 0 &&
        reinterpret_cast<std::uintptr_t>(workspace) % 32U != 0U) {
      throw std::invalid_argument(
          "ACLNN BatchNorm inference workspace is not 32-byte aligned");
    }
    void* effective_workspace =
        state_->workspace_size == 0 ? nullptr : workspace;
    check_aclnn(aclnnBatchNormElemt(effective_workspace,
                                    state_->workspace_size,
                                    state_->executor,
                                    state_->stream),
                "aclnnBatchNormElemt");
  }

 private:
  struct State {
    ~State() {
      if (executor != nullptr) {
        log_destroy(aclDestroyAclOpExecutor(executor),
                    "aclDestroyAclOpExecutor(BatchNorm inference)");
      }
      for (aclTensor* tensor :
           {output, invstd, mean, bias, weight, input}) {
        if (tensor != nullptr) {
          log_destroy(aclDestroyTensor(tensor),
                      "aclDestroyTensor(BatchNorm inference)");
        }
      }
    }

    aclTensor* input = nullptr;
    aclTensor* weight = nullptr;
    aclTensor* bias = nullptr;
    aclTensor* mean = nullptr;
    aclTensor* invstd = nullptr;
    aclTensor* output = nullptr;
    aclOpExecutor* executor = nullptr;
    std::array<void*, 6> pointers = {};
    aclrtStream stream = nullptr;
    std::size_t workspace_size = 0;
  };

  BatchnormInferencePlan plan_;
  std::unique_ptr<State> state_;
};

}  // namespace

bool aclnn_batchnorm_inference_status_is_unsupported(
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

TestTensor batchnorm_reference_data_tensor(const TestTensor& tensor) {
  return tensor;
}

std::unique_ptr<NormalizationExecutable>
build_batchnorm_inference_reference(
    const BatchnormInferenceTestCase& test_case) {
  return std::make_unique<AclnnBatchnormInferenceExecutable>(test_case);
}

}  // namespace flagdnn::testing
