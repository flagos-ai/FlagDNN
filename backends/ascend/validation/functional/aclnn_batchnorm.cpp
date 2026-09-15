/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "validation/functional/aclnn_batchnorm.hpp"

#include "validation/acl_runtime.hpp"
#include "validation/tensor_io.hpp"

#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include <aclnnop/aclnn_batch_norm.h>

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
namespace io = flagdnn::validation::ascend::tensor_io;

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
  if (aclnn_batchnorm_status_is_unsupported(status, message)) {
    throw AclnnBatchnormUnsupportedError(status, message);
  }
  throw std::runtime_error(message);
}

void log_destroy(aclnnStatus status, std::string_view operation) {
  if (status != 0) {
    std::cerr << "Ascend ACLNN BatchNorm cleanup: "
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
      "ACLNN BatchNorm supports FP32, FP16, and BF16 only");
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
          "ACLNN BatchNorm binding UID is duplicated");
    }
    result = &binding;
  }
  if (result == nullptr || result->device_pointer == nullptr) {
    throw std::invalid_argument("ACLNN BatchNorm binding is missing or null");
  }
  return *result;
}

void require_aligned(void* pointer, std::int64_t uid) {
  if (reinterpret_cast<std::uintptr_t>(pointer) % 32U != 0U) {
    throw std::invalid_argument("ACLNN BatchNorm binding UID " +
                                std::to_string(uid) +
                                " is not 32-byte aligned");
  }
}

aclTensor* create_data_tensor(const TestTensor& tensor, void* pointer) {
  const std::size_t storage_count = io::storage_element_count(tensor);
  if (storage_count >
      static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
    throw std::overflow_error("ACLNN BatchNorm storage shape overflows int64");
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
    throw std::runtime_error("aclCreateTensor(BatchNorm data) returned null");
  }
  return result;
}

aclTensor* create_parameter_tensor(std::size_t channels,
                                   flagdnnDataType_t data_type,
                                   void* pointer) {
  if (channels >
      static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
    throw std::overflow_error("ACLNN BatchNorm channel count overflows int64");
  }
  const std::int64_t dimension = static_cast<std::int64_t>(channels);
  constexpr std::int64_t stride = 1;
  aclTensor* result = aclCreateTensor(&dimension,
                                      1,
                                      acl_data_type(data_type),
                                      &stride,
                                      0,
                                      ACL_FORMAT_NCHW,
                                      &dimension,
                                      1,
                                      pointer);
  if (result == nullptr) {
    throw std::runtime_error(
        "aclCreateTensor(BatchNorm parameter) returned null");
  }
  return result;
}

class AclnnBatchnormExecutable final : public NormalizationExecutable {
 public:
  explicit AclnnBatchnormExecutable(const BatchnormTestCase& test_case)
      : plan_(plan_aclnn_batchnorm(test_case)) {}

  void prepare(std::span<const flagdnnBinding_t> bindings,
               flagdnnStream_t stream) override {
    if (stream == nullptr) {
      throw std::invalid_argument("ACLNN BatchNorm prepare stream is null");
    }
    if (state_ != nullptr) {
      acl::check_acl(aclrtSynchronizeStream(state_->stream),
                     "aclrtSynchronizeStream(before BatchNorm reprepare)");
    }
    const acl::BatchnormPlan& operation = plan_.operation;
    const std::array<std::int64_t, 10> uids = {
        operation.input.uid,
        operation.scale.uid,
        operation.bias.uid,
        operation.previous_running_mean.uid,
        operation.previous_running_variance.uid,
        operation.output.uid,
        operation.mean.uid,
        operation.inverse_standard_deviation.uid,
        operation.next_running_mean.uid,
        operation.next_running_variance.uid,
    };
    std::array<const flagdnnBinding_t*, 10> binding = {};
    for (std::size_t index = 0; index < uids.size(); ++index) {
      binding[index] = &find_binding(bindings, uids[index]);
      require_aligned(binding[index]->device_pointer, binding[index]->uid);
    }
    if (state_ != nullptr && state_->executor != nullptr) {
      for (std::size_t index = 0; index < binding.size(); ++index) {
        if (binding[index]->device_pointer != state_->pointers[index]) {
          throw std::invalid_argument(
              "ACLNN BatchNorm cannot replace an unexecuted plan");
        }
      }
      if (reinterpret_cast<aclrtStream>(stream) != state_->stream) {
        throw std::invalid_argument(
            "ACLNN BatchNorm cannot move an unexecuted plan to another stream");
      }
      return;
    }

    auto candidate = std::make_unique<State>();
    for (std::size_t index = 0; index < binding.size(); ++index) {
      candidate->pointers[index] = binding[index]->device_pointer;
    }
    candidate->stream = reinterpret_cast<aclrtStream>(stream);
    create_tensors(*candidate, operation);

    candidate->workspace_size = create_executor(*candidate, operation);
    candidate->statistic_bytes = io::checked_multiply(
        operation.channels, sizeof(float), "ACLNN BatchNorm statistic bytes");

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
      throw std::logic_error("ACLNN BatchNorm execute called before prepare");
    }
    const acl::BatchnormPlan& operation = plan_.operation;
    const std::array<std::int64_t, 10> uids = {
        operation.input.uid,
        operation.scale.uid,
        operation.bias.uid,
        operation.previous_running_mean.uid,
        operation.previous_running_variance.uid,
        operation.output.uid,
        operation.mean.uid,
        operation.inverse_standard_deviation.uid,
        operation.next_running_mean.uid,
        operation.next_running_variance.uid,
    };
    for (std::size_t index = 0; index < uids.size(); ++index) {
      if (find_binding(bindings, uids[index]).device_pointer !=
          state_->pointers[index]) {
        throw std::invalid_argument(
            "ACLNN BatchNorm prepared binding address changed");
      }
    }
    if (stream == nullptr ||
        reinterpret_cast<aclrtStream>(stream) != state_->stream) {
      throw std::invalid_argument("ACLNN BatchNorm prepared stream changed");
    }
    if (workspace_size < state_->workspace_size ||
        (state_->workspace_size != 0 && workspace == nullptr)) {
      throw std::invalid_argument("ACLNN BatchNorm workspace is too small");
    }
    if (state_->workspace_size != 0 &&
        reinterpret_cast<std::uintptr_t>(workspace) % 32U != 0U) {
      throw std::invalid_argument(
          "ACLNN BatchNorm workspace is not 32-byte aligned");
    }

    acl::check_acl(aclrtMemcpyAsync(state_->pointers[8],
                                    state_->statistic_bytes,
                                    state_->pointers[3],
                                    state_->statistic_bytes,
                                    ACL_MEMCPY_DEVICE_TO_DEVICE,
                                    state_->stream),
                   "aclrtMemcpyAsync(BatchNorm running mean)");
    acl::check_acl(aclrtMemcpyAsync(state_->pointers[9],
                                    state_->statistic_bytes,
                                    state_->pointers[4],
                                    state_->statistic_bytes,
                                    ACL_MEMCPY_DEVICE_TO_DEVICE,
                                    state_->stream),
                   "aclrtMemcpyAsync(BatchNorm running variance)");
    std::size_t execution_workspace_size = state_->workspace_size;
    if (state_->executor == nullptr) {
      create_tensors(*state_, operation);
      execution_workspace_size = create_executor(*state_, operation);
      if (execution_workspace_size > state_->workspace_size) {
        throw std::runtime_error(
            "ACLNN BatchNorm workspace grew after prepare");
      }
    }
    void* effective_workspace =
        execution_workspace_size == 0 ? nullptr : workspace;
    const aclnnStatus execute_status =
        aclnnBatchNorm(effective_workspace,
                       execution_workspace_size,
                       state_->executor,
                       state_->stream);
    // Non-repeatable executors are released by the ACLNN second-phase call.
    // aclDestroyAclOpExecutor is valid only after repeatable mode is enabled.
    state_->executor = nullptr;
    if (execute_status != 0) {
      destroy_tensors(*state_);
      check_aclnn(execute_status, "aclnnBatchNorm");
    }

    // CANN 9.0 BatchNorm executors cannot be marked repeatable. Retire the
    // user-owned descriptors after the asynchronous one-shot call completes;
    // a later execute creates both descriptors and an executor again.
    acl::check_acl(aclrtSynchronizeStream(state_->stream),
                   "aclrtSynchronizeStream(after BatchNorm)");
    destroy_tensors(*state_);
  }

 private:
  struct State {
    ~State() {
      if (executor != nullptr) {
        // ACLNN owns a non-repeatable executor after workspace planning and
        // provides no supported API for abandoning it before execution.
        return;
      }
      for (aclTensor* tensor : {saved_invstd,
                                saved_mean,
                                output,
                                running_variance,
                                running_mean,
                                bias,
                                weight,
                                input}) {
        if (tensor != nullptr) {
          log_destroy(aclDestroyTensor(tensor),
                      "aclDestroyTensor(BatchNorm)");
        }
      }
    }

    aclTensor* input = nullptr;
    aclTensor* weight = nullptr;
    aclTensor* bias = nullptr;
    aclTensor* running_mean = nullptr;
    aclTensor* running_variance = nullptr;
    aclTensor* output = nullptr;
    aclTensor* saved_mean = nullptr;
    aclTensor* saved_invstd = nullptr;
    aclOpExecutor* executor = nullptr;
    std::array<void*, 10> pointers = {};
    aclrtStream stream = nullptr;
    std::size_t workspace_size = 0;
    std::size_t statistic_bytes = 0;
  };

  static void create_tensors(State& state,
                             const acl::BatchnormPlan& operation) {
    if (state.input != nullptr || state.weight != nullptr ||
        state.bias != nullptr || state.running_mean != nullptr ||
        state.running_variance != nullptr || state.output != nullptr ||
        state.saved_mean != nullptr || state.saved_invstd != nullptr) {
      throw std::logic_error(
          "ACLNN BatchNorm tensor descriptors already exist");
    }
    state.input = create_data_tensor(operation.input, state.pointers[0]);
    state.weight = create_parameter_tensor(
        operation.channels, operation.scale.data_type, state.pointers[1]);
    state.bias = create_parameter_tensor(
        operation.channels, operation.bias.data_type, state.pointers[2]);
    state.running_mean = create_parameter_tensor(
        operation.channels, FLAGDNN_DATA_FLOAT32, state.pointers[8]);
    state.running_variance = create_parameter_tensor(
        operation.channels, FLAGDNN_DATA_FLOAT32, state.pointers[9]);
    state.output = create_data_tensor(operation.output, state.pointers[5]);
    state.saved_mean = create_parameter_tensor(
        operation.channels, FLAGDNN_DATA_FLOAT32, state.pointers[6]);
    state.saved_invstd = create_parameter_tensor(
        operation.channels, FLAGDNN_DATA_FLOAT32, state.pointers[7]);
  }

  static void destroy_tensors(State& state) noexcept {
    const std::array<aclTensor**, 8> tensors = {
        &state.saved_invstd,
        &state.saved_mean,
        &state.output,
        &state.running_variance,
        &state.running_mean,
        &state.bias,
        &state.weight,
        &state.input,
    };
    for (aclTensor** slot : tensors) {
      aclTensor* tensor = std::exchange(*slot, nullptr);
      if (tensor != nullptr) {
        log_destroy(aclDestroyTensor(tensor),
                    "aclDestroyTensor(BatchNorm)");
      }
    }
  }

  static std::size_t create_executor(
      State& state, const acl::BatchnormPlan& operation) {
    if (state.executor != nullptr) {
      throw std::logic_error(
          "ACLNN BatchNorm executor already exists");
    }
    std::uint64_t workspace_size = 0;
    check_aclnn_query(aclnnBatchNormGetWorkspaceSize(
                          state.input,
                          state.weight,
                          state.bias,
                          state.running_mean,
                          state.running_variance,
                          true,
                          operation.momentum,
                          operation.epsilon,
                          state.output,
                          state.saved_mean,
                          state.saved_invstd,
                          &workspace_size,
                          &state.executor),
                      "aclnnBatchNormGetWorkspaceSize");
    if (state.executor == nullptr ||
        workspace_size > std::numeric_limits<std::size_t>::max()) {
      throw std::runtime_error("ACLNN BatchNorm returned an invalid plan");
    }
    return static_cast<std::size_t>(workspace_size);
  }

  AclnnBatchnormPlan plan_;
  std::unique_ptr<State> state_;
};

}  // namespace

bool aclnn_batchnorm_status_is_unsupported(
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

AclnnBatchnormPlan plan_aclnn_batchnorm(
    const BatchnormTestCase& test_case) {
  AclnnBatchnormPlan result;
  result.operation = acl::plan_batchnorm(test_case);
  return result;
}

TestTensor batchnorm_reference_data_tensor(const TestTensor& tensor) {
  return acl::batchnorm_reference_data_tensor(tensor);
}

std::unique_ptr<NormalizationExecutable> build_batchnorm_reference(
    const BatchnormTestCase& test_case) {
  return std::make_unique<AclnnBatchnormExecutable>(test_case);
}

}  // namespace flagdnn::testing
