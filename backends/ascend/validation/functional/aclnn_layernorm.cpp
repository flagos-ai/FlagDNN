/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "validation/functional/aclnn_layernorm.hpp"

#include "validation/acl_runtime.hpp"
#include "validation/tensor_io.hpp"

#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include <aclnnop/aclnn_layer_norm.h>

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
  if (aclnn_layernorm_status_is_unsupported(status, message)) {
    throw AclnnLayernormUnsupportedError(status, message);
  }
  throw std::runtime_error(message);
}

void log_destroy(aclnnStatus status, std::string_view operation) {
  if (status != 0) {
    std::cerr << "Ascend ACLNN LayerNorm cleanup: "
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
      "ACLNN LayerNorm supports FP32, FP16, and BF16 only");
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
          "ACLNN LayerNorm binding UID is duplicated");
    }
    result = &binding;
  }
  if (result == nullptr || result->device_pointer == nullptr) {
    throw std::invalid_argument("ACLNN LayerNorm binding is missing or null");
  }
  return *result;
}

void require_aligned(void* pointer, std::int64_t uid) {
  if (reinterpret_cast<std::uintptr_t>(pointer) % 32U != 0U) {
    throw std::invalid_argument(
        "ACLNN LayerNorm binding UID " + std::to_string(uid) +
        " is not 32-byte aligned");
  }
}

aclTensor* create_tensor(const TestTensor& tensor, void* pointer) {
  const std::size_t storage_count = tensor_io::storage_element_count(tensor);
  if (storage_count >
      static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
    throw std::overflow_error("ACLNN LayerNorm storage shape overflows int64");
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
    throw std::runtime_error("aclCreateTensor(LayerNorm) returned null");
  }
  return result;
}

TestTensor parameter_view(const TestTensor& tensor,
                          const std::vector<std::int64_t>& dimensions) {
  TestTensor result = tensor;
  result.dimensions = dimensions;
  result.strides.resize(dimensions.size());
  std::int64_t stride = 1;
  for (std::size_t axis = dimensions.size(); axis != 0; --axis) {
    result.strides[axis - 1] = stride;
    stride *= dimensions[axis - 1];
  }
  return result;
}

class AclnnLayernormExecutable final : public NormalizationExecutable {
 public:
  explicit AclnnLayernormExecutable(const LayernormTestCase& test_case)
      : plan_(plan_aclnn_layernorm(test_case)) {}

  void prepare(std::span<const flagdnnBinding_t> bindings,
               flagdnnStream_t stream) override {
    if (stream == nullptr) {
      throw std::invalid_argument("ACLNN LayerNorm prepare stream is null");
    }
    if (state_ != nullptr) {
      acl::check_acl(
          aclrtSynchronizeStream(state_->stream),
          "aclrtSynchronizeStream(before ACLNN LayerNorm reprepare)");
    }
    const auto& operation = plan_.operation;
    const std::array<const flagdnnBinding_t*, 6> binding = {
        &find_binding(bindings, operation.input.uid),
        &find_binding(bindings, operation.scale.uid),
        &find_binding(bindings, operation.bias.uid),
        &find_binding(bindings, operation.output.uid),
        &find_binding(bindings, operation.mean.uid),
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
    candidate->normalized_shape = aclCreateIntArray(
        plan_.normalized_shape.data(), plan_.normalized_shape.size());
    if (candidate->normalized_shape == nullptr) {
      throw std::runtime_error(
          "aclCreateIntArray(LayerNorm normalized shape) returned null");
    }
    const TestTensor scale_view =
        parameter_view(operation.scale, plan_.normalized_shape);
    const TestTensor bias_view =
        parameter_view(operation.bias, plan_.normalized_shape);
    candidate->input = create_tensor(operation.input, binding[0]->device_pointer);
    candidate->scale = create_tensor(scale_view, binding[1]->device_pointer);
    candidate->bias = create_tensor(bias_view, binding[2]->device_pointer);
    candidate->output =
        create_tensor(operation.output, binding[3]->device_pointer);
    candidate->mean = create_tensor(operation.mean, binding[4]->device_pointer);
    candidate->inverse_variance = create_tensor(
        operation.inverse_variance, binding[5]->device_pointer);

    std::uint64_t workspace = 0;
    check_aclnn_query(aclnnLayerNormGetWorkspaceSize(candidate->input,
                                               candidate->normalized_shape,
                                               candidate->scale,
                                               candidate->bias,
                                               operation.epsilon,
                                               candidate->output,
                                               candidate->mean,
                                               candidate->inverse_variance,
                                               &workspace,
                                               &candidate->executor),
                "aclnnLayerNormGetWorkspaceSize");
    if (candidate->executor == nullptr) {
      throw std::runtime_error("ACLNN LayerNorm returned a null executor");
    }
    check_aclnn(aclSetAclOpExecutorRepeatable(candidate->executor),
                "aclSetAclOpExecutorRepeatable(LayerNorm)");
    if (workspace > std::numeric_limits<std::size_t>::max()) {
      throw std::overflow_error("ACLNN LayerNorm workspace overflows size_t");
    }
    candidate->workspace_size = static_cast<std::size_t>(workspace);
    state_ = std::move(candidate);
  }

  [[nodiscard]] std::size_t workspace_size() const noexcept override {
    return state_ == nullptr ? 0U : state_->workspace_size;
  }

  void execute(std::span<const flagdnnBinding_t> bindings,
               void* workspace,
               std::size_t workspace_size,
               flagdnnStream_t stream) override {
    if (state_ == nullptr) {
      throw std::logic_error("ACLNN LayerNorm execute called before prepare");
    }
    const auto& operation = plan_.operation;
    const std::array<std::int64_t, 6> uids = {
        operation.input.uid,
        operation.scale.uid,
        operation.bias.uid,
        operation.output.uid,
        operation.mean.uid,
        operation.inverse_variance.uid,
    };
    for (std::size_t index = 0; index < uids.size(); ++index) {
      if (find_binding(bindings, uids[index]).device_pointer !=
          state_->pointers[index]) {
        throw std::invalid_argument(
            "ACLNN LayerNorm repeatable binding address changed");
      }
    }
    if (stream == nullptr ||
        reinterpret_cast<aclrtStream>(stream) != state_->stream) {
      throw std::invalid_argument("ACLNN LayerNorm repeatable stream changed");
    }
    if (workspace_size < state_->workspace_size ||
        (state_->workspace_size != 0 && workspace == nullptr)) {
      throw std::invalid_argument("ACLNN LayerNorm workspace is too small");
    }
    if (state_->workspace_size != 0 &&
        reinterpret_cast<std::uintptr_t>(workspace) % 32U != 0U) {
      throw std::invalid_argument(
          "ACLNN LayerNorm workspace is not 32-byte aligned");
    }
    check_aclnn(aclnnLayerNorm(state_->workspace_size == 0 ? nullptr
                                                            : workspace,
                               state_->workspace_size,
                               state_->executor,
                               state_->stream),
                "aclnnLayerNorm");
  }

 private:
  struct State {
    ~State() {
      if (executor != nullptr) {
        log_destroy(aclDestroyAclOpExecutor(executor),
                    "aclDestroyAclOpExecutor(LayerNorm)");
      }
      for (aclTensor* tensor : {inverse_variance,
                                mean,
                                output,
                                bias,
                                scale,
                                input}) {
        if (tensor != nullptr) {
          log_destroy(aclDestroyTensor(tensor),
                      "aclDestroyTensor(LayerNorm)");
        }
      }
      if (normalized_shape != nullptr) {
        log_destroy(aclDestroyIntArray(normalized_shape),
                    "aclDestroyIntArray(LayerNorm normalized shape)");
      }
    }

    aclTensor* input = nullptr;
    aclTensor* scale = nullptr;
    aclTensor* bias = nullptr;
    aclTensor* output = nullptr;
    aclTensor* mean = nullptr;
    aclTensor* inverse_variance = nullptr;
    aclIntArray* normalized_shape = nullptr;
    aclOpExecutor* executor = nullptr;
    std::array<void*, 6> pointers = {};
    aclrtStream stream = nullptr;
    std::size_t workspace_size = 0;
  };

  AclnnLayernormPlan plan_;
  std::unique_ptr<State> state_;
};

}  // namespace

bool aclnn_layernorm_status_is_unsupported(
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

std::unique_ptr<NormalizationExecutable> build_layernorm_reference(
    const LayernormTestCase& test_case) {
  return std::make_unique<AclnnLayernormExecutable>(test_case);
}

}  // namespace flagdnn::testing
