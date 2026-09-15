/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "validation/functional/aclnn_convolution.hpp"

#include "validation/acl_runtime.hpp"
#include "validation/tensor_io.hpp"

#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include <aclnnop/aclnn_convolution.h>

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

void log_destroy_status(aclnnStatus status, std::string_view operation) {
  if (status != 0) {
    std::cerr << "Ascend ACLNN convolution validation cleanup: "
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
      "ACLNN convolution supports FP32, FP16, and BF16 only");
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
          "ACLNN convolution binding UID is duplicated");
    }
    result = &binding;
  }
  if (result == nullptr || result->device_pointer == nullptr) {
    throw std::invalid_argument(
        "ACLNN convolution binding is missing or null");
  }
  return *result;
}

void require_aligned(void* pointer, std::int64_t uid) {
  if (reinterpret_cast<std::uintptr_t>(pointer) % 32U != 0U) {
    throw std::invalid_argument(
        "ACLNN convolution binding UID " + std::to_string(uid) +
        " is not 32-byte aligned");
  }
}

aclFormat convolution_tensor_format(std::size_t rank) {
  switch (rank) {
    case 3U:
      return ACL_FORMAT_NCL;
    case 4U:
      return ACL_FORMAT_NCHW;
    case 5U:
      return ACL_FORMAT_NCDHW;
    default:
      throw std::invalid_argument(
          "ACLNN convolution tensor rank must be 3, 4, or 5");
  }
}

aclTensor* create_tensor(const TestTensor& tensor, void* pointer) {
  const std::size_t storage_count = tensor_io::storage_element_count(tensor);
  if (storage_count >
      static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
    throw std::overflow_error(
        "ACLNN convolution storage shape overflows int64");
  }
  const std::int64_t storage_dimension =
      static_cast<std::int64_t>(storage_count);
  aclTensor* result = aclCreateTensor(tensor.dimensions.data(),
                                      tensor.dimensions.size(),
                                      acl_data_type(tensor.data_type),
                                      tensor.strides.data(),
                                      0,
                                      convolution_tensor_format(
                                          tensor.dimensions.size()),
                                      &storage_dimension,
                                      1,
                                      pointer);
  if (result == nullptr) {
    std::string message = "aclCreateTensor(convolution) returned null";
    const char* recent = aclGetRecentErrMsg();
    if (recent != nullptr && recent[0] != '\0') {
      message += ": ";
      message += recent;
    }
    throw std::runtime_error(std::move(message));
  }
  return result;
}

aclIntArray* create_int_array(std::span<const std::int64_t> values,
                              std::string_view name) {
  aclIntArray* result = aclCreateIntArray(values.data(), values.size());
  if (result == nullptr) {
    throw std::runtime_error("aclCreateIntArray(convolution " +
                             std::string(name) + ") returned null");
  }
  return result;
}

struct CopySpan {
  std::size_t source_offset = 0;
  std::size_t destination_offset = 0;
  std::size_t byte_count = 0;
};

std::size_t explicit_padding_before(const AclnnConvolutionFpropPlan& plan,
                                    std::size_t tensor_axis) {
  const std::size_t spatial_rank = plan.input.dimensions.size() - 2U;
  const std::size_t spatial_axis = tensor_axis - 2U;
  const std::size_t padding_index =
      (spatial_rank - 1U - spatial_axis) * 2U;
  return static_cast<std::size_t>(plan.explicit_padding[padding_index]);
}

std::vector<CopySpan> make_copy_spans(
    const AclnnConvolutionFpropPlan& plan) {
  if (!plan.requires_explicit_padding()) {
    return {};
  }
  const std::size_t element_size =
      tensor_io::data_type_size(plan.input.data_type);
  const std::size_t logical_count = tensor_io::element_count(plan.input);
  std::vector<CopySpan> result;
  result.reserve(logical_count);
  for (std::size_t logical_index = 0; logical_index < logical_count;
       ++logical_index) {
    std::size_t remaining = logical_index;
    std::size_t destination_element = 0;
    for (std::size_t axis = plan.input.dimensions.size(); axis != 0U;
         --axis) {
      const std::size_t current = axis - 1U;
      const std::size_t dimension =
          static_cast<std::size_t>(plan.input.dimensions[current]);
      std::size_t coordinate = remaining % dimension;
      remaining /= dimension;
      if (current >= 2U) {
        coordinate = tensor_io::checked_add(
            coordinate,
            explicit_padding_before(plan, current),
            "ACLNN convolution padded coordinate");
      }
      destination_element = tensor_io::checked_add(
          destination_element,
          tensor_io::checked_multiply(
              coordinate,
              static_cast<std::size_t>(
                  plan.convolution_input.strides[current]),
              "ACLNN convolution padded destination offset"),
          "ACLNN convolution padded destination offset");
    }
    const std::size_t source_element =
        tensor_io::physical_offset_unchecked(logical_index, plan.input);
    const CopySpan next = {
        tensor_io::checked_multiply(source_element,
                                    element_size,
                                    "ACLNN convolution source byte offset"),
        tensor_io::checked_multiply(
            destination_element,
            element_size,
            "ACLNN convolution destination byte offset"),
        element_size};
    if (!result.empty() &&
        result.back().source_offset + result.back().byte_count ==
            next.source_offset &&
        result.back().destination_offset + result.back().byte_count ==
            next.destination_offset) {
      result.back().byte_count = tensor_io::checked_add(
          result.back().byte_count,
          next.byte_count,
          "ACLNN convolution coalesced copy size");
    } else {
      result.push_back(next);
    }
  }
  return result;
}

class AclnnConvolutionExecutable final : public ConvolutionExecutable {
 public:
  explicit AclnnConvolutionExecutable(const ConvolutionTestCase& test_case)
      : plan_(plan_aclnn_convolution_fprop(test_case)),
        copy_spans_(make_copy_spans(plan_)) {}

  void prepare(std::span<const flagdnnBinding_t> bindings,
               flagdnnStream_t stream) override {
    if (stream == nullptr) {
      throw std::invalid_argument("ACLNN convolution prepare stream is null");
    }
    if (state_ != nullptr) {
      acl::check_acl(
          aclrtSynchronizeStream(state_->stream),
          "aclrtSynchronizeStream(before ACLNN convolution reprepare)");
    }
    const flagdnnBinding_t& input =
        find_binding(bindings, plan_.input.uid);
    const flagdnnBinding_t& filter =
        find_binding(bindings, plan_.filter.uid);
    const flagdnnBinding_t& output =
        find_binding(bindings, plan_.output.uid);
    require_aligned(input.device_pointer, input.uid);
    require_aligned(filter.device_pointer, filter.uid);
    require_aligned(output.device_pointer, output.uid);

    auto candidate = std::make_unique<State>();
    candidate->input_pointer = input.device_pointer;
    candidate->filter_pointer = filter.device_pointer;
    candidate->output_pointer = output.device_pointer;
    candidate->stream = reinterpret_cast<aclrtStream>(stream);
    void* convolution_input_pointer = input.device_pointer;
    if (plan_.requires_explicit_padding()) {
      candidate->padded_input = std::make_unique<acl::DeviceBuffer>(
          tensor_io::encoded_byte_count(plan_.convolution_input));
      convolution_input_pointer = candidate->padded_input->opaque();
    }
    candidate->input =
        create_tensor(plan_.convolution_input, convolution_input_pointer);
    candidate->filter = create_tensor(plan_.filter, filter.device_pointer);
    candidate->output = create_tensor(plan_.output, output.device_pointer);
    candidate->stride = create_int_array(plan_.stride, "stride");
    candidate->padding =
        create_int_array(plan_.convolution_padding, "padding");
    candidate->dilation = create_int_array(plan_.dilation, "dilation");
    candidate->output_padding =
        create_int_array(plan_.output_padding, "output padding");

    std::uint64_t workspace_size = 0;
    const aclnnStatus query_status = aclnnConvolutionGetWorkspaceSize(
        candidate->input,
        candidate->filter,
        nullptr,
        candidate->stride,
        candidate->padding,
        candidate->dilation,
        plan_.transposed,
        candidate->output_padding,
        plan_.groups,
        candidate->output,
        plan_.cube_math_type,
        &workspace_size,
        &candidate->executor);
    if (query_status != 0) {
      const std::string message = aclnn_error_message(
          query_status, "aclnnConvolutionGetWorkspaceSize");
      if (aclnn_convolution_status_is_unsupported(query_status, message)) {
        throw AclnnConvolutionUnsupportedError(query_status, message);
      }
      throw std::runtime_error(message);
    }
    if (candidate->executor == nullptr ||
        workspace_size > std::numeric_limits<std::size_t>::max()) {
      throw std::runtime_error("ACLNN convolution returned an invalid plan");
    }
    candidate->workspace_size = static_cast<std::size_t>(workspace_size);
    check_aclnn(aclSetAclOpExecutorRepeatable(candidate->executor),
                "aclSetAclOpExecutorRepeatable(convolution)");

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
      throw std::logic_error(
          "ACLNN convolution execute called before prepare");
    }
    const flagdnnBinding_t& input =
        find_binding(bindings, plan_.input.uid);
    const flagdnnBinding_t& filter =
        find_binding(bindings, plan_.filter.uid);
    const flagdnnBinding_t& output =
        find_binding(bindings, plan_.output.uid);
    if (input.device_pointer != state_->input_pointer ||
        filter.device_pointer != state_->filter_pointer ||
        output.device_pointer != state_->output_pointer) {
      throw std::invalid_argument(
          "ACLNN convolution repeatable executor binding address changed");
    }
    if (reinterpret_cast<aclrtStream>(stream) != state_->stream) {
      throw std::invalid_argument(
          "ACLNN convolution repeatable executor stream changed");
    }
    if (workspace_size < state_->workspace_size ||
        (state_->workspace_size != 0U && workspace == nullptr)) {
      throw std::invalid_argument("ACLNN convolution workspace is too small");
    }

    if (state_->padded_input != nullptr) {
      acl::check_acl(
          aclrtMemsetAsync(state_->padded_input->opaque(),
                           state_->padded_input->size(),
                           0,
                           state_->padded_input->size(),
                           state_->stream),
          "aclrtMemsetAsync(convolution explicit padding)");
      const auto* source =
          static_cast<const std::uint8_t*>(state_->input_pointer);
      auto* destination = static_cast<std::uint8_t*>(
          state_->padded_input->opaque());
      for (const CopySpan& copy : copy_spans_) {
        acl::check_acl(
            aclrtMemcpyAsync(destination + copy.destination_offset,
                             state_->padded_input->size() -
                                 copy.destination_offset,
                             source + copy.source_offset,
                             copy.byte_count,
                             ACL_MEMCPY_DEVICE_TO_DEVICE,
                             state_->stream),
            "aclrtMemcpyAsync(convolution explicit padding)");
      }
    }

    void* effective_workspace =
        state_->workspace_size == 0U ? nullptr : workspace;
    check_aclnn(aclnnConvolution(effective_workspace,
                                 state_->workspace_size,
                                 state_->executor,
                                 state_->stream),
                "aclnnConvolution");
  }

 private:
  struct State {
    ~State() {
      if (executor != nullptr) {
        log_destroy_status(aclDestroyAclOpExecutor(executor),
                           "aclDestroyAclOpExecutor(convolution)");
      }
      for (aclTensor* tensor : {output, filter, input}) {
        if (tensor != nullptr) {
          log_destroy_status(aclDestroyTensor(tensor),
                             "aclDestroyTensor(convolution)");
        }
      }
      for (aclIntArray* array :
           {output_padding, dilation, padding, stride}) {
        if (array != nullptr) {
          log_destroy_status(aclDestroyIntArray(array),
                             "aclDestroyIntArray(convolution)");
        }
      }
    }

    aclTensor* input = nullptr;
    aclTensor* filter = nullptr;
    aclTensor* output = nullptr;
    aclIntArray* stride = nullptr;
    aclIntArray* padding = nullptr;
    aclIntArray* dilation = nullptr;
    aclIntArray* output_padding = nullptr;
    aclOpExecutor* executor = nullptr;
    std::unique_ptr<acl::DeviceBuffer> padded_input;
    void* input_pointer = nullptr;
    void* filter_pointer = nullptr;
    void* output_pointer = nullptr;
    aclrtStream stream = nullptr;
    std::size_t workspace_size = 0;
  };

  AclnnConvolutionFpropPlan plan_;
  std::vector<CopySpan> copy_spans_;
  std::unique_ptr<State> state_;
};

}  // namespace

bool aclnn_convolution_status_is_unsupported(
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

std::unique_ptr<ConvolutionExecutable> build_convolution_reference(
    const ConvolutionTestCase& test_case) {
  return std::make_unique<AclnnConvolutionExecutable>(test_case);
}

}  // namespace flagdnn::testing
