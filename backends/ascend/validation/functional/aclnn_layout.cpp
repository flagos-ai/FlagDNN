/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "validation/functional/aclnn_layout.hpp"

#include "validation/acl_runtime.hpp"
#include "validation/tensor_io.hpp"

#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include <aclnnop/aclnn_flatten.h>
#include <aclnnop/aclnn_permute.h>
#include <aclnnop/aclnn_slice_v2.h>

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
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

enum class StageKind { kFlatten, kPermute, kSliceV2 };

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

void check_query(aclnnStatus status, std::string_view operation) {
  if (status == 0) {
    return;
  }
  const std::string message = aclnn_error_message(status, operation);
  if (explicitly_unsupported(status, message)) {
    throw AclnnLayoutUnsupportedError(status, message);
  }
  throw std::runtime_error(message);
}

void check_aclnn(aclnnStatus status, std::string_view operation) {
  if (status != 0) {
    throw std::runtime_error(aclnn_error_message(status, operation));
  }
}

void log_destroy_status(aclnnStatus status, std::string_view operation) {
  if (status != 0) {
    std::cerr << "Ascend ACLNN layout validation cleanup: "
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
      "ACLNN layout supports FP32, FP16, and BF16 only");
}

std::vector<std::int64_t> contiguous_strides(
    const std::vector<std::int64_t>& dimensions) {
  std::vector<std::int64_t> result(dimensions.size());
  std::int64_t stride = 1;
  for (std::size_t axis = dimensions.size(); axis != 0; --axis) {
    const std::int64_t dimension = dimensions[axis - 1];
    if (dimension <= 0 ||
        stride > std::numeric_limits<std::int64_t>::max() / dimension) {
      throw std::overflow_error("ACLNN layout contiguous stride overflows");
    }
    result[axis - 1] = stride;
    stride *= dimension;
  }
  return result;
}

std::int64_t checked_product(std::span<const std::int64_t> dimensions) {
  std::int64_t result = 1;
  for (const std::int64_t dimension : dimensions) {
    if (dimension <= 0 ||
        result > std::numeric_limits<std::int64_t>::max() / dimension) {
      throw std::overflow_error("ACLNN layout shape product overflows");
    }
    result *= dimension;
  }
  return result;
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
      throw std::invalid_argument("ACLNN layout binding UID is duplicated");
    }
    result = &binding;
  }
  if (result == nullptr || result->device_pointer == nullptr) {
    throw std::invalid_argument("ACLNN layout binding is missing or null");
  }
  return *result;
}

void require_aligned(void* pointer, std::int64_t uid) {
  if (reinterpret_cast<std::uintptr_t>(pointer) % 32U != 0U) {
    throw std::invalid_argument(
        "ACLNN layout binding UID " + std::to_string(uid) +
        " is not 32-byte aligned");
  }
}

aclTensor* create_tensor(const TestTensor& tensor, void* pointer) {
  const std::size_t storage_count = tensor_io::storage_element_count(tensor);
  if (storage_count >
      static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
    throw std::overflow_error("ACLNN layout storage shape overflows int64");
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
    std::string message = "aclCreateTensor(layout) returned null";
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
    throw std::runtime_error("aclCreateIntArray(" + std::string(name) +
                             ") returned null");
  }
  return result;
}

bool group_is_collapsible(const TestTensor& tensor,
                          std::size_t begin,
                          std::size_t end) {
  if (begin >= end || end > tensor.dimensions.size()) {
    return false;
  }
  for (std::size_t axis = begin; axis + 1 < end; ++axis) {
    const std::int64_t dimension = tensor.dimensions[axis + 1];
    const std::int64_t inner_stride = tensor.strides[axis + 1];
    if (dimension >
            std::numeric_limits<std::int64_t>::max() / inner_stride ||
        tensor.strides[axis] != dimension * inner_stride) {
      return false;
    }
  }
  return true;
}

std::pair<std::vector<std::int64_t>, std::vector<std::int64_t>>
collapse_output_to_2d(const TestTensor& output) {
  const std::size_t rank = output.dimensions.size();
  if (rank == 1) {
    if (output.dimensions[0] >
        std::numeric_limits<std::int64_t>::max() / output.strides[0]) {
      throw std::overflow_error(
          "ACLNN layout collapsed output stride overflows");
    }
    return {{1, output.dimensions[0]},
            {output.dimensions[0] * output.strides[0],
             output.strides[0]}};
  }
  for (std::size_t split = 1; split < rank; ++split) {
    if (!group_is_collapsible(output, 0, split) ||
        !group_is_collapsible(output, split, rank)) {
      continue;
    }
    return {{checked_product(std::span(output.dimensions).first(split)),
             checked_product(std::span(output.dimensions).subspan(split))},
            {output.strides[split - 1], output.strides.back()}};
  }
  throw std::invalid_argument(
      "ACLNN logical reshape output cannot be represented by a 2D view");
}

std::int64_t direct_flatten_axis(const LayoutTestCase& test_case) {
  if (test_case.output.dimensions.size() != 2) {
    return -1;
  }
  const std::span<const std::int64_t> input(test_case.input.dimensions);
  for (std::size_t axis = 0; axis < input.size(); ++axis) {
    const std::int64_t leading =
        checked_product(input.first(axis));
    const std::int64_t trailing =
        checked_product(input.subspan(axis));
    if (leading == test_case.output.dimensions[0] &&
        trailing == test_case.output.dimensions[1]) {
      return static_cast<std::int64_t>(axis);
    }
  }
  return -1;
}

class AclnnLayoutExecutable final : public LayoutExecutable {
 public:
  explicit AclnnLayoutExecutable(const LayoutTestCase& test_case)
      : plan_(plan_aclnn_layout(test_case)) {}

  void prepare(std::span<const flagdnnBinding_t> bindings,
               flagdnnStream_t stream) override {
    if (stream == nullptr) {
      throw std::invalid_argument("ACLNN layout prepare stream is null");
    }
    if (state_ != nullptr) {
      acl::check_acl(
          aclrtSynchronizeStream(state_->stream),
          "aclrtSynchronizeStream(before ACLNN layout reprepare)");
    }
    const flagdnnBinding_t& input = find_binding(bindings, plan_.input.uid);
    const flagdnnBinding_t& output = find_binding(bindings, plan_.output.uid);
    require_aligned(input.device_pointer, input.uid);
    require_aligned(output.device_pointer, output.uid);

    auto candidate = std::make_unique<State>();
    candidate->tensors.reserve(4);
    candidate->arrays.reserve(4);
    candidate->stages.reserve(2);
    candidate->input_pointer = input.device_pointer;
    candidate->output_pointer = output.device_pointer;
    candidate->stream = reinterpret_cast<aclrtStream>(stream);
    switch (plan_.operation) {
      case LayoutOperation::kReshape:
        prepare_reshape(*candidate, input.device_pointer,
                        output.device_pointer);
        break;
      case LayoutOperation::kTranspose:
        prepare_transpose(*candidate, input.device_pointer,
                          output.device_pointer);
        break;
      case LayoutOperation::kSlice:
        prepare_slice(*candidate, input.device_pointer,
                      output.device_pointer);
        break;
    }
    candidate->workspace_size = 0;
    for (const Stage& stage : candidate->stages) {
      candidate->workspace_size =
          std::max(candidate->workspace_size, stage.workspace_size);
    }

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
      throw std::logic_error("ACLNN layout execute called before prepare");
    }
    const flagdnnBinding_t& input = find_binding(bindings, plan_.input.uid);
    const flagdnnBinding_t& output = find_binding(bindings, plan_.output.uid);
    if (input.device_pointer != state_->input_pointer ||
        output.device_pointer != state_->output_pointer) {
      throw std::invalid_argument(
          "ACLNN layout repeatable executor binding address changed");
    }
    if (reinterpret_cast<aclrtStream>(stream) != state_->stream) {
      throw std::invalid_argument(
          "ACLNN layout repeatable executor stream changed");
    }
    if (workspace_size < state_->workspace_size ||
        (state_->workspace_size != 0 && workspace == nullptr)) {
      throw std::invalid_argument("ACLNN layout workspace is too small");
    }

    for (const Stage& stage : state_->stages) {
      void* effective_workspace =
          stage.workspace_size == 0 ? nullptr : workspace;
      aclnnStatus status = 0;
      std::string_view operation;
      switch (stage.kind) {
        case StageKind::kFlatten:
          operation = "aclnnFlatten";
          status = aclnnFlatten(effective_workspace,
                                stage.workspace_size,
                                stage.executor,
                                state_->stream);
          break;
        case StageKind::kPermute:
          operation = "aclnnPermute";
          status = aclnnPermute(effective_workspace,
                                stage.workspace_size,
                                stage.executor,
                                state_->stream);
          break;
        case StageKind::kSliceV2:
          operation = "aclnnSliceV2";
          status = aclnnSliceV2(effective_workspace,
                                stage.workspace_size,
                                stage.executor,
                                state_->stream);
          break;
      }
      check_aclnn(status, operation);
    }
  }

 private:
  struct Stage {
    StageKind kind = StageKind::kFlatten;
    aclOpExecutor* executor = nullptr;
    std::size_t workspace_size = 0;
  };

  struct State {
    ~State() {
      for (auto iterator = stages.rbegin(); iterator != stages.rend();
           ++iterator) {
        if (iterator->executor != nullptr) {
          log_destroy_status(aclDestroyAclOpExecutor(iterator->executor),
                             "aclDestroyAclOpExecutor(layout)");
        }
      }
      for (auto iterator = arrays.rbegin(); iterator != arrays.rend();
           ++iterator) {
        if (*iterator != nullptr) {
          log_destroy_status(aclDestroyIntArray(*iterator),
                             "aclDestroyIntArray(layout)");
        }
      }
      for (auto iterator = tensors.rbegin(); iterator != tensors.rend();
           ++iterator) {
        if (*iterator != nullptr) {
          log_destroy_status(aclDestroyTensor(*iterator),
                             "aclDestroyTensor(layout)");
        }
      }
      intermediate.reset();
    }

    std::vector<aclTensor*> tensors;
    std::vector<aclIntArray*> arrays;
    std::vector<Stage> stages;
    std::unique_ptr<acl::DeviceBuffer> intermediate;
    void* input_pointer = nullptr;
    void* output_pointer = nullptr;
    aclrtStream stream = nullptr;
    std::size_t workspace_size = 0;
  };

  static void add_flatten_stage(State& state,
                                aclTensor* input,
                                std::int64_t axis,
                                aclTensor* output) {
    state.stages.push_back({StageKind::kFlatten, nullptr, 0});
    Stage& stage = state.stages.back();
    std::uint64_t workspace_size = 0;
    check_query(aclnnFlattenGetWorkspaceSize(input,
                                              axis,
                                              output,
                                              &workspace_size,
                                              &stage.executor),
                "aclnnFlattenGetWorkspaceSize");
    if (stage.executor == nullptr ||
        workspace_size > std::numeric_limits<std::size_t>::max()) {
      throw std::runtime_error("ACLNN Flatten returned an invalid plan");
    }
    stage.workspace_size = static_cast<std::size_t>(workspace_size);
    check_aclnn(aclSetAclOpExecutorRepeatable(stage.executor),
                "aclSetAclOpExecutorRepeatable(Flatten)");
  }

  void prepare_reshape(State& state,
                       void* input_pointer,
                       void* output_pointer) const {
    if (!plan_.materializes_reshape) {
      aclTensor* input = create_tensor(plan_.input, input_pointer);
      state.tensors.push_back(input);
      aclTensor* output = create_tensor(plan_.output, output_pointer);
      state.tensors.push_back(output);
      add_flatten_stage(state, input, plan_.flatten_axis, output);
      return;
    }

    const std::int64_t element_count =
        checked_product(plan_.input.dimensions);
    const TestTensor flat{0,
                          plan_.input.data_type,
                          {1, element_count},
                          {element_count, 1},
                          0};
    const TestTensor target_view{
        0,
        plan_.output.data_type,
        plan_.collapsed_output_dimensions,
        contiguous_strides(plan_.collapsed_output_dimensions),
        0};
    const TestTensor collapsed_output{0,
                                      plan_.output.data_type,
                                      plan_.collapsed_output_dimensions,
                                      plan_.collapsed_output_strides,
                                      0};
    state.intermediate = std::make_unique<acl::DeviceBuffer>(
        tensor_io::encoded_byte_count(flat));
    if (reinterpret_cast<std::uintptr_t>(state.intermediate->opaque()) % 32U !=
        0U) {
      throw std::runtime_error(
          "ACLNN layout intermediate is not 32-byte aligned");
    }

    aclTensor* first_input = create_tensor(plan_.input, input_pointer);
    state.tensors.push_back(first_input);
    aclTensor* first_output =
        create_tensor(flat, state.intermediate->opaque());
    state.tensors.push_back(first_output);
    add_flatten_stage(state, first_input, 0, first_output);

    aclTensor* second_input =
        create_tensor(target_view, state.intermediate->opaque());
    state.tensors.push_back(second_input);
    aclTensor* second_output =
        create_tensor(collapsed_output, output_pointer);
    state.tensors.push_back(second_output);
    add_flatten_stage(state, second_input, 1, second_output);
  }

  void prepare_transpose(State& state,
                         void* input_pointer,
                         void* output_pointer) const {
    aclTensor* input = create_tensor(plan_.input, input_pointer);
    state.tensors.push_back(input);
    aclTensor* output = create_tensor(plan_.output, output_pointer);
    state.tensors.push_back(output);
    aclIntArray* permutation =
        create_int_array(plan_.parameters, "permutation");
    state.arrays.push_back(permutation);

    state.stages.push_back({StageKind::kPermute, nullptr, 0});
    Stage& stage = state.stages.back();
    std::uint64_t workspace_size = 0;
    check_query(aclnnPermuteGetWorkspaceSize(input,
                                              permutation,
                                              output,
                                              &workspace_size,
                                              &stage.executor),
                "aclnnPermuteGetWorkspaceSize");
    if (stage.executor == nullptr ||
        workspace_size > std::numeric_limits<std::size_t>::max()) {
      throw std::runtime_error("ACLNN Permute returned an invalid plan");
    }
    stage.workspace_size = static_cast<std::size_t>(workspace_size);
    check_aclnn(aclSetAclOpExecutorRepeatable(stage.executor),
                "aclSetAclOpExecutorRepeatable(Permute)");
  }

  void prepare_slice(State& state,
                     void* input_pointer,
                     void* output_pointer) const {
    aclTensor* input = create_tensor(plan_.input, input_pointer);
    state.tensors.push_back(input);
    aclTensor* output = create_tensor(plan_.output, output_pointer);
    state.tensors.push_back(output);
    aclIntArray* begin = create_int_array(plan_.slice_begin, "slice begin");
    state.arrays.push_back(begin);
    aclIntArray* end = create_int_array(plan_.slice_end, "slice end");
    state.arrays.push_back(end);
    std::vector<std::int64_t> axes(plan_.slice_begin.size());
    std::iota(axes.begin(), axes.end(), std::int64_t{0});
    aclIntArray* slice_axes = create_int_array(axes, "slice axes");
    state.arrays.push_back(slice_axes);
    aclIntArray* strides =
        create_int_array(plan_.parameters, "slice strides");
    state.arrays.push_back(strides);

    state.stages.push_back({StageKind::kSliceV2, nullptr, 0});
    Stage& stage = state.stages.back();
    std::uint64_t workspace_size = 0;
    check_query(aclnnSliceV2GetWorkspaceSize(input,
                                             begin,
                                             end,
                                             slice_axes,
                                             strides,
                                             output,
                                             &workspace_size,
                                             &stage.executor),
                "aclnnSliceV2GetWorkspaceSize");
    if (stage.executor == nullptr ||
        workspace_size > std::numeric_limits<std::size_t>::max()) {
      throw std::runtime_error(
          "ACLNN SliceV2 returned an invalid plan");
    }
    stage.workspace_size = static_cast<std::size_t>(workspace_size);
    check_aclnn(aclSetAclOpExecutorRepeatable(stage.executor),
                "aclSetAclOpExecutorRepeatable(SliceV2)");
  }

  AclnnLayoutPlan plan_;
  std::unique_ptr<State> state_;
};

}  // namespace

AclnnLayoutPlan plan_aclnn_layout(const LayoutTestCase& test_case) {
  validate_layout_case(test_case);
  AclnnLayoutPlan result;
  result.operation = test_case.operation;
  result.input = test_case.input;
  result.output = test_case.output;
  (void)acl_data_type(result.input.data_type);
  (void)tensor_io::encoded_byte_count(result.input);
  (void)tensor_io::encoded_byte_count(result.output);

  switch (result.operation) {
    case LayoutOperation::kReshape: {
      const std::int64_t axis = direct_flatten_axis(test_case);
      if (axis >= 0) {
        result.flatten_axis = axis;
      } else {
        auto [dimensions, strides] = collapse_output_to_2d(result.output);
        result.collapsed_output_dimensions = std::move(dimensions);
        result.collapsed_output_strides = std::move(strides);
        result.materializes_reshape = true;
      }
      break;
    }
    case LayoutOperation::kTranspose:
      result.parameters = test_case.permutation;
      break;
    case LayoutOperation::kSlice:
      result.parameters = test_case.slice_strides;
      result.slice_begin.reserve(test_case.slices.size());
      result.slice_end.reserve(test_case.slices.size());
      for (const auto& [begin, end] : test_case.slices) {
        result.slice_begin.push_back(begin);
        result.slice_end.push_back(end);
      }
      break;
  }
  return result;
}

std::unique_ptr<LayoutExecutable> build_layout_reference(
    const LayoutTestCase& test_case) {
  return std::make_unique<AclnnLayoutExecutable>(test_case);
}

}  // namespace flagdnn::testing
