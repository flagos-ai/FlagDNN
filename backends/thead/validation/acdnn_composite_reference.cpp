// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "acdnn_composite_reference.hpp"

#include "acdnn_reference.hpp"
#include "numeric_types.hpp"
#include "pointwise_reference.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace flagdnn::validation::thead {
namespace {

constexpr std::string_view kMultiplyPrimitive = "acdnnOpTensor(MUL)";
constexpr std::string_view kAddPrimitive = "acdnnOpTensor(ADD)";
constexpr std::string_view kBackendMultiplyPrimitive =
    "acdnnBackendExecute(POINTWISE_MUL)";
constexpr std::string_view kBackendAddPrimitive =
    "acdnnBackendExecute(POINTWISE_ADD)";

std::size_t align_workspace(std::size_t value) {
  constexpr std::size_t kAlignment = 256;
  if (value > std::numeric_limits<std::size_t>::max() - (kAlignment - 1)) {
    throw std::overflow_error("THead acDNN AddSquare workspace overflows");
  }
  return (value + kAlignment - 1) / kAlignment * kAlignment;
}

std::size_t storage_bytes(const flagdnn::testing::TestTensor &tensor) {
  if ((tensor.data_type != FLAGDNN_DATA_FLOAT32 &&
       tensor.data_type != FLAGDNN_DATA_FLOAT16 &&
       tensor.data_type != FLAGDNN_DATA_BFLOAT16) ||
      tensor.dimensions.empty() ||
      tensor.dimensions.size() != tensor.strides.size()) {
    throw std::invalid_argument(
        "THead acDNN AddSquare DAG requires floating tensors of positive "
        "rank");
  }
  std::uint64_t last = 0;
  for (std::size_t axis = 0; axis < tensor.dimensions.size(); ++axis) {
    const std::int64_t dimension = tensor.dimensions[axis];
    const std::int64_t stride = tensor.strides[axis];
    if (dimension <= 0 || stride <= 0) {
      throw std::invalid_argument(
          "THead acDNN AddSquare tensor geometry is invalid");
    }
    const std::uint64_t extent =
        static_cast<std::uint64_t>(dimension - 1);
    const std::uint64_t step = static_cast<std::uint64_t>(stride);
    if (extent != 0 &&
        step > (std::numeric_limits<std::uint64_t>::max() - last) / extent) {
      throw std::overflow_error("THead acDNN AddSquare storage overflows");
    }
    last += extent * step;
  }
  const std::size_t scalar_bytes = element_size(tensor.data_type);
  if (last >= std::numeric_limits<std::size_t>::max() / scalar_bytes) {
    throw std::overflow_error("THead acDNN AddSquare storage is too large");
  }
  return (static_cast<std::size_t>(last) + 1U) * scalar_bytes;
}

CapabilityRecord operation_capability(std::string primitive,
                                      ReferencePath path,
                                      const CapabilityRecord &source) {
  return {
      .status = source.status,
      .path = path,
      .reference_plan = {std::move(primitive)},
      .constraints = source.constraints,
      .reason_code = source.status == CapabilityStatus::kProbeRequired
                         ? "real_device_qualification_pending"
                         : "",
      .detail = "acDNN stable primitive in AddSquare reference DAG",
  };
}

class AcdnnAddSquare final : public flagdnn::testing::TestExecutable {
 public:
  AcdnnAddSquare(flagdnn::testing::TestTensor left,
                 flagdnn::testing::TestTensor right,
                 flagdnn::testing::TestTensor output,
                 const CapabilityRecord &capability)
      : left_(std::move(left)),
        right_(std::move(right)),
        output_(std::move(output)) {
    const ReferenceSelection selection = select_reference(capability);
    if (!std::holds_alternative<ReferencePlan>(selection)) {
      throw std::invalid_argument(
          "unsupported AddSquare case reached acDNN DAG construction");
    }
    const ReferencePlan &plan = std::get<ReferencePlan>(selection);
    const bool stable =
        plan.path == ReferencePath::kStablePrimitive &&
        plan.primitives ==
            std::vector<std::string>{std::string(kMultiplyPrimitive),
                                     std::string(kAddPrimitive)};
    const bool backend =
        plan.path == ReferencePath::kBackendDescriptor &&
        plan.primitives ==
            std::vector<std::string>{std::string(kBackendMultiplyPrimitive),
                                     std::string(kBackendAddPrimitive)};
    if (!stable && !backend) {
      throw std::invalid_argument("THead AddSquare acDNN DAG plan mismatch");
    }
    const std::int64_t maximum_uid =
        std::max({left_.uid, right_.uid, output_.uid});
    if (maximum_uid > std::numeric_limits<std::int64_t>::max() - 2) {
      throw std::overflow_error("THead AddSquare reference UID overflows");
    }
    intermediate_ = output_;
    intermediate_.uid = maximum_uid + 1;
    right_alias_ = right_;
    right_alias_.uid = maximum_uid + 2;

    multiply_ = make_acdnn_pointwise_reference(
        {.mode = FLAGDNN_POINTWISE_MUL,
         .inputs = {right_, right_alias_},
         .output = intermediate_,
         .alpha = 1.0},
        operation_capability(
            std::string(stable ? kMultiplyPrimitive
                               : kBackendMultiplyPrimitive),
            plan.path, capability));
    add_ = make_acdnn_pointwise_reference(
        {.mode = FLAGDNN_POINTWISE_ADD,
         .inputs = {left_, intermediate_},
         .output = output_,
         .alpha = 1.0},
        operation_capability(
            std::string(stable ? kAddPrimitive : kBackendAddPrimitive),
            plan.path, capability));

    intermediate_size_ = storage_bytes(output_);
    operation_workspace_offset_ = align_workspace(intermediate_size_);
    operation_workspace_size_ =
        std::max(multiply_->workspace_size(), add_->workspace_size());
    if (operation_workspace_offset_ >
        std::numeric_limits<std::size_t>::max() - operation_workspace_size_) {
      throw std::overflow_error("THead acDNN AddSquare workspace is too large");
    }
    workspace_size_ = operation_workspace_offset_ + operation_workspace_size_;
  }

  [[nodiscard]] std::size_t workspace_size() const noexcept override {
    return workspace_size_;
  }

  void execute(std::span<const flagdnnBinding_t> bindings, void *workspace,
               std::size_t workspace_size,
               flagdnnStream_t stream) override {
    if (workspace == nullptr || workspace_size < workspace_size_) {
      throw std::invalid_argument(
          "THead acDNN AddSquare reference workspace is too small");
    }
    std::map<std::int64_t, void *> pointers;
    for (const flagdnnBinding_t &binding : bindings) {
      if (binding.device_pointer == nullptr ||
          !pointers.emplace(binding.uid, binding.device_pointer).second) {
        throw std::invalid_argument(
            "THead acDNN AddSquare bindings are null or duplicated");
      }
    }
    if (pointers.size() != 3 || !pointers.contains(left_.uid) ||
        !pointers.contains(right_.uid) || !pointers.contains(output_.uid)) {
      throw std::invalid_argument(
          "THead acDNN AddSquare requires left/right/output bindings");
    }
    const std::array<flagdnnBinding_t, 3> multiply_bindings = {{
        {right_.uid, pointers.at(right_.uid)},
        {right_alias_.uid, pointers.at(right_.uid)},
        {intermediate_.uid, workspace},
    }};
    const std::array<flagdnnBinding_t, 3> add_bindings = {{
        {left_.uid, pointers.at(left_.uid)},
        {intermediate_.uid, workspace},
        {output_.uid, pointers.at(output_.uid)},
    }};
    void *operation_workspace =
        static_cast<std::byte *>(workspace) + operation_workspace_offset_;
    multiply_->execute(multiply_bindings,
                       multiply_->workspace_size() == 0
                           ? nullptr
                           : operation_workspace,
                       multiply_->workspace_size(), stream);
    add_->execute(add_bindings,
                  add_->workspace_size() == 0 ? nullptr
                                              : operation_workspace,
                  add_->workspace_size(), stream);
  }

 private:
  flagdnn::testing::TestTensor left_;
  flagdnn::testing::TestTensor right_;
  flagdnn::testing::TestTensor output_;
  flagdnn::testing::TestTensor intermediate_;
  flagdnn::testing::TestTensor right_alias_;
  std::size_t intermediate_size_ = 0;
  std::size_t operation_workspace_offset_ = 0;
  std::size_t operation_workspace_size_ = 0;
  std::size_t workspace_size_ = 0;
  std::unique_ptr<flagdnn::testing::TestExecutable> multiply_;
  std::unique_ptr<flagdnn::testing::TestExecutable> add_;
};

}  // namespace

std::unique_ptr<flagdnn::testing::TestExecutable>
make_acdnn_add_square_reference(
    const flagdnn::testing::TestTensor &left,
    const flagdnn::testing::TestTensor &right,
    const flagdnn::testing::TestTensor &output,
    const CapabilityRecord &capability) {
  return std::make_unique<AcdnnAddSquare>(left, right, output, capability);
}

}  // namespace flagdnn::validation::thead
