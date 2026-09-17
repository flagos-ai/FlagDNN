/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "tensor_reference.hpp"

#include "tensor_io.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace flagdnn::validation::hygon {
namespace {

hipdnnDataType_t hipdnn_data_type(flagdnnDataType_t data_type) {
  switch (data_type) {
    case FLAGDNN_DATA_INT32:
      throw std::invalid_argument(
          "INT32 is not supported by this validation adapter");

    case FLAGDNN_DATA_FLOAT32:
      return HIPDNN_DATA_FLOAT;
    case FLAGDNN_DATA_FLOAT16:
      return HIPDNN_DATA_HALF;
    case FLAGDNN_DATA_BFLOAT16:
      return HIPDNN_DATA_BFLOAT16;
    case FLAGDNN_DATA_BOOLEAN:
    case FLAGDNN_DATA_FP8_E8M0:
    case FLAGDNN_DATA_FP8_E4M3:
    case FLAGDNN_DATA_FP8_E5M2:
      break;
  }
  throw std::invalid_argument("data type has no tensor reference mapping");
}

bool is_reduction(HipdnnTensorKind kind) noexcept {
  return kind == HipdnnTensorKind::kReductionAdd ||
         kind == HipdnnTensorKind::kReductionAverage ||
         kind == HipdnnTensorKind::kReductionMultiply;
}

hipdnnReduceTensorOp_t reduction_mode(HipdnnTensorKind kind) {
  switch (kind) {
  case HipdnnTensorKind::kReductionAdd:
    return HIPDNN_REDUCE_TENSOR_ADD;
  case HipdnnTensorKind::kReductionAverage:
    return HIPDNN_REDUCE_TENSOR_AVG;
  case HipdnnTensorKind::kReductionMultiply:
    return HIPDNN_REDUCE_TENSOR_MUL;
  case HipdnnTensorKind::kSlice:
  case HipdnnTensorKind::kUnavailable:
    break;
  }
  throw std::invalid_argument("tensor operation is not a reduction");
}

bool checked_multiply(std::int64_t left, std::int64_t right,
                      std::int64_t &result) noexcept {
  if (left <= 0 || right <= 0 ||
      left > std::numeric_limits<std::int64_t>::max() / right) {
    return false;
  }
  result = left * right;
  return true;
}

std::vector<std::int64_t>
dense_strides(std::span<const std::int64_t> dimensions) {
  std::vector<std::int64_t> result(dimensions.size());
  std::int64_t stride = 1;
  for (std::size_t axis = dimensions.size(); axis != 0; --axis) {
    const std::int64_t dimension = dimensions[axis - 1];
    if (dimension <= 0) {
      throw std::invalid_argument("dense tensor dimension must be positive");
    }
    result[axis - 1] = stride;
    if (!checked_multiply(stride, dimension, stride)) {
      throw std::overflow_error("dense tensor strides overflow");
    }
  }
  return result;
}

bool metadata_fits_int(const ReferenceTensor &tensor,
                       bool allow_scalar) noexcept {
  if (tensor.dimensions.size() != tensor.strides.size() ||
      (!allow_scalar && tensor.dimensions.empty()) ||
      tensor.dimensions.size() > 8) {
    return false;
  }
  for (std::size_t axis = 0; axis < tensor.dimensions.size(); ++axis) {
    if (tensor.dimensions[axis] <= 0 || tensor.strides[axis] <= 0 ||
        tensor.dimensions[axis] > std::numeric_limits<int>::max() ||
        tensor.strides[axis] > std::numeric_limits<int>::max()) {
      return false;
    }
  }
  return true;
}

bool metadata_contract_valid(const ReferenceTensor &tensor,
                             bool allow_scalar) noexcept {
  if (tensor.dimensions.size() != tensor.strides.size() ||
      (!allow_scalar && tensor.dimensions.empty())) {
    return false;
  }
  for (std::size_t axis = 0; axis < tensor.dimensions.size(); ++axis) {
    if (tensor.dimensions[axis] <= 0 || tensor.strides[axis] <= 0) {
      return false;
    }
  }
  return true;
}

bool is_non_overlapping(const ReferenceTensor &tensor) noexcept {
  if (tensor.dimensions.empty()) {
    return true;
  }
  std::vector<std::size_t> axes;
  axes.reserve(tensor.dimensions.size());
  for (std::size_t axis = 0; axis < tensor.dimensions.size(); ++axis) {
    if (tensor.dimensions[axis] > 1) {
      axes.push_back(axis);
    }
  }
  std::sort(axes.begin(), axes.end(), [&](std::size_t left, std::size_t right) {
    return tensor.strides[left] < tensor.strides[right];
  });
  std::uint64_t span = 1;
  for (std::size_t axis : axes) {
    const std::uint64_t stride =
        static_cast<std::uint64_t>(tensor.strides[axis]);
    const std::uint64_t dimension =
        static_cast<std::uint64_t>(tensor.dimensions[axis]);
    if (stride < span ||
        dimension - 1 >
            (std::numeric_limits<std::uint64_t>::max() - span) / stride) {
      return false;
    }
    span += (dimension - 1) * stride;
  }
  return true;
}

std::vector<int> checked_ints(std::span<const std::int64_t> values,
                              std::string_view role) {
  std::vector<int> result;
  result.reserve(values.size());
  for (std::int64_t value : values) {
    if (value <= 0 || value > std::numeric_limits<int>::max()) {
      throw std::invalid_argument(std::string(role) +
                                  " does not fit hipDNN int metadata");
    }
    result.push_back(static_cast<int>(value));
  }
  return result;
}

void set_tensor_descriptor(hipdnnTensorDescriptor_t descriptor,
                           const ReferenceTensor &tensor) {
  const std::vector<int> dimensions =
      checked_ints(tensor.dimensions, "tensor dimensions");
  const std::vector<int> strides =
      checked_ints(tensor.strides, "tensor strides");
  check_hipdnn_status(hipdnnSetTensorNdDescriptor(
                          descriptor, hipdnn_data_type(tensor.data_type),
                          static_cast<int>(dimensions.size()),
                          dimensions.data(), strides.data()),
                      "hipdnnSetTensorNdDescriptor");
}

ReferenceTensor minimum_rank_two(ReferenceTensor tensor) {
  if (tensor.dimensions.size() == 1) {
    tensor.dimensions.insert(tensor.dimensions.begin(), 1);
    tensor.strides.insert(tensor.strides.begin(), 1);
  }
  return tensor;
}

std::int32_t normalized_axis(const HipdnnTensorOperation &operation,
                             std::size_t rank) {
  std::int32_t axis = operation.reduction_axis;
  if (axis < 0) {
    axis += static_cast<std::int32_t>(rank);
  }
  return axis;
}

ReferenceTensor
reduction_output_descriptor(const HipdnnTensorOperation &operation,
                            const ReferenceTensor &input,
                            const ReferenceTensor &output) {
  ReferenceTensor result = output;
  const std::size_t axis = static_cast<std::size_t>(
      normalized_axis(operation, input.dimensions.size()));
  if (!operation.keep_dimensions) {
    std::int64_t inserted_stride = 1;
    if (axis < output.dimensions.size()) {
      std::int64_t candidate = 1;
      if (checked_multiply(output.strides[axis], output.dimensions[axis],
                           candidate) &&
          candidate <= std::numeric_limits<int>::max()) {
        inserted_stride = candidate;
      }
    }
    result.dimensions.insert(
        result.dimensions.begin() + static_cast<std::ptrdiff_t>(axis), 1);
    result.strides.insert(result.strides.begin() +
                              static_cast<std::ptrdiff_t>(axis),
                          inserted_stride);
  }
  return minimum_rank_two(std::move(result));
}

ReferenceTensor reduction_input_descriptor(const ReferenceTensor &input) {
  return minimum_rank_two(input);
}

HipdnnCapability
structural_capability(const HipdnnTensorOperation &operation,
                      std::span<const ReferenceTensor> tensors) {
  if (operation.kind == HipdnnTensorKind::kUnavailable) {
    return HipdnnCapability::vendor_unsupported(
        operation.unavailable_reason.empty()
            ? "no exact validated hipDNN tensor primitive"
            : operation.unavailable_reason);
  }
  if (tensors.size() != 2) {
    return HipdnnCapability::invalid_adapter_contract(
        "hipDNN tensor primitive requires one input and one output");
  }
  const ReferenceTensor &input = tensors[0];
  const ReferenceTensor &output = tensors[1];
  if (input.uid == output.uid) {
    return HipdnnCapability::invalid_adapter_contract(
        "hipDNN tensor input/output UIDs must differ");
  }
  if (input.data_type != output.data_type) {
    return HipdnnCapability::invalid_adapter_contract(
        "hipDNN tensor input/output data types must match");
  }
  const std::size_t element_size = tensor_io::data_type_size(input.data_type);
  if (input.binding_byte_offset % element_size != 0 ||
      output.binding_byte_offset % element_size != 0) {
    return HipdnnCapability::invalid_adapter_contract(
        "tensor binding byte offset is not element aligned");
  }
  if (!metadata_contract_valid(input, false) ||
      !metadata_contract_valid(output, true)) {
    return HipdnnCapability::invalid_adapter_contract(
        "tensor dimensions/strides metadata is invalid");
  }

  if (operation.kind == HipdnnTensorKind::kSlice) {
    if (input.data_type != FLAGDNN_DATA_FLOAT32) {
      return HipdnnCapability::vendor_unsupported(
          "slice is validated only for exact FP32 hipdnnTransformTensor");
    }
    const std::size_t rank = input.dimensions.size();
    if (output.dimensions.size() != rank || operation.slices.size() != rank ||
        operation.slice_strides.size() != rank) {
      return HipdnnCapability::invalid_adapter_contract(
          "slice tensor/attribute ranks do not match");
    }
    if (rank < 2 || rank > 3 || !metadata_fits_int(input, false) ||
        !metadata_fits_int(output, false)) {
      return HipdnnCapability::vendor_unsupported(
          "slice requires representable rank-2/rank-3 tensor metadata");
    }
    if (!is_non_overlapping(input)) {
      return HipdnnCapability::vendor_unsupported(
          "slice input strides overlap and are not representable");
    }
    std::vector<std::int64_t> expected_strides;
    try {
      expected_strides = dense_strides(output.dimensions);
    } catch (const std::exception &) {
      return HipdnnCapability::vendor_unsupported(
          "slice dense output strides overflow");
    }
    if (output.strides != expected_strides) {
      return HipdnnCapability::vendor_unsupported(
          "hipdnnTransformTensor slice requires a dense output descriptor");
    }

    ReferenceTensor effective = input;
    effective.dimensions = output.dimensions;
    std::uint64_t start_elements = 0;
    for (std::size_t axis = 0; axis < rank; ++axis) {
      const auto [start, limit] = operation.slices[axis];
      const std::int64_t step = operation.slice_strides[axis];
      if (start < 0 || limit <= start || limit > input.dimensions[axis] ||
          step <= 0 ||
          output.dimensions[axis] != (limit - start + step - 1) / step) {
        return HipdnnCapability::invalid_adapter_contract(
            "slice range/shape does not match positive-step semantics");
      }
      std::int64_t effective_stride = 1;
      if (!checked_multiply(input.strides[axis], step, effective_stride) ||
          effective_stride > std::numeric_limits<int>::max()) {
        return HipdnnCapability::vendor_unsupported(
            "slice effective stride does not fit hipDNN metadata");
      }
      effective.strides[axis] = effective_stride;
      const std::uint64_t stride =
          static_cast<std::uint64_t>(input.strides[axis]);
      const std::uint64_t coordinate = static_cast<std::uint64_t>(start);
      if (coordinate != 0 &&
          stride >
              (std::numeric_limits<std::uint64_t>::max() - start_elements) /
                  coordinate) {
        return HipdnnCapability::vendor_unsupported(
            "slice input pointer offset overflows");
      }
      start_elements += coordinate * stride;
    }
    if (!is_non_overlapping(effective) ||
        start_elements >
            std::numeric_limits<std::size_t>::max() / element_size) {
      return HipdnnCapability::vendor_unsupported(
          "slice positive-step input view is not representable exactly");
    }
    // DTK hipDNN merges adjacent dimensions using an integer stride ratio.
    // For a [3, 3] view with strides [7, 2], it reads the second row at
    // offset 6 instead of 7. Equivalent padded descriptors have the same
    // failure. Keep these views outside the exact-reference capability.
    for (std::size_t axis = 1; axis < rank; ++axis) {
      const std::int64_t outer_stride = effective.strides[axis - 1];
      const std::int64_t inner_stride = effective.strides[axis];
      if (effective.dimensions[axis - 1] > 1 &&
          outer_stride / inner_stride == effective.dimensions[axis] &&
          outer_stride % inner_stride != 0) {
        return HipdnnCapability::vendor_unsupported(
            "hipdnnTransformTensor slice cannot preserve non-integral "
            "packed stride ratios on the validated DTK stack");
      }
    }
    return {};
  }

  if (!is_reduction(operation.kind)) {
    return HipdnnCapability::invalid_adapter_contract(
        "unknown hipDNN tensor operation kind");
  }
  if (input.data_type == FLAGDNN_DATA_BFLOAT16) {
    return HipdnnCapability::vendor_unsupported(
        "current Hygon hipDNN ReduceTensor BF16 has no validated executable "
        "primitive");
  }
  if (input.data_type != FLAGDNN_DATA_FLOAT32 &&
      input.data_type != FLAGDNN_DATA_FLOAT16) {
    return HipdnnCapability::vendor_unsupported(
        "reduction is validated only for FP32/FP16");
  }
  if (!metadata_fits_int(input, false) || !metadata_fits_int(output, true) ||
      !is_non_overlapping(input) || !is_non_overlapping(output)) {
    return HipdnnCapability::vendor_unsupported(
        "reduction tensor metadata is overlapping or not representable");
  }
  const std::size_t rank = input.dimensions.size();
  const std::int32_t axis = normalized_axis(operation, rank);
  if (axis < 0 || axis >= static_cast<std::int32_t>(rank)) {
    return HipdnnCapability::invalid_adapter_contract(
        "reduction axis is out of range");
  }
  std::vector<std::int64_t> expected = input.dimensions;
  if (operation.keep_dimensions) {
    expected[static_cast<std::size_t>(axis)] = 1;
  } else {
    expected.erase(expected.begin() + axis);
  }
  if (output.dimensions != expected) {
    return HipdnnCapability::invalid_adapter_contract(
        "reduction output shape does not match axis semantics");
  }
  const ReferenceTensor input_descriptor = reduction_input_descriptor(input);
  const ReferenceTensor output_descriptor =
      reduction_output_descriptor(operation, input, output);
  if (!metadata_fits_int(input_descriptor, false) ||
      !metadata_fits_int(output_descriptor, false) ||
      input_descriptor.dimensions.size() !=
          output_descriptor.dimensions.size()) {
    return HipdnnCapability::vendor_unsupported(
        "normalized reduction descriptors do not fit hipDNN");
  }
  return {};
}

std::size_t slice_source_byte_offset(const HipdnnTensorOperation &operation,
                                     const ReferenceTensor &input) {
  std::uint64_t element_offset = 0;
  for (std::size_t axis = 0; axis < operation.slices.size(); ++axis) {
    element_offset += static_cast<std::uint64_t>(operation.slices[axis].first) *
                      static_cast<std::uint64_t>(input.strides[axis]);
  }
  return static_cast<std::size_t>(element_offset) *
         tensor_io::data_type_size(input.data_type);
}

ReferenceTensor slice_input_descriptor(const HipdnnTensorOperation &operation,
                                       const ReferenceTensor &input,
                                       const ReferenceTensor &output) {
  ReferenceTensor result = input;
  result.dimensions = output.dimensions;
  for (std::size_t axis = 0; axis < result.strides.size(); ++axis) {
    result.strides[axis] *= operation.slice_strides[axis];
  }
  return result;
}

void *binding_pointer(std::span<const flagdnnBinding_t> bindings,
                      const ReferenceTensor &tensor,
                      std::size_t additional_byte_offset = 0) {
  const auto found = std::find_if(bindings.begin(), bindings.end(),
                                  [&](const flagdnnBinding_t &binding) {
                                    return binding.uid == tensor.uid;
                                  });
  if (found == bindings.end() || found->device_pointer == nullptr) {
    throw std::invalid_argument("hipDNN tensor binding is missing");
  }
  if (additional_byte_offset >
      std::numeric_limits<std::size_t>::max() - tensor.binding_byte_offset) {
    throw std::overflow_error("hipDNN tensor binding offset overflows");
  }
  return static_cast<void *>(
      static_cast<std::uint8_t *>(found->device_pointer) +
      tensor.binding_byte_offset + additional_byte_offset);
}

} // namespace

HipdnnTensorOperation make_hipdnn_slice_operation(
    std::vector<std::pair<std::int64_t, std::int64_t>> slices,
    std::vector<std::int64_t> strides) {
  HipdnnTensorOperation result;
  result.kind = HipdnnTensorKind::kSlice;
  result.slices = std::move(slices);
  result.slice_strides = std::move(strides);
  return result;
}

HipdnnTensorOperation
make_hipdnn_reduction_operation(flagdnnReductionMode_t mode, std::int32_t axis,
                                bool keep_dimensions) {
  HipdnnTensorOperation result;
  switch (mode) {
  case FLAGDNN_REDUCTION_ADD:
    result.kind = HipdnnTensorKind::kReductionAdd;
    break;
  case FLAGDNN_REDUCTION_AVG:
    result.kind = HipdnnTensorKind::kReductionAverage;
    break;
  case FLAGDNN_REDUCTION_MUL:
    result.kind = HipdnnTensorKind::kReductionMultiply;
    break;
  default:
    return make_hipdnn_tensor_unavailable(
        "reduction mode has no exact hipDNN primitive");
  }
  result.reduction_axis = axis;
  result.keep_dimensions = keep_dimensions;
  return result;
}

HipdnnTensorOperation make_hipdnn_tensor_unavailable(std::string reason) {
  HipdnnTensorOperation result;
  result.kind = HipdnnTensorKind::kUnavailable;
  result.unavailable_reason = std::move(reason);
  return result;
}

ReferenceTensor dense_reference_tensor(const ReferenceTensor &tensor) {
  ReferenceTensor result = tensor;
  result.strides = dense_strides(result.dimensions);
  return result;
}

HipdnnCapability
hipdnn_tensor_capability(const HipdnnTensorOperation &operation,
                         std::span<const ReferenceTensor> tensors) {
  const HipdnnCapability structural = structural_capability(operation, tensors);
  if (!structural.supported) {
    return structural;
  }
  return {};
}

std::string_view hipdnn_tensor_kind_name(HipdnnTensorKind kind) noexcept {
  switch (kind) {
  case HipdnnTensorKind::kSlice:
    return "slice";
  case HipdnnTensorKind::kReductionAdd:
    return "reduction_add";
  case HipdnnTensorKind::kReductionAverage:
    return "reduction_avg";
  case HipdnnTensorKind::kReductionMultiply:
    return "reduction_mul";
  case HipdnnTensorKind::kUnavailable:
    return "unavailable";
  }
  return "unknown";
}

class HipdnnTensorPlan::Impl final {
public:
  Impl(HipdnnTensorOperation operation, std::vector<ReferenceTensor> tensors)
      : operation_(std::move(operation)), tensors_(std::move(tensors)) {
    const HipdnnCapability capability =
        structural_capability(operation_, tensors_);
    if (!capability.supported) {
      throw std::invalid_argument("invalid hipDNN tensor plan: " +
                                  capability.reason);
    }

    check_hipdnn_status(hipdnnCreate(&handle_), "hipdnnCreate");
    try {
      check_hipdnn_status(hipdnnCreateTensorDescriptor(&input_descriptor_),
                          "hipdnnCreateTensorDescriptor(input)");
      check_hipdnn_status(hipdnnCreateTensorDescriptor(&output_descriptor_),
                          "hipdnnCreateTensorDescriptor(output)");
      if (operation_.kind == HipdnnTensorKind::kSlice) {
        set_tensor_descriptor(
            input_descriptor_,
            slice_input_descriptor(operation_, tensors_[0], tensors_[1]));
        set_tensor_descriptor(output_descriptor_, tensors_[1]);
        source_byte_offset_ = slice_source_byte_offset(operation_, tensors_[0]);
      } else {
        const ReferenceTensor input = reduction_input_descriptor(tensors_[0]);
        const ReferenceTensor output =
            reduction_output_descriptor(operation_, tensors_[0], tensors_[1]);
        set_tensor_descriptor(input_descriptor_, input);
        set_tensor_descriptor(output_descriptor_, output);
        check_hipdnn_status(
            hipdnnCreateReduceTensorDescriptor(&reduction_descriptor_),
            "hipdnnCreateReduceTensorDescriptor");
        check_hipdnn_status(
            hipdnnSetReduceTensorDescriptor(
                reduction_descriptor_, reduction_mode(operation_.kind),
                HIPDNN_DATA_FLOAT, HIPDNN_PROPAGATE_NAN,
                HIPDNN_REDUCE_TENSOR_NO_INDICES, HIPDNN_32BIT_INDICES),
            "hipdnnSetReduceTensorDescriptor");
        check_hipdnn_status(hipdnnGetReductionWorkspaceSize(
                                handle_, reduction_descriptor_,
                                input_descriptor_, output_descriptor_,
                                &workspace_size_),
                            "hipdnnGetReductionWorkspaceSize");
      }
    } catch (...) {
      cleanup();
      throw;
    }
  }

  ~Impl() { cleanup(); }

  [[nodiscard]] std::size_t workspace_size() const noexcept {
    return workspace_size_;
  }

  void execute(std::span<const flagdnnBinding_t> bindings, void *workspace,
               std::size_t workspace_size, flagdnnStream_t stream) {
    check_hipdnn_status(
        hipdnnSetStream(handle_, reinterpret_cast<hipdnnStream_t>(stream)),
        "hipdnnSetStream");
    constexpr float one = 1.0F;
    constexpr float zero = 0.0F;
    void *input = binding_pointer(
        bindings, tensors_[0],
        operation_.kind == HipdnnTensorKind::kSlice ? source_byte_offset_ : 0);
    void *output = binding_pointer(bindings, tensors_[1]);
    if (operation_.kind == HipdnnTensorKind::kSlice) {
      check_hipdnn_status(hipdnnTransformTensor(handle_, &one,
                                                input_descriptor_, input, &zero,
                                                output_descriptor_, output),
                          "hipdnnTransformTensor(slice)");
      return;
    }
    if (workspace_size < workspace_size_ ||
        (workspace_size_ != 0 && workspace == nullptr)) {
      throw std::invalid_argument("hipDNN reduction workspace is too small");
    }
    check_hipdnn_status(
        hipdnnReduceTensor(handle_, reduction_descriptor_, nullptr, 0,
                           workspace, workspace_size_, &one, input_descriptor_,
                           input, &zero, output_descriptor_, output),
        "hipdnnReduceTensor");
  }

private:
  void cleanup() noexcept {
    if (reduction_descriptor_ != nullptr) {
      (void)hipdnnDestroyReduceTensorDescriptor(reduction_descriptor_);
      reduction_descriptor_ = nullptr;
    }
    if (output_descriptor_ != nullptr) {
      (void)hipdnnDestroyTensorDescriptor(output_descriptor_);
      output_descriptor_ = nullptr;
    }
    if (input_descriptor_ != nullptr) {
      (void)hipdnnDestroyTensorDescriptor(input_descriptor_);
      input_descriptor_ = nullptr;
    }
    if (handle_ != nullptr) {
      (void)hipdnnDestroy(handle_);
      handle_ = nullptr;
    }
  }

  HipdnnTensorOperation operation_;
  std::vector<ReferenceTensor> tensors_;
  hipdnnHandle_t handle_ = nullptr;
  hipdnnTensorDescriptor_t input_descriptor_ = nullptr;
  hipdnnTensorDescriptor_t output_descriptor_ = nullptr;
  hipdnnReduceTensorDescriptor_t reduction_descriptor_ = nullptr;
  std::size_t source_byte_offset_ = 0;
  std::size_t workspace_size_ = 0;
};

HipdnnTensorPlan::HipdnnTensorPlan(HipdnnTensorOperation operation,
                                   std::vector<ReferenceTensor> tensors)
    : impl_(std::make_unique<Impl>(std::move(operation), std::move(tensors))) {}

HipdnnTensorPlan::~HipdnnTensorPlan() = default;
HipdnnTensorPlan::HipdnnTensorPlan(HipdnnTensorPlan &&) noexcept = default;
HipdnnTensorPlan &
HipdnnTensorPlan::operator=(HipdnnTensorPlan &&) noexcept = default;

std::size_t HipdnnTensorPlan::workspace_size() const noexcept {
  return impl_->workspace_size();
}

void HipdnnTensorPlan::execute(std::span<const flagdnnBinding_t> bindings,
                               void *workspace, std::size_t workspace_size,
                               flagdnnStream_t stream) {
  impl_->execute(bindings, workspace, workspace_size, stream);
}

} // namespace flagdnn::validation::hygon
