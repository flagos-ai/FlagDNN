/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "backends/mthreads/validation/mudnn_reduction.hpp"

#include "backends/mthreads/validation/musa_driver.hpp"
#include "backends/mthreads/validation/tensor_io.hpp"

#include <mudnn.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace flagdnn::validation::mthreads {
namespace {

musa::dnn::Tensor::Type mudnn_data_type(flagdnnDataType_t data_type) {
  switch (data_type) {
    case FLAGDNN_DATA_FLOAT32:
      return musa::dnn::Tensor::Type::FLOAT;
    case FLAGDNN_DATA_FLOAT16:
      return musa::dnn::Tensor::Type::HALF;
    case FLAGDNN_DATA_BFLOAT16:
      return musa::dnn::Tensor::Type::BFLOAT16;
    case FLAGDNN_DATA_BOOLEAN:
    case FLAGDNN_DATA_FP8_E4M3:
    case FLAGDNN_DATA_FP8_E5M2:
      break;
  }
  throw std::invalid_argument(
      "muDNN Reduction tensor type is unsupported");
}

musa::dnn::Reduce::Mode mudnn_reduction_mode(
    flagdnnReductionMode_t mode) {
  switch (mode) {
    case FLAGDNN_REDUCTION_ADD:
      return musa::dnn::Reduce::Mode::ADD;
    case FLAGDNN_REDUCTION_AVG:
      return musa::dnn::Reduce::Mode::MEAN;
    case FLAGDNN_REDUCTION_MUL:
      // muDNN distinguishes the elementwise-style MUL mode from the
      // product-reduction mode. torch_musa's prod integration uses PROD.
      return musa::dnn::Reduce::Mode::PROD;
  }
  throw std::invalid_argument("muDNN Reduction mode is unsupported");
}

void validate_tensor(const TensorDescriptor& tensor,
                     std::string_view name,
                     bool allow_scalar) {
  if (tensor.uid <= 0 || tensor.dimensions.size() > 8 ||
      tensor.dimensions.size() != tensor.strides.size() ||
      (!allow_scalar && tensor.dimensions.empty())) {
    throw std::invalid_argument(
        std::string(name) + " muDNN Reduction descriptor is invalid");
  }
  for (std::size_t axis = 0; axis < tensor.dimensions.size(); ++axis) {
    if (tensor.dimensions[axis] <= 0 || tensor.strides[axis] <= 0) {
      throw std::invalid_argument(
          std::string(name) +
          " muDNN Reduction dimensions/strides must be positive");
    }
  }
  static_cast<void>(mudnn_data_type(tensor.data_type));
  static_cast<void>(tensor_io::element_count(tensor));
  static_cast<void>(tensor_io::storage_element_count(tensor));
}

std::int32_t normalized_axis(const MudnnReductionDescriptor& descriptor) {
  std::int32_t axis = descriptor.axis;
  const auto rank =
      static_cast<std::int32_t>(descriptor.input.dimensions.size());
  if (axis < 0) {
    axis += rank;
  }
  if (axis < 0 || axis >= rank) {
    throw std::invalid_argument("muDNN Reduction axis is out of range");
  }
  return axis;
}

void validate_descriptor(const MudnnReductionDescriptor& descriptor) {
  validate_tensor(descriptor.input, "input", false);
  validate_tensor(descriptor.output, "output", true);
  if (descriptor.input.uid == descriptor.output.uid ||
      descriptor.input.data_type != descriptor.output.data_type) {
    throw std::invalid_argument(
        "muDNN Reduction requires distinct same-type tensors");
  }
  static_cast<void>(mudnn_reduction_mode(descriptor.mode));
  const std::int32_t axis = normalized_axis(descriptor);
  std::vector<std::int64_t> expected = descriptor.input.dimensions;
  if (descriptor.keep_dimensions) {
    expected[static_cast<std::size_t>(axis)] = 1;
  } else {
    expected.erase(expected.begin() + axis);
  }
  if (descriptor.output.dimensions != expected) {
    throw std::invalid_argument(
        "muDNN Reduction output shape is invalid");
  }
}

TensorDescriptor full_rank_output(
    const MudnnReductionDescriptor& descriptor,
    std::int32_t axis) {
  TensorDescriptor result = descriptor.output;
  if (!descriptor.keep_dimensions) {
    const auto position = static_cast<std::size_t>(axis);
    result.dimensions.insert(result.dimensions.begin() + axis, 1);
    result.strides.insert(
        result.strides.begin() + axis,
        descriptor.input.strides[position]);
  }
  return result;
}

bool is_row_major_contiguous(const TensorDescriptor& tensor) {
  std::int64_t running = 1;
  for (std::size_t trailing = 0;
       trailing < tensor.dimensions.size(); ++trailing) {
    const std::size_t axis = tensor.dimensions.size() - 1 - trailing;
    if (tensor.dimensions[axis] != 1 && tensor.strides[axis] != running) {
      return false;
    }
    if (tensor.dimensions[axis] >
        std::numeric_limits<std::int64_t>::max() / running) {
      throw std::overflow_error(
          "muDNN Reduction dense extent overflows int64");
    }
    running *= tensor.dimensions[axis];
  }
  return true;
}

std::vector<std::int64_t> contiguous_strides(
    const std::vector<std::int64_t>& dimensions) {
  std::vector<std::int64_t> result(dimensions.size());
  std::int64_t running = 1;
  for (std::size_t axis = dimensions.size(); axis != 0; --axis) {
    result[axis - 1] = running;
    if (dimensions[axis - 1] >
        std::numeric_limits<std::int64_t>::max() / running) {
      throw std::overflow_error(
          "muDNN Reduction dense stride overflows int64");
    }
    running *= dimensions[axis - 1];
  }
  return result;
}

std::size_t tensor_bytes(const TensorDescriptor& tensor) {
  const std::size_t elements = tensor_io::element_count(tensor);
  const std::size_t width = tensor_io::data_type_size(tensor.data_type);
  if (elements > std::numeric_limits<std::size_t>::max() / width) {
    throw std::overflow_error("muDNN Reduction tensor bytes overflow");
  }
  return elements * width;
}

std::size_t align_up(std::size_t value, std::size_t alignment) {
  const std::size_t remainder = value % alignment;
  if (remainder == 0) {
    return value;
  }
  const std::size_t increment = alignment - remainder;
  if (value > std::numeric_limits<std::size_t>::max() - increment) {
    throw std::overflow_error("muDNN Reduction workspace alignment overflows");
  }
  return value + increment;
}

std::size_t checked_add(std::size_t left,
                        std::size_t right,
                        std::string_view description) {
  if (left > std::numeric_limits<std::size_t>::max() - right) {
    throw std::overflow_error(std::string(description) + " overflows");
  }
  return left + right;
}

void configure_tensor_descriptor(musa::dnn::Tensor& tensor,
                                 const TensorDescriptor& descriptor) {
  check_mudnn(
      tensor.SetType(mudnn_data_type(descriptor.data_type)),
      "muDNN Tensor::SetType(Reduction)");
  check_mudnn(
      tensor.SetNdInfo(
          static_cast<std::int64_t>(descriptor.dimensions.size()),
          descriptor.dimensions.data(),
          descriptor.strides.data()),
      "muDNN Tensor::SetNdInfo(Reduction)");
}

void configure_tensor(musa::dnn::Tensor& tensor,
                      const TensorDescriptor& descriptor,
                      void* pointer) {
  check_mudnn(
      tensor.SetAddr(pointer), "muDNN Tensor::SetAddr(Reduction)");
  configure_tensor_descriptor(tensor, descriptor);
}

void* binding_pointer(
    const TensorDescriptor& tensor,
    const std::unordered_map<std::int64_t, void*>& bindings) {
  const auto found = bindings.find(tensor.uid);
  if (found == bindings.end()) {
    throw std::invalid_argument("muDNN Reduction binding UID is missing");
  }
  return found->second;
}

}  // namespace

struct MudnnReductionOperation::Impl {
  explicit Impl(MudnnReductionDescriptor value)
      : descriptor(std::move(value)),
        handle(0) {
    validate_descriptor(descriptor);
    axis = normalized_axis(descriptor);
    output_view = full_rank_output(descriptor, axis);
    bridge_layout = !is_row_major_contiguous(descriptor.input) ||
                    !is_row_major_contiguous(descriptor.output);
    reduction_input = descriptor.input;
    reduction_output = output_view;
    if (bridge_layout) {
      reduction_input.strides =
          contiguous_strides(reduction_input.dimensions);
      reduction_input.binding_byte_offset = 0;
      reduction_output.strides =
          contiguous_strides(reduction_output.dimensions);
      reduction_output.binding_byte_offset = 0;
    }
    check_mudnn(
        reduction.SetMode(mudnn_reduction_mode(descriptor.mode)),
        "muDNN Reduce::SetMode");
    const int reduction_axis = axis;
    check_mudnn(
        reduction.SetDim(1, &reduction_axis), "muDNN Reduce::SetDim");

    musa::dnn::Tensor input;
    musa::dnn::Tensor output;
    configure_tensor_descriptor(input, reduction_input);
    configure_tensor_descriptor(output, reduction_output);
    check_mudnn(
        reduction.GetWorkspaceSize(
            handle, scratch_bytes, output, input),
        "muDNN Reduce::GetWorkspaceSize");
    if (!bridge_layout) {
      workspace_bytes = scratch_bytes;
      return;
    }
    dense_input_offset = 0;
    dense_output_offset =
        align_up(tensor_bytes(reduction_input), kWorkspaceAlignment);
    scratch_offset = align_up(
        checked_add(dense_output_offset,
                    tensor_bytes(reduction_output),
                    "muDNN Reduction dense output workspace"),
        kWorkspaceAlignment);
    workspace_bytes = checked_add(
        scratch_offset,
        scratch_bytes,
        "muDNN Reduction total workspace");
  }

  static constexpr std::size_t kWorkspaceAlignment = 256;

  MudnnReductionDescriptor descriptor;
  std::int32_t axis = 0;
  TensorDescriptor output_view;
  TensorDescriptor reduction_input;
  TensorDescriptor reduction_output;
  musa::dnn::Handle handle;
  musa::dnn::Reduce reduction;
  musa::dnn::Permute permute;
  bool bridge_layout = false;
  std::size_t dense_input_offset = 0;
  std::size_t dense_output_offset = 0;
  std::size_t scratch_offset = 0;
  std::size_t scratch_bytes = 0;
  std::size_t workspace_bytes = 0;
};

MudnnReductionOperation::MudnnReductionOperation(
    MudnnReductionDescriptor descriptor)
    : implementation_(std::make_unique<Impl>(std::move(descriptor))) {}

MudnnReductionOperation::~MudnnReductionOperation() = default;

std::size_t MudnnReductionOperation::workspace_size() const noexcept {
  return implementation_->workspace_bytes;
}

void MudnnReductionOperation::execute(
    std::span<const flagdnnBinding_t> raw_bindings,
    void* workspace,
    std::size_t workspace_size,
    flagdnnStream_t stream) {
  Impl& state = *implementation_;
  if (stream == nullptr || raw_bindings.size() != 2 ||
      workspace_size < state.workspace_bytes ||
      (state.workspace_bytes != 0 && workspace == nullptr)) {
    throw std::invalid_argument(
        "muDNN Reduction execute arguments are invalid");
  }
  std::unordered_map<std::int64_t, void*> bindings;
  bindings.reserve(raw_bindings.size());
  for (const flagdnnBinding_t& binding : raw_bindings) {
    if (binding.device_pointer == nullptr ||
        !bindings.emplace(binding.uid, binding.device_pointer).second) {
      throw std::invalid_argument("muDNN Reduction binding is invalid");
    }
  }

  const musaStream_t musa_stream = reinterpret_cast<musaStream_t>(stream);
  check_mudnn(
      state.handle.SetStream(musa_stream),
      "muDNN Handle::SetStream(Reduction)");
  if (state.handle.GetStream() != musa_stream) {
    throw std::runtime_error(
        "muDNN Reduction did not retain the caller stream");
  }

  musa::dnn::Tensor input;
  musa::dnn::Tensor output;
  configure_tensor(
      input,
      state.descriptor.input,
      binding_pointer(state.descriptor.input, bindings));
  configure_tensor(
      output,
      state.output_view,
      binding_pointer(state.descriptor.output, bindings));

  musa::dnn::Tensor dense_input;
  musa::dnn::Tensor dense_output;
  musa::dnn::Tensor* reduction_input = &input;
  musa::dnn::Tensor* reduction_output = &output;
  void* scratch = workspace;
  if (state.bridge_layout) {
    auto* workspace_bytes = static_cast<std::byte*>(workspace);
    configure_tensor(
        dense_input,
        state.reduction_input,
        workspace_bytes + state.dense_input_offset);
    configure_tensor(
        dense_output,
        state.reduction_output,
        workspace_bytes + state.dense_output_offset);
    check_mudnn(
        state.permute.Run(state.handle, dense_input, input),
        "muDNN Permute::Run(Reduction gather)");
    reduction_input = &dense_input;
    reduction_output = &dense_output;
    scratch = state.scratch_bytes == 0
                  ? nullptr
                  : workspace_bytes + state.scratch_offset;
  }

  bool workspace_issued = false;
  const musa::dnn::MemoryMaintainer maintainer =
      [scratch, scratch_size = state.scratch_bytes, &workspace_issued](
          std::size_t requested) -> musa::dnn::MemoryHandler {
    if (requested == 0) {
      return musa::dnn::MemoryHandler(nullptr, [](void*) {});
    }
    if (workspace_issued || scratch == nullptr ||
        requested > scratch_size) {
      throw std::runtime_error(
          "muDNN Reduction workspace request exceeds its contract");
    }
    workspace_issued = true;
    return musa::dnn::MemoryHandler(scratch, [](void*) {});
  };
  check_mudnn(
      state.reduction.Run(
          state.handle, *reduction_output, *reduction_input, maintainer),
      "muDNN Reduce::Run");
  if (state.bridge_layout) {
    check_mudnn(
        state.permute.Run(state.handle, output, dense_output),
        "muDNN Permute::Run(Reduction scatter)");
  }
}

}  // namespace flagdnn::validation::mthreads
