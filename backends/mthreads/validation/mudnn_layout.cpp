/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "backends/mthreads/validation/mudnn_layout.hpp"

#include "backends/mthreads/validation/musa_driver.hpp"
#include "backends/mthreads/validation/tensor_io.hpp"

#include <mudnn.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace flagdnn::validation::mthreads {
namespace {

musa::dnn::Tensor::Type mudnn_data_type(flagdnnDataType_t data_type) {
  switch (data_type) {
    case FLAGDNN_DATA_INT32:
      return musa::dnn::Tensor::Type::INT32;

    case FLAGDNN_DATA_FLOAT32:
      return musa::dnn::Tensor::Type::FLOAT;
    case FLAGDNN_DATA_FLOAT16:
      return musa::dnn::Tensor::Type::HALF;
    case FLAGDNN_DATA_BFLOAT16:
      return musa::dnn::Tensor::Type::BFLOAT16;
    case FLAGDNN_DATA_BOOLEAN:
      return musa::dnn::Tensor::Type::BOOL;
    case FLAGDNN_DATA_FP8_E8M0:
    case FLAGDNN_DATA_FP8_E4M3:
    case FLAGDNN_DATA_FP8_E5M2:
      return musa::dnn::Tensor::Type::UINT8;
  }
  throw std::invalid_argument("muDNN layout tensor type is unsupported");
}

std::size_t element_count(const TensorDescriptor& tensor) {
  std::size_t result = 1;
  for (const std::int64_t dimension : tensor.dimensions) {
    const std::size_t value = static_cast<std::size_t>(dimension);
    if (value != 0 &&
        result > std::numeric_limits<std::size_t>::max() / value) {
      throw std::overflow_error("muDNN layout element count overflows");
    }
    result *= value;
  }
  return result;
}

std::vector<std::int64_t> contiguous_strides(
    std::span<const std::int64_t> dimensions) {
  std::vector<std::int64_t> result(dimensions.size());
  std::int64_t stride = 1;
  for (std::size_t axis = dimensions.size(); axis != 0; --axis) {
    result[axis - 1] = stride;
    const std::int64_t dimension = dimensions[axis - 1];
    if (dimension > std::numeric_limits<std::int64_t>::max() / stride) {
      throw std::overflow_error("muDNN layout dense stride overflows");
    }
    stride *= dimension;
  }
  return result;
}

void validate_tensor(const TensorDescriptor& tensor,
                     std::string_view name) {
  if (tensor.uid <= 0 || tensor.dimensions.empty() ||
      tensor.dimensions.size() > 8 ||
      tensor.dimensions.size() != tensor.strides.size()) {
    throw std::invalid_argument(
        std::string(name) + " muDNN layout descriptor is invalid");
  }
  for (std::size_t axis = 0; axis < tensor.dimensions.size(); ++axis) {
    if (tensor.dimensions[axis] <= 0 || tensor.strides[axis] <= 0) {
      throw std::invalid_argument(
          std::string(name) + " muDNN layout dimensions are invalid");
    }
  }
  static_cast<void>(mudnn_data_type(tensor.data_type));
  static_cast<void>(element_count(tensor));
  static_cast<void>(tensor_io::storage_element_count(tensor));
}

void validate_descriptor(const MudnnLayoutDescriptor& descriptor) {
  validate_tensor(descriptor.input, "input");
  validate_tensor(descriptor.output, "output");
  if (descriptor.input.uid == descriptor.output.uid ||
      descriptor.input.data_type != descriptor.output.data_type) {
    throw std::invalid_argument(
        "muDNN layout requires distinct same-type tensors");
  }

  const std::size_t rank = descriptor.input.dimensions.size();
  switch (descriptor.mode) {
    case MudnnLayoutMode::kReshape:
      if (element_count(descriptor.input) !=
              element_count(descriptor.output) ||
          !descriptor.permutation.empty() || !descriptor.slices.empty() ||
          !descriptor.slice_strides.empty()) {
        throw std::invalid_argument(
            "muDNN reshape descriptor is invalid");
      }
      break;
    case MudnnLayoutMode::kTranspose: {
      if (descriptor.output.dimensions.size() != rank ||
          descriptor.permutation.size() != rank ||
          !descriptor.slices.empty() ||
          !descriptor.slice_strides.empty()) {
        throw std::invalid_argument(
            "muDNN transpose descriptor is invalid");
      }
      std::unordered_set<std::int64_t> axes;
      for (std::size_t axis = 0; axis < rank; ++axis) {
        const std::int64_t source = descriptor.permutation[axis];
        if (source < 0 || source >= static_cast<std::int64_t>(rank) ||
            !axes.insert(source).second ||
            descriptor.output.dimensions[axis] !=
                descriptor.input.dimensions[
                    static_cast<std::size_t>(source)]) {
          throw std::invalid_argument(
              "muDNN transpose permutation is invalid");
        }
      }
      break;
    }
    case MudnnLayoutMode::kSlice:
      if (descriptor.output.dimensions.size() != rank ||
          descriptor.slices.size() != rank ||
          descriptor.slice_strides.size() != rank ||
          !descriptor.permutation.empty()) {
        throw std::invalid_argument("muDNN slice descriptor is invalid");
      }
      for (std::size_t axis = 0; axis < rank; ++axis) {
        const auto [start, limit] = descriptor.slices[axis];
        const std::int64_t step = descriptor.slice_strides[axis];
        if (start < 0 || limit <= start ||
            limit > descriptor.input.dimensions[axis] || step <= 0 ||
            descriptor.output.dimensions[axis] !=
                1 + (limit - start - 1) / step) {
          throw std::invalid_argument("muDNN slice range is invalid");
        }
      }
      break;
  }
}

void* binding_pointer(
    const TensorDescriptor& tensor,
    const std::unordered_map<std::int64_t, void*>& bindings) {
  const auto found = bindings.find(tensor.uid);
  if (found == bindings.end()) {
    throw std::invalid_argument("muDNN layout binding UID is missing");
  }
  return found->second;
}

void* offset_pointer(void* pointer,
                     std::int64_t element_offset,
                     std::size_t element_size) {
  if (element_offset < 0 ||
      static_cast<std::uint64_t>(element_offset) >
          std::numeric_limits<std::size_t>::max() / element_size) {
    throw std::overflow_error("muDNN layout element offset overflows");
  }
  const std::size_t byte_offset =
      static_cast<std::size_t>(element_offset) * element_size;
  const std::uintptr_t base = reinterpret_cast<std::uintptr_t>(pointer);
  if (base > std::numeric_limits<std::uintptr_t>::max() - byte_offset) {
    throw std::overflow_error("muDNN layout pointer offset overflows");
  }
  return reinterpret_cast<void*>(base + byte_offset);
}

void configure_tensor(musa::dnn::Tensor& tensor,
                      const TensorDescriptor& descriptor,
                      void* pointer) {
  check_mudnn(tensor.SetAddr(pointer), "muDNN Tensor::SetAddr(layout)");
  check_mudnn(tensor.SetType(mudnn_data_type(descriptor.data_type)),
              "muDNN Tensor::SetType(layout)");
  check_mudnn(
      tensor.SetNdInfo(
          static_cast<std::int64_t>(descriptor.dimensions.size()),
          descriptor.dimensions.data(),
          descriptor.strides.data()),
      "muDNN Tensor::SetNdInfo(layout)");
}

}  // namespace

struct MudnnLayoutOperation::Impl {
  explicit Impl(MudnnLayoutDescriptor value)
      : descriptor(std::move(value)), handle(0) {
    validate_descriptor(descriptor);
    const std::size_t count = element_count(descriptor.output);
    if (descriptor.mode == MudnnLayoutMode::kReshape) {
      const std::size_t width =
          tensor_io::data_type_size(descriptor.input.data_type);
      if (count > std::numeric_limits<std::size_t>::max() / width) {
        throw std::overflow_error("muDNN reshape workspace overflows");
      }
      workspace_bytes = count * width;
      return;
    }

    logical_dimensions = descriptor.output.dimensions;
    logical_input_strides.resize(logical_dimensions.size());
    if (descriptor.mode == MudnnLayoutMode::kTranspose) {
      for (std::size_t axis = 0; axis < logical_dimensions.size(); ++axis) {
        logical_input_strides[axis] = descriptor.input.strides[
            static_cast<std::size_t>(descriptor.permutation[axis])];
      }
      return;
    }

    for (std::size_t axis = 0; axis < logical_dimensions.size(); ++axis) {
      const std::int64_t input_stride = descriptor.input.strides[axis];
      const std::int64_t step = descriptor.slice_strides[axis];
      const std::int64_t start = descriptor.slices[axis].first;
      if (step > std::numeric_limits<std::int64_t>::max() / input_stride) {
        throw std::overflow_error(
            "muDNN slice effective stride overflows");
      }
      logical_input_strides[axis] = step * input_stride;
      if (start != 0 &&
          input_stride >
              (std::numeric_limits<std::int64_t>::max() -
               input_base_elements) /
                  start) {
        throw std::overflow_error("muDNN slice input base overflows");
      }
      input_base_elements += start * input_stride;
    }
  }

  MudnnLayoutDescriptor descriptor;
  musa::dnn::Handle handle;
  musa::dnn::Permute permute;
  std::vector<std::int64_t> logical_dimensions;
  std::vector<std::int64_t> logical_input_strides;
  std::int64_t input_base_elements = 0;
  std::size_t workspace_bytes = 0;
};

MudnnLayoutOperation::MudnnLayoutOperation(
    MudnnLayoutDescriptor descriptor)
    : implementation_(std::make_unique<Impl>(std::move(descriptor))) {}

MudnnLayoutOperation::~MudnnLayoutOperation() = default;

std::size_t MudnnLayoutOperation::workspace_size() const noexcept {
  return implementation_->workspace_bytes;
}

void MudnnLayoutOperation::execute(
    std::span<const flagdnnBinding_t> raw_bindings,
    void* workspace,
    std::size_t workspace_size,
    flagdnnStream_t stream) {
  Impl& state = *implementation_;
  if (stream == nullptr || raw_bindings.size() != 2 ||
      workspace_size < state.workspace_bytes ||
      (state.workspace_bytes != 0 && workspace == nullptr)) {
    throw std::invalid_argument("muDNN layout execute arguments are invalid");
  }
  std::unordered_map<std::int64_t, void*> bindings;
  bindings.reserve(raw_bindings.size());
  for (const flagdnnBinding_t& binding : raw_bindings) {
    if (binding.device_pointer == nullptr ||
        !bindings.emplace(binding.uid, binding.device_pointer).second) {
      throw std::invalid_argument("muDNN layout binding is invalid");
    }
  }

  const musaStream_t musa_stream = reinterpret_cast<musaStream_t>(stream);
  check_mudnn(state.handle.SetStream(musa_stream),
              "muDNN Handle::SetStream(layout)");
  if (state.handle.GetStream() != musa_stream) {
    throw std::runtime_error("muDNN layout did not retain the caller stream");
  }

  void* input_pointer = binding_pointer(state.descriptor.input, bindings);
  void* output_pointer = binding_pointer(state.descriptor.output, bindings);
  if (state.descriptor.mode == MudnnLayoutMode::kReshape) {
    TensorDescriptor dense_input = state.descriptor.input;
    dense_input.strides = contiguous_strides(dense_input.dimensions);
    TensorDescriptor dense_output = state.descriptor.output;
    dense_output.strides = contiguous_strides(dense_output.dimensions);

    musa::dnn::Tensor input;
    musa::dnn::Tensor first_output;
    configure_tensor(input, state.descriptor.input, input_pointer);
    configure_tensor(first_output, dense_input, workspace);
    check_mudnn(state.permute.Run(state.handle, first_output, input),
                "muDNN Permute::Run(reshape gather)");

    musa::dnn::Tensor second_input;
    musa::dnn::Tensor output;
    configure_tensor(second_input, dense_output, workspace);
    configure_tensor(output, state.descriptor.output, output_pointer);
    check_mudnn(state.permute.Run(state.handle, output, second_input),
                "muDNN Permute::Run(reshape scatter)");
    return;
  }

  TensorDescriptor logical_input = state.descriptor.input;
  logical_input.dimensions = state.logical_dimensions;
  logical_input.strides = state.logical_input_strides;
  input_pointer = offset_pointer(
      input_pointer,
      state.input_base_elements,
      tensor_io::data_type_size(state.descriptor.input.data_type));
  musa::dnn::Tensor input;
  musa::dnn::Tensor output;
  configure_tensor(input, logical_input, input_pointer);
  configure_tensor(output, state.descriptor.output, output_pointer);
  check_mudnn(state.permute.Run(state.handle, output, input),
              "muDNN Permute::Run(layout)");
}

}  // namespace flagdnn::validation::mthreads
