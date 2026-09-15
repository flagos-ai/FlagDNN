/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "backends/mthreads/validation/mudnn_add.hpp"

#include "backends/mthreads/validation/musa_driver.hpp"

#include <mudnn.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace flagdnn::validation::mthreads {
namespace {

musa::dnn::Tensor::Type mudnn_data_type(flagdnnDataType_t data_type) {
  switch (data_type) {
    case FLAGDNN_DATA_INT32:
      throw std::invalid_argument(
          "INT32 is not supported by this validation adapter");

    case FLAGDNN_DATA_FLOAT32:
      return musa::dnn::Tensor::Type::FLOAT;
    case FLAGDNN_DATA_FLOAT16:
      return musa::dnn::Tensor::Type::HALF;
    case FLAGDNN_DATA_BFLOAT16:
      return musa::dnn::Tensor::Type::BFLOAT16;
    case FLAGDNN_DATA_FP8_E8M0:
    case FLAGDNN_DATA_FP8_E4M3:
    case FLAGDNN_DATA_FP8_E5M2:
    case FLAGDNN_DATA_BOOLEAN:
      break;
  }
  throw std::invalid_argument("muDNN Add tensor data type is unsupported");
}

void validate_tensor(const TensorDescriptor& tensor,
                     std::string_view name) {
  if (tensor.uid <= 0 || tensor.dimensions.empty() ||
      tensor.dimensions.size() != tensor.strides.size()) {
    throw std::invalid_argument(
        std::string(name) + " muDNN tensor descriptor is invalid");
  }
  for (std::size_t axis = 0; axis < tensor.dimensions.size(); ++axis) {
    if (tensor.dimensions[axis] <= 0 || tensor.strides[axis] <= 0) {
      throw std::invalid_argument(
          std::string(name) +
          " muDNN tensor dimensions/strides must be positive");
    }
  }
  static_cast<void>(mudnn_data_type(tensor.data_type));
}

bool broadcasts_to(const TensorDescriptor& input,
                   const TensorDescriptor& output) {
  if (input.dimensions.size() > output.dimensions.size()) {
    return false;
  }
  const std::size_t leading =
      output.dimensions.size() - input.dimensions.size();
  for (std::size_t axis = 0; axis < input.dimensions.size(); ++axis) {
    const std::int64_t input_dimension = input.dimensions[axis];
    const std::int64_t output_dimension =
        output.dimensions[leading + axis];
    if (input_dimension != 1 && input_dimension != output_dimension) {
      return false;
    }
  }
  return true;
}

void validate_descriptor(const MudnnAddDescriptor& descriptor) {
  validate_tensor(descriptor.left, "left");
  validate_tensor(descriptor.right, "right");
  validate_tensor(descriptor.output, "output");
  if (descriptor.left.uid == descriptor.right.uid ||
      descriptor.left.uid == descriptor.output.uid ||
      descriptor.right.uid == descriptor.output.uid) {
    throw std::invalid_argument("muDNN Add tensor UIDs must be unique");
  }
  if (descriptor.left.data_type != descriptor.right.data_type ||
      descriptor.left.data_type != descriptor.output.data_type) {
    throw std::invalid_argument("muDNN Add tensor data types must match");
  }
  if (!broadcasts_to(descriptor.left, descriptor.output) ||
      !broadcasts_to(descriptor.right, descriptor.output)) {
    throw std::invalid_argument(
        "muDNN Add input shapes do not broadcast to output");
  }
  if (!std::isfinite(descriptor.alpha)) {
    throw std::invalid_argument("muDNN Add alpha must be finite");
  }
}

void* binding_pointer(
    const TensorDescriptor& tensor,
    const std::unordered_map<std::int64_t, void*>& bindings) {
  const auto found = bindings.find(tensor.uid);
  if (found == bindings.end()) {
    throw std::invalid_argument("muDNN Add binding UID is missing");
  }
  return found->second;
}

void configure_tensor(musa::dnn::Tensor& tensor,
                      const TensorDescriptor& descriptor,
                      void* pointer) {
  check_mudnn(tensor.SetAddr(pointer), "muDNN Tensor::SetAddr");
  check_mudnn(
      tensor.SetType(mudnn_data_type(descriptor.data_type)),
      "muDNN Tensor::SetType");
  check_mudnn(
      tensor.SetNdInfo(
          static_cast<std::int64_t>(descriptor.dimensions.size()),
          descriptor.dimensions.data(),
          descriptor.strides.data()),
      "muDNN Tensor::SetNdInfo");
}

}  // namespace

struct MudnnAddOperation::Impl {
  explicit Impl(MudnnAddDescriptor value)
      : descriptor(std::move(value)), handle(0) {
    validate_descriptor(descriptor);
    const bool scaled = descriptor.alpha != 1.0;
    check_mudnn(
        binary.SetMode(
            scaled ? musa::dnn::Binary::Mode::ADD_ALPHA
                   : musa::dnn::Binary::Mode::ADD),
        "muDNN Binary::SetMode(Add)");
    if (scaled) {
      check_mudnn(
          binary.SetAlpha(descriptor.alpha), "muDNN Binary::SetAlpha");
    }
  }

  MudnnAddDescriptor descriptor;
  musa::dnn::Handle handle;
  musa::dnn::Binary binary;
};

MudnnAddOperation::MudnnAddOperation(MudnnAddDescriptor descriptor)
    : implementation_(
          std::make_unique<Impl>(std::move(descriptor))) {}

MudnnAddOperation::~MudnnAddOperation() = default;

void MudnnAddOperation::execute(
    std::span<const flagdnnBinding_t> raw_bindings,
    void* workspace,
    std::size_t workspace_size,
    flagdnnStream_t stream) {
  static_cast<void>(workspace);
  if (workspace_size != 0 || stream == nullptr ||
      raw_bindings.size() != 3) {
    throw std::invalid_argument("muDNN Add execute arguments are invalid");
  }
  std::unordered_map<std::int64_t, void*> bindings;
  bindings.reserve(raw_bindings.size());
  for (const flagdnnBinding_t& binding : raw_bindings) {
    if (binding.device_pointer == nullptr ||
        !bindings.emplace(binding.uid, binding.device_pointer).second) {
      throw std::invalid_argument("muDNN Add binding is invalid");
    }
  }

  Impl& state = *implementation_;
  const musaStream_t musa_stream = reinterpret_cast<musaStream_t>(stream);
  check_mudnn(state.handle.SetStream(musa_stream),
              "muDNN Handle::SetStream");
  if (state.handle.GetStream() != musa_stream) {
    throw std::runtime_error("muDNN did not retain the caller stream");
  }

  musa::dnn::Tensor left;
  musa::dnn::Tensor right;
  musa::dnn::Tensor output;
  configure_tensor(
      left,
      state.descriptor.left,
      binding_pointer(state.descriptor.left, bindings));
  configure_tensor(
      right,
      state.descriptor.right,
      binding_pointer(state.descriptor.right, bindings));
  configure_tensor(
      output,
      state.descriptor.output,
      binding_pointer(state.descriptor.output, bindings));
  check_mudnn(
      state.binary.Run(state.handle, output, left, right),
      "muDNN Binary::Run(Add)");
}

}  // namespace flagdnn::validation::mthreads
