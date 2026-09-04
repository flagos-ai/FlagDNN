/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "backends/mthreads/validation/mudnn_matmul.hpp"

#include "backends/mthreads/validation/musa_driver.hpp"
#include "backends/mthreads/validation/tensor_io.hpp"

#include <mudnn.h>

#include <algorithm>
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
  throw std::invalid_argument("muDNN Matmul tensor type is unsupported");
}

void validate_tensor(const TensorDescriptor& tensor,
                     std::string_view name) {
  if (tensor.uid <= 0 || tensor.dimensions.size() < 2 ||
      tensor.dimensions.size() > 8 ||
      tensor.dimensions.size() != tensor.strides.size()) {
    throw std::invalid_argument(
        std::string(name) + " muDNN Matmul descriptor is invalid");
  }
  for (std::size_t axis = 0; axis < tensor.dimensions.size(); ++axis) {
    if (tensor.dimensions[axis] <= 0 || tensor.strides[axis] <= 0) {
      throw std::invalid_argument(
          std::string(name) +
          " muDNN Matmul dimensions/strides must be positive");
    }
  }
  static_cast<void>(mudnn_data_type(tensor.data_type));
  static_cast<void>(tensor_io::element_count(tensor));
  static_cast<void>(tensor_io::storage_element_count(tensor));
}

std::size_t checked_multiply(std::size_t left,
                             std::size_t right,
                             std::string_view description) {
  if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right) {
    throw std::overflow_error(std::string(description) + " overflows");
  }
  return left * right;
}

std::size_t checked_add(std::size_t left,
                        std::size_t right,
                        std::string_view description) {
  if (left > std::numeric_limits<std::size_t>::max() - right) {
    throw std::overflow_error(std::string(description) + " overflows");
  }
  return left + right;
}

std::size_t align_up(std::size_t value, std::size_t alignment) {
  const std::size_t remainder = value % alignment;
  if (remainder == 0) {
    return value;
  }
  return checked_add(
      value, alignment - remainder, "muDNN Matmul workspace alignment");
}

std::vector<std::int64_t> broadcast_batch_dimensions(
    const TensorDescriptor& a,
    const TensorDescriptor& b) {
  const std::size_t a_rank = a.dimensions.size() - 2;
  const std::size_t b_rank = b.dimensions.size() - 2;
  const std::size_t rank = std::max(a_rank, b_rank);
  std::vector<std::int64_t> result(rank, 1);
  for (std::size_t trailing = 0; trailing < rank; ++trailing) {
    const std::int64_t a_dimension =
        trailing < a_rank ? a.dimensions[a_rank - 1 - trailing] : 1;
    const std::int64_t b_dimension =
        trailing < b_rank ? b.dimensions[b_rank - 1 - trailing] : 1;
    if (a_dimension != b_dimension && a_dimension != 1 &&
        b_dimension != 1) {
      throw std::invalid_argument(
          "muDNN Matmul batch dimensions do not broadcast");
    }
    result[rank - 1 - trailing] = std::max(a_dimension, b_dimension);
  }
  return result;
}

void validate_descriptor(const MudnnMatmulDescriptor& descriptor) {
  validate_tensor(descriptor.a, "A");
  validate_tensor(descriptor.b, "B");
  validate_tensor(descriptor.output, "output");
  if (descriptor.a.uid == descriptor.b.uid ||
      descriptor.a.uid == descriptor.output.uid ||
      descriptor.b.uid == descriptor.output.uid ||
      descriptor.a.data_type != descriptor.b.data_type ||
      descriptor.a.data_type != descriptor.output.data_type ||
      descriptor.a.dimensions.back() !=
          descriptor.b.dimensions[descriptor.b.dimensions.size() - 2]) {
    throw std::invalid_argument(
        "muDNN Matmul tensor identities, types, or contraction are invalid");
  }
  std::vector<std::int64_t> expected =
      broadcast_batch_dimensions(descriptor.a, descriptor.b);
  expected.push_back(
      descriptor.a.dimensions[descriptor.a.dimensions.size() - 2]);
  expected.push_back(descriptor.b.dimensions.back());
  if (descriptor.output.dimensions != expected) {
    throw std::invalid_argument("muDNN Matmul output shape is invalid");
  }
}

bool matrix_layout_supported(const TensorDescriptor& tensor) {
  const std::size_t rank = tensor.dimensions.size();
  return tensor.strides[rank - 1] == 1 &&
         tensor.strides[rank - 2] >= tensor.dimensions[rank - 1];
}

bool direct_matrix(const MudnnMatmulDescriptor& descriptor) {
  return descriptor.a.dimensions.size() == 2 &&
         descriptor.b.dimensions.size() == 2 &&
         descriptor.output.dimensions.size() == 2 &&
         matrix_layout_supported(descriptor.a) &&
         matrix_layout_supported(descriptor.b) &&
         matrix_layout_supported(descriptor.output);
}

bool direct_batch_matrix(const MudnnMatmulDescriptor& descriptor) {
  if (descriptor.output.dimensions.size() != 3 ||
      descriptor.a.dimensions.size() < 2 ||
      descriptor.a.dimensions.size() > 3 ||
      descriptor.b.dimensions.size() < 2 ||
      descriptor.b.dimensions.size() > 3 ||
      !matrix_layout_supported(descriptor.a) ||
      !matrix_layout_supported(descriptor.b) ||
      !matrix_layout_supported(descriptor.output)) {
    return false;
  }
  const std::int64_t batch = descriptor.output.dimensions[0];
  const auto batch_supported = [batch](const TensorDescriptor& tensor) {
    return tensor.dimensions.size() == 2 || tensor.dimensions[0] == 1 ||
           tensor.dimensions[0] == batch;
  };
  return batch_supported(descriptor.a) && batch_supported(descriptor.b);
}

TensorDescriptor matrix_view(const TensorDescriptor& tensor) {
  const std::size_t rank = tensor.dimensions.size();
  TensorDescriptor result = tensor;
  result.dimensions = {
      tensor.dimensions[rank - 2], tensor.dimensions[rank - 1]};
  result.strides = {tensor.strides[rank - 2], tensor.strides[rank - 1]};
  result.binding_byte_offset = 0;
  return result;
}

TensorDescriptor dense_matrix(const TensorDescriptor& tensor) {
  TensorDescriptor result = matrix_view(tensor);
  result.strides = {result.dimensions[1], 1};
  return result;
}

std::size_t tensor_bytes(const TensorDescriptor& tensor) {
  return checked_multiply(
      tensor_io::element_count(tensor),
      tensor_io::data_type_size(tensor.data_type),
      "muDNN Matmul dense tensor bytes");
}

void configure_tensor_descriptor(musa::dnn::Tensor& tensor,
                                 const TensorDescriptor& descriptor) {
  check_mudnn(tensor.SetType(mudnn_data_type(descriptor.data_type)),
              "muDNN Tensor::SetType(Matmul)");
  check_mudnn(
      tensor.SetNdInfo(
          static_cast<std::int64_t>(descriptor.dimensions.size()),
          descriptor.dimensions.data(),
          descriptor.strides.data()),
      "muDNN Tensor::SetNdInfo(Matmul)");
}

void configure_tensor(musa::dnn::Tensor& tensor,
                      const TensorDescriptor& descriptor,
                      void* pointer) {
  check_mudnn(tensor.SetAddr(pointer), "muDNN Tensor::SetAddr(Matmul)");
  configure_tensor_descriptor(tensor, descriptor);
}

void* binding_pointer(
    const TensorDescriptor& tensor,
    const std::unordered_map<std::int64_t, void*>& bindings) {
  const auto found = bindings.find(tensor.uid);
  if (found == bindings.end()) {
    throw std::invalid_argument("muDNN Matmul binding UID is missing");
  }
  return found->second;
}

void* offset_pointer(void* pointer,
                     std::int64_t element_offset,
                     std::size_t element_size) {
  if (element_offset < 0 ||
      static_cast<std::uint64_t>(element_offset) >
          std::numeric_limits<std::size_t>::max() / element_size) {
    throw std::overflow_error("muDNN Matmul element offset overflows");
  }
  const std::size_t byte_offset =
      static_cast<std::size_t>(element_offset) * element_size;
  const std::uintptr_t base = reinterpret_cast<std::uintptr_t>(pointer);
  if (base > std::numeric_limits<std::uintptr_t>::max() - byte_offset) {
    throw std::overflow_error("muDNN Matmul pointer offset overflows");
  }
  return reinterpret_cast<void*>(base + byte_offset);
}

std::int64_t batch_offset(
    const TensorDescriptor& tensor,
    std::span<const std::int64_t> coordinates) {
  const std::size_t tensor_rank = tensor.dimensions.size() - 2;
  if (tensor_rank > coordinates.size()) {
    throw std::invalid_argument("muDNN Matmul batch rank is invalid");
  }
  const std::size_t leading = coordinates.size() - tensor_rank;
  std::int64_t result = 0;
  for (std::size_t axis = 0; axis < tensor_rank; ++axis) {
    if (tensor.dimensions[axis] == 1) {
      continue;
    }
    const std::int64_t coordinate = coordinates[leading + axis];
    if (coordinate >
        (std::numeric_limits<std::int64_t>::max() - result) /
            tensor.strides[axis]) {
      throw std::overflow_error("muDNN Matmul batch offset overflows");
    }
    result += coordinate * tensor.strides[axis];
  }
  return result;
}

musa::dnn::MemoryMaintainer workspace_maintainer(
    void* scratch,
    std::size_t scratch_size,
    bool& workspace_issued) {
  return [scratch, scratch_size, &workspace_issued](
             std::size_t requested) -> musa::dnn::MemoryHandler {
    if (requested == 0) {
      return musa::dnn::MemoryHandler(nullptr, [](void*) {});
    }
    if (workspace_issued || scratch == nullptr || requested > scratch_size) {
      throw std::runtime_error(
          "muDNN Matmul workspace request exceeds its contract");
    }
    workspace_issued = true;
    return musa::dnn::MemoryHandler(scratch, [](void*) {});
  };
}

}  // namespace

struct MudnnMatmulOperation::Impl {
  explicit Impl(MudnnMatmulDescriptor value)
      : descriptor(std::move(value)), handle(0) {
    validate_descriptor(descriptor);
    use_direct_matrix = direct_matrix(descriptor);
    use_direct_batch = direct_batch_matrix(descriptor);
    batch_dimensions =
        broadcast_batch_dimensions(descriptor.a, descriptor.b);
    batch_count = 1;
    for (const std::int64_t dimension : batch_dimensions) {
      batch_count = checked_multiply(
          batch_count,
          static_cast<std::size_t>(dimension),
          "muDNN Matmul batch count");
    }
    dense_a = dense_matrix(descriptor.a);
    dense_b = dense_matrix(descriptor.b);
    dense_output = dense_matrix(descriptor.output);

    const auto configure_matrix = [this](musa::dnn::MatMul& operation) {
      check_mudnn(operation.SetComputeMode(
                      descriptor.a.data_type == FLAGDNN_DATA_FLOAT32
                          ? musa::dnn::MatMul::ComputeMode::SCALAR
                          : musa::dnn::MatMul::ComputeMode::TENSOR),
                  "muDNN MatMul::SetComputeMode");
      check_mudnn(operation.SetTranspose(false, false),
                  "muDNN MatMul::SetTranspose");
      check_mudnn(operation.SetAlpha(1.0), "muDNN MatMul::SetAlpha");
      check_mudnn(operation.SetBeta(0.0), "muDNN MatMul::SetBeta");
    };
    const auto configure_batch = [this](
                                     musa::dnn::BatchMatMul& operation) {
      check_mudnn(operation.SetComputeMode(
                      descriptor.a.data_type == FLAGDNN_DATA_FLOAT32
                          ? musa::dnn::BatchMatMul::ComputeMode::SCALAR
                          : musa::dnn::BatchMatMul::ComputeMode::TENSOR),
                  "muDNN BatchMatMul::SetComputeMode");
      check_mudnn(operation.SetTranspose(false, false),
                  "muDNN BatchMatMul::SetTranspose");
      check_mudnn(operation.SetAlpha(1.0),
                  "muDNN BatchMatMul::SetAlpha");
      check_mudnn(operation.SetBeta(0.0),
                  "muDNN BatchMatMul::SetBeta");
    };
    configure_matrix(matmul);
    configure_batch(batch_matmul);

    musa::dnn::Tensor a;
    musa::dnn::Tensor b;
    musa::dnn::Tensor output;
    if (use_direct_matrix || use_direct_batch) {
      configure_tensor_descriptor(a, descriptor.a);
      configure_tensor_descriptor(b, descriptor.b);
      configure_tensor_descriptor(output, descriptor.output);
      if (use_direct_matrix) {
        check_mudnn(
            matmul.GetWorkspaceSize(handle, scratch_bytes, output, a, b),
            "muDNN MatMul::GetWorkspaceSize");
      } else {
        check_mudnn(batch_matmul.GetWorkspaceSize(
                        handle, scratch_bytes, output, a, b),
                    "muDNN BatchMatMul::GetWorkspaceSize");
      }
      workspace_bytes = scratch_bytes;
      return;
    }

    configure_tensor_descriptor(a, dense_a);
    configure_tensor_descriptor(b, dense_b);
    configure_tensor_descriptor(output, dense_output);
    check_mudnn(
        matmul.GetWorkspaceSize(handle, scratch_bytes, output, a, b),
        "muDNN MatMul::GetWorkspaceSize(bridge)");
    dense_a_offset = 0;
    dense_b_offset = align_up(tensor_bytes(dense_a), kWorkspaceAlignment);
    dense_output_offset = align_up(
        checked_add(dense_b_offset,
                    tensor_bytes(dense_b),
                    "muDNN Matmul dense B workspace"),
        kWorkspaceAlignment);
    scratch_offset = align_up(
        checked_add(dense_output_offset,
                    tensor_bytes(dense_output),
                    "muDNN Matmul dense output workspace"),
        kWorkspaceAlignment);
    workspace_bytes = checked_add(
        scratch_offset, scratch_bytes, "muDNN Matmul total workspace");
  }

  static constexpr std::size_t kWorkspaceAlignment = 256;

  MudnnMatmulDescriptor descriptor;
  TensorDescriptor dense_a;
  TensorDescriptor dense_b;
  TensorDescriptor dense_output;
  std::vector<std::int64_t> batch_dimensions;
  std::size_t batch_count = 1;
  musa::dnn::Handle handle;
  musa::dnn::MatMul matmul;
  musa::dnn::BatchMatMul batch_matmul;
  musa::dnn::Permute permute;
  bool use_direct_matrix = false;
  bool use_direct_batch = false;
  std::size_t dense_a_offset = 0;
  std::size_t dense_b_offset = 0;
  std::size_t dense_output_offset = 0;
  std::size_t scratch_offset = 0;
  std::size_t scratch_bytes = 0;
  std::size_t workspace_bytes = 0;
};

MudnnMatmulOperation::MudnnMatmulOperation(
    MudnnMatmulDescriptor descriptor)
    : implementation_(std::make_unique<Impl>(std::move(descriptor))) {}

MudnnMatmulOperation::~MudnnMatmulOperation() = default;

std::size_t MudnnMatmulOperation::workspace_size() const noexcept {
  return implementation_->workspace_bytes;
}

void MudnnMatmulOperation::execute(
    std::span<const flagdnnBinding_t> raw_bindings,
    void* workspace,
    std::size_t workspace_size,
    flagdnnStream_t stream) {
  Impl& state = *implementation_;
  if (stream == nullptr || raw_bindings.size() != 3 ||
      workspace_size < state.workspace_bytes ||
      (state.workspace_bytes != 0 && workspace == nullptr)) {
    throw std::invalid_argument("muDNN Matmul execute arguments are invalid");
  }
  std::unordered_map<std::int64_t, void*> bindings;
  bindings.reserve(raw_bindings.size());
  for (const flagdnnBinding_t& binding : raw_bindings) {
    if (binding.device_pointer == nullptr ||
        !bindings.emplace(binding.uid, binding.device_pointer).second) {
      throw std::invalid_argument("muDNN Matmul binding is invalid");
    }
  }

  const musaStream_t musa_stream = reinterpret_cast<musaStream_t>(stream);
  check_mudnn(state.handle.SetStream(musa_stream),
              "muDNN Handle::SetStream(Matmul)");
  if (state.handle.GetStream() != musa_stream) {
    throw std::runtime_error("muDNN Matmul did not retain the caller stream");
  }

  void* const a_pointer = binding_pointer(state.descriptor.a, bindings);
  void* const b_pointer = binding_pointer(state.descriptor.b, bindings);
  void* const output_pointer =
      binding_pointer(state.descriptor.output, bindings);
  if (state.use_direct_matrix || state.use_direct_batch) {
    musa::dnn::Tensor a;
    musa::dnn::Tensor b;
    musa::dnn::Tensor output;
    configure_tensor(a, state.descriptor.a, a_pointer);
    configure_tensor(b, state.descriptor.b, b_pointer);
    configure_tensor(output, state.descriptor.output, output_pointer);
    bool workspace_issued = false;
    const musa::dnn::MemoryMaintainer maintainer = workspace_maintainer(
        workspace, state.scratch_bytes, workspace_issued);
    if (state.use_direct_matrix) {
      check_mudnn(state.matmul.Run(state.handle, output, a, b, maintainer),
                  "muDNN MatMul::Run");
    } else {
      check_mudnn(
          state.batch_matmul.Run(state.handle, output, a, b, maintainer),
          "muDNN BatchMatMul::Run");
    }
    return;
  }

  auto* workspace_bytes = static_cast<std::byte*>(workspace);
  void* const dense_a_pointer = workspace_bytes + state.dense_a_offset;
  void* const dense_b_pointer = workspace_bytes + state.dense_b_offset;
  void* const dense_output_pointer =
      workspace_bytes + state.dense_output_offset;
  void* const scratch = state.scratch_bytes == 0
                            ? nullptr
                            : workspace_bytes + state.scratch_offset;
  const std::size_t element_size =
      tensor_io::data_type_size(state.descriptor.a.data_type);
  std::vector<std::int64_t> coordinates(state.batch_dimensions.size(), 0);
  for (std::size_t batch = 0; batch < state.batch_count; ++batch) {
    std::size_t remaining = batch;
    for (std::size_t axis = coordinates.size(); axis != 0; --axis) {
      const std::size_t current = axis - 1;
      const std::size_t dimension = static_cast<std::size_t>(
          state.batch_dimensions[current]);
      coordinates[current] =
          static_cast<std::int64_t>(remaining % dimension);
      remaining /= dimension;
    }

    TensorDescriptor a_view = matrix_view(state.descriptor.a);
    TensorDescriptor b_view = matrix_view(state.descriptor.b);
    TensorDescriptor output_view = matrix_view(state.descriptor.output);
    musa::dnn::Tensor source_a;
    musa::dnn::Tensor source_b;
    musa::dnn::Tensor destination_output;
    configure_tensor(
        source_a,
        a_view,
        offset_pointer(a_pointer,
                       batch_offset(state.descriptor.a, coordinates),
                       element_size));
    configure_tensor(
        source_b,
        b_view,
        offset_pointer(b_pointer,
                       batch_offset(state.descriptor.b, coordinates),
                       element_size));
    configure_tensor(
        destination_output,
        output_view,
        offset_pointer(
            output_pointer,
            batch_offset(state.descriptor.output, coordinates),
            element_size));

    musa::dnn::Tensor dense_a;
    musa::dnn::Tensor dense_b;
    musa::dnn::Tensor dense_output;
    configure_tensor(dense_a, state.dense_a, dense_a_pointer);
    configure_tensor(dense_b, state.dense_b, dense_b_pointer);
    configure_tensor(dense_output, state.dense_output, dense_output_pointer);
    check_mudnn(state.permute.Run(state.handle, dense_a, source_a),
                "muDNN Permute::Run(Matmul gather A)");
    check_mudnn(state.permute.Run(state.handle, dense_b, source_b),
                "muDNN Permute::Run(Matmul gather B)");
    bool workspace_issued = false;
    const musa::dnn::MemoryMaintainer maintainer = workspace_maintainer(
        scratch, state.scratch_bytes, workspace_issued);
    check_mudnn(state.matmul.Run(
                    state.handle, dense_output, dense_a, dense_b, maintainer),
                "muDNN MatMul::Run(bridge)");
    check_mudnn(
        state.permute.Run(state.handle, destination_output, dense_output),
        "muDNN Permute::Run(Matmul scatter output)");
  }
}

}  // namespace flagdnn::validation::mthreads
