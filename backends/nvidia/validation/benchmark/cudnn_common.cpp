/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "cudnn_common.hpp"

#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace flagdnn::benchmarking::cudnn_detail {
namespace {

std::int64_t storage_element_count(const TensorSpec& input) {
  std::int64_t maximum_offset = 0;
  for (std::size_t axis = 0; axis < input.dimensions.size(); ++axis) {
    maximum_offset +=
        (input.dimensions[axis] - 1) * input.strides[axis];
  }
  return maximum_offset + 1;
}

}  // namespace

void check_cudnn(cudnnStatus_t status, const char* operation) {
  if (status != CUDNN_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(operation) + " failed: " +
                             cudnnGetErrorString(status));
  }
}

void check_frontend(fe::error_t status, const char* operation) {
  if (status.is_bad()) {
    throw std::runtime_error(std::string(operation) + " failed: " +
                             status.get_message());
  }
}

void build_frontend_layout_graph_or_unsupported(
    fe::graph::Graph& graph,
    cudnnHandle_t handle,
    std::string_view operation) {
  check_frontend(graph.validate(), "cuDNN layout graph validation");
  check_frontend(graph.build_operation_graph(handle),
                 "cuDNN layout operation graph lowering");

  const auto require_plan_stage = [operation](fe::error_t status,
                                               std::string_view stage) {
    if (status.is_good()) {
      return;
    }
    const fe::error_code_t code = status.get_code();
    const std::string message = status.get_message();
    if (code == fe::error_code_t::HEURISTIC_QUERY_FAILED ||
        code == fe::error_code_t::GRAPH_NOT_SUPPORTED ||
        code == fe::error_code_t::GRAPH_EXECUTION_PLAN_CREATION_FAILED) {
      throw BenchmarkUnsupportedError(
          std::string("cuDNN Frontend native ") +
          std::string(operation) +
          " graph validated and lowered, but the backend has no "
          "standalone execution plan during " +
          std::string(stage));
    }
    throw std::runtime_error(
        std::string("cuDNN Frontend native ") +
        std::string(operation) + " " + std::string(stage) +
        " failed: " + message);
  };

  require_plan_stage(
      graph.create_execution_plans(
          {fe::HeurMode_t::A, fe::HeurMode_t::FALLBACK}),
      "execution-plan discovery");
  require_plan_stage(graph.check_support(handle), "support check");
  require_plan_stage(graph.build_plans(handle), "plan build");
}

fe::DataType_t data_type(flagdnnDataType_t type) {
  switch (type) {
    case FLAGDNN_DATA_FLOAT32:
      return fe::DataType_t::FLOAT;
    case FLAGDNN_DATA_FLOAT16:
      return fe::DataType_t::HALF;
    case FLAGDNN_DATA_BFLOAT16:
      return fe::DataType_t::BFLOAT16;
    case FLAGDNN_DATA_BOOLEAN:
      return fe::DataType_t::BOOLEAN;
    case FLAGDNN_DATA_FP8_E4M3:
      return fe::DataType_t::FP8_E4M3;
    case FLAGDNN_DATA_FP8_E5M2:
      return fe::DataType_t::FP8_E5M2;
  }
  throw std::invalid_argument("unsupported cuDNN reference data type");
}

cudnnDataType_t cudnn_data_type(flagdnnDataType_t type) {
  switch (type) {
    case FLAGDNN_DATA_FLOAT32:
      return CUDNN_DATA_FLOAT;
    case FLAGDNN_DATA_FLOAT16:
      return CUDNN_DATA_HALF;
    case FLAGDNN_DATA_BFLOAT16:
      return CUDNN_DATA_BFLOAT16;
    case FLAGDNN_DATA_BOOLEAN:
      return CUDNN_DATA_BOOLEAN;
    case FLAGDNN_DATA_FP8_E4M3:
      return CUDNN_DATA_FP8_E4M3;
    case FLAGDNN_DATA_FP8_E5M2:
      return CUDNN_DATA_FP8_E5M2;
  }
  throw std::invalid_argument("unsupported cuDNN reference data type");
}

void set_tensor_descriptor(cudnnTensorDescriptor_t descriptor,
                           const TensorSpec& specification) {
  if (specification.dimensions.size() != specification.strides.size() ||
      specification.dimensions.size() >
          static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument("cuDNN tensor metadata is invalid");
  }
  std::vector<int> dimensions;
  std::vector<int> strides;
  dimensions.reserve(specification.dimensions.size());
  strides.reserve(specification.strides.size());
  for (std::size_t axis = 0; axis < specification.dimensions.size(); ++axis) {
    const std::int64_t dimension = specification.dimensions[axis];
    const std::int64_t stride = specification.strides[axis];
    if (dimension <= 0 || stride <= 0 ||
        dimension > std::numeric_limits<int>::max() ||
        stride > std::numeric_limits<int>::max()) {
      throw std::invalid_argument(
          "cuDNN tensor dimension or stride is out of range");
    }
    dimensions.push_back(static_cast<int>(dimension));
    strides.push_back(static_cast<int>(stride));
  }
  check_cudnn(
      cudnnSetTensorNdDescriptor(descriptor,
                                 cudnn_data_type(specification.data_type),
                                 static_cast<int>(dimensions.size()),
                                 dimensions.data(),
                                 strides.data()),
      "cudnnSetTensorNdDescriptor");
}

TensorSpec padded_to_rank_four(const TensorSpec& input,
                               std::size_t logical_rank) {
  if (logical_rank == 0 || logical_rank > 4 ||
      input.dimensions.size() > logical_rank) {
    throw std::invalid_argument(
        "cuDNN pointwise reference received an invalid logical rank");
  }
  TensorSpec result = input;
  const std::int64_t storage_span = storage_element_count(input);
  const std::size_t broadcast_leading =
      logical_rank - result.dimensions.size();
  result.dimensions.insert(
      result.dimensions.begin(), broadcast_leading, 1);
  result.strides.insert(
      result.strides.begin(), broadcast_leading, storage_span);
  if (logical_rank < 4) {
    result.dimensions.insert(result.dimensions.begin(), 1);
    result.strides.insert(result.strides.begin(), storage_span);
  }
  result.dimensions.resize(4, 1);
  result.strides.resize(4, 1);
  return result;
}

TensorSpec compact_unary_tensor(const TensorSpec& input,
                                std::size_t logical_rank) {
  TensorSpec result = padded_to_rank_four(input, logical_rank);
  std::int64_t elements = 1;
  std::int64_t channels = 1;
  for (const std::int64_t dimension : input.dimensions) {
    elements *= dimension;
    if (dimension > 1) {
      channels = dimension;
    }
  }

  if (input.data_type == FLAGDNN_DATA_BOOLEAN) {
    result.dimensions = {1, 1, 1, elements};
    result.strides = {elements, elements, elements, 1};
  } else {
    // Standalone cuDNN unary engines prefer an {N,C,1,1} view and Identity
    // has no engine when C == 1. Preserve that view without placing the full
    // tensor extent in C: the rightmost non-unit logical dimension is a
    // divisor of a compact tensor and keeps large benchmark shapes launchable.
    result.dimensions = {elements / channels, channels, 1, 1};
    result.strides = {channels, 1, 1, 1};
  }
  return result;
}

TensorSpec padded_to_minimum_rank_four(const TensorSpec& input) {
  if (input.dimensions.empty() || input.dimensions.size() > 8 ||
      input.dimensions.size() != input.strides.size()) {
    throw std::invalid_argument("cuDNN reduction tensor rank is invalid");
  }
  TensorSpec result = input;
  const std::int64_t storage_span = storage_element_count(input);
  while (result.dimensions.size() < 4) {
    result.dimensions.insert(result.dimensions.begin(), 1);
    result.strides.insert(result.strides.begin(), storage_span);
  }
  return result;
}

TensorSpec batchnorm_inference_nhwc_tensor(const TensorSpec& input) {
  if (input.dimensions.size() != 4 || input.strides.size() != 4) {
    throw std::invalid_argument(
        "cuDNN BatchNorm Inference reference requires rank-four tensors");
  }
  TensorSpec result = input;
  const std::int64_t channels = input.dimensions[1];
  const std::int64_t height = input.dimensions[2];
  const std::int64_t width = input.dimensions[3];
  result.strides = {
      channels * height * width, 1, width * channels, channels};
  return result;
}

std::shared_ptr<fe::graph::Tensor_attributes> make_tensor(
    const std::shared_ptr<fe::graph::Graph>& graph,
    const TensorSpec& specification,
    std::string_view name,
    bool output) {
  fe::graph::Tensor_attributes attributes;
  attributes.set_name(std::string(name))
      .set_uid(specification.uid)
      .set_data_type(data_type(specification.data_type))
      .set_dim(specification.dimensions)
      .set_stride(specification.strides);
  if (output) {
    attributes.set_output(true);
  }
  return graph->tensor(std::move(attributes));
}

void require_tensor_count(const BenchmarkCase& specification,
                          std::size_t expected) {
  if (specification.tensors.size() != expected) {
    throw std::invalid_argument("case tensor count is invalid");
  }
}

BindingMap make_binding_map(
    std::span<const flagdnnBinding_t> bindings) {
  BindingMap pointers;
  pointers.reserve(bindings.size());
  for (const flagdnnBinding_t& binding : bindings) {
    if (binding.uid <= 0 || binding.device_pointer == nullptr) {
      throw std::invalid_argument(
          "cuDNN reference binding UID or pointer is invalid");
    }
    if (!pointers.emplace(binding.uid, binding.device_pointer).second) {
      throw std::invalid_argument("cuDNN reference binding UID is duplicate");
    }
  }
  return pointers;
}

void* pointer_for(const BindingMap& pointers,
                  std::int64_t uid,
                  std::string_view operation) {
  const auto found = pointers.find(uid);
  if (found == pointers.end()) {
    throw std::invalid_argument(
        "cuDNN " + std::string(operation) +
        " binding is missing a required UID");
  }
  return found->second;
}

ExecutableBase::ExecutableBase() {
  check_cudnn(cudnnCreate(&handle_), "cudnnCreate");
}

ExecutableBase::~ExecutableBase() {
  if (handle_ != nullptr) {
    (void)cudnnDestroy(handle_);
  }
}

std::size_t ExecutableBase::workspace_size() const noexcept {
  return workspace_size_;
}

cudnnHandle_t ExecutableBase::handle() const noexcept {
  return handle_;
}

void ExecutableBase::begin_execute(void* workspace,
                                   std::size_t workspace_size,
                                   flagdnnStream_t stream) {
  if (workspace_size < workspace_size_ ||
      (workspace_size_ != 0 && workspace == nullptr)) {
    throw std::invalid_argument("cuDNN reference workspace is too small");
  }
  const cudaStream_t cuda_stream =
      reinterpret_cast<cudaStream_t>(stream);
  if (!stream_initialized_ || cuda_stream != stream_) {
    check_cudnn(cudnnSetStream(handle_, cuda_stream), "cudnnSetStream");
    stream_ = cuda_stream;
    stream_initialized_ = true;
  }
}

void ExecutableBase::set_workspace_size(std::int64_t workspace_size) {
  if (workspace_size < 0 ||
      static_cast<std::uint64_t>(workspace_size) >
          std::numeric_limits<std::size_t>::max()) {
    throw std::runtime_error("cuDNN returned an invalid workspace size");
  }
  workspace_size_ = static_cast<std::size_t>(workspace_size);
}

void ExecutableBase::set_workspace_size(
    std::size_t workspace_size) noexcept {
  workspace_size_ = workspace_size;
}

}  // namespace flagdnn::benchmarking::cudnn_detail
