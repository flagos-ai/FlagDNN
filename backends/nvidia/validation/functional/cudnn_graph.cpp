/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "validation/functional/cudnn_graph.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include "validation/functional/cudnn_precision.hpp"
#include "validation/tensor_io.hpp"

namespace flagdnn::testing::cuda {

void check_cuda_runtime(cudaError_t status, std::string_view operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) +
                             " failed: " + cudaGetErrorString(status));
  }
}

void check_cudnn(cudnnStatus_t status, std::string_view operation) {
  if (status != CUDNN_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(operation) +
                             " failed: " + cudnnGetErrorString(status));
  }
}

void check_cudnn_frontend(cfe::error_t status, std::string_view operation) {
  if (status.is_bad()) {
    throw std::runtime_error(std::string(operation) +
                             " failed: " + status.get_message());
  }
}

cfe::DataType_t cudnn_frontend_data_type(flagdnnDataType_t data_type) {
  switch (data_type) {
    case FLAGDNN_DATA_INT32:
      return cfe::DataType_t::INT32;
    case FLAGDNN_DATA_FLOAT32:
      return cfe::DataType_t::FLOAT;
    case FLAGDNN_DATA_FLOAT16:
      return cfe::DataType_t::HALF;
    case FLAGDNN_DATA_BFLOAT16:
      return cfe::DataType_t::BFLOAT16;
    case FLAGDNN_DATA_BOOLEAN:
      return cfe::DataType_t::BOOLEAN;
    case FLAGDNN_DATA_FP8_E4M3:
      return cfe::DataType_t::FP8_E4M3;
    case FLAGDNN_DATA_FP8_E8M0:
      return cfe::DataType_t::FP8_E8M0;
    case FLAGDNN_DATA_FP8_E5M2:
      return cfe::DataType_t::FP8_E5M2;
  }
  throw std::invalid_argument("unsupported cuDNN Graph data type");
}

TestTensor padded_to_rank_four(const TestTensor& tensor,
                               std::size_t logical_rank) {
  if (logical_rank == 0 || logical_rank > 4 || tensor.dimensions.empty() ||
      tensor.dimensions.size() > logical_rank ||
      tensor.dimensions.size() != tensor.strides.size()) {
    throw std::invalid_argument(
        "cuDNN pointwise tensor received an invalid logical rank");
  }
  TestTensor result = tensor;
  const std::int64_t storage_span =
      static_cast<std::int64_t>(storage_element_count(tensor));
  const std::size_t leading = logical_rank - result.dimensions.size();
  result.dimensions.insert(result.dimensions.begin(), leading, 1);
  result.strides.insert(result.strides.begin(), leading, storage_span);
  if (logical_rank < 4) {
    result.dimensions.insert(result.dimensions.begin(), 1);
    result.strides.insert(result.strides.begin(), storage_span);
  }
  result.dimensions.resize(4, 1);
  result.strides.resize(4, 1);
  return result;
}

TestTensor flatten_compact_tensor(const TestTensor& tensor) {
  TestTensor result = tensor;
  const std::int64_t elements =
      static_cast<std::int64_t>(element_count(tensor));
  if (tensor.data_type == FLAGDNN_DATA_BOOLEAN) {
    result.dimensions = {1, elements, 1, 1};
    result.strides = {elements, 1, elements, elements};
  } else {
    result.dimensions = {1, elements, 1, 1};
    result.strides = {elements, 1, 1, 1};
  }
  return result;
}

TestTensor canonicalize_pointwise_tensor(const TestTensor& tensor,
                                         std::size_t logical_rank) {
  if (logical_rank == 0 || logical_rank > 4 || tensor.dimensions.empty() ||
      tensor.dimensions.size() > logical_rank ||
      tensor.dimensions.size() != tensor.strides.size()) {
    throw std::invalid_argument("cuDNN pointwise tensor rank is invalid");
  }
  if (logical_rank == 4) {
    return padded_to_rank_four(tensor, logical_rank);
  }

  TestTensor aligned = tensor;
  const std::size_t leading = logical_rank - tensor.dimensions.size();
  const std::int64_t storage_span =
      static_cast<std::int64_t>(storage_element_count(tensor));
  aligned.dimensions.insert(aligned.dimensions.begin(), leading, 1);
  aligned.strides.insert(aligned.strides.begin(), leading, storage_span);

  TestTensor result = aligned;
  if (logical_rank == 1) {
    result.dimensions = {1, aligned.dimensions[0], 1, 1};
    result.strides = {storage_span, aligned.strides[0], storage_span,
                      storage_span};
  } else if (logical_rank == 2) {
    result.dimensions = {aligned.dimensions[0], aligned.dimensions[1], 1, 1};
    result.strides = {aligned.strides[0], aligned.strides[1],
                      aligned.strides[0], aligned.strides[0]};
  } else {
    result.dimensions = {aligned.dimensions[0], aligned.dimensions[2],
                         aligned.dimensions[1], 1};
    result.strides = {aligned.strides[0], aligned.strides[2],
                      aligned.strides[1], aligned.strides[1]};
  }
  return result;
}

bool has_same_physical_mapping(const TestTensor& left,
                               const TestTensor& right) {
  if (left.dimensions != right.dimensions) {
    return false;
  }
  for (std::size_t axis = 0; axis < left.dimensions.size(); ++axis) {
    if (left.dimensions[axis] > 1 &&
        left.strides[axis] != right.strides[axis]) {
      return false;
    }
  }
  return true;
}

std::shared_ptr<cfe::graph::Tensor_attributes> make_cudnn_tensor(
    const std::shared_ptr<cfe::graph::Graph>& graph, const TestTensor& tensor,
    std::string_view name) {
  return graph->tensor(
      cfe::graph::Tensor_attributes()
          .set_name(std::string(name))
          .set_uid(tensor.uid)
          .set_data_type(cudnn_frontend_data_type(tensor.data_type))
          .set_dim(tensor.dimensions)
          .set_stride(tensor.strides));
}

CudnnBindingMap make_cudnn_binding_map(
    std::span<const flagdnnBinding_t> bindings) {
  CudnnBindingMap result;
  result.reserve(bindings.size());
  for (const flagdnnBinding_t& binding : bindings) {
    if (binding.uid <= 0 || binding.device_pointer == nullptr) {
      throw std::invalid_argument(
          "cuDNN binding UID and pointer must be valid");
    }
    if (!result.emplace(binding.uid, binding.device_pointer).second) {
      throw std::invalid_argument("cuDNN binding UID is duplicated");
    }
  }
  return result;
}

CudnnGraphExecutable::CudnnGraphExecutable() {
  check_cudnn(cudnnCreate(&handle_), "cudnnCreate");
}

CudnnGraphExecutable::~CudnnGraphExecutable() {
  if (handle_ != nullptr) {
    (void)cudnnDestroy(handle_);
  }
}

std::size_t CudnnGraphExecutable::workspace_size() const noexcept {
  return workspace_size_;
}

cudnnHandle_t CudnnGraphExecutable::handle() const noexcept { return handle_; }

void CudnnGraphExecutable::begin_execute(void* workspace,
                                         std::size_t workspace_size,
                                         flagdnnStream_t stream) {
  if (workspace_size < workspace_size_ ||
      (workspace_size_ != 0 && workspace == nullptr)) {
    throw std::invalid_argument("cuDNN Graph workspace is too small");
  }
  const cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  if (!stream_initialized_ || cuda_stream != stream_) {
    check_cudnn(cudnnSetStream(handle_, cuda_stream), "cudnnSetStream");
    stream_ = cuda_stream;
    stream_initialized_ = true;
  }
}

void CudnnGraphExecutable::set_workspace_size(std::int64_t workspace_size) {
  if (workspace_size < 0 || static_cast<std::uint64_t>(workspace_size) >
                                std::numeric_limits<std::size_t>::max()) {
    throw std::runtime_error("cuDNN returned an invalid workspace size");
  }
  workspace_size_ = static_cast<std::size_t>(workspace_size);
}

void build_cudnn_plans(cfe::graph::Graph& graph, cudnnHandle_t handle,
                       int input_precision,
                       std::span<const std::int64_t> allowed_engines) {
  require_cudnn_precision_environment(input_precision);
  check_cudnn_frontend(graph.validate(), "cuDNN graph validation");
  check_cudnn_frontend(graph.build_operation_graph(handle),
                       "cuDNN operation graph build");
  check_cudnn_frontend(graph.create_execution_plans(
                           {cfe::HeurMode_t::A, cfe::HeurMode_t::FALLBACK}),
                       "cuDNN execution plan discovery");
  if (input_precision == 1) {
    // Some TF32 matmul plans report TENSOR_CORE without DOWN_CONVERT_INPUTS.
    graph.deselect_numeric_notes(
        {cfe::NumericalNote_t::TENSOR_CORE,
         cfe::NumericalNote_t::DOWN_CONVERT_INPUTS,
         cfe::NumericalNote_t::REDUCED_PRECISION_REDUCTION});
  } else if (input_precision == 2) {
    // Tensor-core plans use TF32 for FP32 operands. DOWN_CONVERT_INPUTS
    // is not advertised by these engines and must not be required.
    graph.select_numeric_notes({cfe::NumericalNote_t::TENSOR_CORE});
    graph.deselect_numeric_notes(
        {cfe::NumericalNote_t::REDUCED_PRECISION_REDUCTION});
  } else if (input_precision != 0) {
    throw std::invalid_argument("invalid cuDNN input precision");
  }
  if (!allowed_engines.empty()) {
    std::vector<std::string> excluded;
    for (std::int64_t index = 0; index < graph.get_execution_plan_count();
         ++index) {
      std::int64_t engine = -1;
      std::unordered_map<cfe::KnobType_t, std::int64_t> knobs;
      check_cudnn_frontend(
          graph.get_engine_and_knobs_at_index(index, engine, knobs),
          "cuDNN engine identity query");
      if (std::find(allowed_engines.begin(), allowed_engines.end(), engine) ==
          allowed_engines.end())
        excluded.push_back("eng" + std::to_string(engine) + "_");
    }
    graph.deselect_engines(excluded);
  }
  check_cudnn_frontend(graph.check_support(handle),
                       "cuDNN graph support check");
  check_cudnn_frontend(graph.build_plans(handle), "cuDNN execution plan build");
}

namespace {
class BuiltCudnnGraph final : public CudnnGraphExecutable {
 public:
  explicit BuiltCudnnGraph(std::shared_ptr<cfe::graph::Graph> graph)
      : graph_(std::move(graph)) {
    build_cudnn_plans(*graph_, handle());
    std::int64_t size = 0;
    check_cudnn_frontend(graph_->get_workspace_size(size),
                         "cuDNN graph workspace query");
    set_workspace_size(size);
  }
  void execute(std::span<const flagdnnBinding_t> bindings, void* workspace,
               std::size_t size, flagdnnStream_t stream) override {
    begin_execute(workspace, size, stream);
    auto pointers = make_cudnn_binding_map(bindings);
    check_cudnn_frontend(graph_->execute(handle(), pointers, workspace),
                         "cuDNN graph execute");
  }

 private:
  std::shared_ptr<cfe::graph::Graph> graph_;
};
}  // namespace

std::unique_ptr<TestExecutable> build_cudnn_graph(
    std::shared_ptr<cfe::graph::Graph> graph) {
  return std::make_unique<BuiltCudnnGraph>(std::move(graph));
}

}  // namespace flagdnn::testing::cuda
