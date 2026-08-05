/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/add.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <cudnn_frontend.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace flagdnn::testing {
namespace {

namespace fe = cudnn_frontend;

void check_cuda(cudaError_t status, std::string_view operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(
        std::string(operation) + " failed: " + cudaGetErrorString(status));
  }
}

void check_cudnn(cudnnStatus_t status, std::string_view operation) {
  if (status != CUDNN_STATUS_SUCCESS) {
    throw std::runtime_error(
        std::string(operation) + " failed: " + cudnnGetErrorString(status));
  }
}

void check_frontend(fe::error_t status, std::string_view operation) {
  if (status.is_bad()) {
    throw std::runtime_error(
        std::string(operation) + " failed: " + status.get_message());
  }
}

fe::DataType_t cudnn_data_type(flagdnnDataType_t data_type) {
  switch (data_type) {
    case FLAGDNN_DATA_FLOAT32:
      return fe::DataType_t::FLOAT;
    case FLAGDNN_DATA_FLOAT16:
      return fe::DataType_t::HALF;
    case FLAGDNN_DATA_BFLOAT16:
      return fe::DataType_t::BFLOAT16;
    case FLAGDNN_DATA_FP8_E4M3:
    case FLAGDNN_DATA_FP8_E5M2:
      break;
    case FLAGDNN_DATA_BOOLEAN:
      break;
  }
  throw std::invalid_argument("cuDNN Add requires a floating data type");
}

std::int64_t storage_element_count(const TestTensor& tensor) {
  std::int64_t result = 1;
  for (std::size_t axis = 0; axis < tensor.dimensions.size(); ++axis) {
    result += (tensor.dimensions[axis] - 1) * tensor.strides[axis];
  }
  return result;
}

TestTensor canonicalize_tensor(const TestTensor& tensor,
                               std::size_t logical_rank) {
  if (logical_rank == 0 || logical_rank > 4 ||
      tensor.dimensions.size() > logical_rank ||
      tensor.dimensions.size() != tensor.strides.size()) {
    throw std::invalid_argument(
        "cuDNN Add reference supports logical ranks one through four");
  }

  TestTensor aligned = tensor;
  const std::int64_t storage_span = storage_element_count(tensor);
  const std::size_t leading = logical_rank - tensor.dimensions.size();
  aligned.dimensions.insert(aligned.dimensions.begin(), leading, 1);
  aligned.strides.insert(aligned.strides.begin(), leading, storage_span);

  if (logical_rank == 4) {
    return aligned;
  }

  TestTensor result = aligned;
  if (logical_rank == 1) {
    result.dimensions = {1, aligned.dimensions[0], 1, 1};
    result.strides = {storage_span,
                      aligned.strides[0],
                      storage_span,
                      storage_span};
  } else if (logical_rank == 2) {
    result.dimensions = {
        aligned.dimensions[0], aligned.dimensions[1], 1, 1};
    result.strides = {aligned.strides[0],
                      aligned.strides[1],
                      aligned.strides[0],
                      aligned.strides[0]};
  } else {
    result.dimensions = {aligned.dimensions[0],
                         aligned.dimensions[2],
                         aligned.dimensions[1],
                         1};
    result.strides = {aligned.strides[0],
                      aligned.strides[2],
                      aligned.strides[1],
                      aligned.strides[1]};
  }
  return result;
}

std::shared_ptr<fe::graph::Tensor_attributes> make_tensor(
    const std::shared_ptr<fe::graph::Graph>& graph,
    const TestTensor& tensor,
    std::string name) {
  return graph->tensor(
      fe::graph::Tensor_attributes()
          .set_name(std::move(name))
          .set_uid(tensor.uid)
          .set_data_type(cudnn_data_type(tensor.data_type))
          .set_dim(tensor.dimensions)
          .set_stride(tensor.strides));
}

class CudnnHandle {
 public:
  CudnnHandle() { check_cudnn(cudnnCreate(&value_), "cudnnCreate"); }

  ~CudnnHandle() {
    if (value_ != nullptr) {
      (void)cudnnDestroy(value_);
    }
  }

  CudnnHandle(const CudnnHandle&) = delete;
  CudnnHandle& operator=(const CudnnHandle&) = delete;

  [[nodiscard]] cudnnHandle_t get() const noexcept { return value_; }

 private:
  cudnnHandle_t value_ = nullptr;
};

class DeviceScalar {
 public:
  DeviceScalar(flagdnnDataType_t data_type, double value) {
    switch (data_type) {
      case FLAGDNN_DATA_FLOAT32:
        allocate_and_copy(static_cast<float>(value));
        return;
      case FLAGDNN_DATA_FLOAT16:
        allocate_and_copy(__float2half_rn(static_cast<float>(value)));
        return;
      case FLAGDNN_DATA_BFLOAT16:
        allocate_and_copy(__float2bfloat16_rn(static_cast<float>(value)));
        return;
      case FLAGDNN_DATA_FP8_E4M3:
      case FLAGDNN_DATA_FP8_E5M2:
        break;
      case FLAGDNN_DATA_BOOLEAN:
        break;
    }
    throw std::invalid_argument("cuDNN Add alpha type is invalid");
  }

  ~DeviceScalar() {
    if (pointer_ != nullptr) {
      (void)cudaFree(pointer_);
    }
  }

  DeviceScalar(const DeviceScalar&) = delete;
  DeviceScalar& operator=(const DeviceScalar&) = delete;

  [[nodiscard]] void* get() const noexcept { return pointer_; }

 private:
  template <typename Value>
  void allocate_and_copy(const Value& value) {
    check_cuda(cudaMalloc(&pointer_, sizeof(Value)), "cudaMalloc(alpha)");
    try {
      check_cuda(cudaMemcpy(pointer_,
                            &value,
                            sizeof(Value),
                            cudaMemcpyHostToDevice),
                 "cudaMemcpy(alpha)");
    } catch (...) {
      (void)cudaFree(pointer_);
      pointer_ = nullptr;
      throw;
    }
  }

  void* pointer_ = nullptr;
};

class CudnnAddExecutable final : public AddExecutable {
 public:
  explicit CudnnAddExecutable(const AddTestCase& test_case)
      : graph_(std::make_shared<fe::graph::Graph>()) {
    validate_add_case(test_case);
    const std::size_t logical_rank = test_case.output.dimensions.size();
    const TestTensor left_spec =
        canonicalize_tensor(test_case.left, logical_rank);
    const TestTensor right_spec =
        canonicalize_tensor(test_case.right, logical_rank);
    const TestTensor output_spec =
        canonicalize_tensor(test_case.output, logical_rank);

    graph_->set_name(test_case.name + "::cudnn")
        .set_io_data_type(cudnn_data_type(test_case.left.data_type))
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    const auto left = make_tensor(graph_, left_spec, "left");
    const auto right = make_tensor(graph_, right_spec, "right");

    std::shared_ptr<fe::graph::Tensor_attributes> right_operand = right;
    if (test_case.alpha != 1.0) {
      alpha_uid_ = std::numeric_limits<std::int64_t>::max();
      alpha_ = std::make_unique<DeviceScalar>(
          test_case.right.data_type, test_case.alpha);
      TestTensor alpha_spec;
      alpha_spec.uid = alpha_uid_;
      alpha_spec.data_type = right_spec.data_type;
      alpha_spec.dimensions = {1, 1, 1, 1};
      alpha_spec.strides = {1, 1, 1, 1};
      const auto alpha = make_tensor(graph_, alpha_spec, "alpha");

      auto scaled_right = graph_->pointwise(
          right,
          alpha,
          fe::graph::Pointwise_attributes()
              .set_name("scale_right")
              .set_mode(fe::PointwiseMode_t::MUL)
              .set_compute_data_type(fe::DataType_t::FLOAT));
      scaled_right->set_name("scaled_right")
          .set_data_type(cudnn_data_type(right_spec.data_type))
          .set_dim(right_spec.dimensions)
          .set_stride(right_spec.strides);
      right_operand = std::move(scaled_right);
    }

    auto output = graph_->pointwise(
        left,
        right_operand,
        fe::graph::Pointwise_attributes()
            .set_name("add")
            .set_mode(fe::PointwiseMode_t::ADD)
            .set_compute_data_type(fe::DataType_t::FLOAT));
    output->set_name("output")
        .set_uid(output_spec.uid)
        .set_data_type(cudnn_data_type(output_spec.data_type))
        .set_dim(output_spec.dimensions)
        .set_stride(output_spec.strides)
        .set_output(true);

    check_frontend(graph_->build(handle_.get(), {fe::HeurMode_t::A}),
                   "cuDNN Add graph build");
    std::int64_t workspace_size = 0;
    check_frontend(graph_->get_workspace_size(workspace_size),
                   "cuDNN Add workspace query");
    if (workspace_size < 0) {
      throw std::runtime_error("cuDNN returned a negative workspace size");
    }
    workspace_size_ = static_cast<std::size_t>(workspace_size);
  }

  [[nodiscard]] std::size_t workspace_size() const noexcept override {
    return workspace_size_;
  }

  void execute(std::span<const flagdnnBinding_t> bindings,
               void* workspace,
               std::size_t workspace_size,
               flagdnnStream_t stream) override {
    if (workspace_size < workspace_size_ ||
        (workspace_size_ != 0 && workspace == nullptr)) {
      throw std::invalid_argument("cuDNN Add workspace is too small");
    }
    const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    if (!stream_initialized_ || stream_ != cuda_stream) {
      check_cudnn(cudnnSetStream(handle_.get(), cuda_stream),
                  "cudnnSetStream");
      stream_ = cuda_stream;
      stream_initialized_ = true;
    }

    std::unordered_map<std::int64_t, void*> pointers;
    pointers.reserve(bindings.size() + 1);
    for (const flagdnnBinding_t& binding : bindings) {
      if (binding.uid <= 0 || binding.device_pointer == nullptr ||
          !pointers.emplace(binding.uid, binding.device_pointer).second) {
        throw std::invalid_argument("cuDNN Add binding is invalid");
      }
    }
    if (alpha_ != nullptr &&
        !pointers.emplace(alpha_uid_, alpha_->get()).second) {
      throw std::invalid_argument("cuDNN Add alpha UID collides with a binding");
    }
    check_frontend(graph_->execute(handle_.get(), pointers, workspace),
                   "cuDNN Add graph execute");
  }

 private:
  CudnnHandle handle_;
  std::unique_ptr<DeviceScalar> alpha_;
  std::shared_ptr<fe::graph::Graph> graph_;
  std::int64_t alpha_uid_ = 0;
  std::size_t workspace_size_ = 0;
  cudaStream_t stream_ = nullptr;
  bool stream_initialized_ = false;
};

}  // namespace

std::unique_ptr<AddExecutable> build_add_reference(
    const AddTestCase& test_case) {
  return std::make_unique<CudnnAddExecutable>(test_case);
}

}  // namespace flagdnn::testing
