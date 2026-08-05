/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_NVIDIA_VALIDATION_FUNCTIONAL_CUDNN_GRAPH_HPP_
#define FLAGDNN_BACKENDS_NVIDIA_VALIDATION_FUNCTIONAL_CUDNN_GRAPH_HPP_

#include "common/common.hpp"

#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <cudnn_frontend.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string_view>
#include <unordered_map>

namespace flagdnn::testing::cuda {

namespace cfe = cudnn_frontend;
using CudnnBindingMap = std::unordered_map<std::int64_t, void*>;

void check_cuda_runtime(cudaError_t status, std::string_view operation);
void check_cudnn(cudnnStatus_t status, std::string_view operation);
void check_cudnn_frontend(cfe::error_t status, std::string_view operation);

[[nodiscard]] cfe::DataType_t cudnn_frontend_data_type(
    flagdnnDataType_t data_type);
[[nodiscard]] TestTensor padded_to_rank_four(const TestTensor& tensor,
                                             std::size_t logical_rank);
[[nodiscard]] TestTensor flatten_compact_tensor(const TestTensor& tensor);
[[nodiscard]] TestTensor canonicalize_pointwise_tensor(
    const TestTensor& tensor,
    std::size_t logical_rank);
[[nodiscard]] bool has_same_physical_mapping(const TestTensor& left,
                                             const TestTensor& right);

[[nodiscard]] std::shared_ptr<cfe::graph::Tensor_attributes> make_cudnn_tensor(
    const std::shared_ptr<cfe::graph::Graph>& graph,
    const TestTensor& tensor,
    std::string_view name);
[[nodiscard]] CudnnBindingMap make_cudnn_binding_map(
    std::span<const flagdnnBinding_t> bindings);

class CudnnGraphExecutable : public TestExecutable {
 public:
  CudnnGraphExecutable();
  ~CudnnGraphExecutable() override;

  CudnnGraphExecutable(const CudnnGraphExecutable&) = delete;
  CudnnGraphExecutable& operator=(const CudnnGraphExecutable&) = delete;

  [[nodiscard]] std::size_t workspace_size() const noexcept final;

 protected:
  [[nodiscard]] cudnnHandle_t handle() const noexcept;
  void begin_execute(void* workspace,
                     std::size_t workspace_size,
                     flagdnnStream_t stream);
  void set_workspace_size(std::int64_t workspace_size);

 private:
  cudnnHandle_t handle_ = nullptr;
  std::size_t workspace_size_ = 0;
  cudaStream_t stream_ = nullptr;
  bool stream_initialized_ = false;
};

}  // namespace flagdnn::testing::cuda

#endif  // FLAGDNN_BACKENDS_NVIDIA_VALIDATION_FUNCTIONAL_CUDNN_GRAPH_HPP_
