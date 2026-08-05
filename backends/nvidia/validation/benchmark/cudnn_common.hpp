/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_NVIDIA_VALIDATION_BENCHMARK_CUDNN_COMMON_HPP_
#define FLAGDNN_BACKENDS_NVIDIA_VALIDATION_BENCHMARK_CUDNN_COMMON_HPP_

#include "common/benchmark_provider.hpp"

#include <cudnn.h>
#include <cudnn_frontend.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string_view>
#include <unordered_map>

namespace flagdnn::benchmarking::cudnn_detail {

namespace fe = cudnn_frontend;

using BindingMap = std::unordered_map<std::int64_t, void*>;

void check_cudnn(cudnnStatus_t status, const char* operation);
void check_frontend(fe::error_t status, const char* operation);
void build_frontend_layout_graph_or_unsupported(
    fe::graph::Graph& graph,
    cudnnHandle_t handle,
    std::string_view operation);

[[nodiscard]] fe::DataType_t data_type(flagdnnDataType_t type);
[[nodiscard]] cudnnDataType_t cudnn_data_type(flagdnnDataType_t type);
void set_tensor_descriptor(cudnnTensorDescriptor_t descriptor,
                           const TensorSpec& specification);

[[nodiscard]] TensorSpec padded_to_rank_four(const TensorSpec& input,
                                             std::size_t logical_rank);
[[nodiscard]] TensorSpec compact_unary_tensor(const TensorSpec& input,
                                              std::size_t logical_rank);
[[nodiscard]] TensorSpec padded_to_minimum_rank_four(
    const TensorSpec& input);
[[nodiscard]] TensorSpec batchnorm_inference_nhwc_tensor(
    const TensorSpec& input);
[[nodiscard]] std::shared_ptr<fe::graph::Tensor_attributes> make_tensor(
    const std::shared_ptr<fe::graph::Graph>& graph,
    const TensorSpec& specification,
    std::string_view name,
    bool output);

void require_tensor_count(const BenchmarkCase& specification,
                          std::size_t expected);
[[nodiscard]] BindingMap make_binding_map(
    std::span<const flagdnnBinding_t> bindings);
[[nodiscard]] void* pointer_for(const BindingMap& pointers,
                                std::int64_t uid,
                                std::string_view operation);

class ExecutableBase : public BenchmarkExecutable {
 public:
  ExecutableBase();
  ~ExecutableBase() override;

  ExecutableBase(const ExecutableBase&) = delete;
  ExecutableBase& operator=(const ExecutableBase&) = delete;

  [[nodiscard]] std::size_t workspace_size() const noexcept final;

 protected:
  [[nodiscard]] cudnnHandle_t handle() const noexcept;
  void begin_execute(void* workspace,
                     std::size_t workspace_size,
                     flagdnnStream_t stream);
  void set_workspace_size(std::int64_t workspace_size);
  void set_workspace_size(std::size_t workspace_size) noexcept;

 private:
  cudnnHandle_t handle_ = nullptr;
  std::size_t workspace_size_ = 0;
  cudaStream_t stream_ = nullptr;
  bool stream_initialized_ = false;
};

}  // namespace flagdnn::benchmarking::cudnn_detail

#endif  // FLAGDNN_BACKENDS_NVIDIA_VALIDATION_BENCHMARK_CUDNN_COMMON_HPP_
