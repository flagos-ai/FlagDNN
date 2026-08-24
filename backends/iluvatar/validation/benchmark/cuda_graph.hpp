// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef FLAGDNN_BACKENDS_ILUVATAR_VALIDATION_BENCHMARK_CUDA_GRAPH_HPP_
#define FLAGDNN_BACKENDS_ILUVATAR_VALIDATION_BENCHMARK_CUDA_GRAPH_HPP_

#include "benchmark/benchmark_phase.hpp"

#include <cuda_runtime_api.h>
#include <flagdnn/flagdnn.h>

#include <cstddef>
#include <functional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace flagdnn::iluvatar::validation::benchmark {

class StreamCaptureUnsupported final : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

void check_cuda_runtime(cudaError_t status, const char *operation);
[[nodiscard]] bool
cuda_capture_status_is_capability(cudaError_t status) noexcept;
[[nodiscard]] bool cuda_capture_status_is_fatal(cudaError_t status) noexcept;

class CudaStream final {
public:
  CudaStream();
  ~CudaStream() noexcept;
  CudaStream(const CudaStream &) = delete;
  CudaStream &operator=(const CudaStream &) = delete;

  [[nodiscard]] cudaStream_t get() const noexcept { return stream_; }
  [[nodiscard]] flagdnnStream_t opaque() const noexcept {
    return reinterpret_cast<flagdnnStream_t>(stream_);
  }
  void synchronize() const;

private:
  cudaStream_t stream_ = nullptr;
};

class CudaDeviceBuffer final {
public:
  explicit CudaDeviceBuffer(std::size_t bytes);
  ~CudaDeviceBuffer() noexcept;
  CudaDeviceBuffer(const CudaDeviceBuffer &) = delete;
  CudaDeviceBuffer &operator=(const CudaDeviceBuffer &) = delete;

  [[nodiscard]] void *at(std::size_t byte_offset = 0) const;
  [[nodiscard]] std::size_t size() const noexcept { return bytes_; }
  void copy_from_host(const void *source, std::size_t bytes,
                      std::size_t byte_offset, cudaStream_t stream) const;
  void copy_to_host(void *destination, std::size_t bytes,
                    std::size_t byte_offset, cudaStream_t stream) const;

private:
  void *pointer_ = nullptr;
  std::size_t bytes_ = 0;
};

class CapturedExecutionBatch final {
public:
  CapturedExecutionBatch(cudaStream_t stream, int execution_count,
                         const std::function<void()> &submit);
  ~CapturedExecutionBatch() noexcept;
  CapturedExecutionBatch(const CapturedExecutionBatch &) = delete;
  CapturedExecutionBatch &operator=(const CapturedExecutionBatch &) = delete;

  void launch(cudaStream_t stream) const;
  [[nodiscard]] int execution_count() const noexcept {
    return execution_count_;
  }

private:
  cudaGraph_t graph_ = nullptr;
  cudaGraphExec_t executable_ = nullptr;
  int execution_count_ = 0;
};

class CudaEventTimer final {
public:
  CudaEventTimer();
  ~CudaEventTimer() noexcept;
  CudaEventTimer(const CudaEventTimer &) = delete;
  CudaEventTimer &operator=(const CudaEventTimer &) = delete;

  [[nodiscard]] double
  measure_microseconds_per_execution(cudaStream_t stream,
                                     const CapturedExecutionBatch &batch) const;

private:
  cudaEvent_t start_ = nullptr;
  cudaEvent_t stop_ = nullptr;
};

[[nodiscard]] double percentile(std::span<const double> samples,
                                double fraction);
void require_positive_finite_samples(std::span<const double> samples,
                                     std::size_t expected_count,
                                     std::string_view provider);

} // namespace flagdnn::iluvatar::validation::benchmark

#endif // FLAGDNN_BACKENDS_ILUVATAR_VALIDATION_BENCHMARK_CUDA_GRAPH_HPP_
