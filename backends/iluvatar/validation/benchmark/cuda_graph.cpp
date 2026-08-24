// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "benchmark/cuda_graph.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <limits>
#include <string>
#include <utility>

namespace flagdnn::iluvatar::validation::benchmark {
namespace {

[[noreturn]] void throw_cuda(cudaError_t status, const char *operation) {
  const char *detail = cudaGetErrorString(status);
  throw std::runtime_error(
      std::string(operation == nullptr ? "CoreX CUDA runtime operation"
                                       : operation) +
      " failed: " + (detail == nullptr ? "unknown CUDA error" : detail) +
      " (status=" + std::to_string(static_cast<int>(status)) + ")");
}

void require_capture_restored(cudaStream_t stream, const char *operation) {
  cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusActive;
  check_cuda_runtime(cudaStreamIsCapturing(stream, &capture_status), operation);
  if (capture_status != cudaStreamCaptureStatusNone) {
    throw std::runtime_error(std::string(operation) +
                             " left the stream in capture state");
  }
}

void destroy_graph(cudaGraph_t graph) noexcept {
  if (graph != nullptr) {
    (void)cudaGraphDestroy(graph);
  }
}

} // namespace

void check_cuda_runtime(cudaError_t status, const char *operation) {
  if (status != cudaSuccess) {
    throw_cuda(status, operation);
  }
}

bool cuda_capture_status_is_capability(cudaError_t status) noexcept {
  return status == cudaErrorStreamCaptureUnsupported;
}

bool cuda_capture_status_is_fatal(cudaError_t status) noexcept {
  return status != cudaSuccess && !cuda_capture_status_is_capability(status);
}

CudaStream::CudaStream() {
  check_cuda_runtime(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
                     "cudaStreamCreateWithFlags");
}

CudaStream::~CudaStream() noexcept {
  if (stream_ != nullptr) {
    (void)cudaStreamDestroy(stream_);
  }
}

void CudaStream::synchronize() const {
  check_cuda_runtime(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
}

CudaDeviceBuffer::CudaDeviceBuffer(std::size_t bytes)
    : bytes_(std::max<std::size_t>(bytes, 1)) {
  check_cuda_runtime(cudaMalloc(&pointer_, bytes_), "cudaMalloc");
}

CudaDeviceBuffer::~CudaDeviceBuffer() noexcept {
  if (pointer_ != nullptr) {
    (void)cudaFree(pointer_);
  }
}

void *CudaDeviceBuffer::at(std::size_t byte_offset) const {
  if (byte_offset >= bytes_) {
    throw std::invalid_argument("CUDA device buffer offset is out of range");
  }
  return static_cast<void *>(static_cast<std::byte *>(pointer_) + byte_offset);
}

void CudaDeviceBuffer::copy_from_host(const void *source, std::size_t bytes,
                                      std::size_t byte_offset,
                                      cudaStream_t stream) const {
  if (source == nullptr || byte_offset > bytes_ ||
      bytes > bytes_ - byte_offset) {
    throw std::invalid_argument("CUDA host-to-device copy is out of range");
  }
  check_cuda_runtime(cudaMemcpyAsync(at(byte_offset), source, bytes,
                                     cudaMemcpyHostToDevice, stream),
                     "cudaMemcpyAsync(host-to-device)");
}

void CudaDeviceBuffer::copy_to_host(void *destination, std::size_t bytes,
                                    std::size_t byte_offset,
                                    cudaStream_t stream) const {
  if (destination == nullptr || byte_offset > bytes_ ||
      bytes > bytes_ - byte_offset) {
    throw std::invalid_argument("CUDA device-to-host copy is out of range");
  }
  check_cuda_runtime(cudaMemcpyAsync(destination, at(byte_offset), bytes,
                                     cudaMemcpyDeviceToHost, stream),
                     "cudaMemcpyAsync(device-to-host)");
}

CapturedExecutionBatch::CapturedExecutionBatch(
    cudaStream_t stream, int execution_count,
    const std::function<void()> &submit)
    : execution_count_(execution_count) {
  if (stream == nullptr || execution_count <= 0 || !submit) {
    throw std::invalid_argument("CUDA Graph capture request is invalid");
  }

  cudaError_t status =
      cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
  if (cuda_capture_status_is_capability(status)) {
    throw StreamCaptureUnsupported("cudaStreamBeginCapture is unsupported");
  }
  check_cuda_runtime(status, "cudaStreamBeginCapture");

  try {
    for (int iteration = 0; iteration < execution_count_; ++iteration) {
      submit();
    }
  } catch (...) {
    std::exception_ptr original = std::current_exception();
    cudaGraph_t abandoned = nullptr;
    const cudaError_t end_status = cudaStreamEndCapture(stream, &abandoned);
    destroy_graph(abandoned);
    require_capture_restored(stream,
                             "cudaStreamIsCapturing(after callback failure)");
    if (end_status != cudaSuccess) {
      throw_cuda(end_status, "cudaStreamEndCapture(after callback failure)");
    }
    std::rethrow_exception(original);
  }

  status = cudaStreamEndCapture(stream, &graph_);
  if (cuda_capture_status_is_capability(status)) {
    destroy_graph(std::exchange(graph_, nullptr));
    require_capture_restored(
        stream, "cudaStreamIsCapturing(after unsupported capture)");
    throw StreamCaptureUnsupported("cudaStreamEndCapture is unsupported");
  }
  if (status != cudaSuccess) {
    destroy_graph(std::exchange(graph_, nullptr));
    require_capture_restored(stream,
                             "cudaStreamIsCapturing(after failed capture)");
    throw_cuda(status, "cudaStreamEndCapture");
  }
  require_capture_restored(stream, "cudaStreamIsCapturing(after capture)");
  if (graph_ == nullptr) {
    throw std::runtime_error("CUDA Graph capture returned a null graph");
  }

  status = cudaGraphInstantiate(&executable_, graph_, nullptr, nullptr, 0);
  if (cuda_capture_status_is_capability(status)) {
    throw StreamCaptureUnsupported("cudaGraphInstantiate is unsupported");
  }
  check_cuda_runtime(status, "cudaGraphInstantiate");
}

CapturedExecutionBatch::~CapturedExecutionBatch() noexcept {
  if (executable_ != nullptr) {
    (void)cudaGraphExecDestroy(executable_);
  }
  destroy_graph(graph_);
}

void CapturedExecutionBatch::launch(cudaStream_t stream) const {
  if (executable_ == nullptr || stream == nullptr) {
    throw std::logic_error("CUDA Graph executable is unavailable");
  }
  const cudaError_t status = cudaGraphLaunch(executable_, stream);
  if (cuda_capture_status_is_capability(status)) {
    throw StreamCaptureUnsupported("cudaGraphLaunch is unsupported");
  }
  check_cuda_runtime(status, "cudaGraphLaunch");
}

CudaEventTimer::CudaEventTimer() {
  check_cuda_runtime(cudaEventCreate(&start_), "cudaEventCreate(start)");
  try {
    check_cuda_runtime(cudaEventCreate(&stop_), "cudaEventCreate(stop)");
  } catch (...) {
    (void)cudaEventDestroy(start_);
    start_ = nullptr;
    throw;
  }
}

CudaEventTimer::~CudaEventTimer() noexcept {
  if (stop_ != nullptr) {
    (void)cudaEventDestroy(stop_);
  }
  if (start_ != nullptr) {
    (void)cudaEventDestroy(start_);
  }
}

double CudaEventTimer::measure_microseconds_per_execution(
    cudaStream_t stream, const CapturedExecutionBatch &batch) const {
  check_cuda_runtime(cudaEventRecord(start_, stream), "cudaEventRecord(start)");
  batch.launch(stream);
  check_cuda_runtime(cudaEventRecord(stop_, stream), "cudaEventRecord(stop)");
  check_cuda_runtime(cudaEventSynchronize(stop_), "cudaEventSynchronize(stop)");
  float elapsed_milliseconds = 0.0F;
  check_cuda_runtime(cudaEventElapsedTime(&elapsed_milliseconds, start_, stop_),
                     "cudaEventElapsedTime");
  const double result = static_cast<double>(elapsed_milliseconds) * 1000.0 /
                        static_cast<double>(batch.execution_count());
  if (!std::isfinite(result) || result <= 0.0) {
    throw std::runtime_error("CoreX event timing is not positive and finite");
  }
  return result;
}

double percentile(std::span<const double> samples, double fraction) {
  if (samples.empty() || !std::isfinite(fraction) || fraction < 0.0 ||
      fraction > 1.0) {
    throw std::invalid_argument("benchmark percentile request is invalid");
  }
  std::vector<double> sorted(samples.begin(), samples.end());
  std::sort(sorted.begin(), sorted.end());
  const std::size_t rank = std::max<std::size_t>(
      1, static_cast<std::size_t>(
             std::ceil(fraction * static_cast<double>(sorted.size()))));
  return sorted[rank - 1];
}

void require_positive_finite_samples(std::span<const double> samples,
                                     std::size_t expected_count,
                                     std::string_view provider) {
  if (samples.size() != expected_count || expected_count == 0) {
    throw std::runtime_error(std::string(provider) +
                             " benchmark sample count mismatch");
  }
  for (const double sample : samples) {
    if (!std::isfinite(sample) || sample <= 0.0) {
      throw std::runtime_error(std::string(provider) +
                               " benchmark sample is not positive and finite");
    }
  }
}

} // namespace flagdnn::iluvatar::validation::benchmark
