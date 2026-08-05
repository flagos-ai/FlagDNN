/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_CUDA_CONTEXT_HPP_
#define FLAGDNN_BACKENDS_CUDA_CONTEXT_HPP_

#include <cuda.h>

#include <cstdint>
#include <string>

namespace flagdnn::cuda {

struct EngineBuildContext {
  CUdevice device = 0;
  CUcontext context = nullptr;
  std::string target_fingerprint;
  std::string device_identity;
};

class ContextGuard {
 public:
  explicit ContextGuard(CUcontext context);
  ~ContextGuard();

  ContextGuard(const ContextGuard&) = delete;
  ContextGuard& operator=(const ContextGuard&) = delete;

 private:
  bool active_ = false;
};

class CudaContext {
 public:
  explicit CudaContext(std::int32_t device_ordinal);
  ~CudaContext();

  CudaContext(const CudaContext&) = delete;
  CudaContext& operator=(const CudaContext&) = delete;

  [[nodiscard]] const std::string& target_fingerprint() const noexcept;
  [[nodiscard]] EngineBuildContext engine_build_context() const;

 private:
  CUdevice device_ = 0;
  CUcontext context_ = nullptr;
  std::int32_t architecture_ = 0;
  std::string target_fingerprint_;
  std::string device_identity_;
};

}  // namespace flagdnn::cuda

#endif  // FLAGDNN_BACKENDS_CUDA_CONTEXT_HPP_
