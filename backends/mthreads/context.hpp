/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_MTHREADS_CONTEXT_HPP_
#define FLAGDNN_BACKENDS_MTHREADS_CONTEXT_HPP_

#include <musa.h>

#include <cstdint>
#include <string>

namespace flagdnn::mthreads {

struct EngineBuildContext {
  MUdevice device = 0;
  MUcontext context = nullptr;
  std::string target_fingerprint;
  std::string device_identity;
};

class ContextGuard final {
 public:
  ContextGuard(MUcontext context, MUdevice device);
  ~ContextGuard();

  ContextGuard(const ContextGuard&) = delete;
  ContextGuard& operator=(const ContextGuard&) = delete;

 private:
  MUcontext requested_ = nullptr;
  MUcontext saved_ = nullptr;
  bool pushed_ = false;
};

class MthreadsContext final {
 public:
  explicit MthreadsContext(std::int32_t device_ordinal);
  ~MthreadsContext();

  MthreadsContext(const MthreadsContext&) = delete;
  MthreadsContext& operator=(const MthreadsContext&) = delete;

  [[nodiscard]] const std::string& target_fingerprint() const noexcept;
  [[nodiscard]] EngineBuildContext engine_build_context() const;

 private:
  MUdevice device_ = 0;
  MUcontext context_ = nullptr;
  std::string target_fingerprint_;
  std::string device_identity_;
};

}  // namespace flagdnn::mthreads

#endif  // FLAGDNN_BACKENDS_MTHREADS_CONTEXT_HPP_
