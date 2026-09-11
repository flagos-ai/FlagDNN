// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef FLAGDNN_BACKENDS_THEAD_CONTEXT_HPP_
#define FLAGDNN_BACKENDS_THEAD_CONTEXT_HPP_

#include <cuda.h>

#include <cstdint>
#include <string>

namespace flagdnn::thead {

struct EngineBuildContext {
  CUdevice device = 0;
  CUcontext context = nullptr;
  std::string target_fingerprint;
  std::string device_identity;
};

class ContextGuard final {
 public:
  explicit ContextGuard(CUcontext context);
  ~ContextGuard();

  ContextGuard(const ContextGuard&) = delete;
  ContextGuard& operator=(const ContextGuard&) = delete;

 private:
  bool pushed_ = false;
};

class TheadContext final {
 public:
  explicit TheadContext(std::int32_t device_ordinal);
  ~TheadContext();

  TheadContext(const TheadContext&) = delete;
  TheadContext& operator=(const TheadContext&) = delete;

  [[nodiscard]] const std::string& target_fingerprint() const noexcept;
  [[nodiscard]] EngineBuildContext engine_build_context() const;

 private:
  CUdevice device_ = 0;
  CUcontext context_ = nullptr;
  std::string target_fingerprint_;
  std::string device_identity_;
};

}  // namespace flagdnn::thead

#endif  // FLAGDNN_BACKENDS_THEAD_CONTEXT_HPP_
