/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BENCHMARK_COMMON_PLATFORM_PROVIDER_SKELETON_HPP_
#define FLAGDNN_BENCHMARK_COMMON_PLATFORM_PROVIDER_SKELETON_HPP_

#include "benchmark_provider.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace flagdnn::benchmarking {

/*
 * Bring-up helper for a new native reference provider. It makes every missing
 * capability explicit while the platform runner and vendor adapter are being
 * implemented. It must not be used as a release correctness provider.
 */
class PlatformProviderSkeleton final : public BenchmarkProvider {
 public:
  PlatformProviderSkeleton(std::string name, std::string reason)
      : name_(std::move(name)), reason_(std::move(reason)) {
    if (name_.empty() || reason_.empty()) {
      throw std::invalid_argument(
          "platform provider skeleton name and reason must be nonempty");
    }
  }

  [[nodiscard]] std::string_view name() const noexcept override {
    return name_;
  }

  [[nodiscard]] ProviderCapability capability(
      const BenchmarkCase&) const override {
    return ProviderCapability::unsupported(reason_);
  }

  [[nodiscard]] std::unique_ptr<BenchmarkExecutable> build(
      const BenchmarkCase&) override {
    throw std::logic_error(
        name_ + " provider build called for an unsupported capability: " +
        reason_);
  }

 private:
  std::string name_;
  std::string reason_;
};

}  // namespace flagdnn::benchmarking

#endif  // FLAGDNN_BENCHMARK_COMMON_PLATFORM_PROVIDER_SKELETON_HPP_
