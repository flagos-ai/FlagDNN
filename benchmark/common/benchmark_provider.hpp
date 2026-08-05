/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BENCHMARK_COMMON_BENCHMARK_PROVIDER_HPP_
#define FLAGDNN_BENCHMARK_COMMON_BENCHMARK_PROVIDER_HPP_

#include "case.hpp"

#include <cstddef>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace flagdnn::benchmarking {

struct ProviderCapability {
  bool supported = true;
  std::string reason;

  [[nodiscard]] static ProviderCapability unsupported(std::string why) {
    return {false, std::move(why)};
  }
};

class BenchmarkUnsupportedError final : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

class BenchmarkExecutable {
 public:
  virtual ~BenchmarkExecutable() = default;

  [[nodiscard]] virtual std::size_t workspace_size() const noexcept = 0;
  virtual void execute(std::span<const flagdnnBinding_t> bindings,
                       void* workspace,
                       std::size_t workspace_size,
                       flagdnnStream_t stream) = 0;
};

class BenchmarkProvider {
 public:
  virtual ~BenchmarkProvider() = default;

  [[nodiscard]] virtual std::string_view name() const noexcept = 0;
  [[nodiscard]] virtual ProviderCapability capability(
      const BenchmarkCase&) const {
    return {};
  }
  [[nodiscard]] virtual std::unique_ptr<BenchmarkExecutable> build(
      const BenchmarkCase& specification) = 0;
};

}  // namespace flagdnn::benchmarking

#endif  // FLAGDNN_BENCHMARK_COMMON_BENCHMARK_PROVIDER_HPP_
