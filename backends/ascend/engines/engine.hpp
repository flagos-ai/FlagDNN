/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_ASCEND_ENGINES_ENGINE_HPP_
#define FLAGDNN_BACKENDS_ASCEND_ENGINES_ENGINE_HPP_

#include "backends/ascend/context.hpp"
#include "backends/backend_api.h"

#include <cstddef>
#include <memory>

namespace flagdnn::ascend {

// Ascend has one execution path. Keep JIT/Python implementation details behind
// the same concrete engine interface used by NVIDIA.
class ExecutionEngine final {
public:
  ExecutionEngine(const EngineBuildContext &context,
                  const flagdnnBackendBuildInputV2 &input);
  ~ExecutionEngine();
  ExecutionEngine(const ExecutionEngine &) = delete;
  ExecutionEngine &operator=(const ExecutionEngine &) = delete;

  [[nodiscard]] std::size_t workspace_size() const noexcept;
  void execute(void *native_stream, const flagdnnBackendBindingV2 bindings[],
               std::size_t binding_count, void *workspace,
               std::size_t workspace_size) const;

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

[[nodiscard]] std::unique_ptr<ExecutionEngine>
create_execution_engine(const EngineBuildContext &context,
                        const flagdnnBackendBuildInputV2 &input);

} // namespace flagdnn::ascend

#endif // FLAGDNN_BACKENDS_ASCEND_ENGINES_ENGINE_HPP_
