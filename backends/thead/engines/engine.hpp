// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef FLAGDNN_BACKENDS_THEAD_ENGINES_ENGINE_HPP_
#define FLAGDNN_BACKENDS_THEAD_ENGINES_ENGINE_HPP_

#include "backends/backend_api.h"
#include "backends/thead/artifact.hpp"
#include "backends/thead/context.hpp"

#include <cuda.h>

#include <cstddef>
#include <memory>

namespace flagdnn::thead {

class ExecutionEngine {
 public:
  virtual ~ExecutionEngine() = default;

  [[nodiscard]] virtual std::size_t workspace_size() const noexcept = 0;
  virtual void execute(CUstream stream,
                       const flagdnnBackendBindingV2 bindings[],
                       std::size_t binding_count,
                       void* workspace,
                       std::size_t workspace_size) const = 0;
};

[[nodiscard]] std::unique_ptr<ExecutionEngine> create_execution_engine(
    const EngineBuildContext& context,
    ExecutionProgramArtifact artifact);

[[nodiscard]] std::unique_ptr<ExecutionEngine>
create_libtriton_jit_engine(const EngineBuildContext& context,
                            ExecutionProgramArtifact artifact);

}  // namespace flagdnn::thead

#endif  // FLAGDNN_BACKENDS_THEAD_ENGINES_ENGINE_HPP_
