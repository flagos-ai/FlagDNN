/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_ILUVATAR_ENGINES_LIBTRITON_JIT_HPP_
#define FLAGDNN_BACKENDS_ILUVATAR_ENGINES_LIBTRITON_JIT_HPP_

#include "backends/backend_api.h"
#include "backends/iluvatar/context.hpp"

#include <cuda.h>

#include <cstddef>
#include <memory>

namespace flagdnn::iluvatar {

class IluvatarExecutable {
public:
  virtual ~IluvatarExecutable() = default;

  [[nodiscard]] virtual std::size_t workspace_size() const noexcept = 0;

  virtual void execute(CUstream stream, const flagdnnBackendBindingV2 *bindings,
                       std::size_t binding_count, void *workspace,
                       std::size_t workspace_size) = 0;
};

[[nodiscard]] std::unique_ptr<IluvatarExecutable>
create_libtriton_jit_executable(const EngineBuildContext &context,
                                const flagdnnBackendBuildInputV2 &input);

} // namespace flagdnn::iluvatar

#endif // FLAGDNN_BACKENDS_ILUVATAR_ENGINES_LIBTRITON_JIT_HPP_
