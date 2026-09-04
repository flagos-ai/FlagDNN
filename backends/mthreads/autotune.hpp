/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_MTHREADS_AUTOTUNE_HPP_
#define FLAGDNN_BACKENDS_MTHREADS_AUTOTUNE_HPP_

#include "backends/mthreads/artifact.hpp"
#include "backends/mthreads/context.hpp"

#include <cstddef>
#include <functional>
#include <string>

namespace flagdnn::mthreads {

struct AutotuneCallbacks {
  std::function<void(std::size_t candidate_index)> prepare;
  std::function<void(
      std::size_t candidate_index, unsigned int iterations)>
      warmup;
  std::function<float(
      std::size_t candidate_index, unsigned int iterations)>
      measure;
};

[[nodiscard]] std::size_t select_mthreads_autotune_candidate(
    const EngineBuildContext& context,
    const StageArtifact& stage,
    std::string measurement_identity,
    const AutotuneCallbacks& callbacks);

}  // namespace flagdnn::mthreads

#endif  // FLAGDNN_BACKENDS_MTHREADS_AUTOTUNE_HPP_
