/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_CUDA_AUTOTUNE_HPP_
#define FLAGDNN_BACKENDS_CUDA_AUTOTUNE_HPP_

#include "backends/nvidia/artifact.hpp"
#include "backends/nvidia/context.hpp"

#include <cstddef>

namespace flagdnn::cuda {

[[nodiscard]] std::size_t select_autotune_candidate(
    const EngineBuildContext& context,
    std::size_t workspace_size,
    const CudaStageArtifact& stage);

}  // namespace flagdnn::cuda

#endif  // FLAGDNN_BACKENDS_CUDA_AUTOTUNE_HPP_
