/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_AUTOTUNE_POLICY_HPP_
#define FLAGDNN_BACKENDS_AUTOTUNE_POLICY_HPP_

#include <cstddef>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace flagdnn::backend::autotune {

// Backend-independent description of one tuning decision. Candidate binaries,
// streams, events, and device allocations deliberately remain in the platform
// adapter (for example backends/nvidia/autotune.cpp).
struct SelectionRequest {
  std::string candidate_identity;
  std::string device_identity;
  std::string measurement_identity;
  std::filesystem::path cache_path;
  std::vector<std::string> candidate_ids;
  unsigned int warmup_milliseconds = 0;
  unsigned int benchmark_milliseconds = 1;
};

struct SelectionResult {
  std::size_t candidate_index = 0;
  bool cache_hit = false;
  std::vector<float> median_milliseconds;
};

using WarmupCallback = std::function<void(
    std::size_t candidate_index, unsigned int iterations)>;
using MeasureCallback =
    std::function<float(
        std::size_t candidate_index, unsigned int iterations)>;

// Returns the cached candidate only when every identity and candidate-space
// check succeeds. Backends may use this before compiling/preparing the full
// tuning space. A selected candidate that later fails to prepare should be
// discarded and followed by a normal tuning pass.
[[nodiscard]] std::optional<std::size_t> find_cached_candidate(
    const SelectionRequest& request) noexcept;

void discard_cached_candidate(const SelectionRequest& request) noexcept;

// Owns the policy shared by every device backend: cache lookup, calibrated
// millisecond-budget warmup/measurement, median statistics, stable
// best-candidate selection, and atomic cache publication. MeasureCallback
// returns the average milliseconds per launch for the requested batch.
// Callbacks own all platform-specific work.
[[nodiscard]] SelectionResult select_best_candidate(
    const SelectionRequest& request,
    const WarmupCallback& warmup,
    const MeasureCallback& measure);

}  // namespace flagdnn::backend::autotune

#endif  // FLAGDNN_BACKENDS_AUTOTUNE_POLICY_HPP_
