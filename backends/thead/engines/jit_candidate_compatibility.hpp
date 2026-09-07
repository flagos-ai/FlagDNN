// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef FLAGDNN_BACKENDS_THEAD_ENGINES_JIT_CANDIDATE_COMPATIBILITY_HPP_
#define FLAGDNN_BACKENDS_THEAD_ENGINES_JIT_CANDIDATE_COMPATIBILITY_HPP_

#include "backends/thead/artifact.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace flagdnn::thead {

// Only this exception is eligible for per-candidate filtering. Artifact,
// ABI, JIT compilation, and runtime failures use their original exception
// types and therefore fail the complete Graph build.
class CandidateCompatibilityError final : public std::runtime_error {
 public:
  explicit CandidateCompatibilityError(std::string message)
      : std::runtime_error(std::move(message)) {}
};

struct CandidateDeviceLimits {
  unsigned int maximum_threads_per_block = 0;
  std::array<unsigned int, 3> maximum_block = {0, 0, 0};
  std::array<unsigned int, 3> maximum_grid = {0, 0, 0};
  unsigned int maximum_shared_memory = 0;
};

inline void validate_candidate_resources(
    const KernelVariant& variant,
    const CandidateDeviceLimits& limits) {
  std::size_t threads = 1;
  for (std::size_t dimension = 0; dimension < 3; ++dimension) {
    if (variant.block[dimension] > limits.maximum_block[dimension] ||
        variant.grid[dimension] > limits.maximum_grid[dimension]) {
      throw CandidateCompatibilityError(
          "THead autotune candidate exceeds a PPU launch-dimension limit");
    }
    if (variant.block[dimension] == 0 ||
        threads > std::numeric_limits<std::size_t>::max() /
                      variant.block[dimension]) {
      throw CandidateCompatibilityError(
          "THead autotune candidate block size is invalid");
    }
    threads *= variant.block[dimension];
  }
  if (threads > limits.maximum_threads_per_block) {
    throw CandidateCompatibilityError(
        "THead autotune candidate exceeds the PPU threads-per-block limit");
  }
  if (variant.shared_memory > limits.maximum_shared_memory) {
    throw CandidateCompatibilityError(
        "THead autotune candidate exceeds the PPU shared-memory limit");
  }
}

}  // namespace flagdnn::thead

#endif  // FLAGDNN_BACKENDS_THEAD_ENGINES_JIT_CANDIDATE_COMPATIBILITY_HPP_
