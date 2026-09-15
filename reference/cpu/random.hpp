/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#ifndef FLAGDNN_REFERENCE_CPU_RANDOM_HPP_
#define FLAGDNN_REFERENCE_CPU_RANDOM_HPP_
#include <cstdint>
#include <vector>
namespace flagdnn::reference::cpu {
struct RngParameters {
  std::vector<std::int64_t> shape;
  std::int64_t seed = 0, offset = 0;
  int distribution = 1, uniform_bits = 24;
  double probability = 0.5;
};
std::vector<std::vector<float>> evaluate_rng(const RngParameters& parameters);
}  // namespace flagdnn::reference::cpu
#endif
