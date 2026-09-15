/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "reference/cpu/random.hpp"

#include <array>
#include <cmath>
#include <numeric>
#include <stdexcept>
namespace flagdnn::reference::cpu {
namespace {
// Philox4x32-10: scalar reference from the counter/key round definition.
std::array<std::uint32_t, 4> philox(std::uint64_t seed, std::uint64_t offset) {
  std::array<std::uint32_t, 4> c = {static_cast<std::uint32_t>(offset),
                                    static_cast<std::uint32_t>(offset >> 32), 0,
                                    0};
  std::uint32_t k0 = static_cast<std::uint32_t>(seed),
                k1 = static_cast<std::uint32_t>(seed >> 32);
  for (int round = 0; round < 10; ++round) {
    const std::uint64_t a = 0xD2511F53ULL * c[0], b = 0xCD9E8D57ULL * c[2];
    c = {static_cast<std::uint32_t>(b >> 32) ^ c[1] ^ k0,
         static_cast<std::uint32_t>(b),
         static_cast<std::uint32_t>(a >> 32) ^ c[3] ^ k1,
         static_cast<std::uint32_t>(a)};
    k0 += 0x9E3779B9U;
    k1 += 0xBB67AE85U;
  }
  return c;
}
}  // namespace
std::vector<std::vector<float>> evaluate_rng(const RngParameters& parameters) {
  const auto count =
      std::accumulate(parameters.shape.begin(), parameters.shape.end(),
                      std::size_t{1}, std::multiplies<>());
  std::vector<float> values(count);
  const auto bits = parameters.uniform_bits;
  for (std::size_t index = 0; index < count; ++index) {
    const auto r =
        philox(static_cast<std::uint64_t>(parameters.seed),
               static_cast<std::uint64_t>(parameters.offset) + index);
    if (parameters.distribution == 1)
      values[index] =
          static_cast<float>(r[0] >> (32 - bits)) * std::ldexp(1.0F, -bits);
    else if (parameters.distribution == 2) {
      const double u1 = (static_cast<double>(r[0] >> 8) + 1.0) *
                        std::ldexp(1.0, -24),
                   u2 = static_cast<double>(r[1] >> 8) * std::ldexp(1.0, -24);
      values[index] = static_cast<float>(std::sqrt(-2.0 * std::log(u1)) *
                                         std::cos(6.283185307179586 * u2));
    } else
      values[index] = static_cast<double>(r[0] >> 8) * std::ldexp(1.0, -24) <
                              parameters.probability
                          ? 1.0F
                          : 0.0F;
  }
  if (count >= 4096) {
    double sum = 0, square = 0;
    for (auto value : values) {
      sum += value;
      square += static_cast<double>(value) * value;
    }
    const double mean = sum / count, variance = square / count - mean * mean;
    const double expected_mean = parameters.distribution == 1 ? 0.5
                                 : parameters.distribution == 2
                                     ? 0.0
                                     : parameters.probability;
    const double expected_variance =
        parameters.distribution == 1 ? 1.0 / 12.0
        : parameters.distribution == 2
            ? 1.0
            : parameters.probability * (1.0 - parameters.probability);
    if (std::abs(mean - expected_mean) > 0.07 ||
        std::abs(variance - expected_variance) > 0.09)
      throw std::runtime_error(
          "Philox stream failed distribution mean/variance check");
  }
  return {values};
}
}  // namespace flagdnn::reference::cpu
