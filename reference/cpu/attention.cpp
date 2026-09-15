/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "reference/cpu/attention.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
namespace flagdnn::reference::cpu {
std::int64_t logical_offset(const AttentionTensor& tensor, std::int64_t b,
                            std::int64_t h, std::int64_t s, std::int64_t d) {
  const auto& dimensions = tensor.dimensions;
  return (((b * dimensions[1] + h) * dimensions[2] + s) * dimensions[3]) + d;
}

AttentionForwardResult evaluate_attention_forward(
    const AttentionParameters& test_case, std::span<const float> q,
    std::span<const float> k, std::span<const float> v,
    std::span<const float> bias) {
  const std::int64_t batch = test_case.q.dimensions[0];
  const std::int64_t query_heads = test_case.q.dimensions[1];
  const std::int64_t key_heads = test_case.k.dimensions[1];
  const std::int64_t value_heads = test_case.v.dimensions[1];
  const std::int64_t sequence_q = test_case.q.dimensions[2];
  const std::int64_t sequence_kv = test_case.k.dimensions[2];
  const std::int64_t head_dimension = test_case.q.dimensions[3];
  const std::int64_t value_dimension = test_case.v.dimensions[3];
  const float scale = test_case.options.attention_scale.value_or(
      1.0F / std::sqrt(static_cast<float>(head_dimension)));
  const std::int64_t shift = test_case.options.diagonal_alignment ==
                                     AttentionDiagonalAlignment::kBottomRight
                                 ? sequence_kv - sequence_q
                                 : 0;
  const std::int64_t minimum_diagonal =
      test_case.options.diagonal_band_left_bound.has_value()
          ? 1 - *test_case.options.diagonal_band_left_bound + shift
          : std::numeric_limits<std::int32_t>::min();
  const std::int64_t maximum_diagonal =
      test_case.options.diagonal_band_right_bound.has_value()
          ? *test_case.options.diagonal_band_right_bound + shift
          : std::numeric_limits<std::int32_t>::max();
  AttentionForwardResult result;
  result.output.resize(static_cast<std::size_t>(batch * query_heads *
                                                sequence_q * value_dimension));
  result.stats.resize(
      static_cast<std::size_t>(batch * query_heads * sequence_q));
  std::vector<double> scores(static_cast<std::size_t>(sequence_kv));
  std::vector<double> probabilities(static_cast<std::size_t>(sequence_kv));

  for (std::int64_t b = 0; b < batch; ++b) {
    for (std::int64_t h = 0; h < query_heads; ++h) {
      const std::int64_t kh = h / (query_heads / key_heads);
      const std::int64_t vh = h / (query_heads / value_heads);
      for (std::int64_t m = 0; m < sequence_q; ++m) {
        double maximum = -std::numeric_limits<double>::infinity();
        for (std::int64_t n = 0; n < sequence_kv; ++n) {
          const std::int64_t diagonal = n - m;
          if (diagonal < minimum_diagonal || diagonal > maximum_diagonal) {
            scores[static_cast<std::size_t>(n)] =
                -std::numeric_limits<double>::infinity();
            continue;
          }
          double score = 0.0;
          for (std::int64_t d = 0; d < head_dimension; ++d) {
            score += static_cast<double>(
                         q[logical_offset(test_case.q, b, h, m, d)]) *
                     static_cast<double>(
                         k[logical_offset(test_case.k, b, kh, n, d)]);
          }
          score *= static_cast<double>(scale);
          if (test_case.bias.has_value()) {
            const AttentionTensor& bias_specification = *test_case.bias;
            const std::int64_t bias_batch =
                bias_specification.dimensions[0] == 1 ? 0 : b;
            const std::int64_t bias_head =
                bias_specification.dimensions[1] == 1 ? 0 : h;
            score += bias[logical_offset(bias_specification, bias_batch,
                                         bias_head, m, n)];
          }
          scores[static_cast<std::size_t>(n)] = score;
          maximum = std::max(maximum, score);
        }
        double denominator = 0.0;
        for (std::int64_t n = 0; n < sequence_kv; ++n) {
          const double probability =
              std::isfinite(scores[static_cast<std::size_t>(n)])
                  ? std::exp(scores[static_cast<std::size_t>(n)] - maximum)
                  : 0.0;
          probabilities[static_cast<std::size_t>(n)] = probability;
          denominator += probability;
        }
        if (!(denominator > 0.0) || !std::isfinite(denominator)) {
          throw std::runtime_error("host SDPA produced an empty attention row");
        }
        const std::size_t stats_index =
            static_cast<std::size_t>((b * query_heads + h) * sequence_q + m);
        result.stats[stats_index] =
            static_cast<float>(maximum + std::log(denominator));
        for (std::int64_t d = 0; d < value_dimension; ++d) {
          double output = 0.0;
          for (std::int64_t n = 0; n < sequence_kv; ++n) {
            output += probabilities[static_cast<std::size_t>(n)] / denominator *
                      static_cast<double>(
                          v[logical_offset(test_case.v, b, vh, n, d)]);
          }
          result.output[static_cast<std::size_t>(logical_offset(
              test_case.output, b, h, m, d))] = static_cast<float>(output);
        }
      }
    }
  }
  return result;
}

AttentionBackwardResult evaluate_attention_backward(
    const AttentionParameters& c, std::span<const float> q,
    std::span<const float> k, std::span<const float> v,
    std::span<const float> doutput, std::span<const float> bias,
    const AttentionForwardResult& primal) {
  std::vector<double> dq(q.size()), dk(k.size()), dv(v.size()),
      dbias(bias.size());
  const auto batch = c.q.dimensions[0];
  const auto heads = c.q.dimensions[1];
  const auto sq = c.q.dimensions[2];
  const auto sk = c.k.dimensions[2];
  const auto dimension = c.q.dimensions[3];
  const auto value_dimension = c.v.dimensions[3];
  const double scale = c.options.attention_scale.value_or(
      1.0F / std::sqrt(static_cast<float>(dimension)));
  const auto shift =
      c.options.diagonal_alignment == AttentionDiagonalAlignment::kBottomRight
          ? sk - sq
          : 0;
  const auto lower = c.options.diagonal_band_left_bound.has_value()
                         ? 1 - *c.options.diagonal_band_left_bound + shift
                         : std::numeric_limits<std::int64_t>::min();
  const auto upper = c.options.diagonal_band_right_bound.has_value()
                         ? *c.options.diagonal_band_right_bound + shift
                         : std::numeric_limits<std::int64_t>::max();
  std::vector<double> probabilities(sk), dp(sk);
  for (std::int64_t b = 0; b < batch; ++b) {
    for (std::int64_t h = 0; h < heads; ++h) {
      const auto kh = h / (heads / c.k.dimensions[1]);
      const auto vh = h / (heads / c.v.dimensions[1]);
      for (std::int64_t m = 0; m < sq; ++m) {
        double delta = 0;
        for (std::int64_t n = 0; n < sk; ++n) {
          probabilities[n] = dp[n] = 0;
          if (n - m < lower || n - m > upper) continue;
          double score = 0;
          for (std::int64_t d = 0; d < dimension; ++d) {
            score += static_cast<double>(q[logical_offset(c.q, b, h, m, d)]) *
                     k[logical_offset(c.k, b, kh, n, d)];
          }
          score *= scale;
          if (c.bias.has_value()) {
            score +=
                bias[logical_offset(*c.bias, c.bias->dimensions[0] == 1 ? 0 : b,
                                    c.bias->dimensions[1] == 1 ? 0 : h, m, n)];
          }
          probabilities[n] =
              std::exp(score - primal.stats[(b * heads + h) * sq + m]);
          for (std::int64_t d = 0; d < value_dimension; ++d) {
            dp[n] += static_cast<double>(
                         doutput[logical_offset(c.doutput, b, h, m, d)]) *
                     v[logical_offset(c.v, b, vh, n, d)];
          }
          delta += probabilities[n] * dp[n];
        }
        for (std::int64_t n = 0; n < sk; ++n) {
          const double ds = probabilities[n] * (dp[n] - delta);
          for (std::int64_t d = 0; d < dimension; ++d) {
            dq[logical_offset(c.q, b, h, m, d)] +=
                scale * ds * k[logical_offset(c.k, b, kh, n, d)];
            dk[logical_offset(c.k, b, kh, n, d)] +=
                scale * ds * q[logical_offset(c.q, b, h, m, d)];
          }
          for (std::int64_t d = 0; d < value_dimension; ++d) {
            dv[logical_offset(c.v, b, vh, n, d)] +=
                probabilities[n] *
                doutput[logical_offset(c.doutput, b, h, m, d)];
          }
          if (c.dbias.has_value()) {
            dbias[logical_offset(*c.dbias, c.dbias->dimensions[0] == 1 ? 0 : b,
                                 c.dbias->dimensions[1] == 1 ? 0 : h, m, n)] +=
                ds;
          }
        }
      }
    }
  }
  return {{dq.begin(), dq.end()},
          {dk.begin(), dk.end()},
          {dv.begin(), dv.end()},
          {dbias.begin(), dbias.end()}};
}

}  // namespace flagdnn::reference::cpu
