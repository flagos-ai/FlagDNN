/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include "common/attention.hpp"
#include "host_runner.hpp"
#include <limits>

namespace flagdnn::testing::hygon_functional::host {
// Dense scalar evaluation of the FP8 attention equations. It deliberately has
// no knowledge of GPU tiles, launch parameters, or online-softmax scheduling.
inline std::size_t attention_offset(const TestTensor &t, std::int64_t b,
                                    std::int64_t h, std::int64_t m,
                                    std::int64_t d) {
  return ((b * t.dimensions[1] + h) * t.dimensions[2] + m) * t.dimensions[3] +
         d;
}
inline float quantize(float value, flagdnnDataType_t type) {
  const std::array<float, 1> input{value};
  return io::decode(io::encode(input, type), type, 1)[0];
}
struct AttentionProbabilities {
  std::vector<double> p;
  std::vector<float> stats;
  double maximum = 0;
};
template <class Case>
AttentionProbabilities
fp8_probabilities(const Case &c, const Values &values,
                  std::span<const float> bias = {},
                  std::span<const float> saved_stats = {}) {
  const auto B = c.q.dimensions[0], H = c.q.dimensions[1],
             M = c.q.dimensions[2], N = c.k.dimensions[2],
             D = c.q.dimensions[3];
  const auto shift =
      c.options.diagonal_alignment == AttentionDiagonalAlignment::kBottomRight
          ? N - M
          : 0;
  const auto lower = c.options.diagonal_band_left_bound
                         ? 1 - *c.options.diagonal_band_left_bound + shift
                         : std::numeric_limits<std::int64_t>::min();
  const auto upper = c.options.diagonal_band_right_bound
                         ? *c.options.diagonal_band_right_bound + shift
                         : std::numeric_limits<std::int64_t>::max();
  const double scale =
      c.options.attention_scale.value_or(1.0F / std::sqrt(float(D))) *
      double(c.descale_q.value) * c.descale_k.value;
  AttentionProbabilities result{std::vector<double>(B * H * M * N),
                                std::vector<float>(B * H * M), 0};
  for (std::int64_t b = 0; b < B; ++b)
    for (std::int64_t h = 0; h < H; ++h)
      for (std::int64_t m = 0; m < M; ++m) {
        const auto row = (b * H + h) * M + m, kh = h / (H / c.k.dimensions[1]);
        std::vector<double> scores(N, -std::numeric_limits<double>::infinity());
        double maximum = -std::numeric_limits<double>::infinity(), sum = 0;
        for (std::int64_t n = 0; n < N; ++n) {
          if (n - m < lower || n - m > upper)
            continue;
          double score = 0;
          for (std::int64_t d = 0; d < D; ++d)
            score += double(values[0][attention_offset(c.q, b, h, m, d)]) *
                     values[1][attention_offset(c.k, b, kh, n, d)];
          score *= scale;
          if constexpr (requires { c.bias; })
            if (c.bias)
              score += bias[attention_offset(
                  *c.bias, c.bias->dimensions[0] == 1 ? 0 : b,
                  c.bias->dimensions[1] == 1 ? 0 : h, m, n)];
          scores[n] = score;
          maximum = std::max(maximum, score);
        }
        for (auto &score : scores) {
          score = std::isfinite(score) ? std::exp(score - maximum) : 0;
          sum += score;
        }
        if (!(sum > 0))
          throw std::runtime_error("empty FP8 attention reference row");
        result.stats[row] = float(maximum + std::log(sum));
        for (std::int64_t n = 0; n < N; ++n) {
          result.p[row * N + n] =
              saved_stats.empty()
                  ? scores[n] / sum
                  : scores[n] * std::exp(maximum - saved_stats[row]);
          result.maximum = std::max(result.maximum, result.p[row * N + n]);
        }
      }
  return result;
}
inline Values fp8_forward_reference(const SdpaFp8TestCase &c,
                                    const Values &values) {
  const auto probabilities =
      fp8_probabilities(c, values,
                        c.bias ? std::span<const float>(values.back())
                               : std::span<const float>{});
  const auto B = c.q.dimensions[0], H = c.q.dimensions[1],
             M = c.q.dimensions[2], N = c.k.dimensions[2],
             D = c.v.dimensions[3];
  std::vector<float> quantized_probabilities(probabilities.p.size());
  for (std::size_t i = 0; i < probabilities.p.size(); ++i)
    quantized_probabilities[i] =
        quantize(float(probabilities.p[i] * c.scale_s.value), c.q.data_type);
  std::vector<float> output(io::element_count(c.output));
  double amax = 0;
  for (std::int64_t b = 0; b < B; ++b)
    for (std::int64_t h = 0; h < H; ++h)
      for (std::int64_t m = 0; m < M; ++m)
        for (std::int64_t d = 0; d < D; ++d) {
          double sum = 0;
          const auto row = (b * H + h) * M + m,
                     vh = h / (H / c.v.dimensions[1]);
          for (std::int64_t n = 0; n < N; ++n)
            sum += quantized_probabilities[row * N + n] *
                   double(values[2][attention_offset(c.v, b, vh, n, d)]);
          sum *= double(c.descale_s.value) * c.descale_v.value;
          amax = std::max(amax, std::abs(sum));
          output[attention_offset(c.output, b, h, m, d)] =
              float(sum * c.scale_o.value);
        }
  Values result{output};
  if (c.stats)
    result.push_back(probabilities.stats);
  result.push_back({float(probabilities.maximum)});
  result.push_back({float(amax)});
  return result;
}
inline Values fp8_backward_reference(const SdpaFp8BackwardTestCase &c,
                                     const Values &values) {
  const auto probabilities = fp8_probabilities(c, values, {}, values[5]);
  const auto B = c.q.dimensions[0], H = c.q.dimensions[1],
             M = c.q.dimensions[2], N = c.k.dimensions[2],
             D = c.q.dimensions[3];
  std::vector<double> dq(io::element_count(c.dq)), dk(io::element_count(c.dk)),
      dv(io::element_count(c.dv));
  double amax_dp = 0;
  const double scale =
      c.options.attention_scale.value_or(1.0F / std::sqrt(float(D)));
  for (std::int64_t b = 0; b < B; ++b)
    for (std::int64_t h = 0; h < H; ++h)
      for (std::int64_t m = 0; m < M; ++m) {
        const auto row = (b * H + h) * M + m, kh = h / (H / c.k.dimensions[1]);
        double delta = 0;
        for (std::int64_t d = 0; d < D; ++d)
          delta += double(values[3][attention_offset(c.output, b, h, m, d)]) *
                   values[4][attention_offset(c.doutput, b, h, m, d)];
        delta *= double(c.descale_o.value) * c.descale_doutput.value;
        for (std::int64_t n = 0; n < N; ++n) {
          double dp = 0;
          for (std::int64_t d = 0; d < D; ++d)
            dp += double(values[4][attention_offset(c.doutput, b, h, m, d)]) *
                  values[2][attention_offset(c.v, b, kh, n, d)];
          dp *= double(c.descale_doutput.value) * c.descale_v.value;
          const double ds = probabilities.p[row * N + n] * (dp - delta) * scale;
          amax_dp = std::max(amax_dp, std::abs(ds));
          const double quantized_ds =
              quantize(float(ds * c.scale_dp.value), c.q.data_type) *
              double(c.descale_dp.value);
          const double quantized_p =
              quantize(float(probabilities.p[row * N + n] * c.scale_s.value),
                       c.q.data_type) *
              double(c.descale_s.value);
          for (std::int64_t d = 0; d < D; ++d) {
            dq[attention_offset(c.dq, b, h, m, d)] +=
                quantized_ds * values[1][attention_offset(c.k, b, kh, n, d)] *
                c.descale_k.value;
            dk[attention_offset(c.dk, b, kh, n, d)] +=
                quantized_ds * values[0][attention_offset(c.q, b, h, m, d)] *
                c.descale_q.value;
            dv[attention_offset(c.dv, b, kh, n, d)] +=
                quantized_p *
                values[4][attention_offset(c.doutput, b, h, m, d)] *
                c.descale_doutput.value;
          }
        }
      }
  Values result;
  std::vector<float> maxima;
  const std::array scales{c.scale_dq.value, c.scale_dk.value, c.scale_dv.value};
  for (const auto *gradient : {&dq, &dk, &dv}) {
    double maximum = 0;
    std::vector<float> output;
    for (auto value : *gradient) {
      maximum = std::max(maximum, std::abs(value));
      output.push_back(float(value * scales[result.size()]));
    }
    result.push_back(std::move(output));
    maxima.push_back(float(maximum));
  }
  for (const auto maximum : maxima)
    result.push_back({maximum});
  result.push_back({float(amax_dp)});
  return result;
}
} // namespace flagdnn::testing::hygon_functional::host
