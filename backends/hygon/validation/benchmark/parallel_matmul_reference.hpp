/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include "reference/cpu/matrix.hpp"

#include <algorithm>
#include <future>
#include <limits>
#include <stdexcept>

namespace flagdnn::validation::hygon {

// Partition independent outputs, packing columns for large row strides. Each
// task calls the independently validated public CPU implementation with the
// original reduction order and precision. Batch broadcasting remains intact.
inline std::vector<float> parallel_matmul_reference(
    const reference::cpu::MatmulParameters& parameters,
    std::span<const float> a, std::span<const float> b,
    std::size_t requested_workers = 4,
    std::size_t minimum_work = 8U * 1024U * 1024U) {
  const auto count = [](const std::vector<std::int64_t>& shape) {
    if (shape.size() < 2)
      throw std::invalid_argument("CPU MatMul partition requires rank >= 2");
    std::size_t size = 1;
    for (const auto dimension : shape) {
      if (dimension <= 0 ||
          static_cast<std::uint64_t>(dimension) >
              std::numeric_limits<std::size_t>::max() / size)
        throw std::invalid_argument("CPU MatMul partition shape is invalid");
      size *= static_cast<std::size_t>(dimension);
    }
    return size;
  };
  const auto& as = parameters.a_shape;
  const auto& bs = parameters.b_shape;
  const auto& ys = parameters.output_shape;
  const auto a_count = count(as), b_count = count(bs), y_count = count(ys);
  const auto m = static_cast<std::size_t>(ys[ys.size() - 2]);
  const auto n = static_cast<std::size_t>(ys.back());
  const auto k = static_cast<std::size_t>(as.back());
  if (requested_workers == 0 || a.size() != a_count || b.size() != b_count ||
      as.size() > ys.size() || bs.size() > ys.size() ||
      as[as.size() - 2] != static_cast<std::int64_t>(m) ||
      bs[bs.size() - 2] != static_cast<std::int64_t>(k) ||
      bs.back() != static_cast<std::int64_t>(n))
    throw std::invalid_argument("CPU MatMul partition dimensions differ");
  for (std::size_t trailing = 0; trailing + 2 < ys.size(); ++trailing) {
    const auto ad = trailing + 2 < as.size() ? as[as.size() - 3 - trailing] : 1;
    const auto bd = trailing + 2 < bs.size() ? bs[bs.size() - 3 - trailing] : 1;
    if ((ad != bd && ad != 1 && bd != 1) ||
        ys[ys.size() - 3 - trailing] != std::max(ad, bd))
      throw std::invalid_argument("CPU MatMul partition batch shape differs");
  }
  const auto worker_limit = std::min(requested_workers, std::size_t{4});
  // Keep automatic/default precision interpretation entirely in the public
  // oracle. Hygon benchmark callers provide explicit IEEE or TF32 precision.
  if (worker_limit == 1 ||
      (parameters.input_precision != 1 && parameters.input_precision != 2) ||
      static_cast<long double>(y_count) * k < minimum_work)
    return reference::cpu::evaluate_matmul(parameters, a, b);

  // The public oracle walks B by columns. Packing narrow column tiles avoids
  // page-sized strides without changing any product, sum or rounding step.
  // Explicit precision is required above: automatic precision depends on N.
  constexpr std::size_t tile_columns = 32;
  if (n >= tile_columns) {
    const auto tiles = n / tile_columns + (n % tile_columns != 0);
    const auto workers = std::min(worker_limit, tiles);
    const auto b_rows = b_count / n;
    const auto output_rows = y_count / n;
    std::vector<float> result(y_count);
    std::vector<std::future<void>> pending;
    pending.reserve(workers);
    for (std::size_t worker = 0; worker < workers; ++worker) {
      pending.push_back(std::async(std::launch::async, [&, worker] {
        for (std::size_t tile = worker; tile < tiles; tile += workers) {
          const auto first = tile * tile_columns;
          const auto columns = std::min(tile_columns, n - first);
          auto part = parameters;
          part.b_shape.back() = static_cast<std::int64_t>(columns);
          part.output_shape.back() = static_cast<std::int64_t>(columns);
          std::vector<float> packed(b_rows * columns);
          for (std::size_t row = 0; row < b_rows; ++row)
            std::copy_n(b.data() + row * n + first, columns,
                        packed.data() + row * columns);
          const auto output = reference::cpu::evaluate_matmul(part, a, packed);
          if (output.size() != output_rows * columns)
            throw std::runtime_error("CPU MatMul column output size differs");
          for (std::size_t row = 0; row < output_rows; ++row)
            std::copy_n(output.data() + row * columns, columns,
                        result.data() + row * n + first);
        }
      }));
    }
    for (auto& task : pending) task.get();
    return result;
  }

  const auto workers = std::min(worker_limit, m);
  if (workers == 1)
    return reference::cpu::evaluate_matmul(parameters, a, b);
  const auto a_batches = a_count / (m * k);
  const auto output_batches = y_count / (m * n);
  std::vector<std::future<std::vector<float>>> pending;
  pending.reserve(workers);
  for (std::size_t worker = 0; worker < workers; ++worker) {
    const auto first = (m / workers) * worker + std::min(m % workers, worker);
    const auto rows = m / workers + (worker < m % workers ? 1U : 0U);
    auto part = parameters;
    part.a_shape[as.size() - 2] = static_cast<std::int64_t>(rows);
    part.output_shape[ys.size() - 2] = static_cast<std::int64_t>(rows);
    pending.push_back(std::async(std::launch::async,
        [part = std::move(part), a, b, a_batches, m, k, first, rows] {
          std::vector<float> packed(a_batches * rows * k);
          for (std::size_t batch = 0; batch < a_batches; ++batch)
            std::copy_n(a.data() + (batch * m + first) * k, rows * k,
                        packed.data() + batch * rows * k);
          return reference::cpu::evaluate_matmul(part, packed, b);
        }));
  }
  std::vector<float> result(y_count);
  for (std::size_t worker = 0; worker < workers; ++worker) {
    const auto first = (m / workers) * worker + std::min(m % workers, worker);
    const auto rows = m / workers + (worker < m % workers ? 1U : 0U);
    const auto part = pending[worker].get();
    if (part.size() != output_batches * rows * n)
      throw std::runtime_error("CPU MatMul partition output size differs");
    for (std::size_t batch = 0; batch < output_batches; ++batch)
      std::copy_n(part.data() + batch * rows * n, rows * n,
                  result.data() + (batch * m + first) * n);
  }
  return result;
}
}  // namespace flagdnn::validation::hygon
