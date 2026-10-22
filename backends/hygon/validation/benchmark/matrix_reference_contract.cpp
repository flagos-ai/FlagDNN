/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "parallel_matmul_reference.hpp"

#include <bit>
#include <chrono>
#include <iostream>
#include <numeric>

namespace cpu = flagdnn::reference::cpu;
namespace hv = flagdnn::validation::hygon;
using Shape = std::vector<std::int64_t>;

std::size_t count(const Shape& shape) {
  return std::accumulate(shape.begin(), shape.end(), std::size_t{1},
                         std::multiplies<>());
}
std::vector<float> values(std::size_t size, int seed) {
  std::vector<float> result(size);
  for (std::size_t i = 0; i < size; ++i)
    result[i] = static_cast<float>(static_cast<int>((i * 13 + seed) % 41) - 20)
                / 13.0F;
  return result;
}
void equal_bits(std::span<const float> actual, std::span<const float> expected) {
  if (actual.size() != expected.size())
    throw std::runtime_error("partition changed output size");
  for (std::size_t i = 0; i < actual.size(); ++i)
    if (std::bit_cast<std::uint32_t>(actual[i]) !=
        std::bit_cast<std::uint32_t>(expected[i]))
      throw std::runtime_error("partition changed output bits at " +
                               std::to_string(i));
}
template <class F> void rejects(F action) {
  try { action(); } catch (const std::invalid_argument&) { return; }
  throw std::runtime_error("invalid CPU partition accepted");
}
int main() {
  try {
    const std::vector<cpu::MatmulParameters> geometry = {
        {{5, 7}, {7, 9}, {5, 9}},
        {{2, 5, 7}, {1, 7, 3}, {2, 5, 3}},
        {{1, 5, 7}, {3, 7, 9}, {3, 5, 9}},
        {{2, 1, 5, 7}, {3, 7, 9}, {2, 3, 5, 9}},
        {{1, 2, 1, 5, 7}, {4, 1, 3, 7, 9}, {4, 2, 3, 5, 9}},
        {{2, 1, 7}, {1, 7, 9}, {2, 1, 9}},
        {{2, 17, 32}, {32, 24}, {2, 17, 24}},
        {{5, 7}, {7, 32}, {5, 32}},
        {{2, 1, 5, 7}, {3, 7, 33}, {2, 3, 5, 33}},
        {{1, 2, 1, 5, 7}, {4, 1, 3, 7, 65}, {4, 2, 3, 5, 65}},
        {{2, 1, 7}, {1, 7, 129}, {2, 1, 129}}};
    std::size_t checked = 0;
    for (auto parameters : geometry) {
      auto a = values(count(parameters.a_shape), 3);
      auto b = values(count(parameters.b_shape), 7);
      // Cancellation-sensitive rows catch changes to reduction order.
      for (std::size_t i = 0; i + 2 < a.size(); i += 7) {
        a[i] = 16777216.0F; a[i + 1] = 1.0F; a[i + 2] = -16777216.0F;
      }
      for (const auto type : {FLAGDNN_DATA_FLOAT32, FLAGDNN_DATA_FLOAT16,
                              FLAGDNN_DATA_BFLOAT16}) {
        parameters.input_type = type;
        for (const int precision : {0, 1, 2}) {
          if (precision == 2 && type != FLAGDNN_DATA_FLOAT32) continue;
          parameters.input_precision = precision;
          const auto expected = cpu::evaluate_matmul(parameters, a, b);
          for (const std::size_t workers : {1U, 2U, 3U, 4U, 9U}) {
            equal_bits(hv::parallel_matmul_reference(parameters, a, b,
                                                     workers, 0), expected);
            ++checked;
          }
        }
      }
    }
    auto p = geometry.front(); p.input_precision = 1;
    const auto a = values(count(p.a_shape), 1), b = values(count(p.b_shape), 2);
    rejects([&] { (void)hv::parallel_matmul_reference(p, a, b, 0, 0); });
    rejects([&] { (void)hv::parallel_matmul_reference(p, {}, b, 4, 0); });
    auto wrong = p; wrong.output_shape[0] += 1;
    rejects([&] { (void)hv::parallel_matmul_reference(wrong, a, b, 4, 0); });
    wrong = geometry[1]; wrong.output_shape[0] = 3;
    rejects([&] { (void)hv::parallel_matmul_reference(wrong,
        values(count(wrong.a_shape), 1), values(count(wrong.b_shape), 2), 4, 0); });
    p = {{2, 512, 512}, {2, 512, 512}, {2, 512, 512}, FLAGDNN_DATA_FLOAT32, 1};
    const auto large_a = values(count(p.a_shape), 3);
    const auto large_b = values(count(p.b_shape), 5);
    using Clock = std::chrono::steady_clock;
    const auto begin = Clock::now();
    const auto serial = cpu::evaluate_matmul(p, large_a, large_b);
    const auto middle = Clock::now();
    const auto parallel = hv::parallel_matmul_reference(p, large_a, large_b);
    const auto end = Clock::now();
    equal_bits(parallel, serial);
    std::cout << "PASS CPU MatMul partition contracts configurations=" << checked
              << " large_outputs=" << parallel.size()
              << " serial_seconds=" << std::chrono::duration<double>(middle-begin).count()
              << " parallel_seconds=" << std::chrono::duration<double>(end-middle).count()
              << '\n';
  } catch (const std::exception& error) {
    std::cerr << "FAIL " << error.what() << '\n';
    return 1;
  }
}
