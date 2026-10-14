/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "reference/cpu/normalization.hpp"
#include "reference/cpu/pointwise.hpp"
#include "reference/cpu/matrix.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace cpu = flagdnn::reference::cpu;
using Values = std::vector<float>;
void close(std::span<const float> actual, std::span<const float> expected) {
  if (actual.size() != expected.size()) throw std::runtime_error("output size differs");
  for (std::size_t i = 0; i < actual.size(); ++i)
    if (!std::isfinite(actual[i]) || std::abs(actual[i] - expected[i]) > 2e-6F)
      throw std::runtime_error("normalization result differs");
}
template <class F> void rejects(F f) {
  try { f(); } catch (const std::invalid_argument&) { return; }
  throw std::runtime_error("invalid normalization input accepted");
}
int main() {
  try {
    const Values cancellation{16777216,1,-16777216};
    close(cpu::evaluate_reduction({{3}, -1, FLAGDNN_REDUCTION_ADD, true}, cancellation),
          Values{0});
    close(cpu::evaluate_reduction({{3}, -1, FLAGDNN_REDUCTION_ADD}, cancellation),
          Values{1});
    close(cpu::evaluate_reduction({{2,3}, 0, FLAGDNN_REDUCTION_AVG, true},
        Values{1,2,3,5,6,7}), Values{3,4,5});
    close(cpu::evaluate_reduction({{2,3}, -1, FLAGDNN_REDUCTION_MUL, true},
        Values{1,2,3,4,5,6}), Values{6,120});
    const auto layer = cpu::evaluate_row_normalization(
        false, Values{1,3, 5,7}, Values{2,3}, Values{-1,2}, 3.0);
    close(layer.at(0), Values{-2,3.5F, -2,3.5F});
    close(layer.at(1), Values{2,6});
    close(layer.at(2), Values{0.5F,0.5F});
    const auto rms = cpu::evaluate_row_normalization(
        true, Values{3,4}, Values{2,3}, Values{-1,2}, 3.5);
    close(rms.at(0), Values{0.5F,5});
    close(rms.at(1), Values{0});
    close(rms.at(2), Values{0.25F});
    const auto stable = cpu::evaluate_row_normalization(false,
        Values{1048575,1048577}, Values{1,1}, Values{0,0}, 3.0);
    close(stable[0], Values{-0.5F,0.5F});
    close(stable[1], Values{1048576});

    const Values input{1,3, 2,4, 5,7, 6,8};
    const auto batch = cpu::evaluate_batchnorm_training(input, 2,
        Values{3,6}, Values{1,-1}, Values{0,1}, Values{1,2}, 4.0, 0.25);
    close(batch.at(0), Values{-2,0,-7,-3,2,4,1,5});
    close(batch.at(1), Values{4,5});
    close(batch.at(2), Values{1.0F/3,1.0F/3});
    close(batch.at(3), Values{1,2});
    close(batch.at(4), Values{29.0F/12,19.0F/6});
    close(cpu::evaluate_batchnorm_inference(input, 2, batch[1], batch[2],
        Values{3,6}, Values{1,-1}), batch[0]);
    const auto singleton = cpu::evaluate_batchnorm_training(Values{2,4}, 1,
        Values{1,1}, Values{0,0}, Values{0,0}, Values{2,4}, 1.0, 0.5);
    close(singleton[0], Values{0,0});
    close(singleton[4], Values{1,2});
    // Ascend's saved mean rounds to FP32 before variance/output computation.
    // A double-mean replacement would produce variance 1 instead of 2 here.
    const auto rounded_mean = cpu::evaluate_batchnorm_training(
        Values{16777216,16777218}, 2, Values{1}, Values{0}, Values{0},
        Values{0}, 2.0, 1.0);
    close(rounded_mean[0], Values{0,1});
    close(rounded_mean[1], Values{16777216});
    close(rounded_mean[2], Values{0.5F});
    close(rounded_mean[4], Values{4});
    Values variance{0,3,8};
    cpu::variance_to_inverse_standard_deviation(variance, 1.0);
    close(variance, Values{1,0.5F,1.0F/3});

    rejects([] { (void)cpu::evaluate_row_normalization(false, Values{1,2,3},
        Values{1,1}, Values{0,0}, 1e-5); });
    rejects([] { (void)cpu::evaluate_row_normalization(true,
        Values{std::numeric_limits<float>::infinity()}, Values{1}, Values{0}, 1e-5); });
    rejects([] { Values v{-1}; cpu::variance_to_inverse_standard_deviation(v, 1.0); });
    rejects([] { Values v{1}; cpu::variance_to_inverse_standard_deviation(v, 0.0); });
    rejects([&] { (void)cpu::evaluate_batchnorm_training(input, 3,
        Values{1,1}, Values{0,0}, Values{0,0}, Values{1,1}, 1e-5, 0.1); });

    // Backward consumes logits; it must retain the tiny gradient at +20.
    if (!(cpu::sigmoid_backward(1,20) > 0) || cpu::sigmoid(0) != 0.5F ||
        cpu::sigmoid_backward(4,0) != 1.0F ||
        cpu::sigmoid_backward(1,-20) != cpu::sigmoid_backward(1,20))
      throw std::runtime_error("sigmoid logit contract differs");
    for (float value : {-3.0F, -1.0F, 0.0F, 1.0F, 3.0F}) {
      const auto exact = cpu::gelu_exact(value);
      if (std::abs(cpu::gelu_exact(value) - cpu::gelu_exact(-value) - value) > 1e-6F ||
          std::abs(cpu::gelu_approximate_tanh(value) - exact) > 5e-4F ||
          std::abs(cpu::gelu_accurate_logit(value) - exact) > 5e-6F)
        throw std::runtime_error("GELU reference contract differs");
    }
    std::cout << "PASS CPU normalization and activation contracts\n";
  } catch (const std::exception& error) {
    std::cerr << "FAIL " << error.what() << '\n';
    return 1;
  }
}
