/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "cases.hpp"

#include <iterator>
#include <utility>
#include <vector>

namespace flagdnn::benchmarking {
namespace {

void append(std::vector<BenchmarkCase>& destination,
            std::vector<BenchmarkCase> source) {
  destination.insert(destination.end(),
                     std::make_move_iterator(source.begin()),
                     std::make_move_iterator(source.end()));
}

}  // namespace

std::vector<BenchmarkCase> all_cases() {
  std::vector<BenchmarkCase> result;
  append(result, relu_cases());
  append(result, add_cases());
  append(result, reduction_cases());
  append(result, conv_fprop_cases());
  append(result, conv_dgrad_cases());
  append(result, conv_wgrad_cases());
  append(result, layernorm_cases());
  append(result, rmsnorm_cases());
  append(result, batchnorm_cases());
  append(result, batchnorm_inference_cases());
  append(result, matmul_cases());
  append(result, reshape_cases());
  append(result, transpose_cases());
  append(result, slice_cases());
  return result;
}

}  // namespace flagdnn::benchmarking
