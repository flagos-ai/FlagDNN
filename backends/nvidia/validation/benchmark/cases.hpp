/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_NVIDIA_VALIDATION_BENCHMARK_CASES_HPP_
#define FLAGDNN_BACKENDS_NVIDIA_VALIDATION_BENCHMARK_CASES_HPP_

#include <algorithm>
#include <span>
#include <vector>

#include "common/case.hpp"

namespace flagdnn::benchmarking {

// NVIDIA performance workloads require a standalone cuDNN reference. Keep
// these confirmed unsupported shapes out of this catalog, while retaining
// the shared workloads for functional validation and other platforms.
inline std::vector<BenchmarkCase> cudnn_benchmark_cases(
    std::span<const BenchmarkCase> cases) {
  std::vector<BenchmarkCase> result;
  result.reserve(cases.size());
  for (const BenchmarkCase& specification : cases) {
    if (specification.operation == Operation::kPointwise) {
      if (specification.pointwise_mode == FLAGDNN_POINTWISE_IDENTITY &&
          !specification.tensors.empty() &&
          specification.tensors.front().dimensions ==
              std::vector<std::int64_t>{1, 1, 1}) {
        continue;
      }
      const bool unsupported_boolean_shape = std::any_of(
          specification.tensors.begin(), specification.tensors.end(),
          [](const TensorSpec& tensor) {
            return tensor.data_type == FLAGDNN_DATA_BOOLEAN &&
                   (tensor.dimensions ==
                        std::vector<std::int64_t>{3, 257, 513} ||
                    tensor.dimensions ==
                        std::vector<std::int64_t>{3, 7, 65, 129} ||
                    tensor.dimensions ==
                        std::vector<std::int64_t>{5, 7, 65, 129});
          });
      if (unsupported_boolean_shape) {
        continue;
      }
    }
    if (specification.operation == Operation::kReduction &&
        !specification.tensors.empty() &&
        specification.tensors.front().data_type == FLAGDNN_DATA_BFLOAT16) {
      // BF16 SUM -> FP32 coverage is supplied by the fp32_output suite.
      continue;
    }
    result.push_back(specification);
  }
  return result;
}

}  // namespace flagdnn::benchmarking

#endif  // FLAGDNN_BACKENDS_NVIDIA_VALIDATION_BENCHMARK_CASES_HPP_
