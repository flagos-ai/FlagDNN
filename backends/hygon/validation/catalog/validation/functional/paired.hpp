/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include "common/common.hpp"
#include <functional>
#include <map>
#include <numeric>
#include <string>
#include <type_traits>
namespace flagdnn::testing::catalog {
using Cases = std::map<std::string, std::vector<TestTensor>>;
extern Cases captured;
void record(std::string name, std::vector<TestTensor> tensors);
} // namespace flagdnn::testing::catalog
namespace flagdnn::testing::cuda {
struct PairedTolerance {
  double absolute, relative;
};
inline std::size_t element_count(const TestTensor &tensor) {
  return std::accumulate(tensor.dimensions.begin(), tensor.dimensions.end(),
                         std::size_t{1}, std::multiplies<>());
}
template <class Case, class Build, class Input, class Reference,
          class Tolerance, class... Args>
int run_paired_cases(int, char **, std::span<const Case> cases, const char *,
                     Build &&, Input &&, Reference &&, Tolerance &&,
                     Args &&...) {
  static_assert(std::is_same_v<std::invoke_result_t<Input, const Case &>,
                               std::vector<std::vector<float>>>);
  for (const auto &value : cases) {
    auto tensors = value.inputs;
    tensors.insert(tensors.end(), value.outputs.begin(), value.outputs.end());
    catalog::record(value.name, std::move(tensors));
  }
  return 0;
}
} // namespace flagdnn::testing::cuda
