/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include "hipdnn_reference.hpp"
#include <algorithm>
#include <cctype>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
namespace flagdnn::testing::hygon {
template <class Case>
int skip_cases(int argc, const std::vector<Case> &cases,
               std::string_view operation, bool benchmark,
               std::string_view reason = "Hygon compiler/validation adapter is "
                                         "not implemented for this operator") {
  if (argc != 3)
    return 2;
  if (cases.empty())
    throw std::invalid_argument("empty capability case catalog");
  std::string marker(operation);
  std::transform(marker.begin(), marker.end(), marker.begin(),
                 [](unsigned char c) { return std::toupper(c); });
  for (const auto &value : cases) {
    std::vector<validation::hygon::ReferenceTensor> tensors;
    if constexpr (requires { value.inputs; }) {
      for (const auto &input : value.inputs)
        tensors.push_back(validation::hygon::as_reference_tensor(input));
    } else {
      tensors.push_back(validation::hygon::as_reference_tensor(value.input));
    }
    if constexpr (requires { value.outputs; }) {
      for (const auto &output : value.outputs)
        tensors.push_back(validation::hygon::as_reference_tensor(output));
    } else {
      tensors.push_back(validation::hygon::as_reference_tensor(value.output));
    }
    std::cout << "[SKIP][hipdnn] op=" << operation << " case=" << value.name
              << " reason=" << reason << ' '
              << validation::hygon::describe_reference_tensors(tensors) << '\n';
  }
  std::cout << "FLAGDNN_" << marker
            << (benchmark ? "_BENCHMARK" : "_FUNCTIONAL")
            << ": SKIP cases=" << cases.size()
            << " executed=0 skipped=" << cases.size() << '\n';
  return 77;
}
} // namespace flagdnn::testing::hygon
