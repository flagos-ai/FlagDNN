// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "common/pointwise.hpp"
#include "functional/runner_support.hpp"
#include "reference/cpu/pointwise.hpp"
#include <cstring>
#include <stdexcept>
namespace flagdnn::iluvatar::validation::functional {
using Bytes = std::vector<std::uint8_t>;
inline Bytes raw_pattern(const testing::TestTensor &tensor,
                         std::size_t salt = 0) {
  Bytes result(element_count(tensor) * data_type_size(tensor.data_type));
  for (std::size_t i = 0; i < result.size(); ++i)
    result[i] = static_cast<std::uint8_t>(i * 37 + salt * 83);
  return result;
}

} // namespace flagdnn::iluvatar::validation::functional
