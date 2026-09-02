/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_HYGON_VALIDATION_FUNCTIONAL_ACCURACY_HPP_
#define FLAGDNN_BACKENDS_HYGON_VALIDATION_FUNCTIONAL_ACCURACY_HPP_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace flagdnn::testing::hygon_functional {

struct Accuracy {
  double maximum_absolute = 0.0;
  double maximum_relative = 0.0;
};

inline Accuracy compare_outputs(std::span<const float> actual,
                                std::span<const float> reference,
                                double absolute_tolerance,
                                double relative_tolerance,
                                std::string_view case_name,
                                std::string_view reference_name = "hipDNN") {
  if (actual.size() != reference.size()) {
    throw std::runtime_error("FlagDNN and " + std::string(reference_name) +
                             " output sizes differ");
  }
  Accuracy result;
  for (std::size_t index = 0; index < actual.size(); ++index) {
    const double left = actual[index];
    const double right = reference[index];
    const double absolute = std::abs(left - right);
    const double relative =
        absolute / std::max({std::abs(left), std::abs(right), 1.0e-30});
    result.maximum_absolute = std::max(result.maximum_absolute, absolute);
    result.maximum_relative = std::max(result.maximum_relative, relative);
    if (!std::isfinite(absolute) ||
        (absolute > absolute_tolerance && relative > relative_tolerance)) {
      std::ostringstream message;
      message << case_name << " differs at output element " << index
              << ": FlagDNN=" << left << ", " << reference_name << '='
              << right
              << ", abs=" << absolute << ", rel=" << relative
              << ", atol=" << absolute_tolerance
              << ", rtol=" << relative_tolerance;
      throw std::runtime_error(message.str());
    }
  }
  return result;
}

} // namespace flagdnn::testing::hygon_functional

#endif // FLAGDNN_BACKENDS_HYGON_VALIDATION_FUNCTIONAL_ACCURACY_HPP_
