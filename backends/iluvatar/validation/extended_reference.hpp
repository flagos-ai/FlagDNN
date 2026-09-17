// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef FLAGDNN_ILUVATAR_EXTENDED_REFERENCE_HPP_
#define FLAGDNN_ILUVATAR_EXTENDED_REFERENCE_HPP_
#include "common/index.hpp"
#include "common/normalization_extended.hpp"
#include "common/resample.hpp"
#include "common/statistics.hpp"
#include <stdexcept>
namespace flagdnn::iluvatar::validation {
std::string missing_graph_reference_reason();

// CoreX 7.6.5 has no backend graph API for the remaining extended operations.
// Supported classic APIs are specialized below and must execute, not skip.
template <class Case> std::string extended_reference_skip(const Case &) {
  return missing_graph_reference_reason();
}
template <class Case>
std::unique_ptr<testing::TestExecutable>
build_extended_reference(const Case &) {
  throw std::logic_error("missing adapter reached a supported DNN case");
}
std::string extended_reference_skip(const testing::StatisticsTestCase &);
std::unique_ptr<testing::TestExecutable>
build_extended_reference(const testing::StatisticsTestCase &);
std::string
extended_reference_skip(const testing::ExtendedNormalizationTestCase &);
std::string extended_reference_skip(const testing::ResampleTestCase &);
std::string extended_reference_skip(const testing::IndexTestCase &);
std::unique_ptr<testing::TestExecutable>
build_extended_reference(const testing::ExtendedNormalizationTestCase &);
std::unique_ptr<testing::TestExecutable>
build_extended_reference(const testing::ResampleTestCase &);
std::unique_ptr<testing::TestExecutable>
build_extended_reference(const testing::IndexTestCase &);
} // namespace flagdnn::iluvatar::validation
#endif
