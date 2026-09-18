// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef FLAGDNN_THEAD_ACDNN_EXTENDED_REFERENCE_HPP_
#define FLAGDNN_THEAD_ACDNN_EXTENDED_REFERENCE_HPP_
#include "common/normalization_extended.hpp"
#include "common/resample.hpp"
#include "common/statistics.hpp"
namespace flagdnn::validation::thead {
std::unique_ptr<flagdnn::testing::TestExecutable> make_acdnn_extended_reference(
    const flagdnn::testing::ExtendedNormalizationTestCase&);
std::unique_ptr<flagdnn::testing::TestExecutable> make_acdnn_extended_reference(
    const flagdnn::testing::StatisticsTestCase&);
std::unique_ptr<flagdnn::testing::TestExecutable> make_acdnn_extended_reference(
    const flagdnn::testing::ResampleTestCase&);
}  // namespace flagdnn::validation::thead
#endif
