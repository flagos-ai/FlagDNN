/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include "common/add.hpp"
#include "common/layout.hpp"
#include "common/pointwise.hpp"
#include "common/reduction.hpp"
namespace flagdnn::testing {
int run_ascend_pointwise_dtype_cases(int, char **,
                                     std::span<const PointwiseTestCase>,
                                     bool = false, bool = false);
int run_ascend_add_dtype_cases(int, char **, std::span<const AddTestCase>);
int run_ascend_layout_dtype_cases(int, char **, std::span<const LayoutTestCase>,
                                  bool = false, bool = false);
int run_ascend_reduction_dtype_cases(int, char **,
                                     std::span<const ReductionTestCase>,
                                     bool = false);
} // namespace flagdnn::testing
