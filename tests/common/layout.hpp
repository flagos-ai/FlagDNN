/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_TESTS_COMMON_LAYOUT_HPP_
#define FLAGDNN_TESTS_COMMON_LAYOUT_HPP_

#include "common/common.hpp"

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace flagdnn::testing {

enum class LayoutOperation { kReshape, kTranspose, kSlice };

struct LayoutTestCase {
  std::string name;
  LayoutOperation operation = LayoutOperation::kReshape;
  TestTensor input;
  TestTensor output;
  std::vector<std::int64_t> permutation;
  std::vector<std::pair<std::int64_t, std::int64_t>> slices;
  std::vector<std::int64_t> slice_strides;
  bool autotune = false;
};

using LayoutExecutable = TestExecutable;

[[nodiscard]] std::vector<LayoutTestCase> make_layout_cases(
    LayoutOperation operation);
void validate_layout_case(const LayoutTestCase& test_case);

[[nodiscard]] std::unique_ptr<LayoutExecutable> build_flagdnn_layout(
    flagdnn::Handle& handle,
    const LayoutTestCase& test_case);

/* Implemented by the selected backends/<platform>/validation/functional adapter. */
[[nodiscard]] std::unique_ptr<LayoutExecutable> build_layout_reference(
    const LayoutTestCase& test_case);

/* Implemented by backends/<platform>/validation/functional/layout_runner.cpp. */
int run_layout_functional_test(int argc,
                               char** argv,
                               std::span<const LayoutTestCase> cases,
                               std::string_view suite_name);

}  // namespace flagdnn::testing

#endif  // FLAGDNN_TESTS_COMMON_LAYOUT_HPP_
