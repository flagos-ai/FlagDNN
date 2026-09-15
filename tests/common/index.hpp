/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#ifndef FLAGDNN_TESTS_COMMON_INDEX_HPP_
#define FLAGDNN_TESTS_COMMON_INDEX_HPP_

#include <memory>
#include <span>
#include <string>
#include <vector>

#include "common/common.hpp"

namespace flagdnn::testing {
struct IndexTestCase {
  std::string name;
  std::string operation;
  std::vector<TestTensor> inputs;
  TestTensor output;
  std::int64_t axis = 0;
};
[[nodiscard]] std::vector<IndexTestCase> make_index_cases(
    std::string_view operation);
[[nodiscard]] std::unique_ptr<TestExecutable> build_flagdnn_index(
    flagdnn::Handle& handle, const IndexTestCase& specification);
int run_index_functional_test(int argc, char** argv,
                              std::span<const IndexTestCase> cases,
                              bool benchmark = false);
}  // namespace flagdnn::testing
#endif
