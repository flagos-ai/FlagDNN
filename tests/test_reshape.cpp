/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/layout.hpp"

#include <span>

int main(int argc, char** argv) {
  const auto cases = flagdnn::testing::make_layout_cases(
      flagdnn::testing::LayoutOperation::kReshape);
  return flagdnn::testing::run_layout_functional_test(
      argc,
      argv,
      std::span<const flagdnn::testing::LayoutTestCase>(cases),
      "FLAGDNN_RESHAPE_FUNCTIONAL");
}
