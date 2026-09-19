/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "validation/functional/aclnn_plan.hpp"

int main() {
  using namespace flagdnn::testing;
  // Descriptors do not require device allocation. Repeated ownership transfer
  // catches double destruction when one tensor participates in several lists.
  for (int repeat = 0; repeat < 100; ++repeat) {
    ascend::Plan plan({}, [](ascend::Plan &) {});
    const TestTensor spec{1, FLAGDNN_DATA_FLOAT32, {2, 3}, {3, 1}};
    auto *tensor = plan.tensor(spec, nullptr);
    plan.list({tensor});
    plan.list({tensor, tensor});
  }
}
