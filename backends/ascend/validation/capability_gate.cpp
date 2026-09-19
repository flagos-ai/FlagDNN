/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include <iostream>
#include <string_view>
int main(int argc, char **argv) {
  if (argc != 2)
    return 2;
  const std::string_view op = argv[1];
  std::cout << "CAPABILITY_UNAVAILABLE: ascend: " << op << ": ";
  if (op == "rng")
    std::cout << "CANN ACLNN does not expose the required per-element Philox "
                 "counter and uniform/normal conversion contract";
  else
    std::cout << "Ascend 910B ACLNN does not support the required FP8 storage "
                 "and scaled matrix/attention interfaces";
  std::cout << '\n';
  return 77;
}
