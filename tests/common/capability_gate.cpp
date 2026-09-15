/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include <iostream>
int main() {
  std::cout << "CAPABILITY_UNAVAILABLE: " << FLAGDNN_CAPABILITY_PLATFORM << ": "
            << FLAGDNN_CAPABILITY_OPERATOR
            << " execution has not been implemented for this backend\n";
  return 77;
}
