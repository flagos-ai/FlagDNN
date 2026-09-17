// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#include <cudnn.h>
#include <dlfcn.h>
#include <iostream>
int main() {
  // The audited 7.6.5 ABI has neither FP8 tensor types nor the graph MatMul/MoE
  // operations. A different ABI requires qualification, not a silent skip.
  if (CUDNN_VERSION != 7605 || cudnnGetVersion() != 7605 ||
      dlsym(RTLD_DEFAULT, "cudnnBackendCreateDescriptor") != nullptr) {
    std::cerr << "CoreX DNN capability changed; qualify MatMul/MoE adapter\n";
    return 1;
  }
  std::cout << "DNN_CAPABILITY_UNAVAILABLE: iluvatar: "
            << FLAGDNN_CAPABILITY_OPERATOR << " reason=DNN_API_UNAVAILABLE\n";
  return 77;
}
