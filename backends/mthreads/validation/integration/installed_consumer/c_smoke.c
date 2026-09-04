/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include <flagdnn/flagdnn.h>

#include <string.h>

int main(void) {
  flagdnnTensorDescriptor_t tensor = NULL;
  const int version_matches =
      flagdnnGetVersion() == FLAGDNN_VERSION_NUMBER &&
      strcmp(flagdnnGetVersionString(), FLAGDNN_VERSION_STRING) == 0 &&
      flagdnnGetExecutionContractVersion() ==
          FLAGDNN_EXECUTION_CONTRACT_VERSION;
  if (!version_matches ||
      flagdnnCreateTensorDescriptor(&tensor) != FLAGDNN_STATUS_SUCCESS ||
      tensor == NULL) {
    return 1;
  }
  return flagdnnDestroyTensorDescriptor(tensor) == FLAGDNN_STATUS_SUCCESS
             ? 0
             : 1;
}
