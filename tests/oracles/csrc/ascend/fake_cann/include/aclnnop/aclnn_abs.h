// Copyright 2026 FlagOS Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef FLAGDNN_TEST_FAKE_CANN_ACLNN_ABS_H_
#define FLAGDNN_TEST_FAKE_CANN_ACLNN_ABS_H_

#include <stdint.h>

#include "acl/acl.h"
#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnAbsGetWorkspaceSize(
    const aclTensor* self,
    aclTensor* out,
    uint64_t* workspace_size,
    aclOpExecutor** executor);
aclnnStatus aclnnAbs(
    void* workspace,
    uint64_t workspace_size,
    aclOpExecutor* executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // FLAGDNN_TEST_FAKE_CANN_ACLNN_ABS_H_
