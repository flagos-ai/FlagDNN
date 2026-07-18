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

#ifndef FLAGDNN_TEST_FAKE_CANN_ACL_META_H_
#define FLAGDNN_TEST_FAKE_CANN_ACL_META_H_

#include <stdint.h>

#include "acl/acl.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct aclTensor aclTensor;
typedef struct aclScalar aclScalar;
typedef struct aclOpExecutor aclOpExecutor;
typedef int32_t aclnnStatus;

aclTensor* aclCreateTensor(
    const int64_t* view_dims,
    uint64_t view_dim_num,
    aclDataType data_type,
    const int64_t* strides,
    int64_t storage_offset,
    aclFormat format,
    const int64_t* storage_dims,
    uint64_t storage_dim_num,
    void* data);
aclScalar* aclCreateScalar(void* value, aclDataType data_type);
aclnnStatus aclDestroyTensor(const aclTensor* tensor);
aclnnStatus aclDestroyScalar(const aclScalar* scalar);
aclnnStatus aclSetAclOpExecutorRepeatable(aclOpExecutor* executor);
aclnnStatus aclDestroyAclOpExecutor(aclOpExecutor* executor);

#ifdef __cplusplus
}
#endif

#endif  // FLAGDNN_TEST_FAKE_CANN_ACL_META_H_
