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

#ifndef FLAGDNN_TEST_FAKE_CANN_ACL_H_
#define FLAGDNN_TEST_FAKE_CANN_ACL_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int aclError;
typedef void* aclrtStream;

typedef enum aclDataType {
    ACL_DT_UNDEFINED = -1,
    ACL_FLOAT = 0,
    ACL_FLOAT16 = 1,
    ACL_BF16 = 27,
} aclDataType;

typedef enum aclFormat {
    ACL_FORMAT_ND = 2,
} aclFormat;

typedef enum aclrtMemMallocPolicy {
    ACL_MEM_MALLOC_HUGE_FIRST,
    ACL_MEM_MALLOC_HUGE_ONLY,
    ACL_MEM_MALLOC_NORMAL_ONLY,
} aclrtMemMallocPolicy;

#define ACL_SUCCESS 0

const char* aclGetRecentErrMsg(void);
aclError aclrtMalloc(
    void** workspace, size_t size, aclrtMemMallocPolicy policy);
aclError aclrtFree(void* workspace);
aclError aclrtSynchronizeStream(aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // FLAGDNN_TEST_FAKE_CANN_ACL_H_
