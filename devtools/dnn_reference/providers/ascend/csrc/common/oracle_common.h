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

#ifndef FLAGDNN_TEST_ASCEND_ORACLE_COMMON_H_
#define FLAGDNN_TEST_ASCEND_ORACLE_COMMON_H_

#include <cstddef>
#include <cstdint>

#include "acl/acl.h"
#include "aclnn/acl_meta.h"

namespace flagdnn_test::ascend {

struct ErrorState {
    char* buffer;
    size_t length;
    int first_status;
    bool has_message;
};

struct ExecutorWorkspace {
    aclOpExecutor* executor = nullptr;
    bool executor_owned = false;
    void* workspace = nullptr;
    uint64_t workspace_size = 0;
};

using ExecuteFunction = aclnnStatus (*)(
    void*, uint64_t, aclOpExecutor*, aclrtStream);

void ClearError(ErrorState* state) noexcept;
void RecordWrapperFailure(
    ErrorState* state, const char* stage, int status) noexcept;
void RecordAclFailure(
    ErrorState* state, const char* stage, int status) noexcept;
int ReturnFailure(
    ErrorState* state, const char* stage, int status) noexcept;
bool MapDataType(int32_t code, aclDataType* result) noexcept;
bool MakeExecutorRepeatable(
    ExecutorWorkspace* resources, ErrorState* state) noexcept;
bool AllocateWorkspace(
    ExecutorWorkspace* resources, ErrorState* state) noexcept;
void ExecuteAndSynchronize(
    ExecutorWorkspace* resources,
    aclrtStream stream,
    ExecuteFunction execute,
    const char* execute_stage,
    ErrorState* state) noexcept;
void DestroyExecutor(
    ExecutorWorkspace* resources, ErrorState* state) noexcept;
void FreeWorkspace(
    ExecutorWorkspace* resources, ErrorState* state) noexcept;
void DestroyTensor(
    aclTensor** tensor, const char* stage, ErrorState* state) noexcept;
void DestroyScalar(
    aclScalar** scalar, const char* stage, ErrorState* state) noexcept;

}  // namespace flagdnn_test::ascend

#endif  // FLAGDNN_TEST_ASCEND_ORACLE_COMMON_H_
