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

#include "common/oracle_common.h"

#include <cstdio>

namespace flagdnn_test::ascend {
namespace {

constexpr int32_t kFloat16 = 0;
constexpr int32_t kBFloat16 = 1;
constexpr int32_t kFloat32 = 2;

void AppendText(ErrorState* state, const char* text) noexcept {
    if (state->buffer == nullptr || state->length == 0 || text == nullptr) {
        return;
    }

    size_t used = 0;
    while (used < state->length && state->buffer[used] != '\0') {
        ++used;
    }
    if (used == state->length) {
        return;
    }
    std::snprintf(state->buffer + used, state->length - used, "%s", text);
}

void RecordFailure(
    ErrorState* state,
    const char* stage,
    int status,
    bool include_recent_error) noexcept {
    if (status == ACL_SUCCESS) {
        return;
    }
    if (state->first_status == ACL_SUCCESS) {
        state->first_status = status;
    }
    if (state->has_message) {
        AppendText(state, "; ");
    }
    state->has_message = true;

    char status_text[32] = {};
    std::snprintf(status_text, sizeof(status_text), "%d", status);
    AppendText(state, stage);
    AppendText(state, " failed, status=");
    AppendText(state, status_text);

    if (!include_recent_error) {
        return;
    }
    const char* recent_error = aclGetRecentErrMsg();
    if (recent_error != nullptr && recent_error[0] != '\0') {
        AppendText(state, ", recent_error=");
        AppendText(state, recent_error);
    }
}

}  // namespace

void ClearError(ErrorState* state) noexcept {
    state->first_status = ACL_SUCCESS;
    state->has_message = false;
    if (state->buffer != nullptr && state->length != 0) {
        state->buffer[0] = '\0';
    }
}

void RecordWrapperFailure(
    ErrorState* state, const char* stage, int status) noexcept {
    RecordFailure(state, stage, status, false);
}

void RecordAclFailure(
    ErrorState* state, const char* stage, int status) noexcept {
    RecordFailure(state, stage, status, true);
}

int ReturnFailure(ErrorState* state, const char* stage, int status) noexcept {
    RecordWrapperFailure(state, stage, status);
    return state->first_status;
}

bool MapDataType(int32_t code, aclDataType* result) noexcept {
    switch (code) {
        case kFloat16:
            *result = ACL_FLOAT16;
            return true;
        case kBFloat16:
            *result = ACL_BF16;
            return true;
        case kFloat32:
            *result = ACL_FLOAT;
            return true;
        default:
            return false;
    }
}

bool MakeExecutorRepeatable(
    ExecutorWorkspace* resources, ErrorState* state) noexcept {
    if (state->first_status != ACL_SUCCESS) {
        return false;
    }
    const int status = aclSetAclOpExecutorRepeatable(resources->executor);
    if (status != ACL_SUCCESS) {
        RecordAclFailure(state, "aclSetAclOpExecutorRepeatable", status);
        return false;
    }
    resources->executor_owned = true;
    return true;
}

bool AllocateWorkspace(
    ExecutorWorkspace* resources, ErrorState* state) noexcept {
    if (state->first_status != ACL_SUCCESS) {
        return false;
    }
    if (resources->workspace_size == 0) {
        return true;
    }

    void* workspace = nullptr;
    const int status = aclrtMalloc(
        &workspace,
        resources->workspace_size,
        ACL_MEM_MALLOC_HUGE_FIRST);
    if (status != ACL_SUCCESS) {
        RecordAclFailure(state, "aclrtMalloc(workspace)", status);
        return false;
    }
    resources->workspace = workspace;
    return true;
}

void ExecuteAndSynchronize(
    ExecutorWorkspace* resources,
    aclrtStream stream,
    ExecuteFunction execute,
    const char* execute_stage,
    ErrorState* state) noexcept {
    if (state->first_status != ACL_SUCCESS) {
        return;
    }

    int status = execute(
        resources->workspace,
        resources->workspace_size,
        resources->executor,
        stream);
    RecordAclFailure(state, execute_stage, status);

    status = aclrtSynchronizeStream(stream);
    RecordAclFailure(state, "aclrtSynchronizeStream", status);
}

void DestroyExecutor(
    ExecutorWorkspace* resources, ErrorState* state) noexcept {
    if (!resources->executor_owned || resources->executor == nullptr) {
        return;
    }
    RecordAclFailure(
        state,
        "aclDestroyAclOpExecutor",
        aclDestroyAclOpExecutor(resources->executor));
    resources->executor = nullptr;
    resources->executor_owned = false;
}

void FreeWorkspace(
    ExecutorWorkspace* resources, ErrorState* state) noexcept {
    if (resources->workspace == nullptr) {
        return;
    }
    RecordAclFailure(
        state, "aclrtFree(workspace)", aclrtFree(resources->workspace));
    resources->workspace = nullptr;
}

void DestroyTensor(
    aclTensor** tensor, const char* stage, ErrorState* state) noexcept {
    if (*tensor == nullptr) {
        return;
    }
    RecordAclFailure(state, stage, aclDestroyTensor(*tensor));
    *tensor = nullptr;
}

void DestroyScalar(
    aclScalar** scalar, const char* stage, ErrorState* state) noexcept {
    if (*scalar == nullptr) {
        return;
    }
    RecordAclFailure(state, stage, aclDestroyScalar(*scalar));
    *scalar = nullptr;
}

}  // namespace flagdnn_test::ascend
