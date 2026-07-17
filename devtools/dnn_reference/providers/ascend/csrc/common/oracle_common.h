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
