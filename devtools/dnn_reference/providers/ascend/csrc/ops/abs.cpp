#include <cstddef>
#include <cstdint>

#include "aclnnop/aclnn_abs.h"
#include "common/oracle_common.h"

namespace flagdnn_test::ascend {

extern "C" int flagdnn_test_aclnn_abs(
    const void* input_data,
    const int64_t* input_shape,
    const int64_t* input_strides,
    uint64_t input_rank,
    void* output_data,
    const int64_t* output_shape,
    const int64_t* output_strides,
    uint64_t output_rank,
    int32_t dtype_code,
    void* stream_handle,
    char* error_buffer,
    size_t error_buffer_length) noexcept {
    ErrorState state{error_buffer, error_buffer_length, ACL_SUCCESS, false};
    ClearError(&state);

    if (input_data == nullptr || output_data == nullptr) {
        return ReturnFailure(
            &state, "input/output device pointer is null", -1);
    }
    if (input_rank == 0 || output_rank == 0 ||
        input_shape == nullptr || input_strides == nullptr ||
        output_shape == nullptr || output_strides == nullptr) {
        return ReturnFailure(
            &state, "shape/stride metadata is null or rank is zero", -2);
    }
    if (stream_handle == nullptr) {
        return ReturnFailure(
            &state, "torch_npu current stream pointer is null", -3);
    }

    aclDataType dtype = ACL_DT_UNDEFINED;
    if (!MapDataType(dtype_code, &dtype)) {
        return ReturnFailure(&state, "unsupported dtype code", -4);
    }

    aclTensor* input = nullptr;
    aclTensor* output = nullptr;
    ExecutorWorkspace execution;

    input = aclCreateTensor(
        input_shape,
        input_rank,
        dtype,
        input_strides,
        0,
        ACL_FORMAT_ND,
        input_shape,
        input_rank,
        const_cast<void*>(input_data));
    if (input == nullptr) {
        RecordWrapperFailure(&state, "aclCreateTensor(x)", -5);
    }
    if (state.first_status == ACL_SUCCESS) {
        output = aclCreateTensor(
            output_shape,
            output_rank,
            dtype,
            output_strides,
            0,
            ACL_FORMAT_ND,
            output_shape,
            output_rank,
            output_data);
        if (output == nullptr) {
            RecordWrapperFailure(
                &state, "aclCreateTensor(output)", -6);
        }
    }

    if (state.first_status == ACL_SUCCESS) {
        const int status = aclnnAbsGetWorkspaceSize(
            input,
            output,
            &execution.workspace_size,
            &execution.executor);
        RecordAclFailure(&state, "aclnnAbsGetWorkspaceSize", status);
    }

    MakeExecutorRepeatable(&execution, &state);
    AllocateWorkspace(&execution, &state);
    ExecuteAndSynchronize(
        &execution,
        reinterpret_cast<aclrtStream>(stream_handle),
        aclnnAbs,
        "aclnnAbs",
        &state);
    DestroyExecutor(&execution, &state);
    DestroyTensor(&input, "aclDestroyTensor(x)", &state);
    DestroyTensor(&output, "aclDestroyTensor(output)", &state);
    FreeWorkspace(&execution, &state);
    return state.first_status;
}

}  // namespace flagdnn_test::ascend
