#include <cstddef>
#include <cstdint>

#include "aclnnop/aclnn_add.h"
#include "common/oracle_common.h"

namespace flagdnn_test::ascend {

extern "C" int flagdnn_test_aclnn_add(
    const void* x_data,
    const int64_t* x_shape,
    const int64_t* x_strides,
    uint64_t x_rank,
    const void* y_data,
    const int64_t* y_shape,
    const int64_t* y_strides,
    uint64_t y_rank,
    void* output_data,
    const int64_t* output_shape,
    const int64_t* output_strides,
    uint64_t output_rank,
    int32_t dtype_code,
    double alpha_value,
    void* stream_handle,
    char* error_buffer,
    size_t error_buffer_length) noexcept {
    ErrorState state{error_buffer, error_buffer_length, ACL_SUCCESS, false};
    ClearError(&state);

    if (x_data == nullptr || y_data == nullptr || output_data == nullptr) {
        return ReturnFailure(
            &state, "input/output device pointer is null", -1);
    }
    if (x_rank == 0 || y_rank == 0 || output_rank == 0 ||
        x_shape == nullptr || x_strides == nullptr ||
        y_shape == nullptr || y_strides == nullptr ||
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

    aclTensor* x = nullptr;
    aclTensor* y = nullptr;
    aclTensor* output = nullptr;
    aclScalar* alpha = nullptr;
    ExecutorWorkspace execution;
    float alpha_data = static_cast<float>(alpha_value);

    x = aclCreateTensor(
        x_shape,
        x_rank,
        dtype,
        x_strides,
        0,
        ACL_FORMAT_ND,
        x_shape,
        x_rank,
        const_cast<void*>(x_data));
    if (x == nullptr) {
        RecordWrapperFailure(&state, "aclCreateTensor(x)", -5);
    }
    if (state.first_status == ACL_SUCCESS) {
        y = aclCreateTensor(
            y_shape,
            y_rank,
            dtype,
            y_strides,
            0,
            ACL_FORMAT_ND,
            y_shape,
            y_rank,
            const_cast<void*>(y_data));
        if (y == nullptr) {
            RecordWrapperFailure(&state, "aclCreateTensor(y)", -6);
        }
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
            RecordWrapperFailure(&state, "aclCreateTensor(output)", -7);
        }
    }
    if (state.first_status == ACL_SUCCESS) {
        alpha = aclCreateScalar(&alpha_data, ACL_FLOAT);
        if (alpha == nullptr) {
            RecordWrapperFailure(&state, "aclCreateScalar(alpha)", -8);
        }
    }

    if (state.first_status == ACL_SUCCESS) {
        const int status = aclnnAddGetWorkspaceSize(
            x,
            y,
            alpha,
            output,
            &execution.workspace_size,
            &execution.executor);
        RecordAclFailure(&state, "aclnnAddGetWorkspaceSize", status);
    }

    MakeExecutorRepeatable(&execution, &state);
    AllocateWorkspace(&execution, &state);
    ExecuteAndSynchronize(
        &execution,
        reinterpret_cast<aclrtStream>(stream_handle),
        aclnnAdd,
        "aclnnAdd",
        &state);
    DestroyExecutor(&execution, &state);
    DestroyTensor(&x, "aclDestroyTensor(x)", &state);
    DestroyTensor(&y, "aclDestroyTensor(y)", &state);
    DestroyTensor(&output, "aclDestroyTensor(output)", &state);
    DestroyScalar(&alpha, "aclDestroyScalar", &state);
    FreeWorkspace(&execution, &state);
    return state.first_status;
}

}  // namespace flagdnn_test::ascend
