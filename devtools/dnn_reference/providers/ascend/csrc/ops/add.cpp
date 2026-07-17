#include <cstddef>
#include <cstdint>
#include <new>

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

namespace {

constexpr uint64_t kMaxTensorRank = 8;

struct PreparedAdd {
    int64_t x_shape[kMaxTensorRank] = {};
    int64_t x_strides[kMaxTensorRank] = {};
    int64_t y_shape[kMaxTensorRank] = {};
    int64_t y_strides[kMaxTensorRank] = {};
    int64_t output_shape[kMaxTensorRank] = {};
    int64_t output_strides[kMaxTensorRank] = {};
    uint64_t x_rank = 0;
    uint64_t y_rank = 0;
    uint64_t output_rank = 0;
    aclTensor* x = nullptr;
    aclTensor* y = nullptr;
    aclTensor* output = nullptr;
    aclScalar* alpha = nullptr;
    ExecutorWorkspace execution;
    aclrtStream stream = nullptr;
    float alpha_data = 0.0F;
};

void CopyMetadata(
    int64_t* destination,
    const int64_t* source,
    uint64_t rank) noexcept {
    for (uint64_t index = 0; index < rank; ++index) {
        destination[index] = source[index];
    }
}

void DestroyPreparedAdd(
    PreparedAdd* prepared, ErrorState* state) noexcept {
    DestroyExecutor(&prepared->execution, state);
    DestroyTensor(&prepared->x, "aclDestroyTensor(x)", state);
    DestroyTensor(&prepared->y, "aclDestroyTensor(y)", state);
    DestroyTensor(
        &prepared->output, "aclDestroyTensor(output)", state);
    DestroyScalar(&prepared->alpha, "aclDestroyScalar", state);
    FreeWorkspace(&prepared->execution, state);
}

bool ValidPreparedMetadata(
    const int64_t* shape,
    const int64_t* strides,
    uint64_t rank) noexcept {
    return shape != nullptr && strides != nullptr && rank != 0 &&
        rank <= kMaxTensorRank;
}

}  // namespace

extern "C" int flagdnn_aclnn_add_create(
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
    void** prepared_handle,
    char* error_buffer,
    size_t error_buffer_length) noexcept {
    ErrorState state{error_buffer, error_buffer_length, ACL_SUCCESS, false};
    ClearError(&state);
    if (prepared_handle == nullptr) {
        return ReturnFailure(&state, "prepared handle pointer is null", -20);
    }
    *prepared_handle = nullptr;
    if (x_data == nullptr || y_data == nullptr || output_data == nullptr) {
        return ReturnFailure(
            &state, "input/output device pointer is null", -21);
    }
    if (!ValidPreparedMetadata(x_shape, x_strides, x_rank) ||
        !ValidPreparedMetadata(y_shape, y_strides, y_rank) ||
        !ValidPreparedMetadata(
            output_shape, output_strides, output_rank)) {
        return ReturnFailure(
            &state, "invalid shape/stride metadata or tensor rank", -22);
    }
    if (stream_handle == nullptr) {
        return ReturnFailure(&state, "current stream pointer is null", -23);
    }

    aclDataType dtype = ACL_DT_UNDEFINED;
    if (!MapDataType(dtype_code, &dtype)) {
        return ReturnFailure(&state, "unsupported dtype code", -24);
    }

    PreparedAdd* prepared = new (std::nothrow) PreparedAdd;
    if (prepared == nullptr) {
        return ReturnFailure(&state, "failed to allocate prepared Add", -25);
    }
    prepared->x_rank = x_rank;
    prepared->y_rank = y_rank;
    prepared->output_rank = output_rank;
    prepared->stream = reinterpret_cast<aclrtStream>(stream_handle);
    prepared->alpha_data = static_cast<float>(alpha_value);
    CopyMetadata(prepared->x_shape, x_shape, x_rank);
    CopyMetadata(prepared->x_strides, x_strides, x_rank);
    CopyMetadata(prepared->y_shape, y_shape, y_rank);
    CopyMetadata(prepared->y_strides, y_strides, y_rank);
    CopyMetadata(prepared->output_shape, output_shape, output_rank);
    CopyMetadata(prepared->output_strides, output_strides, output_rank);

    prepared->x = aclCreateTensor(
        prepared->x_shape,
        prepared->x_rank,
        dtype,
        prepared->x_strides,
        0,
        ACL_FORMAT_ND,
        prepared->x_shape,
        prepared->x_rank,
        const_cast<void*>(x_data));
    if (prepared->x == nullptr) {
        RecordWrapperFailure(&state, "aclCreateTensor(x)", -26);
    }
    if (state.first_status == ACL_SUCCESS) {
        prepared->y = aclCreateTensor(
            prepared->y_shape,
            prepared->y_rank,
            dtype,
            prepared->y_strides,
            0,
            ACL_FORMAT_ND,
            prepared->y_shape,
            prepared->y_rank,
            const_cast<void*>(y_data));
        if (prepared->y == nullptr) {
            RecordWrapperFailure(&state, "aclCreateTensor(y)", -27);
        }
    }
    if (state.first_status == ACL_SUCCESS) {
        prepared->output = aclCreateTensor(
            prepared->output_shape,
            prepared->output_rank,
            dtype,
            prepared->output_strides,
            0,
            ACL_FORMAT_ND,
            prepared->output_shape,
            prepared->output_rank,
            output_data);
        if (prepared->output == nullptr) {
            RecordWrapperFailure(&state, "aclCreateTensor(output)", -28);
        }
    }
    if (state.first_status == ACL_SUCCESS) {
        prepared->alpha = aclCreateScalar(
            &prepared->alpha_data, ACL_FLOAT);
        if (prepared->alpha == nullptr) {
            RecordWrapperFailure(&state, "aclCreateScalar(alpha)", -29);
        }
    }
    if (state.first_status == ACL_SUCCESS) {
        const int status = aclnnAddGetWorkspaceSize(
            prepared->x,
            prepared->y,
            prepared->alpha,
            prepared->output,
            &prepared->execution.workspace_size,
            &prepared->execution.executor);
        RecordAclFailure(&state, "aclnnAddGetWorkspaceSize", status);
    }
    MakeExecutorRepeatable(&prepared->execution, &state);
    AllocateWorkspace(&prepared->execution, &state);

    if (state.first_status != ACL_SUCCESS) {
        DestroyPreparedAdd(prepared, &state);
        delete prepared;
        return state.first_status;
    }
    *prepared_handle = prepared;
    return ACL_SUCCESS;
}

extern "C" int flagdnn_aclnn_add_run(
    void* prepared_handle,
    char* error_buffer,
    size_t error_buffer_length) noexcept {
    ErrorState state{error_buffer, error_buffer_length, ACL_SUCCESS, false};
    ClearError(&state);
    if (prepared_handle == nullptr) {
        return ReturnFailure(&state, "prepared Add handle is null", -30);
    }
    PreparedAdd* prepared = static_cast<PreparedAdd*>(prepared_handle);
    const int status = aclnnAdd(
        prepared->execution.workspace,
        prepared->execution.workspace_size,
        prepared->execution.executor,
        prepared->stream);
    RecordAclFailure(&state, "aclnnAdd", status);
    return state.first_status;
}

extern "C" int flagdnn_aclnn_add_destroy(
    void* prepared_handle,
    char* error_buffer,
    size_t error_buffer_length) noexcept {
    ErrorState state{error_buffer, error_buffer_length, ACL_SUCCESS, false};
    ClearError(&state);
    if (prepared_handle == nullptr) {
        return ReturnFailure(&state, "prepared Add handle is null", -31);
    }
    PreparedAdd* prepared = static_cast<PreparedAdd*>(prepared_handle);
    RecordAclFailure(
        &state,
        "aclrtSynchronizeStream",
        aclrtSynchronizeStream(prepared->stream));
    DestroyPreparedAdd(prepared, &state);
    delete prepared;
    return state.first_status;
}

}  // namespace flagdnn_test::ascend
