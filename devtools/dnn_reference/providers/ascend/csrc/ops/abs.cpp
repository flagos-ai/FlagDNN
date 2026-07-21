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

#include <cstddef>
#include <cstdint>
#include <new>

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

namespace {

constexpr uint64_t kMaxTensorRank = 8;

struct PreparedAbs {
    int64_t input_shape[kMaxTensorRank] = {};
    int64_t input_strides[kMaxTensorRank] = {};
    int64_t output_shape[kMaxTensorRank] = {};
    int64_t output_strides[kMaxTensorRank] = {};
    uint64_t input_rank = 0;
    uint64_t output_rank = 0;
    aclTensor* input = nullptr;
    aclTensor* output = nullptr;
    ExecutorWorkspace execution;
    aclrtStream stream = nullptr;
};

void CopyMetadata(
    int64_t* destination,
    const int64_t* source,
    uint64_t rank) noexcept {
    for (uint64_t index = 0; index < rank; ++index) {
        destination[index] = source[index];
    }
}

bool ValidMetadata(
    const int64_t* shape,
    const int64_t* strides,
    uint64_t rank) noexcept {
    return shape != nullptr && strides != nullptr && rank != 0 &&
        rank <= kMaxTensorRank;
}

void DestroyPreparedAbs(
    PreparedAbs* prepared,
    ErrorState* state) noexcept {
    DestroyExecutor(&prepared->execution, state);
    DestroyTensor(&prepared->input, "aclDestroyTensor(x)", state);
    DestroyTensor(
        &prepared->output, "aclDestroyTensor(output)", state);
    FreeWorkspace(&prepared->execution, state);
}

}  // namespace

extern "C" int flagdnn_aclnn_abs_create(
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
    void** prepared_handle,
    char* error_buffer,
    size_t error_buffer_length) noexcept {
    ErrorState state{error_buffer, error_buffer_length, ACL_SUCCESS, false};
    ClearError(&state);
    if (prepared_handle == nullptr) {
        return ReturnFailure(&state, "prepared handle pointer is null", -20);
    }
    *prepared_handle = nullptr;
    if (input_data == nullptr || output_data == nullptr) {
        return ReturnFailure(
            &state, "input/output device pointer is null", -21);
    }
    if (!ValidMetadata(input_shape, input_strides, input_rank) ||
        !ValidMetadata(output_shape, output_strides, output_rank)) {
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
    PreparedAbs* prepared = new (std::nothrow) PreparedAbs;
    if (prepared == nullptr) {
        return ReturnFailure(&state, "failed to allocate prepared Abs", -25);
    }
    prepared->input_rank = input_rank;
    prepared->output_rank = output_rank;
    prepared->stream = reinterpret_cast<aclrtStream>(stream_handle);
    CopyMetadata(prepared->input_shape, input_shape, input_rank);
    CopyMetadata(prepared->input_strides, input_strides, input_rank);
    CopyMetadata(prepared->output_shape, output_shape, output_rank);
    CopyMetadata(prepared->output_strides, output_strides, output_rank);

    prepared->input = aclCreateTensor(
        prepared->input_shape,
        prepared->input_rank,
        dtype,
        prepared->input_strides,
        0,
        ACL_FORMAT_ND,
        prepared->input_shape,
        prepared->input_rank,
        const_cast<void*>(input_data));
    if (prepared->input == nullptr) {
        RecordWrapperFailure(&state, "aclCreateTensor(x)", -26);
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
            RecordWrapperFailure(
                &state, "aclCreateTensor(output)", -27);
        }
    }
    if (state.first_status == ACL_SUCCESS) {
        const int status = aclnnAbsGetWorkspaceSize(
            prepared->input,
            prepared->output,
            &prepared->execution.workspace_size,
            &prepared->execution.executor);
        RecordAclFailure(&state, "aclnnAbsGetWorkspaceSize", status);
    }
    MakeExecutorRepeatable(&prepared->execution, &state);
    AllocateWorkspace(&prepared->execution, &state);

    if (state.first_status != ACL_SUCCESS) {
        DestroyPreparedAbs(prepared, &state);
        delete prepared;
        return state.first_status;
    }
    *prepared_handle = prepared;
    return ACL_SUCCESS;
}

extern "C" int flagdnn_aclnn_abs_run(
    void* prepared_handle,
    char* error_buffer,
    size_t error_buffer_length) noexcept {
    ErrorState state{error_buffer, error_buffer_length, ACL_SUCCESS, false};
    ClearError(&state);
    if (prepared_handle == nullptr) {
        return ReturnFailure(&state, "prepared Abs handle is null", -30);
    }
    PreparedAbs* prepared = static_cast<PreparedAbs*>(prepared_handle);
    const int status = aclnnAbs(
        prepared->execution.workspace,
        prepared->execution.workspace_size,
        prepared->execution.executor,
        prepared->stream);
    RecordAclFailure(&state, "aclnnAbs", status);
    return state.first_status;
}

extern "C" int flagdnn_aclnn_abs_destroy(
    void* prepared_handle,
    char* error_buffer,
    size_t error_buffer_length) noexcept {
    ErrorState state{error_buffer, error_buffer_length, ACL_SUCCESS, false};
    ClearError(&state);
    if (prepared_handle == nullptr) {
        return ReturnFailure(&state, "prepared Abs handle is null", -31);
    }
    PreparedAbs* prepared = static_cast<PreparedAbs*>(prepared_handle);
    RecordAclFailure(
        &state,
        "aclrtSynchronizeStream",
        aclrtSynchronizeStream(prepared->stream));
    DestroyPreparedAbs(prepared, &state);
    delete prepared;
    return state.first_status;
}

}  // namespace flagdnn_test::ascend
