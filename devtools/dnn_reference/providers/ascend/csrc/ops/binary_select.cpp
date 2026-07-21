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

#include "aclnnop/aclnn_s_where.h"
#include "common/oracle_common.h"

namespace flagdnn_test::ascend {
namespace {

constexpr uint64_t kMaxTensorRank = 8;
constexpr size_t kTensorCount = 4;

struct PreparedBinarySelect {
    int64_t shapes[kTensorCount][kMaxTensorRank] = {};
    int64_t strides[kTensorCount][kMaxTensorRank] = {};
    uint64_t ranks[kTensorCount] = {};
    aclTensor* tensors[kTensorCount] = {};
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

void SetMetadata(
    PreparedBinarySelect* prepared,
    size_t index,
    const int64_t* shape,
    const int64_t* strides,
    uint64_t rank) noexcept {
    prepared->ranks[index] = rank;
    CopyMetadata(prepared->shapes[index], shape, rank);
    CopyMetadata(prepared->strides[index], strides, rank);
}

aclTensor* CreateTensor(
    PreparedBinarySelect* prepared,
    size_t index,
    aclDataType dtype,
    void* data) noexcept {
    return aclCreateTensor(
        prepared->shapes[index],
        prepared->ranks[index],
        dtype,
        prepared->strides[index],
        0,
        ACL_FORMAT_ND,
        prepared->shapes[index],
        prepared->ranks[index],
        data);
}

void DestroyPrepared(
    PreparedBinarySelect* prepared,
    ErrorState* state) noexcept {
    static constexpr const char* kDestroyStages[kTensorCount] = {
        "aclDestroyTensor(x)",
        "aclDestroyTensor(y)",
        "aclDestroyTensor(mask)",
        "aclDestroyTensor(output)",
    };
    DestroyExecutor(&prepared->execution, state);
    for (size_t index = 0; index < kTensorCount; ++index) {
        DestroyTensor(
            &prepared->tensors[index], kDestroyStages[index], state);
    }
    FreeWorkspace(&prepared->execution, state);
}

}  // namespace

extern "C" int flagdnn_aclnn_binary_select_create(
    const void* x_data,
    const int64_t* x_shape,
    const int64_t* x_strides,
    uint64_t x_rank,
    const void* y_data,
    const int64_t* y_shape,
    const int64_t* y_strides,
    uint64_t y_rank,
    const void* mask_data,
    const int64_t* mask_shape,
    const int64_t* mask_strides,
    uint64_t mask_rank,
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
        return ReturnFailure(
            &state, "prepared handle pointer is null", -70);
    }
    *prepared_handle = nullptr;
    if (x_data == nullptr || y_data == nullptr || mask_data == nullptr ||
        output_data == nullptr) {
        return ReturnFailure(
            &state, "input/output device pointer is null", -71);
    }
    if (!ValidMetadata(x_shape, x_strides, x_rank) ||
        !ValidMetadata(y_shape, y_strides, y_rank) ||
        !ValidMetadata(mask_shape, mask_strides, mask_rank) ||
        !ValidMetadata(output_shape, output_strides, output_rank)) {
        return ReturnFailure(&state, "invalid tensor metadata", -72);
    }
    if (stream_handle == nullptr) {
        return ReturnFailure(
            &state, "current stream pointer is null", -73);
    }
    aclDataType dtype = ACL_DT_UNDEFINED;
    if (!MapDataType(dtype_code, &dtype) || dtype == ACL_BOOL) {
        return ReturnFailure(&state, "unsupported data dtype code", -74);
    }

    PreparedBinarySelect* prepared =
        new (std::nothrow) PreparedBinarySelect;
    if (prepared == nullptr) {
        return ReturnFailure(
            &state, "failed to allocate prepared binary_select", -75);
    }
    prepared->stream = reinterpret_cast<aclrtStream>(stream_handle);
    SetMetadata(prepared, 0, x_shape, x_strides, x_rank);
    SetMetadata(prepared, 1, y_shape, y_strides, y_rank);
    SetMetadata(prepared, 2, mask_shape, mask_strides, mask_rank);
    SetMetadata(
        prepared, 3, output_shape, output_strides, output_rank);

    void* data[kTensorCount] = {
        const_cast<void*>(x_data),
        const_cast<void*>(y_data),
        const_cast<void*>(mask_data),
        output_data,
    };
    aclDataType dtypes[kTensorCount] = {
        dtype, dtype, ACL_BOOL, dtype};
    static constexpr const char* kCreateStages[kTensorCount] = {
        "aclCreateTensor(x)",
        "aclCreateTensor(y)",
        "aclCreateTensor(mask)",
        "aclCreateTensor(output)",
    };
    for (size_t index = 0;
         index < kTensorCount && state.first_status == ACL_SUCCESS;
         ++index) {
        prepared->tensors[index] =
            CreateTensor(prepared, index, dtypes[index], data[index]);
        if (prepared->tensors[index] == nullptr) {
            RecordWrapperFailure(
                &state, kCreateStages[index], -76 - index);
        }
    }

    if (state.first_status == ACL_SUCCESS) {
        const aclnnStatus status = aclnnSWhereGetWorkspaceSize(
            prepared->tensors[2],
            prepared->tensors[0],
            prepared->tensors[1],
            prepared->tensors[3],
            &prepared->execution.workspace_size,
            &prepared->execution.executor);
        RecordAclFailure(
            &state, "aclnnSWhereGetWorkspaceSize", status);
    }
    MakeExecutorRepeatable(&prepared->execution, &state);
    AllocateWorkspace(&prepared->execution, &state);
    if (state.first_status != ACL_SUCCESS) {
        DestroyPrepared(prepared, &state);
        delete prepared;
        return state.first_status;
    }
    *prepared_handle = prepared;
    return ACL_SUCCESS;
}

extern "C" int flagdnn_aclnn_binary_select_run(
    void* prepared_handle,
    char* error_buffer,
    size_t error_buffer_length) noexcept {
    ErrorState state{error_buffer, error_buffer_length, ACL_SUCCESS, false};
    ClearError(&state);
    if (prepared_handle == nullptr) {
        return ReturnFailure(
            &state, "prepared binary_select handle is null", -81);
    }
    PreparedBinarySelect* prepared =
        static_cast<PreparedBinarySelect*>(prepared_handle);
    RecordAclFailure(
        &state,
        "aclnnSWhere",
        aclnnSWhere(
            prepared->execution.workspace,
            prepared->execution.workspace_size,
            prepared->execution.executor,
            prepared->stream));
    return state.first_status;
}

extern "C" int flagdnn_aclnn_binary_select_destroy(
    void* prepared_handle,
    char* error_buffer,
    size_t error_buffer_length) noexcept {
    ErrorState state{error_buffer, error_buffer_length, ACL_SUCCESS, false};
    ClearError(&state);
    if (prepared_handle == nullptr) {
        return ACL_SUCCESS;
    }
    PreparedBinarySelect* prepared =
        static_cast<PreparedBinarySelect*>(prepared_handle);
    RecordAclFailure(
        &state,
        "aclrtSynchronizeStream",
        aclrtSynchronizeStream(prepared->stream));
    DestroyPrepared(prepared, &state);
    delete prepared;
    return state.first_status;
}

}  // namespace flagdnn_test::ascend
