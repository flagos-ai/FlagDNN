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

#include "aclnnop/aclnn_mean.h"
#include "aclnnop/aclnn_prod.h"
#include "aclnnop/aclnn_reduce_sum.h"
#include "common/oracle_common.h"

namespace flagdnn_test::ascend {
namespace {

constexpr uint64_t kMaxTensorRank = 8;

enum ReductionOperation : int32_t {
    kAdd = 0,
    kAverage = 1,
    kMultiply = 2,
};

struct PreparedReduction {
    int64_t input_shape[kMaxTensorRank] = {};
    int64_t input_strides[kMaxTensorRank] = {};
    int64_t output_shape[kMaxTensorRank] = {};
    int64_t output_strides[kMaxTensorRank] = {};
    int64_t dimensions[kMaxTensorRank] = {};
    uint64_t input_rank = 0;
    uint64_t output_rank = 0;
    uint64_t dimension_count = 0;
    aclTensor* input = nullptr;
    aclTensor* output = nullptr;
    aclIntArray* dimension_array = nullptr;
    ExecutorWorkspace execution;
    ExecuteFunction execute = nullptr;
    aclrtStream stream = nullptr;
};

void CopyMetadata(
    int64_t* destination,
    const int64_t* source,
    uint64_t count) noexcept {
    for (uint64_t index = 0; index < count; ++index) {
        destination[index] = source[index];
    }
}

void DestroyPreparedReduction(
    PreparedReduction* prepared,
    ErrorState* state) noexcept {
    DestroyExecutor(&prepared->execution, state);
    if (prepared->dimension_array != nullptr) {
        RecordAclFailure(
            state,
            "aclDestroyIntArray",
            aclDestroyIntArray(prepared->dimension_array));
        prepared->dimension_array = nullptr;
    }
    DestroyTensor(&prepared->input, "aclDestroyTensor(input)", state);
    DestroyTensor(&prepared->output, "aclDestroyTensor(output)", state);
    FreeWorkspace(&prepared->execution, state);
}

}  // namespace

extern "C" int flagdnn_aclnn_reduction_create(
    int32_t operation,
    const void* input_data,
    const int64_t* input_shape,
    const int64_t* input_strides,
    uint64_t input_rank,
    void* output_data,
    const int64_t* output_shape,
    const int64_t* output_strides,
    uint64_t output_rank,
    int32_t dtype_code,
    const int64_t* dimensions,
    uint64_t dimension_count,
    bool keepdim,
    void* stream_handle,
    void** prepared_handle,
    char* error_buffer,
    size_t error_buffer_length) noexcept {
    ErrorState state{error_buffer, error_buffer_length, ACL_SUCCESS, false};
    ClearError(&state);
    if (prepared_handle == nullptr) {
        return ReturnFailure(&state, "prepared handle pointer is null", -80);
    }
    *prepared_handle = nullptr;
    if (input_data == nullptr || output_data == nullptr ||
        input_shape == nullptr || input_strides == nullptr ||
        output_shape == nullptr || output_strides == nullptr ||
        dimensions == nullptr || input_rank == 0 ||
        input_rank > kMaxTensorRank || output_rank == 0 ||
        output_rank > kMaxTensorRank || dimension_count == 0 ||
        dimension_count > input_rank || stream_handle == nullptr) {
        return ReturnFailure(&state, "invalid reduction arguments", -81);
    }
    aclDataType dtype = ACL_DT_UNDEFINED;
    if (!MapDataType(dtype_code, &dtype)) {
        return ReturnFailure(&state, "unsupported dtype code", -82);
    }
    PreparedReduction* prepared = new (std::nothrow) PreparedReduction;
    if (prepared == nullptr) {
        return ReturnFailure(
            &state, "failed to allocate prepared reduction", -83);
    }
    prepared->input_rank = input_rank;
    prepared->output_rank = output_rank;
    prepared->dimension_count = dimension_count;
    prepared->stream = reinterpret_cast<aclrtStream>(stream_handle);
    CopyMetadata(prepared->input_shape, input_shape, input_rank);
    CopyMetadata(prepared->input_strides, input_strides, input_rank);
    CopyMetadata(prepared->output_shape, output_shape, output_rank);
    CopyMetadata(prepared->output_strides, output_strides, output_rank);
    CopyMetadata(prepared->dimensions, dimensions, dimension_count);

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
    prepared->dimension_array =
        aclCreateIntArray(prepared->dimensions, prepared->dimension_count);
    if (prepared->input == nullptr || prepared->output == nullptr ||
        prepared->dimension_array == nullptr) {
        RecordWrapperFailure(
            &state, "failed to create reduction metadata", -84);
    }

    aclnnStatus status = ACL_SUCCESS;
    if (state.first_status == ACL_SUCCESS && operation == kAdd) {
        status = aclnnReduceSumGetWorkspaceSize(
            prepared->input,
            prepared->dimension_array,
            keepdim,
            dtype,
            prepared->output,
            &prepared->execution.workspace_size,
            &prepared->execution.executor);
        prepared->execute = aclnnReduceSum;
    } else if (state.first_status == ACL_SUCCESS && operation == kAverage) {
        status = aclnnMeanGetWorkspaceSize(
            prepared->input,
            prepared->dimension_array,
            keepdim,
            dtype,
            prepared->output,
            &prepared->execution.workspace_size,
            &prepared->execution.executor);
        prepared->execute = aclnnMean;
    } else if (state.first_status == ACL_SUCCESS && operation == kMultiply) {
        if (prepared->dimension_count != 1) {
            RecordWrapperFailure(
                &state, "aclnnProdDim requires exactly one dimension", -85);
        } else {
            status = aclnnProdDimGetWorkspaceSize(
                prepared->input,
                prepared->dimensions[0],
                keepdim,
                dtype,
                prepared->output,
                &prepared->execution.workspace_size,
                &prepared->execution.executor);
            prepared->execute = aclnnProdDim;
        }
    } else if (state.first_status == ACL_SUCCESS) {
        RecordWrapperFailure(
            &state, "unsupported reduction operation code", -86);
    }
    if (state.first_status == ACL_SUCCESS) {
        RecordAclFailure(
            &state, "reduction GetWorkspaceSize", status);
    }
    if (state.first_status == ACL_SUCCESS) {
        MakeExecutorRepeatable(&prepared->execution, &state);
        AllocateWorkspace(&prepared->execution, &state);
    }
    if (state.first_status != ACL_SUCCESS) {
        DestroyPreparedReduction(prepared, &state);
        delete prepared;
        return state.first_status;
    }
    *prepared_handle = prepared;
    return ACL_SUCCESS;
}

extern "C" int flagdnn_aclnn_reduction_run(
    void* prepared_handle,
    char* error_buffer,
    size_t error_buffer_length) noexcept {
    ErrorState state{error_buffer, error_buffer_length, ACL_SUCCESS, false};
    ClearError(&state);
    if (prepared_handle == nullptr) {
        return ReturnFailure(&state, "prepared reduction handle is null", -90);
    }
    PreparedReduction* prepared =
        static_cast<PreparedReduction*>(prepared_handle);
    RecordAclFailure(
        &state,
        "aclnn reduction execution",
        prepared->execute(
            prepared->execution.workspace,
            prepared->execution.workspace_size,
            prepared->execution.executor,
            prepared->stream));
    return state.first_status;
}

extern "C" int flagdnn_aclnn_reduction_destroy(
    void* prepared_handle,
    char* error_buffer,
    size_t error_buffer_length) noexcept {
    ErrorState state{error_buffer, error_buffer_length, ACL_SUCCESS, false};
    ClearError(&state);
    if (prepared_handle == nullptr) {
        return ReturnFailure(&state, "prepared reduction handle is null", -91);
    }
    PreparedReduction* prepared =
        static_cast<PreparedReduction*>(prepared_handle);
    RecordAclFailure(
        &state,
        "aclrtSynchronizeStream",
        aclrtSynchronizeStream(prepared->stream));
    DestroyPreparedReduction(prepared, &state);
    delete prepared;
    return state.first_status;
}

}  // namespace flagdnn_test::ascend
