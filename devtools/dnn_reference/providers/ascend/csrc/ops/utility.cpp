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

#include "aclnnop/aclnn_arange.h"
#include "aclnnop/aclnn_cast.h"
#include "aclnnop/aclnn_cat.h"
#include "aclnnop/aclnn_expand.h"
#include "aclnnop/aclnn_permute.h"
#include "aclnnop/aclnn_slice_v2.h"
#include "common/oracle_common.h"

namespace flagdnn_test::ascend {
namespace {

constexpr uint64_t kMaxTensorRank = 8;
constexpr uint64_t kMaxInputs = 3;
constexpr uint64_t kMaxParameters = 4 * kMaxTensorRank;
constexpr uint64_t kMaxExecutions = 2;

enum UtilityOperation : int32_t {
    kIdentity = 0,
    kTranspose = 1,
    kSlice = 2,
    kConcatenate = 3,
    kGenIndex = 4,
};

struct PreparedUtility {
    int32_t operation = -1;
    aclDataType dtype = ACL_DT_UNDEFINED;
    int64_t input_shapes[kMaxInputs][kMaxTensorRank] = {};
    int64_t input_strides[kMaxInputs][kMaxTensorRank] = {};
    uint64_t input_ranks[kMaxInputs] = {};
    int64_t output_shape[kMaxTensorRank] = {};
    int64_t output_strides[kMaxTensorRank] = {};
    int64_t gen_index_shape[kMaxTensorRank] = {};
    int64_t gen_index_strides[kMaxTensorRank] = {};
    uint64_t output_rank = 0;
    int64_t parameters[kMaxParameters] = {};
    uint64_t parameter_count = 0;
    int64_t scalar_values[3] = {0, 0, 1};
    aclTensor* inputs[kMaxInputs] = {};
    void* input_data[kMaxInputs] = {};
    aclTensor* gen_index_view = nullptr;
    aclTensor* output = nullptr;
    aclIntArray* arrays[4] = {};
    aclScalar* scalars[3] = {};
    aclTensorList* tensor_list = nullptr;
    ExecutorWorkspace executions[kMaxExecutions];
    ExecuteFunction execute_functions[kMaxExecutions] = {};
    uint64_t execution_count = 0;
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

aclTensor* CreateTensor(
    const int64_t* shape,
    const int64_t* strides,
    uint64_t rank,
    aclDataType dtype,
    void* data) noexcept {
    return aclCreateTensor(
        shape,
        rank,
        dtype,
        strides,
        0,
        ACL_FORMAT_ND,
        shape,
        rank,
        data);
}

void DestroyIntArray(
    aclIntArray** array,
    const char* stage,
    ErrorState* state) noexcept {
    if (*array == nullptr) {
        return;
    }
    RecordAclFailure(state, stage, aclDestroyIntArray(*array));
    *array = nullptr;
}

void DestroyTensorList(
    aclTensorList** list,
    ErrorState* state) noexcept {
    if (*list == nullptr) {
        return;
    }
    RecordAclFailure(
        state, "aclDestroyTensorList", aclDestroyTensorList(*list));
    *list = nullptr;
}

void DestroyPreparedUtility(
    PreparedUtility* prepared,
    ErrorState* state) noexcept {
    for (uint64_t index = 0; index < prepared->execution_count; ++index) {
        DestroyExecutor(&prepared->executions[index], state);
    }
    DestroyTensorList(&prepared->tensor_list, state);
    for (uint64_t index = 0; index < 4; ++index) {
        DestroyIntArray(
            &prepared->arrays[index], "aclDestroyIntArray", state);
    }
    for (uint64_t index = 0; index < 3; ++index) {
        DestroyScalar(
            &prepared->scalars[index], "aclDestroyScalar", state);
    }
    for (uint64_t index = 0; index < kMaxInputs; ++index) {
        DestroyTensor(
            &prepared->inputs[index], "aclDestroyTensor(input)", state);
    }
    DestroyTensor(
        &prepared->gen_index_view,
        "aclDestroyTensor(gen_index_view)",
        state);
    DestroyTensor(
        &prepared->output, "aclDestroyTensor(output)", state);
    for (uint64_t index = 0; index < prepared->execution_count; ++index) {
        FreeWorkspace(&prepared->executions[index], state);
    }
}

bool CreateArray(
    PreparedUtility* prepared,
    uint64_t array_index,
    const int64_t* values,
    uint64_t count,
    ErrorState* state) noexcept {
    prepared->arrays[array_index] = aclCreateIntArray(values, count);
    if (prepared->arrays[array_index] == nullptr) {
        RecordWrapperFailure(state, "aclCreateIntArray", -50);
        return false;
    }
    return true;
}

void PrepareExecution(
    PreparedUtility* prepared,
    uint64_t index,
    aclnnStatus status,
    ExecuteFunction execute,
    const char* stage,
    ErrorState* state) noexcept {
    RecordAclFailure(state, stage, status);
    if (state->first_status != ACL_SUCCESS) {
        return;
    }
    prepared->execute_functions[index] = execute;
    prepared->execution_count = index + 1;
    MakeExecutorRepeatable(&prepared->executions[index], state);
    AllocateWorkspace(&prepared->executions[index], state);
}

void ConfigureOperation(
    PreparedUtility* prepared,
    uint64_t input_count,
    ErrorState* state) noexcept {
    ExecutorWorkspace* first = &prepared->executions[0];
    if (prepared->operation == kIdentity) {
        const aclnnStatus status = aclnnCastGetWorkspaceSize(
            prepared->inputs[0],
            prepared->dtype,
            prepared->output,
            &first->workspace_size,
            &first->executor);
        PrepareExecution(
            prepared,
            0,
            status,
            aclnnCast,
            "aclnnCastGetWorkspaceSize",
            state);
        return;
    }
    if (prepared->operation == kTranspose) {
        if (prepared->parameter_count != prepared->input_ranks[0] ||
            !CreateArray(
                prepared,
                0,
                prepared->parameters,
                prepared->parameter_count,
                state)) {
            if (state->first_status == ACL_SUCCESS) {
                RecordWrapperFailure(
                    state, "invalid transpose permutation", -51);
            }
            return;
        }
        const aclnnStatus status = aclnnPermuteGetWorkspaceSize(
            prepared->inputs[0],
            prepared->arrays[0],
            prepared->output,
            &first->workspace_size,
            &first->executor);
        PrepareExecution(
            prepared,
            0,
            status,
            aclnnPermute,
            "aclnnPermuteGetWorkspaceSize",
            state);
        return;
    }
    if (prepared->operation == kSlice) {
        const uint64_t rank = prepared->input_ranks[0];
        if (prepared->parameter_count != 4 * rank) {
            RecordWrapperFailure(state, "invalid slice parameters", -52);
            return;
        }
        for (uint64_t index = 0; index < 4; ++index) {
            if (!CreateArray(
                    prepared,
                    index,
                    prepared->parameters + index * rank,
                    rank,
                    state)) {
                return;
            }
        }
        const aclnnStatus status = aclnnSliceV2GetWorkspaceSize(
            prepared->inputs[0],
            prepared->arrays[0],
            prepared->arrays[1],
            prepared->arrays[2],
            prepared->arrays[3],
            prepared->output,
            &first->workspace_size,
            &first->executor);
        PrepareExecution(
            prepared,
            0,
            status,
            aclnnSliceV2,
            "aclnnSliceV2GetWorkspaceSize",
            state);
        return;
    }
    if (prepared->operation == kConcatenate) {
        if (prepared->parameter_count != 1) {
            RecordWrapperFailure(state, "invalid concatenate axis", -53);
            return;
        }
        const aclTensor* tensors[kMaxInputs] = {};
        for (uint64_t index = 0; index < input_count; ++index) {
            tensors[index] = prepared->inputs[index];
        }
        prepared->tensor_list = aclCreateTensorList(tensors, input_count);
        if (prepared->tensor_list == nullptr) {
            RecordWrapperFailure(state, "aclCreateTensorList", -54);
            return;
        }
        // aclTensorList owns the aclTensor wrappers passed to it. Relinquish
        // the individual pointers so cleanup does not destroy them twice.
        for (uint64_t index = 0; index < input_count; ++index) {
            prepared->inputs[index] = nullptr;
        }
        const aclnnStatus status = aclnnCatGetWorkspaceSize(
            prepared->tensor_list,
            prepared->parameters[0],
            prepared->output,
            &first->workspace_size,
            &first->executor);
        PrepareExecution(
            prepared,
            0,
            status,
            aclnnCat,
            "aclnnCatGetWorkspaceSize",
            state);
        return;
    }
    if (prepared->operation == kGenIndex) {
        if (input_count != 2 || prepared->input_ranks[1] != 1 ||
            prepared->parameter_count != 1) {
            RecordWrapperFailure(
                state, "gen_index requires input and scratch tensors", -55);
            return;
        }
        prepared->scalar_values[1] = prepared->input_shapes[1][0];
        for (uint64_t index = 0; index < 3; ++index) {
            prepared->scalars[index] = aclCreateScalar(
                &prepared->scalar_values[index], ACL_INT64);
            if (prepared->scalars[index] == nullptr) {
                RecordWrapperFailure(state, "aclCreateScalar", -56);
                return;
            }
        }
        const aclnnStatus arange_status = aclnnArangeGetWorkspaceSize(
            prepared->scalars[0],
            prepared->scalars[1],
            prepared->scalars[2],
            prepared->inputs[1],
            &first->workspace_size,
            &first->executor);
        PrepareExecution(
            prepared,
            0,
            arange_status,
            aclnnArange,
            "aclnnArangeGetWorkspaceSize",
            state);
        if (state->first_status != ACL_SUCCESS ||
            !CreateArray(
                prepared,
                0,
                prepared->output_shape,
                prepared->output_rank,
                state)) {
            return;
        }
        const uint64_t axis =
            static_cast<uint64_t>(prepared->parameters[0]);
        if (axis >= prepared->output_rank) {
            RecordWrapperFailure(state, "invalid gen_index axis", -58);
            return;
        }
        int64_t running_stride = 1;
        for (uint64_t reverse = prepared->output_rank;
             reverse > 0;
             --reverse) {
            const uint64_t index = reverse - 1;
            prepared->gen_index_shape[index] =
                index == axis ? prepared->scalar_values[1] : 1;
            prepared->gen_index_strides[index] = running_stride;
            running_stride *= prepared->gen_index_shape[index];
        }
        prepared->gen_index_view = CreateTensor(
            prepared->gen_index_shape,
            prepared->gen_index_strides,
            prepared->output_rank,
            prepared->dtype,
            prepared->input_data[1]);
        if (prepared->gen_index_view == nullptr) {
            RecordWrapperFailure(
                state, "aclCreateTensor(gen_index_view)", -59);
            return;
        }
        ExecutorWorkspace* second = &prepared->executions[1];
        const aclnnStatus expand_status = aclnnExpandGetWorkspaceSize(
            prepared->gen_index_view,
            prepared->arrays[0],
            prepared->output,
            &second->workspace_size,
            &second->executor);
        PrepareExecution(
            prepared,
            1,
            expand_status,
            aclnnExpand,
            "aclnnExpandGetWorkspaceSize",
            state);
        return;
    }
    RecordWrapperFailure(state, "unsupported utility operation code", -57);
}

}  // namespace

extern "C" int flagdnn_aclnn_utility_create(
    int32_t operation,
    const void* input0_data,
    const int64_t* input0_shape,
    const int64_t* input0_strides,
    uint64_t input0_rank,
    const void* input1_data,
    const int64_t* input1_shape,
    const int64_t* input1_strides,
    uint64_t input1_rank,
    const void* input2_data,
    const int64_t* input2_shape,
    const int64_t* input2_strides,
    uint64_t input2_rank,
    uint64_t input_count,
    void* output_data,
    const int64_t* output_shape,
    const int64_t* output_strides,
    uint64_t output_rank,
    int32_t dtype_code,
    const int64_t* parameters,
    uint64_t parameter_count,
    void* stream_handle,
    void** prepared_handle,
    char* error_buffer,
    size_t error_buffer_length) noexcept {
    ErrorState state{error_buffer, error_buffer_length, ACL_SUCCESS, false};
    ClearError(&state);
    if (prepared_handle == nullptr) {
        return ReturnFailure(&state, "prepared handle pointer is null", -60);
    }
    *prepared_handle = nullptr;
    if (input_count == 0 || input_count > kMaxInputs ||
        output_data == nullptr || output_shape == nullptr ||
        output_strides == nullptr || output_rank == 0 ||
        output_rank > kMaxTensorRank || stream_handle == nullptr ||
        parameter_count > kMaxParameters ||
        (parameter_count != 0 && parameters == nullptr)) {
        return ReturnFailure(&state, "invalid utility arguments", -61);
    }
    const void* input_data[kMaxInputs] = {
        input0_data, input1_data, input2_data};
    const int64_t* input_shapes[kMaxInputs] = {
        input0_shape, input1_shape, input2_shape};
    const int64_t* input_strides[kMaxInputs] = {
        input0_strides, input1_strides, input2_strides};
    const uint64_t input_ranks[kMaxInputs] = {
        input0_rank, input1_rank, input2_rank};
    for (uint64_t index = 0; index < input_count; ++index) {
        if (input_data[index] == nullptr || input_shapes[index] == nullptr ||
            input_strides[index] == nullptr || input_ranks[index] == 0 ||
            input_ranks[index] > kMaxTensorRank) {
            return ReturnFailure(&state, "invalid utility input", -62);
        }
    }
    aclDataType dtype = ACL_DT_UNDEFINED;
    if (!MapDataType(dtype_code, &dtype)) {
        return ReturnFailure(&state, "unsupported dtype code", -63);
    }
    PreparedUtility* prepared = new (std::nothrow) PreparedUtility;
    if (prepared == nullptr) {
        return ReturnFailure(
            &state, "failed to allocate prepared utility", -64);
    }
    prepared->operation = operation;
    prepared->dtype = dtype;
    prepared->output_rank = output_rank;
    prepared->parameter_count = parameter_count;
    prepared->stream = reinterpret_cast<aclrtStream>(stream_handle);
    CopyMetadata(prepared->output_shape, output_shape, output_rank);
    CopyMetadata(prepared->output_strides, output_strides, output_rank);
    CopyMetadata(prepared->parameters, parameters, parameter_count);

    for (uint64_t index = 0; index < input_count; ++index) {
        prepared->input_ranks[index] = input_ranks[index];
        prepared->input_data[index] = const_cast<void*>(input_data[index]);
        CopyMetadata(
            prepared->input_shapes[index],
            input_shapes[index],
            input_ranks[index]);
        CopyMetadata(
            prepared->input_strides[index],
            input_strides[index],
            input_ranks[index]);
        prepared->inputs[index] = CreateTensor(
            prepared->input_shapes[index],
            prepared->input_strides[index],
            prepared->input_ranks[index],
            dtype,
            const_cast<void*>(input_data[index]));
        if (prepared->inputs[index] == nullptr) {
            RecordWrapperFailure(&state, "aclCreateTensor(input)", -65);
            break;
        }
    }
    if (state.first_status == ACL_SUCCESS) {
        prepared->output = CreateTensor(
            prepared->output_shape,
            prepared->output_strides,
            prepared->output_rank,
            dtype,
            output_data);
        if (prepared->output == nullptr) {
            RecordWrapperFailure(&state, "aclCreateTensor(output)", -66);
        }
    }
    if (state.first_status == ACL_SUCCESS) {
        ConfigureOperation(prepared, input_count, &state);
    }
    if (state.first_status != ACL_SUCCESS) {
        DestroyPreparedUtility(prepared, &state);
        delete prepared;
        return state.first_status;
    }
    *prepared_handle = prepared;
    return ACL_SUCCESS;
}

extern "C" int flagdnn_aclnn_utility_run(
    void* prepared_handle,
    char* error_buffer,
    size_t error_buffer_length) noexcept {
    ErrorState state{error_buffer, error_buffer_length, ACL_SUCCESS, false};
    ClearError(&state);
    if (prepared_handle == nullptr) {
        return ReturnFailure(&state, "prepared utility handle is null", -70);
    }
    PreparedUtility* prepared =
        static_cast<PreparedUtility*>(prepared_handle);
    for (uint64_t index = 0; index < prepared->execution_count; ++index) {
        ExecutorWorkspace* execution = &prepared->executions[index];
        const aclnnStatus status = prepared->execute_functions[index](
            execution->workspace,
            execution->workspace_size,
            execution->executor,
            prepared->stream);
        RecordAclFailure(&state, "aclnn utility execution", status);
        if (state.first_status != ACL_SUCCESS) {
            break;
        }
    }
    return state.first_status;
}

extern "C" int flagdnn_aclnn_utility_destroy(
    void* prepared_handle,
    char* error_buffer,
    size_t error_buffer_length) noexcept {
    ErrorState state{error_buffer, error_buffer_length, ACL_SUCCESS, false};
    ClearError(&state);
    if (prepared_handle == nullptr) {
        return ReturnFailure(&state, "prepared utility handle is null", -71);
    }
    PreparedUtility* prepared =
        static_cast<PreparedUtility*>(prepared_handle);
    RecordAclFailure(
        &state,
        "aclrtSynchronizeStream",
        aclrtSynchronizeStream(prepared->stream));
    DestroyPreparedUtility(prepared, &state);
    delete prepared;
    return state.first_status;
}

}  // namespace flagdnn_test::ascend
