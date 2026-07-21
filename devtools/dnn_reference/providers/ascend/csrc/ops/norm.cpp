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

#include "aclnnop/aclnn_batch_norm_elemt.h"
#include "aclnnop/aclnn_batch_norm_stats.h"
#include "aclnnop/aclnn_layer_norm.h"
#include "aclnnop/aclnn_rms_norm.h"
#include "common/oracle_common.h"

namespace flagdnn_test::ascend {

struct NormTensorArgument {
    const void* data;
    const int64_t* shape;
    const int64_t* strides;
    uint64_t rank;
    int32_t dtype_code;
};

namespace {

constexpr uint64_t kMaxTensorRank = 8;
constexpr uint64_t kMaxTensors = 10;
constexpr uint64_t kMaxIntParameters = 8;

enum NormOperation : int32_t {
    kLayerNorm = 0,
    kRmsNorm = 1,
    kBatchNormStats = 2,
    kBatchNormInference = 3,
};

struct PreparedNorm {
    int64_t shapes[kMaxTensors][kMaxTensorRank] = {};
    int64_t strides[kMaxTensors][kMaxTensorRank] = {};
    uint64_t ranks[kMaxTensors] = {};
    aclTensor* tensors[kMaxTensors] = {};
    uint64_t input_count = 0;
    uint64_t output_count = 0;
    int64_t int_parameters[kMaxIntParameters] = {};
    uint64_t int_parameter_count = 0;
    aclIntArray* int_array = nullptr;
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

aclTensor* Output(PreparedNorm* prepared, uint64_t index) noexcept {
    return prepared->tensors[prepared->input_count + index];
}

aclFormat ActivationFormat(uint64_t rank) noexcept {
    switch (rank) {
        case 2:
            return ACL_FORMAT_NC;
        case 3:
            return ACL_FORMAT_NCL;
        case 4:
            return ACL_FORMAT_NCHW;
        case 5:
            return ACL_FORMAT_NCDHW;
        default:
            return ACL_FORMAT_ND;
    }
}

void DestroyPreparedNorm(
    PreparedNorm* prepared,
    ErrorState* state) noexcept {
    DestroyExecutor(&prepared->execution, state);
    if (prepared->int_array != nullptr) {
        RecordAclFailure(
            state,
            "aclDestroyIntArray",
            aclDestroyIntArray(prepared->int_array));
        prepared->int_array = nullptr;
    }
    for (uint64_t index = 0;
         index < prepared->input_count + prepared->output_count;
         ++index) {
        DestroyTensor(
            &prepared->tensors[index], "aclDestroyTensor(norm)", state);
    }
    FreeWorkspace(&prepared->execution, state);
}

bool CreateArguments(
    PreparedNorm* prepared,
    const NormTensorArgument* arguments,
    uint64_t count,
    uint64_t offset,
    int32_t operation,
    ErrorState* state) noexcept {
    if (arguments == nullptr || offset + count > kMaxTensors) {
        RecordWrapperFailure(state, "invalid norm tensor arguments", -100);
        return false;
    }
    for (uint64_t index = 0; index < count; ++index) {
        const NormTensorArgument& argument = arguments[index];
        const uint64_t target = offset + index;
        if (argument.data == nullptr || argument.shape == nullptr ||
            argument.strides == nullptr || argument.rank == 0 ||
            argument.rank > kMaxTensorRank) {
            RecordWrapperFailure(state, "invalid norm tensor metadata", -101);
            return false;
        }
        aclDataType dtype = ACL_DT_UNDEFINED;
        if (!MapDataType(argument.dtype_code, &dtype)) {
            RecordWrapperFailure(state, "unsupported norm tensor dtype", -102);
            return false;
        }
        prepared->ranks[target] = argument.rank;
        CopyMetadata(
            prepared->shapes[target], argument.shape, argument.rank);
        CopyMetadata(
            prepared->strides[target], argument.strides, argument.rank);
        aclFormat format = ACL_FORMAT_ND;
        const bool is_batchnorm_activation =
            (operation == kBatchNormStats && target == 0) ||
            (operation == kBatchNormInference &&
             (target == 0 || target == prepared->input_count));
        if (is_batchnorm_activation) {
            format = ActivationFormat(argument.rank);
        }
        prepared->tensors[target] = aclCreateTensor(
            prepared->shapes[target],
            prepared->ranks[target],
            dtype,
            prepared->strides[target],
            0,
            format,
            prepared->shapes[target],
            prepared->ranks[target],
            const_cast<void*>(argument.data));
        if (prepared->tensors[target] == nullptr) {
            RecordWrapperFailure(state, "aclCreateTensor(norm)", -103);
            return false;
        }
    }
    return true;
}

}  // namespace

extern "C" int flagdnn_aclnn_norm_create(
    int32_t operation,
    const NormTensorArgument* inputs,
    uint64_t input_count,
    const NormTensorArgument* outputs,
    uint64_t output_count,
    const int64_t* int_parameters,
    uint64_t int_parameter_count,
    double parameter0,
    double parameter1,
    void* stream_handle,
    void** prepared_handle,
    char* error_buffer,
    size_t error_buffer_length) noexcept {
    ErrorState state{error_buffer, error_buffer_length, ACL_SUCCESS, false};
    ClearError(&state);
    if (prepared_handle == nullptr) {
        return ReturnFailure(&state, "prepared handle pointer is null", -110);
    }
    *prepared_handle = nullptr;
    if (input_count + output_count > kMaxTensors || input_count == 0 ||
        output_count == 0 || int_parameter_count > kMaxIntParameters ||
        (int_parameter_count != 0 && int_parameters == nullptr) ||
        stream_handle == nullptr) {
        return ReturnFailure(&state, "invalid norm arguments", -111);
    }
    PreparedNorm* prepared = new (std::nothrow) PreparedNorm;
    if (prepared == nullptr) {
        return ReturnFailure(
            &state, "failed to allocate prepared norm", -112);
    }
    prepared->input_count = input_count;
    prepared->output_count = output_count;
    prepared->int_parameter_count = int_parameter_count;
    prepared->stream = reinterpret_cast<aclrtStream>(stream_handle);
    CopyMetadata(
        prepared->int_parameters, int_parameters, int_parameter_count);
    CreateArguments(prepared, inputs, input_count, 0, operation, &state);
    if (state.first_status == ACL_SUCCESS) {
        CreateArguments(
            prepared,
            outputs,
            output_count,
            input_count,
            operation,
            &state);
    }

    aclnnStatus status = ACL_SUCCESS;
    if (state.first_status == ACL_SUCCESS && operation == kLayerNorm) {
        if (input_count != 3 || output_count != 3 ||
            int_parameter_count == 0) {
            RecordWrapperFailure(&state, "invalid layernorm arguments", -113);
        } else {
            prepared->int_array = aclCreateIntArray(
                prepared->int_parameters, prepared->int_parameter_count);
            if (prepared->int_array == nullptr) {
                RecordWrapperFailure(
                    &state, "aclCreateIntArray(normalized_shape)", -114);
            } else {
                status = aclnnLayerNormGetWorkspaceSize(
                    prepared->tensors[0],
                    prepared->int_array,
                    prepared->tensors[1],
                    prepared->tensors[2],
                    parameter0,
                    Output(prepared, 0),
                    Output(prepared, 1),
                    Output(prepared, 2),
                    &prepared->execution.workspace_size,
                    &prepared->execution.executor);
                prepared->execute = aclnnLayerNorm;
            }
        }
    } else if (state.first_status == ACL_SUCCESS && operation == kRmsNorm) {
        if (input_count != 2 || output_count != 2) {
            RecordWrapperFailure(&state, "invalid rmsnorm arguments", -115);
        } else {
            status = aclnnRmsNormGetWorkspaceSize(
                prepared->tensors[0],
                prepared->tensors[1],
                parameter0,
                Output(prepared, 0),
                Output(prepared, 1),
                &prepared->execution.workspace_size,
                &prepared->execution.executor);
            prepared->execute = aclnnRmsNorm;
        }
    } else if (
        state.first_status == ACL_SUCCESS &&
        operation == kBatchNormStats) {
        if (input_count != 1 || output_count != 2) {
            RecordWrapperFailure(
                &state, "invalid batchnorm stats arguments", -116);
        } else {
            status = aclnnBatchNormStatsGetWorkspaceSize(
                prepared->tensors[0],
                parameter0,
                Output(prepared, 0),
                Output(prepared, 1),
                &prepared->execution.workspace_size,
                &prepared->execution.executor);
            prepared->execute = aclnnBatchNormStats;
        }
    } else if (
        state.first_status == ACL_SUCCESS &&
        operation == kBatchNormInference) {
        if (input_count != 5 || output_count != 1) {
            RecordWrapperFailure(
                &state, "invalid batchnorm inference arguments", -117);
        } else {
            status = aclnnBatchNormElemtGetWorkspaceSize(
                prepared->tensors[0],
                prepared->tensors[1],
                prepared->tensors[2],
                prepared->tensors[3],
                prepared->tensors[4],
                parameter0,
                Output(prepared, 0),
                &prepared->execution.workspace_size,
                &prepared->execution.executor);
            prepared->execute = aclnnBatchNormElemt;
        }
    } else if (state.first_status == ACL_SUCCESS) {
        RecordWrapperFailure(&state, "unsupported norm operation", -118);
    }
    if (state.first_status == ACL_SUCCESS) {
        RecordAclFailure(&state, "norm GetWorkspaceSize", status);
    }
    if (state.first_status == ACL_SUCCESS) {
        MakeExecutorRepeatable(&prepared->execution, &state);
    }
    if (state.first_status == ACL_SUCCESS) {
        AllocateWorkspace(&prepared->execution, &state);
    }
    if (state.first_status != ACL_SUCCESS) {
        DestroyPreparedNorm(prepared, &state);
        delete prepared;
        return state.first_status;
    }
    *prepared_handle = prepared;
    return ACL_SUCCESS;
}

extern "C" int flagdnn_aclnn_norm_run(
    void* prepared_handle,
    char* error_buffer,
    size_t error_buffer_length) noexcept {
    ErrorState state{error_buffer, error_buffer_length, ACL_SUCCESS, false};
    ClearError(&state);
    if (prepared_handle == nullptr) {
        return ReturnFailure(&state, "prepared norm handle is null", -120);
    }
    PreparedNorm* prepared = static_cast<PreparedNorm*>(prepared_handle);
    RecordAclFailure(
        &state,
        "aclnn norm execution",
        prepared->execute(
            prepared->execution.workspace,
            prepared->execution.workspace_size,
            prepared->execution.executor,
            prepared->stream));
    return state.first_status;
}

extern "C" int flagdnn_aclnn_norm_destroy(
    void* prepared_handle,
    char* error_buffer,
    size_t error_buffer_length) noexcept {
    ErrorState state{error_buffer, error_buffer_length, ACL_SUCCESS, false};
    ClearError(&state);
    if (prepared_handle == nullptr) {
        return ReturnFailure(&state, "prepared norm handle is null", -121);
    }
    PreparedNorm* prepared = static_cast<PreparedNorm*>(prepared_handle);
    RecordAclFailure(
        &state,
        "aclrtSynchronizeStream",
        aclrtSynchronizeStream(prepared->stream));
    DestroyPreparedNorm(prepared, &state);
    delete prepared;
    return state.first_status;
}

}  // namespace flagdnn_test::ascend
