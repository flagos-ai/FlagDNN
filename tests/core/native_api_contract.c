#include <flagdnn/flagdnn.h>

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

static int require_status(flagdnnStatus_t actual,
                          flagdnnStatus_t expected,
                          int failure_code) {
  if (actual != expected) {
    return failure_code;
  }
  if (expected != FLAGDNN_STATUS_SUCCESS &&
      strlen(flagdnnGetLastErrorString()) == 0U) {
    return failure_code + 100;
  }
  return 0;
}

int main(void) {
  flagdnnTensorDescriptor_t input = NULL;
  flagdnnTensorDescriptor_t output = NULL;
  flagdnnTensorDescriptor_t scalar = NULL;
  flagdnnOperationDescriptor_t operation = NULL;
  flagdnnOperationDescriptor_t add_operation = NULL;
  flagdnnOperationDescriptor_t pointwise_operation = NULL;
  flagdnnOperationDescriptor_t reduction_operation = NULL;
  flagdnnOperationDescriptor_t legacy_reduction_operation = NULL;
  flagdnnOperationDescriptor_t convolution_operation = NULL;
  flagdnnOperationDescriptor_t generic_operation = NULL;
  flagdnnGraph_t graph = NULL;
  int result = require_status(
      flagdnnDestroy(NULL), FLAGDNN_STATUS_INVALID_VALUE, 1);
  if (result != 0) {
    return result;
  }
  result = require_status(
      flagdnnCreateTensorDescriptor(NULL), FLAGDNN_STATUS_INVALID_VALUE, 2);
  if (result != 0) {
    return result;
  }
  result = require_status(flagdnnCreateTensorDescriptor(&input),
                          FLAGDNN_STATUS_SUCCESS,
                          3);
  if (result != 0 || input == NULL) {
    return result == 0 ? 4 : result;
  }

  int32_t is_virtual = -1;
  result = require_status(
      flagdnnGetTensorDescriptorVirtual(input, &is_virtual),
      FLAGDNN_STATUS_NOT_INITIALIZED,
      94);
  if (result != 0) {
    return result;
  }

  size_t size_in_bytes = 0;
  result = require_status(
      flagdnnGetTensorSizeInBytes(input, &size_in_bytes),
      FLAGDNN_STATUS_NOT_INITIALIZED,
      5);
  if (result != 0) {
    return result;
  }

  const int64_t dimensions[2] = {2, 3};
  const int64_t strides[2] = {3, 1};
  result = require_status(
      flagdnnSetTensorNdDescriptor(input,
                                   0,
                                   FLAGDNN_DATA_FLOAT32,
                                   2,
                                   dimensions,
                                   strides),
      FLAGDNN_STATUS_INVALID_VALUE,
      6);
  if (result != 0) {
    return result;
  }
  result = require_status(
      flagdnnSetTensorNdDescriptor(input,
                                   17,
                                   FLAGDNN_DATA_FLOAT32,
                                   2,
                                   dimensions,
                                   strides),
      FLAGDNN_STATUS_SUCCESS,
      7);
  if (result != 0) {
    return result;
  }
  result = require_status(
      flagdnnGetTensorDescriptorVirtual(input, &is_virtual),
      FLAGDNN_STATUS_SUCCESS,
      95);
  if (result != 0 || is_virtual != 0) {
    return result == 0 ? 96 : result;
  }
  result = require_status(
      flagdnnSetTensorDescriptorVirtual(input, 2),
      FLAGDNN_STATUS_INVALID_VALUE,
      97);
  if (result != 0) {
    return result;
  }
  result = require_status(
      flagdnnSetTensorDescriptorVirtual(input, 1),
      FLAGDNN_STATUS_SUCCESS,
      98);
  if (result != 0) {
    return result;
  }
  result = require_status(
      flagdnnGetTensorDescriptorVirtual(input, &is_virtual),
      FLAGDNN_STATUS_SUCCESS,
      99);
  if (result != 0 || is_virtual != 1) {
    return result == 0 ? 100 : result;
  }
  result = require_status(
      flagdnnSetTensorNdDescriptor(input,
                                   17,
                                   FLAGDNN_DATA_FLOAT32,
                                   2,
                                   dimensions,
                                   strides),
      FLAGDNN_STATUS_SUCCESS,
      101);
  if (result != 0) {
    return result;
  }
  result = require_status(
      flagdnnGetTensorDescriptorVirtual(input, &is_virtual),
      FLAGDNN_STATUS_SUCCESS,
      102);
  if (result != 0 || is_virtual != 0) {
    return result == 0 ? 103 : result;
  }

  result = require_status(
      flagdnnGetTensorSizeInBytes(input, &size_in_bytes),
      FLAGDNN_STATUS_SUCCESS,
      8);
  if (result != 0 || size_in_bytes != 24U) {
    return result == 0 ? 9 : result;
  }

  int64_t uid = 0;
  flagdnnDataType_t data_type = FLAGDNN_DATA_FLOAT16;
  int32_t rank = 0;
  int64_t actual_dimensions[2] = {0, 0};
  int64_t actual_strides[2] = {0, 0};
  result = require_status(
      flagdnnGetTensorNdDescriptor(input,
                                   1,
                                   &uid,
                                   &data_type,
                                   &rank,
                                   actual_dimensions,
                                   actual_strides),
      FLAGDNN_STATUS_INVALID_VALUE,
      10);
  if (result != 0) {
    return result;
  }
  result = require_status(
      flagdnnGetTensorNdDescriptor(input,
                                   2,
                                   &uid,
                                   &data_type,
                                   &rank,
                                   actual_dimensions,
                                   actual_strides),
      FLAGDNN_STATUS_SUCCESS,
      11);
  if (result != 0 || uid != 17 || data_type != FLAGDNN_DATA_FLOAT32 ||
      rank != 2 || actual_dimensions[0] != 2 ||
      actual_dimensions[1] != 3 || actual_strides[0] != 3 ||
      actual_strides[1] != 1) {
    return result == 0 ? 12 : result;
  }

  result = require_status(flagdnnCreateTensorDescriptor(&scalar),
                          FLAGDNN_STATUS_SUCCESS,
                          49);
  if (result != 0 || scalar == NULL) {
    return result == 0 ? 50 : result;
  }
  result = require_status(
      flagdnnSetTensorNdDescriptor(
          scalar, 19, FLAGDNN_DATA_FLOAT32, 0, NULL, NULL),
      FLAGDNN_STATUS_SUCCESS,
      51);
  if (result != 0) {
    return result;
  }
  result = require_status(
      flagdnnGetTensorSizeInBytes(scalar, &size_in_bytes),
      FLAGDNN_STATUS_SUCCESS,
      52);
  if (result != 0 || size_in_bytes != 4U) {
    return result == 0 ? 53 : result;
  }
  uid = 0;
  data_type = FLAGDNN_DATA_FLOAT16;
  rank = -1;
  result = require_status(
      flagdnnGetTensorNdDescriptor(
          scalar, 0, &uid, &data_type, &rank, NULL, NULL),
      FLAGDNN_STATUS_SUCCESS,
      54);
  if (result != 0 || uid != 19 || data_type != FLAGDNN_DATA_FLOAT32 ||
      rank != 0) {
    return result == 0 ? 55 : result;
  }

  result = require_status(flagdnnCreateTensorDescriptor(&output),
                          FLAGDNN_STATUS_SUCCESS,
                          13);
  if (result != 0) {
    return result;
  }
  result = require_status(
      flagdnnSetTensorNdDescriptor(
          output, 18, FLAGDNN_DATA_FLOAT16, 2, dimensions, strides),
      FLAGDNN_STATUS_SUCCESS,
      33);
  if (result != 0) {
    return result;
  }
  result = require_status(
      flagdnnGetTensorSizeInBytes(output, &size_in_bytes),
      FLAGDNN_STATUS_SUCCESS,
      34);
  if (result != 0 || size_in_bytes != 12U) {
    return result == 0 ? 35 : result;
  }
  result = require_status(
      flagdnnSetTensorNdDescriptor(
          output, 18, FLAGDNN_DATA_BFLOAT16, 2, dimensions, strides),
      FLAGDNN_STATUS_SUCCESS,
      36);
  if (result != 0) {
    return result;
  }
  result = require_status(
      flagdnnGetTensorSizeInBytes(output, &size_in_bytes),
      FLAGDNN_STATUS_SUCCESS,
      37);
  if (result != 0 || size_in_bytes != 12U) {
    return result == 0 ? 38 : result;
  }
  const int64_t padded_strides[2] = {7, 2};
  result = require_status(
      flagdnnSetTensorNdDescriptor(output,
                                   18,
                                   FLAGDNN_DATA_FLOAT32,
                                   2,
                                   dimensions,
                                   padded_strides),
      FLAGDNN_STATUS_SUCCESS,
      39);
  if (result != 0) {
    return result;
  }
  result = require_status(
      flagdnnGetTensorSizeInBytes(output, &size_in_bytes),
      FLAGDNN_STATUS_SUCCESS,
      40);
  if (result != 0 || size_in_bytes != 48U) {
    return result == 0 ? 41 : result;
  }
  result = require_status(
      flagdnnSetTensorNdDescriptor(output,
                                   18,
                                   FLAGDNN_DATA_FLOAT32,
                                   2,
                                   dimensions,
                                   strides),
      FLAGDNN_STATUS_SUCCESS,
      14);
  if (result != 0) {
    return result;
  }

  result = require_status(
      flagdnnCreateOperationDescriptor(FLAGDNN_OPERATION_RELU, &operation),
      FLAGDNN_STATUS_SUCCESS,
      15);
  if (result != 0 || operation == NULL) {
    return result == 0 ? 16 : result;
  }
  result = require_status(
      flagdnnSetOperationDescriptorName(operation, "relu_contract"),
      FLAGDNN_STATUS_SUCCESS,
      119);
  if (result != 0) {
    return result;
  }
  result = require_status(
      flagdnnSetOperationDescriptorComputeDataType(
          operation, FLAGDNN_DATA_FLOAT32),
      FLAGDNN_STATUS_SUCCESS,
      120);
  if (result != 0) {
    return result;
  }
  result = require_status(
      flagdnnSetOperationDescriptorComputeDataType(
          operation, (flagdnnDataType_t)99),
      FLAGDNN_STATUS_INVALID_VALUE,
      121);
  if (result != 0) {
    return result;
  }
  flagdnnOperation_t operation_type = FLAGDNN_OPERATION_ADD;
  result = require_status(
      flagdnnGetOperationDescriptorType(operation, &operation_type),
      FLAGDNN_STATUS_SUCCESS,
      17);
  if (result != 0 || operation_type != FLAGDNN_OPERATION_RELU) {
    return result == 0 ? 18 : result;
  }
  result = require_status(
      flagdnnSetMatmulOperationDescriptor(
          operation, input, input, output),
      FLAGDNN_STATUS_INVALID_VALUE,
      142);
  if (result != 0) {
    return result;
  }
  result = require_status(
      flagdnnSetAddOperationDescriptor(operation, input, output, output),
      FLAGDNN_STATUS_INVALID_VALUE,
      19);
  if (result != 0) {
    return result;
  }
  result = require_status(
      flagdnnSetReluOperationDescriptor(operation, input, output),
      FLAGDNN_STATUS_SUCCESS,
      20);
  if (result != 0) {
    return result;
  }

  result = require_status(
      flagdnnCreateOperationDescriptor(FLAGDNN_OPERATION_POINTWISE,
                                       &pointwise_operation),
      FLAGDNN_STATUS_SUCCESS,
      78);
  if (result != 0 || pointwise_operation == NULL) {
    return result == 0 ? 79 : result;
  }
  operation_type = FLAGDNN_OPERATION_RELU;
  result = require_status(
      flagdnnGetOperationDescriptorType(pointwise_operation, &operation_type),
      FLAGDNN_STATUS_SUCCESS,
      80);
  if (result != 0 || operation_type != FLAGDNN_OPERATION_POINTWISE) {
    return result == 0 ? 81 : result;
  }
  result = require_status(
      flagdnnSetPointwiseUnaryOperationDescriptor(
          pointwise_operation, input, FLAGDNN_POINTWISE_ADD, output),
      FLAGDNN_STATUS_NOT_SUPPORTED,
      82);
  if (result != 0) {
    return result;
  }
  result = require_status(
      flagdnnSetPointwiseBinaryOperationDescriptor(
          pointwise_operation,
          input,
          output,
          FLAGDNN_POINTWISE_ABS,
          output),
      FLAGDNN_STATUS_NOT_SUPPORTED,
      83);
  if (result != 0) {
    return result;
  }
  result = require_status(
      flagdnnSetPointwiseBinaryOperationDescriptorWithAlpha(
          pointwise_operation,
          input,
          output,
          FLAGDNN_POINTWISE_ADD,
          output,
          NAN),
      FLAGDNN_STATUS_INVALID_VALUE,
      84);
  if (result != 0) {
    return result;
  }
  result = require_status(
      flagdnnSetPointwiseBinaryOperationDescriptor(
          pointwise_operation,
          input,
          output,
          FLAGDNN_POINTWISE_ADD,
          output),
      FLAGDNN_STATUS_SUCCESS,
      85);
  if (result != 0) {
    return result;
  }
  result = require_status(
      flagdnnSetPointwiseUnaryOperationDescriptor(
          pointwise_operation, input, FLAGDNN_POINTWISE_SUB, output),
      FLAGDNN_STATUS_NOT_SUPPORTED,
      88);
  if (result != 0) {
    return result;
  }
  result = require_status(
      flagdnnSetPointwiseBinaryOperationDescriptorWithAlpha(
          pointwise_operation,
          input,
          output,
          FLAGDNN_POINTWISE_MUL,
          output,
          0.5),
      FLAGDNN_STATUS_INVALID_VALUE,
      89);
  if (result != 0) {
    return result;
  }
  result = require_status(
      flagdnnSetPointwiseBinaryOperationDescriptorWithAlpha(
          pointwise_operation,
          input,
          output,
          FLAGDNN_POINTWISE_SUB,
          output,
          -2.0),
      FLAGDNN_STATUS_SUCCESS,
      90);
  if (result != 0) {
    return result;
  }
  const flagdnnPointwiseMode_t binary_modes[] = {
      FLAGDNN_POINTWISE_SUB,
      FLAGDNN_POINTWISE_MUL,
      FLAGDNN_POINTWISE_DIV,
      FLAGDNN_POINTWISE_MIN,
      FLAGDNN_POINTWISE_MAX,
      FLAGDNN_POINTWISE_MOD,
      FLAGDNN_POINTWISE_POW,
      FLAGDNN_POINTWISE_CMP_EQ,
      FLAGDNN_POINTWISE_CMP_NEQ,
      FLAGDNN_POINTWISE_CMP_GT,
      FLAGDNN_POINTWISE_CMP_GE,
      FLAGDNN_POINTWISE_CMP_LT,
      FLAGDNN_POINTWISE_CMP_LE,
      FLAGDNN_POINTWISE_LOGICAL_AND,
      FLAGDNN_POINTWISE_LOGICAL_OR,
      FLAGDNN_POINTWISE_SIGMOID_BWD,
  };
  for (size_t mode_index = 0;
       mode_index < sizeof(binary_modes) / sizeof(binary_modes[0]);
       ++mode_index) {
    result = require_status(
        flagdnnSetPointwiseBinaryOperationDescriptor(
            pointwise_operation,
            input,
            output,
            binary_modes[mode_index],
            output),
        FLAGDNN_STATUS_SUCCESS,
        91 + (int)mode_index);
    if (result != 0) {
      return result;
    }
  }
  result = require_status(
      flagdnnSetPointwiseTernaryOperationDescriptor(
          pointwise_operation,
          input,
          output,
          input,
          FLAGDNN_POINTWISE_BINARY_SELECT,
          output),
      FLAGDNN_STATUS_SUCCESS,
      107);
  if (result != 0) {
    return result;
  }
  result = require_status(
      flagdnnSetPointwiseTernaryOperationDescriptor(
          pointwise_operation,
          input,
          output,
          input,
          FLAGDNN_POINTWISE_MUL,
          output),
      FLAGDNN_STATUS_NOT_SUPPORTED,
      108);
  if (result != 0) {
    return result;
  }

  const flagdnnPointwiseMode_t unary_modes[] = {
      FLAGDNN_POINTWISE_ABS,
      FLAGDNN_POINTWISE_SIGMOID_FWD,
      FLAGDNN_POINTWISE_TANH_FWD,
      FLAGDNN_POINTWISE_ELU_FWD,
      FLAGDNN_POINTWISE_GELU_FWD,
      FLAGDNN_POINTWISE_SOFTPLUS_FWD,
      FLAGDNN_POINTWISE_SWISH_FWD,
      FLAGDNN_POINTWISE_GELU_APPROX_TANH_FWD,
  };
  for (size_t mode_index = 0;
       mode_index < sizeof(unary_modes) / sizeof(unary_modes[0]);
       ++mode_index) {
    result = require_status(
        flagdnnSetPointwiseUnaryOperationDescriptor(
            pointwise_operation, input, unary_modes[mode_index], output),
        FLAGDNN_STATUS_SUCCESS,
        86 + (int)mode_index);
    if (result != 0) {
      return result;
    }
  }
  flagdnnPointwiseAttributes_t pointwise_attributes =
      FLAGDNN_POINTWISE_ATTRIBUTES_INITIALIZER;
  pointwise_attributes.flags = FLAGDNN_POINTWISE_ATTRIBUTE_SWISH_BETA;
  pointwise_attributes.swish_beta = 1.25;
  result = require_status(
      flagdnnSetPointwiseUnaryOperationDescriptorWithAttributes(
          pointwise_operation,
          input,
          FLAGDNN_POINTWISE_SWISH_FWD,
          output,
          &pointwise_attributes),
      FLAGDNN_STATUS_SUCCESS,
      200);
  if (result != 0) {
    return result;
  }
  result = require_status(
      flagdnnSetPointwiseUnaryOperationDescriptorWithAttributes(
          pointwise_operation,
          input,
          FLAGDNN_POINTWISE_GELU_FWD,
          output,
          NULL),
      FLAGDNN_STATUS_SUCCESS,
      201);
  if (result != 0) {
    return result;
  }
  pointwise_attributes.struct_size = 0U;
  result = require_status(
      flagdnnSetPointwiseUnaryOperationDescriptorWithAttributes(
          pointwise_operation,
          input,
          FLAGDNN_POINTWISE_SWISH_FWD,
          output,
          &pointwise_attributes),
      FLAGDNN_STATUS_INVALID_VALUE,
      202);
  if (result != 0) {
    return result;
  }
  pointwise_attributes =
      (flagdnnPointwiseAttributes_t)
          FLAGDNN_POINTWISE_ATTRIBUTES_INITIALIZER;
  pointwise_attributes.version = 999U;
  result = require_status(
      flagdnnSetPointwiseUnaryOperationDescriptorWithAttributes(
          pointwise_operation,
          input,
          FLAGDNN_POINTWISE_SWISH_FWD,
          output,
          &pointwise_attributes),
      FLAGDNN_STATUS_INVALID_VALUE,
      203);
  if (result != 0) {
    return result;
  }
  pointwise_attributes =
      (flagdnnPointwiseAttributes_t)
          FLAGDNN_POINTWISE_ATTRIBUTES_INITIALIZER;
  pointwise_attributes.flags = UINT64_C(1) << 63;
  result = require_status(
      flagdnnSetPointwiseUnaryOperationDescriptorWithAttributes(
          pointwise_operation,
          input,
          FLAGDNN_POINTWISE_SWISH_FWD,
          output,
          &pointwise_attributes),
      FLAGDNN_STATUS_INVALID_VALUE,
      204);
  if (result != 0) {
    return result;
  }
  pointwise_attributes =
      (flagdnnPointwiseAttributes_t)
          FLAGDNN_POINTWISE_ATTRIBUTES_INITIALIZER;
  pointwise_attributes.flags = FLAGDNN_POINTWISE_ATTRIBUTE_ELU_ALPHA;
  result = require_status(
      flagdnnSetPointwiseUnaryOperationDescriptorWithAttributes(
          pointwise_operation,
          input,
          FLAGDNN_POINTWISE_SWISH_FWD,
          output,
          &pointwise_attributes),
      FLAGDNN_STATUS_INVALID_VALUE,
      205);
  if (result != 0) {
    return result;
  }
  pointwise_attributes =
      (flagdnnPointwiseAttributes_t)
          FLAGDNN_POINTWISE_ATTRIBUTES_INITIALIZER;
  pointwise_attributes.flags = FLAGDNN_POINTWISE_ATTRIBUTE_SWISH_BETA;
  pointwise_attributes.swish_beta = NAN;
  result = require_status(
      flagdnnSetPointwiseUnaryOperationDescriptorWithAttributes(
          pointwise_operation,
          input,
          FLAGDNN_POINTWISE_SWISH_FWD,
          output,
          &pointwise_attributes),
      FLAGDNN_STATUS_INVALID_VALUE,
      206);
  if (result != 0) {
    return result;
  }
  pointwise_attributes =
      (flagdnnPointwiseAttributes_t)
          FLAGDNN_POINTWISE_ATTRIBUTES_INITIALIZER;
  pointwise_attributes.flags =
      FLAGDNN_POINTWISE_ATTRIBUTE_SOFTPLUS_BETA;
  pointwise_attributes.softplus_beta = 0.0;
  result = require_status(
      flagdnnSetPointwiseUnaryOperationDescriptorWithAttributes(
          pointwise_operation,
          input,
          FLAGDNN_POINTWISE_SOFTPLUS_FWD,
          output,
          &pointwise_attributes),
      FLAGDNN_STATUS_INVALID_VALUE,
      207);
  if (result != 0) {
    return result;
  }
  pointwise_attributes =
      (flagdnnPointwiseAttributes_t)
          FLAGDNN_POINTWISE_ATTRIBUTES_INITIALIZER;
  pointwise_attributes.flags =
      FLAGDNN_POINTWISE_ATTRIBUTE_RELU_LOWER_CLIP |
      FLAGDNN_POINTWISE_ATTRIBUTE_RELU_UPPER_CLIP;
  pointwise_attributes.relu_lower_clip = 1.0;
  pointwise_attributes.relu_upper_clip = -1.0;
  result = require_status(
      flagdnnSetPointwiseUnaryOperationDescriptorWithAttributes(
          pointwise_operation,
          input,
          FLAGDNN_POINTWISE_RELU_FWD,
          output,
          &pointwise_attributes),
      FLAGDNN_STATUS_INVALID_VALUE,
      208);
  if (result != 0) {
    return result;
  }
  result = require_status(
      flagdnnDestroyOperationDescriptor(pointwise_operation),
      FLAGDNN_STATUS_SUCCESS,
      87);
  if (result != 0) {
    return result;
  }
  pointwise_operation = NULL;

  result = require_status(
      flagdnnCreateOperationDescriptor(
          FLAGDNN_OPERATION_ADD, &add_operation),
      FLAGDNN_STATUS_SUCCESS,
      42);
  if (result != 0 || add_operation == NULL) {
    return result == 0 ? 43 : result;
  }
  result = require_status(
      flagdnnSetAddOperationDescriptorWithAlpha(
          add_operation, input, output, output, NAN),
      FLAGDNN_STATUS_INVALID_VALUE,
      44);
  if (result != 0) {
    return result;
  }
  result = require_status(
      flagdnnSetAddOperationDescriptorWithAlpha(
          add_operation, input, output, output, INFINITY),
      FLAGDNN_STATUS_INVALID_VALUE,
      45);
  if (result != 0) {
    return result;
  }
  result = require_status(
      flagdnnSetAddOperationDescriptorWithAlpha(
          add_operation, input, output, output, 0.5),
      FLAGDNN_STATUS_SUCCESS,
      46);
  if (result != 0) {
    return result;
  }
  result = require_status(
      flagdnnSetAddOperationDescriptor(
          add_operation, input, output, output),
      FLAGDNN_STATUS_SUCCESS,
      47);
  if (result != 0) {
    return result;
  }
  result = require_status(
      flagdnnDestroyOperationDescriptor(add_operation),
      FLAGDNN_STATUS_SUCCESS,
      48);
  if (result != 0) {
    return result;
  }
  add_operation = NULL;

  if (FLAGDNN_OPERATION_REDUCTION != FLAGDNN_OPERATION_REDUCTION_SUM ||
      FLAGDNN_REDUCTION_ADD != FLAGDNN_REDUCTION_SUM) {
    return 57;
  }
  result = require_status(
      flagdnnCreateOperationDescriptor(
          FLAGDNN_OPERATION_REDUCTION, &reduction_operation),
      FLAGDNN_STATUS_SUCCESS,
      58);
  if (result != 0 || reduction_operation == NULL) {
    return result == 0 ? 59 : result;
  }
  result = require_status(
      flagdnnSetReductionOperationDescriptor(
          reduction_operation,
          input,
          (flagdnnReductionMode_t)99,
          1,
          1,
          output),
      FLAGDNN_STATUS_INVALID_VALUE,
      60);
  if (result != 0) {
    return result;
  }
  result = require_status(
      flagdnnSetReductionOperationDescriptor(reduction_operation,
                                             input,
                                             FLAGDNN_REDUCTION_AVG,
                                             1,
                                             2,
                                             output),
      FLAGDNN_STATUS_INVALID_VALUE,
      61);
  if (result != 0) {
    return result;
  }
  result = require_status(
      flagdnnSetReductionOperationDescriptor(reduction_operation,
                                             input,
                                             FLAGDNN_REDUCTION_MUL,
                                             1,
                                             1,
                                             output),
      FLAGDNN_STATUS_SUCCESS,
      62);
  if (result != 0) {
    return result;
  }
  result = require_status(
      flagdnnCreateOperationDescriptor(FLAGDNN_OPERATION_REDUCTION_SUM,
                                       &legacy_reduction_operation),
      FLAGDNN_STATUS_SUCCESS,
      63);
  if (result != 0 || legacy_reduction_operation == NULL) {
    return result == 0 ? 64 : result;
  }
  result = require_status(
      flagdnnSetReductionSumOperationDescriptor(
          legacy_reduction_operation, input, 1, 1, output),
      FLAGDNN_STATUS_SUCCESS,
      65);
  if (result != 0) {
    return result;
  }
  operation_type = FLAGDNN_OPERATION_RELU;
  result = require_status(
      flagdnnGetOperationDescriptorType(reduction_operation, &operation_type),
      FLAGDNN_STATUS_SUCCESS,
      66);
  if (result != 0 || operation_type != FLAGDNN_OPERATION_REDUCTION) {
    return result == 0 ? 67 : result;
  }
  result = require_status(
      flagdnnDestroyOperationDescriptor(legacy_reduction_operation),
      FLAGDNN_STATUS_SUCCESS,
      68);
  if (result != 0) {
    return result;
  }
  legacy_reduction_operation = NULL;
  result = require_status(
      flagdnnDestroyOperationDescriptor(reduction_operation),
      FLAGDNN_STATUS_SUCCESS,
      69);
  if (result != 0) {
    return result;
  }
  reduction_operation = NULL;

  result = require_status(
      flagdnnCreateOperationDescriptor(FLAGDNN_OPERATION_CONV2D_FPROP,
                                       &convolution_operation),
      FLAGDNN_STATUS_SUCCESS,
      70);
  if (result != 0 || convolution_operation == NULL) {
    return result == 0 ? 71 : result;
  }
  const int64_t pre_padding[2] = {1, 0};
  const int64_t post_padding[2] = {2, 3};
  const int64_t convolution_stride[2] = {1, 2};
  const int64_t convolution_dilation[2] = {1, 1};
  result = require_status(
      flagdnnSetConvolutionFpropOperationDescriptor(convolution_operation,
                                                    input,
                                                    output,
                                                    0,
                                                    pre_padding,
                                                    post_padding,
                                                    convolution_stride,
                                                    convolution_dilation,
                                                    1,
                                                    output),
      FLAGDNN_STATUS_NOT_SUPPORTED,
      75);
  if (result != 0) {
    return result;
  }
  result = require_status(
      flagdnnSetConvolutionFpropOperationDescriptor(convolution_operation,
                                                    input,
                                                    output,
                                                    4,
                                                    pre_padding,
                                                    post_padding,
                                                    convolution_stride,
                                                    convolution_dilation,
                                                    1,
                                                    output),
      FLAGDNN_STATUS_NOT_SUPPORTED,
      77);
  if (result != 0) {
    return result;
  }
  result = require_status(
      flagdnnSetConvolutionFpropOperationDescriptor(convolution_operation,
                                                    input,
                                                    output,
                                                    2,
                                                    pre_padding,
                                                    post_padding,
                                                    convolution_stride,
                                                    convolution_dilation,
                                                    1,
                                                    output),
      FLAGDNN_STATUS_SUCCESS,
      76);
  if (result != 0) {
    return result;
  }
  result = require_status(
      flagdnnSetConv2dFpropOperationDescriptorWithAsymmetricPadding(
          convolution_operation,
          input,
          output,
          pre_padding,
          NULL,
          convolution_stride,
          convolution_dilation,
          1,
          output),
      FLAGDNN_STATUS_INVALID_VALUE,
      72);
  if (result != 0) {
    return result;
  }
  result = require_status(
      flagdnnSetConv2dFpropOperationDescriptorWithAsymmetricPadding(
          convolution_operation,
          input,
          output,
          pre_padding,
          post_padding,
          convolution_stride,
          convolution_dilation,
          1,
          output),
      FLAGDNN_STATUS_SUCCESS,
      73);
  if (result != 0) {
    return result;
  }
  result = require_status(
      flagdnnDestroyOperationDescriptor(convolution_operation),
      FLAGDNN_STATUS_SUCCESS,
      74);
  if (result != 0) {
    return result;
  }
  convolution_operation = NULL;

  result = require_status(
      flagdnnCreateOperationDescriptorByName("Invalid-Kind",
                                              &generic_operation),
      FLAGDNN_STATUS_INVALID_VALUE,
      124);
  if (result != 0 || generic_operation != NULL) {
    return result == 0 ? 125 : result;
  }
  result = require_status(
      flagdnnCreateOperationDescriptorByName("relu", &generic_operation),
      FLAGDNN_STATUS_SUCCESS,
      126);
  if (result != 0 || generic_operation == NULL) {
    return result == 0 ? 127 : result;
  }
  operation_type = FLAGDNN_OPERATION_RELU;
  result = require_status(
      flagdnnGetOperationDescriptorType(generic_operation, &operation_type),
      FLAGDNN_STATUS_SUCCESS,
      128);
  if (result != 0 || operation_type != FLAGDNN_OPERATION_CUSTOM) {
    return result == 0 ? 129 : result;
  }
  result = require_status(
      flagdnnSetOperationDescriptorInput(
          generic_operation, "input", input, 2),
      FLAGDNN_STATUS_INVALID_VALUE,
      130);
  if (result != 0) {
    return result;
  }
  result = require_status(
      flagdnnSetOperationDescriptorInput(
          generic_operation, "input", input, 0),
      FLAGDNN_STATUS_SUCCESS,
      131);
  if (result != 0) {
    return result;
  }
  result = require_status(
      flagdnnSetOperationDescriptorInput(
          generic_operation, "input", input, 0),
      FLAGDNN_STATUS_INVALID_VALUE,
      132);
  if (result != 0) {
    return result;
  }
  result = require_status(
      flagdnnSetOperationDescriptorOutput(
          generic_operation, "output", output, 0),
      FLAGDNN_STATUS_SUCCESS,
      133);
  if (result != 0) {
    return result;
  }
  const int64_t generic_axes[2] = {0, 1};
  result = require_status(
      flagdnnSetOperationDescriptorAttributeInt64(
          generic_operation, "integer_value", 7),
      FLAGDNN_STATUS_SUCCESS,
      134);
  if (result != 0) {
    return result;
  }
  result = require_status(
      flagdnnSetOperationDescriptorAttributeDouble(
          generic_operation, "double_value", 0.5),
      FLAGDNN_STATUS_SUCCESS,
      135);
  if (result != 0) {
    return result;
  }
  result = require_status(
      flagdnnSetOperationDescriptorAttributeBoolean(
          generic_operation, "boolean_value", 1),
      FLAGDNN_STATUS_SUCCESS,
      136);
  if (result != 0) {
    return result;
  }
  result = require_status(
      flagdnnSetOperationDescriptorAttributeString(
          generic_operation, "string_value", "contract"),
      FLAGDNN_STATUS_SUCCESS,
      137);
  if (result != 0) {
    return result;
  }
  result = require_status(
      flagdnnSetOperationDescriptorAttributeInt64Array(
          generic_operation, "array_value", generic_axes, 2),
      FLAGDNN_STATUS_SUCCESS,
      138);
  if (result != 0) {
    return result;
  }
  result = require_status(
      flagdnnFinalizeOperationDescriptor(generic_operation),
      FLAGDNN_STATUS_SUCCESS,
      139);
  if (result != 0) {
    return result;
  }
  result = require_status(
      flagdnnSetOperationDescriptorAttributeInt64(
          generic_operation, "after_finalize", 1),
      FLAGDNN_STATUS_INVALID_VALUE,
      140);
  if (result != 0) {
    return result;
  }
  result = require_status(
      flagdnnDestroyOperationDescriptor(generic_operation),
      FLAGDNN_STATUS_SUCCESS,
      141);
  if (result != 0) {
    return result;
  }
  generic_operation = NULL;

  result = require_status(flagdnnCreateGraph(&graph),
                          FLAGDNN_STATUS_SUCCESS,
                          21);
  if (result != 0 || graph == NULL) {
    return result == 0 ? 22 : result;
  }
  result = require_status(
      flagdnnSetGraphName(graph, "native_api_contract"),
      FLAGDNN_STATUS_SUCCESS,
      122);
  if (result != 0) {
    return result;
  }
  result = require_status(flagdnnGraphAddOperation(graph, operation),
                          FLAGDNN_STATUS_SUCCESS,
                          23);
  if (result != 0) {
    return result;
  }
  size_t operation_count = 0;
  result = require_status(
      flagdnnGetGraphOperationCount(graph, &operation_count),
      FLAGDNN_STATUS_SUCCESS,
      24);
  if (result != 0 || operation_count != 1U) {
    return result == 0 ? 25 : result;
  }
  result = require_status(flagdnnValidateGraph(graph),
                          FLAGDNN_STATUS_SUCCESS,
                          26);
  if (result != 0) {
    return result;
  }
  result = require_status(flagdnnFinalizeGraph(graph),
                          FLAGDNN_STATUS_SUCCESS,
                          76);
  if (result != 0) {
    return result;
  }
  result = require_status(
      flagdnnSetGraphName(graph, "finalized"),
      FLAGDNN_STATUS_INVALID_VALUE,
      123);
  if (result != 0) {
    return result;
  }
  result = require_status(flagdnnGraphAddOperation(graph, operation),
                          FLAGDNN_STATUS_INVALID_VALUE,
                          27);
  if (result != 0) {
    return result;
  }
  result = require_status(flagdnnFinalizeGraph(graph),
                          FLAGDNN_STATUS_INVALID_VALUE,
                          28);
  if (result != 0) {
    return result;
  }

  result = require_status(
      flagdnnDestroyOperationDescriptor(operation),
      FLAGDNN_STATUS_SUCCESS,
      29);
  if (result != 0) {
    return result;
  }
  result = require_status(flagdnnDestroyTensorDescriptor(output),
                          FLAGDNN_STATUS_SUCCESS,
                          30);
  if (result != 0) {
    return result;
  }
  result = require_status(flagdnnDestroyTensorDescriptor(scalar),
                          FLAGDNN_STATUS_SUCCESS,
                          56);
  if (result != 0) {
    return result;
  }
  result = require_status(flagdnnDestroyTensorDescriptor(input),
                          FLAGDNN_STATUS_SUCCESS,
                          31);
  if (result != 0) {
    return result;
  }
  return require_status(
      flagdnnDestroyGraph(graph), FLAGDNN_STATUS_SUCCESS, 32);
}
