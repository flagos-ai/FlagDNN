#include <flagdnn/flagdnn.h>

#include <stddef.h>

static int configure_tensor(
    flagdnnTensorDescriptor_t descriptor,
    int64_t uid,
    flagdnnDataType_t data_type,
    const int64_t dimensions[4],
    const int64_t strides[4]) {
  return flagdnnSetTensorNdDescriptor(
             descriptor, uid, data_type, 4, dimensions, strides) ==
                 FLAGDNN_STATUS_SUCCESS
             ? 0
             : 1;
}

static int validate_batchnorm_inference(
    flagdnnTensorDescriptor_t x,
    flagdnnTensorDescriptor_t mean,
    flagdnnTensorDescriptor_t inv_variance,
    flagdnnTensorDescriptor_t scale,
    flagdnnTensorDescriptor_t bias,
    flagdnnTensorDescriptor_t y,
    flagdnnStatus_t expected_status) {
  flagdnnOperationDescriptor_t operation = NULL;
  flagdnnGraph_t graph = NULL;
  flagdnnStatus_t status = flagdnnCreateOperationDescriptorByName(
      "batchnorm_inference", &operation);
  if (status != FLAGDNN_STATUS_SUCCESS || operation == NULL) {
    return 10;
  }
#define REQUIRE_SUCCESS(expression, code)                 \
  do {                                                    \
    if ((expression) != FLAGDNN_STATUS_SUCCESS) {         \
      if (graph != NULL) {                                \
        flagdnnDestroyGraph(graph);                       \
      }                                                   \
      flagdnnDestroyOperationDescriptor(operation);       \
      return (code);                                      \
    }                                                     \
  } while (0)
  REQUIRE_SUCCESS(
      flagdnnSetOperationDescriptorName(operation, "batchnorm_inference"),
      11);
  REQUIRE_SUCCESS(flagdnnSetOperationDescriptorComputeDataType(
                      operation, FLAGDNN_DATA_FLOAT32),
                  12);
  REQUIRE_SUCCESS(
      flagdnnSetOperationDescriptorInput(operation, "x", x, 0), 13);
  REQUIRE_SUCCESS(
      flagdnnSetOperationDescriptorInput(operation, "mean", mean, 0), 14);
  REQUIRE_SUCCESS(flagdnnSetOperationDescriptorInput(
                      operation, "inv_variance", inv_variance, 0),
                  15);
  REQUIRE_SUCCESS(
      flagdnnSetOperationDescriptorInput(operation, "scale", scale, 0),
      16);
  REQUIRE_SUCCESS(
      flagdnnSetOperationDescriptorInput(operation, "bias", bias, 0),
      17);
  REQUIRE_SUCCESS(
      flagdnnSetOperationDescriptorOutput(operation, "y", y, 0), 18);
  REQUIRE_SUCCESS(flagdnnFinalizeOperationDescriptor(operation), 19);
  REQUIRE_SUCCESS(flagdnnCreateGraph(&graph), 20);
  REQUIRE_SUCCESS(flagdnnGraphAddOperation(graph, operation), 21);
  status = flagdnnValidateGraph(graph);
  if (flagdnnDestroyGraph(graph) != FLAGDNN_STATUS_SUCCESS ||
      flagdnnDestroyOperationDescriptor(operation) !=
          FLAGDNN_STATUS_SUCCESS) {
    return 22;
  }
#undef REQUIRE_SUCCESS
  return status == expected_status ? 0 : 23;
}

int main(void) {
  const int64_t data_dimensions[4] = {2, 8, 4, 4};
  const int64_t data_strides[4] = {128, 16, 4, 1};
  const int64_t parameter_dimensions[4] = {1, 8, 1, 1};
  const int64_t parameter_strides[4] = {8, 1, 1, 1};
  const int64_t bad_parameter_dimensions[4] = {1, 7, 1, 1};
  const int64_t bad_parameter_strides[4] = {7, 1, 1, 1};

  flagdnnTensorDescriptor_t x = NULL;
  flagdnnTensorDescriptor_t mean = NULL;
  flagdnnTensorDescriptor_t inv_variance = NULL;
  flagdnnTensorDescriptor_t scale = NULL;
  flagdnnTensorDescriptor_t bias = NULL;
  flagdnnTensorDescriptor_t bad_bias = NULL;
  flagdnnTensorDescriptor_t y = NULL;
  if (flagdnnCreateTensorDescriptor(&x) != FLAGDNN_STATUS_SUCCESS ||
      flagdnnCreateTensorDescriptor(&mean) != FLAGDNN_STATUS_SUCCESS ||
      flagdnnCreateTensorDescriptor(&inv_variance) !=
          FLAGDNN_STATUS_SUCCESS ||
      flagdnnCreateTensorDescriptor(&scale) != FLAGDNN_STATUS_SUCCESS ||
      flagdnnCreateTensorDescriptor(&bias) != FLAGDNN_STATUS_SUCCESS ||
      flagdnnCreateTensorDescriptor(&bad_bias) != FLAGDNN_STATUS_SUCCESS ||
      flagdnnCreateTensorDescriptor(&y) != FLAGDNN_STATUS_SUCCESS ||
      configure_tensor(x,
                       1,
                       FLAGDNN_DATA_FLOAT16,
                       data_dimensions,
                       data_strides) != 0 ||
      configure_tensor(mean,
                       2,
                       FLAGDNN_DATA_FLOAT32,
                       parameter_dimensions,
                       parameter_strides) != 0 ||
      configure_tensor(inv_variance,
                       3,
                       FLAGDNN_DATA_FLOAT32,
                       parameter_dimensions,
                       parameter_strides) != 0 ||
      configure_tensor(scale,
                       4,
                       FLAGDNN_DATA_FLOAT32,
                       parameter_dimensions,
                       parameter_strides) != 0 ||
      configure_tensor(bias,
                       5,
                       FLAGDNN_DATA_FLOAT32,
                       parameter_dimensions,
                       parameter_strides) != 0 ||
      configure_tensor(bad_bias,
                       6,
                       FLAGDNN_DATA_FLOAT32,
                       bad_parameter_dimensions,
                       bad_parameter_strides) != 0 ||
      configure_tensor(y,
                       7,
                       FLAGDNN_DATA_FLOAT16,
                       data_dimensions,
                       data_strides) != 0) {
    return 1;
  }

  int result = validate_batchnorm_inference(
      x, mean, inv_variance, scale, bias, y, FLAGDNN_STATUS_SUCCESS);
  if (result == 0) {
    result = validate_batchnorm_inference(
        x,
        mean,
        inv_variance,
        scale,
        bad_bias,
        y,
        FLAGDNN_STATUS_INVALID_VALUE);
  }

  if (flagdnnDestroyTensorDescriptor(y) != FLAGDNN_STATUS_SUCCESS ||
      flagdnnDestroyTensorDescriptor(bad_bias) != FLAGDNN_STATUS_SUCCESS ||
      flagdnnDestroyTensorDescriptor(bias) != FLAGDNN_STATUS_SUCCESS ||
      flagdnnDestroyTensorDescriptor(scale) != FLAGDNN_STATUS_SUCCESS ||
      flagdnnDestroyTensorDescriptor(inv_variance) !=
          FLAGDNN_STATUS_SUCCESS ||
      flagdnnDestroyTensorDescriptor(mean) != FLAGDNN_STATUS_SUCCESS ||
      flagdnnDestroyTensorDescriptor(x) != FLAGDNN_STATUS_SUCCESS) {
    return 2;
  }
  return result;
}
