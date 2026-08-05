#include <flagdnn/flagdnn.h>

#include <stddef.h>
#include <string.h>

static int configure_tensor(
    flagdnnTensorDescriptor_t descriptor,
    int64_t uid,
    const int64_t dimensions[4],
    const int64_t strides[4]) {
  return flagdnnSetTensorNdDescriptor(
             descriptor, uid, FLAGDNN_DATA_FLOAT32, 4, dimensions, strides) ==
                 FLAGDNN_STATUS_SUCCESS
             ? 0
             : 1;
}

static int validate_backward(
    const char* operation_name,
    const char* second_port,
    const char* output_port,
    flagdnnTensorDescriptor_t loss,
    flagdnnTensorDescriptor_t other,
    flagdnnTensorDescriptor_t output,
    int64_t convolution_mode,
    flagdnnStatus_t expected_status) {
  const int64_t padding[2] = {1, 1};
  const int64_t stride[2] = {1, 1};
  const int64_t dilation[2] = {1, 1};
  flagdnnOperationDescriptor_t operation = NULL;
  flagdnnGraph_t graph = NULL;
  flagdnnStatus_t status = flagdnnCreateOperationDescriptorByName(
      operation_name, &operation);
  if (status != FLAGDNN_STATUS_SUCCESS || operation == NULL) {
    return 10;
  }
#define REQUIRE_SUCCESS(expression, code)             \
  do {                                                \
    if ((expression) != FLAGDNN_STATUS_SUCCESS) {     \
      if (graph != NULL) {                           \
        flagdnnDestroyGraph(graph);                    \
      }                                                \
      flagdnnDestroyOperationDescriptor(operation);    \
      return (code);                                  \
    }                                                 \
  } while (0)
  REQUIRE_SUCCESS(flagdnnSetOperationDescriptorInput(
                      operation, "dy", loss, 0),
                  11);
  REQUIRE_SUCCESS(flagdnnSetOperationDescriptorInput(
                      operation, second_port, other, 0),
                  12);
  REQUIRE_SUCCESS(flagdnnSetOperationDescriptorOutput(
                      operation, output_port, output, 0),
                  13);
  REQUIRE_SUCCESS(flagdnnSetOperationDescriptorAttributeInt64(
                      operation, "spatial_rank", 2),
                  14);
  REQUIRE_SUCCESS(flagdnnSetOperationDescriptorAttributeInt64Array(
                      operation, "pre_padding", padding, 2),
                  15);
  REQUIRE_SUCCESS(flagdnnSetOperationDescriptorAttributeInt64Array(
                      operation, "post_padding", padding, 2),
                  16);
  REQUIRE_SUCCESS(flagdnnSetOperationDescriptorAttributeInt64Array(
                      operation, "stride", stride, 2),
                  17);
  REQUIRE_SUCCESS(flagdnnSetOperationDescriptorAttributeInt64Array(
                      operation, "dilation", dilation, 2),
                  18);
  REQUIRE_SUCCESS(flagdnnSetOperationDescriptorAttributeInt64(
                      operation, "groups", 1),
                  19);
  REQUIRE_SUCCESS(flagdnnSetOperationDescriptorAttributeInt64(
                      operation, "convolution_mode", convolution_mode),
                  20);
  REQUIRE_SUCCESS(flagdnnFinalizeOperationDescriptor(operation), 21);
  REQUIRE_SUCCESS(flagdnnCreateGraph(&graph), 22);
  REQUIRE_SUCCESS(flagdnnGraphAddOperation(graph, operation), 23);
  status = flagdnnValidateGraph(graph);
  if (flagdnnDestroyGraph(graph) != FLAGDNN_STATUS_SUCCESS ||
      flagdnnDestroyOperationDescriptor(operation) !=
          FLAGDNN_STATUS_SUCCESS) {
    return 24;
  }
#undef REQUIRE_SUCCESS
  return status == expected_status ? 0 : 25;
}

int main(void) {
  const int64_t loss_dimensions[4] = {1, 4, 5, 5};
  const int64_t loss_strides[4] = {100, 25, 5, 1};
  const int64_t filter_dimensions[4] = {4, 2, 3, 3};
  const int64_t filter_strides[4] = {18, 9, 3, 1};
  const int64_t image_dimensions[4] = {1, 2, 5, 5};
  const int64_t image_strides[4] = {50, 25, 5, 1};
  flagdnnTensorDescriptor_t loss = NULL;
  flagdnnTensorDescriptor_t filter = NULL;
  flagdnnTensorDescriptor_t image = NULL;
  if (flagdnnCreateTensorDescriptor(&loss) != FLAGDNN_STATUS_SUCCESS ||
      flagdnnCreateTensorDescriptor(&filter) != FLAGDNN_STATUS_SUCCESS ||
      flagdnnCreateTensorDescriptor(&image) != FLAGDNN_STATUS_SUCCESS ||
      configure_tensor(loss, 1, loss_dimensions, loss_strides) != 0 ||
      configure_tensor(filter, 2, filter_dimensions, filter_strides) != 0 ||
      configure_tensor(image, 3, image_dimensions, image_strides) != 0) {
    return 1;
  }

  int result = validate_backward("convolution_dgrad",
                                 "w",
                                 "dx",
                                 loss,
                                 filter,
                                 image,
                                 0,
                                 FLAGDNN_STATUS_SUCCESS);
  if (result == 0) {
    result = validate_backward("convolution_wgrad",
                               "x",
                               "dw",
                               loss,
                               image,
                               filter,
                               1,
                               FLAGDNN_STATUS_SUCCESS);
  }
  if (result == 0) {
    result = validate_backward("convolution_dgrad",
                               "w",
                               "dx",
                               loss,
                               filter,
                               image,
                               2,
                               FLAGDNN_STATUS_INVALID_VALUE);
  }

  if (flagdnnDestroyTensorDescriptor(image) != FLAGDNN_STATUS_SUCCESS ||
      flagdnnDestroyTensorDescriptor(filter) != FLAGDNN_STATUS_SUCCESS ||
      flagdnnDestroyTensorDescriptor(loss) != FLAGDNN_STATUS_SUCCESS) {
    return 2;
  }
  return result;
}
