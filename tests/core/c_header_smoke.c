#include <flagdnn/flagdnn.h>

#include <stddef.h>
#include <string.h>

int main(void) {
  if (FLAGDNN_OPERATION_REDUCTION != FLAGDNN_OPERATION_REDUCTION_SUM ||
      FLAGDNN_REDUCTION_ADD != FLAGDNN_REDUCTION_SUM ||
      FLAGDNN_OPERATION_POINTWISE == FLAGDNN_OPERATION_CONVOLUTION_FPROP ||
      FLAGDNN_POINTWISE_ABS == FLAGDNN_POINTWISE_NOT_SET ||
      FLAGDNN_DATA_BOOLEAN != 3 || FLAGDNN_POINTWISE_SUB != 17 ||
      FLAGDNN_POINTWISE_POW != 23 ||
      FLAGDNN_POINTWISE_LOGICAL_NOT != 24 ||
      FLAGDNN_POINTWISE_CMP_EQ != 25 ||
      FLAGDNN_POINTWISE_LOGICAL_OR != 32 ||
      FLAGDNN_POINTWISE_SIGMOID_FWD != 33 ||
      FLAGDNN_POINTWISE_TANH_FWD != 34 ||
      FLAGDNN_POINTWISE_ELU_FWD != 35 ||
      FLAGDNN_POINTWISE_GELU_FWD != 36 ||
      FLAGDNN_POINTWISE_SOFTPLUS_FWD != 37 ||
      FLAGDNN_POINTWISE_SWISH_FWD != 38 ||
      FLAGDNN_POINTWISE_GELU_APPROX_TANH_FWD != 39 ||
      FLAGDNN_POINTWISE_SIGMOID_BWD != 40 ||
      FLAGDNN_POINTWISE_BINARY_SELECT != 41 ||
      FLAGDNN_BUILD_OPTION_AUTOTUNE != UINT64_C(4) ||
      FLAGDNN_BUILD_OPTION_FLAGS_ALL != UINT64_C(7)) {
    return 5;
  }
  if (flagdnnGetVersion() != 100U) {
    return 1;
  }
  if (strcmp(flagdnnGetVersionString(), FLAGDNN_VERSION_STRING) != 0) {
    return 2;
  }
  if (flagdnnGetErrorString(FLAGDNN_STATUS_SUCCESS) == NULL) {
    return 3;
  }
  const flagdnnBuildOptions_t options = FLAGDNN_BUILD_OPTIONS_INITIALIZER;
  if (options.struct_size != sizeof(flagdnnBuildOptions_t) ||
      options.version != FLAGDNN_BUILD_OPTIONS_VERSION || options.flags != 0U) {
    return 4;
  }
  const flagdnnPointwiseAttributes_t pointwise =
      FLAGDNN_POINTWISE_ATTRIBUTES_INITIALIZER;
  if (pointwise.struct_size != sizeof(flagdnnPointwiseAttributes_t) ||
      pointwise.version != FLAGDNN_POINTWISE_ATTRIBUTES_VERSION ||
      pointwise.flags != 0U || pointwise.relu_lower_clip != 0.0 ||
      pointwise.relu_upper_clip != 0.0 ||
      pointwise.relu_lower_clip_slope != 0.0 ||
      pointwise.swish_beta != 1.0 || pointwise.elu_alpha != 1.0 ||
      pointwise.softplus_beta != 1.0 ||
      FLAGDNN_POINTWISE_ATTRIBUTE_FLAGS_ALL != UINT64_C(63)) {
    return 6;
  }
  return 0;
}
