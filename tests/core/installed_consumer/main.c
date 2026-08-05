#include <flagdnn/flagdnn.h>

#include <string.h>

int main(void) {
  flagdnnStatus_t (*set_tensor_virtual)(
      flagdnnTensorDescriptor_t,
      int32_t) = flagdnnSetTensorDescriptorVirtual;
  flagdnnStatus_t (*get_tensor_virtual)(
      flagdnnTensorDescriptor_t,
      int32_t*) = flagdnnGetTensorDescriptorVirtual;
  flagdnnStatus_t (*set_pointwise)(flagdnnOperationDescriptor_t,
                                   flagdnnTensorDescriptor_t,
                                   flagdnnPointwiseMode_t,
                                   flagdnnTensorDescriptor_t) =
      flagdnnSetPointwiseUnaryOperationDescriptor;
  flagdnnStatus_t (*set_pointwise_with_attributes)(
      flagdnnOperationDescriptor_t,
      flagdnnTensorDescriptor_t,
      flagdnnPointwiseMode_t,
      flagdnnTensorDescriptor_t,
      const flagdnnPointwiseAttributes_t*) =
      flagdnnSetPointwiseUnaryOperationDescriptorWithAttributes;
  flagdnnStatus_t (*set_binary_pointwise)(
      flagdnnOperationDescriptor_t,
      flagdnnTensorDescriptor_t,
      flagdnnTensorDescriptor_t,
      flagdnnPointwiseMode_t,
      flagdnnTensorDescriptor_t,
      double) = flagdnnSetPointwiseBinaryOperationDescriptorWithAlpha;
  flagdnnStatus_t (*set_ternary_pointwise)(
      flagdnnOperationDescriptor_t,
      flagdnnTensorDescriptor_t,
      flagdnnTensorDescriptor_t,
      flagdnnTensorDescriptor_t,
      flagdnnPointwiseMode_t,
      flagdnnTensorDescriptor_t) =
      flagdnnSetPointwiseTernaryOperationDescriptor;
  flagdnnStatus_t (*set_reduction)(flagdnnOperationDescriptor_t,
                                   flagdnnTensorDescriptor_t,
                                   flagdnnReductionMode_t,
                                   int32_t,
                                   int32_t,
                                   flagdnnTensorDescriptor_t) =
      flagdnnSetReductionOperationDescriptor;
  flagdnnStatus_t (*set_asymmetric_conv)(
      flagdnnOperationDescriptor_t,
      flagdnnTensorDescriptor_t,
      flagdnnTensorDescriptor_t,
      const int64_t[2],
      const int64_t[2],
      const int64_t[2],
      const int64_t[2],
      int64_t,
      flagdnnTensorDescriptor_t) =
      flagdnnSetConv2dFpropOperationDescriptorWithAsymmetricPadding;
  flagdnnStatus_t (*set_convolution)(flagdnnOperationDescriptor_t,
                                    flagdnnTensorDescriptor_t,
                                    flagdnnTensorDescriptor_t,
                                    int32_t,
                                    const int64_t[],
                                    const int64_t[],
                                    const int64_t[],
                                    const int64_t[],
                                    int64_t,
                                    flagdnnTensorDescriptor_t) =
      flagdnnSetConvolutionFpropOperationDescriptor;
  const flagdnnPointwiseAttributes_t pointwise_attributes =
      FLAGDNN_POINTWISE_ATTRIBUTES_INITIALIZER;
  return flagdnnGetVersion() == 100U &&
                 strcmp(flagdnnGetVersionString(),
                        FLAGDNN_VERSION_STRING) == 0 &&
                 set_tensor_virtual != NULL &&
                 get_tensor_virtual != NULL &&
                 set_pointwise != NULL &&
                 set_pointwise_with_attributes != NULL &&
                 set_binary_pointwise != NULL &&
                 set_ternary_pointwise != NULL &&
                 set_reduction != NULL &&
                 set_asymmetric_conv != NULL &&
                 set_convolution != NULL &&
                 FLAGDNN_OPERATION_REDUCTION ==
                     FLAGDNN_OPERATION_REDUCTION_SUM &&
                 FLAGDNN_OPERATION_CONVOLUTION_FPROP ==
                     FLAGDNN_OPERATION_CONV2D_FPROP &&
                 FLAGDNN_POINTWISE_SUB == 17 &&
                 FLAGDNN_POINTWISE_POW == 23 &&
                 FLAGDNN_DATA_BOOLEAN == 3 &&
                 FLAGDNN_POINTWISE_LOGICAL_NOT == 24 &&
                 FLAGDNN_POINTWISE_CMP_EQ == 25 &&
                 FLAGDNN_POINTWISE_LOGICAL_OR == 32 &&
                 FLAGDNN_POINTWISE_SIGMOID_FWD == 33 &&
                 FLAGDNN_POINTWISE_TANH_FWD == 34 &&
                 FLAGDNN_POINTWISE_ELU_FWD == 35 &&
                 FLAGDNN_POINTWISE_GELU_FWD == 36 &&
                 FLAGDNN_POINTWISE_SOFTPLUS_FWD == 37 &&
                 FLAGDNN_POINTWISE_SWISH_FWD == 38 &&
                 FLAGDNN_POINTWISE_GELU_APPROX_TANH_FWD == 39 &&
                 FLAGDNN_POINTWISE_SIGMOID_BWD == 40 &&
                 FLAGDNN_POINTWISE_BINARY_SELECT == 41 &&
                 pointwise_attributes.struct_size ==
                     sizeof(flagdnnPointwiseAttributes_t) &&
                 pointwise_attributes.version ==
                     FLAGDNN_POINTWISE_ATTRIBUTES_VERSION
             ? 0
             : 1;
}
