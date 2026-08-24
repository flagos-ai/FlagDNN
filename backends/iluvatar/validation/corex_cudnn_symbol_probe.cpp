// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include <cudnn.h>

#include <type_traits>

#define FLAGDNN_REQUIRE_CUDNN_DECLARATION(symbol)                              \
  static_assert(std::is_pointer_v<decltype(&symbol)>)

FLAGDNN_REQUIRE_CUDNN_DECLARATION(cudnnTransformTensor);
FLAGDNN_REQUIRE_CUDNN_DECLARATION(cudnnGetErrorString);
FLAGDNN_REQUIRE_CUDNN_DECLARATION(cudnnOpTensor);
FLAGDNN_REQUIRE_CUDNN_DECLARATION(cudnnReduceTensor);
FLAGDNN_REQUIRE_CUDNN_DECLARATION(cudnnActivationForward);
FLAGDNN_REQUIRE_CUDNN_DECLARATION(cudnnActivationBackward);
FLAGDNN_REQUIRE_CUDNN_DECLARATION(cudnnConvolutionForward);
FLAGDNN_REQUIRE_CUDNN_DECLARATION(cudnnConvolutionBackwardData);
FLAGDNN_REQUIRE_CUDNN_DECLARATION(cudnnConvolutionBackwardFilter);
FLAGDNN_REQUIRE_CUDNN_DECLARATION(cudnnBatchNormalizationForwardTraining);
FLAGDNN_REQUIRE_CUDNN_DECLARATION(cudnnBatchNormalizationForwardInference);
FLAGDNN_REQUIRE_CUDNN_DECLARATION(cudnnBatchNormalizationBackward);
FLAGDNN_REQUIRE_CUDNN_DECLARATION(cudnnCreateFlashAttnDescriptor);
FLAGDNN_REQUIRE_CUDNN_DECLARATION(cudnnDestroyFlashAttnDescriptor);
FLAGDNN_REQUIRE_CUDNN_DECLARATION(cudnnGetFlashAttnBuffers);
FLAGDNN_REQUIRE_CUDNN_DECLARATION(cudnnFlashAttnForward);
FLAGDNN_REQUIRE_CUDNN_DECLARATION(cudnnFlashAttnBackward);

#undef FLAGDNN_REQUIRE_CUDNN_DECLARATION
