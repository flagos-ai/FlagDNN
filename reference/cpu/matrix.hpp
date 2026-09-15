/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#ifndef FLAGDNN_REFERENCE_CPU_MATRIX_HPP_
#define FLAGDNN_REFERENCE_CPU_MATRIX_HPP_
#include <flagdnn/flagdnn.h>

#include <cstdint>
#include <span>
#include <vector>
namespace flagdnn::reference::cpu {
struct MatmulParameters {
  std::vector<std::int64_t> a_shape, b_shape, output_shape;
  flagdnnDataType_t input_type = FLAGDNN_DATA_FLOAT32;
  int input_precision = 0;
};
std::vector<float> evaluate_matmul(const MatmulParameters& parameters,
                                   std::span<const float> a,
                                   std::span<const float> b);
enum class ConvolutionDirection { kForward, kInputGradient, kWeightGradient };
struct ConvolutionParameters {
  std::vector<std::int64_t> x_shape, w_shape, y_shape;
  std::vector<std::int64_t> stride, pre_padding, dilation;
  ConvolutionDirection direction = ConvolutionDirection::kForward;
  std::int64_t groups = 1;
  int input_precision = 0;
  bool flip_kernel = false;
};
std::vector<float> evaluate_convolution(
    const ConvolutionParameters& parameters,
    const std::vector<std::vector<float>>& inputs);
struct ReductionParameters {
  std::vector<std::int64_t> input_shape;
  std::int32_t axis = 0;
  flagdnnReductionMode_t mode = FLAGDNN_REDUCTION_ADD;
};
std::vector<float> evaluate_reduction(const ReductionParameters& parameters,
                                      std::span<const float> input);
}  // namespace flagdnn::reference::cpu
#endif
