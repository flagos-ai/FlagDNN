/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "common/convolution.hpp"
#include <array>
#include <span>
namespace flagdnn::testing {
namespace {

ConvolutionTestCase private_fprop_stride2_low_ci_case() {
  ConvolutionTestCase result;
  result.name = "conv2d_fprop_fp32_hygon_private_stride2_low_ci_"
                "1x3x64x64_by_64x3x3x3";
  result.direction = ConvolutionDirection::kFprop;
  result.x = {
      63890, FLAGDNN_DATA_FLOAT32, {1, 3, 64, 64}, {12288, 4096, 64, 1}};
  result.w = {63891, FLAGDNN_DATA_FLOAT32, {64, 3, 3, 3}, {27, 9, 3, 1}};
  result.y = {
      63892, FLAGDNN_DATA_FLOAT32, {1, 64, 32, 32}, {65536, 1024, 32, 1}};
  result.pre_padding = {1, 1};
  result.post_padding = {1, 1};
  result.stride = {2, 2};
  result.dilation = {1, 1};
  result.groups = 1;
  result.absolute_tolerance = 5.0e-2;
  result.relative_tolerance = 5.0e-2;
  result.autotune = true;
  return result;
}

ConvolutionTestCase private_dgrad_stride2_block_pointer_case() {
  ConvolutionTestCase result;
  result.name = "conv2d_dgrad_fp16_hygon_private_stride2_block_pointer_"
                "tail_1x5x41x43_by_33x5x3x3";
  result.direction = ConvolutionDirection::kDgrad;
  result.x = {64090, FLAGDNN_DATA_FLOAT16, {1, 5, 41, 43}, {8815, 1763, 43, 1}};
  result.w = {64091, FLAGDNN_DATA_FLOAT16, {33, 5, 3, 3}, {45, 9, 3, 1}};
  result.y = {
      64092, FLAGDNN_DATA_FLOAT16, {1, 33, 21, 22}, {15246, 462, 22, 1}};
  result.pre_padding = {1, 1};
  result.post_padding = {1, 1};
  result.stride = {2, 2};
  result.dilation = {1, 1};
  result.groups = 1;
  result.absolute_tolerance = 2.0e-2;
  result.relative_tolerance = 7.0e-2;
  result.autotune = true;
  return result;
}

ConvolutionTestCase private_wgrad_1x1_split_tail_case() {
  ConvolutionTestCase result;
  result.name = "conv2d_wgrad_fp16_hygon_private_1x1_split_tail_"
                "8x64x28x28_by_128x64x1x1";
  result.direction = ConvolutionDirection::kWgrad;
  result.x = {
      63990, FLAGDNN_DATA_FLOAT16, {8, 64, 28, 28}, {50176, 784, 28, 1}};
  result.w = {63991, FLAGDNN_DATA_FLOAT16, {128, 64, 1, 1}, {64, 1, 1, 1}};
  result.y = {
      63992, FLAGDNN_DATA_FLOAT16, {8, 128, 28, 28}, {100352, 784, 28, 1}};
  result.pre_padding = {0, 0};
  result.post_padding = {0, 0};
  result.stride = {1, 1};
  result.dilation = {1, 1};
  result.groups = 1;
  result.absolute_tolerance = 8.0e-2;
  result.relative_tolerance = 1.0e-1;
  result.autotune = true;
  return result;
}

ConvolutionTestCase private_wgrad_stem_split_case() {
  ConvolutionTestCase result;
  result.name = "conv2d_wgrad_fp16_hygon_private_stem_split_"
                "1x3x640x640_by_96x3x3x3";
  result.direction = ConvolutionDirection::kWgrad;
  result.x = {
      64390, FLAGDNN_DATA_FLOAT16, {1, 3, 640, 640}, {1228800, 409600, 640, 1}};
  result.w = {64391, FLAGDNN_DATA_FLOAT16, {96, 3, 3, 3}, {27, 9, 3, 1}};
  result.y = {64392,
              FLAGDNN_DATA_FLOAT16,
              {1, 96, 320, 320},
              {9830400, 102400, 320, 1}};
  result.pre_padding = {1, 1};
  result.post_padding = {1, 1};
  result.stride = {2, 2};
  result.dilation = {1, 1};
  result.groups = 1;
  result.absolute_tolerance = 8.0e-2;
  result.relative_tolerance = 1.0e-1;
  result.autotune = true;
  return result;
}

ConvolutionTestCase private_wgrad_multirow_split_tail_case() {
  ConvolutionTestCase result;
  result.name = "conv2d_wgrad_fp16_hygon_private_multirow_split_tail_"
                "2x5x35x37_by_33x5x3x3";
  result.direction = ConvolutionDirection::kWgrad;
  result.x = {64190, FLAGDNN_DATA_FLOAT16, {2, 5, 35, 37}, {6475, 1295, 37, 1}};
  result.w = {64191, FLAGDNN_DATA_FLOAT16, {33, 5, 3, 3}, {45, 9, 3, 1}};
  result.y = {
      64192, FLAGDNN_DATA_FLOAT16, {2, 33, 18, 19}, {11286, 342, 19, 1}};
  result.pre_padding = {1, 1};
  result.post_padding = {1, 1};
  result.stride = {2, 2};
  result.dilation = {1, 1};
  result.groups = 1;
  result.absolute_tolerance = 8.0e-2;
  result.relative_tolerance = 1.0e-1;
  result.autotune = true;
  return result;
}

ConvolutionTestCase private_wgrad_p5_rowmajor_tail_case() {
  ConvolutionTestCase result;
  result.name = "conv2d_wgrad_fp16_hygon_private_p5_rowmajor_tail_"
                "1x5x40x40_by_33x5x3x3";
  result.direction = ConvolutionDirection::kWgrad;
  result.x = {64290, FLAGDNN_DATA_FLOAT16, {1, 5, 40, 40}, {8000, 1600, 40, 1}};
  result.w = {64291, FLAGDNN_DATA_FLOAT16, {33, 5, 3, 3}, {45, 9, 3, 1}};
  result.y = {
      64292, FLAGDNN_DATA_FLOAT16, {1, 33, 20, 20}, {13200, 400, 20, 1}};
  result.pre_padding = {1, 1};
  result.post_padding = {1, 1};
  result.stride = {2, 2};
  result.dilation = {1, 1};
  result.groups = 1;
  result.absolute_tolerance = 8.0e-2;
  result.relative_tolerance = 1.0e-1;
  result.autotune = true;
  return result;
}

} // namespace

int run_private_convolution_regressions(int argc, char **argv) {
  const std::array cases = {private_fprop_stride2_low_ci_case(),
                            private_dgrad_stride2_block_pointer_case(),
                            private_wgrad_1x1_split_tail_case(),
                            private_wgrad_stem_split_case(),
                            private_wgrad_multirow_split_tail_case(),
                            private_wgrad_p5_rowmajor_tail_case()};
  bool executed = false;
  for (const auto &value : cases) {
    const int status = run_convolution_functional_test(
        argc, argv, std::span<const ConvolutionTestCase>(&value, 1),
        value.direction);
    if (status != 0 && status != 77)
      return status;
    executed |= status == 0;
  }
  return executed ? 0 : 77;
}
} // namespace flagdnn::testing
int main(int argc, char **argv) {
  return flagdnn::testing::run_private_convolution_regressions(argc, argv);
}
