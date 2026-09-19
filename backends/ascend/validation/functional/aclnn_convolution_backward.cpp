/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "validation/functional/aclnn_extended.hpp"
#include "validation/functional/aclnn_plan.hpp"
#include <aclnnop/aclnn_constant_pad_nd.h>
#include <aclnnop/aclnn_convolution_backward.h>
#include <aclnnop/aclnn_flip.h>
#include <aclnnop/aclnn_slice.h>
namespace flagdnn::testing {
std::unique_ptr<TestExecutable>
build_aclnn_convolution_backward(const ConvolutionTestCase &c) {
  using namespace ascend;
  if (c.input_precision == 2)
    throw Unsupported("Ascend 910B ACLNN exposes HF32, not the CUDA TF32 "
                      "input-precision contract");
  const bool dx = c.direction == ConvolutionDirection::kDgrad;
  std::vector<TestTensor> specs = dx ? std::vector<TestTensor>{c.y, c.w, c.x}
                                     : std::vector<TestTensor>{c.y, c.x, c.w};
  return std::make_unique<Plan>(std::move(specs), [c, dx](Plan &p) {
    const auto rank = c.stride.size();
    auto *dy = p.ports[0];
    auto *input = dx ? p.temporary(c.x) : p.ports[1];
    auto *weight = dx ? p.ports[1] : p.temporary(c.w);
    auto *grad_input = dx ? p.ports[2] : nullptr;
    auto *grad_weight = dx ? nullptr : p.ports[2];
    std::vector<std::int64_t> flip_axes;
    for (std::size_t i = 0; i < rank; ++i)
      flip_axes.push_back(i + 2);
    auto *flip = p.array(flip_axes);
    if (dx && c.mode == ConvolutionMode::kConvolution) {
      auto *reversed = p.temporary(c.w);
      p.add(aclnnFlip, [&](auto *w, auto **e) {
        return aclnnFlipGetWorkspaceSize(weight, flip, reversed, w, e);
      });
      weight = reversed;
    }
    if (!dx && c.mode == ConvolutionMode::kConvolution)
      grad_weight = p.temporary(c.w);
    auto padding = c.pre_padding;
    const bool asymmetric = c.pre_padding != c.post_padding;
    if (asymmetric) {
      auto padded = c.x;
      std::vector<std::int64_t> pads;
      for (std::size_t i = rank; i > 0; --i) {
        pads.push_back(c.pre_padding[i - 1]);
        pads.push_back(c.post_padding[i - 1]);
        padded.dimensions[i + 1] +=
            c.pre_padding[i - 1] + c.post_padding[i - 1];
      }
      padded.strides = dense_strides(padded.dimensions);
      auto *expanded = p.temporary(padded);
      if (!dx) {
        auto *pad = p.array(pads);
        auto *zero = p.scalar(0);
        p.add(aclnnConstantPadNd, [&](auto *w, auto **e) {
          return aclnnConstantPadNdGetWorkspaceSize(input, pad, zero, expanded,
                                                    w, e);
        });
      }
      input = expanded;
      if (dx)
        grad_input = p.temporary(padded);
      padding.assign(rank, 0);
    }
    const auto format = rank == 1   ? ACL_FORMAT_NCL
                        : rank == 2 ? ACL_FORMAT_NCHW
                                    : ACL_FORMAT_NCDHW;
    dy = p.formatted(dy, format);
    input = p.formatted(input, format);
    weight = p.formatted(weight, format);
    if (grad_input)
      grad_input = p.formatted(grad_input, format);
    if (grad_weight)
      grad_weight = p.formatted(grad_weight, format);
    auto *strides = p.array(c.stride);
    auto *pads = p.array(padding);
    auto *dilation = p.array(c.dilation);
    auto *zeros = p.array(std::vector<std::int64_t>(rank, 0));
    auto *mask = p.output_mask(dx, !dx, false);
    p.add(aclnnConvolutionBackward, [&](auto *w, auto **e) {
      return aclnnConvolutionBackwardGetWorkspaceSize(
          dy, input, weight, nullptr, strides, pads, dilation, false, zeros,
          static_cast<int>(c.groups), mask, 0, grad_input, grad_weight, nullptr,
          w, e);
    });
    if (dx && asymmetric) {
      auto shape = c.x.dimensions;
      for (std::size_t i = 0; i < rank; ++i)
        shape[i + 2] += c.pre_padding[i] + c.post_padding[i];
      for (std::size_t i = 0; i < rank; ++i) {
        shape[i + 2] = c.x.dimensions[i + 2];
        auto *output = i + 1 == rank
                           ? p.ports[2]
                           : p.temporary(TestTensor{0, c.x.data_type, shape,
                                                    dense_strides(shape)});
        output = p.formatted(output, format);
        auto *source = grad_input;
        p.add(aclnnSlice, [&](auto *w, auto **e) {
          return aclnnSliceGetWorkspaceSize(
              source, i + 2, c.pre_padding[i],
              c.pre_padding[i] + c.x.dimensions[i + 2], 1, output, w, e);
        });
        grad_input = output;
      }
    }
    if (!dx && c.mode == ConvolutionMode::kConvolution)
      p.add(aclnnFlip, [&](auto *w, auto **e) {
        return aclnnFlipGetWorkspaceSize(grad_weight, flip,
                                         p.formatted(p.ports[2], format), w, e);
      });
  });
}
} // namespace flagdnn::testing
