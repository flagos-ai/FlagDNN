/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "validation/functional/aclnn_extended.hpp"
#include "validation/functional/aclnn_math.hpp"
#include <aclnnop/aclnn_avgpool3d.h>
#include <aclnnop/aclnn_constant_pad_nd.h>
#include <aclnnop/aclnn_max_pool.h>
#include <aclnnop/aclnn_upsample_bilinear_2d.h>
#include <limits>
namespace flagdnn::testing {
std::unique_ptr<TestExecutable>
build_aclnn_resample(const ResampleTestCase &c) {
  using namespace ascend;
  auto specs = c.inputs;
  specs.insert(specs.end(), c.outputs.begin(), c.outputs.end());
  if (c.outputs.size() != 1)
    throw std::runtime_error(
        "Ascend MaxPool window-relative index reference is "
        "not implemented; the common catalog has no index outputs");
  return std::make_unique<Plan>(std::move(specs), [c](Plan &p) {
    const auto rank = c.inputs[0].dimensions.size();
    const auto format = rank == 3   ? ACL_FORMAT_NCL
                        : rank == 4 ? ACL_FORMAT_NCHW
                                    : ACL_FORMAT_NCDHW;
    Math m{p};
    const bool promote = c.inputs[0].data_type == FLAGDNN_DATA_BFLOAT16;
    const auto input =
        promote ? m.input(0) : Math::Value{p.ports[0], c.inputs[0].dimensions};
    const auto output = promote
                            ? m.temp(c.outputs[0].dimensions)
                            : Math::Value{p.ports[1], c.outputs[0].dimensions};
    auto *x = p.formatted(input.tensor, format);
    auto *y = p.formatted(output.tensor, format);
    if (c.mode == 3) {
      auto *output =
          p.array({c.outputs[0].dimensions[2], c.outputs[0].dimensions[3]});
      p.add(aclnnUpsampleBilinear2d, [&](auto *w, auto **e) {
        return aclnnUpsampleBilinear2dGetWorkspaceSize(
            x, output, c.align_corners, 0, 0, y, w, e);
      });
    } else if (c.mode == 5) {
      auto pads = c.pre;
      pads.insert(pads.end(), c.post.begin(), c.post.end());
      bool explicit_padding = false;
      for (std::size_t i = 0; i < c.window.size(); ++i)
        explicit_padding |= c.pre[i] + c.post[i] > c.window[i];
      if (explicit_padding) {
        auto shape = input.shape;
        std::vector<std::int64_t> padding;
        for (std::size_t i = 0; i < c.window.size(); ++i)
          shape[i + 2] += c.pre[i] + c.post[i];
        for (std::size_t i = c.window.size(); i > 0; --i) {
          padding.push_back(c.pre[i - 1]);
          padding.push_back(c.post[i - 1]);
        }
        auto *padded = p.formatted(
            p.temporary(TestTensor{
                0, promote ? FLAGDNN_DATA_FLOAT32 : c.inputs[0].data_type,
                shape, dense_strides(shape)}),
            format);
        auto *amounts = p.array(padding);
        auto *fill = p.scalar(-std::numeric_limits<double>::infinity());
        p.add(aclnnConstantPadNd, [&](auto *w, auto **e) {
          return aclnnConstantPadNdGetWorkspaceSize(x, amounts, fill, padded, w,
                                                    e);
        });
        x = padded;
        std::fill(pads.begin(), pads.end(), 0);
      }
      auto *padding = p.array(pads);
      auto *kernel = p.array(c.window);
      auto *stride = p.array(c.stride);
      auto *dilation = p.array(std::vector<std::int64_t>(c.window.size(), 1));
      p.add(aclnnMaxPool, [&](auto *w, auto **e) {
        return aclnnMaxPoolGetWorkspaceSize(x, kernel, stride, 0, padding,
                                            dilation, 0, y, w, e);
      });
    } else {
      if (c.pre != c.post)
        throw Unsupported("ACLNN AvgPool requires symmetric padding");
      auto window = c.window, strides = c.stride, padding = c.pre;
      if (window.size() == 1) {
        auto xd = c.inputs[0].dimensions, yd = c.outputs[0].dimensions;
        xd.insert(xd.begin() + 2, 1);
        yd.insert(yd.begin() + 2, 1);
        x = p.formatted(p.alias(input.tensor, xd, dense_strides(xd)),
                        ACL_FORMAT_NCHW);
        y = p.formatted(p.alias(output.tensor, yd, dense_strides(yd)),
                        ACL_FORMAT_NCHW);
        window.insert(window.begin(), 1);
        strides.insert(strides.begin(), 1);
        padding.insert(padding.begin(), 0);
      }
      // The CANN 9 AvgPool2d padded path rejects repeatable executors.
      // A unit-depth AvgPool3d has identical arithmetic and can be prepared
      // once for both correctness repetitions and steady-state timing.
      if (window.size() == 2) {
        auto xd = c.inputs[0].dimensions, yd = c.outputs[0].dimensions;
        while (xd.size() < 5)
          xd.insert(xd.begin() + 2, 1);
        while (yd.size() < 5)
          yd.insert(yd.begin() + 2, 1);
        x = p.formatted(p.alias(input.tensor, xd, dense_strides(xd)),
                        ACL_FORMAT_NCDHW);
        y = p.formatted(p.alias(output.tensor, yd, dense_strides(yd)),
                        ACL_FORMAT_NCDHW);
        window.insert(window.begin(), 1);
        strides.insert(strides.begin(), 1);
        padding.insert(padding.begin(), 0);
      }
      auto *kernel = p.array(window);
      auto *stride = p.array(strides);
      auto *pads = p.array(padding);
      p.add(aclnnAvgPool3d, [&](auto *w, auto **e) {
        return aclnnAvgPool3dGetWorkspaceSize(x, kernel, stride, pads, false,
                                              c.mode == 2, 0, y, w, e);
      });
    }
    if (promote)
      m.output(output, 1);
  });
}
} // namespace flagdnn::testing
