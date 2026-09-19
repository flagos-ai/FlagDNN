/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "validation/functional/aclnn_extended.hpp"
#include "validation/functional/aclnn_math.hpp"
#include "validation/functional/aclnn_plan.hpp"
#include <aclnnop/aclnn_elu_backward.h>
#include <aclnnop/aclnn_gelu_backward_v2.h>
#include <aclnnop/aclnn_leaky_relu.h>
#include <aclnnop/aclnn_leaky_relu_backward.h>
#include <aclnnop/aclnn_lt_scalar.h>
#include <aclnnop/aclnn_mul.h>
#include <aclnnop/aclnn_silu_backward.h>
#include <aclnnop/aclnn_softplus_backward.h>
#include <aclnnop/aclnn_tanh.h>
#include <aclnnop/aclnn_tanh_backward.h>
namespace flagdnn::testing {
std::unique_ptr<TestExecutable>
build_aclnn_activation_backward(const PointwiseTestCase &c) {
  using namespace ascend;
  auto specs = c.inputs;
  specs.push_back(c.output);
  return std::make_unique<Plan>(std::move(specs), [c](Plan &p) {
    auto *dy = p.ports[0];
    auto *x = p.ports[1];
    auto *dx = p.ports[2];
    switch (c.mode) {
    case FLAGDNN_POINTWISE_RELU_BWD: {
      if (c.attributes.relu_lower_clip != 0.0 ||
          (c.attributes.flags & FLAGDNN_POINTWISE_ATTRIBUTE_RELU_UPPER_CLIP)) {
        Math m{p};
        auto input = m.input(1), grad = m.input(0);
        auto shifted = m.shift(input, -c.attributes.relu_lower_clip);
        auto result = m.temp(input.shape);
        auto *slope = p.scalar(c.attributes.relu_lower_clip_slope);
        p.add(aclnnLeakyReluBackward, [&](auto *w, auto **e) {
          return aclnnLeakyReluBackwardGetWorkspaceSize(
              grad.tensor, shifted.tensor, slope, false, result.tensor, w, e);
        });
        if (c.attributes.flags & FLAGDNN_POINTWISE_ATTRIBUTE_RELU_UPPER_CLIP) {
          auto activation = m.temp(input.shape);
          p.add(aclnnLeakyRelu, [&](auto *w, auto **e) {
            return aclnnLeakyReluGetWorkspaceSize(shifted.tensor, slope,
                                                  activation.tensor, w, e);
          });
          activation = m.shift(activation, c.attributes.relu_lower_clip);
          auto *mask =
              p.temporary(TestTensor{0, FLAGDNN_DATA_BOOLEAN, input.shape,
                                     dense_strides(input.shape)});
          auto *upper = p.scalar(c.attributes.relu_upper_clip);
          p.add(aclnnLtScalar, [&](auto *w, auto **e) {
            return aclnnLtScalarGetWorkspaceSize(activation.tensor, upper, mask,
                                                 w, e);
          });
          auto multiplier = m.temp(input.shape);
          p.add(aclnnCast, [&](auto *w, auto **e) {
            return aclnnCastGetWorkspaceSize(mask, ACL_FLOAT, multiplier.tensor,
                                             w, e);
          });
          result = m.mul(result, multiplier);
        }
        m.output(result, 2);
        break;
      }
      auto *slope = p.scalar(c.attributes.relu_lower_clip_slope);
      p.add(aclnnLeakyReluBackward, [&](auto *w, auto **e) {
        return aclnnLeakyReluBackwardGetWorkspaceSize(dy, x, slope, false, dx,
                                                      w, e);
      });
      break;
    }
    case FLAGDNN_POINTWISE_TANH_BWD: {
      auto *y = p.temporary(c.inputs[1]);
      p.add(aclnnTanh, [&](auto *w, auto **e) {
        return aclnnTanhGetWorkspaceSize(x, y, w, e);
      });
      p.add(aclnnTanhBackward, [&](auto *w, auto **e) {
        return aclnnTanhBackwardGetWorkspaceSize(dy, y, dx, w, e);
      });
      break;
    }
    case FLAGDNN_POINTWISE_ELU_BWD: {
      auto *alpha = p.scalar(c.attributes.elu_alpha);
      auto *one = p.scalar(1.0);
      p.add(aclnnEluBackward, [&](auto *w, auto **e) {
        return aclnnEluBackwardGetWorkspaceSize(dy, alpha, one, one, false, x,
                                                dx, w, e);
      });
      break;
    }
    case FLAGDNN_POINTWISE_GELU_BWD:
    case FLAGDNN_POINTWISE_GELU_APPROX_TANH_BWD: {
      char exact[] = "none", approx[] = "tanh";
      char *mode = c.mode == FLAGDNN_POINTWISE_GELU_BWD ? exact : approx;
      p.add(aclnnGeluBackwardV2, [&](auto *w, auto **e) {
        return aclnnGeluBackwardV2GetWorkspaceSize(dy, x, mode, dx, w, e);
      });
      break;
    }
    case FLAGDNN_POINTWISE_SOFTPLUS_BWD: {
      auto *beta = p.scalar(c.attributes.softplus_beta);
      auto *threshold = p.scalar(1.0e30);
      p.add(aclnnSoftplusBackward, [&](auto *w, auto **e) {
        return aclnnSoftplusBackwardGetWorkspaceSize(dy, x, beta, threshold, dx,
                                                     w, e);
      });
      break;
    }
    case FLAGDNN_POINTWISE_SWISH_BWD: {
      if (c.attributes.swish_beta != 1.0) {
        auto *scaled = p.temporary(c.inputs[1]);
        auto *beta = p.scalar(c.attributes.swish_beta);
        p.add(aclnnMuls, [&](auto *w, auto **e) {
          return aclnnMulsGetWorkspaceSize(x, beta, scaled, w, e);
        });
        x = scaled;
      }
      p.add(aclnnSiluBackward, [&](auto *w, auto **e) {
        return aclnnSiluBackwardGetWorkspaceSize(dy, x, dx, w, e);
      });
      break;
    }
    default:
      throw std::invalid_argument(
          "unrecognized Ascend activation backward mode");
    }
  });
}
} // namespace flagdnn::testing
