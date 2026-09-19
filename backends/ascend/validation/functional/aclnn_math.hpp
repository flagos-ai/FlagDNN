/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include "validation/functional/aclnn_plan.hpp"
#include <aclnnop/aclnn_add.h>
#include <aclnnop/aclnn_cast.h>
#include <aclnnop/aclnn_exp.h>
#include <aclnnop/aclnn_expand.h>
#include <aclnnop/aclnn_logsumexp.h>
#include <aclnnop/aclnn_matmul.h>
#include <aclnnop/aclnn_mean.h>
#include <aclnnop/aclnn_mul.h>
#include <aclnnop/aclnn_nan_to_num.h>
#include <aclnnop/aclnn_permute.h>
#include <aclnnop/aclnn_reduce_sum.h>
#include <aclnnop/aclnn_rsqrt.h>
#include <aclnnop/aclnn_sub.h>
namespace flagdnn::testing::ascend {
struct Math {
  struct Value {
    aclTensor *tensor;
    std::vector<std::int64_t> shape;
  };
  Plan &p;
  Value temp(std::vector<std::int64_t> shape) {
    TestTensor s{0, FLAGDNN_DATA_FLOAT32, shape, dense_strides(shape)};
    return {p.temporary(s), std::move(shape)};
  }
  Value input(std::size_t i) {
    if (p.specs[i].data_type == FLAGDNN_DATA_FLOAT32)
      return {p.ports[i], p.specs[i].dimensions};
    auto out = temp(p.specs[i].dimensions);
    p.add(aclnnCast, [&](auto *w, auto **e) {
      return aclnnCastGetWorkspaceSize(p.ports[i], ACL_FLOAT, out.tensor, w, e);
    });
    return out;
  }
  void output(Value a, std::size_t i) {
    if (p.specs[i].data_type == FLAGDNN_DATA_FLOAT32) {
      p.copy(a.tensor, p.ports[i]);
      return;
    }
    p.add(aclnnCast, [&](auto *w, auto **e) {
      return aclnnCastGetWorkspaceSize(a.tensor, dtype(p.specs[i].data_type),
                                       p.ports[i], w, e);
    });
  }
  static std::vector<std::int64_t> broadcast(Value a, Value b) {
    auto shape = a.shape;
    shape.insert(shape.begin(),
                 std::max(a.shape.size(), b.shape.size()) - shape.size(), 1);
    for (std::size_t i = 0; i < b.shape.size(); ++i)
      shape[shape.size() - 1 - i] = std::max(shape[shape.size() - 1 - i],
                                             b.shape[b.shape.size() - 1 - i]);
    return shape;
  }
  Value mul(Value a, Value b) {
    auto out = temp(broadcast(a, b));
    p.add(aclnnMul, [&](auto *w, auto **e) {
      return aclnnMulGetWorkspaceSize(a.tensor, b.tensor, out.tensor, w, e);
    });
    return out;
  }
  Value add(Value a, Value b, double alpha = 1) {
    auto out = temp(broadcast(a, b));
    auto *scale = p.scalar(alpha);
    p.add(aclnnAdd, [&](auto *w, auto **e) {
      return aclnnAddGetWorkspaceSize(a.tensor, b.tensor, scale, out.tensor, w,
                                      e);
    });
    return out;
  }
  Value scale(Value a, double value) {
    if (value == 1)
      return a;
    auto out = temp(a.shape);
    auto *s = p.scalar(value);
    p.add(aclnnMuls, [&](auto *w, auto **e) {
      return aclnnMulsGetWorkspaceSize(a.tensor, s, out.tensor, w, e);
    });
    return out;
  }
  Value shift(Value a, double value) {
    if (value == 0)
      return a;
    auto out = temp(a.shape);
    auto *s = p.scalar(value);
    auto *one = p.scalar(1);
    p.add(aclnnAdds, [&](auto *w, auto **e) {
      return aclnnAddsGetWorkspaceSize(a.tensor, s, one, out.tensor, w, e);
    });
    return out;
  }
  Value rsqrt(Value a) {
    auto out = temp(a.shape);
    p.add(aclnnRsqrt, [&](auto *w, auto **e) {
      return aclnnRsqrtGetWorkspaceSize(a.tensor, out.tensor, w, e);
    });
    return out;
  }
  Value reshape(Value a, std::vector<std::int64_t> shape) {
    auto dense = temp(a.shape);
    p.copy(a.tensor, dense.tensor);
    return {p.alias(dense.tensor, shape, dense_strides(shape)),
            std::move(shape)};
  }
  Value transpose(Value a) {
    auto shape = a.shape;
    std::swap(shape[shape.size() - 1], shape[shape.size() - 2]);
    std::vector<std::int64_t> axes;
    for (std::size_t i = 0; i < shape.size(); ++i)
      axes.push_back(i);
    std::swap(axes[axes.size() - 1], axes[axes.size() - 2]);
    return {p.permuted(a.tensor, axes), std::move(shape)};
  }
  Value expand(Value a, std::vector<std::int64_t> shape) {
    if (a.shape == shape)
      return a;
    auto out = temp(shape);
    auto *dims = p.array(shape);
    p.add(aclnnExpand, [&](auto *w, auto **e) {
      return aclnnExpandGetWorkspaceSize(a.tensor, dims, out.tensor, w, e);
    });
    return out;
  }
  Value mm(Value a, Value b) {
    auto shape = a.shape;
    shape.back() = b.shape.back();
    auto out = temp(shape);
    p.add(aclnnMatmul, [&](auto *w, auto **e) {
      return aclnnMatmulGetWorkspaceSize(a.tensor, b.tensor, out.tensor, 0, w,
                                         e);
    });
    return out;
  }
  Value exp(Value a) {
    auto out = temp(a.shape);
    p.add(aclnnExp, [&](auto *w, auto **e) {
      return aclnnExpGetWorkspaceSize(a.tensor, out.tensor, w, e);
    });
    auto clean = temp(a.shape);
    p.add(aclnnNanToNum, [&](auto *w, auto **e) {
      return aclnnNanToNumGetWorkspaceSize(out.tensor, 0, 0, 0, clean.tensor, w,
                                           e);
    });
    return clean;
  }
  Value lse(Value a) {
    auto shape = a.shape;
    shape.back() = 1;
    auto out = temp(shape);
    auto *dims = p.array({static_cast<std::int64_t>(a.shape.size() - 1)});
    p.add(aclnnLogSumExp, [&](auto *w, auto **e) {
      return aclnnLogSumExpGetWorkspaceSize(a.tensor, dims, true, out.tensor, w,
                                            e);
    });
    return out;
  }
  Value reduce(Value a, const std::vector<std::int64_t> &axes, bool mean) {
    if (axes.empty())
      return a;
    auto shape = a.shape;
    for (auto i : axes)
      shape[i] = 1;
    if (shape == a.shape)
      return a;
    auto out = temp(shape);
    auto *dims = p.array(axes);
    if (mean)
      p.add(aclnnMean, [&](auto *w, auto **e) {
        return aclnnMeanGetWorkspaceSize(a.tensor, dims, true, ACL_FLOAT,
                                         out.tensor, w, e);
      });
    else {
      p.add(aclnnReduceSum, [&](auto *w, auto **e) {
        return aclnnReduceSumGetWorkspaceSize(a.tensor, dims, true, ACL_FLOAT,
                                              out.tensor, w, e);
      });
    }
    return out;
  }
};
} // namespace flagdnn::testing::ascend
