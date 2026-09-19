/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "validation/functional/aclnn_extended.hpp"
#include "validation/functional/aclnn_plan.hpp"
#include <aclnnop/aclnn_arange.h>
#include <aclnnop/aclnn_cat.h>
#include <aclnnop/aclnn_copy.h>
#include <aclnnop/aclnn_expand.h>
namespace flagdnn::testing {
std::unique_ptr<TestExecutable> build_aclnn_index(const IndexTestCase &c) {
  using namespace ascend;
  (void)dtype(c.output.data_type);
  auto specs = c.inputs;
  specs.push_back(c.output);
  return std::make_unique<Plan>(std::move(specs), [c](Plan &p) {
    if (c.operation == "concatenate") {
      if (c.inputs.size() == 1) {
        p.copy(p.ports.front(), p.ports.back());
        return;
      }
      auto inputs = p.ports;
      auto *output = inputs.back();
      inputs.pop_back();
      auto *list = p.list(inputs);
      p.add(aclnnCat, [&](auto *w, auto **e) {
        return aclnnCatGetWorkspaceSize(list, c.axis, output, w, e);
      });
    } else {
      const auto axis =
          (c.axis + static_cast<std::int64_t>(c.output.dimensions.size())) %
          static_cast<std::int64_t>(c.output.dimensions.size());
      auto spec = c.output;
      spec.dimensions.assign(spec.dimensions.size(), 1);
      spec.dimensions[axis] = c.output.dimensions[axis];
      spec.strides = dense_strides(spec.dimensions);
      auto *indices = p.temporary(spec);
      auto *start = p.scalar(0), *end = p.scalar(c.output.dimensions[axis]),
           *step = p.scalar(1);
      // Arange accepts a flat output; view the same temporary as rank one.
      p.add(aclnnArange, [&](auto *w, auto **e) {
        return aclnnArangeGetWorkspaceSize(start, end, step, indices, w, e);
      });
      auto *size = p.array(c.output.dimensions);
      p.add(aclnnExpand, [&](auto *w, auto **e) {
        return aclnnExpandGetWorkspaceSize(indices, size, p.ports.back(), w, e);
      });
    }
  });
}
} // namespace flagdnn::testing
