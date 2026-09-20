/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "validation/functional/dtype_runner.hpp"
#include "reference/cpu/pointwise.hpp"
#include "validation/functional/paired.hpp"
#include <aclnnop/aclnn_abs.h>
#include <aclnnop/aclnn_add.h>
#include <aclnnop/aclnn_cast.h>
#include <aclnnop/aclnn_div.h>
#include <aclnnop/aclnn_eq_scalar.h>
#include <aclnnop/aclnn_eq_tensor.h>
#include <aclnnop/aclnn_flatten.h>
#include <aclnnop/aclnn_fmod_tensor.h>
#include <aclnnop/aclnn_ge_tensor.h>
#include <aclnnop/aclnn_gt_scalar.h>
#include <aclnnop/aclnn_gt_tensor.h>
#include <aclnnop/aclnn_le_tensor.h>
#include <aclnnop/aclnn_logical_and.h>
#include <aclnnop/aclnn_logical_not.h>
#include <aclnnop/aclnn_logical_or.h>
#include <aclnnop/aclnn_lt_scalar.h>
#include <aclnnop/aclnn_lt_tensor.h>
#include <aclnnop/aclnn_masked_fill_scalar.h>
#include <aclnnop/aclnn_maximum.h>
#include <aclnnop/aclnn_mean.h>
#include <aclnnop/aclnn_minimum.h>
#include <aclnnop/aclnn_mul.h>
#include <aclnnop/aclnn_ne_tensor.h>
#include <aclnnop/aclnn_neg.h>
#include <aclnnop/aclnn_permute.h>
#include <aclnnop/aclnn_pow_tensor_tensor.h>
#include <aclnnop/aclnn_prod.h>
#include <aclnnop/aclnn_reduce_sum.h>
#include <aclnnop/aclnn_relu.h>
#include <aclnnop/aclnn_s_where.h>
#include <aclnnop/aclnn_slice_v2.h>
#include <aclnnop/aclnn_sub.h>

namespace flagdnn::testing {
namespace {
using namespace ascend;
bool floating(flagdnnDataType_t t) {
  return t == FLAGDNN_DATA_FLOAT32 || t == FLAGDNN_DATA_FLOAT16 ||
         t == FLAGDNN_DATA_BFLOAT16;
}
bool pointwise_supplement(const PointwiseTestCase &c) {
  return std::any_of(c.inputs.begin(), c.inputs.end(), [&](const auto &t) {
    return !floating(t.data_type) && (t.data_type != FLAGDNN_DATA_BOOLEAN ||
                                      c.mode == FLAGDNN_POINTWISE_IDENTITY);
  });
}
std::optional<std::vector<std::vector<std::uint8_t>>>
cpu_integer_divmod_reference(
    const PointwiseTestCase &c,
    const std::vector<std::vector<std::uint8_t>> &inputs) {
  if ((c.mode != FLAGDNN_POINTWISE_DIV && c.mode != FLAGDNN_POINTWISE_MOD) ||
      c.output.data_type != FLAGDNN_DATA_INT32 ||
      std::any_of(c.inputs.begin(), c.inputs.end(), [](const auto &t) {
        return t.data_type != FLAGDNN_DATA_INT32;
      }))
    return std::nullopt;
  validate_pointwise_case(c);
  for (std::size_t k = 0; k < c.inputs.size(); ++k)
    if (inputs[k].size() != io::encoded_byte_count(c.inputs[k]))
      throw std::runtime_error("incorrect CPU reference input size");
  const auto value_at = [&](std::size_t k, std::size_t index) {
    const auto &t = c.inputs[k];
    const auto leading = c.output.dimensions.size() - t.dimensions.size();
    std::size_t offset = 0;
    for (std::size_t axis = c.output.dimensions.size(); axis != 0; --axis) {
      const auto coordinate = index % c.output.dimensions[axis - 1];
      index /= c.output.dimensions[axis - 1];
      if (axis - 1 >= leading && t.dimensions[axis - 1 - leading] != 1)
        offset += coordinate * t.strides[axis - 1 - leading];
    }
    std::int32_t value;
    std::memcpy(&value, inputs[k].data() + offset * sizeof(value),
                sizeof(value));
    return value;
  };
  const auto count = io::element_count(c.output);
  std::vector<std::uint8_t> output(count * sizeof(std::int32_t));
  for (std::size_t i = 0; i < count; ++i) {
    // Keep the full INT32 bit pattern, including values beyond FP32 precision.
    const auto value = reference::cpu::pointwise_integer_reference(
        c.mode, value_at(0, i), value_at(1, i), false, 1);
    std::memcpy(output.data() + i * sizeof(value), &value, sizeof(value));
  }
  return std::vector<std::vector<std::uint8_t>>{std::move(output)};
}
std::unique_ptr<TestExecutable>
reference_pointwise(const PointwiseTestCase &c) {
  if (c.inputs[0].data_type == FLAGDNN_DATA_INT32 &&
      (c.mode == FLAGDNN_POINTWISE_DIV || c.mode == FLAGDNN_POINTWISE_MOD))
    throw Unsupported("CANN 9.0 ACLNN integer division/remainder do not "
                      "preserve the shared full-range INT32 contract "
                      "(including values above 2^24 and zero divisors)");
  for (const auto &t : c.inputs) {
    (void)dtype(t.data_type);
  }
  auto specs = c.inputs;
  specs.push_back(c.output);
  return std::make_unique<Plan>(std::move(specs), [c](Plan &p) {
    auto *a = p.ports[0];
    auto *b = c.inputs.size() > 1 ? p.ports[1] : nullptr;
    auto *out = p.ports.back();
    auto *alpha = c.inputs[0].data_type == FLAGDNN_DATA_INT32
                      ? p.integer_scalar(static_cast<std::int64_t>(c.alpha))
                      : p.scalar(c.alpha);
    if (c.inputs[0].data_type == FLAGDNN_DATA_INT32 && c.alpha != 1.0 &&
        (c.mode == FLAGDNN_POINTWISE_ADD || c.mode == FLAGDNN_POINTWISE_SUB)) {
      // ACLNN's alpha path converts through floating arithmetic. Use its
      // integer tensor multiplication to preserve modulo-2^32 overflow.
      auto *factor = p.temporary(TestTensor{0, FLAGDNN_DATA_INT32, {1}, {1}});
      const auto value = static_cast<std::int32_t>(c.alpha);
      std::vector<std::uint8_t> bytes(sizeof(value));
      std::memcpy(bytes.data(), &value, sizeof(value));
      p.initialize(factor, bytes);
      auto *scaled = p.temporary(c.inputs[1]);
      p.add(aclnnMul, [&](auto *w, auto **e) {
        return aclnnMulGetWorkspaceSize(b, factor, scaled, w, e);
      });
      b = scaled;
      alpha = p.integer_scalar(1);
    }
    switch (c.mode) {
    case FLAGDNN_POINTWISE_IDENTITY:
      p.copy(a, out);
      break;
    case FLAGDNN_POINTWISE_ADD:
      p.add(aclnnAdd, [&](auto *w, auto **e) {
        return aclnnAddGetWorkspaceSize(a, b, alpha, out, w, e);
      });
      break;
    case FLAGDNN_POINTWISE_SUB:
      p.add(aclnnSub, [&](auto *w, auto **e) {
        return aclnnSubGetWorkspaceSize(a, b, alpha, out, w, e);
      });
      break;
    case FLAGDNN_POINTWISE_MUL:
      p.add(aclnnMul, [&](auto *w, auto **e) {
        return aclnnMulGetWorkspaceSize(a, b, out, w, e);
      });
      break;
    case FLAGDNN_POINTWISE_DIV:
      p.add(aclnnDivMod, [&](auto *w, auto **e) {
        return aclnnDivModGetWorkspaceSize(a, b, 1, out, w, e);
      });
      break;
    case FLAGDNN_POINTWISE_MIN:
      p.add(aclnnMinimum, [&](auto *w, auto **e) {
        return aclnnMinimumGetWorkspaceSize(a, b, out, w, e);
      });
      break;
    case FLAGDNN_POINTWISE_MAX:
      p.add(aclnnMaximum, [&](auto *w, auto **e) {
        return aclnnMaximumGetWorkspaceSize(a, b, out, w, e);
      });
      break;
    case FLAGDNN_POINTWISE_MOD:
      p.add(aclnnFmodTensor, [&](auto *w, auto **e) {
        return aclnnFmodTensorGetWorkspaceSize(a, b, out, w, e);
      });
      break;
    case FLAGDNN_POINTWISE_POW:
      p.add(aclnnPowTensorTensor, [&](auto *w, auto **e) {
        return aclnnPowTensorTensorGetWorkspaceSize(a, b, out, w, e);
      });
      break;
    case FLAGDNN_POINTWISE_CMP_EQ:
      p.add(aclnnEqTensor, [&](auto *w, auto **e) {
        return aclnnEqTensorGetWorkspaceSize(a, b, out, w, e);
      });
      break;
    case FLAGDNN_POINTWISE_CMP_NEQ:
      p.add(aclnnNeTensor, [&](auto *w, auto **e) {
        return aclnnNeTensorGetWorkspaceSize(a, b, out, w, e);
      });
      break;
    case FLAGDNN_POINTWISE_CMP_GT:
      p.add(aclnnGtTensor, [&](auto *w, auto **e) {
        return aclnnGtTensorGetWorkspaceSize(a, b, out, w, e);
      });
      break;
    case FLAGDNN_POINTWISE_CMP_GE:
      p.add(aclnnGeTensor, [&](auto *w, auto **e) {
        return aclnnGeTensorGetWorkspaceSize(a, b, out, w, e);
      });
      break;
    case FLAGDNN_POINTWISE_CMP_LT:
      p.add(aclnnLtTensor, [&](auto *w, auto **e) {
        return aclnnLtTensorGetWorkspaceSize(a, b, out, w, e);
      });
      break;
    case FLAGDNN_POINTWISE_CMP_LE:
      p.add(aclnnLeTensor, [&](auto *w, auto **e) {
        return aclnnLeTensorGetWorkspaceSize(a, b, out, w, e);
      });
      break;
    case FLAGDNN_POINTWISE_LOGICAL_AND:
      p.add(aclnnLogicalAnd, [&](auto *w, auto **e) {
        return aclnnLogicalAndGetWorkspaceSize(a, b, out, w, e);
      });
      break;
    case FLAGDNN_POINTWISE_LOGICAL_OR:
      p.add(aclnnLogicalOr, [&](auto *w, auto **e) {
        return aclnnLogicalOrGetWorkspaceSize(a, b, out, w, e);
      });
      break;
    case FLAGDNN_POINTWISE_LOGICAL_NOT:
      p.add(aclnnLogicalNot, [&](auto *w, auto **e) {
        return aclnnLogicalNotGetWorkspaceSize(a, out, w, e);
      });
      break;
    case FLAGDNN_POINTWISE_NEG:
      p.add(aclnnNeg, [&](auto *w, auto **e) {
        return aclnnNegGetWorkspaceSize(a, out, w, e);
      });
      break;
    case FLAGDNN_POINTWISE_ABS:
      p.add(aclnnAbs, [&](auto *w, auto **e) {
        return aclnnAbsGetWorkspaceSize(a, out, w, e);
      });
      break;
    case FLAGDNN_POINTWISE_RELU_FWD:
      p.add(aclnnRelu, [&](auto *w, auto **e) {
        return aclnnReluGetWorkspaceSize(a, out, w, e);
      });
      break;
    case FLAGDNN_POINTWISE_BINARY_SELECT:
      p.add(aclnnSWhere, [&](auto *w, auto **e) {
        return aclnnSWhereGetWorkspaceSize(p.ports[2], a, b, out, w, e);
      });
      break;
    default:
      throw std::invalid_argument("unhandled Ascend dtype pointwise mode");
    }
    if (c.inputs[0].data_type == FLAGDNN_DATA_INT32 &&
        c.mode == FLAGDNN_POINTWISE_POW) {
      auto bool_temp = [&](const TestTensor &t) {
        auto s = t;
        s.data_type = FLAGDNN_DATA_BOOLEAN;
        return p.temporary(s);
      };
      auto *negative = bool_temp(c.inputs[1]);
      auto *greater = bool_temp(c.inputs[0]);
      auto *less = bool_temp(c.inputs[0]);
      auto *zero_base = bool_temp(c.inputs[0]);
      auto *either = bool_temp(c.inputs[0]);
      auto *regular = bool_temp(c.inputs[0]);
      auto *mask = bool_temp(c.output);
      auto *zero = p.integer_scalar(0);
      auto *one = p.integer_scalar(1);
      auto *minus_one = p.integer_scalar(-1);
      p.add(aclnnLtScalar, [&](auto *w, auto **e) {
        return aclnnLtScalarGetWorkspaceSize(b, zero, negative, w, e);
      });
      p.add(aclnnGtScalar, [&](auto *w, auto **e) {
        return aclnnGtScalarGetWorkspaceSize(a, one, greater, w, e);
      });
      p.add(aclnnLtScalar, [&](auto *w, auto **e) {
        return aclnnLtScalarGetWorkspaceSize(a, minus_one, less, w, e);
      });
      p.add(aclnnEqScalar, [&](auto *w, auto **e) {
        return aclnnEqScalarGetWorkspaceSize(a, zero, zero_base, w, e);
      });
      p.add(aclnnLogicalOr, [&](auto *w, auto **e) {
        return aclnnLogicalOrGetWorkspaceSize(greater, less, either, w, e);
      });
      p.add(aclnnLogicalOr, [&](auto *w, auto **e) {
        return aclnnLogicalOrGetWorkspaceSize(either, zero_base, regular, w, e);
      });
      p.add(aclnnLogicalAnd, [&](auto *w, auto **e) {
        return aclnnLogicalAndGetWorkspaceSize(regular, negative, mask, w, e);
      });
      p.add(aclnnInplaceMaskedFillScalar, [&](auto *w, auto **e) {
        return aclnnInplaceMaskedFillScalarGetWorkspaceSize(out, mask, zero, w,
                                                            e);
      });
    }
  });
}
struct PointwiseCase : PointwiseTestCase {
  std::vector<TestTensor> outputs;
};
struct LayoutCase : LayoutTestCase {
  std::vector<TestTensor> inputs, outputs;
};
struct ReductionCase : ReductionTestCase {
  std::vector<TestTensor> inputs, outputs;
};
} // namespace
int run_ascend_pointwise_dtype_cases(int argc, char **argv,
                                     std::span<const PointwiseTestCase> cases,
                                     bool benchmark, bool all) {
  std::vector<PointwiseCase> selected;
  for (const auto &c : cases)
    if (all || pointwise_supplement(c))
      selected.push_back({c, {c.output}});
  if (selected.empty())
    return 0;
  return ascend::run_paired_cases<PointwiseCase>(
      argc, argv, selected, "FLAGDNN_POINTWISE_CASE", build_flagdnn_pointwise,
      [](const PointwiseCase &c) {
        std::vector<std::vector<std::uint8_t>> inputs;
        for (std::size_t k = 0; k < c.inputs.size(); ++k) {
          const auto &t = c.inputs[k];
          if (t.data_type == FLAGDNN_DATA_INT32)
            inputs.push_back(ascend::integer_bytes(t, [&](std::size_t i) {
              return pointwise_integer_input(i, k, c.mode);
            }));
          else {
            std::vector<float> v(ascend::io::element_count(t));
            for (std::size_t i = 0; i < v.size(); ++i)
              v[i] = t.data_type == FLAGDNN_DATA_BOOLEAN ? float((i + k) % 2)
                                                         : float(1U << (i % 6));
            inputs.push_back(
                ascend::io::encode(ascend::io::scatter(v, t), t.data_type));
          }
        }
        return inputs;
      },
      [](const PointwiseCase &c) {
        auto r = static_cast<const PointwiseTestCase &>(c);
        r.output = c.outputs[0];
        return reference_pointwise(r);
      },
      [](const PointwiseCase &, std::size_t) {
        return ascend::PairedTolerance{0, 0};
      },
      benchmark, cpu_integer_divmod_reference);
}
int run_ascend_add_dtype_cases(int argc, char **argv,
                               std::span<const AddTestCase> cases) {
  std::vector<PointwiseTestCase> converted;
  for (const auto &c : cases)
    if (!floating(c.left.data_type)) {
      PointwiseTestCase p;
      p.name = c.name;
      p.mode = FLAGDNN_POINTWISE_ADD;
      p.inputs = {c.left, c.right};
      p.input_domains = {PointwiseInputDomain::kReal,
                         PointwiseInputDomain::kReal};
      p.output = c.output;
      p.alpha = c.alpha;
      converted.push_back(std::move(p));
    }
  return run_ascend_pointwise_dtype_cases(argc, argv, converted);
}
int run_ascend_layout_dtype_cases(int argc, char **argv,
                                  std::span<const LayoutTestCase> cases,
                                  bool benchmark, bool all) {
  std::vector<LayoutCase> selected;
  for (const auto &c : cases)
    if (all || !floating(c.input.data_type))
      selected.push_back({c, {c.input}, {c.output}});
  if (selected.empty())
    return 0;
  return ascend::run_paired_cases<LayoutCase>(
      argc, argv, selected, "FLAGDNN_LAYOUT_CASE", build_flagdnn_layout,
      [](const LayoutCase &c) {
        const auto &t = c.input;
        if (t.data_type == FLAGDNN_DATA_INT32)
          return std::vector<std::vector<std::uint8_t>>{
              ascend::integer_bytes(t, [](std::size_t i) {
                std::uint32_t bits = 0;
                for (std::size_t b = 0; b < 4; ++b)
                  bits |= std::uint32_t((i * 37 + b * 83) % 256) << (8 * b);
                return static_cast<std::int32_t>(bits);
              })};
        std::vector<float> v(ascend::io::element_count(t));
        for (std::size_t i = 0; i < v.size(); ++i)
          v[i] = t.data_type == FLAGDNN_DATA_BOOLEAN ? float(i % 2)
                                                     : float(1U << (i % 6));
        return std::vector<std::vector<std::uint8_t>>{
            ascend::io::encode(ascend::io::scatter(v, t), t.data_type)};
      },
      [](const LayoutCase &c) -> std::unique_ptr<TestExecutable> {
        (void)ascend::dtype(c.input.data_type);
        auto specs = c.inputs;
        specs.insert(specs.end(), c.outputs.begin(), c.outputs.end());
        return std::make_unique<ascend::Plan>(specs, [c](ascend::Plan &p) {
          auto *x = p.ports[0];
          auto *y = p.ports[1];
          if (c.operation == LayoutOperation::kTranspose) {
            auto *permutation = p.array(c.permutation);
            p.add(aclnnPermute, [&](auto *w, auto **e) {
              return aclnnPermuteGetWorkspaceSize(x, permutation, y, w, e);
            });
          } else if (c.operation == LayoutOperation::kSlice) {
            std::vector<std::int64_t> starts, ends, axes;
            for (std::size_t i = 0; i < c.slices.size(); ++i) {
              starts.push_back(c.slices[i].first);
              ends.push_back(c.slices[i].second);
              axes.push_back(i);
            }
            auto *start = p.array(starts);
            auto *end = p.array(ends);
            auto *axis = p.array(axes);
            auto *steps = p.array(c.slice_strides);
            p.add(aclnnSliceV2, [&](auto *w, auto **e) {
              return aclnnSliceV2GetWorkspaceSize(x, start, end, axis, steps, y,
                                                  w, e);
            });
          } else {
            const auto count =
                static_cast<std::int64_t>(ascend::io::element_count(c.input));
            auto spec = c.input;
            spec.dimensions = {1, count};
            spec.strides = {count, 1};
            auto *flat = p.temporary(spec);
            if (c.input.strides == ascend::dense_strides(c.input.dimensions))
              flat = p.alias(x, {1, count}, {count, 1});
            else
              p.add(aclnnFlatten, [&](auto *w, auto **e) {
                return aclnnFlattenGetWorkspaceSize(x, 0, flat, w, e);
              });
            auto *shaped =
                p.alias(flat, c.outputs[0].dimensions,
                        ascend::dense_strides(c.outputs[0].dimensions));
            p.copy(shaped, y);
          }
        });
      },
      [](const LayoutCase &, std::size_t) {
        return ascend::PairedTolerance{0, 0};
      },
      benchmark);
}
int run_ascend_reduction_dtype_cases(int argc, char **argv,
                                     std::span<const ReductionTestCase> cases,
                                     bool benchmark) {
  std::vector<ReductionCase> selected;
  for (const auto &c : cases)
    if (!floating(c.input.data_type) || c.output.data_type != c.input.data_type)
      selected.push_back({c, {c.input}, {c.output}});
  if (selected.empty())
    return 0;
  return ascend::run_paired_cases<ReductionCase>(
      argc, argv, selected, "FLAGDNN_REDUCTION_CASE", build_flagdnn_reduction,
      [](const ReductionCase &c) {
        return std::vector<std::vector<float>>{reduction_host_input(c)};
      },
      [](const ReductionCase &c) -> std::unique_ptr<TestExecutable> {
        auto specs = c.inputs;
        specs.insert(specs.end(), c.outputs.begin(), c.outputs.end());
        return std::make_unique<ascend::Plan>(specs, [c](ascend::Plan &p) {
          auto *x = p.ports[0];
          auto *y = p.ports[1];
          auto *axis = p.array({c.axis});
          if (c.input.data_type != FLAGDNN_DATA_FLOAT32) {
            auto s = c.input;
            s.data_type = FLAGDNN_DATA_FLOAT32;
            auto *converted = p.temporary(s);
            p.add(aclnnCast, [&](auto *w, auto **e) {
              return aclnnCastGetWorkspaceSize(x, ACL_FLOAT, converted, w, e);
            });
            x = converted;
          }
          if (c.mode == FLAGDNN_REDUCTION_ADD)
            p.add(aclnnReduceSum, [&](auto *w, auto **e) {
              return aclnnReduceSumGetWorkspaceSize(x, axis, c.keep_dimensions,
                                                    ACL_FLOAT, y, w, e);
            });
          else if (c.mode == FLAGDNN_REDUCTION_AVG)
            p.add(aclnnMean, [&](auto *w, auto **e) {
              return aclnnMeanGetWorkspaceSize(x, axis, c.keep_dimensions,
                                               ACL_FLOAT, y, w, e);
            });
          else
            p.add(aclnnProdDim, [&](auto *w, auto **e) {
              return aclnnProdDimGetWorkspaceSize(x, c.axis, c.keep_dimensions,
                                                  ACL_FLOAT, y, w, e);
            });
        });
      },
      [](const ReductionCase &c, std::size_t) {
        return ascend::PairedTolerance{c.absolute_tolerance,
                                       c.relative_tolerance};
      },
      benchmark);
}
} // namespace flagdnn::testing
