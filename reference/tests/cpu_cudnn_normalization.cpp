/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "cpu_cudnn_benchmark.hpp"
#include "common/normalization.hpp"
#include "reference/cpu/normalization.hpp"
#include "validation/functional/cudnn_extended.hpp"

#include <tuple>

namespace flagdnn::reference::tests {
namespace {
double tolerance(flagdnnDataType_t type) {
  return type == FLAGDNN_DATA_BFLOAT16 ? 8e-3 :
         type == FLAGDNN_DATA_FLOAT16 ? 1e-3 : 2e-4;
}
std::size_t spatial(const t::TestTensor& tensor) {
  std::size_t result = 1;
  for (std::size_t i = 2; i < tensor.dimensions.size(); ++i)
    result *= static_cast<std::size_t>(tensor.dimensions[i]);
  return result;
}
template<class Case> bool select(const Case& c, std::set<std::pair<int, bool>>& seen) {
  return io::element_count(c.x) <= 100000 &&
      seen.insert({int(c.x.data_type), c.x.dimensions.back() % 2 != 0}).second;
}
}
void normalization(Suite& suite) {
  normalization_helpers(suite);
  std::set<std::pair<int, bool>> seen;
  for (const auto& c : t::make_layernorm_cases()) {
    if (!select(c, seen)) continue;
    suite.run(c.name, [&] {
      const auto r = suite.check({c.x, c.scale, c.bias}, {c.y, c.mean, c.inv_variance},
          t::build_layernorm_reference(c), [&](const Arrays& x) {
            return cpu::evaluate_row_normalization(false, x[0], x[1], x[2], c.epsilon);
          }, c.absolute_tolerance, c.relative_tolerance);
      auto spec = formula_case(b::Operation::kLayernorm, {c.x, c.scale, c.bias}, {c.y, c.mean, c.inv_variance});
      spec.normalization.epsilon = c.epsilon;
      check_formula(spec, r, c.absolute_tolerance, c.relative_tolerance);
    });
  }
  seen.clear();
  for (const auto& c : t::make_rmsnorm_cases()) {
    if (!select(c, seen)) continue;
    suite.run(c.name, [&] {
      const auto r = suite.check({c.x, c.scale, c.bias}, {c.y, c.inv_variance},
          t::build_rmsnorm_reference(c), [&](const Arrays& x) {
            const auto out = cpu::evaluate_row_normalization(true, x[0], x[1], x[2], c.epsilon);
            return Arrays{out[0], out[2]};
          }, c.absolute_tolerance, c.relative_tolerance);
      auto spec = formula_case(b::Operation::kRmsnorm, {c.x, c.scale, c.bias}, {c.y, c.inv_variance});
      spec.normalization.epsilon = c.epsilon;
      check_formula(spec, r, c.absolute_tolerance, c.relative_tolerance);
    });
  }
  seen.clear();
  for (const auto& c : t::make_batchnorm_cases()) {
    if (!select(c, seen)) continue;
    suite.run(c.name, [&] {
      std::vector inputs{t::batchnorm_reference_data_tensor(c.x), c.scale,
          c.bias, c.previous_running_mean, c.previous_running_variance};
      auto values = inputs_for(inputs);
      for (auto& v : values[4]) v = 0.25F + std::abs(v);
      const auto r = suite.check(inputs, {t::batchnorm_reference_data_tensor(c.y), c.mean,
                  c.inv_variance, c.next_running_mean, c.next_running_variance},
          t::build_batchnorm_reference(c), [&](const Arrays& x) {
            return cpu::evaluate_batchnorm_training(x[0], spatial(c.x),
                x[1], x[2], x[3], x[4], c.epsilon, c.momentum);
          }, c.absolute_tolerance, c.relative_tolerance, values);
      auto spec = formula_case(b::Operation::kBatchnorm, inputs,
          {t::batchnorm_reference_data_tensor(c.y), c.mean, c.inv_variance,
           c.next_running_mean, c.next_running_variance});
      spec.normalization.epsilon = c.epsilon; spec.normalization.momentum = c.momentum;
      check_formula(spec, r, c.absolute_tolerance, c.relative_tolerance);
    });
  }
  seen.clear();
  for (const auto& c : t::make_batchnorm_inference_cases()) {
    if (!select(c, seen)) continue;
    suite.run(c.name, [&] {
      std::vector inputs{t::batchnorm_reference_data_tensor(c.x), c.mean,
                         c.inv_variance, c.scale, c.bias};
      auto values = inputs_for(inputs);
      for (auto& v : values[2]) v = 0.25F + std::abs(v);
      const auto r = suite.check(inputs, {t::batchnorm_reference_data_tensor(c.y)},
          t::build_batchnorm_inference_reference(c), [&](const Arrays& x) {
            return Arrays{cpu::evaluate_batchnorm_inference(x[0], spatial(c.x),
                x[1], x[2], x[3], x[4])};
          }, c.absolute_tolerance, c.relative_tolerance, values);
      check_formula(formula_case(b::Operation::kBatchnormInference, inputs,
          {t::batchnorm_reference_data_tensor(c.y)}), r,
          c.absolute_tolerance, c.relative_tolerance);
    });
  }
  for (const auto op : {"instancenorm", "adalayernorm", "layernorm_backward",
                        "rmsnorm_backward", "batchnorm_backward",
                        "instancenorm_backward", "adalayernorm_backward"}) {
    std::set<std::pair<int, bool>> covered;
    for (const auto& c : t::make_extended_normalization_cases(op)) {
      const bool backward = c.operation.ends_with("_backward");
      const auto& data = c.inputs[backward ? 1 : 0];
      if (io::element_count(data) > 100000 ||
          !covered.insert({int(data.data_type), data.dimensions.back() % 2 != 0}).second)
        continue;
      suite.run(c.name, [&] {
        cpu::NormalizationParameters p{c.operation, data.dimensions,
            c.inputs[backward ? 2 : 1].dimensions, c.axes, c.epsilon};
        auto values = inputs_for(c.inputs);
        if (backward)
          for (auto& v : values.back()) v = 0.5F + std::abs(v);
        const auto tol = tolerance(data.data_type);
        suite.check(c.inputs, c.outputs, t::build_cudnn_extended_normalization(c),
            [&](const Arrays& x) { return cpu::evaluate_normalization(p, x); },
            tol, tol, values);
        if (backward) {
          // Validate the saved-statistics helper against cuDNN forward outputs,
          // then use those independently computed statistics for backward.
          cpu::populate_normalization_statistics(p, values);
          t::ExtendedNormalizationTestCase forward;
          forward.name = c.name + "_saved_statistics";
          forward.operation = c.operation.substr(0, c.operation.size() - 9);
          forward.axes = c.axes;
          forward.epsilon = c.epsilon;
          auto bias = c.inputs[2]; bias.uid = 103;
          auto mean = c.inputs[3]; mean.uid = 104;
          auto inv = c.inputs.back(); inv.uid = 105;
          auto y = data; y.uid = 106;
          auto affine = c.inputs[2];
          if (c.operation == "rmsnorm_backward" || c.operation == "layernorm_backward" ||
              c.operation == "batchnorm_backward")
            affine.data_type = bias.data_type = data.data_type;
          if (c.operation == "rmsnorm_backward") {
            t::RmsnormTestCase f{forward.name, data, affine, bias, y, inv};
            f.epsilon = c.epsilon;
            const auto r = suite.execute({data, affine, bias}, {y, inv},
                t::build_rmsnorm_reference(f),
                {values[1], values[2], Values(values[2].size(), 0)});
            compare(r.outputs[1], values[3], 2e-4, 2e-4);
          } else if (c.operation == "layernorm_backward") {
            t::LayernormTestCase f{forward.name, data, affine, bias, y, mean, inv};
            f.epsilon = c.epsilon;
            const auto r = suite.execute({data, affine, bias}, {y, mean, inv},
                t::build_layernorm_reference(f),
                {values[1], values[2], Values(values[2].size(), 0)});
            compare(r.outputs[1], values[3], 2e-4, 2e-4);
            compare(r.outputs[2], values[4], 2e-4, 2e-4);
          } else if (c.operation == "batchnorm_backward") {
            t::BatchnormTestCase f;
            f.name = forward.name; f.x = data; f.scale = affine; f.bias = bias;
            f.y = y; f.mean = mean; f.inv_variance = inv; f.epsilon = c.epsilon;
            f.previous_running_mean = mean; f.previous_running_mean.uid = 107;
            f.previous_running_variance = inv; f.previous_running_variance.uid = 108;
            f.next_running_mean = mean; f.next_running_mean.uid = 109;
            f.next_running_variance = inv; f.next_running_variance.uid = 110;
            const auto r = suite.execute(
                {t::batchnorm_reference_data_tensor(data), f.scale, bias,
                 f.previous_running_mean, f.previous_running_variance},
                {t::batchnorm_reference_data_tensor(y), mean, inv,
                 f.next_running_mean, f.next_running_variance},
                t::build_batchnorm_reference(f),
                {values[1], values[2], Values(values[2].size(), 0),
                 Values(values[2].size(), 0), Values(values[2].size(), 1)});
            compare(r.outputs[1], values[3], 2e-4, 2e-4);
            compare(r.outputs[2], values[4], 2e-4, 2e-4);
          } else {
            forward.inputs = {data, c.inputs[2], bias};
            forward.outputs = {y, mean, inv};
            const auto r = suite.execute(forward.inputs, forward.outputs,
                t::build_cudnn_extended_normalization(forward),
                {values[1], values[2], Values(values[2].size(), 0)});
            compare(r.outputs[1], values[3], 2e-4, 2e-4);
            compare(r.outputs[2], values[4], 2e-4, 2e-4);
          }
          suite.check(c.inputs, c.outputs, t::build_cudnn_extended_normalization(c),
              [&](const Arrays& x) { return cpu::evaluate_normalization(p, x); },
              tol, tol, values);
        }
      });
    }
  }
}
}  // namespace flagdnn::reference::tests
