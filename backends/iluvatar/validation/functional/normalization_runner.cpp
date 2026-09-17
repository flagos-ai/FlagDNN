// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "common/normalization.hpp"
#include "functional/runner_support.hpp"
#include "reference/cpu/normalization.hpp"
#include <algorithm>
#include <cmath>

namespace flagdnn::testing {
namespace {

namespace functional = iluvatar::validation::functional;

functional::PlannedOutput output(const TestTensor &tensor,
                                 double absolute_tolerance,
                                 double relative_tolerance, const char *label) {
  return {tensor, absolute_tolerance, relative_tolerance, label};
}

} // namespace

int run_layernorm_functional_test(int argc, char **argv,
                                  std::span<const LayernormTestCase> cases) {
  functional::FunctionalSuite suite(argc, argv, "layernorm",
                                    "FLAGDNN_LAYERNORM_FUNCTIONAL");
  for (const LayernormTestCase &test_case : cases) {
    validate_normalization_case(test_case);
    suite.run(
        {.operation = "layernorm",
         .case_name = test_case.name,
         .inputs = {{test_case.x, functional::InputDomain::kReal, {}},
                    {test_case.scale, functional::InputDomain::kUnitScale, {}},
                    {test_case.bias, functional::InputDomain::kReal, {}}},
         .outputs = {output(test_case.y, test_case.absolute_tolerance,
                            test_case.relative_tolerance, "y"),
                     output(test_case.mean, test_case.absolute_tolerance,
                            test_case.relative_tolerance, "mean"),
                     output(test_case.inv_variance,
                            test_case.absolute_tolerance,
                            test_case.relative_tolerance, "inv_variance")}},
        [&suite, &test_case] {
          return build_flagdnn_layernorm(suite.handle(), test_case);
        },
        [&test_case] { return build_layernorm_reference(test_case); });
  }
  return suite.finish();
}

int run_rmsnorm_functional_test(int argc, char **argv,
                                std::span<const RmsnormTestCase> cases) {
  functional::FunctionalSuite suite(argc, argv, "rmsnorm",
                                    "FLAGDNN_RMSNORM_FUNCTIONAL");
  for (const RmsnormTestCase &test_case : cases) {
    validate_normalization_case(test_case);
    functional::HostReference host_reference;
    if (test_case.x.data_type != FLAGDNN_DATA_FLOAT16) {
      host_reference = [&test_case](const auto &inputs) {
        const auto width = inputs[1].size(), rows = inputs[0].size() / width;
        std::vector<float> y(inputs[0].size()), inverse(rows);
        for (std::size_t row = 0; row < rows; ++row) {
          double square_sum = 0;
          for (std::size_t col = 0; col < width; ++col) {
            const double value = inputs[0][row * width + col];
            square_sum += value * value;
          }
          const double inv =
              1 / std::sqrt(square_sum / width + test_case.epsilon);
          inverse[row] = static_cast<float>(inv);
          for (std::size_t col = 0; col < width; ++col)
            y[row * width + col] = static_cast<float>(
                inputs[0][row * width + col] * inv * inputs[1][col] +
                inputs[2][col]);
        }
        return std::vector<std::vector<float>>{std::move(y),
                                               std::move(inverse)};
      };
    }
    suite.run(
        {.operation = "rmsnorm",
         .case_name = test_case.name,
         .inputs = {{test_case.x, functional::InputDomain::kReal, {}},
                    {test_case.scale, functional::InputDomain::kUnitScale, {}},
                    {test_case.bias, functional::InputDomain::kReal, {}}},
         .outputs = {output(test_case.y, test_case.absolute_tolerance,
                            test_case.relative_tolerance, "y"),
                     output(test_case.inv_variance,
                            test_case.absolute_tolerance,
                            test_case.relative_tolerance, "inv_variance")}},
        [&suite, &test_case] {
          return build_flagdnn_rmsnorm(suite.handle(), test_case);
        },
        [&test_case] { return build_rmsnorm_reference(test_case); },
        host_reference);
  }
  return suite.finish();
}

int run_batchnorm_functional_test(int argc, char **argv,
                                  std::span<const BatchnormTestCase> cases) {
  functional::FunctionalSuite suite(argc, argv, "batchnorm",
                                    "FLAGDNN_BATCHNORM_FUNCTIONAL");
  for (const BatchnormTestCase &test_case : cases) {
    validate_normalization_case(test_case);
    functional::HostReference host_reference;
    if (test_case.x.data_type != FLAGDNN_DATA_FLOAT32) {
      // The classic CoreX training reference only qualifies FP32. Validate
      // low-precision production, including running statistics, on the CPU.
      host_reference = [&test_case](const auto &inputs) {
        std::vector<std::size_t> axes;
        for (std::size_t axis = 0; axis < test_case.x.dimensions.size(); ++axis)
          if (axis != 1)
            axes.push_back(axis);
        auto outputs = reference::cpu::evaluate_normalization(
            {"batchnorm", test_case.x.dimensions, test_case.scale.dimensions,
             axes, test_case.epsilon},
            inputs);
        const auto channels = outputs[1].size();
        const double count = static_cast<double>(inputs[0].size() / channels);
        std::vector<float> mean(channels), variance(channels);
        for (std::size_t c = 0; c < channels; ++c) {
          const double inverse = outputs[2][c];
          const double population =
              std::max(0.0, 1.0 / (inverse * inverse) - test_case.epsilon);
          const double unbiased =
              count > 1 ? population * count / (count - 1) : 0;
          mean[c] = static_cast<float>((1 - test_case.momentum) * inputs[3][c] +
                                       test_case.momentum * outputs[1][c]);
          variance[c] =
              static_cast<float>((1 - test_case.momentum) * inputs[4][c] +
                                 test_case.momentum * unbiased);
        }
        outputs.push_back(std::move(mean));
        outputs.push_back(std::move(variance));
        return outputs;
      };
    }
    suite.run(
        {.operation = "batchnorm",
         .case_name = test_case.name,
         .inputs = {{test_case.x, functional::InputDomain::kReal, {}},
                    {test_case.scale, functional::InputDomain::kUnitScale, {}},
                    {test_case.bias, functional::InputDomain::kReal, {}},
                    {test_case.previous_running_mean,
                     functional::InputDomain::kReal,
                     {}},
                    {test_case.previous_running_variance,
                     functional::InputDomain::kVariance,
                     {}}},
         .outputs =
             {output(test_case.y, test_case.absolute_tolerance,
                     test_case.relative_tolerance, "y"),
              output(test_case.mean, test_case.absolute_tolerance,
                     test_case.relative_tolerance, "mean"),
              output(test_case.inv_variance, test_case.absolute_tolerance,
                     test_case.relative_tolerance, "inv_variance"),
              output(test_case.next_running_mean, test_case.absolute_tolerance,
                     test_case.relative_tolerance, "next_running_mean"),
              output(test_case.next_running_variance,
                     test_case.absolute_tolerance, test_case.relative_tolerance,
                     "next_running_variance")}},
        [&suite, &test_case] {
          return build_flagdnn_batchnorm(suite.handle(), test_case);
        },
        [&test_case] { return build_batchnorm_reference(test_case); },
        host_reference);
  }
  return suite.finish();
}

int run_batchnorm_inference_functional_test(
    int argc, char **argv, std::span<const BatchnormInferenceTestCase> cases) {
  functional::FunctionalSuite suite(argc, argv, "batchnorm_inference",
                                    "FLAGDNN_BATCHNORM_INFERENCE_FUNCTIONAL");
  for (const BatchnormInferenceTestCase &test_case : cases) {
    validate_normalization_case(test_case);
    suite.run(
        {.operation = "batchnorm_inference",
         .case_name = test_case.name,
         .inputs = {{test_case.x, functional::InputDomain::kReal, {}},
                    {test_case.mean, functional::InputDomain::kReal, {}},
                    {test_case.inv_variance,
                     functional::InputDomain::kVariance,
                     {}},
                    {test_case.scale, functional::InputDomain::kUnitScale, {}},
                    {test_case.bias, functional::InputDomain::kReal, {}}},
         .outputs = {output(test_case.y, test_case.absolute_tolerance,
                            test_case.relative_tolerance, "y")}},
        [&suite, &test_case] {
          return build_flagdnn_batchnorm_inference(suite.handle(), test_case);
        },
        [&test_case] {
          return build_batchnorm_inference_reference(test_case);
        });
  }
  return suite.finish();
}

} // namespace flagdnn::testing
