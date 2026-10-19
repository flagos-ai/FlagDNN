// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cctype>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/pointwise.hpp"
#include "functional/raw_reference.hpp"
#include "functional/cpu_reference.hpp"
#include "functional/runner_support.hpp"

namespace flagdnn::testing {
namespace {

std::string operation_from_marker(std::string_view marker) {
  constexpr std::string_view prefix = "FLAGDNN_";
  constexpr std::string_view suffix = "_FUNCTIONAL";
  if (!marker.starts_with(prefix) || !marker.ends_with(suffix) ||
      marker.size() <= prefix.size() + suffix.size()) {
    throw std::invalid_argument("pointwise suite marker is invalid");
  }
  std::string result(marker.substr(
      prefix.size(), marker.size() - prefix.size() - suffix.size()));
  std::transform(result.begin(), result.end(), result.begin(),
                 [](unsigned char character) {
                   return static_cast<char>(std::tolower(character));
                 });
  return result;
}

} // namespace

int run_pointwise_functional_test(int argc, char **argv,
                                  std::span<const PointwiseTestCase> cases,
                                  std::string_view suite_name) {
  namespace functional = iluvatar::validation::functional;
  const std::string operation = operation_from_marker(suite_name);
  functional::FunctionalSuite suite(argc, argv, operation,
                                    std::string(suite_name));
  for (const PointwiseTestCase &test_case : cases) {
    validate_pointwise_case(test_case);
    functional::CasePlan plan{
        .operation = operation,
        .case_name = test_case.name,
        .inputs = {},
        .outputs = {{test_case.output, test_case.absolute_tolerance,
                     test_case.relative_tolerance, "output"}},
    };
    for (std::size_t index = 0; index < test_case.inputs.size(); ++index) {
      plan.inputs.push_back(
          {test_case.inputs[index],
           functional::pointwise_input_domain(
               static_cast<int>(test_case.input_domains[index])),
           {}});
    }
    if (test_case.mode == FLAGDNN_POINTWISE_IDENTITY &&
        test_case.inputs[0].data_type != FLAGDNN_DATA_FLOAT32 &&
        test_case.inputs[0].data_type != FLAGDNN_DATA_FLOAT16) {
      const auto values = functional::raw_pattern(test_case.inputs[0]);
      functional::BuildExecutable reference;
      if (test_case.inputs[0].data_type == FLAGDNN_DATA_BOOLEAN)
        reference = [&] { return build_pointwise_reference(test_case); };
      suite.run_raw(
          plan,
          [&] { return build_flagdnn_pointwise(suite.handle(), test_case); },
          {values}, {reference::cpu::evaluate_identity_bytes(values)}, reference);
      continue;
    }
    if (test_case.inputs[0].data_type == FLAGDNN_DATA_INT32) {
      functional::run_integer_case(
          suite, plan, test_case.mode,
          static_cast<std::int32_t>(test_case.alpha),
          [&] { return build_flagdnn_pointwise(suite.handle(), test_case); });
      continue;
    }
    const bool sigmoid_composition =
        (operation == "swish_backward" || operation == "softplus_backward") &&
        test_case.inputs[0].data_type != FLAGDNN_DATA_BFLOAT16;
    const bool leaky_backward =
        operation == "leaky_relu_backward" &&
        test_case.attributes.flags ==
            FLAGDNN_POINTWISE_ATTRIBUTE_RELU_LOWER_CLIP_SLOPE &&
        test_case.inputs[0].data_type != FLAGDNN_DATA_BFLOAT16;
    const bool classic_backward =
        test_case.inputs[0].data_type != FLAGDNN_DATA_BFLOAT16 &&
        (leaky_backward || sigmoid_composition ||
         operation == "tanh_backward" ||
         ((operation == "elu_backward" || operation == "relu_backward") &&
          test_case.attributes.flags == 0));
    if (operation.ends_with("_backward") && operation != "sigmoid_backward" &&
        !classic_backward) {
      suite.run(
          plan,
          [&] { return build_flagdnn_pointwise(suite.handle(), test_case); },
          {}, functional::pointwise_cpu_reference(
                  plan, test_case.mode, test_case.alpha, test_case.attributes));
      continue;
    }
    suite.run(
        plan,
        [&suite, &test_case] {
          return build_flagdnn_pointwise(suite.handle(), test_case);
        },
        [&test_case] { return build_pointwise_reference(test_case); },
        {}, false,
        functional::pointwise_cpu_reference(
            plan, test_case.mode, test_case.alpha, test_case.attributes));
  }
  return suite.finish();
}

} // namespace flagdnn::testing
