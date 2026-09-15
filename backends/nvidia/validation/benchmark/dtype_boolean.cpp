/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include <stdexcept>

#include "validation/functional/pointwise_runner.hpp"
namespace flagdnn::testing {
int run_cudnn_boolean_benchmark(int argc, char** argv,
                                std::string_view operation) {
  PointwiseCaseDefinition definition;
  definition.operation_name = operation;
  if (operation == "logical_not")
    definition.mode = FLAGDNN_POINTWISE_LOGICAL_NOT;
  else if (operation == "logical_and")
    definition.mode = FLAGDNN_POINTWISE_LOGICAL_AND;
  else if (operation == "logical_or")
    definition.mode = FLAGDNN_POINTWISE_LOGICAL_OR;
  else
    throw std::invalid_argument("unknown boolean benchmark operation");
  definition.input_domain = PointwiseInputDomain::kLogical;
  const auto cases = operation == "logical_not"
                         ? make_unary_pointwise_cases(definition)
                         : make_binary_pointwise_cases(definition);
  return run_cudnn_pointwise_tests(argc, argv, cases,
                                   "FLAGDNN_BOOLEAN_BENCHMARK", true);
}
}  // namespace flagdnn::testing
