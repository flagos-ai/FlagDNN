/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "common/dtype_runner.hpp"
#include "validation/functional/dtype_runner.hpp"
#include <iostream>
#include <unordered_map>
namespace flagdnn::testing {
int run_native_precision_benchmark(int, char **, std::string_view);
int run_native_dtype_benchmark(int argc, char **argv,
                               std::string_view operation,
                               std::string_view category) {
  const bool benchmark = category != "functional_dtype";
  if (category == "precision")
    return run_native_precision_benchmark(argc, argv, operation);
  if (operation == "reduction")
    return run_ascend_reduction_dtype_cases(argc, argv, make_reduction_cases(),
                                            benchmark);
  if (operation == "reshape" || operation == "transpose" ||
      operation == "slice")
    return run_ascend_layout_dtype_cases(
        argc, argv,
        make_layout_cases(operation == "reshape" ? LayoutOperation::kReshape
                          : operation == "transpose"
                              ? LayoutOperation::kTranspose
                              : LayoutOperation::kSlice),
        benchmark, category == "copy");
  if (operation == "add" && !benchmark)
    return run_ascend_add_dtype_cases(argc, argv, make_add_cases());
  const std::unordered_map<std::string_view, flagdnnPointwiseMode_t> modes = {
      {"add", FLAGDNN_POINTWISE_ADD},
      {"sub", FLAGDNN_POINTWISE_SUB},
      {"mul", FLAGDNN_POINTWISE_MUL},
      {"div", FLAGDNN_POINTWISE_DIV},
      {"mod", FLAGDNN_POINTWISE_MOD},
      {"pow", FLAGDNN_POINTWISE_POW},
      {"min", FLAGDNN_POINTWISE_MIN},
      {"max", FLAGDNN_POINTWISE_MAX},
      {"neg", FLAGDNN_POINTWISE_NEG},
      {"abs", FLAGDNN_POINTWISE_ABS},
      {"relu", FLAGDNN_POINTWISE_RELU_FWD},
      {"identity", FLAGDNN_POINTWISE_IDENTITY},
      {"cmp_eq", FLAGDNN_POINTWISE_CMP_EQ},
      {"cmp_neq", FLAGDNN_POINTWISE_CMP_NEQ},
      {"cmp_lt", FLAGDNN_POINTWISE_CMP_LT},
      {"cmp_le", FLAGDNN_POINTWISE_CMP_LE},
      {"cmp_gt", FLAGDNN_POINTWISE_CMP_GT},
      {"cmp_ge", FLAGDNN_POINTWISE_CMP_GE},
      {"logical_not", FLAGDNN_POINTWISE_LOGICAL_NOT},
      {"logical_and", FLAGDNN_POINTWISE_LOGICAL_AND},
      {"logical_or", FLAGDNN_POINTWISE_LOGICAL_OR},
      {"binary_select", FLAGDNN_POINTWISE_BINARY_SELECT}};
  PointwiseCaseDefinition definition;
  definition.operation_name = operation;
  definition.mode = modes.at(operation);
  if (category == "boolean")
    definition.input_domain = PointwiseInputDomain::kLogical;
  const bool unary = operation == "neg" || operation == "abs" ||
                     operation == "relu" || operation == "identity" ||
                     operation == "logical_not";
  const auto cases = operation == "binary_select"
                         ? make_binary_select_cases(definition)
                     : unary ? make_unary_pointwise_cases(definition)
                             : make_binary_pointwise_cases(definition);
  return run_ascend_pointwise_dtype_cases(argc, argv, cases, benchmark,
                                          category == "boolean" ||
                                              category == "copy");
}
} // namespace flagdnn::testing
