/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "common/layout.hpp"
#include "host_runner.hpp"
#include "reference/cpu/layout.hpp"
namespace flagdnn::testing {
int run_layout_functional_test(int argc, char **argv,
                               std::span<const LayoutTestCase> cases,
                               std::string_view suite_name) {
  namespace host = hygon_functional::host;
  namespace cpu = reference::cpu;
  const std::string operation =
      cases.front().operation == LayoutOperation::kReshape     ? "reshape"
      : cases.front().operation == LayoutOperation::kTranspose ? "transpose"
                                                               : "slice";
  (void)suite_name;
  return host::run_suite(
      argc, argv, cases, operation, false,
      [&](const LayoutTestCase &c, const auto &handle, auto &stream) {
        validate_layout_case(c);
        const cpu::LayoutParameters parameters{
            operation,     c.input.dimensions, c.output.dimensions,
            c.permutation, c.slices,           c.slice_strides};
        host::run_copy(
            c.name, std::vector<TestTensor>{c.input}, c.output, handle, stream,
            [&](std::size_t i) {
              return std::pair<std::size_t, std::size_t>{
                  0, cpu::layout_source_index(parameters, i)};
            },
            [&](flagdnn::Handle &h) { return build_flagdnn_layout(h, c); });
      });
}
} // namespace flagdnn::testing
