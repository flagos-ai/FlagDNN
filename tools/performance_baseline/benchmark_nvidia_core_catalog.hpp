/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#ifndef FLAGDNN_TOOLS_NVIDIA_CORE_CATALOG_HPP_
#define FLAGDNN_TOOLS_NVIDIA_CORE_CATALOG_HPP_
#include "../../benchmark/common/cases.hpp"
#include "../../benchmark/common/runner.hpp"

// Capture the cases emitted by each real benchmark entry point. This retains
// its factory selection, input domains and attribute overrides verbatim.
namespace flagdnn::benchmarking {
inline std::vector<BenchmarkCase> captured_standalone_cases;
inline int capture_standalone_catalog(int, char**, std::span<const BenchmarkCase> cases,
                                      std::string_view) {
  captured_standalone_cases.assign(cases.begin(), cases.end());
  return 0;
}
}
#define run_benchmark_suite capture_standalone_catalog
#define main standalone_catalog_entry_mod
#include "../../benchmark/test_mod.cpp"
#undef main
#define main standalone_catalog_entry_pow
#include "../../benchmark/test_pow.cpp"
#undef main
#define main standalone_catalog_entry_div
#include "../../benchmark/test_div.cpp"
#undef main
#define main standalone_catalog_entry_cmp_eq
#include "../../benchmark/test_cmp_eq.cpp"
#undef main
#define main standalone_catalog_entry_cmp_neq
#include "../../benchmark/test_cmp_neq.cpp"
#undef main
#define main standalone_catalog_entry_cmp_lt
#include "../../benchmark/test_cmp_lt.cpp"
#undef main
#define main standalone_catalog_entry_cmp_le
#include "../../benchmark/test_cmp_le.cpp"
#undef main
#define main standalone_catalog_entry_cmp_gt
#include "../../benchmark/test_cmp_gt.cpp"
#undef main
#define main standalone_catalog_entry_cmp_ge
#include "../../benchmark/test_cmp_ge.cpp"
#undef main
#define main standalone_catalog_entry_logical_and
#include "../../benchmark/test_logical_and.cpp"
#undef main
#define main standalone_catalog_entry_logical_or
#include "../../benchmark/test_logical_or.cpp"
#undef main
#define main standalone_catalog_entry_binary_select
#include "../../benchmark/test_binary_select.cpp"
#undef main
#define main standalone_catalog_entry_neg
#include "../../benchmark/test_neg.cpp"
#undef main
#define main standalone_catalog_entry_sqrt
#include "../../benchmark/test_sqrt.cpp"
#undef main
#define main standalone_catalog_entry_rsqrt
#include "../../benchmark/test_rsqrt.cpp"
#undef main
#define main standalone_catalog_entry_reciprocal
#include "../../benchmark/test_reciprocal.cpp"
#undef main
#define main standalone_catalog_entry_ceil
#include "../../benchmark/test_ceil.cpp"
#undef main
#define main standalone_catalog_entry_floor
#include "../../benchmark/test_floor.cpp"
#undef main
#define main standalone_catalog_entry_exp
#include "../../benchmark/test_exp.cpp"
#undef main
#define main standalone_catalog_entry_log
#include "../../benchmark/test_log.cpp"
#undef main
#define main standalone_catalog_entry_erf
#include "../../benchmark/test_erf.cpp"
#undef main
#define main standalone_catalog_entry_sin
#include "../../benchmark/test_sin.cpp"
#undef main
#define main standalone_catalog_entry_cos
#include "../../benchmark/test_cos.cpp"
#undef main
#define main standalone_catalog_entry_tan
#include "../../benchmark/test_tan.cpp"
#undef main
#define main standalone_catalog_entry_leaky_relu
#include "../../benchmark/test_leaky_relu.cpp"
#undef main
#define main standalone_catalog_entry_leaky_relu_backward
#include "../../benchmark/test_leaky_relu_backward.cpp"
#undef main
#define main standalone_catalog_entry_gelu
#include "../../benchmark/test_gelu.cpp"
#undef main
#define main standalone_catalog_entry_gelu_backward
#include "../../benchmark/test_gelu_backward.cpp"
#undef main
#define main standalone_catalog_entry_gelu_approx_tanh
#include "../../benchmark/test_gelu_approx_tanh.cpp"
#undef main
#define main standalone_catalog_entry_gelu_approx_tanh_backward
#include "../../benchmark/test_gelu_approx_tanh_backward.cpp"
#undef main
#define main standalone_catalog_entry_softplus
#include "../../benchmark/test_softplus.cpp"
#undef main
#define main standalone_catalog_entry_elu_backward
#include "../../benchmark/test_elu_backward.cpp"
#undef main
#define main standalone_catalog_entry_softplus_backward
#include "../../benchmark/test_softplus_backward.cpp"
#undef main
#define main standalone_catalog_entry_matmul
#include "../../benchmark/test_matmul.cpp"
#undef main
#define main standalone_catalog_entry_batchnorm_inference
#include "../../benchmark/test_batchnorm_inference.cpp"
#undef main
#define main standalone_catalog_entry_layernorm
#include "../../benchmark/test_layernorm.cpp"
#undef main
#define main standalone_catalog_entry_rmsnorm
#include "../../benchmark/test_rmsnorm.cpp"
#undef main
#define main standalone_catalog_entry_conv_fprop
#include "../../benchmark/test_conv_fprop.cpp"
#undef main
#define main standalone_catalog_entry_conv_dgrad
#include "../../benchmark/test_conv_dgrad.cpp"
#undef main
#define main standalone_catalog_entry_conv_wgrad
#include "../../benchmark/test_conv_wgrad.cpp"
#undef main
#undef run_benchmark_suite
namespace standalone_cudnn {
struct CatalogEntry { const char* name; int (*entry)(int, char**); };
inline const CatalogEntry catalog_entries[] = {
  {"mod", standalone_catalog_entry_mod},
  {"pow", standalone_catalog_entry_pow},
  {"div", standalone_catalog_entry_div},
  {"cmp_eq", standalone_catalog_entry_cmp_eq},
  {"cmp_neq", standalone_catalog_entry_cmp_neq},
  {"cmp_lt", standalone_catalog_entry_cmp_lt},
  {"cmp_le", standalone_catalog_entry_cmp_le},
  {"cmp_gt", standalone_catalog_entry_cmp_gt},
  {"cmp_ge", standalone_catalog_entry_cmp_ge},
  {"logical_and", standalone_catalog_entry_logical_and},
  {"logical_or", standalone_catalog_entry_logical_or},
  {"binary_select", standalone_catalog_entry_binary_select},
  {"neg", standalone_catalog_entry_neg},
  {"sqrt", standalone_catalog_entry_sqrt},
  {"rsqrt", standalone_catalog_entry_rsqrt},
  {"reciprocal", standalone_catalog_entry_reciprocal},
  {"ceil", standalone_catalog_entry_ceil},
  {"floor", standalone_catalog_entry_floor},
  {"exp", standalone_catalog_entry_exp},
  {"log", standalone_catalog_entry_log},
  {"erf", standalone_catalog_entry_erf},
  {"sin", standalone_catalog_entry_sin},
  {"cos", standalone_catalog_entry_cos},
  {"tan", standalone_catalog_entry_tan},
  {"leaky_relu", standalone_catalog_entry_leaky_relu},
  {"leaky_relu_backward", standalone_catalog_entry_leaky_relu_backward},
  {"gelu", standalone_catalog_entry_gelu},
  {"gelu_backward", standalone_catalog_entry_gelu_backward},
  {"gelu_approx_tanh", standalone_catalog_entry_gelu_approx_tanh},
  {"gelu_approx_tanh_backward", standalone_catalog_entry_gelu_approx_tanh_backward},
  {"softplus", standalone_catalog_entry_softplus},
  {"elu_backward", standalone_catalog_entry_elu_backward},
  {"softplus_backward", standalone_catalog_entry_softplus_backward},
  {"matmul", standalone_catalog_entry_matmul},
  {"batchnorm_inference", standalone_catalog_entry_batchnorm_inference},
  {"layernorm", standalone_catalog_entry_layernorm},
  {"rmsnorm", standalone_catalog_entry_rmsnorm},
  {"conv_fprop", standalone_catalog_entry_conv_fprop},
  {"conv_dgrad", standalone_catalog_entry_conv_dgrad},
  {"conv_wgrad", standalone_catalog_entry_conv_wgrad},
};
}
#endif
