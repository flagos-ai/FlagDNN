/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "benchmark/native.hpp"
#include "common/convolution.hpp"
#include "common/dtype_runner.hpp"
#include "common/layout.hpp"
#include "common/matmul.hpp"
#include "common/pointwise.hpp"
#include "common/reduction.hpp"
#include "validation/functional/paired.hpp"
#include <cstdlib>
#include <iostream>
#include <stdexcept>
namespace flagdnn::testing::catalog {
Cases captured;
void record(std::string name, std::vector<TestTensor> tensors) {
  if (!captured.emplace(std::move(name), std::move(tensors)).second)
    throw std::runtime_error("duplicate catalog case");
}
bool same_tensor(const TestTensor &a, const TestTensor &b) {
  return a.uid == b.uid && a.data_type == b.data_type &&
         a.dimensions == b.dimensions && a.strides == b.strides &&
         a.binding_byte_offset == b.binding_byte_offset;
}
} // namespace flagdnn::testing::catalog
namespace flagdnn::benchmarking {
int run_hygon_benchmark_suite(int, char **,
                              std::span<const BenchmarkCase> cases,
                              std::string_view, const NativeBuilder &) {
  for (const auto &value : cases) {
    std::vector<testing::TestTensor> tensors;
    for (const auto &t : value.tensors)
      tensors.push_back(
          {t.uid, t.data_type, t.dimensions, t.strides, t.binding_byte_offset});
    testing::catalog::record(value.name, std::move(tensors));
  }
  return 0;
}
} // namespace flagdnn::benchmarking
namespace flagdnn::testing {
int run_hygon_dtype_catalog(int, char **, std::string_view, std::string_view);
// No vendor builders are executed by the catalog probe.
std::unique_ptr<TestExecutable> build_matmul_reference(const MatmulTestCase &) {
  return {};
}
std::unique_ptr<TestExecutable>
build_convolution_reference(const ConvolutionTestCase &) {
  return {};
}
std::unique_ptr<TestExecutable>
build_reduction_reference(const ReductionTestCase &) {
  return {};
}
std::unique_ptr<TestExecutable> build_layout_reference(const LayoutTestCase &) {
  return {};
}
int run_cudnn_pointwise_tests(int, char **,
                              std::span<const PointwiseTestCase> cases,
                              std::string_view, bool) {
  // native_copy/boolean supplies identity or logical modes only. Mirror the
  // dtype exclusions in NVIDIA pointwise_runner; the Python guard tracks it.
  for (const auto &value : cases) {
    const auto type = value.inputs.front().data_type;
    if ((type == FLAGDNN_DATA_INT32 &&
         value.mode != FLAGDNN_POINTWISE_IDENTITY) ||
        type == FLAGDNN_DATA_FP8_E8M0)
      continue;
    auto tensors = value.inputs;
    tensors.push_back(value.output);
    catalog::record(value.name, std::move(tensors));
  }
  return 0;
}
} // namespace flagdnn::testing
int main() {
  using namespace flagdnn::testing;
  try {
    std::size_t groups = 0, cases = 0;
    for (const auto &[operation, category] :
         std::vector<std::pair<std::string, std::string>>{
             {"logical_and", "boolean"},
             {"logical_or", "boolean"},
             {"logical_not", "boolean"},
             {"identity", "copy"},
             {"reshape", "copy"},
             {"transpose", "copy"},
             {"slice", "copy"},
             {"matmul", "precision"},
             {"conv_fprop", "precision"},
             {"conv_dgrad", "precision"},
             {"conv_wgrad", "precision"},
             {"reduction", "fp32_output"}}) {
      for (const int precision :
           category == "precision" ? std::vector{1, 2} : std::vector{0}) {
        if (setenv("FLAGDNN_INPUT_PRECISION", std::to_string(precision).c_str(),
                   1) != 0)
          throw std::runtime_error("setenv failed");
        catalog::captured.clear();
        if (run_native_dtype_benchmark(3, nullptr, operation, category))
          throw std::runtime_error("NVIDIA catalog failed");
        auto expected = std::move(catalog::captured);
        catalog::captured.clear();
        if (run_hygon_dtype_catalog(3, nullptr, operation, category))
          throw std::runtime_error("Hygon catalog failed");
        const auto &actual = catalog::captured;
        if (expected.empty() || actual.size() != expected.size())
          throw std::runtime_error(operation +
                                   ": case count differs from NVIDIA");
        for (const auto &[name, tensors] : expected) {
          if (!actual.contains(name) ||
              actual.at(name).size() != tensors.size())
            throw std::runtime_error("missing NVIDIA case/tensors: " + name);
          for (std::size_t i = 0; i < tensors.size(); ++i)
            if (!catalog::same_tensor(tensors[i], actual.at(name)[i]))
              throw std::runtime_error(
                  "shape/dtype/stride/offset differs from NVIDIA: " + name);
        }
        ++groups;
        cases += expected.size();
      }
    }
    std::cout << "PASS NVIDIA/Hygon native catalogs: groups=" << groups
              << " cases=" << cases
              << " (names, UID, dtype, shape, strides, binding offsets)\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
