/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "native.hpp"

#include "common/index.hpp"
#include "equivalent_baseline.hpp"
#include "reference/cpu/index.hpp"

#include <cstdlib>
#include <iostream>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace flagdnn::testing {
namespace {

namespace bm = benchmarking;

class IndexExecutable final : public bm::BenchmarkExecutable {
 public:
  explicit IndexExecutable(std::unique_ptr<TestExecutable> executable)
      : executable_(std::move(executable)) {}
  void prepare(std::span<const flagdnnBinding_t> bindings,
               flagdnnStream_t stream) override {
    executable_->prepare(bindings, stream);
  }
  std::size_t workspace_size() const noexcept override {
    return executable_->workspace_size();
  }
  void execute(std::span<const flagdnnBinding_t> bindings, void* workspace,
               std::size_t bytes, flagdnnStream_t stream) override {
    executable_->execute(bindings, workspace, bytes, stream);
  }

 private:
  std::unique_ptr<TestExecutable> executable_;
};

bm::TensorSpec tensor(const TestTensor& value) {
  return {value.uid, value.data_type, value.dimensions, value.strides,
          value.binding_byte_offset};
}

bool has_nvidia_reference(const IndexTestCase& value) {
  // These exclusions match NVIDIA's native index catalog. No other dtype's
  // timings can stand in for unavailable INT32, bool, or FP8 concatenate cases.
  return value.operation == "gen_index" ||
         value.output.data_type == FLAGDNN_DATA_FLOAT32 ||
         value.output.data_type == FLAGDNN_DATA_FLOAT16 ||
         value.output.data_type == FLAGDNN_DATA_BFLOAT16;
}

}  // namespace

int run_native_index_benchmark(int argc, char** argv,
                               std::string_view operation) {
  if (argc != 3) {
    std::cerr << "usage: index benchmark COMPILER_EXECUTABLE COMPILER_ENTRY\n";
    return 2;
  }
  try {
    const auto catalog = make_index_cases(operation);
    std::vector<bm::BenchmarkCase> cases;
    std::map<std::string, IndexTestCase, std::less<>> native_cases;
    const char* filter = std::getenv("FLAGDNN_BENCHMARK_CASE");
    const bool filtered = filter != nullptr && filter[0] != '\0';
    std::size_t excluded = 0;
    for (const auto& value : catalog) {
      if (!has_nvidia_reference(value)) {
        if (!filtered || value.name == filter) {
          ++excluded;
          std::cout
              << (filtered ? "[SKIP]" : "[EXCLUDED]")
              << "[nvidia_baseline] op=" << operation
              << " case=" << value.name
              << " reason=NVIDIA native concatenate reference "
                 "supports only fp32/fp16/bfloat16\n";
        }
        continue;
      }
      bm::BenchmarkCase spec;
      spec.name = value.name;
      // Index operations use their native graph builder and independent CPU
      // oracle. The empty graph is only a carrier for the common timing path;
      // explicit metadata supplies the operation and axis for baseline lookup.
      spec.operation = bm::Operation::kGraph;
      for (const auto& input : value.inputs)
        spec.tensors.push_back(tensor(input));
      spec.tensors.push_back(tensor(value.output));
      spec.output_count = 1;
      spec.absolute_tolerance = 0.0;
      spec.relative_tolerance = 0.0;
      spec.benchmark = {10, 20, 50};
      cases.push_back(std::move(spec));
      native_cases.emplace(value.name, value);
    }
    const std::string suite = operation == "concatenate"
                                  ? "FLAGDNN_CONCATENATE_BENCHMARK"
                                  : "FLAGDNN_GEN_INDEX_BENCHMARK";
    if (excluded != 0) {
      std::cout << "[baseline_coverage] op=" << operation
                << " excluded_without_nvidia_reference=" << excluded << '\n';
      if (filtered) {
        std::cout << suite << ": SKIP cases=1 executed=0 skipped=1\n";
        return 77;
      }
    }

    return bm::run_hygon_benchmark_suite(
        argc, argv, cases, suite,
        [&](flagdnn::Handle& handle, const bm::BenchmarkCase& spec) {
          return std::make_unique<IndexExecutable>(
              build_flagdnn_index(handle, native_cases.at(spec.name)));
        },
        [&](const bm::BenchmarkCase& spec,
            const std::vector<std::vector<float>>& inputs) {
          const auto& value = native_cases.at(spec.name);
          if (inputs.size() != value.inputs.size())
            throw std::invalid_argument("index CPU oracle input count differs");
          reference::cpu::IndexParameters parameters;
          parameters.operation = value.operation;
          parameters.output_shape = value.output.dimensions;
          parameters.axis = value.axis;
          for (std::size_t i = 0; i < value.inputs.size(); ++i) {
            const auto& dimensions = value.inputs[i].dimensions;
            parameters.input_shapes.push_back(dimensions);
            const auto count = std::accumulate(
                dimensions.begin(), dimensions.end(), std::size_t{1},
                std::multiplies<>());
            if (inputs[i].size() != count)
              throw std::invalid_argument("index CPU oracle input size differs");
          }
          const auto count = std::accumulate(
              value.output.dimensions.begin(), value.output.dimensions.end(),
              std::size_t{1}, std::multiplies<>());
          std::vector<float> output(count);
          for (std::size_t i = 0; i < count; ++i) {
            const auto [source, index] =
                reference::cpu::index_source(parameters, i);
            output[i] = value.operation == "gen_index"
                            ? static_cast<float>(index)
                            : inputs.at(source).at(index);
          }
          return std::vector<std::vector<float>>{std::move(output)};
        },
        [&](const bm::BenchmarkCase& spec) {
          const auto& value = native_cases.at(spec.name);
          return bm::NativeBaselineMetadata{value.operation, value.axis};
        });
  } catch (const std::exception& error) {
    std::cerr << "index benchmark failed: " << error.what() << '\n';
    return 1;
  }
}

}  // namespace flagdnn::testing
