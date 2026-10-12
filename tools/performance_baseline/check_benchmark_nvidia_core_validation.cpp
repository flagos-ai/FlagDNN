/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
/* GPU-free checks for core catalog selection and full-output validation.
   c++ -O2 -std=c++20 -I. -Iinclude -Ibenchmark -Ibuild/nvidia/generated/include \
     tools/performance_baseline/check_benchmark_nvidia_core_validation.cpp \
     build/nvidia/benchmark/libflagdnn_benchmark_cases.a \
     build/nvidia/backends/nvidia/validation/libflagdnn_validation_nvidia_tensor_io.a \
     -o /tmp/check_benchmark_nvidia_core_validation
   /tmp/check_benchmark_nvidia_core_validation [core-cases.txt]
*/
#include "benchmark_nvidia_core_catalog.hpp"
#include "benchmark_nvidia_core_validation.hpp"
#include "../../backends/nvidia/validation/benchmark/cases.hpp"

#include <fstream>
#include <iostream>
#include <map>
#include <set>

namespace {
using namespace standalone_cudnn;
namespace bench = flagdnn::benchmarking;
namespace io = flagdnn::validation::nvidia::tensor_io;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

template <typename Function>
void require_rejection(Function&& function, const char* message) {
  try {
    function();
  } catch (const std::runtime_error&) {
    return;
  }
  throw std::runtime_error(message);
}

void check_storage() {
  const bench::TensorSpec tensor{17, FLAGDNN_DATA_FLOAT32, {2, 3}, {5, 1}, 16};
  const std::vector<std::uint8_t> encoded(8 * sizeof(float), 0x39);
  for (const bool input : {false, true}) {
    const auto storage = make_core_tensor_storage(tensor, encoded, input, sizeof(float));
    auto current = storage.initial;
    require(validate_core_storage(storage, current) > 2 * kCoreGuardBytes,
            "guards/binding offset were not counted");
    for (const auto offset : {std::size_t{0}, storage.binding_start - 1,
                             storage.binding_start + 3 * sizeof(float),
                             storage.initial.size() - 1}) {
      current = storage.initial;
      current[offset] ^= 1;
      require_rejection([&] { validate_core_storage(storage, current); },
                        "padding or guard write escaped validation");
    }
    current = storage.initial;
    current[storage.binding_start + 7 * sizeof(float)] ^= 1;
    if (input) {
      require_rejection([&] { validate_core_storage(storage, current); },
                        "input modification escaped validation");
    } else {
      validate_core_storage(storage, current);
    }
  }
  // Only logical bits may change, including the partially occupied final byte.
  const bench::TensorSpec boolean{18, FLAGDNN_DATA_BOOLEAN, {2, 3}, {5, 1}, 0};
  const auto storage = make_core_tensor_storage(boolean, std::vector<std::uint8_t>{0}, false, 1);
  auto current = storage.initial;
  current[storage.binding_start] = 0xE7;
  validate_core_storage(storage, current);
  current[storage.binding_start] |= 0x08;
  require_rejection([&] { validate_core_storage(storage, current); },
                    "nonlogical BOOLEAN bit write escaped validation");
}

void check_comparison_and_poison() {
  for (const auto type : {FLAGDNN_DATA_FLOAT32, FLAGDNN_DATA_FLOAT16, FLAGDNN_DATA_BFLOAT16}) {
    const std::vector<float> values{0, -0.0F, 0.0001F, -0.0001F, 1, -2, 31, -250, 65504};
    const auto rounded = io::decode(io::encode(values, type, io::BooleanEncoding::kByte),
                                    type, values.size(), io::BooleanEncoding::kByte);
    for (const float expected : rounded) {
      const std::vector<float> poison{core_output_poison(expected, 0.07, 0.07)};
      const auto stored = io::decode(io::encode(poison, type, io::BooleanEncoding::kByte),
                                     type, 1, io::BooleanEncoding::kByte);
      require(std::isfinite(stored.front()), "finite poison overflowed output dtype");
      ValidationAccuracy accuracy;
      std::size_t checked = 0;
      const std::vector<float> reference{expected};
      require_rejection([&] {
        compare_core_values(stored, reference, 0.07, 0.07, accuracy, checked);
      }, "unwritten finite poison passed output comparison");
    }
  }
  std::vector<float> expected(1025, 1), actual = expected;
  ValidationAccuracy accuracy;
  std::size_t checked = 0;
  compare_core_values(actual, expected, 0, 0, accuracy, checked);
  require(checked == expected.size(), "comparison did not check every element");
  actual.back() = -1;
  checked = 0;
  require_rejection([&] {
    compare_core_values(actual, expected, 0, 0, accuracy, checked);
  }, "late output corruption escaped comparison");
  require(checked == expected.size(), "comparison missed the last element");
  actual.back() = std::numeric_limits<float>::quiet_NaN();
  require_rejection([&] {
    compare_core_values(actual, expected, 0, 0, accuracy, checked);
  }, "nonfinite output escaped comparison");
  require_rejection([&] { core_output_poison(0, 2, 2); },
                    "indistinguishable poison should be rejected");
}

void check_catalog(const char* case_file) {
  std::set<std::string> requested;
  if (case_file) {
    std::ifstream input(case_file);
    require(bool(input), "cannot open requested core catalog");
    for (std::string name; std::getline(input, name);) {
      if (!name.empty() && name.back() == '\r') name.pop_back();
      if (!name.empty()) requested.insert(name);
    }
    require(requested.size() == 291, "requested core catalog must contain 291 cases");
  }
  std::map<std::string, std::size_t> expected_counts;
  for (const char* operation : {"abs", "add", "add_square", "elu", "max", "min", "mul",
                               "relu", "relu_backward", "scale", "sigmoid", "sigmoid_backward",
                               "sub", "swish", "swish_backward", "tanh", "tanh_backward"})
    expected_counts[operation] = 15;
  expected_counts["identity"] = 12;
  expected_counts["batchnorm"] = 24;
  std::map<std::string, std::size_t> actual_counts;
  std::set<std::string> actual_names;
  for (const auto& entry : catalog_entries) {
    if (!expected_counts.contains(entry.name)) continue;
    entry.entry(0, nullptr);
    for (const auto& spec : bench::cudnn_benchmark_cases(bench::captured_standalone_cases)) {
      const auto type = spec.tensors.front().data_type;
      const bool batchnorm = spec.operation == bench::Operation::kBatchnorm;
      if (type != FLAGDNN_DATA_BFLOAT16 && !(batchnorm && type == FLAGDNN_DATA_FLOAT16)) continue;
      require(actual_names.insert(spec.name).second, "duplicate core case name");
      ++actual_counts[entry.name];
      const auto config = core_benchmark_configuration(spec);
      require(config.warmup_iterations == (batchnorm ? 5 : 10) &&
                  config.sample_count == (batchnorm ? 10 : 20) &&
                  config.iterations_per_sample == (batchnorm ? 20 : 50),
              "core timing differs from required configuration");
      if (batchnorm) {
        require(spec.tensors.size() == 10 && spec.output_count == 5,
                "BatchNorm must retain all five outputs");
        require(spec.tensors[1].data_type == type && spec.tensors[2].data_type == type,
                "BatchNorm scale/bias must retain low precision");
        for (const auto index : {3, 4, 6, 7, 8, 9})
          require(spec.tensors[index].data_type == FLAGDNN_DATA_FLOAT32,
                  "BatchNorm statistics must remain FLOAT32");
        require(spec.input_domains.size() == 5 &&
                    spec.input_domains[4] == bench::InputDomain::kPositive,
                "BatchNorm running variance domain was lost");
      }
      if (std::string_view(entry.name) == "add_square") {
        require(spec.graph.intermediates.size() == 1 && spec.graph.nodes.size() == 2,
                "add_square intermediate/node graph was lost");
        require(spec.graph.intermediates.front().data_type == FLAGDNN_DATA_BFLOAT16,
                "add_square square intermediate must quantize to BF16");
      }
    }
  }
  require(actual_counts == expected_counts, "main core catalog operation counts differ");
  require(actual_names.size() == 291, "core missing-cases selection differs from 291");
  if (case_file) require(actual_names == requested, "core names differ from exact requested inventory");
}
}  // namespace

int main(int argc, char** argv) {
  try {
    require(argc <= 2, "usage: check_benchmark_nvidia_core_validation [core-cases.txt]");
    check_storage();
    check_comparison_and_poison();
    check_catalog(argc == 2 ? argv[1] : nullptr);
    std::cout << "PASS: core guards, full comparisons, finite poison, exact 291-case catalog\n";
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
