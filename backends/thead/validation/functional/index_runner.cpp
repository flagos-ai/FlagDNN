// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#include "common/index.hpp"
#include "cpu_validation.hpp"
#include "functional/pointwise_runner_support.hpp"
#include <cctype>
#include <iostream>

namespace flagdnn::testing {
int run_index_functional_test(int argc, char** argv,
                             std::span<const IndexTestCase> cases, bool benchmark) {
  namespace tv = validation::thead;
  namespace bm = benchmarking;
  try {
    if (argc != 3 || cases.empty() || benchmark)
      throw std::invalid_argument("index functional requires compiler and cases");
    const auto operation = cases.front().operation;
    std::string label = operation;
    for (auto& c : label) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    const char* filter = std::getenv(("FLAGDNN_" + label + "_CASE").c_str());
    tv::check_driver(cuInit(0), "index init");
    CUdevice device;
    tv::check_driver(cuDeviceGet(&device, 0), "index device");
    tv::PrimaryContext primary(device);
    tv::ScopedCurrentContext current(primary.get());
    tv::DeviceStream stream;
    tv::functional::TemporaryCache cache;
    flagdnn::Handle handle("thead", 0);
    handle.set_compiler(argv[1], argv[2], cache.path().string());
    std::size_t executed = 0;
    for (const auto& t : cases) {
      if (filter && filter[0] && t.name.find(filter) == std::string::npos) continue;
      bm::BenchmarkCase spec;
      spec.name = t.name;
      spec.operation = bm::Operation::kGraph;
      for (const auto& x : t.inputs)
        spec.tensors.push_back({x.uid, x.data_type, x.dimensions, x.strides, x.binding_byte_offset});
      const auto& y = t.output;
      spec.tensors.push_back({y.uid, y.data_type, y.dimensions, y.strides, y.binding_byte_offset});
      const bm::thead::NativeBaselineMetadata metadata{operation, t.axis};
      tv::TestBenchmarkExecutable executable(build_flagdnn_index(handle, t));
      tv::run_cpu_validated_case(spec, executable, stream, &metadata);
      ++executed;
    }
    if (!executed) throw std::invalid_argument("index filter matched no cases");
    std::cout << "FLAGDNN_" << label << "_FUNCTIONAL: PASS cases=" << executed
              << " executed=" << executed << " skipped=0\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "index functional failed: " << e.what() << '\n';
    return 1;
  }
}
}  // namespace flagdnn::testing
