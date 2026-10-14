// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#include "common/index.hpp"
#include "benchmark/equivalent.hpp"
#include "capability.hpp"
#include "functional/pointwise_runner_support.hpp"
#include <acdnn.h>
#include <cctype>
#include <iostream>

namespace flagdnn::testing {
int run_thead_index_benchmark(int argc, char** argv, std::string_view operation) {
  namespace tv = validation::thead;
  namespace bm = benchmarking;
  try {
    const auto cases = make_index_cases(operation);
    if (argc == 2 && std::string_view(argv[1]) == "--dump-cases") {
      for (const auto& t : cases) std::cout << operation << '\t' << t.name << '\n';
      return 0;
    }
    if (argc != 3) throw std::invalid_argument("index benchmark requires compiler");
    std::string label(operation);
    for (auto& c : label) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    const char* filter = std::getenv("FLAGDNN_BENCHMARK_CASE");
    const auto catalog = tv::CapabilityCatalog::load(FLAGDNN_THEAD_ACDNN_CAPABILITY_CATALOG);
    catalog.validate_versions(FLAGDNN_THEAD_PPU_SDK_VERSION, ACDNN_VERSION, acdnnGetVersion());
    tv::check_driver(cuInit(0), "index init");
    CUdevice device;
    tv::check_driver(cuDeviceGet(&device, 0), "index device");
    tv::PrimaryContext primary(device);
    tv::ScopedCurrentContext current(primary.get());
    tv::DeviceStream stream;
    tv::functional::TemporaryCache cache;
    flagdnn::Handle handle("thead", 0);
    handle.set_compiler(argv[1], argv[2], cache.path().string());
    std::size_t executed = 0, skipped = 0;
    for (const auto& t : cases) {
      if (filter && filter[0] && t.name != filter) continue;
      if (catalog.lookup(operation, t.name).status != tv::CapabilityStatus::kUnsupported)
        throw std::runtime_error("index fallback requires unsupported native capability");
      const auto dtype = t.output.data_type;
      if (operation == "concatenate" && dtype != FLAGDNN_DATA_FLOAT32 &&
          dtype != FLAGDNN_DATA_FLOAT16 && dtype != FLAGDNN_DATA_BFLOAT16) {
        std::cout << "[SKIP][nvidia_baseline] op=" << operation << " case=" << t.name
                  << " reason=nvidia_reference_unavailable sdk=" << FLAGDNN_THEAD_PPU_SDK_VERSION
                  << " acdnn_header=" << ACDNN_VERSION << " acdnn_runtime=" << acdnnGetVersion()
                  << " target=" << handle.target_fingerprint()
                  << " dtype=" << tv::functional::data_type_name(dtype)
                  << " layout=" << tv::functional::layout_name(t.output)
                  << " shape=" << tv::functional::shape_name(t.output) << '\n';
        ++skipped;
        continue;
      }
      bm::BenchmarkCase spec;
      spec.name = t.name;
      spec.operation = bm::Operation::kGraph;
      for (const auto& x : t.inputs)
        spec.tensors.push_back({x.uid, x.data_type, x.dimensions, x.strides, x.binding_byte_offset});
      const auto& y = t.output;
      spec.tensors.push_back({y.uid, y.data_type, y.dimensions, y.strides, y.binding_byte_offset});
      const bm::thead::NativeBaselineMetadata metadata{std::string(operation), t.axis};
      tv::TestBenchmarkExecutable executable(build_flagdnn_index(handle, t));
      tv::run_equivalent_benchmark(spec, executable, stream, &metadata);
      ++executed;
    }
    if (!executed && !skipped) throw std::invalid_argument("index filter matched no cases");
    std::cout << "FLAGDNN_" << label << "_BENCHMARK: " << (executed ? "PASS" : "SKIP")
              << " cases=" << executed + skipped << " comparable_executed=" << executed
              << " reference_skipped=" << skipped << " equivalent_executed=" << executed << '\n';
    return executed ? 0 : 77;
  } catch (const std::exception& e) {
    std::cerr << "index benchmark failed: " << e.what() << '\n';
    return 1;
  }
}
}  // namespace flagdnn::testing
