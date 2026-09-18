// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include <acdnn.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <iostream>
#include <string>

#include "functional/pointwise_runner_support.hpp"
#include "unsupported_cases.hpp"

#ifndef FLAGDNN_CAPABILITY_SUITE
#define FLAGDNN_CAPABILITY_SUITE "functional"
#endif

int main(int argc, char **argv) {
  namespace tv = flagdnn::validation::thead;
  const std::string operation = FLAGDNN_CAPABILITY_OPERATOR;
  std::string label = operation;
  std::transform(
      label.begin(), label.end(), label.begin(),
      [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
  const bool benchmark = std::string(FLAGDNN_CAPABILITY_SUITE) == "benchmark";
  const std::string suite =
      "FLAGDNN_" + label + (benchmark ? "_BENCHMARK" : "_FUNCTIONAL");
  try {
    const auto cases = tv::unsupported_cases(operation);
    if (argc == 2 && std::string_view(argv[1]) == "--dump-cases") {
      for (const auto &test : cases) {
        std::cout << operation << '\t' << test.name << '\n';
      }
      return 0;
    }
    const auto catalog =
        tv::CapabilityCatalog::load(FLAGDNN_THEAD_ACDNN_CAPABILITY_CATALOG);
    catalog.validate_versions(FLAGDNN_THEAD_PPU_SDK_VERSION, ACDNN_VERSION,
                              static_cast<std::int64_t>(acdnnGetVersion()));
    const char *filter = std::getenv(("FLAGDNN_" + label + "_CASE").c_str());
    std::size_t skipped = 0;
    for (const auto &test : cases) {
      if (filter && test.name.find(filter) == std::string::npos) continue;
      const auto &record = catalog.lookup(operation, test.name);
      if (record.status != tv::CapabilityStatus::kUnsupported) {
        throw std::runtime_error(
            "capability gate requires an unsupported record: " + test.name);
      }
      std::cout << "[SKIP][acdnn] op=" << operation << " case=" << test.name
                << " reason=" << record.reason_code
                << " sdk=" << FLAGDNN_THEAD_PPU_SDK_VERSION
                << " acdnn_header=" << ACDNN_VERSION
                << " acdnn_runtime=" << acdnnGetVersion()
                << " target=thead dtype="
                << tv::functional::data_type_name(test.tensor.data_type)
                << " layout=" << tv::functional::layout_name(test.tensor)
                << " shape=" << tv::functional::shape_name(test.tensor) << '\n';
      ++skipped;
    }
    if (skipped == 0) throw std::runtime_error("case filter matched no cases");
    std::cout << suite << ": SKIP cases=" << skipped
              << (benchmark ? " comparable_executed=0 reference_skipped="
                            : " executed=0 skipped=")
              << skipped << '\n';
    return 77;
  } catch (const std::exception &error) {
    std::cerr << suite << ": FAIL reason=" << error.what() << '\n';
    return 1;
  }
}
