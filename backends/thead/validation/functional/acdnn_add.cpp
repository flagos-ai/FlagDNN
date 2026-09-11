// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "common/add.hpp"
#include "pointwise_reference.hpp"

#include "capability.hpp"

#include <memory>
#include <stdexcept>

#ifndef FLAGDNN_THEAD_ACDNN_CAPABILITY_CATALOG
#define FLAGDNN_THEAD_ACDNN_CAPABILITY_CATALOG "capability.json"
#endif

namespace flagdnn::testing {

std::unique_ptr<AddExecutable>
build_add_reference(const AddTestCase &test_case) {
  using namespace flagdnn::validation::thead;
  static const CapabilityCatalog catalog =
      CapabilityCatalog::load(FLAGDNN_THEAD_ACDNN_CAPABILITY_CATALOG);
  const CapabilityRecord &record = catalog.lookup("add", test_case.name);
  if (record.status == CapabilityStatus::kUnsupported) {
    throw std::invalid_argument(
        "unsupported Add case reached acDNN reference builder");
  }
  return make_acdnn_pointwise_reference(
      {.mode = FLAGDNN_POINTWISE_ADD,
       .inputs = {test_case.left, test_case.right},
       .output = test_case.output,
       .alpha = test_case.alpha},
      record);
}

}  // namespace flagdnn::testing
