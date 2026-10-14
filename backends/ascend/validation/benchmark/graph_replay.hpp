/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include "validation/acl_runtime.hpp"

namespace flagdnn::testing::ascend {

// Capture construction and destruction are outside the measured window.
// EventTimer measures one replay; divide its sample by captured iterations.
class GraphReplay {
 public:
  template <class Function>
  GraphReplay(aclrtStream stream, int iterations, Function &&execute)
      : stream_(stream) {
    namespace acl = flagdnn::validation::ascend;
    if (stream == nullptr || iterations <= 0)
      throw std::invalid_argument("invalid Ascend graph capture configuration");
    acl::check_acl(aclmdlRICaptureBegin(stream_, ACL_MODEL_RI_CAPTURE_MODE_GLOBAL),
                   "aclmdlRICaptureBegin");
    try {
      for (int i = 0; i < iterations; ++i) execute();
    } catch (...) {
      aclmdlRI abandoned = nullptr;
      (void)aclmdlRICaptureEnd(stream_, &abandoned);
      if (abandoned != nullptr) (void)aclmdlRIDestroy(abandoned);
      throw;
    }
    const auto status = aclmdlRICaptureEnd(stream_, &model_);
    if (status != ACL_SUCCESS) {
      if (model_ != nullptr) (void)aclmdlRIDestroy(model_);
      model_ = nullptr;
      acl::check_acl(status, "aclmdlRICaptureEnd");
    }
    if (model_ == nullptr)
      throw std::runtime_error("Ascend capture returned no graph");
  }

  ~GraphReplay() {
    if (model_ != nullptr) {
      // Drain before destroying even when a caller exits through validation.
      (void)aclrtSynchronizeStream(stream_);
      (void)aclmdlRIDestroy(model_);
    }
  }
  GraphReplay(const GraphReplay &) = delete;
  GraphReplay &operator=(const GraphReplay &) = delete;

  void replay() {
    flagdnn::validation::ascend::check_acl(
        aclmdlRIExecuteAsync(model_, stream_), "aclmdlRIExecuteAsync");
  }

 private:
  aclrtStream stream_ = nullptr;
  aclmdlRI model_ = nullptr;
};

}  // namespace flagdnn::testing::ascend
