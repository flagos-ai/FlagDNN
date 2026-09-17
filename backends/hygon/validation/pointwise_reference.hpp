/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_HYGON_VALIDATION_POINTWISE_REFERENCE_HPP_
#define FLAGDNN_BACKENDS_HYGON_VALIDATION_POINTWISE_REFERENCE_HPP_

#include "hipdnn_reference.hpp"

#include <flagdnn/flagdnn.h>

#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace flagdnn::validation::hygon {

enum class HipdnnPointwiseKind {
  kAdd,
  kSub,
  kMul,
  kMin,
  kMax,
  kAddSquare,
  kSigmoid,
  kRelu,
  kTanh,
  kElu,
  kSoftplus,
  kAbs,
  kIdentity,
  kSwish,
  kSigmoidBackward,
  kReluBackward,
  kTanhBackward,
  kEluBackward,
  kSoftplusBackward,
  kSwishBackward,
  kUnavailable,
};

struct HipdnnPointwiseOperation {
  HipdnnPointwiseKind kind = HipdnnPointwiseKind::kUnavailable;
  double alpha = 1.0;
  double activation_coefficient = 0.0;
  double swish_beta = 1.0;
  std::string unavailable_reason = "no exact hipDNN pointwise primitive";
};

[[nodiscard]] HipdnnPointwiseOperation
make_hipdnn_pointwise_operation(flagdnnPointwiseMode_t mode,
                                const flagdnnPointwiseAttributes_t &attributes,
                                double alpha = 1.0);

[[nodiscard]] HipdnnCapability
hipdnn_pointwise_capability(const HipdnnPointwiseOperation &operation,
                            std::span<const ReferenceTensor> tensors,
                            bool inputs_are_finite = true);

/*
 * HIP stream capture is a separate capability from direct primitive
 * execution. Keep the qualification result explicit so benchmark code never
 * probes a known VM-faulting replay path in-process.
 */
[[nodiscard]] HipdnnCapability
hipdnn_pointwise_capture_capability(const HipdnnPointwiseOperation &operation);

[[nodiscard]] std::vector<ReferenceTensor>
hipdnn_pointwise_descriptor_tensors(const HipdnnPointwiseOperation &operation,
                                    std::span<const ReferenceTensor> tensors);

[[nodiscard]] bool
hipdnn_pointwise_uses_sequence(HipdnnPointwiseKind kind) noexcept;

class HipdnnPointwisePlan final {
public:
  HipdnnPointwisePlan(HipdnnPointwiseOperation operation,
                      std::vector<ReferenceTensor> tensors);
  ~HipdnnPointwisePlan();

  HipdnnPointwisePlan(const HipdnnPointwisePlan &) = delete;
  HipdnnPointwisePlan &operator=(const HipdnnPointwisePlan &) = delete;
  HipdnnPointwisePlan(HipdnnPointwisePlan &&) noexcept;
  HipdnnPointwisePlan &operator=(HipdnnPointwisePlan &&) noexcept;

  [[nodiscard]] std::size_t workspace_size() const noexcept;
  void execute(std::span<const flagdnnBinding_t> bindings, void *workspace,
               std::size_t workspace_size, flagdnnStream_t stream);

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace flagdnn::validation::hygon

#endif // FLAGDNN_BACKENDS_HYGON_VALIDATION_POINTWISE_REFERENCE_HPP_
