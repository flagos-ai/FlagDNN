// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef FLAGDNN_BACKENDS_THEAD_VALIDATION_ACDNN_POINTWISE_DAG_HPP_
#define FLAGDNN_BACKENDS_THEAD_VALIDATION_ACDNN_POINTWISE_DAG_HPP_
#include "pointwise_reference.hpp"
namespace flagdnn::validation::thead {
[[nodiscard]] std::vector<std::string> acdnn_pointwise_dag_plan(
    flagdnnPointwiseMode_t mode);
[[nodiscard]] std::unique_ptr<flagdnn::testing::TestExecutable>
make_acdnn_pointwise_dag(const PointwiseReferenceSpecification &specification,
                        const CapabilityRecord &capability);
}  // namespace flagdnn::validation::thead
#endif
