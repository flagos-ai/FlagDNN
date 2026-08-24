// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef FLAGDNN_BACKENDS_ILUVATAR_VALIDATION_BENCHMARK_COREX_CUDNN_CAPABILITY_POLICY_HPP_
#define FLAGDNN_BACKENDS_ILUVATAR_VALIDATION_BENCHMARK_COREX_CUDNN_CAPABILITY_POLICY_HPP_

#include "common/benchmark_provider.hpp"
#include "corex_cudnn_reference.hpp"

#include <string_view>

namespace flagdnn::iluvatar::validation::benchmark {

[[nodiscard]] flagdnn::benchmarking::ProviderCapability
benchmark_capability_from_catalog(
    const CorexCudnnCapabilityCatalog &catalog,
    const flagdnn::benchmarking::BenchmarkCase &specification,
    std::string_view operation);

} // namespace flagdnn::iluvatar::validation::benchmark

#endif // FLAGDNN_BACKENDS_ILUVATAR_VALIDATION_BENCHMARK_COREX_CUDNN_CAPABILITY_POLICY_HPP_
