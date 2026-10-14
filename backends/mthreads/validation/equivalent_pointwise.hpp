/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#ifndef FLAGDNN_MTHREADS_EQUIVALENT_POINTWISE_HPP_
#define FLAGDNN_MTHREADS_EQUIVALENT_POINTWISE_HPP_

#include <functional>

#include "backends/mthreads/validation/paired_timing.hpp"

namespace flagdnn::benchmarking {
struct BenchmarkCase;
}

namespace flagdnn::testing {
struct PointwiseTestCase;
}

namespace flagdnn::validation::mthreads::equivalent {

// CPU checks and fresh-output replay are required, outside event timing.
// A missing/mismatched NVIDIA record or target policy is an error.
void run(const benchmarking::BenchmarkCase& specification, Stream& stream,
         const std::function<void()>& enqueue,
         const std::function<void()>& validate,
         const std::function<void()>& reset_outputs);
void run(const testing::PointwiseTestCase& specification, Stream& stream,
         const std::function<void()>& enqueue,
         const std::function<void()>& validate,
         const std::function<void()>& reset_outputs);

}  // namespace flagdnn::validation::mthreads::equivalent
#endif
