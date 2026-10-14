// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef FLAGDNN_THEAD_CPU_VALIDATION_HPP_
#define FLAGDNN_THEAD_CPU_VALIDATION_HPP_

#include "common/benchmark_provider.hpp"
#include "benchmark/equivalent_baseline.hpp"
#include "common/common.hpp"
#include "tensor_io.hpp"

namespace flagdnn::validation::thead {

// CPU validation is never timed. Index copies are checked byte for byte,
// including dtypes whose values cannot be represented losslessly as float.
std::vector<double> run_cpu_validated_case(
    const benchmarking::BenchmarkCase& specification,
    benchmarking::BenchmarkExecutable& executable, DeviceStream& stream,
    const benchmarking::thead::NativeBaselineMetadata* metadata = nullptr,
    bool benchmark = false);

class TestBenchmarkExecutable final : public benchmarking::BenchmarkExecutable {
 public:
  explicit TestBenchmarkExecutable(std::unique_ptr<testing::TestExecutable> value)
      : value_(std::move(value)) {}
  void prepare(std::span<const flagdnnBinding_t> bindings,
               flagdnnStream_t stream) override { value_->prepare(bindings, stream); }
  std::size_t workspace_size() const noexcept override { return value_->workspace_size(); }
  void execute(std::span<const flagdnnBinding_t> bindings, void* workspace,
               std::size_t bytes, flagdnnStream_t stream) override {
    value_->execute(bindings, workspace, bytes, stream);
  }
 private:
  std::unique_ptr<testing::TestExecutable> value_;
};

}  // namespace flagdnn::validation::thead
#endif
