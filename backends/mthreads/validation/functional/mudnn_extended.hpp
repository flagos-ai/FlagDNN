/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include <deque>
#include <functional>

#include "backends/mthreads/validation/functional/tensor_io_adapter.hpp"
#include "backends/mthreads/validation/musa_driver.hpp"
#include "common/causal_convolution.hpp"
#include "common/fp8_matmul.hpp"
#include "common/index.hpp"
#include "common/moe_matmul.hpp"
#include "common/normalization_extended.hpp"
#include "common/position_embedding.hpp"
#include "common/resample.hpp"
#include "common/statistics.hpp"

namespace flagdnn::testing::mthreads {
namespace md = musa::dnn;
namespace mv = validation::mthreads;
using Shape = std::vector<std::int64_t>;
using Unary = md::Unary::Mode;
using Binary = md::Binary::Mode;
using Reduce = md::Reduce::Mode;

// References use only native muDNN operations. Arithmetic intermediates default
// to FP32; copies and GEMM operands retain their explicitly requested dtype.
class NativeProgram : public TestExecutable {
 public:
  NativeProgram(std::span<const TestTensor> inputs,
                std::span<const TestTensor> outputs, bool raw_storage = false);
  std::size_t workspace_size() const noexcept override { return 0; }
  void execute(std::span<const flagdnnBinding_t>, void*, std::size_t,
               flagdnnStream_t) override;
  std::size_t temporary(const Shape&,
                        flagdnnDataType_t = FLAGDNN_DATA_FLOAT32);
  std::size_t view(std::size_t, const Shape&, const Shape&,
                   std::size_t byte_offset = 0);
  void unary_to(std::size_t out, Unary mode, std::size_t in,
                double alpha = 0.0);
  std::size_t unary(Unary mode, std::size_t in, double alpha = 0.0);
  void binary_to(std::size_t out, Binary mode, std::size_t a, std::size_t b);
  std::size_t binary(Binary mode, std::size_t a, std::size_t b);
  void reduce_to(std::size_t out, Reduce mode, std::size_t in,
                 const std::vector<int>& axes);
  std::size_t reduce(Reduce mode, std::size_t in,
                     const std::vector<int>& axes);
  std::size_t floating(std::size_t in);
  std::size_t output(std::size_t i) const { return input_count_ + i; }
  void add(std::function<void()> operation) {
    operations_.push_back(std::move(operation));
  }
  md::Tensor& tensor(std::size_t i) { return tensors_.at(i); }
  const TestTensor& descriptor(std::size_t i) const {
    return descriptors_.at(i);
  }
  md::Handle& handle() { return handle_; }
  md::MemoryMaintainer maintainer();

 private:
  std::size_t append(TestTensor);
  md::Handle handle_;
  bool raw_storage_;
  std::size_t input_count_, external_count_;
  std::vector<TestTensor> descriptors_;
  std::deque<md::Tensor> tensors_;
  std::vector<void*> addresses_;
  std::vector<std::unique_ptr<mv::DeviceBuffer>> buffers_, workspace_buffers_;
  struct View {
    std::size_t index, parent, offset;
  };
  std::vector<View> views_;
  std::vector<std::function<void()>> operations_;
  std::size_t workspace_cursor_ = 0;
};
std::unique_ptr<TestExecutable> reference(const IndexTestCase&);
std::unique_ptr<TestExecutable> reference(const StatisticsTestCase&);
std::unique_ptr<TestExecutable> reference(
    const ExtendedNormalizationTestCase&);
std::unique_ptr<TestExecutable> reference(const RoPETestCase&);
std::unique_ptr<TestExecutable> reference(const ResampleTestCase&);
std::unique_ptr<TestExecutable> reference(const CausalConvolutionTestCase&);
std::unique_ptr<TestExecutable> reference(const Fp8MatmulTestCase&);
std::unique_ptr<TestExecutable> reference(const MoeMatmulTestCase&);
}  // namespace flagdnn::testing::mthreads
