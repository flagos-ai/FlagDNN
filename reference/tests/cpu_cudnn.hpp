/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#ifndef FLAGDNN_REFERENCE_TESTS_CPU_CUDNN_HPP_
#define FLAGDNN_REFERENCE_TESTS_CPU_CUDNN_HPP_

#include "common/common.hpp"
#include "common/pointwise.hpp"
#include "validation/cuda_driver.hpp"
#include "validation/tensor_io.hpp"

#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <set>
#include <stdexcept>

namespace flagdnn::reference::cpu {}
namespace flagdnn::reference::tests {
namespace t = flagdnn::testing;
namespace cpu = flagdnn::reference::cpu;
namespace gpu = flagdnn::validation::nvidia;
namespace io = gpu::tensor_io;
using Values = std::vector<float>;
using Arrays = std::vector<Values>;
using Bytes = std::vector<std::uint8_t>;
using RawArrays = std::vector<Bytes>;
using Oracle = std::function<Arrays(const Arrays&)>;
constexpr auto encoding = io::BooleanEncoding::kBitPacked;

Values quantize(std::span<const float> values, flagdnnDataType_t type);
Arrays inputs_for(const std::vector<t::TestTensor>& tensors);
void compare(std::span<const float> actual, std::span<const float> expected,
             double atol, double rtol);
t::TestTensor tensor(std::int64_t uid, std::vector<std::int64_t> shape,
                     flagdnnDataType_t type = FLAGDNN_DATA_FLOAT32);
std::vector<std::vector<std::int64_t>> shapes(
    const std::vector<t::TestTensor>& tensors);
struct Result { Arrays inputs, outputs; RawArrays input_bytes, output_bytes; };
class Suite {
 public:
  gpu::Stream stream;
  std::size_t passed = 0, failed = 0;
  void run(const std::string& name, const std::function<void()>& action);
  Result execute(const std::vector<t::TestTensor>& inputs,
                 const std::vector<t::TestTensor>& outputs,
                 std::unique_ptr<t::TestExecutable> executable,
                 Arrays values = {}, RawArrays raw_inputs = {});
  Result check(const std::vector<t::TestTensor>& inputs,
               const std::vector<t::TestTensor>& outputs,
               std::unique_ptr<t::TestExecutable> executable,
               const Oracle& oracle, double atol = 2e-5, double rtol = 2e-5,
               Arrays values = {});
};
void matrix(Suite&);
void matrix_tf32(Suite&);
void normalization(Suite&);
void normalization_helpers(Suite&);
void attention(Suite&);
void storage(Suite&);
void pointwise(Suite&);
void extended(Suite&, const std::string&);
void formulas(Suite&);

}  // namespace flagdnn::reference::tests
#endif
