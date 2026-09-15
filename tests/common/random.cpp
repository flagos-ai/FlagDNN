/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "common/random.hpp"

#include <flagdnn_frontend.h>

#include <array>
#include <cmath>
#include <numeric>
#include <stdexcept>
namespace flagdnn::testing {
namespace {
namespace fe = flagdnn_frontend;
fe::DataType_t frontend_type(flagdnnDataType_t type) {
  if (type == FLAGDNN_DATA_FLOAT32) return fe::DataType_t::FLOAT;
  if (type == FLAGDNN_DATA_FLOAT16) return fe::DataType_t::HALF;
  if (type == FLAGDNN_DATA_BFLOAT16) return fe::DataType_t::BFLOAT16;
  throw std::invalid_argument("RNG requires floating output");
}
void check(fe::error_t status) {
  if (status.is_bad()) throw std::runtime_error(status.get_message());
}
class Executable final : public TestExecutable {
 public:
  Executable(flagdnn::Handle& handle, const RngTestCase& test_case)
      : handle_(handle) {
    const auto& out = test_case.outputs[0];
    auto output =
        graph_.rng(fe::graph::Rng_attributes()
                       .set_name(test_case.name)
                       .set_dim(out.dimensions)
                       .set_stride(out.strides)
                       .set_data_type(frontend_type(out.data_type))
                       .set_seed(test_case.seed)
                       .set_offset(test_case.offset)
                       .set_distribution(static_cast<fe::RngDistribution_t>(
                           test_case.distribution))
                       .set_bernoulli_probability(test_case.probability));
    output->set_uid(out.uid).set_output(true);
    check(graph_.build(handle_, {fe::HeurMode_t::A}));
    std::int64_t size = 0;
    check(graph_.get_workspace_size(size));
    workspace_size_ = static_cast<std::size_t>(size);
  }
  std::size_t workspace_size() const noexcept override {
    return workspace_size_;
  }
  void execute(std::span<const flagdnnBinding_t> bindings, void* workspace,
               std::size_t size, flagdnnStream_t stream) override {
    check(graph_.execute(handle_, bindings, workspace, size, stream));
  }

 private:
  flagdnn::Handle& handle_;
  fe::graph::Graph graph_;
  std::size_t workspace_size_ = 0;
};
}  // namespace
std::vector<RngTestCase> make_rng_cases() {
  const std::vector<std::vector<std::int64_t>> shapes = {
      {1},
      {7},
      {17},
      {65},
      {127},
      {257},
      {513},
      {1025},
      {4096},
      {65536},
      {3, 5, 7},
      {2, 3, 17},
      {2, 4, 31, 33},
      {1, 2, 3, 2, 3, 2, 3, 2}};
  std::vector<RngTestCase> result;
  for (std::size_t index = 0; index < shapes.size(); ++index) {
    for (auto type :
         {FLAGDNN_DATA_FLOAT32, FLAGDNN_DATA_FLOAT16, FLAGDNN_DATA_BFLOAT16}) {
      for (int distribution : {1, 2, 3}) {
        RngTestCase test_case;
        test_case.name = std::string("rng_") +
                         (distribution == 1   ? "uniform"
                          : distribution == 2 ? "normal"
                                              : "bernoulli") +
                         "_" +
                         (type == FLAGDNN_DATA_FLOAT32   ? "fp32"
                          : type == FLAGDNN_DATA_FLOAT16 ? "fp16"
                                                         : "bf16");
        const auto& shape = shapes[index];
        for (auto dim : shape) test_case.name += "_" + std::to_string(dim);
        auto strides = shape;
        std::int64_t stride = index % 2 + 1;
        for (std::size_t axis = shape.size(); axis > 0; --axis) {
          strides[axis - 1] = stride;
          stride *= shape[axis - 1];
        }
        test_case.outputs = {{1, type, shape, strides}};
        test_case.seed =
            index % 3 == 0 ? -1
            : index % 3 == 1
                ? 42
                : 0x1234567800000000LL + static_cast<std::int64_t>(index);
        test_case.offset = index % 3 == 2
                               ? 4294967289LL
                               : static_cast<std::int64_t>(index * 17);
        test_case.distribution = distribution;
        test_case.probability = index == 0   ? 0.0
                                : index == 1 ? 1.0
                                : index % 2  ? 0.25
                                             : 0.75;
        result.push_back(std::move(test_case));
      }
    }
  }
  return result;
}
std::unique_ptr<TestExecutable> build_flagdnn_rng(
    flagdnn::Handle& handle, const RngTestCase& test_case) {
  return std::make_unique<Executable>(handle, test_case);
}
}  // namespace flagdnn::testing
