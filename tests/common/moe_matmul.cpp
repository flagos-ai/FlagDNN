/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "common/moe_matmul.hpp"

#include <flagdnn_frontend.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <stdexcept>
namespace flagdnn::testing {
namespace {
namespace fe = flagdnn_frontend;
fe::DataType_t frontend_type(flagdnnDataType_t type) {
  if (type == FLAGDNN_DATA_INT32) return fe::DataType_t::INT32;
  if (type == FLAGDNN_DATA_FLOAT32) return fe::DataType_t::FLOAT;
  if (type == FLAGDNN_DATA_FLOAT16) return fe::DataType_t::HALF;
  if (type == FLAGDNN_DATA_BFLOAT16) return fe::DataType_t::BFLOAT16;
  if (type == FLAGDNN_DATA_FP8_E4M3) return fe::DataType_t::FP8_E4M3;
  if (type == FLAGDNN_DATA_FP8_E5M2) return fe::DataType_t::FP8_E5M2;
  if (type == FLAGDNN_DATA_FP8_E8M0) return fe::DataType_t::FP8_E8M0;
  throw std::invalid_argument("invalid FP8 matmul test tensor type");
}
void check(fe::error_t status) {
  if (status.is_bad()) throw std::runtime_error(status.get_message());
}
class Executable final : public TestExecutable {
 public:
  Executable(flagdnn::Handle& handle, const MoeMatmulTestCase& test_case)
      : handle_(handle) {
    std::vector<fe::graph::Graph::Tensor> inputs;
    for (const auto& input : test_case.inputs)
      inputs.push_back(
          graph_.tensor(fe::graph::Tensor_attributes()
                            .set_uid(input.uid)
                            .set_data_type(frontend_type(input.data_type))
                            .set_dim(input.dimensions)
                            .set_stride(input.strides)));
    fe::graph::Graph::Tensor output;
    if (test_case.backward)
      output = graph_.moe_grouped_matmul_bwd(
          inputs[0], inputs[1], inputs[2],
          fe::graph::Moe_grouped_matmul_bwd_attributes().set_name(
              test_case.name));
    else
      output = graph_.moe_grouped_matmul(
          inputs[0], inputs[1], inputs[2], test_case.mode ? inputs[3] : nullptr,
          test_case.mode == 2 ? inputs[4] : nullptr,
          fe::graph::Moe_grouped_matmul_attributes()
              .set_name(test_case.name)
              .set_mode(static_cast<fe::MoeGroupedMatmulMode_t>(test_case.mode))
              .set_top_k(test_case.top_k));
    const auto& out = test_case.outputs[0];
    output->set_uid(out.uid)
        .set_data_type(frontend_type(out.data_type))
        .set_dim(out.dimensions)
        .set_stride(out.strides)
        .set_output(true);
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
using Shape = std::vector<std::int64_t>;
std::int64_t elements(const Shape& shape) {
  return std::accumulate(shape.begin(), shape.end(), std::int64_t{1},
                         std::multiplies<>());
}
TestTensor tensor(std::int64_t uid, flagdnnDataType_t type, Shape shape) {
  Shape strides(shape.size());
  std::int64_t stride = 1;
  for (std::size_t axis = shape.size(); axis > 0; --axis) {
    strides[axis - 1] = stride;
    stride *= shape[axis - 1];
  }
  return {uid, type, std::move(shape), std::move(strides), 16};
}
}  // namespace
std::vector<MoeMatmulTestCase> make_moe_matmul_cases(bool backward) {
  // S, E, K, N. Keep large routed groups for NONE/GATHER workloads.
  const std::vector<std::array<std::int64_t, 4>> shapes = {
      {1, 1, 16, 16},   {7, 3, 32, 32},   {16, 2, 32, 64},   {17, 8, 64, 32},
      {31, 5, 64, 64},  {32, 3, 128, 64}, {33, 17, 64, 128}, {4, 8, 128, 128},
      {64, 4, 96, 32},  {5, 3, 256, 64},  {17, 7, 128, 64},  {65, 8, 256, 128},
      {128, 4, 64, 64}, {256, 4, 64, 64}};
  // cuDNN 9.24 SCATTER leaves rows unwritten above eight routed tokens
  // per expert. These replacements keep 14 shapes, top-k 1/2/3, empty
  // experts and arbitrary token order while using executable configurations.
  const std::vector<std::array<std::int64_t, 4>> scatter_shapes = {
      {1, 1, 16, 16},    {7, 3, 32, 32},   {8, 4, 32, 64},    {17, 8, 64, 32},
      {31, 16, 64, 64},  {16, 8, 128, 64}, {33, 17, 64, 128}, {4, 8, 128, 128},
      {32, 16, 96, 32},  {5, 3, 256, 64},  {17, 8, 128, 64},  {21, 8, 256, 128},
      {128, 32, 64, 64}, {256, 64, 64, 64}};
  std::vector<MoeMatmulTestCase> result;
  for (std::size_t index = 0; index < shapes.size(); ++index) {
    for (int format = 0; format < 4; ++format) {
      const auto dtype = format == 0   ? FLAGDNN_DATA_FLOAT16
                         : format == 1 ? FLAGDNN_DATA_BFLOAT16
                         : format == 2 ? FLAGDNN_DATA_FP8_E4M3
                                       : FLAGDNN_DATA_FP8_E5M2;
      for (int mode = 0; mode < (backward ? 1 : 3); ++mode) {
        const auto& shape = mode == 2 ? scatter_shapes[index] : shapes[index];
        const auto [s, e, k, n] = shape;
        const auto top_k =
            std::min(static_cast<int>(index % 3) + 1, static_cast<int>(e));
        const auto routed = s * top_k;
        MoeMatmulTestCase test_case;
        test_case.backward = backward;
        test_case.mode = mode;
        test_case.top_k = top_k;
        test_case.name = std::string(backward ? "moe_grouped_matmul_bwd"
                                              : "moe_grouped_matmul") +
                         (format == 0   ? "_fp16"
                          : format == 1 ? "_bf16"
                          : format == 2 ? "_e4m3"
                                        : "_e5m2") +
                         "_mode" + std::to_string(mode);
        for (auto dim : shape) test_case.name += "_" + std::to_string(dim);
        for (std::int64_t expert = 0; expert < e; ++expert) {
          test_case.offsets.push_back(
              static_cast<std::int32_t>(test_case.token_index.size()));
          for (std::int64_t source = s; source > 0; --source) {
            for (int rank = top_k; rank > 0; --rank) {
              const auto selected =
                  index % 4 == 0 && index < 12 && (mode != 2 || routed <= 8)
                      ? e - 1
                      : ((source - 1) * 7 + (rank - 1) * 3) % e;
              if (selected == expert) {
                test_case.token_index.push_back(
                    static_cast<std::int32_t>(source - 1));
                test_case.token_ks.push_back(rank - 1);
              }
            }
          }
        }
        const auto offset = tensor(3, FLAGDNN_DATA_INT32, {e, 1, 1});
        if (backward)
          test_case.inputs = {tensor(1, dtype, {1, routed, n}),
                              tensor(2, dtype, {1, routed, k}), offset};
        else {
          test_case.inputs = {tensor(1, dtype, {1, mode == 1 ? s : routed, k}),
                              tensor(2, dtype, {e, k, n}), offset};
          if (mode)
            test_case.inputs.push_back(
                tensor(4, FLAGDNN_DATA_INT32, {1, routed, 1}));
          if (mode == 2)
            test_case.inputs.push_back(
                tensor(5, FLAGDNN_DATA_INT32, {1, routed, 1}));
        }
        const auto out_type = format < 2 ? dtype : FLAGDNN_DATA_FLOAT32;
        test_case.outputs = {tensor(
            6, out_type, backward ? Shape{e, k, n} : Shape{1, routed, n})};
        result.push_back(std::move(test_case));
      }
    }
  }
  return result;
}
std::vector<std::vector<float>> moe_matmul_inputs(
    const MoeMatmulTestCase& test_case) {
  std::vector<std::vector<float>> result;
  for (std::size_t input = 0; input < test_case.inputs.size(); ++input) {
    std::vector<float> values(elements(test_case.inputs[input].dimensions));
    if (input >= 2) {
      const auto& source = input == 2   ? test_case.offsets
                           : input == 3 ? test_case.token_index
                                        : test_case.token_ks;
      values.assign(source.begin(), source.end());
    } else {
      for (std::size_t index = 0; index < values.size(); ++index)
        values[index] =
            static_cast<float>(static_cast<int>((index * 17 + input * 7) % 61) -
                               30) /
            19.0F;
    }
    result.push_back(std::move(values));
  }
  return result;
}
std::unique_ptr<TestExecutable> build_flagdnn_moe_matmul(
    flagdnn::Handle& handle, const MoeMatmulTestCase& test_case) {
  return std::make_unique<Executable>(handle, test_case);
}
}  // namespace flagdnn::testing
