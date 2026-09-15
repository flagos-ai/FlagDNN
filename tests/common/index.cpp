/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "common/index.hpp"

#include <flagdnn_frontend.h>

#include <array>
#include <stdexcept>
#include <utility>

namespace flagdnn::testing {
namespace {
namespace fe = ::flagdnn_frontend;
fe::DataType_t frontend_type(flagdnnDataType_t type) {
  switch (type) {
    case FLAGDNN_DATA_FLOAT32:
      return fe::DataType_t::FLOAT;
    case FLAGDNN_DATA_FLOAT16:
      return fe::DataType_t::HALF;
    case FLAGDNN_DATA_BFLOAT16:
      return fe::DataType_t::BFLOAT16;
    case FLAGDNN_DATA_INT32:
      return fe::DataType_t::INT32;
    case FLAGDNN_DATA_BOOLEAN:
      return fe::DataType_t::BOOLEAN;
    case FLAGDNN_DATA_FP8_E4M3:
      return fe::DataType_t::FP8_E4M3;
    case FLAGDNN_DATA_FP8_E5M2:
      return fe::DataType_t::FP8_E5M2;
    case FLAGDNN_DATA_FP8_E8M0:
      return fe::DataType_t::FP8_E8M0;
  }
  throw std::invalid_argument("unsupported index case type");
}
std::string type_name(flagdnnDataType_t type) {
  switch (type) {
    case FLAGDNN_DATA_FLOAT32:
      return "fp32";
    case FLAGDNN_DATA_FLOAT16:
      return "fp16";
    case FLAGDNN_DATA_BFLOAT16:
      return "bf16";
    case FLAGDNN_DATA_INT32:
      return "int32";
    case FLAGDNN_DATA_BOOLEAN:
      return "bool";
    case FLAGDNN_DATA_FP8_E4M3:
      return "fp8_e4m3";
    case FLAGDNN_DATA_FP8_E5M2:
      return "fp8_e5m2";
    case FLAGDNN_DATA_FP8_E8M0:
      return "fp8_e8m0";
  }
  throw std::invalid_argument("unsupported index case type");
}
TestTensor tensor(std::int64_t uid, flagdnnDataType_t type,
                  const std::vector<std::int64_t>& shape, bool padded) {
  std::vector<std::int64_t> strides(shape.size());
  std::int64_t span = padded ? 2 : 1;
  for (std::size_t axis = shape.size(); axis != 0; --axis) {
    strides[axis - 1] = span;
    span *= shape[axis - 1];
    if (padded) span += 3;
  }
  return {uid, type, shape, strides};
}
void check(fe::error_t status) {
  if (status.is_bad()) throw std::runtime_error(status.get_message());
}
class Executable final : public TestExecutable {
 public:
  Executable(flagdnn::Handle& handle, const IndexTestCase& test_case)
      : handle_(handle) {
    graph_.set_name(test_case.name)
        .set_compute_data_type(fe::DataType_t::FLOAT);
    std::vector<fe::graph::Graph::Tensor> inputs;
    for (const auto& input : test_case.inputs) {
      inputs.push_back(
          graph_.tensor(fe::graph::Tensor_attributes()
                            .set_uid(input.uid)
                            .set_data_type(frontend_type(input.data_type))
                            .set_dim(input.dimensions)
                            .set_stride(input.strides)));
    }
    fe::graph::Graph::Tensor output;
    if (test_case.operation == "concatenate") {
      output = graph_.concatenate(
          inputs, fe::graph::Concatenate_attributes().set_axis(test_case.axis));
    } else if (test_case.operation == "gen_index") {
      output = graph_.gen_index(
          fe::graph::Gen_index_attributes()
              .set_dim(test_case.output.dimensions)
              .set_axis(test_case.axis)
              .set_data_type(frontend_type(test_case.output.data_type)));
    } else {
      throw std::invalid_argument("unknown index operation");
    }
    output->set_uid(test_case.output.uid)
        .set_data_type(frontend_type(test_case.output.data_type))
        .set_dim(test_case.output.dimensions)
        .set_stride(test_case.output.strides)
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
}  // namespace

std::vector<IndexTestCase> make_index_cases(std::string_view operation) {
  if (operation != "gen_index" && operation != "concatenate")
    throw std::invalid_argument("unknown index case operation");
  std::vector<std::vector<std::int64_t>> shapes = {{1},
                                                   {17},
                                                   {33},
                                                   {2, 3},
                                                   {3, 7},
                                                   {2, 3, 5},
                                                   {1, 3, 17},
                                                   {2, 4, 33},
                                                   {2, 3, 4, 5},
                                                   {1, 7, 3, 9},
                                                   {2, 2, 2, 2, 3},
                                                   {1, 2, 1, 3, 1, 2, 1, 5},
                                                   {3, 257},
                                                   {2, 1025}};
  if (operation == "gen_index")
    shapes = {{3},   {7},    {17},   {33},   {65},   {127},   {257},
              {513}, {1025}, {2049}, {4096}, {8192}, {16384}, {65536}};
  std::vector<flagdnnDataType_t> types = {FLAGDNN_DATA_INT32,
                                          FLAGDNN_DATA_FLOAT32};
  if (operation == "concatenate")
    types.insert(types.end(), {FLAGDNN_DATA_FLOAT16, FLAGDNN_DATA_BFLOAT16,
                               FLAGDNN_DATA_BOOLEAN, FLAGDNN_DATA_FP8_E4M3,
                               FLAGDNN_DATA_FP8_E5M2, FLAGDNN_DATA_FP8_E8M0});
  std::vector<IndexTestCase> result;
  for (std::size_t index = 0; index < shapes.size(); ++index) {
    for (const auto type : types) {
      IndexTestCase test_case;
      test_case.operation = operation;
      const auto axis = index % shapes[index].size();
      test_case.axis = index % 2
                           ? static_cast<std::int64_t>(axis) -
                                 static_cast<std::int64_t>(shapes[index].size())
                           : static_cast<std::int64_t>(axis);
      test_case.name = std::string(operation) + "_" + type_name(type);
      for (auto dim : shapes[index])
        test_case.name += "_" + std::to_string(dim);
      test_case.name += "_axis" + std::to_string(test_case.axis);
      auto output_shape = shapes[index];
      if (operation == "concatenate") {
        output_shape[axis] = 0;
        for (std::size_t input = 0; input < 1 + index % 4; ++input) {
          auto shape = shapes[index];
          shape[axis] += static_cast<std::int64_t>(input);
          output_shape[axis] += shape[axis];
          test_case.inputs.push_back(
              tensor(100 + input, type, shape, (index + input) % 2 != 0));
        }
      }
      test_case.output = tensor(200, type, output_shape,
                                operation == "concatenate" && index % 3 != 0);
      result.push_back(std::move(test_case));
    }
  }
  return result;
}

std::unique_ptr<TestExecutable> build_flagdnn_index(
    flagdnn::Handle& handle, const IndexTestCase& test_case) {
  return std::make_unique<Executable>(handle, test_case);
}
}  // namespace flagdnn::testing
