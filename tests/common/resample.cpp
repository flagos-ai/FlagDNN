/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "common/resample.hpp"

#include <flagdnn_frontend.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
namespace flagdnn::testing {
namespace {
namespace fe = flagdnn_frontend;
fe::DataType_t frontend_type(flagdnnDataType_t type) {
  if (type == FLAGDNN_DATA_FLOAT32) return fe::DataType_t::FLOAT;
  if (type == FLAGDNN_DATA_FLOAT16) return fe::DataType_t::HALF;
  if (type == FLAGDNN_DATA_BFLOAT16) return fe::DataType_t::BFLOAT16;
  throw std::invalid_argument("resample requires floating input");
}
void check(fe::error_t status) {
  if (status.is_bad()) throw std::runtime_error(status.get_message());
}
class Executable final : public TestExecutable {
 public:
  Executable(flagdnn::Handle& handle, const ResampleTestCase& test_case)
      : handle_(handle) {
    std::vector<fe::graph::Graph::Tensor> inputs;
    for (const auto& input : test_case.inputs)
      inputs.push_back(
          graph_.tensor(fe::graph::Tensor_attributes()
                            .set_uid(input.uid)
                            .set_data_type(frontend_type(input.data_type))
                            .set_dim(input.dimensions)
                            .set_stride(input.strides)));
    auto attributes =
        fe::graph::Resample_attributes()
            .set_name(test_case.name)
            .set_resampling_mode(
                static_cast<fe::ResampleMode_t>(test_case.mode))
            .set_padding_mode(static_cast<fe::PaddingMode_t>(test_case.padding))
            .set_generate_index(test_case.outputs.size() == 2)
            .set_align_corners(test_case.align_corners);
    if (test_case.mode == 3 || test_case.mode == 4) {
      const auto& shape = test_case.outputs[0].dimensions;
      attributes.set_output_dim(
          std::vector<std::int64_t>(shape.begin() + 2, shape.end()));
    } else
      attributes.set_window(test_case.window)
          .set_stride(test_case.stride)
          .set_pre_padding(test_case.pre)
          .set_post_padding(test_case.post);
    const auto outputs = graph_.resample(inputs[0], attributes);
    for (std::size_t index = 0; index < test_case.outputs.size(); ++index) {
      const auto& out = test_case.outputs[index];
      outputs[index]
          ->set_uid(out.uid)
          .set_data_type(out.data_type == FLAGDNN_DATA_INT32
                             ? fe::DataType_t::INT32
                             : frontend_type(out.data_type))
          .set_dim(out.dimensions)
          .set_stride(out.strides)
          .set_output(true);
    }
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
std::vector<ResampleTestCase> make_resample_cases() {
  const std::vector<std::vector<std::int64_t>> shapes = {
      {1, 8, 4, 8},    {2, 8, 7, 9},    {1, 16, 17, 19}, {2, 16, 31, 33},
      {3, 8, 5, 7},    {1, 32, 17, 19}, {2, 32, 31, 33}, {1, 8, 65, 17},
      {2, 64, 7, 9},   {4, 16, 13, 15}, {2, 32, 27, 29}, {1, 64, 17, 19},
      {1, 16, 33, 34}, {2, 8, 13, 9}};
  std::vector<ResampleTestCase> result;
  const auto make_strides = [](const std::vector<std::int64_t>& shape,
                               bool channels_last) {
    std::vector<std::int64_t> strides(shape.size());
    std::vector<std::size_t> order(shape.size());
    std::iota(order.begin(), order.end(), 0);
    if (channels_last) {
      order.erase(order.begin() + 1);
      order.push_back(1);
    }
    std::int64_t stride = 1;
    for (auto axis = order.rbegin(); axis != order.rend(); ++axis) {
      strides[*axis] = stride;
      stride *= shape[*axis];
    }
    return strides;
  };
  for (std::size_t index = 0; index < shapes.size(); ++index) {
    for (auto type :
         {FLAGDNN_DATA_FLOAT32, FLAGDNN_DATA_FLOAT16, FLAGDNN_DATA_BFLOAT16}) {
      for (int mode : {1, 2, 3, 5}) {
        // cuDNN bilinear resampling supports FP32 and exact 2x upsampling.
        if (mode == 3 && type != FLAGDNN_DATA_FLOAT32) continue;
        ResampleTestCase test_case;
        auto shape = shapes[index];
        if (mode == 3 && shape.size() != 4)
          shape = {1 + static_cast<std::int64_t>(index % 2),
                   2 + static_cast<std::int64_t>(index % 3),
                   3 + static_cast<std::int64_t>(index * 2),
                   7 + static_cast<std::int64_t>(index)};
        const auto spatial = shape.size() - 2;
        auto output_shape = shape;
        test_case.mode = mode;
        test_case.padding = mode == 3 || mode == 4 ? 1 : mode == 5 ? 2 : 3;
        test_case.align_corners = false;
        if (mode == 3 || mode == 4) {
          for (std::size_t axis = 0; axis < spatial; ++axis)
            output_shape[axis + 2] = shape[axis + 2] * 2;
        } else {
          for (std::size_t axis = 0; axis < spatial; ++axis) {
            test_case.window.push_back(2 + (index + axis) % 2);
            test_case.stride.push_back(1 + index % 2);
            test_case.pre.push_back(index % 2);
            // cuDNN average pooling uses symmetric explicit padding.
            test_case.post.push_back(
                mode <= 2 ? test_case.pre.back()
                          : static_cast<std::int64_t>(index % 3));
            output_shape[axis + 2] =
                (shape[axis + 2] + test_case.pre.back() +
                 test_case.post.back() - test_case.window.back()) /
                    test_case.stride.back() +
                1;
          }
        }
        test_case.name = std::string("resample_") +
                         (mode == 1   ? "avg_exclude"
                          : mode == 2 ? "avg_include"
                          : mode == 3 ? "bilinear"
                          : mode == 4 ? "nearest"
                                      : "maxpool") +
                         "_" +
                         (type == FLAGDNN_DATA_FLOAT32   ? "fp32"
                          : type == FLAGDNN_DATA_FLOAT16 ? "fp16"
                                                         : "bf16");
        for (auto dim : shape) test_case.name += "_" + std::to_string(dim);
        test_case.inputs = {{1, type, shape, make_strides(shape, mode == 3)}};
        test_case.outputs = {
            {2, type, output_shape, make_strides(output_shape, mode == 3)}};
        result.push_back(std::move(test_case));
      }
    }
  }
  return result;
}
std::vector<std::vector<float>> resample_inputs(
    const ResampleTestCase& test_case) {
  const auto& shape = test_case.inputs[0].dimensions;
  const auto count = std::accumulate(shape.begin(), shape.end(), std::size_t{1},
                                     std::multiplies<>());
  std::vector<float> values(count);
  for (std::size_t index = 0; index < count; ++index)
    values[index] =
        static_cast<float>(static_cast<int>((index * 17) % 61) - 45) / 19.0F;
  return {values};
}
std::unique_ptr<TestExecutable> build_flagdnn_resample(
    flagdnn::Handle& handle, const ResampleTestCase& test_case) {
  return std::make_unique<Executable>(handle, test_case);
}
}  // namespace flagdnn::testing
