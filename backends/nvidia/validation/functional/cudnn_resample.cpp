/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include <algorithm>
#include <memory>
#include <vector>

#include "validation/cuda_driver.hpp"
#include "validation/functional/cudnn_extended.hpp"
#include "validation/functional/cudnn_tensor.hpp"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
namespace flagdnn::testing {
namespace {
using cuda::TensorDescriptor;
class CudnnBilinearSampler final : public cuda::CudnnGraphExecutable {
 public:
  explicit CudnnBilinearSampler(const ResampleTestCase& test_case)
      : input_(test_case.inputs.at(0)),
        output_(test_case.outputs.at(0)),
        input_uid_(test_case.inputs[0].uid),
        output_uid_(test_case.outputs[0].uid) {
    const auto& input = test_case.inputs[0];
    const auto& output = test_case.outputs[0];
    if (input.dimensions.size() != 4 || output.dimensions.size() != 4 ||
        input.data_type != FLAGDNN_DATA_FLOAT32 ||
        output.data_type != FLAGDNN_DATA_FLOAT32 || test_case.mode != 3)
      throw std::invalid_argument(
          "cuDNN bilinear sampler cases require FP32 rank four");
    const auto batch = output.dimensions[0], h = output.dimensions[2],
               w = output.dimensions[3];
    std::vector<float> grid(batch * h * w * 2);
    const auto coordinate = [&](std::int64_t index, std::int64_t in,
                                std::int64_t out) {
      if (in == 1) return 0.0F;
      const double position =
          test_case.align_corners
              ? (out > 1 ? static_cast<double>(index) * (in - 1) / (out - 1)
                         : 0.0)
              : (index + 0.5) * in / out - 0.5;
      return static_cast<float>(
          2.0 * std::clamp(position, 0.0, static_cast<double>(in - 1)) /
              (in - 1) -
          1.0);
    };
    // Grid generation supplies geometry to the cuDNN sampler. All output
    // interpolation and arithmetic run inside cudnnSpatialTfSamplerForward.
    for (std::int64_t n = 0; n < batch; ++n)
      for (std::int64_t row = 0; row < h; ++row)
        for (std::int64_t col = 0; col < w; ++col) {
          const auto index = ((n * h + row) * w + col) * 2;
          grid[index] = coordinate(col, input.dimensions[3], w);
          grid[index + 1] = coordinate(row, input.dimensions[2], h);
        }
    grid_ = std::make_unique<DeviceBuffer>(grid.size() * sizeof(float));
    cuda::check_cuda_runtime(
        cudaMemcpy(grid_->opaque(), grid.data(), grid.size() * sizeof(float),
                   cudaMemcpyHostToDevice),
        "cudaMemcpy(sampling grid)");
    cuda::check_cudnn(cudnnCreateSpatialTransformerDescriptor(&transformer_),
                      "cudnnCreateSpatialTransformerDescriptor");
    std::vector<int> dims(output.dimensions.begin(), output.dimensions.end());
    const auto status = cudnnSetSpatialTransformerNdDescriptor(
        transformer_, CUDNN_SAMPLER_BILINEAR, CUDNN_DATA_FLOAT, 4, dims.data());
    if (status != CUDNN_STATUS_SUCCESS) {
      (void)cudnnDestroySpatialTransformerDescriptor(transformer_);
      transformer_ = nullptr;
      cuda::check_cudnn(status, "cudnnSetSpatialTransformerNdDescriptor");
    }
  }
  ~CudnnBilinearSampler() override {
    if (transformer_)
      (void)cudnnDestroySpatialTransformerDescriptor(transformer_);
  }
  void execute(std::span<const flagdnnBinding_t> bindings, void* workspace,
               std::size_t size, flagdnnStream_t stream) override {
    begin_execute(workspace, size, stream);
    const auto pointers = cuda::make_cudnn_binding_map(bindings);
    const float alpha = 1.0F, beta = 0.0F;
    cuda::check_cudnn(cudnnSpatialTfSamplerForward(
                          handle(), transformer_, &alpha, input_.get(),
                          pointers.at(input_uid_), grid_->opaque(), &beta,
                          output_.get(), pointers.at(output_uid_)),
                      "cudnnSpatialTfSamplerForward");
  }

 private:
  TensorDescriptor input_, output_;
  std::int64_t input_uid_, output_uid_;
  cudnnSpatialTransformerDescriptor_t transformer_ = nullptr;
  std::unique_ptr<DeviceBuffer> grid_;
};
}  // namespace
std::unique_ptr<TestExecutable> build_cudnn_bilinear_resample(
    const ResampleTestCase& test_case) {
  return std::make_unique<CudnnBilinearSampler>(test_case);
}
}  // namespace flagdnn::testing
#pragma GCC diagnostic pop
