// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#include "extended_reference.hpp"
#include "corex_cudnn_raii.hpp"
#include "corex_cudnn_status.hpp"
#include "functional/runner_support.hpp"
#include "reference_tensor.hpp"
#include <algorithm>
#include <cuda_runtime_api.h>
#include <dlfcn.h>
namespace flagdnn::iluvatar::validation {
namespace {
using testing::TestExecutable;
using testing::TestTensor;
void *binding(std::span<const flagdnnBinding_t> bindings, std::int64_t uid) {
  for (const auto &entry : bindings)
    if (entry.uid == uid && entry.device_pointer)
      return entry.device_pointer;
  throw std::invalid_argument("missing extended reference binding");
}
class BatchnormBackward final : public TestExecutable {
  testing::ExtendedNormalizationTestCase c_;
  CorexCudnnHandle handle_;
  CorexCudnnTensorDescriptor x_, dy_, dx_, parameter_;

public:
  explicit BatchnormBackward(const testing::ExtendedNormalizationTestCase &c)
      : c_(c), x_(make_reference_tensor(c.inputs[1])),
        dy_(make_reference_tensor(c.inputs[0])),
        dx_(make_reference_tensor(c.outputs[0])),
        parameter_(make_reference_tensor(c.inputs[2])) {}
  std::size_t workspace_size() const noexcept override { return 0; }
  void execute(std::span<const flagdnnBinding_t> b, void *, std::size_t,
               flagdnnStream_t stream) override {
    handle_.bind_stream(stream);
    const float one = 1, zero = 0;
    check_cudnn(cudnnBatchNormalizationBackward(
                    handle_.get(), CUDNN_BATCHNORM_SPATIAL, &one, &zero, &one,
                    &zero, x_.get(), binding(b, c_.inputs[1].uid), dy_.get(),
                    binding(b, c_.inputs[0].uid), dx_.get(),
                    binding(b, c_.outputs[0].uid), parameter_.get(),
                    binding(b, c_.inputs[2].uid), binding(b, c_.outputs[1].uid),
                    binding(b, c_.outputs[2].uid), c_.epsilon,
                    binding(b, c_.inputs[3].uid), binding(b, c_.inputs[4].uid)),
                "cudnnBatchNormalizationBackward");
  }
};
class Pooling final : public TestExecutable {
  testing::ResampleTestCase c_;
  CorexCudnnHandle handle_;
  CorexCudnnTensorDescriptor x_, y_;
  cudnnPoolingDescriptor_t pooling_ = nullptr;

public:
  explicit Pooling(const testing::ResampleTestCase &c)
      : c_(c), x_(make_reference_tensor(c.inputs[0])),
        y_(make_reference_tensor(c.outputs[0])) {
    check_cudnn(cudnnCreatePoolingDescriptor(&pooling_),
                "cudnnCreatePoolingDescriptor");
    const auto mode = c.mode == 1 ? CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING
                      : c.mode == 2
                          ? CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING
                          : CUDNN_POOLING_MAX;
    std::vector<int> window(c.window.begin(), c.window.end()),
        padding(c.pre.begin(), c.pre.end()),
        stride(c.stride.begin(), c.stride.end());
    const auto status = cudnnSetPoolingNdDescriptor(
        pooling_, mode, CUDNN_PROPAGATE_NAN, static_cast<int>(window.size()),
        window.data(), padding.data(), stride.data());
    if (status != CUDNN_STATUS_SUCCESS) {
      cudnnDestroyPoolingDescriptor(pooling_);
      pooling_ = nullptr;
      check_cudnn(status, "cudnnSetPoolingNdDescriptor");
    }
  }
  ~Pooling() override {
    if (pooling_)
      (void)cudnnDestroyPoolingDescriptor(pooling_);
  }
  std::size_t workspace_size() const noexcept override { return 0; }
  void execute(std::span<const flagdnnBinding_t> b, void *, std::size_t,
               flagdnnStream_t stream) override {
    handle_.bind_stream(stream);
    const float one = 1, zero = 0;
    check_cudnn(cudnnPoolingForward(handle_.get(), pooling_, &one, x_.get(),
                                    binding(b, c_.inputs[0].uid), &zero,
                                    y_.get(), binding(b, c_.outputs[0].uid)),
                "cudnnPoolingForward");
  }
};
class Bilinear final : public TestExecutable {
  testing::ResampleTestCase c_;
  CorexCudnnHandle handle_;
  CorexCudnnTensorDescriptor x_, y_;
  cudnnSpatialTransformerDescriptor_t sampler_ = nullptr;
  std::unique_ptr<functional::DeviceBuffer> grid_;

public:
  explicit Bilinear(const testing::ResampleTestCase &c)
      : c_(c), x_(make_reference_tensor(c.inputs[0])),
        y_(make_reference_tensor(c.outputs[0])) {
    const auto &input = c.inputs[0].dimensions,
               &output = c.outputs[0].dimensions;
    const auto h = output[2], w = output[3];
    std::vector<float> grid(output[0] * h * w * 2);
    const auto coordinate = [&](std::int64_t index, std::int64_t in,
                                std::int64_t out) {
      if (in == 1)
        return 0.F;
      const double position =
          c.align_corners ? (out > 1 ? double(index) * (in - 1) / (out - 1) : 0)
                          : (index + 0.5) * in / out - 0.5;
      return static_cast<float>(
          2 * std::clamp(position, 0.0, double(in - 1)) / (in - 1) - 1);
    };
    for (std::int64_t n = 0; n < output[0]; ++n)
      for (std::int64_t row = 0; row < h; ++row)
        for (std::int64_t col = 0; col < w; ++col) {
          const auto i = ((n * h + row) * w + col) * 2;
          grid[i] = coordinate(col, input[3], w);
          grid[i + 1] = coordinate(row, input[2], h);
        }
    grid_ =
        std::make_unique<functional::DeviceBuffer>(grid.size() * sizeof(float));
    auto status =
        cudaMemcpy(grid_->at(), grid.data(), grid.size() * sizeof(float),
                   cudaMemcpyHostToDevice);
    if (status != cudaSuccess)
      throw std::runtime_error(cudaGetErrorString(status));
    check_cudnn(cudnnCreateSpatialTransformerDescriptor(&sampler_),
                "cudnnCreateSpatialTransformerDescriptor");
    std::vector<int> dims(output.begin(), output.end());
    auto dnn_status = cudnnSetSpatialTransformerNdDescriptor(
        sampler_, CUDNN_SAMPLER_BILINEAR, CUDNN_DATA_FLOAT, 4, dims.data());
    if (dnn_status != CUDNN_STATUS_SUCCESS) {
      cudnnDestroySpatialTransformerDescriptor(sampler_);
      sampler_ = nullptr;
      check_cudnn(dnn_status, "cudnnSetSpatialTransformerNdDescriptor");
    }
  }
  ~Bilinear() override {
    if (sampler_)
      (void)cudnnDestroySpatialTransformerDescriptor(sampler_);
  }
  std::size_t workspace_size() const noexcept override { return 0; }
  void execute(std::span<const flagdnnBinding_t> b, void *, std::size_t,
               flagdnnStream_t stream) override {
    handle_.bind_stream(stream);
    const float one = 1, zero = 0;
    check_cudnn(cudnnSpatialTfSamplerForward(
                    handle_.get(), sampler_, &one, x_.get(),
                    binding(b, c_.inputs[0].uid), grid_->at(), &zero, y_.get(),
                    binding(b, c_.outputs[0].uid)),
                "cudnnSpatialTfSamplerForward");
  }
};
class Concatenate final : public TestExecutable {
  struct Copy {
    std::int64_t uid;
    std::size_t input_offset, output_offset, descriptor;
  };
  testing::IndexTestCase c_;
  CorexCudnnHandle handle_;
  std::vector<CorexCudnnTensorDescriptor> inputs_, outputs_;
  std::vector<Copy> copies_;

public:
  explicit Concatenate(const testing::IndexTestCase &c) : c_(c) {
    const auto rank = static_cast<std::int64_t>(c.output.dimensions.size());
    const auto axis = c.axis < 0 ? c.axis + rank : c.axis;
    if (axis < 0 || axis >= rank)
      throw std::invalid_argument("concatenate reference axis is invalid");
    const auto bytes = flagdnn_data_type_size(c.output.data_type);
    std::size_t base = 0;
    for (const auto &input : c.inputs) {
      // CoreX transform ignores outer strides for a sliced multidimensional
      // output. Submit innermost rows with explicit physical offsets instead.
      auto xrow = input, yrow = c.output;
      xrow.dimensions = yrow.dimensions = {input.dimensions.back()};
      xrow.strides = {input.strides.back()};
      yrow.strides = {c.output.strides.back()};
      inputs_.emplace_back(make_reference_tensor(xrow));
      outputs_.emplace_back(make_reference_tensor(yrow));
      const auto rows =
          functional::element_count(input) / input.dimensions.back();
      for (std::size_t row = 0; row < rows; ++row) {
        std::size_t remaining = row, source = 0, target = base;
        for (std::int64_t dim = rank - 2; dim >= 0; --dim) {
          const auto coordinate = remaining % input.dimensions[dim];
          remaining /= input.dimensions[dim];
          source += coordinate * input.strides[dim];
          target += coordinate * c.output.strides[dim];
        }
        copies_.push_back(
            {input.uid, source * bytes, target * bytes, inputs_.size() - 1});
      }
      base += input.dimensions[axis] * c.output.strides[axis];
    }
  }
  std::size_t workspace_size() const noexcept override { return 0; }
  void execute(std::span<const flagdnnBinding_t> b, void *, std::size_t,
               flagdnnStream_t stream) override {
    handle_.bind_stream(stream);
    const float one = 1, zero = 0;
    auto *out = static_cast<std::byte *>(binding(b, c_.output.uid));
    for (const auto &copy : copies_) {
      auto *in = static_cast<std::byte *>(binding(b, copy.uid));
      check_cudnn(cudnnTransformTensor(handle_.get(), &one,
                                       inputs_[copy.descriptor].get(),
                                       in + copy.input_offset, &zero,
                                       outputs_[copy.descriptor].get(),
                                       out + copy.output_offset),
                  "cudnnTransformTensor(concatenate row)");
    }
  }
};
class BnFinalize final : public TestExecutable {
  testing::StatisticsTestCase c_;
  CorexCudnnHandle handle_;
  CorexCudnnTensorDescriptor stats_, parameters_;
  cudnnFusedOpsPlan_t plan_ = nullptr;
  cudnnFusedOpsConstParamPack_t constants_ = nullptr;
  cudnnFusedOpsVariantParamPack_t variants_ = nullptr;
  std::size_t workspace_ = 0;
  void release() noexcept {
    if (variants_)
      (void)cudnnDestroyFusedOpsVariantParamPack(variants_);
    if (constants_)
      (void)cudnnDestroyFusedOpsConstParamPack(constants_);
    if (plan_)
      (void)cudnnDestroyFusedOpsPlan(plan_);
    variants_ = nullptr;
    constants_ = nullptr;
    plan_ = nullptr;
  }

public:
  explicit BnFinalize(const testing::StatisticsTestCase &c)
      : c_(c), stats_(make_reference_tensor(c.inputs[0])),
        parameters_(make_reference_tensor(c.inputs[2])) {
    try {
      constexpr auto operation = CUDNN_FUSED_BN_FINALIZE_STATISTICS_TRAINING;
      check_cudnn(cudnnCreateFusedOpsPlan(&plan_, operation),
                  "cudnnCreateFusedOpsPlan(BN finalize)");
      check_cudnn(cudnnCreateFusedOpsConstParamPack(&constants_, operation),
                  "cudnnCreateFusedOpsConstParamPack");
      check_cudnn(cudnnCreateFusedOpsVariantParamPack(&variants_, operation),
                  "cudnnCreateFusedOpsVariantParamPack");
      for (auto label :
           {CUDNN_PARAM_YSTATS_DESC, CUDNN_PARAM_BN_SCALEBIAS_MEANVAR_DESC,
            CUDNN_PARAM_BN_EQSCALEBIAS_DESC})
        check_cudnn(cudnnSetFusedOpsConstParamPackAttribute(
                        constants_, label,
                        label == CUDNN_PARAM_YSTATS_DESC ? stats_.get()
                                                         : parameters_.get()),
                    "cudnnSetFusedOpsConstParamPackAttribute(descriptor)");
      auto mode = CUDNN_BATCHNORM_SPATIAL;
      check_cudnn(cudnnSetFusedOpsConstParamPackAttribute(
                      constants_, CUDNN_PARAM_BN_MODE, &mode),
                  "cudnnSetFusedOpsConstParamPackAttribute(mode)");
      auto pointer = CUDNN_PTR_16B_ALIGNED;
      for (auto label :
           {CUDNN_PARAM_BN_EQSCALE_PLACEHOLDER,
            CUDNN_PARAM_BN_EQBIAS_PLACEHOLDER, CUDNN_PARAM_YSUM_PLACEHOLDER,
            CUDNN_PARAM_YSQSUM_PLACEHOLDER, CUDNN_PARAM_BN_SCALE_PLACEHOLDER,
            CUDNN_PARAM_BN_BIAS_PLACEHOLDER,
            CUDNN_PARAM_BN_SAVED_MEAN_PLACEHOLDER,
            CUDNN_PARAM_BN_SAVED_INVSTD_PLACEHOLDER,
            CUDNN_PARAM_BN_RUNNING_MEAN_PLACEHOLDER,
            CUDNN_PARAM_BN_RUNNING_VAR_PLACEHOLDER})
        check_cudnn(cudnnSetFusedOpsConstParamPackAttribute(constants_, label,
                                                            &pointer),
                    "cudnnSetFusedOpsConstParamPackAttribute(pointer)");
      check_cudnn(
          cudnnMakeFusedOpsPlan(handle_.get(), plan_, constants_, &workspace_),
          "cudnnMakeFusedOpsPlan(BN finalize)");
    } catch (...) {
      release();
      throw;
    }
  }
  ~BnFinalize() override { release(); }
  std::size_t workspace_size() const noexcept override { return workspace_; }
  void execute(std::span<const flagdnnBinding_t> b, void *workspace,
               std::size_t bytes, flagdnnStream_t stream) override {
    handle_.bind_stream(stream);
    if (bytes < workspace_)
      throw std::invalid_argument("BN finalize workspace is small");
    const float one = 1, zero = 0;
    // The legacy DNN API updates running statistics in place; give it private
    // output copies so inputs stay immutable across graph replays.
    for (std::size_t i = 4; i < 6; ++i)
      check_cudnn(cudnnTransformTensor(handle_.get(), &one, parameters_.get(),
                                       binding(b, c_.inputs[i].uid), &zero,
                                       parameters_.get(),
                                       binding(b, c_.outputs[i].uid)),
                  "cudnnTransformTensor(BN previous statistics)");
    const auto set = [&](auto label, void *value) {
      check_cudnn(
          cudnnSetFusedOpsVariantParamPackAttribute(variants_, label, value),
          "cudnnSetFusedOpsVariantParamPackAttribute");
    };
    set(CUDNN_PTR_YSUM, binding(b, c_.inputs[0].uid));
    set(CUDNN_PTR_YSQSUM, binding(b, c_.inputs[1].uid));
    set(CUDNN_PTR_BN_SCALE, binding(b, c_.inputs[2].uid));
    set(CUDNN_PTR_BN_BIAS, binding(b, c_.inputs[3].uid));
    set(CUDNN_PTR_BN_EQSCALE, binding(b, c_.outputs[0].uid));
    set(CUDNN_PTR_BN_EQBIAS, binding(b, c_.outputs[1].uid));
    set(CUDNN_PTR_BN_SAVED_MEAN, binding(b, c_.outputs[2].uid));
    set(CUDNN_PTR_BN_SAVED_INVSTD, binding(b, c_.outputs[3].uid));
    set(CUDNN_PTR_BN_RUNNING_MEAN, binding(b, c_.outputs[4].uid));
    set(CUDNN_PTR_BN_RUNNING_VAR, binding(b, c_.outputs[5].uid));
    set(CUDNN_PTR_WORKSPACE, workspace);
    set(CUDNN_SCALAR_SIZE_T_WORKSPACE_SIZE_IN_BYTES, &workspace_);
    auto count = static_cast<std::int64_t>(c_.accum_count);
    set(CUDNN_SCALAR_INT64_T_BN_ACCUMULATION_COUNT, &count);
    set(CUDNN_SCALAR_DOUBLE_BN_EXP_AVG_FACTOR, &c_.momentum);
    set(CUDNN_SCALAR_DOUBLE_BN_EPSILON, &c_.epsilon);
    check_cudnn(cudnnFusedOpsExecute(handle_.get(), plan_, variants_),
                "cudnnFusedOpsExecute(BN finalize)");
  }
};

} // namespace
std::string extended_reference_skip(const testing::StatisticsTestCase &c) {
  if (c.operation == "bn_finalize") {
    // The DNN preflight executes a valid FP32 fused plan and checks all
    // outputs. CoreX 4.4.0 reports SUCCESS without writing any output.
    if (cudnnGetVersion() != 7605)
      throw std::runtime_error(
          "BN finalize DNN capability needs requalification");
    return "SEMANTIC_MISMATCH";
  }
  return missing_graph_reference_reason();
}
std::unique_ptr<testing::TestExecutable>
build_extended_reference(const testing::StatisticsTestCase &c) {
  return std::make_unique<BnFinalize>(c);
}
std::string
extended_reference_skip(const testing::ExtendedNormalizationTestCase &c) {
  return c.operation == "batchnorm_backward" ? ""
                                             : missing_graph_reference_reason();
}
std::string missing_graph_reference_reason() {
  if (cudnnGetVersion() != 7605 ||
      dlsym(RTLD_DEFAULT, "cudnnBackendCreateDescriptor"))
    throw std::runtime_error("CoreX graph ABI changed; missing reference "
                             "adapter needs qualification");
  return "DNN_API_UNAVAILABLE";
}
std::string extended_reference_skip(const testing::ResampleTestCase &c) {
  if (c.mode == 3)
    return c.inputs[0].data_type == FLAGDNN_DATA_FLOAT32 ? ""
                                                         : "DTYPE_UNSUPPORTED";
  if (c.mode != 1 && c.mode != 2 && c.mode != 5)
    return "DNN_API_UNAVAILABLE";
  if (c.outputs.size() != 1 || c.pre != c.post)
    return "ATTRIBUTE_UNSUPPORTED";
  return "";
}
std::string extended_reference_skip(const testing::IndexTestCase &c) {
  if (c.operation != "concatenate")
    return missing_graph_reference_reason();
  const auto type = c.output.data_type;
  // The audited CoreX transform returns SUCCESS but leaves BF16/INT32
  // outputs unchanged. The DNN API probe verifies this with nonzero data.
  if (type == FLAGDNN_DATA_BFLOAT16 || type == FLAGDNN_DATA_INT32 ||
      type == FLAGDNN_DATA_FP8_E4M3 || type == FLAGDNN_DATA_FP8_E5M2 ||
      type == FLAGDNN_DATA_FP8_E8M0)
    return "DTYPE_UNSUPPORTED";
  return "";
}
std::unique_ptr<testing::TestExecutable>
build_extended_reference(const testing::ExtendedNormalizationTestCase &c) {
  return std::make_unique<BatchnormBackward>(c);
}
std::unique_ptr<testing::TestExecutable>
build_extended_reference(const testing::ResampleTestCase &c) {
  if (c.mode == 3)
    return std::make_unique<Bilinear>(c);
  return std::make_unique<Pooling>(c);
}
std::unique_ptr<testing::TestExecutable>
build_extended_reference(const testing::IndexTestCase &c) {
  return std::make_unique<Concatenate>(c);
}
} // namespace flagdnn::iluvatar::validation
