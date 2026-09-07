// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "acdnn_convolution_reference.hpp"

#include "acdnn_reference.hpp"
#include "backend_pointwise_reference.hpp"
#include "numeric_types.hpp"
#include "ppu_driver.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace flagdnn::validation::thead {
namespace {

using flagdnn::testing::ConvolutionDirection;
using flagdnn::testing::ConvolutionMode;
using flagdnn::testing::ConvolutionTestCase;
using flagdnn::testing::TestTensor;

acdnnDataType_t acdnn_data_type(flagdnnDataType_t data_type) {
  switch (data_type) {
    case FLAGDNN_DATA_FLOAT32:
      return ACDNN_DATA_FLOAT;
    case FLAGDNN_DATA_FLOAT16:
      return ACDNN_DATA_HALF;
    case FLAGDNN_DATA_BFLOAT16:
      return ACDNN_DATA_BF16;
    default:
      throw std::invalid_argument(
          "acDNN convolution data type is unsupported");
  }
}

class FilterDescriptor final {
 public:
  FilterDescriptor() {
    check_acdnn(acdnnCreateFilterDescriptor(&descriptor_),
                "acdnnCreateFilterDescriptor");
    if (descriptor_ == nullptr) {
      throw std::runtime_error(
          "acdnnCreateFilterDescriptor returned null");
    }
  }

  ~FilterDescriptor() {
    if (descriptor_ != nullptr) {
      (void)acdnnDestroyFilterDescriptor(descriptor_);
    }
  }

  FilterDescriptor(const FilterDescriptor &) = delete;
  FilterDescriptor &operator=(const FilterDescriptor &) = delete;

  [[nodiscard]] acdnnFilterDescriptor_t get() const noexcept {
    return descriptor_;
  }

  void set(acdnnDataType_t data_type, std::span<const int> dimensions,
           acdnnTensorFormat_t format = ACDNN_TENSOR_NCHW) {
    check_acdnn(
        acdnnSetFilterNdDescriptor(
            descriptor_, data_type, format,
            static_cast<int>(dimensions.size()), dimensions.data()),
        "acdnnSetFilterNdDescriptor");
  }

 private:
  acdnnFilterDescriptor_t descriptor_ = nullptr;
};

class ConvolutionDescriptor final {
 public:
  ConvolutionDescriptor() {
    check_acdnn(acdnnCreateConvolutionDescriptor(&descriptor_),
                "acdnnCreateConvolutionDescriptor");
    if (descriptor_ == nullptr) {
      throw std::runtime_error(
          "acdnnCreateConvolutionDescriptor returned null");
    }
  }

  ~ConvolutionDescriptor() {
    if (descriptor_ != nullptr) {
      (void)acdnnDestroyConvolutionDescriptor(descriptor_);
    }
  }

  ConvolutionDescriptor(const ConvolutionDescriptor &) = delete;
  ConvolutionDescriptor &operator=(const ConvolutionDescriptor &) = delete;

  [[nodiscard]] acdnnConvolutionDescriptor_t get() const noexcept {
    return descriptor_;
  }

  void set(std::span<const int> padding, std::span<const int> stride,
           std::span<const int> dilation, int groups,
           acdnnConvolutionMode_t mode) {
    if (padding.empty() || padding.size() != stride.size() ||
        padding.size() != dilation.size() || groups <= 0) {
      throw std::invalid_argument("invalid acDNN convolution geometry");
    }
    check_acdnn(
        acdnnSetConvolutionNdDescriptor(
            descriptor_, static_cast<int>(padding.size()), padding.data(),
            stride.data(), dilation.data(), mode, ACDNN_DATA_FLOAT),
        "acdnnSetConvolutionNdDescriptor");
    check_acdnn(acdnnSetConvolutionGroupCount(descriptor_, groups),
                "acdnnSetConvolutionGroupCount");
  }

 private:
  acdnnConvolutionDescriptor_t descriptor_ = nullptr;
};

int checked_int(std::int64_t value, const char *description) {
  if (value <= 0 || value > std::numeric_limits<int>::max()) {
    throw std::invalid_argument(std::string(description) +
                                " is outside positive int range");
  }
  return static_cast<int>(value);
}

std::vector<int> checked_positive(std::span<const std::int64_t> values,
                                  const char *description) {
  std::vector<int> result;
  result.reserve(values.size());
  for (const std::int64_t value : values) {
    result.push_back(checked_int(value, description));
  }
  return result;
}

std::vector<int> checked_nonnegative(std::span<const std::int64_t> values,
                                     const char *description) {
  std::vector<int> result;
  result.reserve(values.size());
  for (const std::int64_t value : values) {
    if (value < 0 || value > std::numeric_limits<int>::max()) {
      throw std::invalid_argument(std::string(description) +
                                  " is outside nonnegative int range");
    }
    result.push_back(static_cast<int>(value));
  }
  return result;
}

bool dense_contiguous(const TestTensor &tensor) {
  if (tensor.dimensions.size() != tensor.strides.size()) {
    return false;
  }
  std::int64_t expected = 1;
  for (std::size_t axis = tensor.dimensions.size(); axis != 0; --axis) {
    const std::int64_t dimension = tensor.dimensions[axis - 1];
    if (dimension <= 0 || tensor.strides[axis - 1] != expected ||
        expected > std::numeric_limits<std::int64_t>::max() / dimension) {
      return false;
    }
    expected *= dimension;
  }
  return true;
}

bool channels_last(const TestTensor &tensor) {
  if (tensor.dimensions.size() < 3 || tensor.dimensions.size() > 5 ||
      tensor.dimensions.size() != tensor.strides.size()) {
    return false;
  }
  std::vector<std::int64_t> expected(tensor.dimensions.size());
  expected[1] = 1;
  std::int64_t stride = tensor.dimensions[1];
  if (stride <= 0) {
    return false;
  }
  for (std::size_t axis = tensor.dimensions.size(); axis != 2; --axis) {
    const std::size_t current = axis - 1;
    const std::int64_t dimension = tensor.dimensions[current];
    if (dimension <= 0 ||
        stride > std::numeric_limits<std::int64_t>::max() / dimension) {
      return false;
    }
    expected[current] = stride;
    stride *= dimension;
  }
  expected[0] = stride;
  return tensor.strides == expected;
}

std::size_t tensor_storage_bytes(const TestTensor &tensor) {
  std::uint64_t maximum_offset = 0;
  if (tensor.dimensions.size() != tensor.strides.size()) {
    throw std::invalid_argument(
        "THead fused convolution storage requires valid tensor metadata");
  }
  for (std::size_t axis = 0; axis < tensor.dimensions.size(); ++axis) {
    const std::int64_t dimension = tensor.dimensions[axis];
    const std::int64_t stride = tensor.strides[axis];
    if (dimension <= 0 || stride <= 0) {
      throw std::invalid_argument(
          "THead fused convolution tensor geometry is invalid");
    }
    const std::uint64_t extent = static_cast<std::uint64_t>(dimension - 1);
    const std::uint64_t step = static_cast<std::uint64_t>(stride);
    if (extent != 0 &&
        step > (std::numeric_limits<std::uint64_t>::max() - maximum_offset) /
                   extent) {
      throw std::overflow_error(
          "THead fused convolution tensor storage overflows");
    }
    maximum_offset += extent * step;
  }
  const std::size_t scalar_bytes = element_size(tensor.data_type);
  if (maximum_offset >=
      std::numeric_limits<std::size_t>::max() / scalar_bytes) {
    throw std::overflow_error(
        "THead fused convolution tensor storage is too large");
  }
  return (static_cast<std::size_t>(maximum_offset) + 1U) * scalar_bytes;
}

std::size_t align_workspace(std::size_t value) {
  constexpr std::size_t kAlignment = 256;
  if (value > std::numeric_limits<std::size_t>::max() - (kAlignment - 1)) {
    throw std::overflow_error("THead convolution workspace size overflows");
  }
  return (value + kAlignment - 1) / kAlignment * kAlignment;
}

std::string expected_primitive(ConvolutionDirection direction) {
  switch (direction) {
    case ConvolutionDirection::kFprop:
      return "acdnnConvolutionForward(IMPLICIT_GEMM)";
    case ConvolutionDirection::kDgrad:
      return "acdnnConvolutionBackwardData(ALGO_0)";
    case ConvolutionDirection::kWgrad:
      return "acdnnConvolutionBackwardFilter(ALGO_0)";
  }
  throw std::invalid_argument("unknown convolution direction");
}

std::vector<std::string> asymmetric_reference_plan(
    ConvolutionDirection direction, flagdnnDataType_t data_type) {
  const bool backend_copy = data_type == FLAGDNN_DATA_BFLOAT16;
  if (direction == ConvolutionDirection::kFprop) {
    return {
        "acdnnConvolutionForward(IMPLICIT_GEMM,symmetric-superset)",
        backend_copy
            ? "acdnnBackendExecute(POINTWISE_IDENTITY_FWD,asymmetric-output-slice)"
            : "acdnnTransformTensor(alpha=1,beta=0,asymmetric-output-slice)",
    };
  }
  return {
      backend_copy
          ? "acdnnBackendExecute(POINTWISE_IDENTITY_FWD,asymmetric-gradient-zero)"
          : "acdnnTransformTensor(alpha=1,beta=0,asymmetric-gradient-zero)",
      backend_copy
          ? "acdnnBackendExecute(POINTWISE_IDENTITY_FWD,asymmetric-gradient-pad)"
          : "acdnnTransformTensor(alpha=1,beta=0,asymmetric-gradient-pad)",
      direction == ConvolutionDirection::kDgrad
          ? "acdnnConvolutionBackwardData(ALGO_0,symmetric-superset)"
          : "acdnnConvolutionBackwardFilter(ALGO_0,symmetric-superset)",
  };
}

std::vector<std::int64_t> dense_strides(
    std::span<const std::int64_t> dimensions, bool use_channels_last) {
  if (dimensions.size() < 3 || dimensions.size() > 5) {
    throw std::invalid_argument(
        "asymmetric convolution tensor rank is invalid");
  }
  std::vector<std::int64_t> result(dimensions.size());
  auto multiply = [](std::int64_t left, std::int64_t right) {
    if (left <= 0 || right <= 0 ||
        left > std::numeric_limits<std::int64_t>::max() / right) {
      throw std::overflow_error(
          "asymmetric convolution tensor strides overflow");
    }
    return left * right;
  };
  if (!use_channels_last) {
    std::int64_t stride = 1;
    for (std::size_t axis = dimensions.size(); axis != 0; --axis) {
      result[axis - 1] = stride;
      stride = multiply(stride, dimensions[axis - 1]);
    }
    return result;
  }
  result[1] = 1;
  std::int64_t stride = dimensions[1];
  if (stride <= 0) {
    throw std::invalid_argument(
        "asymmetric convolution channel extent is invalid");
  }
  for (std::size_t axis = dimensions.size(); axis != 2; --axis) {
    const std::size_t current = axis - 1;
    result[current] = stride;
    stride = multiply(stride, dimensions[current]);
  }
  result[0] = stride;
  return result;
}

struct AsymmetricGeometry {
  TestTensor temporary_y;
  std::vector<std::int64_t> symmetric_padding;
  std::size_t subview_byte_offset = 0;
};

AsymmetricGeometry asymmetric_geometry(const ConvolutionTestCase &test_case) {
  const std::size_t rank = test_case.x.dimensions.size();
  const std::size_t spatial_rank = rank >= 2 ? rank - 2 : 0;
  if (rank < 3 || rank > 5 || test_case.w.dimensions.size() != rank ||
      test_case.y.dimensions.size() != rank ||
      test_case.pre_padding.size() != spatial_rank ||
      test_case.post_padding.size() != spatial_rank ||
      test_case.stride.size() != spatial_rank ||
      test_case.dilation.size() != spatial_rank ||
      test_case.pre_padding == test_case.post_padding) {
    throw std::invalid_argument(
        "asymmetric convolution DAG geometry is invalid");
  }
  AsymmetricGeometry result;
  result.temporary_y = test_case.y;
  const std::int64_t maximum_uid =
      std::max({test_case.x.uid, test_case.w.uid, test_case.y.uid});
  if (maximum_uid > std::numeric_limits<std::int64_t>::max() - 2) {
    throw std::overflow_error("asymmetric convolution UID overflows");
  }
  result.temporary_y.uid = maximum_uid + 1;
  result.temporary_y.binding_byte_offset = 0;
  result.symmetric_padding.resize(spatial_rank);
  std::vector<std::int64_t> slice_offsets(spatial_rank);

  // For every spatial axis choose symmetric padding P = pre + k * stride
  // with P >= post.  Output element j from the asymmetric convolution then
  // equals element j + k from the symmetric convolution.  Fprop slices that
  // superset; Dgrad/Wgrad place their incoming Y gradient into the same
  // subview of an otherwise-zero temporary tensor.
  for (std::size_t axis = 0; axis < spatial_rank; ++axis) {
    const std::int64_t pre = test_case.pre_padding[axis];
    const std::int64_t post = test_case.post_padding[axis];
    const std::int64_t step = test_case.stride[axis];
    const std::int64_t dilation = test_case.dilation[axis];
    const std::int64_t input = test_case.x.dimensions[axis + 2];
    const std::int64_t kernel = test_case.w.dimensions[axis + 2];
    if (pre < 0 || post < 0 || step <= 0 || dilation <= 0 || input <= 0 ||
        kernel <= 0) {
      throw std::invalid_argument(
          "asymmetric convolution spatial geometry is invalid");
    }
    std::int64_t padding = pre;
    if (post > pre) {
      const std::int64_t difference = post - pre;
      if (difference >
          std::numeric_limits<std::int64_t>::max() - (step - 1)) {
        throw std::overflow_error(
            "asymmetric convolution padding alignment overflows");
      }
      const std::int64_t increments = (difference + step - 1) / step;
      if (increments >
          (std::numeric_limits<std::int64_t>::max() - pre) / step) {
        throw std::overflow_error(
            "asymmetric convolution symmetric padding overflows");
      }
      padding = pre + increments * step;
    }
    result.symmetric_padding[axis] = padding;
    slice_offsets[axis] = (padding - pre) / step;

    if (kernel - 1 >
        (std::numeric_limits<std::int64_t>::max() - 1) / dilation) {
      throw std::overflow_error(
          "asymmetric convolution effective kernel overflows");
    }
    const std::int64_t effective_kernel = (kernel - 1) * dilation + 1;
    if (padding >
        (std::numeric_limits<std::int64_t>::max() - input) / 2) {
      throw std::overflow_error(
          "asymmetric convolution padded input overflows");
    }
    const std::int64_t padded_input = input + 2 * padding;
    if (padded_input < effective_kernel) {
      throw std::invalid_argument(
          "asymmetric convolution effective kernel exceeds padded input");
    }
    const std::int64_t temporary_extent =
        (padded_input - effective_kernel) / step + 1;
    if (slice_offsets[axis] > temporary_extent ||
        test_case.y.dimensions[axis + 2] >
            temporary_extent - slice_offsets[axis]) {
      throw std::invalid_argument(
          "asymmetric convolution output is outside symmetric superset");
    }
    result.temporary_y.dimensions[axis + 2] = temporary_extent;
  }

  const bool y_channels_last = channels_last(test_case.y);
  if (!y_channels_last && !dense_contiguous(test_case.y)) {
    throw std::invalid_argument(
        "asymmetric convolution output layout is unsupported");
  }
  result.temporary_y.strides =
      dense_strides(result.temporary_y.dimensions, y_channels_last);
  std::size_t element_offset = 0;
  for (std::size_t axis = 0; axis < spatial_rank; ++axis) {
    const std::uint64_t contribution =
        static_cast<std::uint64_t>(slice_offsets[axis]) *
        static_cast<std::uint64_t>(result.temporary_y.strides[axis + 2]);
    if (contribution > std::numeric_limits<std::size_t>::max() -
                           element_offset) {
      throw std::overflow_error(
          "asymmetric convolution subview offset overflows");
    }
    element_offset += static_cast<std::size_t>(contribution);
  }
  const std::size_t scalar_bytes = element_size(test_case.y.data_type);
  if (element_offset >
      std::numeric_limits<std::size_t>::max() / scalar_bytes) {
    throw std::overflow_error(
        "asymmetric convolution subview byte offset overflows");
  }
  result.subview_byte_offset = element_offset * scalar_bytes;
  return result;
}

void *required_pointer(std::span<const flagdnnBinding_t> bindings,
                       std::int64_t uid) {
  void *result = nullptr;
  for (const flagdnnBinding_t &binding : bindings) {
    if (binding.uid != uid) {
      continue;
    }
    if (result != nullptr || binding.device_pointer == nullptr) {
      throw std::invalid_argument(
          "acDNN convolution binding is duplicate or null");
    }
    result = binding.device_pointer;
  }
  if (result == nullptr) {
    throw std::invalid_argument("acDNN convolution binding is missing");
  }
  return result;
}

class AcdnnStridedIdentityCopy final
    : public flagdnn::testing::TestExecutable {
 public:
  AcdnnStridedIdentityCopy(TestTensor input, TestTensor output,
                           std::string primitive)
      : input_(std::move(input)),
        output_(std::move(output)),
        primitive_(std::move(primitive)) {
    if (input_.uid == output_.uid || input_.data_type != output_.data_type ||
        input_.dimensions.empty() || input_.dimensions.size() > 5 ||
        input_.dimensions != output_.dimensions ||
        input_.dimensions.size() != input_.strides.size() ||
        output_.dimensions.size() != output_.strides.size() ||
        primitive_.empty()) {
      throw std::invalid_argument(
          "asymmetric convolution copy metadata is invalid");
    }
    for (std::size_t axis = 0; axis < input_.dimensions.size(); ++axis) {
      if (input_.dimensions[axis] <= 0 || input_.strides[axis] <= 0 ||
          output_.strides[axis] <= 0) {
        throw std::invalid_argument(
            "asymmetric convolution copy geometry is invalid");
      }
    }
    const acdnnDataType_t data_type = acdnn_data_type(input_.data_type);
    if (input_.strides == output_.strides) {
      segments_.push_back({});
      if (uses_backend_descriptor()) {
        delegate_ = make_acdnn_backend_pointwise_reference(
            {.mode = ACDNN_POINTWISE_IDENTITY_FWD,
             .inputs = {input_},
             .output = output_,
             .primitive = primitive_});
        return;
      }
      input_descriptor_.set(
          data_type,
          checked_positive(input_.dimensions,
                           "asymmetric convolution copy input dimension"),
          checked_positive(input_.strides,
                           "asymmetric convolution copy input stride"));
      output_descriptor_.set(
          data_type,
          checked_positive(output_.dimensions,
                           "asymmetric convolution copy output dimension"),
          checked_positive(output_.strides,
                           "asymmetric convolution copy output stride"));
      return;
    }

    std::size_t segment_axis = input_.dimensions.size();
    std::int64_t segment_extent = 0;
    for (std::size_t axis = 0; axis < input_.dimensions.size(); ++axis) {
      if (input_.strides[axis] == 1 && output_.strides[axis] == 1 &&
          input_.dimensions[axis] > segment_extent) {
        segment_axis = axis;
        segment_extent = input_.dimensions[axis];
      }
    }
    if (segment_axis == input_.dimensions.size()) {
      throw std::invalid_argument(
          "asymmetric convolution copy has no contiguous axis");
    }
    std::size_t segment_count = 1;
    for (std::size_t axis = 0; axis < input_.dimensions.size(); ++axis) {
      if (axis == segment_axis) {
        continue;
      }
      const std::size_t extent =
          static_cast<std::size_t>(input_.dimensions[axis]);
      if (extent > std::numeric_limits<std::size_t>::max() / segment_count) {
        throw std::overflow_error(
            "asymmetric convolution copy segment count overflows");
      }
      segment_count *= extent;
    }
    segments_.reserve(segment_count);
    for (std::size_t linear = 0; linear < segment_count; ++linear) {
      std::size_t remaining = linear;
      Segment segment;
      for (std::size_t axis = input_.dimensions.size(); axis != 0; --axis) {
        const std::size_t current = axis - 1;
        if (current == segment_axis) {
          continue;
        }
        const std::size_t extent =
            static_cast<std::size_t>(input_.dimensions[current]);
        const std::size_t coordinate = remaining % extent;
        remaining /= extent;
        add_offset(segment.input_offset, coordinate,
                   input_.strides[current]);
        add_offset(segment.output_offset, coordinate,
                   output_.strides[current]);
      }
      segments_.push_back(segment);
    }
    const std::vector<int> dimensions = {
        1, 1, checked_int(segment_extent, "asymmetric copy segment extent")};
    const std::vector<int> strides = {
        dimensions.back(), dimensions.back(), 1};
    if (uses_backend_descriptor()) {
      TestTensor segment_input = input_;
      TestTensor segment_output = output_;
      segment_input.dimensions = {1, 1, segment_extent};
      segment_input.strides = {segment_extent, segment_extent, 1};
      segment_output.dimensions = segment_input.dimensions;
      segment_output.strides = segment_input.strides;
      const std::size_t scalar_bytes = element_size(input_.data_type);
      segment_input.binding_byte_offset = scalar_bytes;
      segment_output.binding_byte_offset = scalar_bytes;
      delegate_ = make_acdnn_backend_pointwise_reference(
          {.mode = ACDNN_POINTWISE_IDENTITY_FWD,
           .inputs = {segment_input},
           .output = segment_output,
           .primitive = primitive_});
      return;
    }
    input_descriptor_.set(data_type, dimensions, strides);
    output_descriptor_.set(data_type, dimensions, strides);
  }

  void prepare(std::span<const flagdnnBinding_t> bindings,
               flagdnnStream_t stream) override {
    if (delegate_ == nullptr) {
      return;
    }
    for (const Segment &segment : segments_) {
      const auto adjusted = adjusted_bindings(bindings, segment);
      delegate_->prepare(adjusted, stream);
    }
  }

  [[nodiscard]] std::size_t workspace_size() const noexcept override {
    return delegate_ == nullptr ? 0 : delegate_->workspace_size();
  }

  void execute(std::span<const flagdnnBinding_t> bindings, void *workspace,
               std::size_t workspace_size, flagdnnStream_t stream) override {
    if (bindings.size() != 2 || workspace_size != this->workspace_size() ||
        (workspace_size != 0 && workspace == nullptr) || stream == nullptr ||
        (delegate_ == nullptr && workspace != nullptr)) {
      throw std::invalid_argument(
          "asymmetric convolution copy execution is invalid");
    }
    if (delegate_ != nullptr) {
      for (const Segment &segment : segments_) {
        const auto adjusted = adjusted_bindings(bindings, segment);
        delegate_->execute(adjusted, workspace, workspace_size, stream);
      }
      return;
    }
    handle_.set_stream(reinterpret_cast<hggcStream_t>(stream));
    constexpr float alpha = 1.0F;
    constexpr float beta = 0.0F;
    const auto input_address = reinterpret_cast<std::uintptr_t>(
        required_pointer(bindings, input_.uid));
    const auto output_address = reinterpret_cast<std::uintptr_t>(
        required_pointer(bindings, output_.uid));
    const std::size_t scalar_bytes = element_size(input_.data_type);
    for (const Segment &segment : segments_) {
      if (segment.input_offset >
              (std::numeric_limits<std::uintptr_t>::max() - input_address) /
                  scalar_bytes ||
          segment.output_offset >
              (std::numeric_limits<std::uintptr_t>::max() - output_address) /
                  scalar_bytes) {
        throw std::overflow_error(
            "asymmetric convolution copy pointer overflows");
      }
      check_acdnn(
          acdnnTransformTensor(
              handle_.get(), &alpha, input_descriptor_.get(),
              reinterpret_cast<void *>(
                  input_address + segment.input_offset * scalar_bytes),
              &beta, output_descriptor_.get(),
              reinterpret_cast<void *>(
                  output_address + segment.output_offset * scalar_bytes)),
          primitive_);
    }
  }

 private:
  struct Segment {
    std::size_t input_offset = 0;
    std::size_t output_offset = 0;
  };

  static void add_offset(std::size_t &total, std::size_t coordinate,
                         std::int64_t stride) {
    const std::size_t physical_stride = static_cast<std::size_t>(stride);
    if (coordinate != 0 &&
        physical_stride >
            (std::numeric_limits<std::size_t>::max() - total) / coordinate) {
      throw std::overflow_error(
          "asymmetric convolution copy offset overflows");
    }
    total += coordinate * physical_stride;
  }

  [[nodiscard]] bool uses_backend_descriptor() const noexcept {
    return primitive_.starts_with("acdnnBackendExecute(");
  }

  [[nodiscard]] std::array<flagdnnBinding_t, 2> adjusted_bindings(
      std::span<const flagdnnBinding_t> bindings,
      const Segment &segment) const {
    if (bindings.size() != 2) {
      throw std::invalid_argument(
          "asymmetric convolution copy binding count is invalid");
    }
    const auto input_address = reinterpret_cast<std::uintptr_t>(
        required_pointer(bindings, input_.uid));
    const auto output_address = reinterpret_cast<std::uintptr_t>(
        required_pointer(bindings, output_.uid));
    const std::size_t scalar_bytes = element_size(input_.data_type);
    if (segment.input_offset >
            (std::numeric_limits<std::uintptr_t>::max() - input_address) /
                scalar_bytes ||
        segment.output_offset >
            (std::numeric_limits<std::uintptr_t>::max() - output_address) /
                scalar_bytes) {
      throw std::overflow_error(
          "asymmetric convolution copy pointer overflows");
    }
    return {{{input_.uid,
              reinterpret_cast<void *>(
                  input_address + segment.input_offset * scalar_bytes)},
             {output_.uid,
              reinterpret_cast<void *>(
                  output_address + segment.output_offset * scalar_bytes)}}};
  }

  TestTensor input_;
  TestTensor output_;
  std::string primitive_;
  std::vector<Segment> segments_;
  std::unique_ptr<flagdnn::testing::TestExecutable> delegate_;
  AcdnnHandle handle_;
  AcdnnTensorDescriptor input_descriptor_;
  AcdnnTensorDescriptor output_descriptor_;
};

class AcdnnConvolution final
    : public flagdnn::testing::ConvolutionExecutable {
 public:
  AcdnnConvolution(ConvolutionTestCase test_case,
                   const CapabilityRecord &capability)
      : test_case_(std::move(test_case)) {
    const ReferenceSelection selection = select_reference(capability);
    if (!std::holds_alternative<ReferencePlan>(selection)) {
      throw std::invalid_argument(
          "unsupported convolution reached acDNN reference");
    }
    const ReferencePlan &reference = std::get<ReferencePlan>(selection);
    if (reference.path != ReferencePath::kStablePrimitive ||
        reference.primitives !=
            std::vector<std::string>{
                expected_primitive(test_case_.direction)}) {
      throw std::invalid_argument(
          "THead acDNN convolution reference plan mismatch");
    }
    validate_qualified_case();

    std::vector<int> x_dimensions =
        checked_positive(test_case_.x.dimensions, "convolution X dimension");
    std::vector<int> x_strides =
        checked_positive(test_case_.x.strides, "convolution X stride");
    std::vector<int> w_dimensions =
        checked_positive(test_case_.w.dimensions, "convolution W dimension");
    std::vector<int> y_dimensions =
        checked_positive(test_case_.y.dimensions, "convolution Y dimension");
    std::vector<int> y_strides =
        checked_positive(test_case_.y.strides, "convolution Y stride");
    std::vector<int> padding = checked_nonnegative(
        test_case_.pre_padding, "convolution padding");
    std::vector<int> stride =
        checked_positive(test_case_.stride, "convolution stride");
    std::vector<int> dilation =
        checked_positive(test_case_.dilation, "convolution dilation");

    // acDNN runtime 1400 raises SIGFPE when the legacy convolution entry
    // points are given rank-3 NCW/NWC descriptors.  Express the identical
    // one-dimensional operation as a rank-4 convolution with a singleton
    // leading spatial axis.  No data is copied and the explicit strides keep
    // the original physical layout.
    if (test_case_.x.dimensions.size() == 3) {
      const int x_singleton_stride =
          channels_last(test_case_.x) ? x_strides[0] : x_strides[1];
      const int y_singleton_stride =
          channels_last(test_case_.y) ? y_strides[0] : y_strides[1];
      x_dimensions.insert(x_dimensions.begin() + 2, 1);
      x_strides.insert(x_strides.begin() + 2, x_singleton_stride);
      w_dimensions.insert(w_dimensions.begin() + 2, 1);
      y_dimensions.insert(y_dimensions.begin() + 2, 1);
      y_strides.insert(y_strides.begin() + 2, y_singleton_stride);
      padding.insert(padding.begin(), 0);
      stride.insert(stride.begin(), 1);
      dilation.insert(dilation.begin(), 1);
    }
    const acdnnDataType_t data_type = acdnn_data_type(test_case_.x.data_type);
    x_.set(data_type, x_dimensions, x_strides);
    w_.set(data_type, w_dimensions,
           channels_last(test_case_.w) ? ACDNN_TENSOR_NHWC
                                       : ACDNN_TENSOR_NCHW);
    y_.set(data_type, y_dimensions, y_strides);

    convolution_.set(padding, stride, dilation,
                     checked_int(test_case_.groups, "convolution groups"),
                     test_case_.mode == ConvolutionMode::kConvolution
                         ? ACDNN_CONVOLUTION
                         : ACDNN_CROSS_CORRELATION);
    query_workspace();
  }

  [[nodiscard]] std::size_t workspace_size() const noexcept override {
    return workspace_size_;
  }

  void execute(std::span<const flagdnnBinding_t> bindings, void *workspace,
               std::size_t workspace_size, flagdnnStream_t stream) override {
    if (bindings.size() != 3 || stream == nullptr ||
        workspace_size != workspace_size_ ||
        (workspace_size_ != 0 && workspace == nullptr)) {
      throw std::invalid_argument(
          "acDNN convolution execution arguments are invalid");
    }
    handle_.set_stream(reinterpret_cast<hggcStream_t>(stream));
    void *x = required_pointer(bindings, test_case_.x.uid);
    void *w = required_pointer(bindings, test_case_.w.uid);
    void *y = required_pointer(bindings, test_case_.y.uid);
    constexpr float alpha = 1.0F;
    constexpr float beta = 0.0F;
    switch (test_case_.direction) {
      case ConvolutionDirection::kFprop:
        check_acdnn(
            acdnnConvolutionForward(
                handle_.get(), &alpha, x_.get(), x, w_.get(), w,
                convolution_.get(),
                ACDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, workspace,
                workspace_size, &beta, y_.get(), y),
            "acdnnConvolutionForward(IMPLICIT_GEMM)");
        return;
      case ConvolutionDirection::kDgrad:
        check_acdnn(
            acdnnConvolutionBackwardData(
                handle_.get(), &alpha, w_.get(), w, y_.get(), y,
                convolution_.get(), ACDNN_CONVOLUTION_BWD_DATA_ALGO_0,
                workspace, workspace_size, &beta, x_.get(), x),
            "acdnnConvolutionBackwardData(ALGO_0)");
        return;
      case ConvolutionDirection::kWgrad:
        check_acdnn(
            acdnnConvolutionBackwardFilter(
                handle_.get(), &alpha, x_.get(), x, y_.get(), y,
                convolution_.get(), ACDNN_CONVOLUTION_BWD_FILTER_ALGO_0,
                workspace, workspace_size, &beta, w_.get(), w),
            "acdnnConvolutionBackwardFilter(ALGO_0)");
        return;
    }
    throw std::invalid_argument("unknown convolution direction");
  }

 private:
  void validate_qualified_case() const {
    if (test_case_.x.data_type != test_case_.w.data_type ||
        test_case_.x.data_type != test_case_.y.data_type ||
        test_case_.x.dimensions.size() < 3 ||
        test_case_.x.dimensions.size() > 5 ||
        test_case_.w.dimensions.size() != test_case_.x.dimensions.size() ||
        test_case_.y.dimensions.size() != test_case_.x.dimensions.size() ||
        test_case_.pre_padding != test_case_.post_padding ||
        (!dense_contiguous(test_case_.x) && !channels_last(test_case_.x)) ||
        (!dense_contiguous(test_case_.w) && !channels_last(test_case_.w)) ||
        (!dense_contiguous(test_case_.y) && !channels_last(test_case_.y))) {
      throw std::invalid_argument(
          "qualified acDNN convolution requires matching floating tensors, "
          "a supported dense layout, and symmetric padding");
    }
  }

  void query_workspace() {
    std::size_t workspace = 0;
    switch (test_case_.direction) {
      case ConvolutionDirection::kFprop:
        check_acdnn(
            acdnnGetConvolutionForwardWorkspaceSize(
                handle_.get(), x_.get(), w_.get(), convolution_.get(),
                y_.get(), ACDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
                &workspace),
            "acdnnGetConvolutionForwardWorkspaceSize");
        break;
      case ConvolutionDirection::kDgrad:
        check_acdnn(
            acdnnGetConvolutionBackwardDataWorkspaceSize(
                handle_.get(), w_.get(), y_.get(), convolution_.get(),
                x_.get(), ACDNN_CONVOLUTION_BWD_DATA_ALGO_0, &workspace),
            "acdnnGetConvolutionBackwardDataWorkspaceSize");
        break;
      case ConvolutionDirection::kWgrad:
        check_acdnn(
            acdnnGetConvolutionBackwardFilterWorkspaceSize(
                handle_.get(), x_.get(), y_.get(), convolution_.get(),
                w_.get(), ACDNN_CONVOLUTION_BWD_FILTER_ALGO_0, &workspace),
            "acdnnGetConvolutionBackwardFilterWorkspaceSize");
        break;
    }
    workspace_size_ = workspace;
  }

  ConvolutionTestCase test_case_;
  AcdnnHandle handle_;
  AcdnnTensorDescriptor x_;
  FilterDescriptor w_;
  AcdnnTensorDescriptor y_;
  ConvolutionDescriptor convolution_;
  std::size_t workspace_size_ = 0;
};

class AcdnnAsymmetricConvolution final
    : public flagdnn::testing::ConvolutionExecutable {
 public:
  AcdnnAsymmetricConvolution(ConvolutionTestCase test_case,
                             const CapabilityRecord &capability)
      : test_case_(std::move(test_case)),
        geometry_(asymmetric_geometry(test_case_)) {
    const ReferenceSelection selection = select_reference(capability);
    if (!std::holds_alternative<ReferencePlan>(selection) ||
        std::get<ReferencePlan>(selection).path !=
            ReferencePath::kStablePrimitive ||
        std::get<ReferencePlan>(selection).primitives !=
            asymmetric_reference_plan(test_case_.direction,
                                      test_case_.x.data_type) ||
        test_case_.x.data_type != test_case_.w.data_type ||
        test_case_.x.data_type != test_case_.y.data_type ||
        (!dense_contiguous(test_case_.x) &&
         !channels_last(test_case_.x)) ||
        (!dense_contiguous(test_case_.w) &&
         !channels_last(test_case_.w)) ||
        (!dense_contiguous(test_case_.y) &&
         !channels_last(test_case_.y))) {
      throw std::invalid_argument(
          "asymmetric convolution acDNN DAG contract mismatch");
    }

    ConvolutionTestCase symmetric_case = test_case_;
    symmetric_case.y = geometry_.temporary_y;
    symmetric_case.pre_padding = geometry_.symmetric_padding;
    symmetric_case.post_padding = geometry_.symmetric_padding;
    CapabilityRecord symmetric_capability = capability;
    symmetric_capability.reference_plan = {
        expected_primitive(test_case_.direction)};
    convolution_ = std::make_unique<AcdnnConvolution>(
        std::move(symmetric_case), symmetric_capability);
    workspace_size_ = convolution_->workspace_size();

    const std::size_t temporary_bytes =
        tensor_storage_bytes(geometry_.temporary_y);
    temporary_buffer_ = DeviceBuffer(temporary_bytes);

    temporary_subview_tensor_ = geometry_.temporary_y;
    temporary_subview_tensor_.dimensions = test_case_.y.dimensions;
    temporary_subview_tensor_.binding_byte_offset =
        geometry_.subview_byte_offset;
    const bool fprop =
        test_case_.direction == ConvolutionDirection::kFprop;
    const bool backend_copy =
        test_case_.x.data_type == FLAGDNN_DATA_BFLOAT16;
    copy_ = std::make_unique<AcdnnStridedIdentityCopy>(
        fprop ? temporary_subview_tensor_ : test_case_.y,
        fprop ? test_case_.y : temporary_subview_tensor_,
        backend_copy
            ? (fprop
                   ? "acdnnBackendExecute(POINTWISE_IDENTITY_FWD,asymmetric-output-slice)"
                   : "acdnnBackendExecute(POINTWISE_IDENTITY_FWD,asymmetric-gradient-pad)")
            : (fprop
                   ? "acdnnTransformTensor(alpha=1,beta=0,asymmetric-output-slice)"
                   : "acdnnTransformTensor(alpha=1,beta=0,asymmetric-gradient-pad)"));
    workspace_size_ =
        std::max(workspace_size_, copy_->workspace_size());

    if (test_case_.direction != ConvolutionDirection::kFprop) {
      zero_tensor_ = geometry_.temporary_y;
      zero_tensor_.uid = geometry_.temporary_y.uid + 1;
      zero_buffer_ = DeviceBuffer(temporary_bytes);
      zero_host_ = std::vector<std::byte>(temporary_bytes, std::byte{0});
      zero_fill_ = std::make_unique<AcdnnStridedIdentityCopy>(
          zero_tensor_, geometry_.temporary_y,
          backend_copy
              ? "acdnnBackendExecute(POINTWISE_IDENTITY_FWD,asymmetric-gradient-zero)"
              : "acdnnTransformTensor(alpha=1,beta=0,asymmetric-gradient-zero)");
      workspace_size_ =
          std::max(workspace_size_, zero_fill_->workspace_size());
    }
  }

  void prepare(std::span<const flagdnnBinding_t> bindings,
               flagdnnStream_t stream) override {
    if (bindings.size() != 3 || stream == nullptr) {
      throw std::invalid_argument(
          "asymmetric convolution prepare arguments are invalid");
    }
    if (test_case_.direction != ConvolutionDirection::kFprop) {
      copy_to_device_async(zero_buffer_, zero_host_, 0,
                           reinterpret_cast<CUstream>(stream));
    }
    const std::array<flagdnnBinding_t, 3> adjusted =
        adjusted_bindings(bindings);
    convolution_->prepare(adjusted, stream);
    const std::array<flagdnnBinding_t, 2> copy = copy_bindings(bindings);
    copy_->prepare(copy, stream);
    if (zero_fill_ != nullptr) {
      const std::array<flagdnnBinding_t, 2> zero = zero_bindings();
      zero_fill_->prepare(zero, stream);
    }
  }

  [[nodiscard]] std::size_t workspace_size() const noexcept override {
    return workspace_size_;
  }

  void execute(std::span<const flagdnnBinding_t> bindings, void *workspace,
               std::size_t workspace_size, flagdnnStream_t stream) override {
    if (bindings.size() != 3 || stream == nullptr ||
        workspace_size != workspace_size_ ||
        (workspace_size != 0 && workspace == nullptr)) {
      throw std::invalid_argument(
          "asymmetric convolution execution arguments are invalid");
    }
    const std::array<flagdnnBinding_t, 3> adjusted =
        adjusted_bindings(bindings);
    const std::array<flagdnnBinding_t, 2> copy = copy_bindings(bindings);
    if (test_case_.direction == ConvolutionDirection::kFprop) {
      execute_inner(*convolution_, adjusted, workspace, stream);
      execute_inner(*copy_, copy, workspace, stream);
      return;
    }

    const std::array<flagdnnBinding_t, 2> zero = zero_bindings();
    execute_inner(*zero_fill_, zero, workspace, stream);
    execute_inner(*copy_, copy, workspace, stream);
    execute_inner(*convolution_, adjusted, workspace, stream);
  }

 private:
  [[nodiscard]] std::array<flagdnnBinding_t, 3> adjusted_bindings(
      std::span<const flagdnnBinding_t> bindings) const {
    void *const x = required_pointer(bindings, test_case_.x.uid);
    void *const w = required_pointer(bindings, test_case_.w.uid);
    (void)required_pointer(bindings, test_case_.y.uid);
    if (bindings.size() != 3) {
      throw std::invalid_argument(
          "asymmetric convolution binding count is invalid");
    }
    return {{{test_case_.x.uid, x},
             {test_case_.w.uid, w},
             {geometry_.temporary_y.uid, temporary_buffer_.data()}}};
  }

  [[nodiscard]] std::array<flagdnnBinding_t, 2> copy_bindings(
      std::span<const flagdnnBinding_t> bindings) const {
    void *const y = required_pointer(bindings, test_case_.y.uid);
    void *const temporary =
        temporary_buffer_.at(geometry_.subview_byte_offset);
    if (test_case_.direction == ConvolutionDirection::kFprop) {
      return {{{temporary_subview_tensor_.uid, temporary},
               {test_case_.y.uid, y}}};
    }
    return {{{test_case_.y.uid, y},
             {temporary_subview_tensor_.uid, temporary}}};
  }

  [[nodiscard]] std::array<flagdnnBinding_t, 2> zero_bindings() const {
    if (zero_fill_ == nullptr || zero_buffer_.data() == nullptr) {
      throw std::invalid_argument(
          "asymmetric convolution zero-fill plan is unavailable");
    }
    return {{{zero_tensor_.uid, zero_buffer_.data()},
             {geometry_.temporary_y.uid, temporary_buffer_.data()}}};
  }

  static void execute_inner(flagdnn::testing::TestExecutable &executable,
                            std::span<const flagdnnBinding_t> bindings,
                            void *workspace, flagdnnStream_t stream) {
    executable.execute(bindings,
                       executable.workspace_size() == 0 ? nullptr
                                                        : workspace,
                       executable.workspace_size(), stream);
  }

  ConvolutionTestCase test_case_;
  AsymmetricGeometry geometry_;
  TestTensor temporary_subview_tensor_;
  TestTensor zero_tensor_;
  DeviceBuffer temporary_buffer_;
  DeviceBuffer zero_buffer_;
  std::vector<std::byte> zero_host_;
  std::unique_ptr<flagdnn::testing::ConvolutionExecutable> convolution_;
  std::unique_ptr<flagdnn::testing::TestExecutable> copy_;
  std::unique_ptr<flagdnn::testing::TestExecutable> zero_fill_;
  std::size_t workspace_size_ = 0;
};

class AcdnnConvBiasRelu final
    : public flagdnn::testing::CompositeExecutable {
 public:
  AcdnnConvBiasRelu(
      flagdnn::testing::ConvBiasReluTestCase test_case,
      const CapabilityRecord &capability)
      : test_case_(std::move(test_case)) {
    const ReferenceSelection selection = select_reference(capability);
    if (!std::holds_alternative<ReferencePlan>(selection)) {
      throw std::invalid_argument(
          "unsupported ConvBiasRelu reached acDNN reference");
    }
    const ReferencePlan &reference = std::get<ReferencePlan>(selection);
    if (reference.path != ReferencePath::kStablePrimitive ||
        reference.primitives != std::vector<std::string>{
                                    "acdnnConvolutionBiasActivationForward(RELU)"}) {
      throw std::invalid_argument(
          "THead acDNN ConvBiasRelu reference plan mismatch");
    }
    validate_qualified_case();

    const std::vector<int> x_dimensions = checked_positive(
        test_case_.x.dimensions, "ConvBiasRelu X dimension");
    const std::vector<int> x_strides = checked_positive(
        test_case_.x.strides, "ConvBiasRelu X stride");
    const std::vector<int> w_dimensions = checked_positive(
        test_case_.w.dimensions, "ConvBiasRelu W dimension");
    const std::vector<int> bias_dimensions = checked_positive(
        test_case_.bias.dimensions, "ConvBiasRelu bias dimension");
    const std::vector<int> bias_strides = checked_positive(
        test_case_.bias.strides, "ConvBiasRelu bias stride");
    const std::vector<int> output_dimensions = checked_positive(
        test_case_.output.dimensions, "ConvBiasRelu output dimension");
    const std::vector<int> output_strides = checked_positive(
        test_case_.output.strides, "ConvBiasRelu output stride");
    const acdnnDataType_t data_type = acdnn_data_type(test_case_.x.data_type);
    x_.set(data_type, x_dimensions, x_strides);
    w_.set(data_type, w_dimensions, ACDNN_TENSOR_NHWC);
    bias_.set(data_type, bias_dimensions, bias_strides);
    output_.set(data_type, output_dimensions, output_strides);
    const std::vector<int> padding = checked_nonnegative(
        test_case_.padding, "ConvBiasRelu padding");
    const std::vector<int> stride = checked_positive(
        test_case_.stride, "ConvBiasRelu stride");
    const std::vector<int> dilation = checked_positive(
        test_case_.dilation, "ConvBiasRelu dilation");
    convolution_.set(padding, stride, dilation, 1,
                     ACDNN_CROSS_CORRELATION);
    activation_.set(ACDNN_ACTIVATION_RELU,
                    ACDNN_NOT_PROPAGATE_NAN, 0.0);

    std::size_t convolution_workspace = 0;
    check_acdnn(
        acdnnGetConvolutionForwardWorkspaceSize(
            handle_.get(), x_.get(), w_.get(), convolution_.get(),
            output_.get(), ACDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
            &convolution_workspace),
        "acdnnGetConvolutionForwardWorkspaceSize(ConvBiasRelu)");
    convolution_workspace_size_ = convolution_workspace;
    z_offset_ = align_workspace(convolution_workspace_size_);
    z_size_ = tensor_storage_bytes(test_case_.output);
    if (z_offset_ > std::numeric_limits<std::size_t>::max() - z_size_) {
      throw std::overflow_error("ConvBiasRelu reference workspace overflows");
    }
    workspace_size_ = z_offset_ + z_size_;
  }

  [[nodiscard]] std::size_t workspace_size() const noexcept override {
    return workspace_size_;
  }

  void execute(std::span<const flagdnnBinding_t> bindings, void *workspace,
               std::size_t workspace_size, flagdnnStream_t stream) override {
    if (bindings.size() != 4 || stream == nullptr || workspace == nullptr ||
        workspace_size < workspace_size_) {
      throw std::invalid_argument(
          "acDNN ConvBiasRelu execution arguments are invalid");
    }
    handle_.set_stream(reinterpret_cast<hggcStream_t>(stream));
    void *x = required_pointer(bindings, test_case_.x.uid);
    void *w = required_pointer(bindings, test_case_.w.uid);
    void *bias = required_pointer(bindings, test_case_.bias.uid);
    void *output = required_pointer(bindings, test_case_.output.uid);
    void *z = static_cast<std::byte *>(workspace) + z_offset_;
    check_driver(
        cuMemsetD8Async(reinterpret_cast<CUdeviceptr>(z), 0, z_size_,
                        reinterpret_cast<CUstream>(stream)),
        "cuMemsetD8Async(ConvBiasRelu z)");
    constexpr float alpha_one = 1.0F;
    constexpr float alpha_zero = 0.0F;
    check_acdnn(
        acdnnConvolutionBiasActivationForward(
            handle_.get(), &alpha_one, x_.get(), x, w_.get(), w,
            convolution_.get(), ACDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
            workspace, convolution_workspace_size_, &alpha_zero,
            output_.get(), z, bias_.get(), bias, activation_.get(),
            output_.get(), output),
        "acdnnConvolutionBiasActivationForward(RELU)");
  }

 private:
  void validate_qualified_case() const {
    if (test_case_.x.data_type != test_case_.w.data_type ||
        test_case_.x.data_type != test_case_.bias.data_type ||
        test_case_.x.data_type != test_case_.output.data_type ||
        !channels_last(test_case_.x) || !channels_last(test_case_.w) ||
        !channels_last(test_case_.bias) ||
        !channels_last(test_case_.output) || test_case_.padding.size() != 2 ||
        test_case_.stride.size() != 2 || test_case_.dilation.size() != 2 ||
        std::ranges::any_of(test_case_.padding,
                            [](std::int64_t value) { return value < 0; }) ||
        std::ranges::any_of(test_case_.stride,
                            [](std::int64_t value) { return value <= 0; }) ||
        std::ranges::any_of(test_case_.dilation,
                            [](std::int64_t value) { return value <= 0; })) {
      throw std::invalid_argument(
          "qualified acDNN ConvBiasRelu requires matching NHWC tensors "
          "and valid two-dimensional convolution attributes");
    }
  }

  flagdnn::testing::ConvBiasReluTestCase test_case_;
  AcdnnHandle handle_;
  AcdnnTensorDescriptor x_;
  FilterDescriptor w_;
  AcdnnTensorDescriptor bias_;
  AcdnnTensorDescriptor output_;
  ConvolutionDescriptor convolution_;
  AcdnnActivationDescriptor activation_;
  std::size_t convolution_workspace_size_ = 0;
  std::size_t z_offset_ = 0;
  std::size_t z_size_ = 0;
  std::size_t workspace_size_ = 0;
};

}  // namespace

std::unique_ptr<flagdnn::testing::ConvolutionExecutable>
make_acdnn_convolution_reference(
    const flagdnn::testing::ConvolutionTestCase &test_case,
    const CapabilityRecord &capability) {
  if (test_case.pre_padding != test_case.post_padding &&
      capability.path == ReferencePath::kStablePrimitive &&
      capability.reference_plan.size() > 1) {
    return std::make_unique<AcdnnAsymmetricConvolution>(test_case,
                                                        capability);
  }
  return std::make_unique<AcdnnConvolution>(test_case, capability);
}

std::unique_ptr<flagdnn::testing::CompositeExecutable>
make_acdnn_conv_bias_relu_reference(
    const flagdnn::testing::ConvBiasReluTestCase &test_case,
    const CapabilityRecord &capability) {
  return std::make_unique<AcdnnConvBiasRelu>(test_case, capability);
}

}  // namespace flagdnn::validation::thead
