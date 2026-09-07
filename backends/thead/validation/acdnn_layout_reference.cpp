// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "acdnn_layout_reference.hpp"

#include "acdnn_reference.hpp"
#include "backend_pointwise_reference.hpp"
#include "numeric_types.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace flagdnn::validation::thead {
namespace {

std::vector<int> checked_ints(std::span<const std::int64_t> values,
                              std::string_view description) {
  std::vector<int> result;
  result.reserve(values.size());
  for (std::int64_t value : values) {
    if (value < 0 || value > std::numeric_limits<int>::max()) {
      throw std::invalid_argument(std::string(description) +
                                  " is outside int32");
    }
    result.push_back(static_cast<int>(value));
  }
  return result;
}

std::size_t element_count(std::span<const std::int64_t> dimensions) {
  std::size_t result = 1;
  for (std::int64_t dimension : dimensions) {
    if (dimension <= 0 ||
        static_cast<std::uint64_t>(dimension) >
            std::numeric_limits<std::size_t>::max() / result) {
      throw std::overflow_error("acDNN Layout element count overflows");
    }
    result *= static_cast<std::size_t>(dimension);
  }
  return result;
}

std::vector<std::int64_t>
contiguous_strides(std::span<const std::int64_t> dimensions) {
  std::vector<std::int64_t> result(dimensions.size());
  std::int64_t stride = 1;
  for (std::size_t axis = dimensions.size(); axis != 0; --axis) {
    const std::int64_t dimension = dimensions[axis - 1];
    if (dimension <= 0 ||
        stride > std::numeric_limits<std::int64_t>::max() / dimension) {
      throw std::overflow_error("acDNN Layout stride overflows");
    }
    result[axis - 1] = stride;
    stride *= dimension;
  }
  return result;
}

bool legacy_packed_strides(std::span<const std::int64_t> dimensions,
                           std::span<const std::int64_t> strides) {
  const std::size_t rank = dimensions.size();
  if (rank == 0 || rank > 5 || strides.size() != rank) {
    return false;
  }
  if (rank <= 3 ||
      std::ranges::equal(strides, contiguous_strides(dimensions))) {
    return true;
  }
  std::vector<std::int64_t> channels_last(rank);
  channels_last[1] = 1;
  std::int64_t physical_stride = dimensions[1];
  for (std::size_t axis = rank; axis != 2; --axis) {
    const std::size_t current = axis - 1;
    if (dimensions[current] <= 0 || physical_stride <= 0 ||
        physical_stride > std::numeric_limits<std::int64_t>::max() /
                              dimensions[current]) {
      return false;
    }
    channels_last[current] = physical_stride;
    physical_stride *= dimensions[current];
  }
  channels_last[0] = physical_stride;
  return std::ranges::equal(strides, channels_last);
}

acdnnDataType_t acdnn_data_type(flagdnnDataType_t data_type) {
  switch (data_type) {
    case FLAGDNN_DATA_FLOAT32:
      return ACDNN_DATA_FLOAT;
    case FLAGDNN_DATA_FLOAT16:
      return ACDNN_DATA_HALF;
    case FLAGDNN_DATA_BFLOAT16:
      return ACDNN_DATA_BF16;
    case FLAGDNN_DATA_BOOLEAN:
    case FLAGDNN_DATA_FP8_E4M3:
    case FLAGDNN_DATA_FP8_E5M2:
      break;
  }
  throw std::invalid_argument("acDNN Layout data type is unsupported");
}

void validate_acdnn_layout_case(
    const flagdnn::testing::LayoutTestCase &test_case) {
  const auto valid_tensor = [](const flagdnn::testing::TestTensor &tensor) {
    return tensor.uid > 0 && !tensor.dimensions.empty() &&
           tensor.dimensions.size() <= 8 &&
           tensor.dimensions.size() == tensor.strides.size() &&
           std::ranges::all_of(tensor.dimensions,
                               [](std::int64_t value) { return value > 0; }) &&
           std::ranges::all_of(tensor.strides,
                               [](std::int64_t value) { return value > 0; });
  };
  if (test_case.name.empty() || test_case.input.uid == test_case.output.uid ||
      !valid_tensor(test_case.input) || !valid_tensor(test_case.output) ||
      test_case.input.data_type != test_case.output.data_type) {
    throw std::invalid_argument("THead acDNN Layout metadata is invalid");
  }
  const std::size_t rank = test_case.input.dimensions.size();
  if (test_case.operation == flagdnn::testing::LayoutOperation::kReshape) {
    if (element_count(test_case.input.dimensions) !=
            element_count(test_case.output.dimensions) ||
        !test_case.permutation.empty() || !test_case.slices.empty() ||
        !test_case.slice_strides.empty()) {
      throw std::invalid_argument("THead acDNN Reshape metadata is invalid");
    }
    return;
  }
  if (test_case.output.dimensions.size() != rank) {
    throw std::invalid_argument("THead acDNN Layout ranks differ");
  }
  if (test_case.operation == flagdnn::testing::LayoutOperation::kTranspose) {
    std::vector<std::int64_t> sorted = test_case.permutation;
    std::ranges::sort(sorted);
    if (sorted.size() != rank ||
        sorted != [&] {
          std::vector<std::int64_t> axes(rank);
          std::iota(axes.begin(), axes.end(), 0);
          return axes;
        }()) {
      throw std::invalid_argument(
          "THead acDNN Transpose permutation is invalid");
    }
    for (std::size_t axis = 0; axis < rank; ++axis) {
      if (test_case.output.dimensions[axis] !=
          test_case.input.dimensions[static_cast<std::size_t>(
              test_case.permutation[axis])]) {
        throw std::invalid_argument(
            "THead acDNN Transpose shape is invalid");
      }
    }
    return;
  }
  if (test_case.slices.size() != rank ||
      test_case.slice_strides.size() != rank ||
      !test_case.permutation.empty()) {
    throw std::invalid_argument("THead acDNN Slice metadata is invalid");
  }
  for (std::size_t axis = 0; axis < rank; ++axis) {
    const auto [start, limit] = test_case.slices[axis];
    const std::int64_t step = test_case.slice_strides[axis];
    if (start < 0 || limit <= start ||
        limit > test_case.input.dimensions[axis] || step <= 0 ||
        test_case.output.dimensions[axis] !=
            (limit - start + step - 1) / step) {
      throw std::invalid_argument("THead acDNN Slice range is invalid");
    }
  }
}

std::string expected_primitive(flagdnn::testing::LayoutOperation operation) {
  switch (operation) {
    case flagdnn::testing::LayoutOperation::kReshape:
      return "acdnnTransformTensor(flattened,alpha=1,beta=0)";
    case flagdnn::testing::LayoutOperation::kTranspose:
      return "acdnnTransformTensor(permuted-stride,alpha=1,beta=0)";
    case flagdnn::testing::LayoutOperation::kSlice:
      return "acdnnTransformTensor(slice-segments,alpha=1,beta=0)";
  }
  throw std::invalid_argument("unknown THead Layout operation");
}

constexpr std::string_view kBackendLayoutPrimitive =
    "acdnnBackendExecute(POINTWISE_IDENTITY_FWD,layout-map)";
constexpr std::string_view kConvertLayoutInputFp32 =
    "acdnnBackendExecute(POINTWISE_IDENTITY_FWD,convert-input0-fp32)";
constexpr std::string_view kConvertLayoutOutputBfloat16 =
    "acdnnBackendExecute(POINTWISE_IDENTITY_FWD,convert-output-bfloat16)";

class AcdnnLayout final : public flagdnn::testing::LayoutExecutable {
 public:
  AcdnnLayout(flagdnn::testing::LayoutTestCase test_case,
              const CapabilityRecord &capability)
      : test_case_(std::move(test_case)) {
    validate_acdnn_layout_case(test_case_);
    if ((test_case_.input.data_type != FLAGDNN_DATA_FLOAT32 &&
         test_case_.input.data_type != FLAGDNN_DATA_FLOAT16 &&
         test_case_.input.data_type != FLAGDNN_DATA_BFLOAT16) ||
        test_case_.output.data_type != test_case_.input.data_type) {
      throw std::invalid_argument(
          "THead acDNN Layout requires matching floating types");
    }
    const ReferenceSelection selection = select_reference(capability);
    if (!std::holds_alternative<ReferencePlan>(selection)) {
      throw std::invalid_argument(
          "unsupported Layout case reached acDNN construction");
    }
    const ReferencePlan &plan = std::get<ReferencePlan>(selection);
    const bool stable =
        plan.path == ReferencePath::kStablePrimitive &&
        plan.primitives ==
            std::vector<std::string>{expected_primitive(test_case_.operation)};
    const bool backend =
        plan.path == ReferencePath::kBackendDescriptor &&
        plan.primitives ==
            std::vector<std::string>{std::string(kBackendLayoutPrimitive)};
    if (!stable && !backend) {
      throw std::invalid_argument("THead acDNN Layout plan mismatch");
    }

    std::vector<std::int64_t> logical_dimensions;
    std::vector<std::int64_t> logical_input_strides;
    std::vector<std::int64_t> logical_output_strides;
    if (test_case_.operation ==
        flagdnn::testing::LayoutOperation::kReshape) {
      const std::size_t elements = element_count(test_case_.output.dimensions);
      if (elements > static_cast<std::size_t>(
                         std::numeric_limits<int>::max())) {
        throw std::overflow_error("acDNN Reshape element count exceeds int32");
      }
      logical_dimensions = {1, 1, static_cast<std::int64_t>(elements)};
      logical_input_strides = {
          static_cast<std::int64_t>(elements),
          static_cast<std::int64_t>(elements), 1};
      logical_output_strides = logical_input_strides;
    } else if (test_case_.operation ==
               flagdnn::testing::LayoutOperation::kTranspose) {
      // Keep descriptors in the input coordinate system.  Mapping the dense
      // reference-output strides back through the permutation expresses the
      // transpose.  Canonical maps use the stable primitive; general maps
      // use the backend identity descriptor because runtime 1400 rejects
      // their legacy tensor descriptors.
      logical_dimensions = test_case_.input.dimensions;
      logical_input_strides = test_case_.input.strides;
      const std::vector<std::int64_t> dense_output_strides =
          contiguous_strides(test_case_.output.dimensions);
      logical_output_strides.resize(test_case_.permutation.size());
      for (std::size_t output_axis = 0;
           output_axis < test_case_.permutation.size(); ++output_axis) {
        const std::size_t input_axis = static_cast<std::size_t>(
            test_case_.permutation[output_axis]);
        logical_output_strides[input_axis] =
            dense_output_strides[output_axis];
      }
      if (!backend &&
          !legacy_packed_strides(logical_dimensions,
                                 logical_output_strides)) {
        // The inverse NCHW/NHWC transform is canonical only when expressed
        // in output coordinates.  Switch coordinate systems so the input is
        // NHWC/NDHWC and the output is dense NCHW/NCDHW.
        logical_dimensions = test_case_.output.dimensions;
        logical_output_strides = dense_output_strides;
        logical_input_strides.resize(test_case_.permutation.size());
        for (std::size_t output_axis = 0;
             output_axis < test_case_.permutation.size(); ++output_axis) {
          logical_input_strides[output_axis] =
              test_case_.input.strides[static_cast<std::size_t>(
                  test_case_.permutation[output_axis])];
        }
        if (!legacy_packed_strides(logical_dimensions,
                                   logical_input_strides)) {
          throw std::invalid_argument(
              "legacy acDNN Transpose has no canonical descriptor mapping");
        }
      }
    } else {
      const std::size_t rank = test_case_.output.dimensions.size();
      if (backend) {
        logical_dimensions = test_case_.output.dimensions;
        logical_input_strides.resize(rank);
        logical_output_strides = contiguous_strides(logical_dimensions);
        for (std::size_t axis = 0; axis < rank; ++axis) {
          const auto [start, limit] = test_case_.slices[axis];
          (void)limit;
          const std::int64_t step = test_case_.slice_strides[axis];
          if (start > 0 &&
              static_cast<std::uint64_t>(start) >
                  (std::numeric_limits<std::size_t>::max() -
                   backend_input_base_) /
                      static_cast<std::uint64_t>(
                          test_case_.input.strides[axis])) {
            throw std::overflow_error("THead acDNN Slice base overflows");
          }
          backend_input_base_ +=
              static_cast<std::size_t>(start) *
              static_cast<std::size_t>(test_case_.input.strides[axis]);
          if (step >
              std::numeric_limits<std::int64_t>::max() /
                  test_case_.input.strides[axis]) {
            throw std::overflow_error("THead acDNN Slice stride overflows");
          }
          logical_input_strides[axis] =
              test_case_.input.strides[axis] * step;
        }
      } else {
      const std::size_t last = rank - 1;
      const bool contiguous_last_axis =
          test_case_.slice_strides[last] == 1 &&
          test_case_.input.strides[last] == 1;
      segment_length_ =
          contiguous_last_axis
              ? static_cast<std::size_t>(test_case_.output.dimensions[last])
              : 1;
      const std::size_t output_elements =
          element_count(test_case_.output.dimensions);
      const std::size_t segment_count = output_elements / segment_length_;
      segments_.reserve(segment_count);
      for (std::size_t segment = 0; segment < segment_count; ++segment) {
        const std::size_t logical_index = segment * segment_length_;
        std::size_t remaining = logical_index;
        std::size_t input_offset = 0;
        for (std::size_t axis = rank; axis != 0; --axis) {
          const std::size_t current = axis - 1;
          const std::size_t extent = static_cast<std::size_t>(
              test_case_.output.dimensions[current]);
          const std::size_t coordinate = remaining % extent;
          remaining /= extent;
          const auto [start, limit] = test_case_.slices[current];
          (void)limit;
          input_offset +=
              (static_cast<std::size_t>(start) +
               coordinate * static_cast<std::size_t>(
                                test_case_.slice_strides[current])) *
              static_cast<std::size_t>(test_case_.input.strides[current]);
        }
        segments_.push_back(
            {input_offset, logical_index});
      }
      const std::vector<std::int64_t> segment_dimensions = {
          1, 1, static_cast<std::int64_t>(segment_length_)};
      const std::vector<std::int64_t> segment_strides = {
          static_cast<std::int64_t>(segment_length_),
          static_cast<std::int64_t>(segment_length_), 1};
      const acdnnDataType_t data_type =
          acdnn_data_type(test_case_.input.data_type);
      input_.set(data_type,
                 checked_ints(segment_dimensions, "slice segment dimension"),
                 checked_ints(segment_strides, "slice segment stride"));
      output_.set(
          data_type,
          checked_ints(segment_dimensions, "slice segment dimension"),
          checked_ints(segment_strides, "slice segment stride"));
      return;
      }
    }
    if (backend) {
      flagdnn::testing::TestTensor logical_input = test_case_.input;
      flagdnn::testing::TestTensor logical_output = test_case_.output;
      logical_input.dimensions = logical_dimensions;
      logical_input.strides = logical_input_strides;
      logical_output.dimensions = logical_dimensions;
      logical_output.strides = logical_output_strides;
      backend_ = make_acdnn_backend_pointwise_reference(
          {.mode = ACDNN_POINTWISE_IDENTITY_FWD,
           .inputs = {logical_input},
           .output = logical_output,
           .primitive = std::string(kBackendLayoutPrimitive)});
      return;
    }
    const acdnnDataType_t data_type =
        acdnn_data_type(test_case_.input.data_type);
    input_.set(data_type, checked_ints(logical_dimensions, "dimension"),
               checked_ints(logical_input_strides, "input stride"));
    output_.set(data_type,
                checked_ints(logical_dimensions, "dimension"),
                checked_ints(logical_output_strides, "output stride"));
  }

  [[nodiscard]] std::size_t workspace_size() const noexcept override {
    return backend_ == nullptr ? 0 : backend_->workspace_size();
  }

  void execute(std::span<const flagdnnBinding_t> bindings, void *workspace,
               std::size_t workspace_size,
               flagdnnStream_t stream) override {
    if (stream == nullptr || workspace_size != this->workspace_size() ||
        (workspace_size != 0 && workspace == nullptr) ||
        (backend_ == nullptr && workspace != nullptr)) {
      throw std::invalid_argument(
          "THead acDNN Layout workspace or stream is invalid");
    }
    std::map<std::int64_t, void *> pointers;
    for (const flagdnnBinding_t &binding : bindings) {
      if (binding.device_pointer == nullptr ||
          !pointers.emplace(binding.uid, binding.device_pointer).second) {
        throw std::invalid_argument("THead acDNN Layout bindings are invalid");
      }
    }
    if (pointers.size() != 2 || !pointers.contains(test_case_.input.uid) ||
        !pointers.contains(test_case_.output.uid)) {
      throw std::invalid_argument(
          "THead acDNN Layout requires input/output bindings");
    }
    if (backend_ != nullptr) {
      const std::size_t scalar_bytes = element_size(test_case_.input.data_type);
      const auto input_address = reinterpret_cast<std::uintptr_t>(
          pointers.at(test_case_.input.uid));
      if (backend_input_base_ >
          (std::numeric_limits<std::uintptr_t>::max() - input_address) /
              scalar_bytes) {
        throw std::overflow_error("THead acDNN Layout pointer overflows");
      }
      const std::array<flagdnnBinding_t, 2> mapped_bindings = {{
          {test_case_.input.uid,
           reinterpret_cast<void *>(input_address +
                                    backend_input_base_ * scalar_bytes)},
          {test_case_.output.uid, pointers.at(test_case_.output.uid)},
      }};
      backend_->execute(mapped_bindings, workspace, workspace_size, stream);
      return;
    }
    handle_.set_stream(reinterpret_cast<hggcStream_t>(stream));
    constexpr float alpha = 1.0F;
    constexpr float beta = 0.0F;
    const std::size_t scalar_bytes = element_size(test_case_.input.data_type);
    if (!segments_.empty()) {
      const auto input_address = reinterpret_cast<std::uintptr_t>(
          pointers.at(test_case_.input.uid));
      const auto output_address = reinterpret_cast<std::uintptr_t>(
          pointers.at(test_case_.output.uid));
      for (const Segment &segment : segments_) {
        if (segment.input_offset >
                (std::numeric_limits<std::uintptr_t>::max() - input_address) /
                    scalar_bytes ||
            segment.output_offset >
                (std::numeric_limits<std::uintptr_t>::max() - output_address) /
                    scalar_bytes) {
          throw std::overflow_error("THead acDNN Slice pointer overflows");
        }
        check_acdnn(
            acdnnTransformTensor(
                handle_.get(), &alpha, input_.get(),
                reinterpret_cast<void *>(
                    input_address + segment.input_offset * scalar_bytes),
                &beta, output_.get(),
                reinterpret_cast<void *>(
                    output_address + segment.output_offset * scalar_bytes)),
            expected_primitive(test_case_.operation));
      }
      return;
    }
    check_acdnn(acdnnTransformTensor(
                    handle_.get(), &alpha, input_.get(),
                    pointers.at(test_case_.input.uid), &beta,
                    output_.get(), pointers.at(test_case_.output.uid)),
                expected_primitive(test_case_.operation));
  }

 private:
  struct Segment {
    std::size_t input_offset = 0;
    std::size_t output_offset = 0;
  };

  flagdnn::testing::LayoutTestCase test_case_;
  std::size_t segment_length_ = 0;
  std::size_t backend_input_base_ = 0;
  std::vector<Segment> segments_;
  std::unique_ptr<flagdnn::testing::TestExecutable> backend_;
  AcdnnHandle handle_;
  AcdnnTensorDescriptor input_;
  AcdnnTensorDescriptor output_;
};

std::size_t fp32_storage_bytes(
    const flagdnn::testing::TestTensor &tensor) {
  std::size_t maximum_offset = 0;
  if (tensor.dimensions.empty() ||
      tensor.dimensions.size() != tensor.strides.size()) {
    throw std::invalid_argument(
        "converted BF16 Transpose tensor geometry is invalid");
  }
  for (std::size_t axis = 0; axis < tensor.dimensions.size(); ++axis) {
    const std::int64_t dimension = tensor.dimensions[axis];
    const std::int64_t stride = tensor.strides[axis];
    if (dimension <= 0 || stride <= 0) {
      throw std::invalid_argument(
          "converted BF16 Transpose tensor geometry is invalid");
    }
    const std::size_t extent = static_cast<std::size_t>(dimension - 1);
    const std::size_t physical_stride = static_cast<std::size_t>(stride);
    if (physical_stride != 0 &&
        extent >
            std::numeric_limits<std::size_t>::max() / physical_stride) {
      throw std::overflow_error(
          "converted BF16 Transpose storage overflows");
    }
    const std::size_t contribution = extent * physical_stride;
    if (maximum_offset >
        std::numeric_limits<std::size_t>::max() - contribution) {
      throw std::overflow_error(
          "converted BF16 Transpose storage overflows");
    }
    maximum_offset += contribution;
  }
  if (maximum_offset == std::numeric_limits<std::size_t>::max() ||
      maximum_offset + 1 >
          std::numeric_limits<std::size_t>::max() / sizeof(float)) {
    throw std::overflow_error(
        "converted BF16 Transpose storage is too large");
  }
  return (maximum_offset + 1) * sizeof(float);
}

class AcdnnConvertedBfloat16Transpose final
    : public flagdnn::testing::LayoutExecutable {
 public:
  AcdnnConvertedBfloat16Transpose(
      flagdnn::testing::LayoutTestCase test_case,
      const CapabilityRecord &capability)
      : test_case_(std::move(test_case)) {
    validate_acdnn_layout_case(test_case_);
    const std::vector<std::string> expected_plan = {
        std::string(kConvertLayoutInputFp32),
        expected_primitive(flagdnn::testing::LayoutOperation::kTranspose),
        std::string(kConvertLayoutOutputBfloat16),
    };
    const ReferenceSelection selection = select_reference(capability);
    if (test_case_.operation !=
            flagdnn::testing::LayoutOperation::kTranspose ||
        test_case_.input.data_type != FLAGDNN_DATA_BFLOAT16 ||
        test_case_.output.data_type != FLAGDNN_DATA_BFLOAT16 ||
        !std::holds_alternative<ReferencePlan>(selection) ||
        std::get<ReferencePlan>(selection).path !=
            ReferencePath::kBackendDescriptor ||
        std::get<ReferencePlan>(selection).primitives != expected_plan) {
      throw std::invalid_argument(
          "converted BF16 Transpose acDNN DAG plan mismatch");
    }
    const std::int64_t maximum_uid =
        std::max(test_case_.input.uid, test_case_.output.uid);
    if (maximum_uid > std::numeric_limits<std::int64_t>::max() - 2) {
      throw std::overflow_error(
          "converted BF16 Transpose internal UID overflows");
    }
    fp32_input_ = test_case_.input;
    fp32_input_.uid = maximum_uid + 1;
    fp32_input_.data_type = FLAGDNN_DATA_FLOAT32;
    fp32_input_.binding_byte_offset = 0;
    fp32_output_ = test_case_.output;
    fp32_output_.uid = maximum_uid + 2;
    fp32_output_.data_type = FLAGDNN_DATA_FLOAT32;
    fp32_output_.binding_byte_offset = 0;
    input_buffer_ = DeviceBuffer(fp32_storage_bytes(fp32_input_));
    output_buffer_ = DeviceBuffer(fp32_storage_bytes(fp32_output_));

    input_conversion_ = make_acdnn_backend_pointwise_reference(
        {.mode = ACDNN_POINTWISE_IDENTITY_FWD,
         .inputs = {test_case_.input},
         .output = fp32_input_,
         .primitive = std::string(kConvertLayoutInputFp32)});
    flagdnn::testing::LayoutTestCase fp32_case = test_case_;
    fp32_case.input = fp32_input_;
    fp32_case.output = fp32_output_;
    CapabilityRecord transpose_capability = capability;
    transpose_capability.path = ReferencePath::kStablePrimitive;
    transpose_capability.reference_plan = {
        expected_primitive(flagdnn::testing::LayoutOperation::kTranspose)};
    transpose_ =
        std::make_unique<AcdnnLayout>(std::move(fp32_case),
                                      transpose_capability);
    output_conversion_ = make_acdnn_backend_pointwise_reference(
        {.mode = ACDNN_POINTWISE_IDENTITY_FWD,
         .inputs = {fp32_output_},
         .output = test_case_.output,
         .primitive = std::string(kConvertLayoutOutputBfloat16)});
    workspace_size_ =
        std::max({input_conversion_->workspace_size(),
                  transpose_->workspace_size(),
                  output_conversion_->workspace_size()});
  }

  void prepare(std::span<const flagdnnBinding_t> bindings,
               flagdnnStream_t stream) override {
    const auto pointers = binding_map(bindings);
    const std::array<flagdnnBinding_t, 2> input_bindings = {{
        {test_case_.input.uid, pointers.at(test_case_.input.uid)},
        {fp32_input_.uid, input_buffer_.data()},
    }};
    const std::array<flagdnnBinding_t, 2> transpose_bindings = {{
        {fp32_input_.uid, input_buffer_.data()},
        {fp32_output_.uid, output_buffer_.data()},
    }};
    const std::array<flagdnnBinding_t, 2> output_bindings = {{
        {fp32_output_.uid, output_buffer_.data()},
        {test_case_.output.uid, pointers.at(test_case_.output.uid)},
    }};
    input_conversion_->prepare(input_bindings, stream);
    transpose_->prepare(transpose_bindings, stream);
    output_conversion_->prepare(output_bindings, stream);
  }

  [[nodiscard]] std::size_t workspace_size() const noexcept override {
    return workspace_size_;
  }

  void execute(std::span<const flagdnnBinding_t> bindings, void *workspace,
               std::size_t workspace_size,
               flagdnnStream_t stream) override {
    if (workspace_size != workspace_size_ ||
        (workspace_size != 0 && workspace == nullptr)) {
      throw std::invalid_argument(
          "converted BF16 Transpose workspace does not match plan");
    }
    const auto pointers = binding_map(bindings);
    const std::array<flagdnnBinding_t, 2> input_bindings = {{
        {test_case_.input.uid, pointers.at(test_case_.input.uid)},
        {fp32_input_.uid, input_buffer_.data()},
    }};
    const std::array<flagdnnBinding_t, 2> transpose_bindings = {{
        {fp32_input_.uid, input_buffer_.data()},
        {fp32_output_.uid, output_buffer_.data()},
    }};
    const std::array<flagdnnBinding_t, 2> output_bindings = {{
        {fp32_output_.uid, output_buffer_.data()},
        {test_case_.output.uid, pointers.at(test_case_.output.uid)},
    }};
    execute_inner(*input_conversion_, input_bindings, workspace, stream);
    execute_inner(*transpose_, transpose_bindings, workspace, stream);
    execute_inner(*output_conversion_, output_bindings, workspace, stream);
  }

 private:
  [[nodiscard]] std::map<std::int64_t, void *> binding_map(
      std::span<const flagdnnBinding_t> bindings) const {
    std::map<std::int64_t, void *> result;
    for (const flagdnnBinding_t &binding : bindings) {
      if (binding.device_pointer == nullptr ||
          !result.emplace(binding.uid, binding.device_pointer).second) {
        throw std::invalid_argument(
            "converted BF16 Transpose binding is null or duplicate");
      }
    }
    if (result.size() != 2 || !result.contains(test_case_.input.uid) ||
        !result.contains(test_case_.output.uid)) {
      throw std::invalid_argument(
          "converted BF16 Transpose bindings do not match graph");
    }
    return result;
  }

  void execute_inner(flagdnn::testing::TestExecutable &executable,
                     std::span<const flagdnnBinding_t> bindings,
                     void *workspace, flagdnnStream_t stream) const {
    executable.execute(bindings,
                       executable.workspace_size() == 0 ? nullptr
                                                        : workspace,
                       executable.workspace_size(), stream);
  }

  flagdnn::testing::LayoutTestCase test_case_;
  flagdnn::testing::TestTensor fp32_input_;
  flagdnn::testing::TestTensor fp32_output_;
  DeviceBuffer input_buffer_;
  DeviceBuffer output_buffer_;
  std::unique_ptr<flagdnn::testing::TestExecutable> input_conversion_;
  std::unique_ptr<flagdnn::testing::LayoutExecutable> transpose_;
  std::unique_ptr<flagdnn::testing::TestExecutable> output_conversion_;
  std::size_t workspace_size_ = 0;
};

}  // namespace

bool legacy_acdnn_transpose_descriptor_compatible(
    std::span<const std::int64_t> input_dimensions,
    std::span<const std::int64_t> permutation) {
  const std::size_t rank = input_dimensions.size();
  if (rank == 0 || rank > 5 || permutation.size() != rank ||
      std::ranges::any_of(input_dimensions,
                          [](std::int64_t value) { return value <= 0; })) {
    throw std::invalid_argument(
        "legacy acDNN Transpose geometry is invalid");
  }
  std::vector<std::int64_t> sorted(permutation.begin(), permutation.end());
  std::ranges::sort(sorted);
  for (std::size_t axis = 0; axis < rank; ++axis) {
    if (sorted[axis] != static_cast<std::int64_t>(axis)) {
      throw std::invalid_argument(
          "legacy acDNN Transpose permutation is invalid");
    }
  }

  std::vector<std::int64_t> output_dimensions(rank);
  for (std::size_t axis = 0; axis < rank; ++axis) {
    output_dimensions[axis] = input_dimensions[static_cast<std::size_t>(
        permutation[axis])];
  }
  const std::vector<std::int64_t> dense_output_strides =
      contiguous_strides(output_dimensions);
  std::vector<std::int64_t> mapped_output_strides(rank);
  for (std::size_t output_axis = 0; output_axis < rank; ++output_axis) {
    mapped_output_strides[static_cast<std::size_t>(
        permutation[output_axis])] = dense_output_strides[output_axis];
  }

  // First try input coordinates: dense input -> mapped output.
  if (legacy_packed_strides(input_dimensions, mapped_output_strides)) {
    return true;
  }

  // Then try output coordinates: mapped input -> dense output.  This covers
  // the inverse channels-last transform used by the largest public case.
  const std::vector<std::int64_t> dense_input_strides =
      contiguous_strides(input_dimensions);
  std::vector<std::int64_t> mapped_input_strides(rank);
  for (std::size_t output_axis = 0; output_axis < rank; ++output_axis) {
    mapped_input_strides[output_axis] =
        dense_input_strides[static_cast<std::size_t>(
            permutation[output_axis])];
  }
  return legacy_packed_strides(output_dimensions, mapped_input_strides);
}

std::unique_ptr<flagdnn::testing::LayoutExecutable>
make_acdnn_layout_reference(
    const flagdnn::testing::LayoutTestCase &test_case,
    const CapabilityRecord &capability) {
  if (capability.path == ReferencePath::kBackendDescriptor &&
      capability.reference_plan.size() > 1) {
    return std::make_unique<AcdnnConvertedBfloat16Transpose>(
        test_case, capability);
  }
  return std::make_unique<AcdnnLayout>(test_case, capability);
}

}  // namespace flagdnn::validation::thead

namespace flagdnn::testing {

std::unique_ptr<LayoutExecutable> build_layout_reference(
    const LayoutTestCase &test_case) {
  using namespace flagdnn::validation::thead;
  static const CapabilityCatalog catalog = CapabilityCatalog::load(
      FLAGDNN_THEAD_ACDNN_CAPABILITY_CATALOG);
  const std::string operation =
      test_case.operation == LayoutOperation::kReshape
          ? "reshape"
          : test_case.operation == LayoutOperation::kTranspose ? "transpose"
                                                               : "slice";
  return make_acdnn_layout_reference(
      test_case, catalog.lookup(operation, test_case.name));
}

}  // namespace flagdnn::testing
