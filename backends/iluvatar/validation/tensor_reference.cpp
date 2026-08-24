// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "tensor_reference.hpp"

#include "corex_cudnn_raii.hpp"
#include "corex_cudnn_status.hpp"
#include "reference_tensor.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>

namespace flagdnn::iluvatar::validation {
namespace {

void *binding(std::span<const flagdnnBinding_t> bindings, std::int64_t uid) {
  const auto iterator = std::find_if(
      bindings.begin(), bindings.end(),
      [uid](const flagdnnBinding_t &value) { return value.uid == uid; });
  if (iterator == bindings.end() || iterator->device_pointer == nullptr) {
    throw std::invalid_argument("CoreX cuDNN tensor binding is missing");
  }
  return iterator->device_pointer;
}

std::int32_t
normalized_axis(const flagdnn::testing::ReductionTestCase &test_case) {
  std::int32_t axis = test_case.axis;
  if (axis < 0) {
    axis += static_cast<std::int32_t>(test_case.input.dimensions.size());
  }
  if (axis < 0 ||
      axis >= static_cast<std::int32_t>(test_case.input.dimensions.size())) {
    throw std::invalid_argument("CoreX cuDNN reduction axis is invalid");
  }
  return axis;
}

flagdnn::testing::TestTensor
full_rank_output(const flagdnn::testing::ReductionTestCase &test_case) {
  flagdnn::testing::TestTensor result = test_case.output;
  if (!test_case.keep_dimensions) {
    const std::int32_t axis = normalized_axis(test_case);
    result.dimensions.insert(result.dimensions.begin() + axis, 1);
    result.strides.insert(result.strides.begin() + axis,
                          test_case.input.strides[axis]);
  }
  return result;
}

cudnnReduceTensorOp_t reduction_mode(flagdnnReductionMode_t mode) {
  switch (mode) {
  case FLAGDNN_REDUCTION_ADD:
    return CUDNN_REDUCE_TENSOR_ADD;
  case FLAGDNN_REDUCTION_AVG:
    return CUDNN_REDUCE_TENSOR_AVG;
  case FLAGDNN_REDUCTION_MUL:
    return CUDNN_REDUCE_TENSOR_MUL;
  }
  throw std::invalid_argument("CoreX cuDNN reduction mode is invalid");
}

class ReductionExecutable final : public flagdnn::testing::TestExecutable {
public:
  explicit ReductionExecutable(
      const flagdnn::testing::ReductionTestCase &test_case)
      : input_(test_case.input), output_(test_case.output),
        input_descriptor_(make_reference_tensor(test_case.input)),
        output_descriptor_(make_reference_tensor(full_rank_output(test_case))) {
    check_cudnn(cudnnSetReduceTensorDescriptor(
                    reduction_.get(), reduction_mode(test_case.mode),
                    CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN,
                    CUDNN_REDUCE_TENSOR_NO_INDICES, CUDNN_32BIT_INDICES),
                "cudnnSetReduceTensorDescriptor");
    check_cudnn(cudnnGetReductionWorkspaceSize(
                    handle_.get(), reduction_.get(), input_descriptor_.get(),
                    output_descriptor_.get(), &workspace_size_),
                "cudnnGetReductionWorkspaceSize");
  }

  [[nodiscard]] std::size_t workspace_size() const noexcept override {
    return workspace_size_;
  }

  void execute(std::span<const flagdnnBinding_t> bindings, void *workspace,
               std::size_t workspace_size, flagdnnStream_t stream) override {
    if (workspace_size < workspace_size_ ||
        (workspace_size_ != 0 && workspace == nullptr)) {
      throw std::invalid_argument("CoreX cuDNN reduction workspace is small");
    }
    handle_.bind_stream(stream);
    const float one = 1.0F;
    const float zero = 0.0F;
    check_cudnn(cudnnReduceTensor(
                    handle_.get(), reduction_.get(), nullptr, 0, workspace,
                    workspace_size_, &one, input_descriptor_.get(),
                    binding(bindings, input_.uid), &zero,
                    output_descriptor_.get(), binding(bindings, output_.uid)),
                "cudnnReduceTensor");
  }

private:
  flagdnn::testing::TestTensor input_;
  flagdnn::testing::TestTensor output_;
  CorexCudnnHandle handle_;
  CorexCudnnTensorDescriptor input_descriptor_;
  CorexCudnnTensorDescriptor output_descriptor_;
  CorexCudnnReductionDescriptor reduction_;
  std::size_t workspace_size_ = 0;
};

struct LayoutSource {
  flagdnn::testing::TestTensor tensor;
  std::size_t additional_byte_offset = 0;
};

LayoutSource layout_source(const flagdnn::testing::LayoutTestCase &test_case) {
  LayoutSource result{test_case.input, 0};
  switch (test_case.operation) {
  case flagdnn::testing::LayoutOperation::kReshape:
    result.tensor.dimensions = test_case.output.dimensions;
    result.tensor.strides = test_case.output.strides;
    break;
  case flagdnn::testing::LayoutOperation::kTranspose:
    result.tensor.dimensions = test_case.output.dimensions;
    result.tensor.strides.resize(test_case.permutation.size());
    for (std::size_t axis = 0; axis < test_case.permutation.size(); ++axis) {
      result.tensor.strides[axis] =
          test_case.input
              .strides[static_cast<std::size_t>(test_case.permutation[axis])];
    }
    break;
  case flagdnn::testing::LayoutOperation::kSlice: {
    result.tensor.dimensions = test_case.output.dimensions;
    result.tensor.strides.resize(test_case.output.dimensions.size());
    std::size_t element_offset = 0;
    for (std::size_t axis = 0; axis < result.tensor.dimensions.size(); ++axis) {
      result.tensor.strides[axis] =
          test_case.input.strides[axis] * test_case.slice_strides[axis];
      element_offset += static_cast<std::size_t>(test_case.slices[axis].first) *
                        static_cast<std::size_t>(test_case.input.strides[axis]);
    }
    result.additional_byte_offset =
        element_offset * flagdnn_data_type_size(test_case.input.data_type);
    break;
  }
  }
  return result;
}

bool has_contiguous_strides(const flagdnn::testing::TestTensor &tensor) {
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

std::size_t layout_element_count(const flagdnn::testing::TestTensor &tensor) {
  std::size_t result = 1;
  for (const std::int64_t dimension : tensor.dimensions) {
    const std::size_t extent = static_cast<std::size_t>(dimension);
    if (dimension <= 0 ||
        result > std::numeric_limits<std::size_t>::max() / extent) {
      throw std::invalid_argument("CoreX cuDNN layout shape is too large");
    }
    result *= extent;
  }
  return result;
}

std::size_t layout_logical_offset(std::size_t logical_index,
                                  const flagdnn::testing::TestTensor &tensor) {
  std::size_t result = 0;
  for (std::size_t axis = tensor.dimensions.size(); axis != 0; --axis) {
    const std::size_t current = axis - 1;
    const std::size_t dimension =
        static_cast<std::size_t>(tensor.dimensions[current]);
    const std::size_t coordinate = logical_index % dimension;
    logical_index /= dimension;
    const std::size_t stride =
        static_cast<std::size_t>(tensor.strides[current]);
    if (coordinate != 0 &&
        stride > std::numeric_limits<std::size_t>::max() / coordinate) {
      throw std::invalid_argument("CoreX cuDNN layout offset is too large");
    }
    const std::size_t term = coordinate * stride;
    if (result > std::numeric_limits<std::size_t>::max() - term) {
      throw std::invalid_argument("CoreX cuDNN layout offset is too large");
    }
    result += term;
  }
  return result;
}

flagdnn::testing::TestTensor
scalar_tensor(flagdnn::testing::TestTensor tensor) {
  tensor.dimensions.clear();
  tensor.strides.clear();
  tensor.binding_byte_offset = 0;
  return tensor;
}

class LayoutExecutable final : public flagdnn::testing::TestExecutable {
public:
  explicit LayoutExecutable(const flagdnn::testing::LayoutTestCase &test_case)
      : input_(test_case.input), output_(test_case.output),
        source_(layout_source(test_case)),
        source_descriptor_(make_reference_tensor(source_.tensor)),
        output_descriptor_(make_reference_tensor(test_case.output)),
        physical_copy_descriptor_(make_reference_tensor(test_case.input)),
        element_descriptor_(
            make_reference_tensor(scalar_tensor(test_case.input))),
        // A transpose whose output strides are the permuted input strides
        // changes logical indexing without changing physical byte order.
        physical_copy_transpose_(
            test_case.operation ==
                flagdnn::testing::LayoutOperation::kTranspose &&
            has_contiguous_strides(input_) &&
            source_.tensor.dimensions == output_.dimensions &&
            source_.tensor.strides == output_.strides),
        scalar_sequence_(test_case.operation !=
                             flagdnn::testing::LayoutOperation::kReshape &&
                         !physical_copy_transpose_) {}

  [[nodiscard]] std::size_t workspace_size() const noexcept override {
    return 0;
  }

  void execute(std::span<const flagdnnBinding_t> bindings, void *, std::size_t,
               flagdnnStream_t stream) override {
    handle_.bind_stream(stream);
    auto *source = static_cast<std::byte *>(binding(bindings, input_.uid)) +
                   source_.additional_byte_offset;
    auto *output = static_cast<std::byte *>(binding(bindings, output_.uid));
    const float one = 1.0F;
    const float zero = 0.0F;
    if (physical_copy_transpose_) {
      check_cudnn(cudnnTransformTensor(
                      handle_.get(), &one, physical_copy_descriptor_.get(),
                      source, &zero, physical_copy_descriptor_.get(), output),
                  "cudnnTransformTensor(layout physical transpose copy)");
      return;
    }
    if (!scalar_sequence_) {
      check_cudnn(cudnnTransformTensor(handle_.get(), &one,
                                       source_descriptor_.get(), source, &zero,
                                       output_descriptor_.get(), output),
                  "cudnnTransformTensor(layout)");
      return;
    }

    const std::size_t element_size = flagdnn_data_type_size(input_.data_type);
    const std::size_t elements = layout_element_count(output_);
    for (std::size_t index = 0; index < elements; ++index) {
      const std::size_t source_offset =
          layout_logical_offset(index, source_.tensor);
      const std::size_t output_offset = layout_logical_offset(index, output_);
      check_cudnn(cudnnTransformTensor(handle_.get(), &one,
                                       element_descriptor_.get(),
                                       source + source_offset * element_size,
                                       &zero, element_descriptor_.get(),
                                       output + output_offset * element_size),
                  "cudnnTransformTensor(layout scalar sequence)");
    }
  }

private:
  flagdnn::testing::TestTensor input_;
  flagdnn::testing::TestTensor output_;
  LayoutSource source_;
  CorexCudnnHandle handle_;
  CorexCudnnTensorDescriptor source_descriptor_;
  CorexCudnnTensorDescriptor output_descriptor_;
  CorexCudnnTensorDescriptor physical_copy_descriptor_;
  CorexCudnnTensorDescriptor element_descriptor_;
  bool physical_copy_transpose_ = false;
  bool scalar_sequence_ = false;
};

} // namespace

std::unique_ptr<flagdnn::testing::TestExecutable>
make_classic_reduction_reference(
    const flagdnn::testing::ReductionTestCase &test_case) {
  return std::make_unique<ReductionExecutable>(test_case);
}

std::unique_ptr<flagdnn::testing::TestExecutable> make_classic_layout_reference(
    const flagdnn::testing::LayoutTestCase &test_case) {
  return std::make_unique<LayoutExecutable>(test_case);
}

} // namespace flagdnn::iluvatar::validation
