/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "backends/mthreads/validation/mudnn_convolution.hpp"

#include "backends/mthreads/validation/musa_driver.hpp"
#include "backends/mthreads/validation/mudnn_workspace.hpp"
#include "backends/mthreads/validation/tensor_io.hpp"

#include <mudnn.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace flagdnn::validation::mthreads {
namespace {

constexpr std::size_t kWorkspaceAlignment = 256;
// The deployed muDNN can request internal storage from MemoryMaintainer that
// was omitted by its algorithm-workspace query. Keep a validation-only floor;
// the constructor adds a geometry-derived reserve below as well.
constexpr std::size_t kUnreportedWorkspaceReserve = 64 * 1024;
// Backward-data/filter can retain two geometry-sized internal blocks at once.
constexpr std::size_t kBackwardGeometryReserveCopies = 2;

musa::dnn::Tensor::Type mudnn_data_type(flagdnnDataType_t data_type) {
  switch (data_type) {
    case FLAGDNN_DATA_INT32:
      throw std::invalid_argument(
          "INT32 is not supported by this validation adapter");

    case FLAGDNN_DATA_FLOAT32:
      return musa::dnn::Tensor::Type::FLOAT;
    case FLAGDNN_DATA_FLOAT16:
      return musa::dnn::Tensor::Type::HALF;
    case FLAGDNN_DATA_BFLOAT16:
      return musa::dnn::Tensor::Type::BFLOAT16;
    case FLAGDNN_DATA_BOOLEAN:
    case FLAGDNN_DATA_FP8_E8M0:
    case FLAGDNN_DATA_FP8_E4M3:
    case FLAGDNN_DATA_FP8_E5M2:
      break;
  }
  throw std::invalid_argument(
      "muDNN Convolution tensor type is unsupported");
}

std::size_t checked_multiply(std::size_t left,
                             std::size_t right,
                             std::string_view description) {
  if (right != 0 &&
      left > std::numeric_limits<std::size_t>::max() / right) {
    throw std::overflow_error(std::string(description) + " overflows");
  }
  return left * right;
}

std::size_t checked_add(std::size_t left,
                        std::size_t right,
                        std::string_view description) {
  if (left > std::numeric_limits<std::size_t>::max() - right) {
    throw std::overflow_error(std::string(description) + " overflows");
  }
  return left + right;
}

std::size_t align_up(std::size_t value, std::size_t alignment) {
  const std::size_t remainder = value % alignment;
  return remainder == 0
             ? value
             : checked_add(
                   value,
                   alignment - remainder,
                   "muDNN Convolution workspace alignment");
}

std::vector<std::int64_t> contiguous_strides(
    std::span<const std::int64_t> dimensions) {
  std::vector<std::int64_t> result(dimensions.size());
  std::int64_t stride = 1;
  for (std::size_t axis = dimensions.size(); axis != 0; --axis) {
    result[axis - 1] = stride;
    const std::int64_t dimension = dimensions[axis - 1];
    if (dimension >
        std::numeric_limits<std::int64_t>::max() / stride) {
      throw std::overflow_error(
          "muDNN Convolution dense stride overflows");
    }
    stride *= dimension;
  }
  return result;
}

void validate_tensor(const TensorDescriptor& tensor,
                     std::string_view name) {
  if (tensor.uid <= 0 || tensor.dimensions.size() < 3 ||
      tensor.dimensions.size() > 5 ||
      tensor.dimensions.size() != tensor.strides.size()) {
    throw std::invalid_argument(
        std::string(name) +
        " muDNN Convolution descriptor is invalid");
  }
  for (std::size_t axis = 0; axis < tensor.dimensions.size(); ++axis) {
    if (tensor.dimensions[axis] <= 0 || tensor.strides[axis] <= 0) {
      throw std::invalid_argument(
          std::string(name) +
          " muDNN Convolution dimensions/strides must be positive");
    }
  }
  static_cast<void>(mudnn_data_type(tensor.data_type));
  static_cast<void>(tensor_io::element_count(tensor));
  static_cast<void>(tensor_io::storage_element_count(tensor));
}

std::int64_t output_dimension(std::int64_t input,
                              std::int64_t filter,
                              std::int64_t pre_padding,
                              std::int64_t post_padding,
                              std::int64_t stride,
                              std::int64_t dilation) {
  const std::int64_t effective_filter =
      dilation * (filter - 1) + 1;
  return (input + pre_padding + post_padding - effective_filter) /
             stride +
         1;
}

void validate_mudnn_int(std::int64_t value,
                        bool allow_zero,
                        std::string_view description) {
  const std::int64_t minimum = allow_zero ? 0 : 1;
  if (value < minimum ||
      value > std::numeric_limits<int>::max()) {
    throw std::invalid_argument(
        std::string(description) + " is outside the muDNN int range");
  }
}

void validate_descriptor(
    const MudnnConvolutionDescriptor& descriptor) {
  validate_tensor(descriptor.image, "image");
  validate_tensor(descriptor.filter, "filter");
  validate_tensor(descriptor.result, "result");
  if (descriptor.image.uid == descriptor.filter.uid ||
      descriptor.image.uid == descriptor.result.uid ||
      descriptor.filter.uid == descriptor.result.uid ||
      descriptor.image.data_type != descriptor.filter.data_type ||
      descriptor.image.data_type != descriptor.result.data_type ||
      descriptor.image.dimensions.size() !=
          descriptor.filter.dimensions.size() ||
      descriptor.image.dimensions.size() !=
          descriptor.result.dimensions.size()) {
    throw std::invalid_argument(
        "muDNN Convolution tensor identities, types, or ranks are invalid");
  }
  switch (descriptor.direction) {
    case MudnnConvolutionDirection::kFprop:
      if (descriptor.mode !=
          MudnnConvolutionMode::kCrossCorrelation) {
        throw std::invalid_argument(
            "muDNN Convolution FProp requires cross-correlation mode");
      }
      break;
    case MudnnConvolutionDirection::kDgrad:
    case MudnnConvolutionDirection::kWgrad:
      break;
    default:
      throw std::invalid_argument(
          "muDNN Convolution direction is invalid");
  }
  switch (descriptor.mode) {
    case MudnnConvolutionMode::kCrossCorrelation:
    case MudnnConvolutionMode::kConvolution:
      break;
    default:
      throw std::invalid_argument("muDNN Convolution mode is invalid");
  }

  const std::size_t spatial_rank =
      descriptor.image.dimensions.size() - 2;
  if (descriptor.pre_padding.size() != spatial_rank ||
      descriptor.post_padding.size() != spatial_rank ||
      descriptor.stride.size() != spatial_rank ||
      descriptor.dilation.size() != spatial_rank) {
    throw std::invalid_argument(
        "muDNN Convolution spatial attributes have the wrong rank");
  }
  validate_mudnn_int(descriptor.groups, false,
                     "muDNN Convolution group count");
  const std::int64_t channels = descriptor.image.dimensions[1];
  const std::int64_t filters = descriptor.filter.dimensions[0];
  if (channels % descriptor.groups != 0 ||
      filters % descriptor.groups != 0 ||
      descriptor.filter.dimensions[1] !=
          channels / descriptor.groups ||
      descriptor.result.dimensions[0] !=
          descriptor.image.dimensions[0] ||
      descriptor.result.dimensions[1] != filters) {
    throw std::invalid_argument(
        "muDNN Convolution channel geometry is invalid");
  }
  for (std::size_t axis = 0; axis < spatial_rank; ++axis) {
    validate_mudnn_int(
        descriptor.pre_padding[axis], true,
        "muDNN Convolution pre-padding");
    validate_mudnn_int(
        descriptor.post_padding[axis], true,
        "muDNN Convolution post-padding");
    validate_mudnn_int(
        descriptor.stride[axis], false,
        "muDNN Convolution stride");
    validate_mudnn_int(
        descriptor.dilation[axis], false,
        "muDNN Convolution dilation");
    const std::int64_t expected = output_dimension(
        descriptor.image.dimensions[axis + 2],
        descriptor.filter.dimensions[axis + 2],
        descriptor.pre_padding[axis],
        descriptor.post_padding[axis],
        descriptor.stride[axis],
        descriptor.dilation[axis]);
    if (expected <= 0 ||
        descriptor.result.dimensions[axis + 2] != expected) {
      throw std::invalid_argument(
          "muDNN Convolution output geometry is invalid");
    }
  }
}

TensorDescriptor dense_tensor(const TensorDescriptor& tensor) {
  TensorDescriptor result = tensor;
  result.strides = contiguous_strides(result.dimensions);
  result.binding_byte_offset = 0;
  return result;
}

TensorDescriptor convolution_view(const TensorDescriptor& tensor) {
  if (tensor.dimensions.size() != 3) {
    return tensor;
  }
  TensorDescriptor result = tensor;
  result.dimensions = {
      tensor.dimensions[0],
      tensor.dimensions[1],
      1,
      tensor.dimensions[2],
  };
  result.strides = contiguous_strides(result.dimensions);
  result.binding_byte_offset = 0;
  return result;
}

TensorDescriptor dgrad_group_tensor_view(
    const TensorDescriptor& tensor,
    std::int64_t groups) {
  if (groups <= 1 || tensor.dimensions.size() < 2 ||
      tensor.dimensions[1] % groups != 0) {
    throw std::invalid_argument(
        "muDNN Dgrad group tensor view is invalid");
  }
  TensorDescriptor result = tensor;
  result.dimensions[0] = 1;
  result.dimensions[1] /= groups;
  result.strides = contiguous_strides(result.dimensions);
  return result;
}

TensorDescriptor dgrad_group_filter_view(
    const TensorDescriptor& filter,
    std::int64_t groups) {
  if (groups <= 1 || filter.dimensions.empty() ||
      filter.dimensions[0] % groups != 0) {
    throw std::invalid_argument(
        "muDNN Dgrad group filter view is invalid");
  }
  TensorDescriptor result = filter;
  result.dimensions[0] /= groups;
  result.strides = contiguous_strides(result.dimensions);
  return result;
}

std::size_t convolution_axis(std::size_t spatial_rank,
                             std::size_t axis) {
  return spatial_rank == 1 ? 3 : axis + 2;
}

TensorDescriptor padded_image(
    const TensorDescriptor& convolution_image,
    const MudnnConvolutionDescriptor& descriptor) {
  TensorDescriptor result = convolution_image;
  const std::size_t spatial_rank = descriptor.pre_padding.size();
  for (std::size_t axis = 0; axis < spatial_rank; ++axis) {
    const std::size_t target = convolution_axis(spatial_rank, axis);
    const std::int64_t padding =
        descriptor.pre_padding[axis] +
        descriptor.post_padding[axis];
    if (result.dimensions[target] >
        std::numeric_limits<std::int64_t>::max() - padding) {
      throw std::overflow_error(
          "muDNN Convolution padded dimension overflows");
    }
    result.dimensions[target] += padding;
  }
  result.strides = contiguous_strides(result.dimensions);
  return result;
}

bool has_explicit_padding(
    const MudnnConvolutionDescriptor& descriptor) {
  return std::any_of(
             descriptor.pre_padding.begin(),
             descriptor.pre_padding.end(),
             [](std::int64_t value) { return value != 0; }) ||
         std::any_of(
             descriptor.post_padding.begin(),
             descriptor.post_padding.end(),
             [](std::int64_t value) { return value != 0; });
}

std::vector<int> mudnn_padding(
    const MudnnConvolutionDescriptor& descriptor) {
  std::vector<int> result;
  const std::size_t spatial_rank = descriptor.pre_padding.size();
  result.reserve(spatial_rank == 1 ? 4 : spatial_rank * 2);
  for (std::size_t axis = spatial_rank; axis != 0; --axis) {
    result.push_back(
        static_cast<int>(descriptor.pre_padding[axis - 1]));
    result.push_back(
        static_cast<int>(descriptor.post_padding[axis - 1]));
  }
  if (spatial_rank == 1) {
    result.push_back(0);
    result.push_back(0);
  }
  return result;
}

std::vector<int> convolution_attribute(
    std::span<const std::int64_t> values,
    int one_dimensional_leading_value) {
  std::vector<int> result;
  if (values.size() == 1) {
    result.push_back(one_dimensional_leading_value);
  }
  result.reserve(values.size() + (values.size() == 1 ? 1 : 0));
  for (const std::int64_t value : values) {
    result.push_back(static_cast<int>(value));
  }
  return result;
}

std::size_t tensor_bytes(const TensorDescriptor& tensor) {
  return checked_multiply(
      tensor_io::element_count(tensor),
      tensor_io::data_type_size(tensor.data_type),
      "muDNN Convolution dense tensor bytes");
}

TensorDescriptor flattened_filter(
    const TensorDescriptor& dense_filter) {
  std::size_t volume = 1;
  for (std::size_t axis = 2;
       axis < dense_filter.dimensions.size(); ++axis) {
    volume = checked_multiply(
        volume,
        static_cast<std::size_t>(dense_filter.dimensions[axis]),
        "muDNN Convolution filter volume");
  }
  if (volume >
      static_cast<std::size_t>(
          std::numeric_limits<std::int64_t>::max())) {
    throw std::overflow_error(
        "muDNN Convolution flattened filter exceeds int64");
  }
  TensorDescriptor result = dense_filter;
  result.dimensions = {
      dense_filter.dimensions[0],
      dense_filter.dimensions[1],
      static_cast<std::int64_t>(volume),
  };
  result.strides = contiguous_strides(result.dimensions);
  return result;
}

musa::dnn::Tensor::Format convolution_format(
    const TensorDescriptor& tensor) {
  if (tensor.dimensions.size() == 4) {
    return musa::dnn::Tensor::Format::NCHW;
  }
  if (tensor.dimensions.size() == 5) {
    return musa::dnn::Tensor::Format::NCDHW;
  }
  throw std::invalid_argument(
      "muDNN Convolution tensor view must be rank 4 or 5");
}

void configure_permute_tensor(musa::dnn::Tensor& tensor,
                              const TensorDescriptor& descriptor,
                              void* pointer) {
  check_mudnn(
      tensor.SetAddr(pointer),
      "muDNN Tensor::SetAddr(Convolution bridge)");
  check_mudnn(
      tensor.SetType(mudnn_data_type(descriptor.data_type)),
      "muDNN Tensor::SetType(Convolution bridge)");
  check_mudnn(
      tensor.SetNdInfo(
          static_cast<int>(descriptor.dimensions.size()),
          descriptor.dimensions.data(),
          descriptor.strides.data()),
      "muDNN Tensor::SetNdInfo(Convolution bridge)");
}

void configure_convolution_tensor_descriptor(
    musa::dnn::Tensor& tensor,
    const TensorDescriptor& descriptor) {
  check_mudnn(
      tensor.SetType(mudnn_data_type(descriptor.data_type)),
      "muDNN Tensor::SetType(Convolution)");
  check_mudnn(
      tensor.SetFormat(convolution_format(descriptor)),
      "muDNN Tensor::SetFormat(Convolution)");
  check_mudnn(
      tensor.SetNdInfo(
          static_cast<int>(descriptor.dimensions.size()),
          descriptor.dimensions.data(),
          descriptor.strides.data()),
      "muDNN Tensor::SetNdInfo(Convolution)");
}

void configure_convolution_tensor(
    musa::dnn::Tensor& tensor,
    const TensorDescriptor& descriptor,
    void* pointer) {
  check_mudnn(
      tensor.SetAddr(pointer), "muDNN Tensor::SetAddr(Convolution)");
  configure_convolution_tensor_descriptor(tensor, descriptor);
}

void* binding_pointer(
    const TensorDescriptor& tensor,
    const std::unordered_map<std::int64_t, void*>& bindings) {
  const auto found = bindings.find(tensor.uid);
  if (found == bindings.end()) {
    throw std::invalid_argument(
        "muDNN Convolution binding UID is missing");
  }
  return found->second;
}

void* offset_pointer(void* pointer,
                     std::int64_t element_offset,
                     std::size_t element_size) {
  if (element_offset < 0 ||
      static_cast<std::uint64_t>(element_offset) >
          std::numeric_limits<std::size_t>::max() / element_size) {
    throw std::overflow_error(
        "muDNN Convolution element offset overflows");
  }
  const std::size_t byte_offset =
      static_cast<std::size_t>(element_offset) * element_size;
  const std::uintptr_t base =
      reinterpret_cast<std::uintptr_t>(pointer);
  if (base >
      std::numeric_limits<std::uintptr_t>::max() - byte_offset) {
    throw std::overflow_error(
        "muDNN Convolution pointer offset overflows");
  }
  return reinterpret_cast<void*>(base + byte_offset);
}

std::int64_t checked_element_offset(
    std::int64_t first_index,
    std::int64_t first_stride,
    std::int64_t second_index,
    std::int64_t second_stride,
    std::string_view description) {
  if (first_index < 0 || first_stride < 0 ||
      second_index < 0 || second_stride < 0 ||
      (first_stride != 0 &&
       first_index >
           std::numeric_limits<std::int64_t>::max() /
               first_stride) ||
      (second_stride != 0 &&
       second_index >
           std::numeric_limits<std::int64_t>::max() /
               second_stride)) {
    throw std::overflow_error(
        std::string(description) + " offset overflows");
  }
  const std::int64_t first = first_index * first_stride;
  const std::int64_t second = second_index * second_stride;
  if (first >
      std::numeric_limits<std::int64_t>::max() - second) {
    throw std::overflow_error(
        std::string(description) + " offset overflows");
  }
  return first + second;
}

}  // namespace

struct MudnnConvolutionOperation::Impl {
  explicit Impl(MudnnConvolutionDescriptor value)
      : descriptor(std::move(value)), handle(0) {
    validate_descriptor(descriptor);
    dense_image = dense_tensor(descriptor.image);
    dense_filter = dense_tensor(descriptor.filter);
    dense_result = dense_tensor(descriptor.result);
    convolution_image = convolution_view(dense_image);
    convolution_filter = convolution_view(dense_filter);
    convolution_result = convolution_view(dense_result);
    explicit_padding = has_explicit_padding(descriptor);
    padded_convolution_image =
        padded_image(convolution_image, descriptor);
    flattened_filter_view = flattened_filter(dense_filter);
    decompose_grouped_dgrad =
        descriptor.direction == MudnnConvolutionDirection::kDgrad &&
        descriptor.groups > 1;
    if (decompose_grouped_dgrad) {
      dgrad_group_image = dgrad_group_tensor_view(
          convolution_image, descriptor.groups);
      dgrad_group_padded_image = dgrad_group_tensor_view(
          padded_convolution_image, descriptor.groups);
      dgrad_group_filter = dgrad_group_filter_view(
          convolution_filter, descriptor.groups);
      dgrad_group_result = dgrad_group_tensor_view(
          convolution_result, descriptor.groups);
    }

    const std::vector<int> stride =
        convolution_attribute(descriptor.stride, 1);
    const std::vector<int> dilation =
        convolution_attribute(descriptor.dilation, 1);
    const std::vector<int> pad(stride.size(), 0);
    check_mudnn(
        convolution.SetNdInfo(
            static_cast<int>(pad.size()),
            pad.data(),
            stride.data(),
            dilation.data()),
        "muDNN Convolution::SetNdInfo");
    check_mudnn(
        convolution.SetGroups(
            decompose_grouped_dgrad
                ? 1
                : static_cast<int>(descriptor.groups)),
        "muDNN Convolution::SetGroups");
    check_mudnn(convolution.SetComputeMode(
                    (descriptor.image.data_type == FLAGDNN_DATA_FLOAT32 &&
                     descriptor.input_precision != 2)
                        ? musa::dnn::Convolution::ComputeMode::SCALAR
                        : musa::dnn::Convolution::ComputeMode::TENSOR),
                "muDNN Convolution::SetComputeMode");

    if (explicit_padding) {
      const std::vector<int> padding = mudnn_padding(descriptor);
      check_mudnn(
          pad_operation.SetMode(musa::dnn::Pad::Mode::CONSTANT),
          "muDNN Pad::SetMode(Convolution)");
      check_mudnn(
          pad_operation.SetValue(0.0),
          "muDNN Pad::SetValue(Convolution)");
      check_mudnn(
          pad_operation.SetPaddingInfo(
              static_cast<int>(padding.size()), padding.data()),
          "muDNN Pad::SetPaddingInfo(Convolution)");
    }

    reverse_filter =
        descriptor.mode == MudnnConvolutionMode::kConvolution;
    if (reverse_filter) {
      check_mudnn(
          gather.SetMode(
              musa::dnn::GatherX::Mode::GATHER_ELEMENTS),
          "muDNN GatherX::SetMode(Convolution filter flip)");
      check_mudnn(
          gather.SetAxis(2),
          "muDNN GatherX::SetAxis(Convolution filter flip)");
      const std::size_t index_count =
          tensor_io::element_count(flattened_filter_view);
      const std::size_t index_bytes = checked_multiply(
          index_count,
          sizeof(std::int64_t),
          "muDNN Convolution filter index bytes");
      reverse_indices =
          std::make_unique<DeviceBuffer>(index_bytes, 256);
      const std::size_t volume = static_cast<std::size_t>(
          flattened_filter_view.dimensions[2]);
      std::vector<std::int64_t> host_indices(index_count);
      for (std::size_t index = 0; index < index_count; ++index) {
        host_indices[index] = static_cast<std::int64_t>(
            volume - 1 - (index % volume));
      }
      check_musa(
          musaMemcpy(
              reverse_indices->get(),
              host_indices.data(),
              index_bytes,
              musaMemcpyHostToDevice),
          "musaMemcpy(Convolution filter indices)");
    }

    musa::dnn::Tensor image;
    musa::dnn::Tensor filter;
    musa::dnn::Tensor result;
    configure_convolution_tensor_descriptor(
        image,
        decompose_grouped_dgrad
            ? (explicit_padding ? dgrad_group_padded_image
                                : dgrad_group_image)
            : (explicit_padding ? padded_convolution_image
                                : convolution_image));
    configure_convolution_tensor_descriptor(
        filter,
        decompose_grouped_dgrad ? dgrad_group_filter
                                : convolution_filter);
    configure_convolution_tensor_descriptor(
        result,
        decompose_grouped_dgrad ? dgrad_group_result
                                : convolution_result);

    switch (descriptor.direction) {
      case MudnnConvolutionDirection::kFprop:
        check_mudnn(
            convolution.GetRecommendForwardAlgorithm(
                handle, forward_algorithm, result, image, filter),
            "muDNN Convolution::GetRecommendForwardAlgorithm");
        // muDNN 3.1.5 can recommend DIRECT for grouped IEEE FP32 even
        // though Run rejects that configuration. IMPLICIT_GEMM supports it.
        if (descriptor.image.data_type == FLAGDNN_DATA_FLOAT32 &&
            descriptor.input_precision != 2 && descriptor.groups > 1 &&
            forward_algorithm == musa::dnn::Convolution::Algorithm::DIRECT) {
          forward_algorithm =
              musa::dnn::Convolution::Algorithm::IMPLICIT_GEMM;
        }
        check_mudnn(
            convolution.GetForwardWorkspaceSize(
                handle,
                scratch_bytes,
                result,
                image,
                filter,
                forward_algorithm),
            "muDNN Convolution::GetForwardWorkspaceSize");
        break;
      case MudnnConvolutionDirection::kDgrad:
        check_mudnn(
            convolution.GetRecommendBackwardDataAlgorithm(
                handle, dgrad_algorithm, image, result, filter),
            "muDNN Convolution::GetRecommendBackwardDataAlgorithm");
        check_mudnn(
            convolution.GetBackwardDataWorkspaceSize(
                handle,
                scratch_bytes,
                image,
                result,
                filter,
                dgrad_algorithm),
            "muDNN Convolution::GetBackwardDataWorkspaceSize");
        break;
      case MudnnConvolutionDirection::kWgrad:
        check_mudnn(
            convolution.GetRecommendBackwardFilterAlgorithm(
                handle, wgrad_algorithm, filter, image, result),
            "muDNN Convolution::GetRecommendBackwardFilterAlgorithm");
        check_mudnn(
            convolution.GetBackwardFilterWorkspaceSize(
                handle,
                scratch_bytes,
                filter,
                image,
                result,
                wgrad_algorithm),
            "muDNN Convolution::GetBackwardFilterWorkspaceSize");
        break;
    }
    std::size_t geometry_reserve = checked_add(
        tensor_bytes(dense_image),
        tensor_bytes(dense_filter),
        "muDNN Convolution geometry reserve");
    geometry_reserve = checked_add(
        geometry_reserve,
        tensor_bytes(dense_result),
        "muDNN Convolution geometry reserve");
    if (explicit_padding) {
      const std::size_t padded_geometry = checked_add(
          tensor_bytes(padded_convolution_image),
          tensor_bytes(dense_filter),
          "muDNN Convolution padded geometry reserve");
      geometry_reserve = std::max(
          geometry_reserve,
          checked_add(padded_geometry,
                      tensor_bytes(dense_result),
                      "muDNN Convolution padded geometry reserve"));
    }
    if (descriptor.direction != MudnnConvolutionDirection::kFprop) {
      geometry_reserve = checked_multiply(
          geometry_reserve,
          kBackwardGeometryReserveCopies,
          "muDNN Convolution backward geometry reserve");
    }
    geometry_reserve =
        std::max(geometry_reserve, kUnreportedWorkspaceReserve);
    scratch_bytes = checked_add(
        align_up(scratch_bytes, kWorkspaceAlignment),
        align_up(geometry_reserve, kWorkspaceAlignment),
        "muDNN Convolution scratch workspace with SDK reserve");

    image_offset = 0;
    filter_offset = align_up(
        checked_add(
            image_offset,
            tensor_bytes(dense_image),
            "muDNN Convolution image workspace"),
        kWorkspaceAlignment);
    result_offset = align_up(
        checked_add(
            filter_offset,
            tensor_bytes(dense_filter),
            "muDNN Convolution filter workspace"),
        kWorkspaceAlignment);
    std::size_t next = align_up(
        checked_add(
            result_offset,
            tensor_bytes(dense_result),
            "muDNN Convolution result workspace"),
        kWorkspaceAlignment);
    if (explicit_padding) {
      padded_image_offset = next;
      next = align_up(
          checked_add(
              next,
              tensor_bytes(padded_convolution_image),
              "muDNN Convolution padded image workspace"),
          kWorkspaceAlignment);
    }
    if (reverse_filter) {
      reversed_filter_offset = next;
      next = align_up(
          checked_add(
              next,
              tensor_bytes(dense_filter),
              "muDNN Convolution reversed filter workspace"),
          kWorkspaceAlignment);
    }
    scratch_offset = next;
    workspace_bytes = checked_add(
        scratch_offset,
        scratch_bytes,
        "muDNN Convolution total workspace");
  }

  MudnnConvolutionDescriptor descriptor;
  TensorDescriptor dense_image;
  TensorDescriptor dense_filter;
  TensorDescriptor dense_result;
  TensorDescriptor convolution_image;
  TensorDescriptor convolution_filter;
  TensorDescriptor convolution_result;
  TensorDescriptor padded_convolution_image;
  TensorDescriptor flattened_filter_view;
  TensorDescriptor dgrad_group_image;
  TensorDescriptor dgrad_group_padded_image;
  TensorDescriptor dgrad_group_filter;
  TensorDescriptor dgrad_group_result;
  musa::dnn::Handle handle;
  musa::dnn::Convolution convolution;
  musa::dnn::Convolution::Algorithm forward_algorithm =
      musa::dnn::Convolution::Algorithm::IMPLICIT_GEMM;
  musa::dnn::Convolution::AlgorithmBwdData dgrad_algorithm =
      musa::dnn::Convolution::AlgorithmBwdData::IMPLICIT_GEMM;
  musa::dnn::Convolution::AlgorithmBwdFilter wgrad_algorithm =
      musa::dnn::Convolution::AlgorithmBwdFilter::IMPLICIT_GEMM;
  musa::dnn::Permute permute;
  musa::dnn::Pad pad_operation;
  musa::dnn::GatherX gather;
  std::unique_ptr<DeviceBuffer> reverse_indices;
  bool explicit_padding = false;
  bool reverse_filter = false;
  bool decompose_grouped_dgrad = false;
  std::size_t image_offset = 0;
  std::size_t filter_offset = 0;
  std::size_t result_offset = 0;
  std::size_t padded_image_offset = 0;
  std::size_t reversed_filter_offset = 0;
  std::size_t scratch_offset = 0;
  std::size_t scratch_bytes = 0;
  std::size_t workspace_bytes = 0;
};

MudnnConvolutionOperation::MudnnConvolutionOperation(
    MudnnConvolutionDescriptor descriptor)
    : implementation_(
          std::make_unique<Impl>(std::move(descriptor))) {}

MudnnConvolutionOperation::~MudnnConvolutionOperation() = default;

std::size_t MudnnConvolutionOperation::workspace_size() const noexcept {
  return implementation_->workspace_bytes;
}

void MudnnConvolutionOperation::execute(
    std::span<const flagdnnBinding_t> raw_bindings,
    void* workspace,
    std::size_t workspace_size,
    flagdnnStream_t stream) {
  Impl& state = *implementation_;
  if (stream == nullptr || raw_bindings.size() != 3 ||
      workspace_size < state.workspace_bytes ||
      (state.workspace_bytes != 0 && workspace == nullptr)) {
    throw std::invalid_argument(
        "muDNN Convolution execute arguments are invalid");
  }
  std::unordered_map<std::int64_t, void*> bindings;
  bindings.reserve(raw_bindings.size());
  for (const flagdnnBinding_t& binding : raw_bindings) {
    if (binding.device_pointer == nullptr ||
        !bindings.emplace(binding.uid, binding.device_pointer).second) {
      throw std::invalid_argument(
          "muDNN Convolution binding is invalid");
    }
  }

  const musaStream_t musa_stream =
      reinterpret_cast<musaStream_t>(stream);
  check_mudnn(
      state.handle.SetStream(musa_stream),
      "muDNN Handle::SetStream(Convolution)");
  if (state.handle.GetStream() != musa_stream) {
    throw std::runtime_error(
        "muDNN Convolution did not retain the caller stream");
  }

  auto* workspace_bytes = static_cast<std::byte*>(workspace);
  void* const dense_image_pointer =
      workspace_bytes + state.image_offset;
  void* const dense_filter_pointer =
      workspace_bytes + state.filter_offset;
  void* const dense_result_pointer =
      workspace_bytes + state.result_offset;
  void* const padded_image_pointer =
      state.explicit_padding
          ? workspace_bytes + state.padded_image_offset
          : dense_image_pointer;
  void* const reversed_filter_pointer =
      state.reverse_filter
          ? workspace_bytes + state.reversed_filter_offset
          : dense_filter_pointer;
  void* const scratch =
      state.scratch_bytes == 0
          ? nullptr
          : workspace_bytes + state.scratch_offset;

  void* const image_binding =
      binding_pointer(state.descriptor.image, bindings);
  void* const filter_binding =
      binding_pointer(state.descriptor.filter, bindings);
  void* const result_binding =
      binding_pointer(state.descriptor.result, bindings);

  const auto permute = [&](const TensorDescriptor& output_descriptor,
                           void* output_pointer,
                           const TensorDescriptor& input_descriptor,
                           void* input_pointer,
                           std::string_view operation) {
    musa::dnn::Tensor output;
    musa::dnn::Tensor input;
    configure_permute_tensor(
        output, output_descriptor, output_pointer);
    configure_permute_tensor(
        input, input_descriptor, input_pointer);
    check_mudnn(
        state.permute.Run(state.handle, output, input), operation);
  };
  const auto reverse_filter = [&](void* output_pointer,
                                  void* input_pointer) {
    if (state.reverse_indices == nullptr) {
      throw std::logic_error(
          "muDNN Convolution filter indices are unavailable");
    }
    musa::dnn::Tensor output;
    musa::dnn::Tensor input;
    musa::dnn::Tensor indices;
    configure_permute_tensor(
        output, state.flattened_filter_view, output_pointer);
    configure_permute_tensor(
        input, state.flattened_filter_view, input_pointer);
    check_mudnn(
        indices.SetAddr(state.reverse_indices->get()),
        "muDNN Tensor::SetAddr(Convolution filter indices)");
    check_mudnn(
        indices.SetType(musa::dnn::Tensor::Type::INT64),
        "muDNN Tensor::SetType(Convolution filter indices)");
    check_mudnn(
        indices.SetNdInfo(
            static_cast<int>(
                state.flattened_filter_view.dimensions.size()),
            state.flattened_filter_view.dimensions.data(),
            state.flattened_filter_view.strides.data()),
        "muDNN Tensor::SetNdInfo(Convolution filter indices)");
    check_mudnn(
        state.gather.Run(state.handle, output, indices, input),
        "muDNN GatherX::Run(Convolution filter flip)");
  };
  const auto pad_image = [&] {
    musa::dnn::Tensor output;
    musa::dnn::Tensor input;
    configure_convolution_tensor(
        output,
        state.padded_convolution_image,
        padded_image_pointer);
    configure_convolution_tensor(
        input,
        state.convolution_image,
        dense_image_pointer);
    check_mudnn(
        state.pad_operation.Run(state.handle, output, input),
        "muDNN Pad::Run(Convolution)");
  };

  switch (state.descriptor.direction) {
    case MudnnConvolutionDirection::kFprop:
      permute(
          state.dense_image,
          dense_image_pointer,
          state.descriptor.image,
          image_binding,
          "muDNN Permute::Run(Convolution gather image)");
      permute(
          state.dense_filter,
          dense_filter_pointer,
          state.descriptor.filter,
          filter_binding,
          "muDNN Permute::Run(Convolution gather filter)");
      if (state.explicit_padding) {
        pad_image();
      }
      break;
    case MudnnConvolutionDirection::kDgrad:
      permute(
          state.dense_result,
          dense_result_pointer,
          state.descriptor.result,
          result_binding,
          "muDNN Permute::Run(Convolution gather loss)");
      permute(
          state.dense_filter,
          dense_filter_pointer,
          state.descriptor.filter,
          filter_binding,
          "muDNN Permute::Run(Convolution gather filter)");
      if (state.reverse_filter) {
        reverse_filter(
            reversed_filter_pointer, dense_filter_pointer);
      }
      break;
    case MudnnConvolutionDirection::kWgrad:
      permute(
          state.dense_result,
          dense_result_pointer,
          state.descriptor.result,
          result_binding,
          "muDNN Permute::Run(Convolution gather loss)");
      permute(
          state.dense_image,
          dense_image_pointer,
          state.descriptor.image,
          image_binding,
          "muDNN Permute::Run(Convolution gather image)");
      if (state.explicit_padding) {
        pad_image();
      }
      break;
  }

  musa::dnn::Tensor image;
  musa::dnn::Tensor filter;
  musa::dnn::Tensor result;
  configure_convolution_tensor(
      image,
      state.explicit_padding
          ? state.padded_convolution_image
          : state.convolution_image,
      padded_image_pointer);
  configure_convolution_tensor(
      filter,
      state.convolution_filter,
      state.descriptor.direction ==
                  MudnnConvolutionDirection::kDgrad &&
              state.reverse_filter
          ? reversed_filter_pointer
          : (state.descriptor.direction ==
                         MudnnConvolutionDirection::kWgrad &&
                     state.reverse_filter
                 ? reversed_filter_pointer
                 : dense_filter_pointer));
  configure_convolution_tensor(
      result, state.convolution_result, dense_result_pointer);

  const musa::dnn::MemoryMaintainer maintainer =
      make_mudnn_workspace_maintainer(
          scratch, state.scratch_bytes, "muDNN Convolution");
  switch (state.descriptor.direction) {
    case MudnnConvolutionDirection::kFprop:
      check_mudnn(
          state.convolution.Run(
              state.handle,
              result,
              image,
              filter,
              state.forward_algorithm,
              maintainer),
          "muDNN Convolution::Run");
      permute(
          state.descriptor.result,
          result_binding,
          state.dense_result,
          dense_result_pointer,
          "muDNN Permute::Run(Convolution scatter result)");
      break;
    case MudnnConvolutionDirection::kDgrad:
      if (!state.decompose_grouped_dgrad) {
        check_mudnn(
            state.convolution.RunBwdData(
                state.handle,
                image,
                result,
                filter,
                state.dgrad_algorithm,
                maintainer),
            "muDNN Convolution::RunBwdData");
      } else {
        const TensorDescriptor& full_image =
            state.explicit_padding
                ? state.padded_convolution_image
                : state.convolution_image;
        const TensorDescriptor& group_image =
            state.explicit_padding
                ? state.dgrad_group_padded_image
                : state.dgrad_group_image;
        const std::int64_t input_channels_per_group =
            state.dgrad_group_image.dimensions[1];
        const std::int64_t output_channels_per_group =
            state.dgrad_group_result.dimensions[1];
        const std::size_t element_size =
            tensor_io::data_type_size(
                state.descriptor.image.data_type);
        void* const full_filter_pointer =
            state.reverse_filter ? reversed_filter_pointer
                                 : dense_filter_pointer;
        for (std::int64_t batch = 0;
             batch < state.descriptor.image.dimensions[0]; ++batch) {
          for (std::int64_t group = 0;
               group < state.descriptor.groups; ++group) {
            const std::int64_t image_offset =
                checked_element_offset(
                    batch,
                    full_image.strides[0],
                    group * input_channels_per_group,
                    full_image.strides[1],
                    "muDNN grouped Dgrad image");
            const std::int64_t result_offset =
                checked_element_offset(
                    batch,
                    state.convolution_result.strides[0],
                    group * output_channels_per_group,
                    state.convolution_result.strides[1],
                    "muDNN grouped Dgrad loss");
            const std::int64_t filter_offset =
                checked_element_offset(
                    group * output_channels_per_group,
                    state.convolution_filter.strides[0],
                    0,
                    0,
                    "muDNN grouped Dgrad filter");
            musa::dnn::Tensor group_image_tensor;
            musa::dnn::Tensor group_filter_tensor;
            musa::dnn::Tensor group_result_tensor;
            configure_convolution_tensor(
                group_image_tensor,
                group_image,
                offset_pointer(
                    padded_image_pointer,
                    image_offset,
                    element_size));
            configure_convolution_tensor(
                group_filter_tensor,
                state.dgrad_group_filter,
                offset_pointer(
                    full_filter_pointer,
                    filter_offset,
                    element_size));
            configure_convolution_tensor(
                group_result_tensor,
                state.dgrad_group_result,
                offset_pointer(
                    dense_result_pointer,
                    result_offset,
                    element_size));
            const musa::dnn::MemoryMaintainer group_maintainer =
                make_mudnn_workspace_maintainer(
                    scratch,
                    state.scratch_bytes,
                    "muDNN Convolution grouped Dgrad");
            check_mudnn(
                state.convolution.RunBwdData(
                    state.handle,
                    group_image_tensor,
                    group_result_tensor,
                    group_filter_tensor,
                    state.dgrad_algorithm,
                    group_maintainer),
                "muDNN Convolution::RunBwdData(group slice)");
          }
        }
      }
      if (state.explicit_padding) {
        TensorDescriptor cropped = state.convolution_image;
        cropped.strides =
            state.padded_convolution_image.strides;
        std::int64_t crop_offset = 0;
        const std::size_t spatial_rank =
            state.descriptor.pre_padding.size();
        for (std::size_t axis = 0; axis < spatial_rank; ++axis) {
          const std::size_t target =
              convolution_axis(spatial_rank, axis);
          const std::int64_t padding =
              state.descriptor.pre_padding[axis];
          const std::int64_t stride =
              cropped.strides[target];
          if (padding != 0 &&
              stride >
                  (std::numeric_limits<std::int64_t>::max() -
                   crop_offset) /
                      padding) {
            throw std::overflow_error(
                "muDNN Convolution crop offset overflows");
          }
          crop_offset += padding * stride;
        }
        const std::size_t element_size =
            tensor_io::data_type_size(
                state.descriptor.image.data_type);
        permute(
            state.convolution_image,
            dense_image_pointer,
            cropped,
            offset_pointer(
                padded_image_pointer, crop_offset, element_size),
            "muDNN Permute::Run(Convolution crop dgrad)");
      }
      permute(
          state.descriptor.image,
          image_binding,
          state.dense_image,
          dense_image_pointer,
          "muDNN Permute::Run(Convolution scatter dgrad)");
      break;
    case MudnnConvolutionDirection::kWgrad:
      check_mudnn(
          state.convolution.RunBwdFilter(
              state.handle,
              filter,
              image,
              result,
              state.wgrad_algorithm,
              maintainer),
          "muDNN Convolution::RunBwdFilter");
      if (state.reverse_filter) {
        reverse_filter(
            dense_filter_pointer, reversed_filter_pointer);
      }
      permute(
          state.descriptor.filter,
          filter_binding,
          state.dense_filter,
          dense_filter_pointer,
          "muDNN Permute::Run(Convolution scatter wgrad)");
      break;
  }
}

}  // namespace flagdnn::validation::mthreads
