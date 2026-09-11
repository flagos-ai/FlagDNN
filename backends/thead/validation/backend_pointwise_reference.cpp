// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "backend_pointwise_reference.hpp"

#include "acdnn_reference.hpp"
#include "numeric_types.hpp"

#include <acdnn.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace flagdnn::validation::thead {
namespace {

class BackendDescriptor final {
 public:
  explicit BackendDescriptor(acdnnBackendDescriptorType_t type) {
    check_acdnn(acdnnBackendCreateDescriptor(type, &descriptor_),
                "acdnnBackendCreateDescriptor");
  }

  ~BackendDescriptor() {
    if (descriptor_ != nullptr) {
      (void)acdnnBackendDestroyDescriptor(descriptor_);
    }
  }

  BackendDescriptor(const BackendDescriptor &) = delete;
  BackendDescriptor &operator=(const BackendDescriptor &) = delete;

  [[nodiscard]] acdnnBackendDescriptor_t get() const noexcept {
    return descriptor_;
  }

  void set(acdnnBackendAttributeName_t name,
           acdnnBackendAttributeType_t type, std::int64_t count,
           const void *values, std::string_view description) {
    check_acdnn(acdnnBackendSetAttribute(descriptor_, name, type, count,
                                        values),
                description);
  }

  void finalize(std::string_view description) {
    check_acdnn(acdnnBackendFinalize(descriptor_), description);
  }

 private:
  acdnnBackendDescriptor_t descriptor_ = nullptr;
};

acdnnDataType_t backend_data_type(flagdnnDataType_t data_type,
                                   bool fp8_storage_bytes = false) {
  switch (data_type) {
    case FLAGDNN_DATA_FLOAT32:
      return ACDNN_DATA_FLOAT;
    case FLAGDNN_DATA_FLOAT16:
      return ACDNN_DATA_HALF;
    case FLAGDNN_DATA_BFLOAT16:
      return ACDNN_DATA_BF16;
    case FLAGDNN_DATA_BOOLEAN:
      return ACDNN_DATA_BOOL;
    case FLAGDNN_DATA_FP8_E4M3:
    case FLAGDNN_DATA_FP8_E5M2:
      if (fp8_storage_bytes) return ACDNN_DATA_INT8;
      break;
  }
  throw std::invalid_argument(
      "qualified acDNN backend pointwise reference data type is unsupported");
}

bool is_floating(flagdnnDataType_t data_type) {
  return data_type == FLAGDNN_DATA_FLOAT32 ||
         data_type == FLAGDNN_DATA_FLOAT16 ||
         data_type == FLAGDNN_DATA_BFLOAT16;
}

bool is_comparison_mode(acdnnPointwiseMode_t mode) {
  return mode == ACDNN_POINTWISE_CMP_EQ ||
         mode == ACDNN_POINTWISE_CMP_NEQ ||
         mode == ACDNN_POINTWISE_CMP_GT ||
         mode == ACDNN_POINTWISE_CMP_GE ||
         mode == ACDNN_POINTWISE_CMP_LT ||
         mode == ACDNN_POINTWISE_CMP_LE;
}

bool is_logical_mode(acdnnPointwiseMode_t mode) {
  return mode == ACDNN_POINTWISE_LOGICAL_NOT ||
         mode == ACDNN_POINTWISE_LOGICAL_AND ||
         mode == ACDNN_POINTWISE_LOGICAL_OR;
}

bool valid_tensor_types(
    const BackendPointwiseReferenceSpecification &specification) {
  if (specification.fp8_storage_bytes) {
    const auto fp8 = [](flagdnnDataType_t type) {
      return type == FLAGDNN_DATA_FP8_E4M3 || type == FLAGDNN_DATA_FP8_E5M2;
    };
    return specification.mode == ACDNN_POINTWISE_IDENTITY_FWD &&
           specification.inputs.size() == 1 &&
           !specification.constant_one_numerator &&
           specification.alpha1 == 1.0F && specification.alpha2 == 1.0F &&
           ((fp8(specification.inputs[0].data_type) &&
             specification.output.data_type == FLAGDNN_DATA_FLOAT32) ||
            (specification.inputs[0].data_type == FLAGDNN_DATA_FLOAT32 &&
             fp8(specification.output.data_type)));
  }
  if (specification.mode == ACDNN_POINTWISE_IDENTITY_FWD) {
    return specification.inputs.size() == 1 &&
           (is_floating(specification.inputs[0].data_type) ||
            specification.inputs[0].data_type == FLAGDNN_DATA_BOOLEAN) &&
           (is_floating(specification.output.data_type) ||
            specification.output.data_type == FLAGDNN_DATA_BOOLEAN);
  }
  if (is_comparison_mode(specification.mode)) {
    return specification.inputs.size() == 2 &&
           is_floating(specification.inputs[0].data_type) &&
           specification.inputs[1].data_type ==
               specification.inputs[0].data_type &&
           specification.output.data_type == FLAGDNN_DATA_BOOLEAN;
  }
  if (is_logical_mode(specification.mode)) {
    const std::size_t expected_inputs =
        specification.mode == ACDNN_POINTWISE_LOGICAL_NOT ? 1U : 2U;
    return specification.inputs.size() == expected_inputs &&
           specification.output.data_type == FLAGDNN_DATA_BOOLEAN &&
           std::all_of(
               specification.inputs.begin(), specification.inputs.end(),
               [](const flagdnn::testing::TestTensor &input) {
                 return input.data_type == FLAGDNN_DATA_BOOLEAN;
               });
  }
  return std::all_of(
      specification.inputs.begin(), specification.inputs.end(),
      [&specification](const flagdnn::testing::TestTensor &input) {
        return input.data_type == specification.output.data_type;
      });
}

std::vector<std::int64_t> checked_metadata(
    std::span<const std::int64_t> values, std::string_view description) {
  std::vector<std::int64_t> result(values.begin(), values.end());
  if (result.empty() || result.size() > 8 ||
      std::any_of(result.begin(), result.end(), [](std::int64_t value) {
        return value <= 0;
      })) {
    throw std::invalid_argument(std::string(description) + " is invalid");
  }
  return result;
}

std::size_t storage_element_count(
    const flagdnn::testing::TestTensor &tensor) {
  if (tensor.dimensions.empty() ||
      tensor.dimensions.size() != tensor.strides.size()) {
    throw std::invalid_argument(
        "constant-numerator acDNN pointwise geometry is invalid");
  }
  std::size_t maximum_offset = 0;
  for (std::size_t axis = 0; axis < tensor.dimensions.size(); ++axis) {
    const std::int64_t dimension = tensor.dimensions[axis];
    const std::int64_t stride = tensor.strides[axis];
    if (dimension <= 0 || stride <= 0) {
      throw std::invalid_argument(
          "constant-numerator acDNN pointwise geometry is invalid");
    }
    const std::size_t extent = static_cast<std::size_t>(dimension - 1);
    const std::size_t physical_stride = static_cast<std::size_t>(stride);
    if (physical_stride != 0 &&
        extent > std::numeric_limits<std::size_t>::max() / physical_stride) {
      throw std::invalid_argument(
          "constant-numerator acDNN pointwise storage is too large");
    }
    const std::size_t contribution = extent * physical_stride;
    if (maximum_offset >
        std::numeric_limits<std::size_t>::max() - contribution) {
      throw std::invalid_argument(
          "constant-numerator acDNN pointwise storage is too large");
    }
    maximum_offset += contribution;
  }
  if (maximum_offset == std::numeric_limits<std::size_t>::max()) {
    throw std::invalid_argument(
        "constant-numerator acDNN pointwise storage is too large");
  }
  return maximum_offset + 1;
}

void build_tensor_descriptor(BackendDescriptor &descriptor,
                             const flagdnn::testing::TestTensor &tensor,
                             bool fp8_storage_bytes = false) {
  const acdnnDataType_t data_type =
      backend_data_type(tensor.data_type, fp8_storage_bytes);
  const std::vector<std::int64_t> dimensions =
      checked_metadata(tensor.dimensions, "backend tensor dimensions");
  const std::vector<std::int64_t> strides =
      checked_metadata(tensor.strides, "backend tensor strides");
  if (dimensions.size() != strides.size() || tensor.uid <= 0) {
    throw std::invalid_argument("backend tensor metadata is inconsistent");
  }
  std::int64_t alignment = 16;
  if (tensor.binding_byte_offset != 0) {
    alignment = 1;
    while (alignment < 16 &&
           tensor.binding_byte_offset %
                   (static_cast<std::size_t>(alignment) * 2U) ==
               0) {
      alignment *= 2;
    }
  }
  if (alignment < static_cast<std::int64_t>(element_size(tensor.data_type))) {
    throw std::invalid_argument(
        "backend tensor binding alignment is below its scalar size");
  }
  descriptor.set(ACDNN_ATTR_TENSOR_DATA_TYPE, ACDNN_TYPE_DATA_TYPE, 1,
                 &data_type, "acdnnBackendSetAttribute(tensor data type)");
  descriptor.set(ACDNN_ATTR_TENSOR_DIMENSIONS, ACDNN_TYPE_INT64,
                 static_cast<std::int64_t>(dimensions.size()),
                 dimensions.data(),
                 "acdnnBackendSetAttribute(tensor dimensions)");
  descriptor.set(ACDNN_ATTR_TENSOR_STRIDES, ACDNN_TYPE_INT64,
                 static_cast<std::int64_t>(strides.size()), strides.data(),
                 "acdnnBackendSetAttribute(tensor strides)");
  descriptor.set(ACDNN_ATTR_TENSOR_UNIQUE_ID, ACDNN_TYPE_INT64, 1,
                 &tensor.uid,
                 "acdnnBackendSetAttribute(tensor unique id)");
  descriptor.set(ACDNN_ATTR_TENSOR_BYTE_ALIGNMENT, ACDNN_TYPE_INT64, 1,
                 &alignment,
                 "acdnnBackendSetAttribute(tensor alignment)");
  descriptor.finalize("acdnnBackendFinalize(tensor)");
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
          "acDNN backend pointwise binding is duplicate or null");
    }
    result = binding.device_pointer;
  }
  if (result == nullptr) {
    throw std::invalid_argument(
        "acDNN backend pointwise binding is missing");
  }
  return result;
}

class AcdnnBackendPointwiseSegment final
    : public flagdnn::testing::TestExecutable {
 public:
  explicit AcdnnBackendPointwiseSegment(
      BackendPointwiseReferenceSpecification specification)
      : specification_(std::move(specification)),
        pointwise_(ACDNN_BACKEND_POINTWISE_DESCRIPTOR),
        input_(ACDNN_BACKEND_TENSOR_DESCRIPTOR),
        second_input_(ACDNN_BACKEND_TENSOR_DESCRIPTOR),
        output_(ACDNN_BACKEND_TENSOR_DESCRIPTOR),
        operation_(ACDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR),
        operation_graph_(ACDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR),
        heuristics_(ACDNN_BACKEND_ENGINEHEUR_DESCRIPTOR),
        engine_config_(ACDNN_BACKEND_ENGINECFG_DESCRIPTOR),
        plan_(ACDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR) {
    if ((specification_.inputs.size() != 1 &&
         specification_.inputs.size() != 2) ||
        (specification_.constant_one_numerator &&
         (specification_.mode != ACDNN_POINTWISE_DIV ||
          specification_.inputs.size() != 1)) ||
        !std::isfinite(specification_.alpha1) ||
        !std::isfinite(specification_.alpha2) ||
        !valid_tensor_types(specification_) ||
        std::any_of(
            specification_.inputs.begin(), specification_.inputs.end(),
            [this](const flagdnn::testing::TestTensor &input) {
              return input.dimensions != specification_.output.dimensions;
            })) {
      throw std::invalid_argument(
          "qualified acDNN backend pointwise geometry is invalid");
    }
    if (specification_.constant_one_numerator) {
      constexpr std::int64_t kConstantNumeratorUid =
          std::numeric_limits<std::int64_t>::max() - 1;
      if (specification_.inputs[0].uid == kConstantNumeratorUid ||
          specification_.output.uid == kConstantNumeratorUid) {
        throw std::invalid_argument(
            "constant-numerator acDNN pointwise UID collides with graph");
      }
      flagdnn::testing::TestTensor numerator = specification_.inputs[0];
      numerator.uid = kConstantNumeratorUid;
      constant_numerator_uid_ = numerator.uid;
      build_tensor_descriptor(input_, numerator);
      build_tensor_descriptor(second_input_, specification_.inputs[0]);
      const std::size_t elements = storage_element_count(numerator);
      const std::size_t scalar_bytes = element_size(numerator.data_type);
      if (elements >
          std::numeric_limits<std::size_t>::max() / scalar_bytes) {
        throw std::invalid_argument(
            "constant-numerator acDNN pointwise allocation is too large");
      }
      constant_numerator_ = DeviceBuffer(elements * scalar_bytes);
      if (numerator.data_type == FLAGDNN_DATA_FLOAT32) {
        check_driver(cuMemsetD32(constant_numerator_.address(), 0x3f800000U,
                                elements),
                     "cuMemsetD32(acDNN reciprocal numerator)");
      } else {
        const unsigned short one =
            numerator.data_type == FLAGDNN_DATA_FLOAT16 ? 0x3c00U : 0x3f80U;
        check_driver(cuMemsetD16(constant_numerator_.address(), one, elements),
                     "cuMemsetD16(acDNN reciprocal numerator)");
      }
    } else {
      build_tensor_descriptor(input_, specification_.inputs[0],
                              specification_.fp8_storage_bytes);
    }
    if (!specification_.constant_one_numerator &&
        specification_.inputs.size() == 2) {
      build_tensor_descriptor(second_input_, specification_.inputs[1]);
    }
    build_tensor_descriptor(output_, specification_.output,
                            specification_.fp8_storage_bytes);

    const acdnnDataType_t math_precision =
        is_comparison_mode(specification_.mode) ||
                is_logical_mode(specification_.mode)
            ? ACDNN_DATA_BOOL
            : ACDNN_DATA_FLOAT;
    const acdnnNanPropagation_t nan_policy = ACDNN_NOT_PROPAGATE_NAN;
    pointwise_.set(ACDNN_ATTR_POINTWISE_MODE, ACDNN_TYPE_POINTWISE_MODE, 1,
                   &specification_.mode,
                   "acdnnBackendSetAttribute(pointwise mode)");
    pointwise_.set(ACDNN_ATTR_POINTWISE_MATH_PREC, ACDNN_TYPE_DATA_TYPE, 1,
                   &math_precision,
                   "acdnnBackendSetAttribute(pointwise math precision)");
    pointwise_.set(ACDNN_ATTR_POINTWISE_NAN_PROPAGATION,
                   ACDNN_TYPE_NAN_PROPOGATION, 1, &nan_policy,
                   "acdnnBackendSetAttribute(pointwise nan policy)");
    set_pointwise_parameters();
    pointwise_.finalize("acdnnBackendFinalize(pointwise)");

    acdnnBackendDescriptor_t pointwise = pointwise_.get();
    acdnnBackendDescriptor_t input = input_.get();
    acdnnBackendDescriptor_t output = output_.get();
    operation_.set(ACDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR,
                   ACDNN_TYPE_BACKEND_DESCRIPTOR, 1, &pointwise,
                   "acdnnBackendSetAttribute(operation pointwise)");
    acdnnBackendDescriptor_t second_input = second_input_.get();
    if (specification_.mode == ACDNN_POINTWISE_SIGMOID_BWD) {
      operation_.set(ACDNN_ATTR_OPERATION_POINTWISE_XDESC,
                     ACDNN_TYPE_BACKEND_DESCRIPTOR, 1, &second_input,
                     "acdnnBackendSetAttribute(operation backward input)");
      operation_.set(ACDNN_ATTR_OPERATION_POINTWISE_DYDESC,
                     ACDNN_TYPE_BACKEND_DESCRIPTOR, 1, &input,
                     "acdnnBackendSetAttribute(operation upstream gradient)");
      operation_.set(ACDNN_ATTR_OPERATION_POINTWISE_DXDESC,
                     ACDNN_TYPE_BACKEND_DESCRIPTOR, 1, &output,
                     "acdnnBackendSetAttribute(operation input gradient)");
    } else {
      operation_.set(ACDNN_ATTR_OPERATION_POINTWISE_XDESC,
                     ACDNN_TYPE_BACKEND_DESCRIPTOR, 1, &input,
                     "acdnnBackendSetAttribute(operation input)");
      if (specification_.inputs.size() == 2 ||
          specification_.constant_one_numerator) {
        operation_.set(ACDNN_ATTR_OPERATION_POINTWISE_BDESC,
                       ACDNN_TYPE_BACKEND_DESCRIPTOR, 1, &second_input,
                       "acdnnBackendSetAttribute(operation second input)");
      }
      operation_.set(ACDNN_ATTR_OPERATION_POINTWISE_YDESC,
                     ACDNN_TYPE_BACKEND_DESCRIPTOR, 1, &output,
                     "acdnnBackendSetAttribute(operation output)");
    }
    operation_.set(ACDNN_ATTR_OPERATION_POINTWISE_ALPHA1, ACDNN_TYPE_FLOAT,
                   1, &specification_.alpha1,
                   "acdnnBackendSetAttribute(operation alpha1)");
    operation_.set(ACDNN_ATTR_OPERATION_POINTWISE_ALPHA2, ACDNN_TYPE_FLOAT,
                   1, &specification_.alpha2,
                   "acdnnBackendSetAttribute(operation alpha2)");
    operation_.finalize("acdnnBackendFinalize(pointwise operation)");

    acdnnBackendDescriptor_t operation = operation_.get();
    acdnnHandle_t handle = handle_.get();
    operation_graph_.set(ACDNN_ATTR_OPERATIONGRAPH_OPS,
                         ACDNN_TYPE_BACKEND_DESCRIPTOR, 1, &operation,
                         "acdnnBackendSetAttribute(operation graph ops)");
    operation_graph_.set(ACDNN_ATTR_OPERATIONGRAPH_HANDLE,
                         ACDNN_TYPE_HANDLE, 1, &handle,
                         "acdnnBackendSetAttribute(operation graph handle)");
    operation_graph_.finalize("acdnnBackendFinalize(operation graph)");

    acdnnBackendDescriptor_t graph = operation_graph_.get();
    const acdnnBackendHeurMode_t mode = ACDNN_HEUR_MODE_A;
    heuristics_.set(ACDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                    ACDNN_TYPE_BACKEND_DESCRIPTOR, 1, &graph,
                    "acdnnBackendSetAttribute(heuristics graph)");
    heuristics_.set(ACDNN_ATTR_ENGINEHEUR_MODE, ACDNN_TYPE_HEUR_MODE, 1,
                    &mode, "acdnnBackendSetAttribute(heuristics mode)");
    heuristics_.finalize("acdnnBackendFinalize(heuristics)");

    std::int64_t result_count = 0;
    check_acdnn(acdnnBackendGetAttribute(
                    heuristics_.get(), ACDNN_ATTR_ENGINEHEUR_RESULTS,
                    ACDNN_TYPE_BACKEND_DESCRIPTOR, 0, &result_count, nullptr),
                "acdnnBackendGetAttribute(heuristics count)");
    if (result_count <= 0) {
      throw std::runtime_error(
          "acDNN backend pointwise heuristics returned no engine config");
    }
    acdnnBackendDescriptor_t engine_config = engine_config_.get();
    std::int64_t returned_count = 0;
    check_acdnn(acdnnBackendGetAttribute(
                    heuristics_.get(), ACDNN_ATTR_ENGINEHEUR_RESULTS,
                    ACDNN_TYPE_BACKEND_DESCRIPTOR, 1, &returned_count,
                    &engine_config),
                "acdnnBackendGetAttribute(heuristics result)");
    if (returned_count <= 0) {
      throw std::runtime_error(
          "acDNN backend pointwise returned no usable engine config: "
          "advertised=" +
          std::to_string(result_count) + " returned=" +
          std::to_string(returned_count));
    }

    plan_.set(ACDNN_ATTR_EXECUTION_PLAN_HANDLE, ACDNN_TYPE_HANDLE, 1,
              &handle, "acdnnBackendSetAttribute(plan handle)");
    plan_.set(ACDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
              ACDNN_TYPE_BACKEND_DESCRIPTOR, 1, &engine_config,
              "acdnnBackendSetAttribute(plan engine config)");
    plan_.finalize("acdnnBackendFinalize(execution plan)");

    std::int64_t workspace = 0;
    std::int64_t workspace_count = 0;
    check_acdnn(acdnnBackendGetAttribute(
                    plan_.get(), ACDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE,
                    ACDNN_TYPE_INT64, 1, &workspace_count, &workspace),
                "acdnnBackendGetAttribute(plan workspace)");
    if (workspace_count != 1 || workspace < 0 ||
        static_cast<std::uint64_t>(workspace) >
            std::numeric_limits<std::size_t>::max()) {
      throw std::runtime_error(
          "acDNN backend pointwise workspace size is invalid");
    }
    workspace_size_ = static_cast<std::size_t>(workspace);
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
          "acDNN backend pointwise workspace does not match plan");
    }
    handle_.set_stream(reinterpret_cast<hggcStream_t>(stream));
    std::vector<std::int64_t> uids;
    std::vector<void *> pointers;
    uids.reserve(specification_.inputs.size() + 2);
    pointers.reserve(specification_.inputs.size() + 2);
    if (specification_.constant_one_numerator) {
      uids.push_back(constant_numerator_uid_);
      pointers.push_back(constant_numerator_.data());
    }
    for (const flagdnn::testing::TestTensor &input : specification_.inputs) {
      uids.push_back(input.uid);
      pointers.push_back(required_pointer(bindings, input.uid));
    }
    uids.push_back(specification_.output.uid);
    pointers.push_back(
        required_pointer(bindings, specification_.output.uid));
    if (bindings.size() != specification_.inputs.size() + 1) {
      throw std::invalid_argument(
          "acDNN backend pointwise binding count is invalid");
    }
    BackendDescriptor variant_pack(ACDNN_BACKEND_VARIANT_PACK_DESCRIPTOR);
    variant_pack.set(ACDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
                     ACDNN_TYPE_VOID_PTR,
                     static_cast<std::int64_t>(pointers.size()),
                     pointers.data(),
                     "acdnnBackendSetAttribute(variant pointers)");
    variant_pack.set(ACDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, ACDNN_TYPE_INT64,
                     static_cast<std::int64_t>(uids.size()), uids.data(),
                     "acdnnBackendSetAttribute(variant uids)");
    variant_pack.set(ACDNN_ATTR_VARIANT_PACK_WORKSPACE, ACDNN_TYPE_VOID_PTR,
                     1, &workspace,
                     "acdnnBackendSetAttribute(variant workspace)");
    const std::int64_t workspace_bytes =
        static_cast<std::int64_t>(workspace_size);
    variant_pack.set(ACDNN_ATTR_VARIANT_PACK_WORKSPACE_SIZE,
                     ACDNN_TYPE_INT64, 1, &workspace_bytes,
                     "acdnnBackendSetAttribute(variant workspace size)");
    variant_pack.finalize("acdnnBackendFinalize(variant pack)");
    check_acdnn(acdnnBackendExecute(handle_.get(), plan_.get(),
                                    variant_pack.get()),
                specification_.primitive);
  }

 private:
  void set_pointwise_parameters() {
    const auto set_double = [this](acdnnBackendAttributeName_t name,
                                   const double &value,
                                   std::string_view description) {
      pointwise_.set(name, ACDNN_TYPE_DOUBLE, 1, &value, description);
    };
    switch (specification_.mode) {
      case ACDNN_POINTWISE_RELU_FWD:
      case ACDNN_POINTWISE_LEAKYRELU_FWD:
      case ACDNN_POINTWISE_CLIP_RELU_FWD:
        set_double(ACDNN_ATTR_POINTWISE_RELU_LOWER_CLIP,
                   specification_.relu_lower_clip,
                   "acdnnBackendSetAttribute(pointwise lower clip)");
        set_double(ACDNN_ATTR_POINTWISE_RELU_UPPER_CLIP,
                   specification_.relu_upper_clip,
                   "acdnnBackendSetAttribute(pointwise upper clip)");
        set_double(ACDNN_ATTR_POINTWISE_RELU_LOWER_CLIP_SLOPE,
                   specification_.relu_lower_clip_slope,
                   "acdnnBackendSetAttribute(pointwise lower clip slope)");
        break;
      case ACDNN_POINTWISE_ELU_FWD:
        set_double(ACDNN_ATTR_POINTWISE_ELU_ALPHA,
                   specification_.elu_alpha,
                   "acdnnBackendSetAttribute(pointwise ELU alpha)");
        break;
      case ACDNN_POINTWISE_SOFTPLUS_FWD:
        set_double(ACDNN_ATTR_POINTWISE_SOFTPLUS_BETA,
                   specification_.softplus_beta,
                   "acdnnBackendSetAttribute(pointwise Softplus beta)");
        break;
      case ACDNN_POINTWISE_SWISH_FWD:
        set_double(ACDNN_ATTR_POINTWISE_SWISH_BETA,
                   specification_.swish_beta,
                   "acdnnBackendSetAttribute(pointwise Swish beta)");
        break;
      default:
        break;
    }
  }

  BackendPointwiseReferenceSpecification specification_;
  AcdnnHandle handle_;
  BackendDescriptor pointwise_;
  BackendDescriptor input_;
  BackendDescriptor second_input_;
  BackendDescriptor output_;
  BackendDescriptor operation_;
  BackendDescriptor operation_graph_;
  BackendDescriptor heuristics_;
  BackendDescriptor engine_config_;
  BackendDescriptor plan_;
  DeviceBuffer constant_numerator_;
  std::int64_t constant_numerator_uid_ = 0;
  std::size_t workspace_size_ = 0;
};

std::size_t logical_element_count(
    const flagdnn::testing::TestTensor &tensor) {
  std::size_t result = 1;
  for (const std::int64_t dimension : tensor.dimensions) {
    if (dimension <= 0 ||
        static_cast<std::uint64_t>(dimension) >
            std::numeric_limits<std::size_t>::max() / result) {
      throw std::invalid_argument(
          "acDNN pointwise element count is invalid");
    }
    result *= static_cast<std::size_t>(dimension);
  }
  return result;
}

std::size_t largest_power_of_two(std::size_t upper_bound) {
  if (upper_bound == 0) {
    throw std::invalid_argument(
        "segmented acDNN pointwise segment is empty");
  }
  std::size_t result = 1;
  while (result <= upper_bound / 2) {
    result *= 2;
  }
  return result;
}

class AcdnnBackendPointwise final
    : public flagdnn::testing::TestExecutable {
 public:
  explicit AcdnnBackendPointwise(
      BackendPointwiseReferenceSpecification specification)
      : specification_(std::move(specification)) {
    const std::size_t elements =
        logical_element_count(specification_.output);
    bool dense_storage =
        elements == storage_element_count(specification_.output);
    for (const flagdnn::testing::TestTensor &input :
         specification_.inputs) {
      if (logical_element_count(input) != elements) {
        throw std::invalid_argument(
            "segmented acDNN pointwise tensors have different sizes");
      }
      dense_storage =
          dense_storage && elements == storage_element_count(input) &&
          input.strides == specification_.output.strides;
    }
    if (!dense_storage) {
      auto executable =
          std::make_unique<AcdnnBackendPointwiseSegment>(specification_);
      workspace_size_ = executable->workspace_size();
      segments_.push_back(
          {.element_offset = 0, .executable = std::move(executable)});
      return;
    }

    std::size_t element_offset = 0;
    while (element_offset < elements) {
      const std::size_t segment_elements =
          largest_power_of_two(elements - element_offset);
      if (segment_elements >
          static_cast<std::size_t>(
              std::numeric_limits<std::int64_t>::max())) {
        throw std::invalid_argument(
            "segmented acDNN pointwise segment exceeds int64 range");
      }
      BackendPointwiseReferenceSpecification segment = specification_;
      for (flagdnn::testing::TestTensor &input : segment.inputs) {
        input.dimensions = {static_cast<std::int64_t>(segment_elements)};
        input.strides = {1};
        input.binding_byte_offset +=
            element_offset * element_size(input.data_type);
      }
      segment.output.dimensions = {
          static_cast<std::int64_t>(segment_elements)};
      segment.output.strides = {1};
      segment.output.binding_byte_offset +=
          element_offset * element_size(segment.output.data_type);

      auto executable =
          std::make_unique<AcdnnBackendPointwiseSegment>(std::move(segment));
      workspace_size_ =
          std::max(workspace_size_, executable->workspace_size());
      segments_.push_back(
          {.element_offset = element_offset,
           .executable = std::move(executable)});
      element_offset += segment_elements;
    }
  }

  void prepare(std::span<const flagdnnBinding_t> bindings,
               flagdnnStream_t stream) override {
    for (Segment &segment : segments_) {
      const std::vector<flagdnnBinding_t> adjusted =
          adjusted_bindings(bindings, segment.element_offset);
      segment.executable->prepare(adjusted, stream);
    }
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
          "segmented acDNN pointwise workspace does not match plan");
    }
    for (Segment &segment : segments_) {
      const std::vector<flagdnnBinding_t> adjusted =
          adjusted_bindings(bindings, segment.element_offset);
      segment.executable->execute(
          adjusted, workspace, segment.executable->workspace_size(), stream);
    }
  }

 private:
  struct Segment {
    std::size_t element_offset;
    std::unique_ptr<flagdnn::testing::TestExecutable> executable;
  };

  [[nodiscard]] std::vector<flagdnnBinding_t> adjusted_bindings(
      std::span<const flagdnnBinding_t> bindings,
      std::size_t element_offset) const {
    std::map<std::int64_t, flagdnnDataType_t> data_types;
    for (const flagdnn::testing::TestTensor &input :
         specification_.inputs) {
      if (!data_types.emplace(input.uid, input.data_type).second) {
        throw std::invalid_argument(
            "segmented acDNN pointwise input UID is duplicate");
      }
    }
    if (!data_types
             .emplace(specification_.output.uid,
                      specification_.output.data_type)
             .second) {
      throw std::invalid_argument(
          "segmented acDNN pointwise output UID is duplicate");
    }
    if (bindings.size() != data_types.size()) {
      throw std::invalid_argument(
          "segmented acDNN pointwise binding count is invalid");
    }

    std::vector<flagdnnBinding_t> result;
    result.reserve(bindings.size());
    for (const flagdnnBinding_t &binding : bindings) {
      const auto iterator = data_types.find(binding.uid);
      if (iterator == data_types.end() || binding.device_pointer == nullptr) {
        throw std::invalid_argument(
            "segmented acDNN pointwise binding is invalid");
      }
      auto *pointer = static_cast<std::byte *>(binding.device_pointer);
      result.push_back(
          {binding.uid,
           pointer + element_offset * element_size(iterator->second)});
    }
    return result;
  }

  BackendPointwiseReferenceSpecification specification_;
  std::vector<Segment> segments_;
  std::size_t workspace_size_ = 0;
};

}  // namespace

std::unique_ptr<flagdnn::testing::TestExecutable>
make_acdnn_backend_pointwise_reference(
    BackendPointwiseReferenceSpecification specification) {
  return std::make_unique<AcdnnBackendPointwise>(std::move(specification));
}

}  // namespace flagdnn::validation::thead
