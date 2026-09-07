// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "acdnn_reduction_reference.hpp"

#include "acdnn_reference.hpp"
#include "backend_pointwise_reference.hpp"
#include "numeric_types.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#ifndef FLAGDNN_THEAD_ACDNN_CAPABILITY_CATALOG
#define FLAGDNN_THEAD_ACDNN_CAPABILITY_CATALOG "capability.json"
#endif

namespace flagdnn::validation::thead {
namespace {

std::vector<int> checked_ints(std::span<const std::int64_t> values,
                              std::string_view description) {
  std::vector<int> result;
  result.reserve(values.size());
  for (const std::int64_t value : values) {
    if (value <= 0 || value > std::numeric_limits<int>::max()) {
      throw std::invalid_argument(std::string(description) +
                                  " is outside positive int32");
    }
    result.push_back(static_cast<int>(value));
  }
  return result;
}

acdnnReduceTensorOp_t acdnn_mode(flagdnnReductionMode_t mode) {
  switch (mode) {
    case FLAGDNN_REDUCTION_ADD:
      return ACDNN_REDUCE_TENSOR_ADD;
    case FLAGDNN_REDUCTION_AVG:
      return ACDNN_REDUCE_TENSOR_AVG;
    case FLAGDNN_REDUCTION_MUL:
      return ACDNN_REDUCE_TENSOR_MUL;
  }
  throw std::invalid_argument("unsupported THead acDNN reduction mode");
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
  throw std::invalid_argument("acDNN reduction data type is unsupported");
}

std::string primitive(flagdnnReductionMode_t mode) {
  switch (mode) {
    case FLAGDNN_REDUCTION_ADD:
      return "acdnnReduceTensor(ADD,alpha=1,beta=0)";
    case FLAGDNN_REDUCTION_AVG:
      return "acdnnReduceTensor(AVG,alpha=1,beta=0)";
    case FLAGDNN_REDUCTION_MUL:
      return "acdnnReduceTensor(MUL,alpha=1,beta=0)";
  }
  throw std::invalid_argument("unsupported THead acDNN reduction mode");
}

std::string backend_primitive(flagdnnReductionMode_t mode) {
  switch (mode) {
    case FLAGDNN_REDUCTION_ADD:
      return "acdnnBackendExecute(REDUCTION_ADD,compute=fp32)";
    case FLAGDNN_REDUCTION_AVG:
      return "acdnnBackendExecute(REDUCTION_AVG,compute=fp32)";
    case FLAGDNN_REDUCTION_MUL:
      return "acdnnBackendExecute(REDUCTION_MUL,compute=fp32)";
  }
  throw std::invalid_argument("unsupported THead backend reduction mode");
}

constexpr std::string_view kPackDensePrimitive =
    "acdnnBackendExecute(POINTWISE_IDENTITY_FWD,pack-segments)";
constexpr std::string_view kConvertPackPrimitive =
    "acdnnBackendExecute(POINTWISE_IDENTITY_FWD,convert-pack-fp32)";
constexpr std::string_view kConvertOutputPrimitive =
    "acdnnBackendExecute(POINTWISE_IDENTITY_FWD,convert-bfloat16)";

std::vector<std::int64_t>
contiguous_strides(std::span<const std::int64_t> dimensions) {
  std::vector<std::int64_t> result(dimensions.size());
  std::int64_t stride = 1;
  for (std::size_t axis = dimensions.size(); axis != 0; --axis) {
    const std::int64_t dimension = dimensions[axis - 1];
    if (dimension <= 0 ||
        stride > std::numeric_limits<std::int64_t>::max() / dimension) {
      throw std::overflow_error("acDNN reduction dense stride overflows");
    }
    result[axis - 1] = stride;
    stride *= dimension;
  }
  return result;
}

std::size_t tensor_bytes(const flagdnn::testing::TestTensor &tensor) {
  std::size_t elements = 1;
  for (const std::int64_t dimension : tensor.dimensions) {
    if (dimension <= 0 ||
        static_cast<std::uint64_t>(dimension) >
            std::numeric_limits<std::size_t>::max() / elements) {
      throw std::overflow_error("acDNN reduction scratch size overflows");
    }
    elements *= static_cast<std::size_t>(dimension);
  }
  const std::size_t scalar_bytes = element_size(tensor.data_type);
  if (elements > std::numeric_limits<std::size_t>::max() / scalar_bytes) {
    throw std::overflow_error("acDNN reduction scratch size overflows");
  }
  return elements * scalar_bytes;
}

struct PackSegment {
  std::size_t input_offset = 0;
  std::size_t output_offset = 0;
};

class BackendDescriptor final {
 public:
  explicit BackendDescriptor(acdnnBackendDescriptorType_t type) {
    check_acdnn(acdnnBackendCreateDescriptor(type, &descriptor_),
                "acdnnBackendCreateDescriptor(reduction)");
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

void build_backend_tensor(BackendDescriptor &descriptor,
                          const flagdnn::testing::TestTensor &tensor) {
  if (tensor.uid <= 0 || tensor.dimensions.empty() ||
      tensor.dimensions.size() != tensor.strides.size() ||
      tensor.dimensions.size() > 8 ||
      std::ranges::any_of(tensor.dimensions,
                          [](std::int64_t value) { return value <= 0; }) ||
      std::ranges::any_of(tensor.strides,
                          [](std::int64_t value) { return value <= 0; })) {
    throw std::invalid_argument(
        "acDNN backend reduction tensor metadata is invalid");
  }
  const acdnnDataType_t data_type = acdnn_data_type(tensor.data_type);
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
        "acDNN backend reduction alignment is below scalar size");
  }
  descriptor.set(ACDNN_ATTR_TENSOR_DATA_TYPE, ACDNN_TYPE_DATA_TYPE, 1,
                 &data_type,
                 "acdnnBackendSetAttribute(reduction tensor data type)");
  descriptor.set(ACDNN_ATTR_TENSOR_DIMENSIONS, ACDNN_TYPE_INT64,
                 static_cast<std::int64_t>(tensor.dimensions.size()),
                 tensor.dimensions.data(),
                 "acdnnBackendSetAttribute(reduction tensor dimensions)");
  descriptor.set(ACDNN_ATTR_TENSOR_STRIDES, ACDNN_TYPE_INT64,
                 static_cast<std::int64_t>(tensor.strides.size()),
                 tensor.strides.data(),
                 "acdnnBackendSetAttribute(reduction tensor strides)");
  descriptor.set(ACDNN_ATTR_TENSOR_UNIQUE_ID, ACDNN_TYPE_INT64, 1,
                 &tensor.uid,
                 "acdnnBackendSetAttribute(reduction tensor uid)");
  descriptor.set(ACDNN_ATTR_TENSOR_BYTE_ALIGNMENT, ACDNN_TYPE_INT64, 1,
                 &alignment,
                 "acdnnBackendSetAttribute(reduction tensor alignment)");
  descriptor.finalize("acdnnBackendFinalize(reduction tensor)");
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
          "acDNN backend reduction binding is duplicate or null");
    }
    result = binding.device_pointer;
  }
  if (result == nullptr) {
    throw std::invalid_argument(
        "acDNN backend reduction binding is missing");
  }
  return result;
}

class AcdnnBackendReduction final
    : public flagdnn::testing::TestExecutable {
 public:
  AcdnnBackendReduction(flagdnn::testing::TestTensor input,
                        flagdnn::testing::TestTensor output,
                        flagdnnReductionMode_t mode)
      : input_specification_(std::move(input)),
        output_specification_(std::move(output)),
        mode_(mode),
        reduction_(ACDNN_BACKEND_REDUCTION_DESCRIPTOR),
        input_(ACDNN_BACKEND_TENSOR_DESCRIPTOR),
        output_(ACDNN_BACKEND_TENSOR_DESCRIPTOR),
        operation_(ACDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR),
        operation_graph_(ACDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR),
        heuristics_(ACDNN_BACKEND_ENGINEHEUR_DESCRIPTOR),
        engine_config_(ACDNN_BACKEND_ENGINECFG_DESCRIPTOR),
        plan_(ACDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR) {
    if (input_specification_.data_type != output_specification_.data_type ||
        input_specification_.uid == output_specification_.uid) {
      throw std::invalid_argument(
          "acDNN backend reduction tensor types or UIDs are invalid");
    }
    build_backend_tensor(input_, input_specification_);
    build_backend_tensor(output_, output_specification_);

    const acdnnReduceTensorOp_t operator_type = acdnn_mode(mode_);
    const acdnnDataType_t compute_type = ACDNN_DATA_FLOAT;
    reduction_.set(ACDNN_ATTR_REDUCTION_OPERATOR,
                   ACDNN_TYPE_REDUCTION_OPERATOR_TYPE, 1, &operator_type,
                   "acdnnBackendSetAttribute(reduction operator)");
    reduction_.set(ACDNN_ATTR_REDUCTION_COMP_TYPE, ACDNN_TYPE_DATA_TYPE, 1,
                   &compute_type,
                   "acdnnBackendSetAttribute(reduction compute type)");
    reduction_.finalize("acdnnBackendFinalize(reduction descriptor)");

    acdnnBackendDescriptor_t reduction = reduction_.get();
    acdnnBackendDescriptor_t input_descriptor = input_.get();
    acdnnBackendDescriptor_t output_descriptor = output_.get();
    operation_.set(ACDNN_ATTR_OPERATION_REDUCTION_DESC,
                   ACDNN_TYPE_BACKEND_DESCRIPTOR, 1, &reduction,
                   "acdnnBackendSetAttribute(reduction operation)");
    operation_.set(ACDNN_ATTR_OPERATION_REDUCTION_XDESC,
                   ACDNN_TYPE_BACKEND_DESCRIPTOR, 1, &input_descriptor,
                   "acdnnBackendSetAttribute(reduction input)");
    operation_.set(ACDNN_ATTR_OPERATION_REDUCTION_YDESC,
                   ACDNN_TYPE_BACKEND_DESCRIPTOR, 1, &output_descriptor,
                   "acdnnBackendSetAttribute(reduction output)");
    operation_.finalize("acdnnBackendFinalize(reduction operation)");

    acdnnBackendDescriptor_t operation = operation_.get();
    acdnnHandle_t handle = handle_.get();
    operation_graph_.set(ACDNN_ATTR_OPERATIONGRAPH_OPS,
                         ACDNN_TYPE_BACKEND_DESCRIPTOR, 1, &operation,
                         "acdnnBackendSetAttribute(reduction graph ops)");
    operation_graph_.set(ACDNN_ATTR_OPERATIONGRAPH_HANDLE,
                         ACDNN_TYPE_HANDLE, 1, &handle,
                         "acdnnBackendSetAttribute(reduction graph handle)");
    operation_graph_.finalize("acdnnBackendFinalize(reduction graph)");

    acdnnBackendDescriptor_t graph = operation_graph_.get();
    const acdnnBackendHeurMode_t heuristic_mode = ACDNN_HEUR_MODE_A;
    heuristics_.set(ACDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                    ACDNN_TYPE_BACKEND_DESCRIPTOR, 1, &graph,
                    "acdnnBackendSetAttribute(reduction heuristics graph)");
    heuristics_.set(ACDNN_ATTR_ENGINEHEUR_MODE, ACDNN_TYPE_HEUR_MODE, 1,
                    &heuristic_mode,
                    "acdnnBackendSetAttribute(reduction heuristics mode)");
    heuristics_.finalize("acdnnBackendFinalize(reduction heuristics)");

    std::int64_t result_count = 0;
    check_acdnn(acdnnBackendGetAttribute(
                    heuristics_.get(), ACDNN_ATTR_ENGINEHEUR_RESULTS,
                    ACDNN_TYPE_BACKEND_DESCRIPTOR, 0, &result_count, nullptr),
                "acdnnBackendGetAttribute(reduction heuristics count)");
    if (result_count <= 0) {
      throw std::runtime_error(
          "acDNN backend reduction heuristics returned no engine config");
    }
    acdnnBackendDescriptor_t engine_config = engine_config_.get();
    std::int64_t returned_count = 0;
    check_acdnn(acdnnBackendGetAttribute(
                    heuristics_.get(), ACDNN_ATTR_ENGINEHEUR_RESULTS,
                    ACDNN_TYPE_BACKEND_DESCRIPTOR, 1, &returned_count,
                    &engine_config),
                "acdnnBackendGetAttribute(reduction heuristics result)");
    if (returned_count <= 0) {
      throw std::runtime_error(
          "acDNN backend reduction returned no usable engine config");
    }

    plan_.set(ACDNN_ATTR_EXECUTION_PLAN_HANDLE, ACDNN_TYPE_HANDLE, 1,
              &handle,
              "acdnnBackendSetAttribute(reduction plan handle)");
    plan_.set(ACDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
              ACDNN_TYPE_BACKEND_DESCRIPTOR, 1, &engine_config,
              "acdnnBackendSetAttribute(reduction plan engine config)");
    plan_.finalize("acdnnBackendFinalize(reduction execution plan)");

    std::int64_t workspace = 0;
    std::int64_t workspace_count = 0;
    check_acdnn(acdnnBackendGetAttribute(
                    plan_.get(), ACDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE,
                    ACDNN_TYPE_INT64, 1, &workspace_count, &workspace),
                "acdnnBackendGetAttribute(reduction workspace)");
    if (workspace_count != 1 || workspace < 0 ||
        static_cast<std::uint64_t>(workspace) >
            std::numeric_limits<std::size_t>::max()) {
      throw std::runtime_error(
          "acDNN backend reduction workspace size is invalid");
    }
    workspace_size_ = static_cast<std::size_t>(workspace);
  }

  [[nodiscard]] std::size_t workspace_size() const noexcept override {
    return workspace_size_;
  }

  void execute(std::span<const flagdnnBinding_t> bindings, void *workspace,
               std::size_t workspace_size,
               flagdnnStream_t stream) override {
    if (stream == nullptr || workspace_size != workspace_size_ ||
        (workspace_size != 0 && workspace == nullptr) ||
        bindings.size() != 2) {
      throw std::invalid_argument(
          "acDNN backend reduction execution arguments are invalid");
    }
    handle_.set_stream(reinterpret_cast<hggcStream_t>(stream));
    std::array<std::int64_t, 2> uids = {
        input_specification_.uid, output_specification_.uid};
    std::array<void *, 2> pointers = {
        required_pointer(bindings, input_specification_.uid),
        required_pointer(bindings, output_specification_.uid)};
    BackendDescriptor variant_pack(ACDNN_BACKEND_VARIANT_PACK_DESCRIPTOR);
    variant_pack.set(ACDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
                     ACDNN_TYPE_VOID_PTR, 2, pointers.data(),
                     "acdnnBackendSetAttribute(reduction pointers)");
    variant_pack.set(ACDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, ACDNN_TYPE_INT64, 2,
                     uids.data(),
                     "acdnnBackendSetAttribute(reduction uids)");
    variant_pack.set(ACDNN_ATTR_VARIANT_PACK_WORKSPACE, ACDNN_TYPE_VOID_PTR,
                     1, &workspace,
                     "acdnnBackendSetAttribute(reduction workspace pointer)");
    const std::int64_t workspace_bytes =
        static_cast<std::int64_t>(workspace_size);
    variant_pack.set(ACDNN_ATTR_VARIANT_PACK_WORKSPACE_SIZE,
                     ACDNN_TYPE_INT64, 1, &workspace_bytes,
                     "acdnnBackendSetAttribute(reduction workspace size)");
    variant_pack.finalize("acdnnBackendFinalize(reduction variant pack)");
    check_acdnn(acdnnBackendExecute(handle_.get(), plan_.get(),
                                    variant_pack.get()),
                backend_primitive(mode_));
  }

 private:
  flagdnn::testing::TestTensor input_specification_;
  flagdnn::testing::TestTensor output_specification_;
  flagdnnReductionMode_t mode_;
  AcdnnHandle handle_;
  BackendDescriptor reduction_;
  BackendDescriptor input_;
  BackendDescriptor output_;
  BackendDescriptor operation_;
  BackendDescriptor operation_graph_;
  BackendDescriptor heuristics_;
  BackendDescriptor engine_config_;
  BackendDescriptor plan_;
  std::size_t workspace_size_ = 0;
};

class ReduceDescriptor final {
 public:
  ReduceDescriptor() {
    check_acdnn(acdnnCreateReduceTensorDescriptor(&descriptor_),
                "acdnnCreateReduceTensorDescriptor");
  }
  ~ReduceDescriptor() {
    if (descriptor_ != nullptr) {
      (void)acdnnDestroyReduceTensorDescriptor(descriptor_);
    }
  }
  ReduceDescriptor(const ReduceDescriptor &) = delete;
  ReduceDescriptor &operator=(const ReduceDescriptor &) = delete;

  void set(flagdnnReductionMode_t mode) {
    check_acdnn(
        acdnnSetReduceTensorDescriptor(
            descriptor_, acdnn_mode(mode), ACDNN_DATA_FLOAT,
            ACDNN_NOT_PROPAGATE_NAN, ACDNN_REDUCE_TENSOR_NO_INDICES,
            ACDNN_32BIT_INDICES),
        "acdnnSetReduceTensorDescriptor");
  }

  [[nodiscard]] acdnnReduceTensorDescriptor_t get() const noexcept {
    return descriptor_;
  }

 private:
  acdnnReduceTensorDescriptor_t descriptor_ = nullptr;
};

class AcdnnReduction final : public flagdnn::testing::ReductionExecutable {
 public:
  AcdnnReduction(flagdnn::testing::ReductionTestCase test_case,
                 const CapabilityRecord &capability)
      : test_case_(std::move(test_case)) {
    if ((test_case_.input.data_type != FLAGDNN_DATA_FLOAT32 &&
         test_case_.input.data_type != FLAGDNN_DATA_FLOAT16 &&
         test_case_.input.data_type != FLAGDNN_DATA_BFLOAT16) ||
        test_case_.output.data_type != test_case_.input.data_type ||
        test_case_.input.dimensions.empty()) {
      throw std::invalid_argument(
          "THead acDNN reduction reference requires matching floating "
          "tensors and a non-scalar input");
    }
    const ReferenceSelection selection = select_reference(capability);
    if (!std::holds_alternative<ReferencePlan>(selection)) {
      throw std::invalid_argument(
          "unsupported reduction reached acDNN reference");
    }
    const ReferencePlan &plan = std::get<ReferencePlan>(selection);
    const std::vector<std::int64_t> dense_input_strides =
        contiguous_strides(test_case_.input.dimensions);
    const bool needs_pack = test_case_.input.strides != dense_input_strides;
    const bool needs_bfloat_conversion =
        test_case_.input.data_type == FLAGDNN_DATA_BFLOAT16;
    std::vector<std::string> expected_plan;
    if (needs_bfloat_conversion) {
      expected_plan = {std::string(kConvertPackPrimitive),
                       primitive(test_case_.mode),
                       std::string(kConvertOutputPrimitive)};
    } else if (needs_pack) {
      expected_plan = {std::string(kPackDensePrimitive),
                       primitive(test_case_.mode)};
    } else {
      expected_plan = {primitive(test_case_.mode)};
    }
    const ReferencePath expected_path =
        needs_pack || needs_bfloat_conversion
            ? ReferencePath::kBackendDescriptor
            : ReferencePath::kStablePrimitive;
    if (plan.path != expected_path || plan.primitives != expected_plan) {
      throw std::invalid_argument("THead acDNN reduction plan mismatch");
    }
    flagdnn::testing::TestTensor reduction_input = test_case_.input;
    if (needs_pack || needs_bfloat_conversion) {
      constexpr std::int64_t kPackedInputUid =
          std::numeric_limits<std::int64_t>::max() - 7;
      if (test_case_.input.uid == kPackedInputUid ||
          test_case_.output.uid == kPackedInputUid) {
        throw std::invalid_argument("acDNN reduction scratch UID collides");
      }
      packed_input_specification_ = test_case_.input;
      packed_input_specification_.uid = kPackedInputUid;
      if (needs_bfloat_conversion) {
        packed_input_specification_.data_type = FLAGDNN_DATA_FLOAT32;
      }
      packed_input_specification_.strides = dense_input_strides;
      packed_input_specification_.binding_byte_offset = 0;
      std::size_t segment_length = 1;
      const std::size_t rank = test_case_.input.dimensions.size();
      if (test_case_.input.strides.back() == 1) {
        segment_length =
            static_cast<std::size_t>(test_case_.input.dimensions.back());
        for (std::size_t axis = rank - 1; axis != 0; --axis) {
          const std::size_t previous = axis - 1;
          if (test_case_.input.strides[previous] !=
              static_cast<std::int64_t>(segment_length)) {
            break;
          }
          const std::size_t extent = static_cast<std::size_t>(
              test_case_.input.dimensions[previous]);
          if (extent > std::numeric_limits<std::size_t>::max() /
                           segment_length) {
            throw std::overflow_error(
                "acDNN reduction pack segment overflows");
          }
          segment_length *= extent;
        }
      }
      const std::size_t packed_bytes =
          tensor_bytes(packed_input_specification_);
      const std::size_t input_scalar_bytes =
          element_size(test_case_.input.data_type);
      const std::size_t packed_scalar_bytes =
          element_size(packed_input_specification_.data_type);
      const std::size_t elements = packed_bytes / packed_scalar_bytes;
      if (elements % segment_length != 0 ||
          segment_length >
              static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
        throw std::overflow_error("acDNN reduction pack geometry is invalid");
      }
      const std::vector<std::int64_t> segment_dimensions = {
          1, 1, static_cast<std::int64_t>(segment_length)};
      const std::vector<std::int64_t> segment_strides = {
          static_cast<std::int64_t>(segment_length),
          static_cast<std::int64_t>(segment_length), 1};
      flagdnn::testing::TestTensor segment_input = test_case_.input;
      segment_input.dimensions = segment_dimensions;
      segment_input.strides = segment_strides;
      segment_input.binding_byte_offset = input_scalar_bytes;
      flagdnn::testing::TestTensor segment_output =
          packed_input_specification_;
      segment_output.dimensions = segment_dimensions;
      segment_output.strides = segment_strides;
      segment_output.binding_byte_offset = packed_scalar_bytes;
      pack_ = make_acdnn_backend_pointwise_reference(
          {.mode = ACDNN_POINTWISE_IDENTITY_FWD,
           .inputs = {segment_input},
           .output = segment_output,
           .primitive = std::string(needs_bfloat_conversion
                                        ? kConvertPackPrimitive
                                        : kPackDensePrimitive)});
      packed_input_ = DeviceBuffer(packed_bytes);
      pack_segments_.reserve(elements / segment_length);
      for (std::size_t logical_index = 0; logical_index < elements;
           logical_index += segment_length) {
        std::size_t remaining = logical_index;
        std::size_t input_offset = 0;
        for (std::size_t axis = rank; axis != 0; --axis) {
          const std::size_t current = axis - 1;
          const std::size_t extent = static_cast<std::size_t>(
              test_case_.input.dimensions[current]);
          const std::size_t coordinate = remaining % extent;
          remaining /= extent;
          const std::size_t stride = static_cast<std::size_t>(
              test_case_.input.strides[current]);
          if (coordinate != 0 &&
              stride > (std::numeric_limits<std::size_t>::max() -
                        input_offset) /
                           coordinate) {
            throw std::overflow_error(
                "acDNN reduction pack offset overflows");
          }
          input_offset += coordinate * stride;
        }
        pack_segments_.push_back({input_offset, logical_index});
      }
      reduction_input = packed_input_specification_;
    }
    reduction_input_uid_ = reduction_input.uid;
    std::vector<std::int64_t> acdnn_output_dimensions =
        test_case_.output.dimensions;
    std::vector<std::int64_t> acdnn_output_strides =
        test_case_.output.strides;
    if (!test_case_.keep_dimensions) {
      std::int32_t axis = test_case_.axis;
      if (axis < 0) {
        axis += static_cast<std::int32_t>(test_case_.input.dimensions.size());
      }
      const std::size_t insertion = static_cast<std::size_t>(axis);
      const std::int64_t inserted_stride =
          insertion < acdnn_output_dimensions.size()
              ? acdnn_output_dimensions[insertion] *
                    acdnn_output_strides[insertion]
              : 1;
      acdnn_output_dimensions.insert(
          acdnn_output_dimensions.begin() + axis, 1);
      acdnn_output_strides.insert(acdnn_output_strides.begin() + axis,
                                  inserted_stride);
    }
    flagdnn::testing::TestTensor reduction_output = test_case_.output;
    reduction_output.dimensions = acdnn_output_dimensions;
    reduction_output.strides = acdnn_output_strides;
    if (needs_bfloat_conversion) {
      constexpr std::int64_t kReductionOutputUid =
          std::numeric_limits<std::int64_t>::max() - 8;
      if (test_case_.input.uid == kReductionOutputUid ||
          test_case_.output.uid == kReductionOutputUid ||
          packed_input_specification_.uid == kReductionOutputUid) {
        throw std::invalid_argument(
            "acDNN reduction output scratch UID collides");
      }
      reduction_output.uid = kReductionOutputUid;
      reduction_output.data_type = FLAGDNN_DATA_FLOAT32;
      reduction_output.binding_byte_offset = 0;
      reduction_output_specification_ = reduction_output;
      reduction_output_ = DeviceBuffer(tensor_bytes(reduction_output));

      const std::size_t output_elements =
          tensor_bytes(reduction_output) /
          element_size(reduction_output.data_type);
      if (output_elements == 0 ||
          output_elements > static_cast<std::size_t>(
                                std::numeric_limits<std::int64_t>::max())) {
        throw std::overflow_error(
            "acDNN reduction output conversion size is invalid");
      }
      const std::vector<std::int64_t> conversion_dimensions = {
          1, 1, static_cast<std::int64_t>(output_elements)};
      const std::vector<std::int64_t> conversion_strides = {
          static_cast<std::int64_t>(output_elements),
          static_cast<std::int64_t>(output_elements), 1};
      flagdnn::testing::TestTensor conversion_input = reduction_output;
      conversion_input.dimensions = conversion_dimensions;
      conversion_input.strides = conversion_strides;
      conversion_input.binding_byte_offset = 4;
      flagdnn::testing::TestTensor conversion_output = test_case_.output;
      conversion_output.dimensions = conversion_dimensions;
      conversion_output.strides = conversion_strides;
      conversion_output.binding_byte_offset = 2;
      output_conversion_ = make_acdnn_backend_pointwise_reference(
          {.mode = ACDNN_POINTWISE_IDENTITY_FWD,
           .inputs = {conversion_input},
           .output = conversion_output,
           .primitive = std::string(kConvertOutputPrimitive)});
    }
    const acdnnDataType_t input_data_type =
        acdnn_data_type(reduction_input.data_type);
    const acdnnDataType_t output_data_type =
        acdnn_data_type(reduction_output.data_type);
    input_.set(input_data_type,
               checked_ints(reduction_input.dimensions, "input dimension"),
               checked_ints(reduction_input.strides, "input stride"));
    output_.set(output_data_type,
                checked_ints(acdnn_output_dimensions, "output dimension"),
                checked_ints(acdnn_output_strides, "output stride"));
    reduction_.set(test_case_.mode);
    check_acdnn(
        acdnnGetReductionWorkspaceSize(handle_.get(), reduction_.get(),
                                       input_.get(), output_.get(),
                                       &workspace_size_),
        "acdnnGetReductionWorkspaceSize");
  }

  [[nodiscard]] std::size_t workspace_size() const noexcept override {
    // acDNN 1400 rejects null indices/workspace pointers even when the
    // descriptor requests no indices and both reported byte counts are zero.
    return std::max({workspace_size_,
                     pack_ == nullptr ? std::size_t{0}
                                      : pack_->workspace_size(),
                     output_conversion_ == nullptr
                         ? std::size_t{0}
                         : output_conversion_->workspace_size(),
                     std::size_t{1}});
  }

  void execute(std::span<const flagdnnBinding_t> bindings, void *workspace,
               std::size_t workspace_size,
               flagdnnStream_t stream) override {
    if (stream == nullptr || workspace == nullptr ||
        workspace_size < this->workspace_size()) {
      throw std::invalid_argument("THead acDNN reduction execution is invalid");
    }
    std::map<std::int64_t, void *> pointers;
    for (const flagdnnBinding_t &binding : bindings) {
      if (binding.device_pointer == nullptr ||
          !pointers.emplace(binding.uid, binding.device_pointer).second) {
        throw std::invalid_argument("THead reduction bindings are invalid");
      }
    }
    if (pointers.size() != 2 || !pointers.contains(test_case_.input.uid) ||
        !pointers.contains(test_case_.output.uid)) {
      throw std::invalid_argument(
          "THead reduction requires input/output bindings");
    }
    handle_.set_stream(reinterpret_cast<hggcStream_t>(stream));
    void *input_pointer = pointers.at(test_case_.input.uid);
    if (pack_ != nullptr) {
      const std::size_t input_scalar_bytes =
          element_size(test_case_.input.data_type);
      const std::size_t packed_scalar_bytes =
          element_size(packed_input_specification_.data_type);
      const auto input_address =
          reinterpret_cast<std::uintptr_t>(input_pointer);
      const auto output_address =
          reinterpret_cast<std::uintptr_t>(packed_input_.data());
      for (const PackSegment &segment : pack_segments_) {
        if (segment.input_offset >
                (std::numeric_limits<std::uintptr_t>::max() - input_address) /
                    input_scalar_bytes ||
            segment.output_offset >
                (std::numeric_limits<std::uintptr_t>::max() - output_address) /
                    packed_scalar_bytes) {
          throw std::overflow_error("acDNN reduction pack pointer overflows");
        }
        const std::array<flagdnnBinding_t, 2> pack_bindings = {{
            {test_case_.input.uid,
             reinterpret_cast<void *>(
                 input_address +
                 segment.input_offset * input_scalar_bytes)},
            {packed_input_specification_.uid,
             reinterpret_cast<void *>(
                 output_address +
                 segment.output_offset * packed_scalar_bytes)},
        }};
        pack_->prepare(pack_bindings, stream);
        pack_->execute(pack_bindings, workspace, pack_->workspace_size(),
                       stream);
      }
      input_pointer = packed_input_.data();
    }
    constexpr float alpha = 1.0F;
    constexpr float beta = 0.0F;
    void *output_pointer =
        output_conversion_ == nullptr
            ? pointers.at(test_case_.output.uid)
            : reduction_output_.data();
    check_acdnn(
        acdnnReduceTensor(
            handle_.get(), reduction_.get(), workspace, 0, workspace,
            workspace_size_, &alpha, input_.get(),
            input_pointer, &beta, output_.get(),
            output_pointer),
        primitive(test_case_.mode));
    if (output_conversion_ != nullptr) {
      const std::array<flagdnnBinding_t, 2> conversion_bindings = {{
          {reduction_output_specification_.uid, reduction_output_.data()},
          {test_case_.output.uid, pointers.at(test_case_.output.uid)},
      }};
      output_conversion_->prepare(conversion_bindings, stream);
      output_conversion_->execute(
          conversion_bindings, workspace,
          output_conversion_->workspace_size(), stream);
    }
  }

 private:
  flagdnn::testing::ReductionTestCase test_case_;
  flagdnn::testing::TestTensor packed_input_specification_;
  std::unique_ptr<flagdnn::testing::TestExecutable> pack_;
  DeviceBuffer packed_input_;
  std::vector<PackSegment> pack_segments_;
  flagdnn::testing::TestTensor reduction_output_specification_;
  std::unique_ptr<flagdnn::testing::TestExecutable> output_conversion_;
  DeviceBuffer reduction_output_;
  std::int64_t reduction_input_uid_ = 0;
  AcdnnHandle handle_;
  AcdnnTensorDescriptor input_;
  AcdnnTensorDescriptor output_;
  ReduceDescriptor reduction_;
  std::size_t workspace_size_ = 0;
};

}  // namespace

std::unique_ptr<flagdnn::testing::ReductionExecutable>
make_acdnn_reduction_reference(
    const flagdnn::testing::ReductionTestCase &test_case,
    const CapabilityRecord &capability) {
  return std::make_unique<AcdnnReduction>(test_case, capability);
}

}  // namespace flagdnn::validation::thead

namespace flagdnn::testing {

TestTensor reduction_reference_input_tensor(
    const ReductionTestCase &test_case) {
  return test_case.input;
}

std::unique_ptr<ReductionExecutable> build_reduction_reference(
    const ReductionTestCase &test_case) {
  using namespace flagdnn::validation::thead;
  static const CapabilityCatalog catalog = CapabilityCatalog::load(
      FLAGDNN_THEAD_ACDNN_CAPABILITY_CATALOG);
  return make_acdnn_reduction_reference(
      test_case, catalog.lookup("reduction", test_case.name));
}

}  // namespace flagdnn::testing
