// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "acdnn_matmul_reference.hpp"

#include "acdnn_reference.hpp"
#include "numeric_types.hpp"

#include <acdnn_backend.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace flagdnn::validation::thead {
namespace {

constexpr std::string_view kPrimitive = "acdnnBackendExecute(MATMUL)";

void validate_tensor_geometry(
    const flagdnn::testing::TestTensor &tensor,
    std::string_view name) {
  if (tensor.uid <= 0 || tensor.dimensions.size() < 2 ||
      tensor.dimensions.size() > 8 ||
      tensor.dimensions.size() != tensor.strides.size() ||
      std::ranges::any_of(tensor.dimensions,
                          [](std::int64_t value) { return value <= 0; }) ||
      std::ranges::any_of(tensor.strides,
                          [](std::int64_t value) { return value <= 0; })) {
    throw std::invalid_argument(std::string(name) +
                                " geometry is invalid");
  }
}

void validate_matmul_geometry(
    const flagdnn::testing::MatmulTestCase &test_case) {
  validate_tensor_geometry(test_case.a, "acDNN MatMul A");
  validate_tensor_geometry(test_case.b, "acDNN MatMul B");
  validate_tensor_geometry(test_case.output, "acDNN MatMul output");
  if (test_case.a.uid == test_case.b.uid ||
      test_case.a.uid == test_case.output.uid ||
      test_case.b.uid == test_case.output.uid ||
      test_case.a.dimensions.back() !=
          test_case.b.dimensions[test_case.b.dimensions.size() - 2]) {
    throw std::invalid_argument("acDNN MatMul geometry is inconsistent");
  }

  const std::size_t rank = std::max(test_case.a.dimensions.size(),
                                    test_case.b.dimensions.size());
  std::vector<std::int64_t> expected(rank, 1);
  const std::size_t batch_rank = rank - 2;
  for (std::size_t trailing = 0; trailing < batch_rank; ++trailing) {
    const std::int64_t a_dimension =
        trailing + 2 < test_case.a.dimensions.size()
            ? test_case.a.dimensions[test_case.a.dimensions.size() - 3 -
                                     trailing]
            : 1;
    const std::int64_t b_dimension =
        trailing + 2 < test_case.b.dimensions.size()
            ? test_case.b.dimensions[test_case.b.dimensions.size() - 3 -
                                     trailing]
            : 1;
    if (a_dimension != b_dimension && a_dimension != 1 &&
        b_dimension != 1) {
      throw std::invalid_argument(
          "acDNN MatMul batch dimensions are incompatible");
    }
    expected[batch_rank - 1 - trailing] =
        std::max(a_dimension, b_dimension);
  }
  expected[rank - 2] =
      test_case.a.dimensions[test_case.a.dimensions.size() - 2];
  expected[rank - 1] = test_case.b.dimensions.back();
  if (test_case.output.dimensions != expected) {
    throw std::invalid_argument("acDNN MatMul output shape is invalid");
  }
}

class BackendDescriptor final {
 public:
  explicit BackendDescriptor(acdnnBackendDescriptorType_t type) {
    check_acdnn(acdnnBackendCreateDescriptor(type, &descriptor_),
                "acdnnBackendCreateDescriptor(MatMul)");
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
    check_acdnn(
        acdnnBackendSetAttribute(descriptor_, name, type, count, values),
        description);
  }

  void finalize(std::string_view description) {
    check_acdnn(acdnnBackendFinalize(descriptor_), description);
  }

 private:
  acdnnBackendDescriptor_t descriptor_ = nullptr;
};

std::size_t storage_element_count(
    const flagdnn::testing::TestTensor &tensor) {
  std::uint64_t result = 1;
  for (std::size_t axis = 0; axis < tensor.dimensions.size(); ++axis) {
    const auto dimension = static_cast<std::uint64_t>(tensor.dimensions[axis]);
    const auto stride = static_cast<std::uint64_t>(tensor.strides[axis]);
    if (dimension == 0 || stride == 0 ||
        dimension - 1 >
            (std::numeric_limits<std::uint64_t>::max() - result) / stride) {
      throw std::invalid_argument("acDNN MatMul tensor storage overflows");
    }
    result += (dimension - 1) * stride;
  }
  if (result > std::numeric_limits<std::size_t>::max()) {
    throw std::invalid_argument("acDNN MatMul tensor storage is too large");
  }
  return static_cast<std::size_t>(result);
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
  throw std::invalid_argument("acDNN MatMul data type is unsupported");
}

flagdnn::testing::TestTensor padded_tensor(
    const flagdnn::testing::TestTensor &tensor, std::size_t rank) {
  if (tensor.dimensions.size() > rank) {
    throw std::invalid_argument("acDNN MatMul tensor rank is invalid");
  }
  flagdnn::testing::TestTensor result = tensor;
  const std::size_t storage = storage_element_count(tensor);
  if (storage > static_cast<std::size_t>(
                    std::numeric_limits<std::int64_t>::max())) {
    throw std::invalid_argument("acDNN MatMul leading stride overflows");
  }
  const std::size_t leading = rank - tensor.dimensions.size();
  result.dimensions.insert(result.dimensions.begin(), leading, 1);
  result.strides.insert(result.strides.begin(), leading,
                        static_cast<std::int64_t>(storage));
  return result;
}

void build_tensor_descriptor(BackendDescriptor &descriptor,
                             const flagdnn::testing::TestTensor &tensor) {
  if (tensor.uid <= 0 || tensor.dimensions.size() != tensor.strides.size() ||
      tensor.dimensions.size() < 3 || tensor.dimensions.size() > 8) {
    throw std::invalid_argument(
        "qualified acDNN MatMul tensor must have rank 3 through 8");
  }
  if (std::ranges::any_of(tensor.dimensions,
                          [](std::int64_t value) { return value <= 0; }) ||
      std::ranges::any_of(tensor.strides,
                          [](std::int64_t value) { return value <= 0; })) {
    throw std::invalid_argument("acDNN MatMul tensor geometry is invalid");
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
        "acDNN MatMul tensor alignment is below its scalar size");
  }
  descriptor.set(ACDNN_ATTR_TENSOR_DATA_TYPE, ACDNN_TYPE_DATA_TYPE, 1,
                 &data_type,
                 "acdnnBackendSetAttribute(MatMul tensor data type)");
  descriptor.set(ACDNN_ATTR_TENSOR_DIMENSIONS, ACDNN_TYPE_INT64,
                 static_cast<std::int64_t>(tensor.dimensions.size()),
                 tensor.dimensions.data(),
                 "acdnnBackendSetAttribute(MatMul tensor dimensions)");
  descriptor.set(ACDNN_ATTR_TENSOR_STRIDES, ACDNN_TYPE_INT64,
                 static_cast<std::int64_t>(tensor.strides.size()),
                 tensor.strides.data(),
                 "acdnnBackendSetAttribute(MatMul tensor strides)");
  descriptor.set(ACDNN_ATTR_TENSOR_UNIQUE_ID, ACDNN_TYPE_INT64, 1,
                 &tensor.uid,
                 "acdnnBackendSetAttribute(MatMul tensor uid)");
  descriptor.set(ACDNN_ATTR_TENSOR_BYTE_ALIGNMENT, ACDNN_TYPE_INT64, 1,
                 &alignment,
                 "acdnnBackendSetAttribute(MatMul tensor alignment)");
  descriptor.finalize("acdnnBackendFinalize(MatMul tensor)");
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
          "acDNN MatMul binding is duplicate or null");
    }
    result = binding.device_pointer;
  }
  if (result == nullptr) {
    throw std::invalid_argument("acDNN MatMul binding is missing");
  }
  return result;
}

class AcdnnMatmulSegment final : public flagdnn::testing::MatmulExecutable {
 public:
  AcdnnMatmulSegment(flagdnn::testing::MatmulTestCase test_case,
                     const CapabilityRecord &capability)
      : test_case_(std::move(test_case)),
        matmul_(ACDNN_BACKEND_MATMUL_DESCRIPTOR),
        a_(ACDNN_BACKEND_TENSOR_DESCRIPTOR),
        b_(ACDNN_BACKEND_TENSOR_DESCRIPTOR),
        output_(ACDNN_BACKEND_TENSOR_DESCRIPTOR),
        operation_(ACDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR),
        operation_graph_(ACDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR),
        heuristics_(ACDNN_BACKEND_ENGINEHEUR_DESCRIPTOR),
        engine_config_(ACDNN_BACKEND_ENGINECFG_DESCRIPTOR),
        plan_(ACDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR) {
    const ReferenceSelection selection = select_reference(capability);
    if (!std::holds_alternative<ReferencePlan>(selection)) {
      throw std::invalid_argument("unsupported MatMul reached acDNN reference");
    }
    const ReferencePlan &reference = std::get<ReferencePlan>(selection);
    if (reference.path != ReferencePath::kBackendDescriptor ||
        reference.primitives !=
            std::vector<std::string>{std::string(kPrimitive)}) {
      throw std::invalid_argument("THead acDNN MatMul reference plan mismatch");
    }
    validate_matmul_geometry(test_case_);
    if ((test_case_.a.data_type != FLAGDNN_DATA_FLOAT32 &&
         test_case_.a.data_type != FLAGDNN_DATA_FLOAT16 &&
         test_case_.a.data_type != FLAGDNN_DATA_BFLOAT16) ||
        test_case_.b.data_type != test_case_.a.data_type ||
        test_case_.output.data_type != test_case_.a.data_type) {
      throw std::invalid_argument(
          "qualified acDNN MatMul reference requires matching floating "
          "types");
    }

    const std::size_t rank =
        std::max<std::size_t>(3, test_case_.output.dimensions.size());
    const auto a_specification = padded_tensor(test_case_.a, rank);
    const auto b_specification = padded_tensor(test_case_.b, rank);
    const auto output_specification = padded_tensor(test_case_.output, rank);
    build_tensor_descriptor(a_, a_specification);
    build_tensor_descriptor(b_, b_specification);
    build_tensor_descriptor(output_, output_specification);

    constexpr acdnnDataType_t kComputeType = ACDNN_DATA_FLOAT;
    matmul_.set(ACDNN_ATTR_MATMUL_COMP_TYPE, ACDNN_TYPE_DATA_TYPE, 1,
                &kComputeType,
                "acdnnBackendSetAttribute(MatMul compute type)");
    matmul_.finalize("acdnnBackendFinalize(MatMul descriptor)");

    acdnnBackendDescriptor_t matmul = matmul_.get();
    acdnnBackendDescriptor_t a = a_.get();
    acdnnBackendDescriptor_t b = b_.get();
    acdnnBackendDescriptor_t output = output_.get();
    operation_.set(ACDNN_ATTR_OPERATION_MATMUL_DESC,
                   ACDNN_TYPE_BACKEND_DESCRIPTOR, 1, &matmul,
                   "acdnnBackendSetAttribute(MatMul descriptor)");
    operation_.set(ACDNN_ATTR_OPERATION_MATMUL_ADESC,
                   ACDNN_TYPE_BACKEND_DESCRIPTOR, 1, &a,
                   "acdnnBackendSetAttribute(MatMul A)");
    operation_.set(ACDNN_ATTR_OPERATION_MATMUL_BDESC,
                   ACDNN_TYPE_BACKEND_DESCRIPTOR, 1, &b,
                   "acdnnBackendSetAttribute(MatMul B)");
    operation_.set(ACDNN_ATTR_OPERATION_MATMUL_CDESC,
                   ACDNN_TYPE_BACKEND_DESCRIPTOR, 1, &output,
                   "acdnnBackendSetAttribute(MatMul output)");
    operation_.finalize("acdnnBackendFinalize(MatMul operation)");

    acdnnBackendDescriptor_t operation = operation_.get();
    acdnnHandle_t handle = handle_.get();
    operation_graph_.set(ACDNN_ATTR_OPERATIONGRAPH_OPS,
                         ACDNN_TYPE_BACKEND_DESCRIPTOR, 1, &operation,
                         "acdnnBackendSetAttribute(MatMul graph operations)");
    operation_graph_.set(ACDNN_ATTR_OPERATIONGRAPH_HANDLE,
                         ACDNN_TYPE_HANDLE, 1, &handle,
                         "acdnnBackendSetAttribute(MatMul graph handle)");
    operation_graph_.finalize("acdnnBackendFinalize(MatMul graph)");

    acdnnBackendDescriptor_t graph = operation_graph_.get();
    constexpr acdnnBackendHeurMode_t kMode = ACDNN_HEUR_MODE_A;
    heuristics_.set(ACDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                    ACDNN_TYPE_BACKEND_DESCRIPTOR, 1, &graph,
                    "acdnnBackendSetAttribute(MatMul heuristics graph)");
    heuristics_.set(ACDNN_ATTR_ENGINEHEUR_MODE, ACDNN_TYPE_HEUR_MODE, 1,
                    &kMode,
                    "acdnnBackendSetAttribute(MatMul heuristics mode)");
    heuristics_.finalize("acdnnBackendFinalize(MatMul heuristics)");

    std::int64_t result_count = 0;
    check_acdnn(acdnnBackendGetAttribute(
                    heuristics_.get(), ACDNN_ATTR_ENGINEHEUR_RESULTS,
                    ACDNN_TYPE_BACKEND_DESCRIPTOR, 0, &result_count, nullptr),
                "acdnnBackendGetAttribute(MatMul heuristics count)");
    if (result_count <= 0) {
      throw std::runtime_error(
          "acDNN MatMul heuristics returned no engine configuration");
    }
    acdnnBackendDescriptor_t engine_config = engine_config_.get();
    std::int64_t returned_count = 0;
    check_acdnn(acdnnBackendGetAttribute(
                    heuristics_.get(), ACDNN_ATTR_ENGINEHEUR_RESULTS,
                    ACDNN_TYPE_BACKEND_DESCRIPTOR, 1, &returned_count,
                    &engine_config),
                "acdnnBackendGetAttribute(MatMul heuristics result)");
    if (returned_count <= 0) {
      throw std::runtime_error(
          "acDNN MatMul heuristics returned no usable configuration: "
          "advertised=" +
          std::to_string(result_count) +
          " returned=" + std::to_string(returned_count));
    }

    plan_.set(ACDNN_ATTR_EXECUTION_PLAN_HANDLE, ACDNN_TYPE_HANDLE, 1,
              &handle, "acdnnBackendSetAttribute(MatMul plan handle)");
    plan_.set(ACDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
              ACDNN_TYPE_BACKEND_DESCRIPTOR, 1, &engine_config,
              "acdnnBackendSetAttribute(MatMul plan engine)");
    plan_.finalize("acdnnBackendFinalize(MatMul plan)");

    std::int64_t workspace = 0;
    std::int64_t workspace_count = 0;
    check_acdnn(acdnnBackendGetAttribute(
                    plan_.get(), ACDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE,
                    ACDNN_TYPE_INT64, 1, &workspace_count, &workspace),
                "acdnnBackendGetAttribute(MatMul workspace)");
    if (workspace_count != 1 || workspace < 0 ||
        static_cast<std::uint64_t>(workspace) >
            std::numeric_limits<std::size_t>::max()) {
      throw std::runtime_error("acDNN MatMul workspace size is invalid");
    }
    workspace_size_ = static_cast<std::size_t>(workspace);
  }

  [[nodiscard]] std::size_t workspace_size() const noexcept override {
    return workspace_size_;
  }

  void execute(std::span<const flagdnnBinding_t> bindings, void *workspace,
               std::size_t workspace_size,
               flagdnnStream_t stream) override {
    if (bindings.size() != 3 || stream == nullptr ||
        workspace_size != workspace_size_ ||
        (workspace_size_ != 0 && workspace == nullptr)) {
      throw std::invalid_argument("acDNN MatMul execution arguments are invalid");
    }
    handle_.set_stream(reinterpret_cast<hggcStream_t>(stream));
    std::vector<std::int64_t> uids = {
        test_case_.a.uid, test_case_.b.uid, test_case_.output.uid};
    std::vector<void *> pointers = {
        required_pointer(bindings, test_case_.a.uid),
        required_pointer(bindings, test_case_.b.uid),
        required_pointer(bindings, test_case_.output.uid)};
    BackendDescriptor variant_pack(ACDNN_BACKEND_VARIANT_PACK_DESCRIPTOR);
    variant_pack.set(ACDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
                     ACDNN_TYPE_VOID_PTR,
                     static_cast<std::int64_t>(pointers.size()),
                     pointers.data(),
                     "acdnnBackendSetAttribute(MatMul variant pointers)");
    variant_pack.set(ACDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, ACDNN_TYPE_INT64,
                     static_cast<std::int64_t>(uids.size()), uids.data(),
                     "acdnnBackendSetAttribute(MatMul variant uids)");
    variant_pack.set(ACDNN_ATTR_VARIANT_PACK_WORKSPACE, ACDNN_TYPE_VOID_PTR,
                     1, &workspace,
                     "acdnnBackendSetAttribute(MatMul variant workspace)");
    const std::int64_t workspace_bytes =
        static_cast<std::int64_t>(workspace_size);
    variant_pack.set(ACDNN_ATTR_VARIANT_PACK_WORKSPACE_SIZE,
                     ACDNN_TYPE_INT64, 1, &workspace_bytes,
                     "acdnnBackendSetAttribute(MatMul workspace size)");
    variant_pack.finalize("acdnnBackendFinalize(MatMul variant pack)");
    check_acdnn(
        acdnnBackendExecute(handle_.get(), plan_.get(), variant_pack.get()),
        kPrimitive);
  }

 private:
  flagdnn::testing::MatmulTestCase test_case_;
  AcdnnHandle handle_;
  BackendDescriptor matmul_;
  BackendDescriptor a_;
  BackendDescriptor b_;
  BackendDescriptor output_;
  BackendDescriptor operation_;
  BackendDescriptor operation_graph_;
  BackendDescriptor heuristics_;
  BackendDescriptor engine_config_;
  BackendDescriptor plan_;
  std::size_t workspace_size_ = 0;
};

constexpr std::string_view kSegmentedBatchPrimitive =
    "acdnnBackendExecute(MATMUL,batch-segments)";

std::size_t checked_add_byte_offset(std::size_t base,
                                    std::size_t element_offset,
                                    flagdnnDataType_t data_type) {
  const std::size_t scalar_bytes = element_size(data_type);
  if (element_offset >
          std::numeric_limits<std::size_t>::max() / scalar_bytes ||
      base > std::numeric_limits<std::size_t>::max() -
                 element_offset * scalar_bytes) {
    throw std::overflow_error("segmented acDNN MatMul byte offset overflows");
  }
  return base + element_offset * scalar_bytes;
}

flagdnn::testing::TestTensor matrix_segment_tensor(
    const flagdnn::testing::TestTensor &tensor,
    std::size_t element_offset) {
  const std::size_t rank = tensor.dimensions.size();
  if (rank < 2) {
    throw std::invalid_argument(
        "segmented acDNN MatMul tensor rank is invalid");
  }
  const std::int64_t rows = tensor.dimensions[rank - 2];
  const std::int64_t columns = tensor.dimensions[rank - 1];
  const std::int64_t row_stride = tensor.strides[rank - 2];
  const std::int64_t column_stride = tensor.strides[rank - 1];
  const std::uint64_t span =
      1U + static_cast<std::uint64_t>(rows - 1) *
               static_cast<std::uint64_t>(row_stride) +
      static_cast<std::uint64_t>(columns - 1) *
          static_cast<std::uint64_t>(column_stride);
  if (span >
      static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
    throw std::overflow_error(
        "segmented acDNN MatMul matrix span overflows");
  }
  flagdnn::testing::TestTensor result = tensor;
  result.dimensions = {1, rows, columns};
  result.strides = {static_cast<std::int64_t>(span), row_stride,
                    column_stride};
  result.binding_byte_offset = checked_add_byte_offset(
      tensor.binding_byte_offset, element_offset, tensor.data_type);
  return result;
}

std::size_t tensor_batch_offset(
    const flagdnn::testing::TestTensor &tensor, std::size_t output_rank,
    std::span<const std::int64_t> coordinates) {
  const flagdnn::testing::TestTensor padded =
      padded_tensor(tensor, output_rank);
  if (coordinates.size() != output_rank - 2) {
    throw std::invalid_argument(
        "segmented acDNN MatMul coordinate rank is invalid");
  }
  std::size_t result = 0;
  for (std::size_t axis = 0; axis < coordinates.size(); ++axis) {
    const std::int64_t coordinate =
        padded.dimensions[axis] == 1 ? 0 : coordinates[axis];
    const std::uint64_t contribution =
        static_cast<std::uint64_t>(coordinate) *
        static_cast<std::uint64_t>(padded.strides[axis]);
    if (contribution > std::numeric_limits<std::size_t>::max() - result) {
      throw std::overflow_error(
          "segmented acDNN MatMul tensor offset overflows");
    }
    result += static_cast<std::size_t>(contribution);
  }
  return result;
}

class AcdnnSegmentedBatchMatmul final
    : public flagdnn::testing::MatmulExecutable {
 public:
  AcdnnSegmentedBatchMatmul(
      flagdnn::testing::MatmulTestCase test_case,
      const CapabilityRecord &capability)
      : test_case_(std::move(test_case)) {
    validate_matmul_geometry(test_case_);
    const ReferenceSelection selection = select_reference(capability);
    if (!std::holds_alternative<ReferencePlan>(selection) ||
        std::get<ReferencePlan>(selection).path !=
            ReferencePath::kBackendDescriptor ||
        std::get<ReferencePlan>(selection).primitives !=
            std::vector<std::string>{std::string(kSegmentedBatchPrimitive)} ||
        test_case_.output.dimensions.size() <= 3 ||
        test_case_.a.data_type != test_case_.b.data_type ||
        test_case_.a.data_type != test_case_.output.data_type) {
      throw std::invalid_argument(
          "segmented acDNN MatMul reference plan mismatch");
    }
    const std::size_t rank = test_case_.output.dimensions.size();
    std::size_t batch_elements = 1;
    for (std::size_t axis = 0; axis < rank - 2; ++axis) {
      const std::int64_t dimension = test_case_.output.dimensions[axis];
      if (dimension <= 0 ||
          static_cast<std::uint64_t>(dimension) >
              std::numeric_limits<std::size_t>::max() / batch_elements) {
        throw std::overflow_error(
            "segmented acDNN MatMul batch size overflows");
      }
      batch_elements *= static_cast<std::size_t>(dimension);
    }

    CapabilityRecord segment_capability = capability;
    segment_capability.reference_plan = {std::string(kPrimitive)};
    std::vector<std::int64_t> coordinates(rank - 2);
    segments_.reserve(batch_elements);
    for (std::size_t batch = 0; batch < batch_elements; ++batch) {
      std::size_t remaining = batch;
      for (std::size_t axis = rank - 2; axis != 0; --axis) {
        const std::size_t current = axis - 1;
        const std::size_t dimension = static_cast<std::size_t>(
            test_case_.output.dimensions[current]);
        coordinates[current] =
            static_cast<std::int64_t>(remaining % dimension);
        remaining /= dimension;
      }
      const std::size_t a_offset =
          tensor_batch_offset(test_case_.a, rank, coordinates);
      const std::size_t b_offset =
          tensor_batch_offset(test_case_.b, rank, coordinates);
      const std::size_t output_offset =
          tensor_batch_offset(test_case_.output, rank, coordinates);
      flagdnn::testing::MatmulTestCase segment_case = test_case_;
      segment_case.a = matrix_segment_tensor(test_case_.a, a_offset);
      segment_case.b = matrix_segment_tensor(test_case_.b, b_offset);
      segment_case.output =
          matrix_segment_tensor(test_case_.output, output_offset);
      auto executable = std::make_unique<AcdnnMatmulSegment>(
          std::move(segment_case), segment_capability);
      workspace_size_ =
          std::max(workspace_size_, executable->workspace_size());
      segments_.push_back(
          {.a_offset = a_offset,
           .b_offset = b_offset,
           .output_offset = output_offset,
           .executable = std::move(executable)});
    }
  }

  void prepare(std::span<const flagdnnBinding_t> bindings,
               flagdnnStream_t stream) override {
    for (Segment &segment : segments_) {
      const std::array<flagdnnBinding_t, 3> adjusted =
          adjusted_bindings(bindings, segment);
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
          "segmented acDNN MatMul workspace does not match plan");
    }
    for (Segment &segment : segments_) {
      const std::array<flagdnnBinding_t, 3> adjusted =
          adjusted_bindings(bindings, segment);
      segment.executable->execute(
          adjusted, workspace, segment.executable->workspace_size(), stream);
    }
  }

 private:
  struct Segment {
    std::size_t a_offset;
    std::size_t b_offset;
    std::size_t output_offset;
    std::unique_ptr<flagdnn::testing::MatmulExecutable> executable;
  };

  [[nodiscard]] std::array<flagdnnBinding_t, 3> adjusted_bindings(
      std::span<const flagdnnBinding_t> bindings,
      const Segment &segment) const {
    void *const a = required_pointer(bindings, test_case_.a.uid);
    void *const b = required_pointer(bindings, test_case_.b.uid);
    void *const output =
        required_pointer(bindings, test_case_.output.uid);
    if (bindings.size() != 3) {
      throw std::invalid_argument(
          "segmented acDNN MatMul binding count is invalid");
    }
    const std::size_t scalar_bytes = element_size(test_case_.a.data_type);
    return {{
        {test_case_.a.uid,
         static_cast<std::byte *>(a) + segment.a_offset * scalar_bytes},
        {test_case_.b.uid,
         static_cast<std::byte *>(b) + segment.b_offset * scalar_bytes},
        {test_case_.output.uid,
         static_cast<std::byte *>(output) +
             segment.output_offset * scalar_bytes},
    }};
  }

  flagdnn::testing::MatmulTestCase test_case_;
  std::vector<Segment> segments_;
  std::size_t workspace_size_ = 0;
};

}  // namespace

std::unique_ptr<flagdnn::testing::MatmulExecutable>
make_acdnn_matmul_reference(
    const flagdnn::testing::MatmulTestCase &test_case,
    const CapabilityRecord &capability) {
  if (capability.reference_plan ==
      std::vector<std::string>{std::string(kSegmentedBatchPrimitive)}) {
    return std::make_unique<AcdnnSegmentedBatchMatmul>(test_case,
                                                       capability);
  }
  return std::make_unique<AcdnnMatmulSegment>(test_case, capability);
}

}  // namespace flagdnn::validation::thead
