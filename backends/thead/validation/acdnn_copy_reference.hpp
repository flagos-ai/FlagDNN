// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef FLAGDNN_THEAD_ACDNN_COPY_REFERENCE_HPP_
#define FLAGDNN_THEAD_ACDNN_COPY_REFERENCE_HPP_
#include <algorithm>
#include <array>
#include <limits>

#include "acdnn_graph.hpp"
#include "common/layout.hpp"
#include "numeric_types.hpp"

namespace flagdnn::validation::thead {
inline constexpr std::string_view kRawCopyPrimitive =
    "acdnnBackendExecute(POINTWISE_IDENTITY_FWD,raw-int8-storage)";
inline bool requires_raw_copy(flagdnnDataType_t type) {
  return type == FLAGDNN_DATA_INT32 || type == FLAGDNN_DATA_FP8_E4M3 ||
         type == FLAGDNN_DATA_FP8_E5M2 || type == FLAGDNN_DATA_FP8_E8M0;
}
// INT8 values are exactly representable by the SDK's FP32 pointwise math.
// Reinterpret each element as independent bytes: no integer/FP8 arithmetic
// or FP8 conversion is used, and every payload bit is preserved.
class AcdnnRawCopySegment final : public AcdnnGraphReference {
 public:
  AcdnnRawCopySegment(flagdnn::testing::TestTensor input,
                      flagdnn::testing::TestTensor output,
                      std::size_t input_base = 0)
      : input_uid_(input.uid), input_base_(input_base) {
    if (!requires_raw_copy(input.data_type) ||
        output.data_type != input.data_type ||
        input.dimensions != output.dimensions || input.uid == output.uid)
      throw std::invalid_argument("acDNN raw copy tensor mismatch");
    const auto width = element_size(input.data_type);
    for (auto* t : {&input, &output}) {
      for (auto& stride : t->strides) {
        if (stride > std::numeric_limits<std::int64_t>::max() /
                         static_cast<std::int64_t>(width))
          throw std::overflow_error("acDNN raw copy byte stride overflows");
        stride *= width;
      }
      t->dimensions.push_back(width);
      t->strides.push_back(1);
      while (t->dimensions.size() < 4) {
        t->strides.insert(t->strides.begin(),
                          t->strides.front() * t->dimensions.front());
        t->dimensions.insert(t->dimensions.begin(), 1);
      }
    }
    auto& pointwise = descriptor(ACDNN_BACKEND_POINTWISE_DESCRIPTOR);
    pointwise.set(ACDNN_ATTR_POINTWISE_MODE, ACDNN_TYPE_POINTWISE_MODE,
                  ACDNN_POINTWISE_IDENTITY_FWD);
    pointwise.set(ACDNN_ATTR_POINTWISE_MATH_PREC, ACDNN_TYPE_DATA_TYPE,
                  ACDNN_DATA_FLOAT);
    pointwise.finalize();
    auto& operation = descriptor(ACDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR);
    operation.set(ACDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR,
                  ACDNN_TYPE_BACKEND_DESCRIPTOR, pointwise.get());
    operation.set(ACDNN_ATTR_OPERATION_POINTWISE_XDESC,
                  ACDNN_TYPE_BACKEND_DESCRIPTOR,
                  tensor(input, ACDNN_DATA_INT8, 1));
    operation.set(ACDNN_ATTR_OPERATION_POINTWISE_YDESC,
                  ACDNN_TYPE_BACKEND_DESCRIPTOR,
                  tensor(output, ACDNN_DATA_INT8, 1));
    operation.set(ACDNN_ATTR_OPERATION_POINTWISE_ALPHA1, ACDNN_TYPE_FLOAT,
                  1.0F);
    operation.set(ACDNN_ATTR_OPERATION_POINTWISE_ALPHA2, ACDNN_TYPE_FLOAT,
                  1.0F);
    operation.finalize();
    build(operation.get());
  }
  void execute(std::span<const flagdnnBinding_t> bindings, void* workspace,
               std::size_t size, flagdnnStream_t stream) override {
    std::vector<flagdnnBinding_t> adjusted(bindings.begin(), bindings.end());
    for (auto& binding : adjusted)
      if (binding.uid == input_uid_)
        binding.device_pointer =
            static_cast<std::byte*>(binding.device_pointer) + input_base_;
    AcdnnGraphReference::execute(adjusted, workspace, size, stream);
  }

 private:
  std::int64_t input_uid_;
  std::size_t input_base_;
};
// SDK 1400 INT8 identity requires matching packed descriptors. Map the
// copy into maximal contiguous byte spans; each span uses that SDK primitive.
class AcdnnRawCopy final : public flagdnn::testing::TestExecutable {
 public:
  AcdnnRawCopy(const flagdnn::testing::TestTensor& input,
               const flagdnn::testing::TestTensor& output, std::size_t base = 0)
      : input_uid_(input.uid), output_uid_(output.uid) {
    if (input.data_type != output.data_type ||
        input.dimensions != output.dimensions || input.uid == output.uid)
      throw std::invalid_argument("acDNN raw copy shape or dtype mismatch");
    const auto width = element_size(input.data_type);
    std::size_t count = 1;
    for (const auto extent : input.dimensions) count *= extent;
    const auto offset = [](std::size_t index,
                           const flagdnn::testing::TestTensor& tensor) {
      std::size_t result = 0;
      for (std::size_t axis = tensor.dimensions.size(); axis > 0; --axis) {
        result +=
            (index % tensor.dimensions[axis - 1]) * tensor.strides[axis - 1];
        index /= tensor.dimensions[axis - 1];
      }
      return result;
    };
    std::vector<Segment> elements;
    elements.reserve(count);
    for (std::size_t i = 0; i < count; ++i)
      elements.push_back(
          {base + offset(i, input) * width, offset(i, output) * width, width});
    // Traversal order need not match logical dimension order. Coalesce spans
    // in physical order so a transpose view with unchanged storage is a copy.
    std::sort(elements.begin(), elements.end(),
              [](const auto& a, const auto& b) { return a.input < b.input; });
    for (const auto& element : elements) {
      if (!segments_.empty() &&
          segments_.back().input + segments_.back().bytes == element.input &&
          segments_.back().output + segments_.back().bytes == element.output)
        segments_.back().bytes += element.bytes;
      else
        segments_.push_back(element);
    }
    for (const auto& segment : segments_)
      if (!plans_.contains(segment.bytes)) {
        const auto n = static_cast<std::int64_t>(segment.bytes);
        flagdnn::testing::TestTensor x{
            input.uid, FLAGDNN_DATA_FP8_E4M3, {n}, {1}},
            y = x;
        y.uid = output.uid;
        auto plan = std::make_unique<AcdnnRawCopySegment>(x, y);
        workspace_ = std::max(workspace_, plan->workspace_size());
        plans_.emplace(segment.bytes, std::move(plan));
      }
  }
  std::size_t workspace_size() const noexcept override { return workspace_; }
  void execute(std::span<const flagdnnBinding_t> bindings, void* workspace,
               std::size_t size, flagdnnStream_t stream) override {
    if (size < workspace_ || (workspace_ && !workspace))
      throw std::invalid_argument("raw copy workspace mismatch");
    void* input = nullptr;
    void* output = nullptr;
    for (const auto& b : bindings) {
      if (b.uid == input_uid_) input = b.device_pointer;
      if (b.uid == output_uid_) output = b.device_pointer;
    }
    if (bindings.size() != 2 || !input || !output)
      throw std::invalid_argument("raw copy bindings mismatch");
    for (const auto& segment : segments_) {
      auto& plan = *plans_.at(segment.bytes);
      const std::array<flagdnnBinding_t, 2> pair{
          {{input_uid_, static_cast<std::byte*>(input) + segment.input},
           {output_uid_, static_cast<std::byte*>(output) + segment.output}}};
      plan.execute(pair, plan.workspace_size() ? workspace : nullptr,
                   plan.workspace_size(), stream);
    }
  }

 private:
  struct Segment {
    std::size_t input, output, bytes;
  };
  std::int64_t input_uid_, output_uid_;
  std::size_t workspace_ = 0;
  std::vector<Segment> segments_;
  std::map<std::size_t, std::unique_ptr<AcdnnRawCopySegment>> plans_;
};
inline std::unique_ptr<flagdnn::testing::LayoutExecutable>
make_acdnn_raw_layout_reference(const flagdnn::testing::LayoutTestCase& test) {
  using flagdnn::testing::LayoutOperation;
  auto input = test.input, output = test.output;
  std::size_t base = 0;
  if (test.operation == LayoutOperation::kReshape) {
    input.dimensions = output.dimensions;
    input.strides = output.strides;
  } else if (test.operation == LayoutOperation::kTranspose) {
    input.dimensions = output.dimensions;
    for (std::size_t i = 0; i < test.permutation.size(); ++i)
      input.strides[i] = test.input.strides[test.permutation[i]];
  } else {
    input.dimensions = output.dimensions;
    for (std::size_t i = 0; i < input.strides.size(); ++i) {
      base += test.slices[i].first * input.strides[i] *
              element_size(input.data_type);
      input.strides[i] *=
          test.slice_strides.empty() ? 1 : test.slice_strides[i];
    }
  }
  return std::make_unique<AcdnnRawCopy>(input, output, base);
}
}  // namespace flagdnn::validation::thead
#endif
