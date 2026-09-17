/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "capability_cases.hpp"
#include "common/convolution.hpp"
#include "common/dtype_runner.hpp"
#include "common/layout.hpp"
#include "common/matmul.hpp"
#include "common/pointwise.hpp"
#include "common/reduction.hpp"
#include "native.hpp"
#include "precision.hpp"
#include <map>

namespace flagdnn::testing {
namespace {
namespace bm = flagdnn::benchmarking;
class NativeExecutable final : public bm::BenchmarkExecutable {
public:
  explicit NativeExecutable(std::unique_ptr<TestExecutable> value)
      : value_(std::move(value)) {}
  void prepare(std::span<const flagdnnBinding_t> bindings,
               flagdnnStream_t stream) override {
    value_->prepare(bindings, stream);
  }
  std::size_t workspace_size() const noexcept override {
    return value_->workspace_size();
  }
  void execute(std::span<const flagdnnBinding_t> bindings, void *workspace,
               std::size_t bytes, flagdnnStream_t stream) override {
    value_->execute(bindings, workspace, bytes, stream);
  }

private:
  std::unique_ptr<TestExecutable> value_;
};
bm::TensorSpec tensor(const TestTensor &value) {
  return {value.uid, value.data_type, value.dimensions, value.strides,
          value.binding_byte_offset};
}
struct NativeCase {
  std::string name;
  std::vector<bm::TensorSpec> inputs, outputs;
};
struct Catalog {
  std::vector<bm::BenchmarkCase> cases;
  std::map<std::string,
           std::function<std::unique_ptr<TestExecutable>(flagdnn::Handle &)>>
      builders;
  template <class Case, class Build>
  void add(bm::BenchmarkCase specification, Case value, Build build) {
    specification.name = value.name;
    if constexpr (requires { value.absolute_tolerance; }) {
      specification.absolute_tolerance = value.absolute_tolerance;
      specification.relative_tolerance = value.relative_tolerance;
    }
    if constexpr (requires { value.autotune; })
      value.autotune = true;
    builders.emplace(value.name, [value, build](flagdnn::Handle &handle) {
      return build(handle, value);
    });
    cases.push_back(std::move(specification));
  }
  int run(int argc, char **argv, std::string_view operation, bool tf32) const {
#ifdef FLAGDNN_HYGON_CATALOG_PROBE
    // The catalog contract inspects unsupported TF32 shapes without launching.
    tf32 = false;
#endif
    if (tf32) {
      std::vector<NativeCase> skipped;
      for (const auto &value : cases) {
        const auto output_begin =
            value.tensors.end() -
            static_cast<std::ptrdiff_t>(value.output_count);
        skipped.push_back({value.name,
                           {value.tensors.begin(), output_begin},
                           {output_begin, value.tensors.end()}});
      }
      return hygon::skip_cases(
          argc, skipped, operation, true,
          "Hygon implements IEEE dot products; TF32 is unsupported");
    }
    std::string marker(operation);
    std::transform(marker.begin(), marker.end(), marker.begin(),
                   [](unsigned char c) { return std::toupper(c); });
    return bm::run_hygon_benchmark_suite(
        argc, argv, cases, "FLAGDNN_" + marker + "_BENCHMARK",
        [&](flagdnn::Handle &handle, const bm::BenchmarkCase &value) {
          return std::make_unique<NativeExecutable>(
              builders.at(value.name)(handle));
        });
  }
};
} // namespace

int run_native_dtype_benchmark(int argc, char **argv,
                               std::string_view operation,
                               std::string_view category) {
  Catalog catalog;
  if (category == "boolean" ||
      (category == "copy" && operation == "identity")) {
    PointwiseCaseDefinition definition;
    definition.operation_name = operation;
    if (operation == "identity")
      definition.mode = FLAGDNN_POINTWISE_IDENTITY;
    else if (operation == "logical_not")
      definition.mode = FLAGDNN_POINTWISE_LOGICAL_NOT;
    else if (operation == "logical_and")
      definition.mode = FLAGDNN_POINTWISE_LOGICAL_AND;
    else if (operation == "logical_or")
      definition.mode = FLAGDNN_POINTWISE_LOGICAL_OR;
    else
      throw std::invalid_argument("unknown pointwise dtype benchmark");
    definition.input_domain = category == "boolean"
                                  ? PointwiseInputDomain::kLogical
                                  : PointwiseInputDomain::kReal;
    const auto cases = operation == "logical_and" || operation == "logical_or"
                           ? make_binary_pointwise_cases(definition)
                           : make_unary_pointwise_cases(definition);
    for (const auto &value : cases) {
      const auto type = value.inputs.front().data_type;
      if ((type == FLAGDNN_DATA_INT32 &&
           value.mode != FLAGDNN_POINTWISE_IDENTITY) ||
          type == FLAGDNN_DATA_FP8_E8M0)
        continue;
      bm::BenchmarkCase spec;
      spec.operation = bm::Operation::kPointwise;
      spec.pointwise_mode = value.mode;
      spec.pointwise_attributes = value.attributes;
      spec.add_alpha = value.alpha;
      spec.input_domain = category == "boolean" ? bm::InputDomain::kLogical
                                                : bm::InputDomain::kReal;
      for (const auto &input : value.inputs)
        spec.tensors.push_back(tensor(input));
      spec.tensors.push_back(tensor(value.output));
      catalog.add(spec, value, build_flagdnn_pointwise);
    }
  } else if (category == "copy") {
    if (operation != "reshape" && operation != "transpose" &&
        operation != "slice")
      throw std::invalid_argument("unknown layout dtype benchmark");
    const auto mode = operation == "reshape"     ? LayoutOperation::kReshape
                      : operation == "transpose" ? LayoutOperation::kTranspose
                                                 : LayoutOperation::kSlice;
    for (const auto &value : make_layout_cases(mode)) {
      if (value.input.data_type != FLAGDNN_DATA_FLOAT32 &&
          value.input.data_type != FLAGDNN_DATA_FLOAT16 &&
          value.input.data_type != FLAGDNN_DATA_BFLOAT16)
        continue;
      bm::BenchmarkCase spec;
      spec.operation = operation == "reshape"     ? bm::Operation::kReshape
                       : operation == "transpose" ? bm::Operation::kTranspose
                                                  : bm::Operation::kSlice;
      spec.tensors = {tensor(value.input), tensor(value.output)};
      spec.reshape = {value.output.dimensions, value.output.strides, true};
      spec.transpose.permutation = value.permutation;
      spec.slice = {value.slices, value.slice_strides};
      catalog.add(spec, value, build_flagdnn_layout);
    }
  } else if (category == "precision" && operation == "matmul") {
    for (const auto &value : make_matmul_cases()) {
      if (!value.input_precision ||
          value.input_precision !=
              validation::hygon::selected_input_precision() ||
          value.output.dimensions.size() > 3)
        continue;
      bm::BenchmarkCase spec;
      spec.operation = bm::Operation::kMatmul;
      spec.tensors = {tensor(value.a), tensor(value.b), tensor(value.output)};
      catalog.add(spec, value, build_flagdnn_matmul);
    }
  } else if (category == "precision") {
    if (operation != "conv_fprop" && operation != "conv_dgrad" &&
        operation != "conv_wgrad")
      throw std::invalid_argument("unknown precision benchmark operation");
    const auto direction =
        operation == "conv_fprop"   ? ConvolutionDirection::kFprop
        : operation == "conv_dgrad" ? ConvolutionDirection::kDgrad
                                    : ConvolutionDirection::kWgrad;
    for (const auto &value : make_convolution_cases(direction)) {
      if (!value.input_precision ||
          value.input_precision !=
              validation::hygon::selected_input_precision())
        continue;
      bm::BenchmarkCase spec;
      if (direction == ConvolutionDirection::kFprop) {
        spec.operation = bm::Operation::kConvolutionFprop;
        spec.tensors = {tensor(value.x), tensor(value.w), tensor(value.y)};
      } else if (direction == ConvolutionDirection::kDgrad) {
        spec.operation = bm::Operation::kConvolutionDgrad;
        spec.tensors = {tensor(value.y), tensor(value.w), tensor(value.x)};
      } else {
        spec.operation = bm::Operation::kConvolutionWgrad;
        spec.tensors = {tensor(value.y), tensor(value.x), tensor(value.w)};
      }
      spec.convolution = {static_cast<std::int32_t>(value.stride.size()),
                          value.pre_padding,
                          value.post_padding,
                          value.stride,
                          value.dilation,
                          value.groups,
                          value.mode == ConvolutionMode::kConvolution
                              ? bm::ConvolutionMode::kConvolution
                              : bm::ConvolutionMode::kCrossCorrelation};
      catalog.add(spec, value, build_flagdnn_convolution);
    }
  } else if (category == "fp32_output" && operation == "reduction") {
    for (const auto &value : make_reduction_cases()) {
      // Match NVIDIA's fp32_output group; do not invent a smaller Hygon shape
      // set.
      const bool selected =
          value.input.data_type == FLAGDNN_DATA_BFLOAT16
              ? value.input.dimensions.size() == 2 &&
                    value.output.data_type == FLAGDNN_DATA_FLOAT32 &&
                    value.mode == FLAGDNN_REDUCTION_ADD
              : value.input.data_type == FLAGDNN_DATA_FLOAT32 ||
                    value.input.data_type == FLAGDNN_DATA_FLOAT16;
      if (!selected || value.name.find("_to_fp32_") == std::string::npos)
        continue;
      bm::BenchmarkCase spec;
      spec.operation = bm::Operation::kReduction;
      spec.tensors = {tensor(value.input), tensor(value.output)};
      spec.reduction_mode = value.mode;
      spec.reduction_axis = value.axis;
      spec.keep_dimensions = value.keep_dimensions;
      catalog.add(spec, value, build_flagdnn_reduction);
    }
  } else
    throw std::invalid_argument("unknown dtype benchmark category");
  return catalog.run(argc, argv, operation,
                     category == "precision" &&
                         validation::hygon::selected_input_precision() == 2);
}
} // namespace flagdnn::testing
