// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string_view>
#include <type_traits>

#include "acdnn_copy_reference.hpp"
#include "acdnn_graph.hpp"
#include "acdnn_reference.hpp"
#include "backend_pointwise_reference.hpp"
#include "tensor_io.hpp"
#include "benchmark/common/case.hpp"
#include "reference/cpu/benchmark.hpp"
#include "reference/cpu/matrix.hpp"
#include "reference/cpu/pointwise.hpp"

namespace tv = flagdnn::validation::thead;
namespace cpu = flagdnn::reference::cpu;
namespace bm = flagdnn::benchmarking;
using flagdnn::testing::TestTensor;

void probe(acdnnPointwiseMode_t mode, std::string_view name,
           bool expect_not_supported) {
  const bool is_mod = mode == ACDNN_POINTWISE_MOD;
  const bool unary = mode == ACDNN_POINTWISE_LOGICAL_NOT;
  const bool approximate_backward =
      mode == ACDNN_POINTWISE_GELU_APPROX_TANH_BWD;
  const auto dtype = (is_mod || approximate_backward) ? FLAGDNN_DATA_FLOAT32
                                                      : FLAGDNN_DATA_BOOLEAN;
  TestTensor x{1, dtype, {1, 1, 1, 16}, {16, 16, 16, 1}};
  TestTensor y = x;
  y.uid = 2;
  TestTensor z = x;
  z.uid = 3;
  tv::BackendPointwiseReferenceSpecification specification;
  specification.mode = mode;
  specification.inputs = {x};
  if (!unary) specification.inputs.push_back(y);
  specification.output = z;
  specification.primitive = std::string(name);
  try {
    auto reference = tv::make_acdnn_backend_pointwise_reference(specification);
    tv::DeviceStream stream;
    tv::DeviceBuffer a(64), b(64), output(64);
    tv::DeviceBuffer workspace(reference->workspace_size());
    std::array<std::uint8_t, 16> left{}, right{}, actual{}, expected{};
    std::array<float, 16> mod_left{}, mod_right{};
    for (std::size_t i = 0; i < left.size(); ++i) {
      left[i] = (i % 4) / 2;
      right[i] = i % 2;
      mod_left[i] = 5.5F;
      mod_right[i] = 2.0F;
    }
    if (!is_mod && !approximate_backward) {
      bm::BenchmarkCase oracle;
      oracle.operation = bm::Operation::kPointwise;
      oracle.pointwise_mode = mode == ACDNN_POINTWISE_LOGICAL_AND
          ? FLAGDNN_POINTWISE_LOGICAL_AND : mode == ACDNN_POINTWISE_LOGICAL_OR
          ? FLAGDNN_POINTWISE_LOGICAL_OR : FLAGDNN_POINTWISE_LOGICAL_NOT;
      oracle.tensors.push_back({1, FLAGDNN_DATA_BOOLEAN, {16}, {1}});
      std::vector<std::vector<float>> inputs{
          std::vector<float>(left.begin(), left.end())};
      if (!unary) {
        oracle.tensors.push_back({2, FLAGDNN_DATA_BOOLEAN, {16}, {1}});
        inputs.emplace_back(right.begin(), right.end());
      }
      oracle.tensors.push_back({3, FLAGDNN_DATA_BOOLEAN, {16}, {1}});
      const auto values = cpu::evaluate_benchmark(oracle, inputs,
          [](std::span<const float> values, flagdnnDataType_t) {
            return std::vector<float>(values.begin(), values.end());
          }).at(0);
      for (std::size_t i = 0; i < expected.size(); ++i)
        expected[i] = static_cast<std::uint8_t>(values.at(i));
    }
    if (is_mod || approximate_backward) {
      tv::copy_to_device_async<float>(a, mod_left, 0, stream.get());
      tv::copy_to_device_async<float>(b, mod_right, 0, stream.get());
    } else {
      tv::copy_to_device_async<std::uint8_t>(a, left, 0, stream.get());
      tv::copy_to_device_async<std::uint8_t>(b, right, 0, stream.get());
    }
    const std::array<flagdnnBinding_t, 3> bindings{
        {{1, a.data()}, {2, b.data()}, {3, output.data()}}};
    reference->prepare(bindings, stream.opaque());
    reference->execute(bindings, workspace.data(), workspace.size(),
                       stream.opaque());
    tv::copy_from_device_async<std::uint8_t>(actual, output, 0, stream.get());
    tv::check_driver(cuStreamSynchronize(stream.get()),
                     "gap probe synchronize");
    if (expect_not_supported || actual == expected) {
      throw std::runtime_error(std::string(name) +
                               " native primitive changed: requalify before "
                               "replacing the acDNN composite reference");
    }
    std::cout << name << ": confirmed acdnn_semantic_mismatch\n";
  } catch (const tv::AcdnnStatusError& error) {
    const bool rejected_mode =
        error.status() == ACDNN_STATUS_NOT_SUPPORTED ||
        (error.status() == ACDNN_STATUS_BAD_PARAM &&
         std::string_view(error.what())
             .starts_with("acdnnBackendFinalize(pointwise operation)"));
    if (!expect_not_supported || !rejected_mode) {
      throw;
    }
    std::cout << name << ": confirmed descriptor rejection (" << error.what()
              << ")\n";
  }
}

// These use raw SDK descriptors so a validation-wrapper dtype check cannot
// accidentally serve as evidence that the SDK itself rejects INT32.
class Int32PointwiseProbe final : public tv::AcdnnGraphReference {
 public:
  Int32PointwiseProbe(acdnnPointwiseMode_t mode, acdnnDataType_t math) {
    TestTensor x{1, FLAGDNN_DATA_INT32, {1, 1, 4, 16}, {64, 64, 16, 1}};
    auto y = x;
    auto output = x;
    y.uid = 2;
    output.uid = 3;
    auto& pointwise = descriptor(ACDNN_BACKEND_POINTWISE_DESCRIPTOR);
    pointwise.set(ACDNN_ATTR_POINTWISE_MODE, ACDNN_TYPE_POINTWISE_MODE, mode);
    pointwise.set(ACDNN_ATTR_POINTWISE_MATH_PREC, ACDNN_TYPE_DATA_TYPE, math);
    pointwise.finalize();
    auto& operation = descriptor(ACDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR);
    operation.set(ACDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR,
                  ACDNN_TYPE_BACKEND_DESCRIPTOR, pointwise.get());
    operation.set(ACDNN_ATTR_OPERATION_POINTWISE_XDESC,
                  ACDNN_TYPE_BACKEND_DESCRIPTOR, tensor(x, ACDNN_DATA_INT32));
    if (mode != ACDNN_POINTWISE_IDENTITY_FWD)
      operation.set(ACDNN_ATTR_OPERATION_POINTWISE_BDESC,
                    ACDNN_TYPE_BACKEND_DESCRIPTOR, tensor(y, ACDNN_DATA_INT32));
    const bool comparison =
        mode >= ACDNN_POINTWISE_CMP_EQ && mode <= ACDNN_POINTWISE_CMP_LE;
    operation.set(
        ACDNN_ATTR_OPERATION_POINTWISE_YDESC, ACDNN_TYPE_BACKEND_DESCRIPTOR,
        tensor(output, comparison ? ACDNN_DATA_BOOL : ACDNN_DATA_INT32));
    operation.set(ACDNN_ATTR_OPERATION_POINTWISE_ALPHA1, ACDNN_TYPE_FLOAT,
                  1.0F);
    operation.set(ACDNN_ATTR_OPERATION_POINTWISE_ALPHA2, ACDNN_TYPE_FLOAT,
                  1.0F);
    operation.finalize();
    build(operation.get());
  }
};

void probe_raw_copy_and_conversion() {
  tv::DeviceStream stream;
  for (auto type :
       {FLAGDNN_DATA_FLOAT32, FLAGDNN_DATA_FLOAT16, FLAGDNN_DATA_BFLOAT16,
        FLAGDNN_DATA_BOOLEAN, FLAGDNN_DATA_INT32, FLAGDNN_DATA_FP8_E4M3,
        FLAGDNN_DATA_FP8_E5M2, FLAGDNN_DATA_FP8_E8M0}) {
    const auto width = tv::element_size(type);
    TestTensor x{1, type, {7, 3}, {1, 7}}, y{2, type, {7, 3}, {3, 1}};
    tv::AcdnnRawCopy reference(x, y);
    tv::DeviceBuffer a(21 * width), b(21 * width),
        workspace(reference.workspace_size());
    std::vector<std::uint8_t> input(21 * width), actual(input.size());
    for (std::size_t i = 0; i < input.size(); ++i)
      input[i] = (i * 37 + 83) % 256;
    tv::copy_to_device_async<std::uint8_t>(a, input, 0, stream.get());
    const std::array<flagdnnBinding_t, 2> bindings{
        {{1, a.data()}, {2, b.data()}}};
    reference.execute(bindings, workspace.data(), workspace.size(),
                      stream.opaque());
    tv::copy_from_device_async<std::uint8_t>(actual, b, 0, stream.get());
    tv::check_driver(cuStreamSynchronize(stream.get()),
                     "raw copy control synchronize");
    for (std::size_t i = 0; i < 7; ++i)
      for (std::size_t j = 0; j < 3; ++j)
        for (std::size_t k = 0; k < width; ++k)
          if (actual[(i * 3 + j) * width + k] != input[(i + j * 7) * width + k])
            throw std::runtime_error("acDNN byte copy positive control failed");
    std::cout << "raw copy type=" << type << ": exact byte mapping passed\n";
  }
  TestTensor x{1, FLAGDNN_DATA_INT32, {1, 1, 1, 8}, {8, 8, 8, 1}}, y = x;
  y.uid = 2;
  y.data_type = FLAGDNN_DATA_FLOAT32;
  auto reference = tv::make_acdnn_backend_pointwise_reference(
      {.mode = ACDNN_POINTWISE_IDENTITY_FWD,
       .inputs = {x},
       .output = y,
       .primitive = "acdnnBackendExecute(POINTWISE_IDENTITY_FWD,int32-to-fp32-"
                    "control)"});
  const std::array<std::int32_t, 8> input{
      0, 1, -1, 16777217, -16777217, 2147483647, (-2147483647 - 1), 65539};
  std::array<float, 8> actual{};
  tv::DeviceBuffer a(sizeof(input)), b(sizeof(actual)),
      workspace(reference->workspace_size());
  tv::copy_to_device_async<std::int32_t>(a, input, 0, stream.get());
  const std::array<flagdnnBinding_t, 2> bindings{
      {{1, a.data()}, {2, b.data()}}};
  reference->prepare(bindings, stream.opaque());
  reference->execute(bindings, workspace.data(), workspace.size(),
                     stream.opaque());
  tv::copy_from_device_async<float>(actual, b, 0, stream.get());
  tv::check_driver(cuStreamSynchronize(stream.get()),
                   "integer conversion control synchronize");
  for (std::size_t i = 0; i < input.size(); ++i)
    if (actual[i] != static_cast<float>(input[i]))
      throw std::runtime_error(
          "acDNN INT32-to-FP32 conversion positive control failed");
  std::cout << "INT32-to-FP32 conversion: positive control passed\n";
}

class Tf32MatmulProbe final : public tv::AcdnnGraphReference {
 public:
  Tf32MatmulProbe() {
    const TestTensor a{1, FLAGDNN_DATA_FLOAT32, {1, 16, 32}, {512, 32, 1}};
    const TestTensor b{2, FLAGDNN_DATA_FLOAT32, {1, 32, 24}, {768, 24, 1}};
    const TestTensor c{3, FLAGDNN_DATA_FLOAT32, {1, 16, 24}, {384, 24, 1}};
    auto& matmul = descriptor(ACDNN_BACKEND_MATMUL_DESCRIPTOR);
    matmul.set(ACDNN_ATTR_MATMUL_COMP_TYPE, ACDNN_TYPE_DATA_TYPE,
               ACDNN_DATA_TF32);
    matmul.finalize();
    auto& operation = descriptor(ACDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR);
    operation.set(ACDNN_ATTR_OPERATION_MATMUL_DESC,
                  ACDNN_TYPE_BACKEND_DESCRIPTOR, matmul.get());
    operation.set(ACDNN_ATTR_OPERATION_MATMUL_ADESC,
                  ACDNN_TYPE_BACKEND_DESCRIPTOR, tensor(a, ACDNN_DATA_FLOAT));
    operation.set(ACDNN_ATTR_OPERATION_MATMUL_BDESC,
                  ACDNN_TYPE_BACKEND_DESCRIPTOR, tensor(b, ACDNN_DATA_FLOAT));
    operation.set(ACDNN_ATTR_OPERATION_MATMUL_CDESC,
                  ACDNN_TYPE_BACKEND_DESCRIPTOR, tensor(c, ACDNN_DATA_FLOAT));
    operation.finalize();
    build(operation.get());
  }
};

void probe_tf32_matmul() {
  Tf32MatmulProbe reference;
  tv::DeviceStream stream;
  std::array<float, 512> a{};
  std::array<float, 768> b{};
  std::array<float, 384> c{};
  for (std::size_t i = 0; i < a.size(); ++i)
    a[i] = static_cast<float>(static_cast<int>((i * 17) % 61) - 30) / 19.0F;
  for (std::size_t i = 0; i < b.size(); ++i)
    b[i] = static_cast<float>(static_cast<int>((i * 17 + 7) % 61) - 30) / 19.0F;
  tv::DeviceBuffer da(sizeof(a)), db(sizeof(b)), dc(sizeof(c));
  tv::DeviceBuffer workspace(reference.workspace_size());
  tv::copy_to_device_async<float>(da, a, 0, stream.get());
  tv::copy_to_device_async<float>(db, b, 0, stream.get());
  const std::array<flagdnnBinding_t, 3> bindings{
      {{1, da.data()}, {2, db.data()}, {3, dc.data()}}};
  reference.execute(bindings, workspace.data(), workspace.size(),
                    stream.opaque());
  tv::copy_from_device_async<float>(c, dc, 0, stream.get());
  tv::check_driver(cuStreamSynchronize(stream.get()),
                   "TF32 MatMul probe synchronize");
  const auto tf32 = [](float value) {
    return std::bit_cast<float>(std::bit_cast<std::uint32_t>(value) &
                                0xffffe000U);
  };
  // Truncation is the probe's explicit input codec. The reference MatMul
  // receives those quantized values with precision=1, so it does not apply its
  // separate automatic/nearest-even TF32 input policy.
  std::vector<float> quantized_a(a.begin(), a.end()), quantized_b(b.begin(), b.end());
  for (auto& value : quantized_a) value = tf32(value);
  for (auto& value : quantized_b) value = tf32(value);
  const double expected = cpu::evaluate_matmul(
      {{16, 32}, {32, 24}, {16, 24}, FLAGDNN_DATA_FLOAT32, 1},
      quantized_a, quantized_b).at(0);
  const double difference = std::abs(static_cast<double>(c[0]) - expected);
  if (!std::isfinite(c[0]) || std::abs(c[0] - 2.680105925F) > 1.0e-6F ||
      difference <= 5.0e-5 || difference <= 5.0e-5 * std::abs(expected)) {
    throw std::runtime_error(
        "acDNN TF32 MatMul semantics changed; requalify catalog");
  }
  std::cout << "TF32 MatMul: confirmed FP16 operand rounding instead of TF32 "
               "truncation\n";
}

void probe_tf32_convolution() {
  tv::AcdnnHandle handle;
  tv::AcdnnTensorDescriptor x, y;
  x.set(ACDNN_DATA_FLOAT, std::array<int, 4>{1, 8, 7, 9},
        std::array<int, 4>{504, 1, 72, 8});
  y.set(ACDNN_DATA_FLOAT, std::array<int, 4>{1, 8, 4, 5},
        std::array<int, 4>{160, 1, 40, 8});
  acdnnFilterDescriptor_t raw_filter = nullptr;
  tv::check_acdnn(acdnnCreateFilterDescriptor(&raw_filter),
                  "create TF32 filter");
  std::unique_ptr<std::remove_pointer_t<acdnnFilterDescriptor_t>,
                  decltype(&acdnnDestroyFilterDescriptor)>
      filter(raw_filter, acdnnDestroyFilterDescriptor);
  const std::array<int, 4> dimensions{8, 8, 1, 1};
  tv::check_acdnn(
      acdnnSetFilterNdDescriptor(filter.get(), ACDNN_DATA_FLOAT,
                                 ACDNN_TENSOR_NHWC, 4, dimensions.data()),
      "set TF32 filter");
  acdnnConvolutionDescriptor_t raw_convolution = nullptr;
  tv::check_acdnn(acdnnCreateConvolutionDescriptor(&raw_convolution),
                  "create TF32 convolution");
  std::unique_ptr<std::remove_pointer_t<acdnnConvolutionDescriptor_t>,
                  decltype(&acdnnDestroyConvolutionDescriptor)>
      convolution(raw_convolution, acdnnDestroyConvolutionDescriptor);
  const std::array<int, 2> padding{0, 0}, stride{2, 2}, dilation{1, 1};
  tv::check_acdnn(
      acdnnSetConvolutionNdDescriptor(convolution.get(), 2, padding.data(),
                                      stride.data(), dilation.data(),
                                      ACDNN_CROSS_CORRELATION, ACDNN_DATA_TF32),
      "set TF32 convolution");
  for (auto math : {ACDNN_DEFAULT_MATH, ACDNN_TENSOR_OP_MATH, ACDNN_FMA_MATH}) {
    tv::check_acdnn(acdnnSetConvolutionMathType(convolution.get(), math),
                    "set TF32 convolution math type");
    std::size_t bytes = 0;
    const std::array statuses{
        acdnnGetConvolutionForwardWorkspaceSize(
            handle.get(), x.get(), filter.get(), convolution.get(), y.get(),
            ACDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, &bytes),
        acdnnGetConvolutionBackwardDataWorkspaceSize(
            handle.get(), filter.get(), y.get(), convolution.get(), x.get(),
            ACDNN_CONVOLUTION_BWD_DATA_ALGO_0, &bytes),
        acdnnGetConvolutionBackwardFilterWorkspaceSize(
            handle.get(), x.get(), y.get(), convolution.get(), filter.get(),
            ACDNN_CONVOLUTION_BWD_FILTER_ALGO_0, &bytes)};
    for (const auto status : statuses) {
      if (status != ACDNN_STATUS_NOT_SUPPORTED) {
        throw std::runtime_error(
            "acDNN TF32 convolution support changed; requalify catalog");
      }
    }
    std::cout << "TF32 convolution forward/dgrad/wgrad math=" << math
              << ": confirmed ACDNN_STATUS_NOT_SUPPORTED\n";
  }
}

int main() {
  try {
    tv::check_driver(cuInit(0), "cuInit");
    CUdevice device = 0;
    tv::check_driver(cuDeviceGet(&device, 0), "cuDeviceGet");
    tv::PrimaryContext primary(device);
    tv::ScopedCurrentContext current(primary.get());
      probe_raw_copy_and_conversion();
    probe_tf32_convolution();
    probe_tf32_matmul();
    probe(ACDNN_POINTWISE_MOD, "mod", true);
    probe(ACDNN_POINTWISE_GELU_APPROX_TANH_BWD, "gelu_approx_tanh_backward",
          true);
    for (const auto type :
         {ACDNN_BACKEND_OPERATION_GEN_STATS_DESCRIPTOR,
          ACDNN_BACKEND_OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR,
          ACDNN_BACKEND_RESAMPLE_DESCRIPTOR}) {
      acdnnBackendDescriptor_t descriptor = nullptr;
      const auto status = acdnnBackendCreateDescriptor(type, &descriptor);
      if (descriptor) (void)acdnnBackendDestroyDescriptor(descriptor);
      if (status != ACDNN_STATUS_NOT_SUPPORTED) {
        throw std::runtime_error(
            "acDNN extended descriptor support changed; requalify catalog: " +
            std::to_string(type));
      }
      std::cout << "descriptor " << type
                << ": confirmed ACDNN_STATUS_NOT_SUPPORTED\n";
    }
    probe(ACDNN_POINTWISE_LOGICAL_NOT, "logical_not", true);
    probe(ACDNN_POINTWISE_LOGICAL_AND, "logical_and", false);
    probe(ACDNN_POINTWISE_LOGICAL_OR, "logical_or", false);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "acDNN gap requalification: FAIL " << error.what() << '\n';
    return 1;
  }
}
