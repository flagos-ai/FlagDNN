/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
/*
Native explicit-TF32 performance catalog from native_matrix.cpp.
Build with: bash tools/performance_baseline/build_benchmark_nvidia_cudnn.sh /tmp/benchmark_nvidia_precision precision
Run: /tmp/benchmark_nvidia_precision > precision.jsonl 2> precision.log
--list and --self-test do not initialize the GPU. --op NAME and --case EXACT_NAME filter cases.
*/
#define FLAGDNN_CUDNN_NO_MAIN
#include "benchmark_nvidia_cudnn.cpp"
#include "common/matmul.hpp"
#include "common/convolution.hpp"
#include <bit>
#include "reference/cpu/matrix.hpp"

namespace native_precision {
namespace test = flagdnn::testing;
using namespace standalone_cudnn;

bench::TensorSpec tensor(const test::TestTensor& value) {
  return {value.uid, value.data_type, value.dimensions, value.strides,
          value.binding_byte_offset};
}
struct Case {
  std::string op;
  bench::BenchmarkCase spec;
  std::function<std::unique_ptr<test::TestExecutable>()> build;
};
class Executable final : public bench::BenchmarkExecutable {
 public:
  explicit Executable(std::unique_ptr<test::TestExecutable> executable)
      : executable_(std::move(executable)) {}
  std::size_t workspace_size() const noexcept override {
    return executable_->workspace_size();
  }
  void execute(std::span<const flagdnnBinding_t> bindings, void* workspace,
               std::size_t workspace_size, flagdnnStream_t stream) override {
    executable_->execute(bindings, workspace, workspace_size, stream);
  }
 private:
  std::unique_ptr<test::TestExecutable> executable_;
};
struct Provider {
  const Case& value;
  bench::ProviderCapability capability(const bench::BenchmarkCase&) { return {}; }
  std::unique_ptr<bench::BenchmarkExecutable> build(const bench::BenchmarkCase&) {
    return std::make_unique<Executable>(value.build());
  }
};
std::vector<Case> catalog() {
  std::vector<Case> result;
  for (const auto& value : test::make_matmul_cases()) {
    if (value.input_precision != 2 || value.output.dimensions.size() > 3) continue;
    bench::BenchmarkCase spec;
    spec.name = value.name;
    spec.operation = bench::Operation::kMatmul;
    spec.tensors = {tensor(value.a), tensor(value.b), tensor(value.output)};
    spec.absolute_tolerance = value.absolute_tolerance;
    spec.relative_tolerance = value.relative_tolerance;
    result.push_back({"matmul", spec, [value] { return test::build_matmul_reference(value); }});
  }
  for (auto direction : {test::ConvolutionDirection::kFprop,
                         test::ConvolutionDirection::kDgrad,
                         test::ConvolutionDirection::kWgrad}) {
    for (const auto& value : test::make_convolution_cases(direction)) {
      if (value.input_precision != 2) continue;
      bench::BenchmarkCase spec;
      spec.name = value.name;
      std::string op;
      if (direction == test::ConvolutionDirection::kFprop) {
        op = "conv_fprop";
        spec.operation = bench::Operation::kConvolutionFprop;
        spec.tensors = {tensor(value.x), tensor(value.w), tensor(value.y)};
      } else if (direction == test::ConvolutionDirection::kDgrad) {
        op = "conv_dgrad";
        spec.operation = bench::Operation::kConvolutionDgrad;
        spec.tensors = {tensor(value.y), tensor(value.w), tensor(value.x)};
      } else {
        op = "conv_wgrad";
        spec.operation = bench::Operation::kConvolutionWgrad;
        spec.tensors = {tensor(value.y), tensor(value.x), tensor(value.w)};
      }
      spec.convolution = {static_cast<std::int32_t>(value.stride.size()),
          value.pre_padding, value.post_padding, value.stride, value.dilation,
          value.groups, value.mode == test::ConvolutionMode::kConvolution ?
              bench::ConvolutionMode::kConvolution : bench::ConvolutionMode::kCrossCorrelation};
      spec.absolute_tolerance = value.absolute_tolerance;
      spec.relative_tolerance = value.relative_tolerance;
      result.push_back({op, spec, [value] { return test::build_convolution_reference(value); }});
    }
  }
  return result;
}

// Exact input formula from native_matrix.cpp::matrix_inputs, before any TF32
// compute rounding. Device storage remains FP32.
std::vector<float> matrix_input(std::size_t n, std::size_t input, InputDomain) {
  std::vector<float> values(n);
  for (std::size_t i = 0; i < n; ++i) {
    values[i] = static_cast<float>(static_cast<int>((i * 17 + input * 7) % 61) - 30) / 19.0F;
  }
  return values;
}
std::vector<float> tf32_reference(const bench::BenchmarkCase& spec,
    const std::vector<std::vector<float>>& inputs,
    const std::vector<std::size_t>& indices) {
  auto rounded = inputs;
  // Nearest-even TF32 operand rounding, matching reference/cpu/matrix.cpp.
  for (auto& input : rounded) for (float& value : input) {
    const auto bits = std::bit_cast<std::uint32_t>(value);
    value = std::bit_cast<float>((bits + 0xFFFU + ((bits >> 13) & 1U)) & 0xFFFFE000U);
  }
  return sampled_reference<double>(spec, rounded, indices);
}
}  // namespace native_precision

int main(int argc, char** argv) {
  using namespace native_precision;
  try {
    bool list = false, self_test = false;
    std::string op, name;
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--self-test") { self_test = true; continue; }
      if (arg == "--list") { list = true; continue; }
      if (++i == argc) throw std::invalid_argument("missing value for " + arg);
      if (arg == "--op") op = argv[i];
      else if (arg == "--case") name = argv[i];
      else throw std::invalid_argument("unknown option: " + arg);
    }
    auto all = catalog();
    std::erase_if(all, [&](const Case& value) {
      return (!op.empty() && value.op != op) || (!name.empty() && value.spec.name != name);
    });
    if (all.empty()) throw std::invalid_argument("no native TF32 cases matched");
    if (list) {
      for (const auto& value : all) std::cout << value.op << '\t' << value.spec.name << '\n';
      return 0;
    }
    if (self_test) {
      std::set<std::string> tested;
      namespace cpu = flagdnn::reference::cpu;
      for (const auto& value : all) {
        if (!tested.insert(value.op).second) continue;
        const auto& spec = value.spec;
        std::vector<std::vector<float>> inputs;
        for (std::size_t i = 0; i < bench::input_tensor_count(spec); ++i)
          inputs.push_back(matrix_input(product(spec.tensors[i].dimensions), i, InputDomain::kReal));
        std::vector<float> expected;
        if (value.op == "matmul") {
          expected = cpu::evaluate_matmul({spec.tensors[0].dimensions, spec.tensors[1].dimensions,
              spec.tensors[2].dimensions, FLAGDNN_DATA_FLOAT32, 2}, inputs[0], inputs[1]);
        } else {
          const bool fprop = value.op == "conv_fprop", dgrad = value.op == "conv_dgrad";
          cpu::ConvolutionParameters params;
          params.x_shape = spec.tensors[fprop ? 0 : dgrad ? 2 : 1].dimensions;
          params.w_shape = spec.tensors[fprop || dgrad ? 1 : 2].dimensions;
          params.y_shape = spec.tensors[fprop ? 2 : 0].dimensions;
          params.stride = spec.convolution.stride;
          params.pre_padding = spec.convolution.pre_padding;
          params.dilation = spec.convolution.dilation;
          params.direction = fprop ? cpu::ConvolutionDirection::kForward :
              dgrad ? cpu::ConvolutionDirection::kInputGradient : cpu::ConvolutionDirection::kWeightGradient;
          params.groups = spec.convolution.groups;
          params.input_precision = 2;
          params.flip_kernel = spec.convolution.mode == bench::ConvolutionMode::kConvolution;
          std::vector<std::vector<float>> xyz(3);
          if (fprop) { xyz[0] = inputs[0]; xyz[1] = inputs[1]; }
          else if (dgrad) { xyz[2] = inputs[0]; xyz[1] = inputs[1]; }
          else { xyz[2] = inputs[0]; xyz[0] = inputs[1]; }
          expected = cpu::evaluate_convolution(params, xyz);
        }
        const auto ids = sample_indices(expected.size(), 257);
        const auto actual = tf32_reference(spec, inputs, ids);
        std::vector<float> selected;
        for (auto i : ids) selected.push_back(expected.at(i));
        compare_outputs(actual, selected, 0, 0);
        std::cout << value.op << " TF32 sampled oracle matches public CPU oracle exactly\n";
      }
      return 0;
    }
    bench::DriverContext context;
    bench::Stream stream;
    CUdevice device;
    bench::check_cuda(cuCtxGetDevice(&device), "cuCtxGetDevice");
    char device_name[256]{};
    bench::check_cuda(cuDeviceGetName(device_name, sizeof(device_name), device), "cuDeviceGetName");
    int driver = 0, runtime = 0;
    bench::check_cuda(cuDriverGetVersion(&driver), "cuDriverGetVersion");
    const auto status = cudaRuntimeGetVersion(&runtime);
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    std::size_t failures = 0, completed = 0;
    for (const auto& value : all) {
      Measurements measurements;
      std::string error;
      Provider provider{value};
      try {
        measure(value.spec, provider, stream, measurements, matrix_input, tf32_reference);
      } catch (const std::exception& exception) {
        error = exception.what();
        ++failures;
      }
      emit(value.spec, value.op, device_name, driver, runtime, measurements, error,
           "backends/nvidia/validation/benchmark/native_matrix.cpp::run_native_precision_benchmark",
           "backends/nvidia/validation/benchmark/native_matrix.cpp::matrix_inputs", 2);
      std::cerr << ++completed << '/' << all.size() << ' ' << value.spec.name << ' '
                << (error.empty() ? "PASS" : "ERROR: " + error) << std::endl;
    }
    return failures ? 1 : 0;
  } catch (const std::exception& error) {
    std::cerr << "ERROR: " << error.what() << std::endl;
    return 1;
  }
}
