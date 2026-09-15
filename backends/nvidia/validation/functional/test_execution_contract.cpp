/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include <unistd.h>

#include <algorithm>
#include <array>
#include <barrier>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <flagdnn/flagdnn.hpp>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "common/convolution.hpp"
#include "common/layout.hpp"
#include "common/matmul.hpp"
#include "common/pointwise.hpp"
#include "validation/cuda_driver.hpp"
#include "validation/functional/contract_reference.hpp"
#include "validation/tensor_io.hpp"

namespace {
using namespace flagdnn::validation::nvidia;
namespace reference = flagdnn::testing;
namespace reference_cuda = flagdnn::testing::cuda;
using reference_cuda::contract_tensor;
using reference_cuda::run_contract_reference;
std::unique_ptr<reference::TestExecutable> pointwise_reference(
    flagdnnPointwiseMode_t mode, std::vector<reference::TestTensor> inputs,
    reference::TestTensor output,
    flagdnnPointwiseAttributes_t attributes =
        FLAGDNN_POINTWISE_ATTRIBUTES_INITIALIZER) {
  reference::PointwiseTestCase specification;
  specification.name = "execution_contract";
  specification.mode = mode;
  specification.input_domains.assign(inputs.size(),
                                     reference::PointwiseInputDomain::kReal);
  specification.inputs = std::move(inputs);
  specification.output = std::move(output);
  specification.attributes = attributes;
  return reference::build_pointwise_reference(specification);
}

void require(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

class Cache {
 public:
  Cache() {
    auto pattern = (std::filesystem::temp_directory_path() /
                    "flagdnn-nvidia-contract-XXXXXX")
                       .string();
    std::vector<char> name(pattern.begin(), pattern.end());
    name.push_back('\0');
    const char* directory = mkdtemp(name.data());
    if (directory == nullptr) {
      throw std::runtime_error("mkdtemp failed");
    }
    path = directory;
  }
  ~Cache() {
    std::error_code error;
    std::filesystem::remove_all(path, error);
  }
  Cache(const Cache&) = delete;
  Cache& operator=(const Cache&) = delete;
  std::filesystem::path path;
};

void check_build_threads(const char* python, const char* compiler) {
  std::barrier start(2);
  std::array<Cache, 2> caches;
  std::array<std::unique_ptr<flagdnn::Handle>, 2> handles;
  std::array<std::exception_ptr, 2> failures{};
  const auto build = [&](std::size_t index) {
    try {
      handles[index] =
          std::make_unique<flagdnn::Handle>(FLAGDNN_BACKEND_NVIDIA, 0);
      auto& handle = *handles[index];
      handle.set_compiler(python, compiler, caches[index].path.string());
      const std::array<std::int64_t, 1> dimensions{
          32 * static_cast<std::int64_t>(index + 1)};
      const std::array<std::int64_t, 1> strides{1};
      flagdnn::TensorDescriptor input(1, FLAGDNN_DATA_FLOAT32, dimensions,
                                      strides);
      flagdnn::TensorDescriptor output(2, FLAGDNN_DATA_FLOAT32, dimensions,
                                       strides);
      flagdnn::Graph graph;
      graph.relu(input, output);
      graph.finalize();
      flagdnn::Executable executable(handle, graph);
      require(executable.operation_count() == 1,
              "worker build returned an invalid executable");
    } catch (...) {
      failures[index] = std::current_exception();
    }
  };
  std::thread first([&] {
    start.arrive_and_wait();
    build(0);
  });
  std::thread second([&] {
    start.arrive_and_wait();
    build(1);
  });
  first.join();
  second.join();
  // Release the last handles only after the builders' thread-local state has
  // been destroyed. Python shutdown callbacks must not run at this dlclose.
  for (auto& handle : handles) {
    handle.reset();
  }
  for (const auto& failure : failures) {
    if (failure) {
      std::rethrow_exception(failure);
    }
  }
  std::cout << "PASS cold builds on independent threads and handles\n";
  // A premature shutdown callback can acquire the GIL on the releasing main
  // thread; a subsequent build on a fresh worker then deadlocks.
  std::thread reload([&] { build(0); });
  reload.join();
  handles[0].reset();
  if (failures[0]) {
    std::rethrow_exception(failures[0]);
  }
  std::cout
      << "PASS cross-thread handle destruction and subsequent worker build\n";
}

void check_streams_and_graph(flagdnn::Handle& handle, Stream& stream) {
  const std::array<std::int64_t, 1> dims{16}, strides{1};
  flagdnn::TensorDescriptor a(1, FLAGDNN_DATA_FLOAT32, dims, strides);
  flagdnn::TensorDescriptor b(2, FLAGDNN_DATA_FLOAT32, dims, strides);
  flagdnn::TensorDescriptor c(3, FLAGDNN_DATA_FLOAT32, dims, strides);
  flagdnn::TensorDescriptor d(4, FLAGDNN_DATA_FLOAT32, dims, strides);
  b.set_virtual();
  c.set_virtual();
  flagdnn::Graph graph;
  graph.relu(a, b);
  graph.relu(b, c);
  graph.relu(c, d);
  graph.finalize();
  flagdnn::Executable executable(handle, graph);
  DeviceBuffer input(64), output(64), workspace(executable.workspace_size());
  std::array<float, 16> values{};
  for (std::size_t i = 0; i < values.size(); ++i) {
    values[i] = static_cast<float>(i) - 8.0F;
  }
  input.copy_from_host(values.data(), 64, stream.get());
  stream.synchronize();
  const std::array<flagdnnBinding_t, 2> bindings{
      {{4, output.opaque()}, {1, input.opaque()}}};
  auto cudnn_graph = std::make_shared<reference_cuda::cfe::graph::Graph>();
  cudnn_graph->set_io_data_type(reference_cuda::cfe::DataType_t::FLOAT)
      .set_intermediate_data_type(reference_cuda::cfe::DataType_t::FLOAT)
      .set_compute_data_type(reference_cuda::cfe::DataType_t::FLOAT);
  const auto input_spec = reference_cuda::flatten_compact_tensor(
      contract_tensor(1, FLAGDNN_DATA_FLOAT32, dims, strides));
  auto value =
      reference_cuda::make_cudnn_tensor(cudnn_graph, input_spec, "input");
  for (int node = 0; node < 3; ++node)
    value = cudnn_graph->pointwise(
        value, reference_cuda::cfe::graph::Pointwise_attributes().set_mode(
                   reference_cuda::cfe::PointwiseMode_t::RELU_FWD));
  value->set_uid(4)
      .set_output(true)
      .set_data_type(reference_cuda::cfe::DataType_t::FLOAT)
      .set_dim(input_spec.dimensions)
      .set_stride(input_spec.strides);
  auto cudnn = reference_cuda::build_cudnn_graph(std::move(cudnn_graph));
  const auto expected = run_contract_reference<float>(
      *cudnn, bindings, 4, values.size(), stream.get());
  CUcontext context = nullptr;
  check_cuda(cuCtxGetCurrent(&context), "cuCtxGetCurrent");
  for (CUstream selected :
       {CUstream{}, CU_STREAM_LEGACY, CU_STREAM_PER_THREAD, stream.get()}) {
    std::exception_ptr failure;
    std::thread worker([&] {
      try {
        CUcontext before = nullptr;
        check_cuda(cuCtxGetCurrent(&before), "fresh thread context");
        require(before == nullptr, "worker unexpectedly has a context");
        executable.execute(bindings, workspace.opaque(),
                           executable.workspace_size(), selected);
        CUcontext after = nullptr;
        check_cuda(cuCtxGetCurrent(&after), "restored thread context");
        require(after == before, "execute changed caller context");
        check_cuda(cuCtxPushCurrent(context), "worker context");
        const CUresult synchronized = cuStreamSynchronize(selected);
        CUcontext ignored = nullptr;
        (void)cuCtxPopCurrent(&ignored);
        check_cuda(synchronized, "worker stream synchronization");
      } catch (...) {
        failure = std::current_exception();
      }
    });
    worker.join();
    if (failure) {
      std::rethrow_exception(failure);
    }
    std::array<float, 16> actual{};
    output.copy_to_host(actual.data(), 64, stream.get());
    stream.synchronize();
    for (std::size_t i = 0; i < actual.size(); ++i) {
      require(actual[i] == expected[i], "three-node graph result mismatch");
    }
  }
  std::cout
      << "PASS rank-one graph and default/explicit streams on fresh threads\n";
}

void check_fp32_convolution(flagdnn::Handle& handle, Stream& stream,
                            bool p5 = false) {
  // Non-aligned P5 channels exercise K split tails and the final output tile.
  const std::int64_t channels = p5 ? 130 : 64;
  const std::int64_t outputs = p5 ? 257 : 64;
  const std::int64_t input_hw = p5 ? 40 : 32;
  const std::int64_t output_hw = p5 ? 20 : 32;
  const std::int64_t batch = p5 ? 1 : 4;
  const std::array<std::int64_t, 4> xd{batch, channels, input_hw, input_hw};
  const std::array<std::int64_t, 4> xs{channels * input_hw * input_hw,
                                       input_hw * input_hw, input_hw, 1};
  const std::array<std::int64_t, 4> wd{outputs, channels, 3, 3};
  const std::array<std::int64_t, 4> ws{channels * 9, 9, 3, 1};
  const std::array<std::int64_t, 4> yd{batch, outputs, output_hw, output_hw};
  const std::array<std::int64_t, 4> ys{outputs * output_hw * output_hw,
                                       output_hw * output_hw, output_hw, 1};
  const std::array<std::int64_t, 2> one{1, 1};
  const std::array<std::int64_t, 2> stride{p5 ? 2 : 1, p5 ? 2 : 1};
  flagdnn::TensorDescriptor x(11, FLAGDNN_DATA_FLOAT32, xd, xs);
  flagdnn::TensorDescriptor w(12, FLAGDNN_DATA_FLOAT32, wd, ws);
  flagdnn::TensorDescriptor y(13, FLAGDNN_DATA_FLOAT32, yd, ys);
  flagdnn::Graph graph;
  // Preserve the production im2col/P5 pipelines and compare the precision
  // selected by each layout against an equivalent cuDNN plan.
  graph.convolution_fprop(x, w, one, one, stride, one, 1, y);
  graph.finalize();
  flagdnn::Executable executable(handle, graph);
  DeviceBuffer input(x.size_in_bytes()), weight(w.size_in_bytes());
  DeviceBuffer output(y.size_in_bytes());
  DeviceBuffer workspace(executable.workspace_size());
  const std::array<flagdnnBinding_t, 3> bindings{
      {{11, input.opaque()}, {12, weight.opaque()}, {13, output.opaque()}}};
  reference::ConvolutionTestCase specification;
  specification.name = "execution_conv_range";
  specification.input_precision = p5 ? 1 : 2;
  specification.x = contract_tensor(11, FLAGDNN_DATA_FLOAT32, xd, xs);
  specification.w = contract_tensor(12, FLAGDNN_DATA_FLOAT32, wd, ws);
  specification.y = contract_tensor(13, FLAGDNN_DATA_FLOAT32, yd, ys);
  specification.pre_padding = specification.post_padding = {1, 1};
  specification.dilation = {1, 1};
  specification.stride.assign(stride.begin(), stride.end());
  auto cudnn = reference::build_convolution_reference(specification);
  for (const auto& operands : {std::array<float, 2>{100000.0F, 0.00001F},
                               std::array<float, 2>{1.0e-9F, 1.0e9F},
                               std::array<float, 2>{1.0F, 65536.0F}}) {
    const bool cancellation = operands[1] == 65536.0F;
    std::vector<float> host_x(x.size_in_bytes() / 4, operands[0]);
    std::vector<float> host_w(w.size_in_bytes() / 4, operands[1]);
    if (cancellation) {
      for (std::size_t i = 0; i < host_w.size(); ++i) {
        if ((i / 9) % 2 != 0) {
          host_w[i] = -65535.0F;
        }
      }
    }
    std::vector<float> actual(y.size_in_bytes() / 4);
    input.copy_from_host(host_x.data(), x.size_in_bytes(), stream.get());
    weight.copy_from_host(host_w.data(), w.size_in_bytes(), stream.get());
    executable.execute(bindings, workspace.opaque(),
                       executable.workspace_size(), stream.opaque());
    output.copy_to_host(actual.data(), y.size_in_bytes(), stream.get());
    stream.synchronize();
    const auto expected = run_contract_reference<float>(
        *cudnn, bindings, 13, actual.size(), stream.get());
    for (std::size_t i = 0; i < actual.size(); ++i) {
      require(std::isfinite(actual[i]) && std::abs(actual[i] - expected[i]) <=
                                              1.e-4 * std::abs(expected[i]),
              "FP32 convolution differs from cuDNN in range or accuracy");
    }
  }
  std::cout << "PASS FP32 " << (p5 ? "P5 tails" : "general im2col")
            << " convolution large, tiny and cancelling operands\n";
}
void check_layernorm_row_tail(flagdnn::Handle& handle, Stream& stream) {
  constexpr std::int64_t rows = 129, columns = 1024;
  constexpr float sentinel = -999.0F;
  const std::array<std::int64_t, 2> data_dims{rows, columns};
  const std::array<std::int64_t, 2> data_strides{columns, 1};
  const std::array<std::int64_t, 2> affine_dims{1, columns};
  const std::array<std::int64_t, 2> stat_dims{rows, 1}, stat_strides{1, 1};
  flagdnn::TensorDescriptor x(21, FLAGDNN_DATA_FLOAT32, data_dims,
                              data_strides);
  flagdnn::TensorDescriptor w(22, FLAGDNN_DATA_FLOAT32, affine_dims,
                              data_strides);
  flagdnn::TensorDescriptor b(23, FLAGDNN_DATA_FLOAT32, affine_dims,
                              data_strides);
  flagdnn::TensorDescriptor y(24, FLAGDNN_DATA_FLOAT32, data_dims,
                              data_strides);
  flagdnn::TensorDescriptor mean(25, FLAGDNN_DATA_FLOAT32, stat_dims,
                                 stat_strides);
  flagdnn::TensorDescriptor inv(26, FLAGDNN_DATA_FLOAT32, stat_dims,
                                stat_strides);
  flagdnn::OperationDescriptor operation("layernorm");
  operation.set_input("x", x);
  operation.set_input("scale", w);
  operation.set_input("bias", b);
  operation.set_output("y", y);
  operation.set_output("mean", mean);
  operation.set_output("inv_variance", inv);
  operation.set_attribute("epsilon", 1.e-5);
  operation.set_attribute("forward_phase", std::int64_t{2});
  operation.finalize();
  flagdnn::Graph graph;
  graph.add(operation);
  graph.finalize();
  flagdnnBuildOptions_t options = FLAGDNN_BUILD_OPTIONS_INITIALIZER;
  options.flags = FLAGDNN_BUILD_OPTION_AUTOTUNE;
  flagdnn::Executable executable(handle, graph, &options);
  std::vector<float> input((rows + 1) * columns, sentinel);
  std::vector<float> output(input.size(), sentinel);
  std::vector<float> scale(columns, 1.0F), bias(columns, 0.0F);
  std::vector<float> means(rows + 1, sentinel), inverses(rows + 1, sentinel);
  for (std::int64_t row = 0; row < rows; ++row) {
    for (std::int64_t col = 0; col < columns; ++col) {
      input[row * columns + col] = (col % 2 ? 1.0F : -1.0F) + row / 128.0F;
    }
  }
  DeviceBuffer dx(input.size() * 4), dy(output.size() * 4);
  DeviceBuffer dw(columns * 4), db(columns * 4);
  DeviceBuffer dm(means.size() * 4), di(inverses.size() * 4);
  DeviceBuffer workspace(executable.workspace_size());
  dx.copy_from_host(input.data(), input.size() * 4, stream.get());
  dy.copy_from_host(output.data(), output.size() * 4, stream.get());
  dw.copy_from_host(scale.data(), columns * 4, stream.get());
  db.copy_from_host(bias.data(), columns * 4, stream.get());
  dm.copy_from_host(means.data(), means.size() * 4, stream.get());
  di.copy_from_host(inverses.data(), inverses.size() * 4, stream.get());
  const std::array<flagdnnBinding_t, 6> bindings{{{21, dx.opaque()},
                                                  {22, dw.opaque()},
                                                  {23, db.opaque()},
                                                  {24, dy.opaque()},
                                                  {25, dm.opaque()},
                                                  {26, di.opaque()}}};
  executable.execute(bindings, workspace.opaque(), executable.workspace_size(),
                     stream.opaque());
  dy.copy_to_host(output.data(), output.size() * 4, stream.get());
  dm.copy_to_host(means.data(), means.size() * 4, stream.get());
  di.copy_to_host(inverses.data(), inverses.size() * 4, stream.get());
  stream.synchronize();
  auto cudnn_graph = std::make_shared<reference_cuda::cfe::graph::Graph>();
  cudnn_graph->set_io_data_type(reference_cuda::cfe::DataType_t::FLOAT)
      .set_intermediate_data_type(reference_cuda::cfe::DataType_t::FLOAT)
      .set_compute_data_type(reference_cuda::cfe::DataType_t::FLOAT);
  const auto tensor = [&](std::int64_t uid, const auto& dims,
                          const auto& strides) {
    return reference_cuda::make_cudnn_tensor(
        cudnn_graph, contract_tensor(uid, FLAGDNN_DATA_FLOAT32, dims, strides),
        "contract");
  };
  auto results = cudnn_graph->layernorm(
      tensor(21, data_dims, data_strides),
      tensor(22, affine_dims, data_strides),
      tensor(23, affine_dims, data_strides),
      reference_cuda::cfe::graph::Layernorm_attributes()
          .set_forward_phase(reference_cuda::cfe::NormFwdPhase_t::TRAINING)
          .set_epsilon(1.e-5F));
  for (std::size_t index = 0; index < results.size(); ++index) {
    const auto spec =
        index == 0
            ? contract_tensor(24, FLAGDNN_DATA_FLOAT32, data_dims, data_strides)
            : contract_tensor(24 + index, FLAGDNN_DATA_FLOAT32, stat_dims,
                              stat_strides);
    results[index]
        ->set_uid(spec.uid)
        .set_data_type(reference_cuda::cfe::DataType_t::FLOAT)
        .set_dim(spec.dimensions)
        .set_stride(spec.strides)
        .set_output(true);
  }
  auto cudnn = reference_cuda::build_cudnn_graph(std::move(cudnn_graph));
  DeviceBuffer reference_y(rows * columns * 4), reference_mean(rows * 4),
      reference_inv(rows * 4), reference_workspace(cudnn->workspace_size());
  const std::array<flagdnnBinding_t, 6> reference_bindings{
      {{21, dx.opaque()},
       {22, dw.opaque()},
       {23, db.opaque()},
       {24, reference_y.opaque()},
       {25, reference_mean.opaque()},
       {26, reference_inv.opaque()}}};
  cudnn->execute(reference_bindings, reference_workspace.opaque(),
                 cudnn->workspace_size(), stream.opaque());
  std::vector<float> expected_y(rows * columns), expected_mean(rows),
      expected_inv(rows);
  reference_y.copy_to_host(expected_y.data(), expected_y.size() * 4,
                           stream.get());
  reference_mean.copy_to_host(expected_mean.data(), rows * 4, stream.get());
  reference_inv.copy_to_host(expected_inv.data(), rows * 4, stream.get());
  stream.synchronize();
  for (std::int64_t row = 0; row < rows; ++row) {
    require(std::abs(means[row] - expected_mean[row]) < 1.e-6 &&
                std::abs(inverses[row] - expected_inv[row]) < 1.e-6,
            "LayerNorm row statistics differ from cuDNN");
    for (std::int64_t col = 0; col < columns; ++col) {
      require(std::abs(output[row * columns + col] -
                       expected_y[row * columns + col]) < 1.e-5,
              "LayerNorm row output differs from cuDNN");
    }
  }
  require(means.back() == sentinel && inverses.back() == sentinel &&
              std::all_of(output.begin() + rows * columns, output.end(),
                          [](float value) { return value == sentinel; }),
          "LayerNorm grouped launch overwrote the tail guard");
  std::cout << "PASS LayerNorm autotune odd rows and output/statistic guards\n";
}
void check_matrix_transpose_tails(flagdnn::Handle& handle, Stream& stream,
                                  bool wide = false) {
  const std::int64_t rows = wide ? 16 : 257;
  const std::int64_t columns = wide ? 65536 * 16 : 259;
  constexpr std::int64_t guard = 1024;
  constexpr float sentinel = -999.0F;
  const std::array<std::int64_t, 2> xd{rows, columns}, xs{columns, 1};
  const std::array<std::int64_t, 2> yd{columns, rows}, ys{rows, 1};
  const std::array<std::int64_t, 2> permutation{1, 0};
  flagdnn::TensorDescriptor x(31, FLAGDNN_DATA_FLOAT32, xd, xs);
  flagdnn::TensorDescriptor y(32, FLAGDNN_DATA_FLOAT32, yd, ys);
  flagdnn::OperationDescriptor operation("transpose");
  operation.set_input("input", x);
  operation.set_output("output", y);
  operation.set_attribute("permutation", permutation);
  operation.finalize();
  flagdnn::Graph graph;
  graph.add(operation);
  graph.finalize();
  flagdnnBuildOptions_t options = FLAGDNN_BUILD_OPTIONS_INITIALIZER;
  options.flags = FLAGDNN_BUILD_OPTION_AUTOTUNE;
  flagdnn::Executable executable(handle, graph, &options);
  std::vector<float> input(rows * columns + guard, sentinel);
  std::vector<float> output(input.size(), sentinel);
  for (std::int64_t i = 0; i < rows * columns; ++i) {
    input[i] = static_cast<float>(i);
  }
  DeviceBuffer dx(input.size() * sizeof(float));
  DeviceBuffer dy(output.size() * sizeof(float));
  DeviceBuffer workspace(executable.workspace_size());
  dx.copy_from_host(input.data(), input.size() * sizeof(float), stream.get());
  dy.copy_from_host(output.data(), output.size() * sizeof(float), stream.get());
  const std::array<flagdnnBinding_t, 2> bindings{
      {{31, dx.opaque()}, {32, dy.opaque()}}};
  executable.execute(bindings, workspace.opaque(), executable.workspace_size(),
                     stream.opaque());
  dy.copy_to_host(output.data(), output.size() * sizeof(float), stream.get());
  stream.synchronize();
  reference::LayoutTestCase specification;
  specification.name = "execution_transpose_tail";
  specification.operation = reference::LayoutOperation::kTranspose;
  specification.input = contract_tensor(31, FLAGDNN_DATA_FLOAT32, xd, xs);
  specification.output = contract_tensor(32, FLAGDNN_DATA_FLOAT32, yd, ys);
  specification.permutation = {1, 0};
  auto cudnn = reference::build_layout_reference(specification);
  const auto expected = run_contract_reference<float>(
      *cudnn, bindings, 32, rows * columns, stream.get());
  require(std::equal(expected.begin(), expected.end(), output.begin()),
          "tiled matrix transpose tail differs from cuDNN");
  require(std::all_of(output.begin() + rows * columns, output.end(),
                      [](float value) { return value == sentinel; }),
          "tiled matrix transpose overwrote the output guard");
  std::cout << "PASS transpose autotune "
            << (wide ? "wide-grid fallback" : "both-axis tails")
            << " and output guard\n";
}

void check_low_precision_matmul_scratch(flagdnn::Handle& handle,
                                        Stream& stream) {
  constexpr std::int64_t batch = 4;
  constexpr std::size_t guard = 1024;
  constexpr std::uint8_t sentinel = 0x5A;
  for (const auto dtype : {FLAGDNN_DATA_FLOAT16, FLAGDNN_DATA_BFLOAT16}) {
    for (const auto shape : {std::array<std::int64_t, 3>{512, 512, 512},
                             std::array<std::int64_t, 3>{1024, 1024, 1024},
                             std::array<std::int64_t, 3>{513, 515, 520}}) {
      const auto m = shape[0], n = shape[1], k = shape[2];
      const std::array<std::int64_t, 3> ad{batch, m, k}, as{m * k, k, 1};
      const std::array<std::int64_t, 3> bd{batch, k, n}, bs{k * n, n, 1};
      const std::array<std::int64_t, 3> yd{batch, m, n}, ys{m * n, n, 1};
      flagdnn::TensorDescriptor a(51, dtype, ad, as);
      flagdnn::TensorDescriptor b(52, dtype, bd, bs);
      flagdnn::TensorDescriptor y(53, dtype, yd, ys);
      flagdnn::Graph graph;
      graph.matmul(a, b, y);
      graph.finalize();
      std::vector<float> host_a(batch * m * k), host_b(batch * k * n);
      for (std::int64_t i = 0; i < batch * m * k; ++i) {
        host_a[i] = 0.125F * static_cast<float>(1 + (i / k % m) % 3);
      }
      for (std::int64_t i = 0; i < batch * k * n; ++i) {
        host_b[i] = 0.0625F * static_cast<float>(1 + i / (k * n) + i % n % 2);
      }
      const auto encoded_a =
          tensor_io::encode(host_a, dtype, tensor_io::BooleanEncoding::kByte);
      const auto encoded_b =
          tensor_io::encode(host_b, dtype, tensor_io::BooleanEncoding::kByte);
      DeviceBuffer da(a.size_in_bytes()), db(b.size_in_bytes());
      DeviceBuffer dy(y.size_in_bytes() + guard);
      da.copy_from_host(encoded_a.data(), encoded_a.size(), stream.get());
      db.copy_from_host(encoded_b.data(), encoded_b.size(), stream.get());
      const std::array<flagdnnBinding_t, 3> bindings{
          {{51, da.opaque()}, {52, db.opaque()}, {53, dy.opaque()}}};
      reference::MatmulTestCase specification;
      specification.name = "execution_low_precision_matmul";
      specification.a = contract_tensor(51, dtype, ad, as);
      specification.b = contract_tensor(52, dtype, bd, bs);
      specification.output = contract_tensor(53, dtype, yd, ys);
      auto cudnn = reference::build_matmul_reference(specification);
      const auto encoded_expected = run_contract_reference<std::uint8_t>(
          *cudnn, bindings, 53, y.size_in_bytes(), stream.get());
      for (bool autotune : {false, true}) {
        flagdnnBuildOptions_t options = FLAGDNN_BUILD_OPTIONS_INITIALIZER;
        if (autotune) {
          options.flags = FLAGDNN_BUILD_OPTION_AUTOTUNE;
        }
        flagdnn::Executable executable(handle, graph, &options);
        DeviceBuffer workspace(executable.workspace_size() + guard);
        std::vector<std::uint8_t> host_workspace(
            executable.workspace_size() + guard, sentinel);
        std::vector<std::uint8_t> actual(y.size_in_bytes() + guard, sentinel);
        workspace.copy_from_host(host_workspace.data(), host_workspace.size(),
                                 stream.get());
        dy.copy_from_host(actual.data(), actual.size(), stream.get());
        executable.execute(bindings, workspace.opaque(),
                           executable.workspace_size(), stream.opaque());
        workspace.copy_to_host(host_workspace.data(), host_workspace.size(),
                               stream.get());
        dy.copy_to_host(actual.data(), actual.size(), stream.get());
        stream.synchronize();
        require(
            std::equal(encoded_expected.begin(), encoded_expected.end(),
                       actual.begin()),
            "low-precision MatMul mixed batch tails or changed accumulation");
        require(
            std::all_of(actual.begin() + y.size_in_bytes(), actual.end(),
                        [](std::uint8_t value) { return value == sentinel; }),
            "low-precision MatMul overwrote the output guard");
        require(
            std::all_of(host_workspace.begin() + executable.workspace_size(),
                        host_workspace.end(),
                        [](std::uint8_t value) { return value == sentinel; }),
            "TMA MatMul overwrote the workspace guard");
      }
    }
  }
  std::cout << "PASS FP16/BF16 MatMul short-K, persistent, batch tails, "
               "fixed/autotune and scratch guards\n";
}

enum class Fp32MatmulCase {
  Irregular,
  Packed,
  Direct,
  TensorMap,
  TensorMapShort
};

void check_fp32_matmul_layout(
    flagdnn::Handle& handle, Stream& stream,
    Fp32MatmulCase test_case = Fp32MatmulCase::Irregular,
    bool autotune = true) {
  const bool tf32 = test_case != Fp32MatmulCase::Irregular;
  const bool short_tensor_map = test_case == Fp32MatmulCase::TensorMapShort;
  const bool tensor_map =
      test_case == Fp32MatmulCase::TensorMap || short_tensor_map;
  constexpr std::int64_t batch = 4;
  const std::int64_t m = short_tensor_map ? 1024 : (tf32 ? 512 : 513);
  const std::int64_t n =
      short_tensor_map ? 1024
                       : ((test_case == Fp32MatmulCase::Direct || tensor_map)
                              ? 512
                              : (tf32 ? 516 : 515));
  const std::int64_t k =
      tensor_map && !short_tensor_map ? 1024 : (tf32 ? 512 : 518);
  constexpr std::int64_t guard = 1024;
  constexpr float sentinel = -999.0F;
  const std::array<std::int64_t, 3> ad{batch, m, k}, as{m * k, k, 1};
  const std::array<std::int64_t, 3> bd{batch, k, n}, bs{k * n, n, 1};
  const std::array<std::int64_t, 3> yd{batch, m, n}, ys{m * n, n, 1};
  flagdnn::TensorDescriptor a(41, FLAGDNN_DATA_FLOAT32, ad, as);
  flagdnn::TensorDescriptor b(42, FLAGDNN_DATA_FLOAT32, bd, bs);
  flagdnn::TensorDescriptor y(43, FLAGDNN_DATA_FLOAT32, yd, ys);
  flagdnn::Graph graph;
  // Exercise the production packing/TensorMap selection. The reference uses
  // the corresponding explicit IEEE or TF32 precision in its own process.
  graph.matmul(a, b, y);
  graph.finalize();
  flagdnnBuildOptions_t options = FLAGDNN_BUILD_OPTIONS_INITIALIZER;
  options.flags = autotune ? FLAGDNN_BUILD_OPTION_AUTOTUNE : 0;
  flagdnn::Executable executable(handle, graph, &options);
  DeviceBuffer da(a.size_in_bytes()), db(b.size_in_bytes());
  DeviceBuffer alternate_a(tensor_map ? a.size_in_bytes() : 0);
  DeviceBuffer alternate_b(tensor_map ? b.size_in_bytes() : 0);
  DeviceBuffer dy(y.size_in_bytes() + guard * sizeof(float));
  DeviceBuffer workspace(executable.workspace_size() + guard);
  std::array<flagdnnBinding_t, 3> bindings{
      {{41, da.opaque()}, {42, db.opaque()}, {43, dy.opaque()}}};
  reference::MatmulTestCase specification;
  specification.name = "execution_fp32_matmul";
  specification.input_precision = tf32 ? 2 : 1;
  specification.a = contract_tensor(41, FLAGDNN_DATA_FLOAT32, ad, as);
  specification.b = contract_tensor(42, FLAGDNN_DATA_FLOAT32, bd, bs);
  specification.output = contract_tensor(43, FLAGDNN_DATA_FLOAT32, yd, ys);
  auto cudnn = reference::build_matmul_reference(specification);
  unsigned int iteration = 0;
  for (const auto& operands : {std::array<float, 2>{100000.0F, 0.00001F},
                               std::array<float, 2>{1.0e-9F, 1.0e9F},
                               std::array<float, 2>{1.0F, 65536.0F},
                               std::array<float, 2>{1.0F, 65472.0F}}) {
    const bool wide_cancellation = operands[1] == 65472.0F;
    const bool cancellation = operands[1] == 65536.0F || wide_cancellation;
    std::vector<float> host_a(batch * m * k, operands[0]);
    std::vector<float> host_b(batch * k * n, operands[1]);
    std::vector<float> actual(batch * m * n + guard, sentinel);
    for (std::int64_t index = 0; index < batch * m * k; ++index) {
      host_a[index] *= static_cast<float>(1 + (index / k % m) % 3);
    }
    for (std::int64_t index = 0; index < batch * k * n; ++index) {
      const auto sample = index / (k * n);
      const auto column = index % n;
      const auto reduction = index / n % k;
      // Distinct batches and columns detect a mistaken packed matrix view.
      host_b[index] =
          cancellation
              ? (reduction % 2
                     ? (wide_cancellation ? -65472.0F : -65535.0F)
                     : 65536.0F + (wide_cancellation ? 128 : 2) * sample +
                           (wide_cancellation ? 64 : 1) * (column % 2))
              : operands[1] * static_cast<float>(1 + sample + column % 2);
    }
    const bool rebind = tensor_map && (++iteration % 2 == 0);
    auto& input_a = rebind ? alternate_a : da;
    auto& input_b = rebind ? alternate_b : db;
    input_a.copy_from_host(host_a.data(), a.size_in_bytes(), stream.get());
    input_b.copy_from_host(host_b.data(), b.size_in_bytes(), stream.get());
    bindings[0].device_pointer = input_a.opaque();
    bindings[1].device_pointer = input_b.opaque();
    dy.copy_from_host(actual.data(), actual.size() * sizeof(float),
                      stream.get());
    std::vector<std::uint8_t> workspace_bytes(
        executable.workspace_size() + guard, 0xA5);
    workspace.copy_from_host(workspace_bytes.data(), workspace_bytes.size(),
                             stream.get());
    executable.execute(bindings, workspace.opaque(),
                       executable.workspace_size(), stream.opaque());
    dy.copy_to_host(actual.data(), actual.size() * sizeof(float), stream.get());
    workspace.copy_to_host(workspace_bytes.data(), workspace_bytes.size(),
                           stream.get());
    stream.synchronize();
    const auto expected = run_contract_reference<float>(
        *cudnn, bindings, 43, batch * m * n, stream.get());
    for (std::int64_t index = 0; index < batch * m * n; ++index) {
      if (!(std::isfinite(actual[index]) &&
            std::abs(actual[index] - expected[index]) <=
                1.e-4 * std::abs(expected[index])))
        throw std::runtime_error("FP32 MatMul differs from cuDNN: path=" +
                                 std::to_string(static_cast<int>(test_case)) +
                                 " index=" + std::to_string(index) +
                                 " inputs=" + std::to_string(operands[0]) +
                                 "," + std::to_string(operands[1]) +
                                 " actual=" + std::to_string(actual[index]) +
                                 " cuDNN=" + std::to_string(expected[index]));
    }
    require(std::all_of(actual.begin() + batch * m * n, actual.end(),
                        [](float value) { return value == sentinel; }),
            "FP32 MatMul overwrote the output guard");
    require(std::all_of(workspace_bytes.begin() + executable.workspace_size(),
                        workspace_bytes.end(),
                        [](std::uint8_t value) { return value == 0xA5; }),
            "FP32 MatMul overwrote the workspace guard");
  }
  std::cout << "PASS FP32 "
            << (short_tensor_map
                    ? "short-K host TensorMap"
                    : (tensor_map
                           ? "host TensorMap"
                           : (test_case == Fp32MatmulCase::Direct ? "direct"
                                                                  : "packed")))
            << " MatMul " << (tf32 ? "TF32-RNE" : "tf32x3")
            << (autotune ? " autotune" : " fixed")
            << " batches, tails, range/cancellation and scratch guard\n";
}
void check_softplus_extremes(flagdnn::Handle& handle, Stream& stream) {
  constexpr std::int64_t count = 17;
  constexpr float sentinel = -97.0F;
  const std::array<std::int64_t, 1> dimensions{count};
  for (const auto type :
       {FLAGDNN_DATA_FLOAT32, FLAGDNN_DATA_FLOAT16, FLAGDNN_DATA_BFLOAT16}) {
    const float largest = type == FLAGDNN_DATA_FLOAT16 ? 60000.0F : 3.0e38F;
    const float infinity = std::numeric_limits<float>::infinity();
    const std::array<float, 12> values{1000.0F,  -1000.0F,  largest, -largest,
                                       infinity, -infinity, -17.0F,  -20.0F,
                                       -30.0F,   17.0F,     20.0F,   30.0F};
    for (const std::int64_t stride : {1}) {
      const std::array<std::int64_t, 1> strides{stride};
      const auto storage =
          static_cast<std::size_t>((count - 1) * stride + 1) + 16;
      std::vector<float> host_x(storage, sentinel);
      for (std::size_t i = 0; i < count; ++i)
        host_x[i * stride] = values[i % values.size()];
      const auto x_bytes =
          tensor_io::encode(host_x, type, tensor_io::BooleanEncoding::kByte);
      const auto quantized = tensor_io::decode(
          x_bytes, type, storage, tensor_io::BooleanEncoding::kByte);
      const auto initial =
          tensor_io::encode(std::vector<float>(storage, sentinel), type,
                            tensor_io::BooleanEncoding::kByte);
      flagdnn::TensorDescriptor x(81, type, dimensions, strides);
      flagdnn::TensorDescriptor y(82, type, dimensions, strides);
      DeviceBuffer input(x_bytes.size()), output(initial.size());
      input.copy_from_host(x_bytes.data(), x_bytes.size(), stream.get());
      for (const double beta : {0.5, 1.0, 2.0}) {
        flagdnnPointwiseAttributes_t attributes =
            FLAGDNN_POINTWISE_ATTRIBUTES_INITIALIZER;
        attributes.flags = FLAGDNN_POINTWISE_ATTRIBUTE_SOFTPLUS_BETA;
        attributes.softplus_beta = beta;
        flagdnn::OperationDescriptor operation(FLAGDNN_OPERATION_POINTWISE);
        operation.set_pointwise(x, FLAGDNN_POINTWISE_SOFTPLUS_FWD, y,
                                attributes);
        flagdnn::Graph graph;
        graph.add(operation);
        graph.finalize();
        flagdnn::Executable executable(handle, graph);
        DeviceBuffer workspace(executable.workspace_size());
        output.copy_from_host(initial.data(), initial.size(), stream.get());
        const std::array<flagdnnBinding_t, 2> bindings{
            {{81, input.opaque()}, {82, output.opaque()}}};
        executable.execute(bindings, workspace.opaque(),
                           executable.workspace_size(), stream.opaque());
        std::vector<std::uint8_t> actual_bytes(initial.size());
        output.copy_to_host(actual_bytes.data(), actual_bytes.size(),
                            stream.get());
        stream.synchronize();
        const auto actual = tensor_io::decode(
            actual_bytes, type, storage, tensor_io::BooleanEncoding::kByte);
        auto cudnn = pointwise_reference(
            FLAGDNN_POINTWISE_SOFTPLUS_FWD,
            {contract_tensor(81, type, dimensions, strides)},
            contract_tensor(82, type, dimensions, strides), attributes);
        const auto reference_bytes = run_contract_reference<std::uint8_t>(
            *cudnn, bindings, 82, initial.size(), stream.get(), initial);
        const auto expected = tensor_io::decode(
            reference_bytes, type, storage, tensor_io::BooleanEncoding::kByte);
        const double tolerance = type == FLAGDNN_DATA_FLOAT32   ? 5.0e-6
                                 : type == FLAGDNN_DATA_FLOAT16 ? 2.0e-3
                                                                : 1.6e-2;
        for (std::size_t i = 0; i < storage; ++i) {
          if (actual[i] == expected[i]) continue;
          // cuDNN's FP32 log(1 + exp(x)) loses tiny positive tails when
          // 1 + exp(x) rounds to one. Bound that rounding independently of
          // output dtype; padding and nonfinite limits still compare exactly.
          if (i < count && i % stride == 0 && std::isfinite(expected[i]) &&
              std::isfinite(actual[i]) &&
              std::abs(static_cast<double>(actual[i]) - expected[i]) <=
                  1.e-6 / beta + tolerance * std::abs(expected[i]))
            continue;
          std::cerr << "softplus mismatch dtype=" << type << " beta=" << beta
                    << " stride=" << stride << " index=" << i
                    << " input=" << quantized[i] << " actual=" << actual[i]
                    << " expected=" << expected[i] << '\n';
          throw std::runtime_error("softplus boundary mismatch");
        }
        std::cout << "PASS softplus extremes dtype=" << type << " beta=" << beta
                  << " stride=" << stride << '\n';
      }
    }
  }
}

void check_tanh_gradient_boundaries(flagdnn::Handle& handle, Stream& stream) {
  constexpr float sentinel = -97.0F;
  constexpr std::size_t guard = 16;
  for (const auto type :
       {FLAGDNN_DATA_FLOAT32, FLAGDNN_DATA_FLOAT16, FLAGDNN_DATA_BFLOAT16}) {
    constexpr float gradient_scale = 16.0F;
    const float small = type == FLAGDNN_DATA_FLOAT16 ? 1.0e-4F : 1.0e-20F;
    const float infinity = std::numeric_limits<float>::infinity();
    const std::array<float, 23> values{-1000.0F,
                                       -100.0F,
                                       -44.0F,
                                       -40.0F,
                                       -20.0F,
                                       -10.0F,
                                       -8.5F,
                                       -4.0F,
                                       -0.5F,
                                       -0.0F,
                                       0.0F,
                                       0.5F,
                                       4.0F,
                                       8.5F,
                                       10.0F,
                                       20.0F,
                                       40.0F,
                                       44.0F,
                                       100.0F,
                                       1000.0F,
                                       -infinity,
                                       infinity,
                                       std::numeric_limits<float>::quiet_NaN()};
    const std::array<float, 8> gradients{
        1.0F,  -1.0F,  gradient_scale, -gradient_scale,
        small, -small, 0.0F,           -0.0F};
    for (const std::int64_t count : {17, 512, 513}) {
      const std::array<std::int64_t, 1> dimensions{count};
      for (const std::int64_t stride : {1}) {
        const std::array<std::int64_t, 1> strides{stride};
        const auto span = static_cast<std::size_t>((count - 1) * stride + 1);
        const auto storage = span + guard;
        std::vector<float> host_x(storage, sentinel),
            host_dy(storage, sentinel);
        for (std::size_t i = 0; i < static_cast<std::size_t>(count); ++i) {
          host_x[i * stride] = values[i % values.size()];
          host_dy[i * stride] = gradients[i % gradients.size()];
        }
        const auto encode = [type](const std::vector<float>& values) {
          return tensor_io::encode(values, type,
                                   tensor_io::BooleanEncoding::kByte);
        };
        const auto decode = [type,
                             storage](const std::vector<std::uint8_t>& bytes) {
          return tensor_io::decode(bytes, type, storage,
                                   tensor_io::BooleanEncoding::kByte);
        };
        const auto x_bytes = encode(host_x), dy_bytes = encode(host_dy);
        const auto quantized_x = decode(x_bytes),
                   quantized_dy = decode(dy_bytes);
        const auto initial = encode(std::vector<float>(storage, sentinel));
        flagdnn::TensorDescriptor dy(91, type, dimensions, strides);
        flagdnn::TensorDescriptor x(92, type, dimensions, strides);
        flagdnn::TensorDescriptor dx(93, type, dimensions, strides);
        flagdnn::OperationDescriptor operation(FLAGDNN_OPERATION_POINTWISE);
        operation.set_pointwise(dy, x, FLAGDNN_POINTWISE_TANH_BWD, dx);
        flagdnn::Graph graph;
        graph.add(operation);
        graph.finalize();
        const bool autotune = count == 512 && stride == 1;
        flagdnnBuildOptions_t options = FLAGDNN_BUILD_OPTIONS_INITIALIZER;
        options.flags = autotune ? FLAGDNN_BUILD_OPTION_AUTOTUNE : 0;
        flagdnn::Executable executable(handle, graph, &options);
        DeviceBuffer input(x_bytes.size()), loss(dy_bytes.size()),
            output(initial.size());
        std::vector<std::uint8_t> scratch(executable.workspace_size() + guard,
                                          0xA5);
        DeviceBuffer workspace(scratch.size());
        input.copy_from_host(x_bytes.data(), x_bytes.size(), stream.get());
        loss.copy_from_host(dy_bytes.data(), dy_bytes.size(), stream.get());
        output.copy_from_host(initial.data(), initial.size(), stream.get());
        workspace.copy_from_host(scratch.data(), scratch.size(), stream.get());
        const std::array<flagdnnBinding_t, 3> bindings{
            {{91, loss.opaque()}, {92, input.opaque()}, {93, output.opaque()}}};
        executable.execute(bindings, workspace.opaque(),
                           executable.workspace_size(), stream.opaque());
        std::vector<std::uint8_t> actual_bytes(initial.size());
        output.copy_to_host(actual_bytes.data(), actual_bytes.size(),
                            stream.get());
        workspace.copy_to_host(scratch.data(), scratch.size(), stream.get());
        stream.synchronize();
        const auto actual = decode(actual_bytes);
        auto cudnn = pointwise_reference(
            FLAGDNN_POINTWISE_TANH_BWD,
            {contract_tensor(91, type, dimensions, strides),
             contract_tensor(92, type, dimensions, strides)},
            contract_tensor(93, type, dimensions, strides));
        const auto expected = decode(run_contract_reference<std::uint8_t>(
            *cudnn, bindings, 93, initial.size(), stream.get(), initial));
        const double tolerance = type == FLAGDNN_DATA_FLOAT32   ? 5.0e-6
                                 : type == FLAGDNN_DATA_FLOAT16 ? 2.0e-3
                                                                : 1.6e-2;
        for (std::size_t i = 0; i < storage; ++i) {
          if (actual[i] == expected[i] ||
              (std::isnan(actual[i]) && std::isnan(expected[i])))
            continue;
          // cuDNN evaluates an approximate tanh derivative. Its absolute
          // derivative error scales with the supplied upstream gradient.
          if (i < span && i % stride == 0 && std::isfinite(actual[i]) &&
              std::isfinite(expected[i]) &&
              std::abs(static_cast<double>(actual[i]) - expected[i]) <=
                  2.e-5 * std::max(1.0, std::abs(static_cast<double>(
                                            quantized_dy[i]))) +
                      tolerance * std::abs(static_cast<double>(expected[i])))
            continue;
          std::cerr << "tanh gradient mismatch dtype=" << type
                    << " count=" << count << " stride=" << stride
                    << " index=" << i << " input=" << quantized_x[i]
                    << " dy=" << quantized_dy[i] << " actual=" << actual[i]
                    << " expected=" << expected[i] << '\n';
          throw std::runtime_error("tanh gradient boundary mismatch");
        }
        require(std::all_of(scratch.begin() + executable.workspace_size(),
                            scratch.end(),
                            [](std::uint8_t value) { return value == 0xA5; }),
                "tanh gradient overwrote the workspace guard");
        std::cout << "PASS tanh gradient boundaries dtype=" << type
                  << " count=" << count << " stride=" << stride
                  << " autotune=" << autotune << '\n';
      }
    }
  }
}

void check_activation_gradient_extremes(flagdnn::Handle& handle,
                                        Stream& stream) {
  constexpr std::int64_t count = 17;
  constexpr float sentinel = -97.0F;
  const std::array<std::int64_t, 1> dimensions{count};
  for (const auto type :
       {FLAGDNN_DATA_FLOAT32, FLAGDNN_DATA_FLOAT16, FLAGDNN_DATA_BFLOAT16}) {
    // cuDNN GELU backward returns NaN for infinite X. Use finite
    // saturation cases accepted by all three standalone backward operators.
    const std::array<float, 6> values{32.0F,  -32.0F,  80.0F,
                                      -80.0F, 1000.0F, -1000.0F};
    for (const std::int64_t stride : {1}) {
      const std::array<std::int64_t, 1> strides{stride};
      const auto storage =
          static_cast<std::size_t>((count - 1) * stride + 1) + 16;
      std::vector<float> host_x(storage, sentinel), host_dy(storage, sentinel);
      for (std::size_t i = 0; i < count; ++i) {
        host_x[i * stride] = values[i % values.size()];
        host_dy[i * stride] = i % 2 ? -2.0F : 2.0F;
      }
      const auto x_bytes =
          tensor_io::encode(host_x, type, tensor_io::BooleanEncoding::kByte);
      const auto dy_bytes =
          tensor_io::encode(host_dy, type, tensor_io::BooleanEncoding::kByte);
      const auto initial =
          tensor_io::encode(std::vector<float>(storage, sentinel), type,
                            tensor_io::BooleanEncoding::kByte);
      flagdnn::TensorDescriptor dy(71, type, dimensions, strides);
      flagdnn::TensorDescriptor x(72, type, dimensions, strides);
      flagdnn::TensorDescriptor dx(73, type, dimensions, strides);
      DeviceBuffer input(x_bytes.size()), loss(dy_bytes.size()),
          output(initial.size());
      input.copy_from_host(x_bytes.data(), x_bytes.size(), stream.get());
      loss.copy_from_host(dy_bytes.data(), dy_bytes.size(), stream.get());
      for (const auto mode :
           {FLAGDNN_POINTWISE_GELU_BWD, FLAGDNN_POINTWISE_SWISH_BWD,
            FLAGDNN_POINTWISE_GELU_APPROX_TANH_BWD}) {
        flagdnnPointwiseAttributes_t attributes =
            FLAGDNN_POINTWISE_ATTRIBUTES_INITIALIZER;
        if (mode == FLAGDNN_POINTWISE_SWISH_BWD) {
          attributes.flags = FLAGDNN_POINTWISE_ATTRIBUTE_SWISH_BETA;
          attributes.swish_beta = 2.0;
        }
        flagdnn::OperationDescriptor operation(FLAGDNN_OPERATION_POINTWISE);
        operation.set_pointwise(dy, x, mode, dx, 1.0, attributes);
        flagdnn::Graph graph;
        graph.add(operation);
        graph.finalize();
        flagdnn::Executable executable(handle, graph);
        DeviceBuffer workspace(executable.workspace_size());
        output.copy_from_host(initial.data(), initial.size(), stream.get());
        const std::array<flagdnnBinding_t, 3> bindings{
            {{71, loss.opaque()}, {72, input.opaque()}, {73, output.opaque()}}};
        executable.execute(bindings, workspace.opaque(),
                           executable.workspace_size(), stream.opaque());
        std::vector<std::uint8_t> actual_bytes(initial.size());
        output.copy_to_host(actual_bytes.data(), actual_bytes.size(),
                            stream.get());
        stream.synchronize();
        const auto actual = tensor_io::decode(
            actual_bytes, type, storage, tensor_io::BooleanEncoding::kByte);
        auto cudnn = pointwise_reference(
            mode,
            {contract_tensor(71, type, dimensions, strides),
             contract_tensor(72, type, dimensions, strides)},
            contract_tensor(73, type, dimensions, strides), attributes);
        const auto reference_bytes = run_contract_reference<std::uint8_t>(
            *cudnn, bindings, 73, initial.size(), stream.get(), initial);
        const auto reference_values = tensor_io::decode(
            reference_bytes, type, storage, tensor_io::BooleanEncoding::kByte);
        for (std::size_t i = 0; i < storage; ++i) {
          const float expected = reference_values[i];
          if (actual[i] != expected) {
            throw std::runtime_error(
                "activation gradient extreme mismatch mode=" +
                std::to_string(mode) + " dtype=" + std::to_string(type) +
                " stride=" + std::to_string(stride) + " index=" +
                std::to_string(i) + " actual=" + std::to_string(actual[i]) +
                " expected=" + std::to_string(expected));
          }
        }
        std::cout << "PASS activation gradient extremes mode=" << mode
                  << " dtype=" << type << " stride=" << stride << '\n';
      }
    }
  }
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 3 &&
        (argc != 4 || (std::string(argv[3]) != "tensor-map-first" &&
                       std::string(argv[3]) != "short-map-first" &&
                       std::string(argv[3]) != "short-map-only" &&
                       std::string(argv[3]) != "ieee-range" &&
                       std::string(argv[3]) != "packed-only"))) {
      throw std::runtime_error("expected compiler executable and entry");
    }
    if (argc == 4 && std::string(argv[3]) == "ieee-range") {
      DriverContext context;
      Stream stream;
      Cache cache;
      flagdnn::Handle handle(FLAGDNN_BACKEND_NVIDIA, 0);
      handle.set_compiler(argv[1], argv[2], cache.path.string());
      check_fp32_convolution(handle, stream, true);
      check_fp32_matmul_layout(handle, stream);
      return 0;
    }
    if (argc == 4) {
      DriverContext context;
      Stream stream;
      Cache cache;
      flagdnn::Handle handle(FLAGDNN_BACKEND_NVIDIA, 0);
      handle.set_compiler(argv[1], argv[2], cache.path.string());
      const auto first_case = std::string(argv[3]) == "packed-only"
                                  ? Fp32MatmulCase::Packed
                              : std::string(argv[3]) != "tensor-map-first"
                                  ? Fp32MatmulCase::TensorMapShort
                                  : Fp32MatmulCase::TensorMap;
      check_fp32_matmul_layout(handle, stream, first_case, false);
      if (std::string(argv[3]) == "short-map-only" ||
          std::string(argv[3]) == "packed-only") {
        check_fp32_matmul_layout(handle, stream, first_case, true);
        return 0;
      }
      const char* initialized_backend = std::getenv("TRITON_JIT_BACKEND");
      require(initialized_backend != nullptr && initialized_backend[0] != '\0',
              "cached JIT reported readiness without initializing the JIT "
              "environment");
      check_build_threads(argv[1], argv[2]);
      std::cout << "PASS cached-first then threaded raw JIT initialization and "
                   "shutdown\n";
      return 0;
    }
    // Start without a caller-owned CUDA context. This also checks that cached
    // JIT functions remain usable after all initial handles have been released.
    check_build_threads(argv[1], argv[2]);
    DriverContext context;
    Stream stream;
    Cache cache;
    flagdnn::Handle handle(FLAGDNN_BACKEND_NVIDIA, 0);
    handle.set_compiler(argv[1], argv[2], cache.path.string());
    check_softplus_extremes(handle, stream);
    check_activation_gradient_extremes(handle, stream);
    check_tanh_gradient_boundaries(handle, stream);
    check_streams_and_graph(handle, stream);
    check_fp32_convolution(handle, stream);
    check_layernorm_row_tail(handle, stream);
    check_matrix_transpose_tails(handle, stream);
    check_matrix_transpose_tails(handle, stream, true);
    for (const auto test_case :
         {Fp32MatmulCase::Packed, Fp32MatmulCase::Direct,
          Fp32MatmulCase::TensorMap, Fp32MatmulCase::TensorMapShort}) {
      check_fp32_matmul_layout(handle, stream, test_case, false);
      check_fp32_matmul_layout(handle, stream, test_case, true);
    }
    check_low_precision_matmul_scratch(handle, stream);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL " << error.what() << '\n';
    return 1;
  }
}
