#include <flagdnn/frontend.hpp>

#include <cuda.h>

#include <unistd.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check_cuda(CUresult result, const char* operation) {
  if (result == CUDA_SUCCESS) {
    return;
  }
  const char* name = nullptr;
  const char* detail = nullptr;
  (void)cuGetErrorName(result, &name);
  (void)cuGetErrorString(result, &detail);
  throw std::runtime_error(
      std::string(operation) + " failed" +
      (name == nullptr ? "" : std::string(" (") + name + ")") +
      (detail == nullptr ? "" : std::string(": ") + detail));
}

void check_frontend(const flagdnn_frontend::error_t& error,
                    const char* operation) {
  if (error.is_bad()) {
    throw std::runtime_error(
        std::string(operation) + " failed: " + error.get_message());
  }
}

bool process_maps_contains(const std::string& needle) {
  std::ifstream maps("/proc/self/maps");
  std::string line;
  while (std::getline(maps, line)) {
    if (line.find(needle) != std::string::npos) {
      return true;
    }
  }
  return false;
}

void require_native_process_clean(const char* stage) {
  const bool has_python = process_maps_contains("libpython");
  const bool has_torch = process_maps_contains("libtorch");
  std::cout << stage << "_has_libpython=" << std::boolalpha << has_python
            << " " << stage << "_has_libtorch=" << has_torch << '\n';
#if !defined(FLAGDNN_EXPECT_LIBTRITON_JIT)
  if (has_python || has_torch) {
    throw std::runtime_error(
        std::string(stage) + " unexpectedly loaded Python or Torch");
  }
#endif
}

class TemporaryCache {
 public:
  TemporaryCache() {
    std::string pattern =
        (std::filesystem::temp_directory_path() /
         "flagdnn-native-graph-XXXXXX")
            .string();
    std::vector<char> writable(pattern.begin(), pattern.end());
    writable.push_back('\0');
    char* created = mkdtemp(writable.data());
    if (created == nullptr) {
      throw std::runtime_error("mkdtemp failed");
    }
    path_ = created;
  }

  ~TemporaryCache() {
    std::error_code ignored;
    std::filesystem::remove_all(path_, ignored);
  }

  [[nodiscard]] const std::filesystem::path& path() const noexcept {
    return path_;
  }

 private:
  std::filesystem::path path_;
};

class DriverContext {
 public:
  DriverContext() {
    check_cuda(cuInit(0), "cuInit");
    check_cuda(cuDeviceGet(&device_, 0), "cuDeviceGet");
    check_cuda(cuDevicePrimaryCtxRetain(&context_, device_),
               "cuDevicePrimaryCtxRetain");
    check_cuda(cuCtxSetCurrent(context_), "cuCtxSetCurrent");
  }

  ~DriverContext() {
    if (context_ != nullptr) {
      (void)cuDevicePrimaryCtxRelease(device_);
    }
  }

  DriverContext(const DriverContext&) = delete;
  DriverContext& operator=(const DriverContext&) = delete;

 private:
  CUdevice device_ = 0;
  CUcontext context_ = nullptr;
};

class Stream {
 public:
  Stream() {
    check_cuda(cuStreamCreate(&value_, CU_STREAM_NON_BLOCKING),
               "cuStreamCreate");
  }

  ~Stream() {
    if (value_ != nullptr) {
      (void)cuStreamDestroy(value_);
    }
  }

  [[nodiscard]] CUstream get() const noexcept { return value_; }

  [[nodiscard]] flagdnnStream_t opaque() const noexcept {
    return reinterpret_cast<flagdnnStream_t>(value_);
  }

 private:
  CUstream value_ = nullptr;
};

class DeviceBuffer {
 public:
  explicit DeviceBuffer(std::size_t size) : size_(size) {
    if (size_ != 0) {
      check_cuda(cuMemAlloc(&value_, size_), "cuMemAlloc");
    }
  }

  ~DeviceBuffer() {
    if (value_ != 0) {
      (void)cuMemFree(value_);
    }
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  [[nodiscard]] void* opaque() const noexcept {
    return reinterpret_cast<void*>(static_cast<std::uintptr_t>(value_));
  }

  void copy_from(const void* source, CUstream stream) const {
    check_cuda(cuMemcpyHtoDAsync(value_, source, size_, stream),
               "cuMemcpyHtoDAsync");
  }

  void copy_to(void* destination, CUstream stream) const {
    check_cuda(cuMemcpyDtoHAsync(destination, value_, size_, stream),
               "cuMemcpyDtoHAsync");
  }

 private:
  CUdeviceptr value_ = 0;
  std::size_t size_ = 0;
};

void require_invalid_graph(flagdnn::Graph& graph, const char* name) {
  const auto require_rejected = [&](auto&& operation, const char* stage) {
    try {
      operation();
    } catch (const flagdnn::Error& error) {
      if (error.status() == FLAGDNN_STATUS_INVALID_VALUE) {
        return;
      }
      throw std::runtime_error(
          std::string(name) + " returned the wrong status from " + stage);
    }
    throw std::runtime_error(
        std::string(name) + " was not rejected by " + stage);
  };
  require_rejected([&] { graph.validate(); }, "validate");
  require_rejected([&] { graph.finalize(); }, "finalize");
}

void test_invalid_graph_contracts() {
  const std::array<std::int64_t, 1> large_dimensions = {1024};
  const std::array<std::int64_t, 1> small_dimensions = {512};
  const std::array<std::int64_t, 1> strides = {1};

  flagdnn::TensorDescriptor virtual_input(
      40, FLAGDNN_DATA_FLOAT32, large_dimensions, strides);
  virtual_input.set_virtual();
  flagdnn::TensorDescriptor output(
      41, FLAGDNN_DATA_FLOAT32, large_dimensions, strides);
  flagdnn::Graph missing_producer;
  missing_producer.relu(virtual_input, output);
  require_invalid_graph(missing_producer, "missing virtual producer");

  flagdnn::TensorDescriptor first_input(
      42, FLAGDNN_DATA_FLOAT32, large_dimensions, strides);
  flagdnn::TensorDescriptor second_input(
      43, FLAGDNN_DATA_FLOAT32, large_dimensions, strides);
  flagdnn::TensorDescriptor shared_output(
      44, FLAGDNN_DATA_FLOAT32, large_dimensions, strides);
  shared_output.set_virtual();
  flagdnn::Graph missing_external_output;
  missing_external_output.relu(first_input, shared_output);
  require_invalid_graph(missing_external_output, "missing non-virtual output");

  flagdnn::Graph duplicate_producer;
  duplicate_producer.relu(first_input, shared_output);
  duplicate_producer.relu(second_input, shared_output);
  require_invalid_graph(duplicate_producer, "duplicate virtual producer");

  flagdnn::TensorDescriptor large_shared(
      45, FLAGDNN_DATA_FLOAT32, large_dimensions, strides);
  large_shared.set_virtual();
  flagdnn::TensorDescriptor small_shared(
      45, FLAGDNN_DATA_FLOAT32, small_dimensions, strides);
  small_shared.set_virtual();
  flagdnn::TensorDescriptor small_output(
      46, FLAGDNN_DATA_FLOAT32, small_dimensions, strides);
  flagdnn::Graph conflicting_metadata;
  conflicting_metadata.relu(first_input, large_shared);
  conflicting_metadata.relu(small_shared, small_output);
  require_invalid_graph(conflicting_metadata,
                        "conflicting shared UID metadata");
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 3) {
      throw std::invalid_argument(
          "usage: native_nvidia_graph_smoke COMPILER_EXECUTABLE COMPILER_ENTRY");
    }
    require_native_process_clean("startup");
    DriverContext driver;
    Stream stream;
    TemporaryCache cache;

    flagdnn::Handle handle("nvidia", 0);
    handle.set_compiler(argv[1], argv[2], cache.path().string());
    test_invalid_graph_contracts();

    flagdnn_frontend::graph::Graph graph;
    graph.set_name("relu_add_chain")
        .set_io_data_type(flagdnn_frontend::DataType_t::FLOAT)
        .set_intermediate_data_type(flagdnn_frontend::DataType_t::FLOAT)
        .set_compute_data_type(flagdnn_frontend::DataType_t::FLOAT);
    auto input = graph.tensor(
        flagdnn_frontend::graph::Tensor_attributes()
            .set_name("input")
            .set_uid(1)
            .set_dim({1024})
            .set_stride({1}));
    auto bias = graph.tensor(
        flagdnn_frontend::graph::Tensor_attributes()
            .set_name("bias")
            .set_uid(2)
            .set_dim({1024})
            .set_stride({1}));
    auto intermediate = graph.pointwise(
        input,
        flagdnn_frontend::graph::Pointwise_attributes()
            .set_name("relu")
            .set_mode(flagdnn_frontend::PointwiseMode_t::RELU_FWD));
    auto output = graph.pointwise(
        intermediate,
        bias,
        flagdnn_frontend::graph::Pointwise_attributes()
            .set_name("add")
            .set_mode(flagdnn_frontend::PointwiseMode_t::ADD));
    output->set_name("output").set_uid(3).set_output(true);

    check_frontend(graph.build(handle), "graph.build");
    require_native_process_clean("after_build");
    if (!intermediate->get_is_virtual() ||
        intermediate->get_uid() <= 0 ||
        intermediate->get_uid() == 1 ||
        intermediate->get_uid() == 2 ||
        intermediate->get_uid() == 3) {
      throw std::runtime_error(
          "frontend did not assign a distinct virtual tensor UID");
    }

    std::int64_t workspace_size = 0;
    check_frontend(
        graph.get_workspace_size(workspace_size),
        "graph.get_workspace_size");
    if (workspace_size < 1024 * static_cast<std::int64_t>(sizeof(float))) {
      throw std::runtime_error("virtual tensor workspace is too small");
    }

    std::vector<float> host_input(1024);
    std::vector<float> host_bias(1024);
    std::vector<float> host_output(1024, 0.0F);
    for (std::size_t index = 0; index < host_input.size(); ++index) {
      host_input[index] =
          static_cast<float>(static_cast<int>(index % 37) - 18) / 7.0F;
      host_bias[index] =
          static_cast<float>(static_cast<int>(index % 11) - 5) / 13.0F;
    }

    const std::size_t bytes = host_input.size() * sizeof(float);
    DeviceBuffer input_buffer(bytes);
    DeviceBuffer bias_buffer(bytes);
    DeviceBuffer output_buffer(bytes);
    DeviceBuffer workspace(static_cast<std::size_t>(workspace_size));
    input_buffer.copy_from(host_input.data(), stream.get());
    bias_buffer.copy_from(host_bias.data(), stream.get());
    const flagdnn_frontend::VariantPack variant_pack = {
        {1, input_buffer.opaque()},
        {2, bias_buffer.opaque()},
        {3, output_buffer.opaque()},
    };

    const auto missing_workspace =
        graph.execute(handle, variant_pack, nullptr, stream.opaque());
    if (missing_workspace.is_good()) {
      throw std::runtime_error("null graph workspace was not rejected");
    }
    check_frontend(
        graph.execute(
            handle, variant_pack, workspace.opaque(), stream.opaque()),
        "graph.execute");
    output_buffer.copy_to(host_output.data(), stream.get());
    check_cuda(cuStreamSynchronize(stream.get()), "cuStreamSynchronize");

    float maximum_error = 0.0F;
    for (std::size_t index = 0; index < host_output.size(); ++index) {
      const float expected =
          std::max(host_input[index], 0.0F) + host_bias[index];
      maximum_error =
          std::max(maximum_error, std::fabs(host_output[index] - expected));
    }
    if (!std::isfinite(maximum_error) || maximum_error != 0.0F) {
      throw std::runtime_error(
          "multi-operation graph result does not match ReLU + Add");
    }

    const std::array<std::int64_t, 1> generic_dimensions = {1024};
    const std::array<std::int64_t, 1> generic_strides = {1};
    flagdnn::TensorDescriptor generic_input(
        101, FLAGDNN_DATA_FLOAT32, generic_dimensions, generic_strides);
    flagdnn::TensorDescriptor generic_output(
        102, FLAGDNN_DATA_FLOAT32, generic_dimensions, generic_strides);
    flagdnn::OperationDescriptor generic_relu("relu");
    generic_relu.set_input("input", generic_input);
    generic_relu.set_output("output", generic_output);
    generic_relu.finalize();
    generic_relu.set_name("generic_cuda_relu");
    generic_relu.set_compute_data_type(FLAGDNN_DATA_FLOAT32);
    flagdnn::Graph generic_graph;
    generic_graph.set_name("generic_cuda_graph");
    generic_graph.add(generic_relu);
    generic_graph.finalize();
    flagdnn::Executable generic_executable(handle, generic_graph);
    const std::array<flagdnnBinding_t, 2> generic_bindings = {
        flagdnnBinding_t{101, input_buffer.opaque()},
        flagdnnBinding_t{102, output_buffer.opaque()}};
    DeviceBuffer generic_workspace(generic_executable.workspace_size());
    generic_executable.execute(
        generic_bindings,
        generic_workspace.opaque(),
        generic_executable.workspace_size(),
        stream.opaque());
    output_buffer.copy_to(host_output.data(), stream.get());
    check_cuda(cuStreamSynchronize(stream.get()),
               "generic descriptor synchronize");
    for (std::size_t index = 0; index < host_output.size(); ++index) {
      if (host_output[index] != std::max(host_input[index], 0.0F)) {
        throw std::runtime_error(
            "generic CUDA descriptor ReLU result differs");
      }
    }
    require_native_process_clean("after_execute");
    std::cout << "PASS multi_operation_graph operations=2 workspace="
              << workspace_size << " max_abs_error=" << maximum_error
              << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
