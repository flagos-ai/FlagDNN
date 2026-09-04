/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <flagdnn/flagdnn.hpp>
#include <flagdnn_frontend.h>

#include <musa_runtime_api.h>
#include <mudnn.h>

#include <dlfcn.h>
#include <link.h>

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace {

namespace fe = flagdnn_frontend;

void require(bool condition, std::string message) {
  if (!condition) {
    throw std::runtime_error(std::move(message));
  }
}

void check_musa(musaError_t status, std::string_view operation) {
  if (status == musaSuccess) {
    return;
  }
  const char* name = musaGetErrorName(status);
  const char* detail = musaGetErrorString(status);
  throw std::runtime_error(
      std::string(operation) + " failed: " +
      (name == nullptr ? "unknown" : name) + ": " +
      (detail == nullptr ? "unknown" : detail));
}

void check_mudnn(
    musa::dnn::Status status, std::string_view operation) {
  if (status != musa::dnn::Status::SUCCESS) {
    throw std::runtime_error(
        std::string(operation) + " failed with muDNN status " +
        std::to_string(static_cast<int>(status)));
  }
}

void check_frontend(fe::error_t status, std::string_view operation) {
  if (status.is_bad()) {
    throw std::runtime_error(
        std::string(operation) + " failed: " + status.get_message());
  }
}

std::filesystem::path required_environment_path(const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr || *value == '\0') {
    throw std::runtime_error(std::string(name) + " is missing");
  }
  const std::filesystem::path path(value);
  if (!path.is_absolute() || !std::filesystem::exists(path)) {
    throw std::runtime_error(
        std::string(name) + " is not an existing absolute path");
  }
  return std::filesystem::canonical(path);
}

bool is_within(const std::filesystem::path& root,
               const std::filesystem::path& candidate) {
  const std::filesystem::path relative = candidate.lexically_relative(root);
  return candidate == root ||
         (!relative.empty() && !relative.is_absolute() &&
          *relative.begin() != "..");
}

void require_clean_environment() {
  constexpr std::array<const char*, 20> forbidden = {
      "FLAGDNN_BACKEND",
      "FLAGDNN_BACKEND_ROOT",
      "FLAGDNN_KERNEL_SOURCE_ROOT",
      "FLAGDNN_TUNING_ROOT",
      "FLAGDNN_MTHREADS_ENVIRONMENT_REPORT",
      "FLAGDNN_MTHREADS_TRITON_JIT_ROOT",
      "FLAGDNN_MTHREADS_TRITON_JIT_DIR",
      "FLAGDNN_MTHREADS_TRITON_JIT_LIBRARY",
      "FLAGDNN_MTHREADS_TRITON_JIT_INCLUDE_DIR",
      "FLAGDNN_MTHREADS_TRITON_JIT_SCRIPT_DIR",
      "FLAGDNN_TRITON_JIT_LIBRARY",
      "FLAGDNN_TRITON_JIT_INCLUDE_DIR",
      "FLAGDNN_TRITON_JIT_SCRIPT_DIR",
      "FLAGDNN_CODEGEN_COMPILER",
      "FLAGDNN_CODEGEN_PYTHON",
      "FLAGDNN_COMPILER",
      "FLAGDNN_COMPILER_EXECUTABLE",
      "FLAGDNN_EXECUTION_ENGINE",
      "PYTHONDONTWRITEBYTECODE",
      "PYTHONHOME",
  };
  for (const char* name : forbidden) {
    if (std::getenv(name) != nullptr) {
      throw std::runtime_error(
          std::string("inherited override was not unset: ") + name);
    }
  }
  if (std::getenv("PYTHONPATH") != nullptr) {
    throw std::runtime_error("inherited override was not unset: PYTHONPATH");
  }
}

void require_no_python_bytecode(
    const std::filesystem::path& root,
    std::string_view description) {
  for (const auto& entry :
       std::filesystem::recursive_directory_iterator(root)) {
    if (entry.path().filename() == "__pycache__" ||
        (entry.is_regular_file() && entry.path().extension() == ".pyc")) {
      throw std::runtime_error(
          std::string(description) + " wrote Python bytecode: " +
          entry.path().string());
    }
  }
}

void require_mapped_library(
    const std::filesystem::path& expected,
    std::string_view description) {
  void* handle = dlopen(expected.c_str(), RTLD_NOW | RTLD_NOLOAD);
  if (handle == nullptr) {
    const char* detail = dlerror();
    throw std::runtime_error(
        std::string(description) + " is not mapped: " +
        (detail == nullptr ? "unknown dlopen error" : detail));
  }
  struct link_map* mapping = nullptr;
  if (dlinfo(handle, RTLD_DI_LINKMAP, &mapping) != 0 ||
      mapping == nullptr || mapping->l_name == nullptr ||
      *mapping->l_name == '\0') {
    const char* detail = dlerror();
    static_cast<void>(dlclose(handle));
    throw std::runtime_error(
        "cannot identify mapped " + std::string(description) + ": " +
        (detail == nullptr ? "missing link-map path" : detail));
  }
  const std::filesystem::path actual =
      std::filesystem::canonical(mapping->l_name);
  static_cast<void>(dlclose(handle));
  require(
      actual == expected,
      std::string(description) + " mapped outside the isolated SDK: " +
          actual.string());
}

class Stream final {
 public:
  Stream() {
    check_musa(
        musaStreamCreateWithFlags(&value_, musaStreamNonBlocking),
        "musaStreamCreateWithFlags");
  }

  ~Stream() {
    if (value_ != nullptr) {
      static_cast<void>(musaStreamDestroy(value_));
    }
  }

  Stream(const Stream&) = delete;
  Stream& operator=(const Stream&) = delete;

  [[nodiscard]] musaStream_t get() const noexcept { return value_; }
  [[nodiscard]] flagdnnStream_t opaque() const noexcept {
    return reinterpret_cast<flagdnnStream_t>(value_);
  }

 private:
  musaStream_t value_ = nullptr;
};

class DeviceBuffer final {
 public:
  explicit DeviceBuffer(std::size_t bytes)
      : bytes_(bytes == 0 ? 1 : bytes) {
    check_musa(musaMalloc(&value_, bytes_), "musaMalloc");
  }

  ~DeviceBuffer() {
    if (value_ != nullptr) {
      static_cast<void>(musaFree(value_));
    }
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  [[nodiscard]] void* get() const noexcept { return value_; }

  void copy_from(
      const void* source, std::size_t bytes, musaStream_t stream) const {
    require(bytes <= bytes_, "host-to-device copy exceeds allocation");
    check_musa(
        musaMemcpyAsync(
            value_, source, bytes, musaMemcpyHostToDevice, stream),
        "musaMemcpyAsync(host-to-device)");
  }

  void copy_to(
      void* destination, std::size_t bytes, musaStream_t stream) const {
    require(bytes <= bytes_, "device-to-host copy exceeds allocation");
    check_musa(
        musaMemcpyAsync(
            destination, value_, bytes, musaMemcpyDeviceToHost, stream),
        "musaMemcpyAsync(device-to-host)");
  }

  void clear(musaStream_t stream) const {
    check_musa(
        musaMemsetAsync(value_, 0, bytes_, stream), "musaMemsetAsync");
  }

 private:
  void* value_ = nullptr;
  std::size_t bytes_ = 0;
};

class CapturedGraph final {
 public:
  CapturedGraph(musaStream_t stream, const std::function<void()>& enqueue) {
    check_musa(
        musaStreamBeginCapture(stream, musaStreamCaptureModeThreadLocal),
        "musaStreamBeginCapture(installed Add)");
    try {
      enqueue();
    } catch (...) {
      musaGraph_t abandoned = nullptr;
      static_cast<void>(musaStreamEndCapture(stream, &abandoned));
      if (abandoned != nullptr) {
        static_cast<void>(musaGraphDestroy(abandoned));
      }
      throw;
    }
    musaGraph_t graph = nullptr;
    check_musa(
        musaStreamEndCapture(stream, &graph),
        "musaStreamEndCapture(installed Add)");
    require(graph != nullptr, "MUSA Graph capture returned null");
    try {
      check_musa(
          musaGraphGetNodes(graph, nullptr, &node_count_),
          "musaGraphGetNodes(installed Add)");
      require(node_count_ > 0, "MUSA Graph capture produced no nodes");
      check_musa(
          musaGraphInstantiate(&executable_, graph, 0),
          "musaGraphInstantiate(installed Add)");
    } catch (...) {
      static_cast<void>(musaGraphDestroy(graph));
      throw;
    }
    check_musa(
        musaGraphDestroy(graph), "musaGraphDestroy(installed Add)");
  }

  ~CapturedGraph() {
    if (executable_ != nullptr) {
      static_cast<void>(musaGraphExecDestroy(executable_));
    }
  }

  CapturedGraph(const CapturedGraph&) = delete;
  CapturedGraph& operator=(const CapturedGraph&) = delete;

  void launch(musaStream_t stream) const {
    check_musa(
        musaGraphLaunch(executable_, stream),
        "musaGraphLaunch(installed Add)");
  }

  [[nodiscard]] std::size_t node_count() const noexcept {
    return node_count_;
  }

 private:
  musaGraphExec_t executable_ = nullptr;
  std::size_t node_count_ = 0;
};

fe::graph::Graph::Tensor make_tensor(
    fe::graph::Graph& graph, const char* name, std::int64_t uid) {
  return graph.tensor(
      fe::graph::Tensor_attributes()
          .set_name(name)
          .set_uid(uid)
          .set_data_type(fe::DataType_t::FLOAT)
          .set_dim({256})
          .set_stride({1}));
}

std::unique_ptr<fe::graph::Graph> build_add_graph(
    flagdnn::Handle& handle) {
  auto graph = std::make_unique<fe::graph::Graph>();
  graph->set_name("installed_mthreads_add")
      .set_io_data_type(fe::DataType_t::FLOAT)
      .set_intermediate_data_type(fe::DataType_t::FLOAT)
      .set_compute_data_type(fe::DataType_t::FLOAT)
      .set_autotune(true);
  const auto left = make_tensor(*graph, "left", 1);
  const auto right = make_tensor(*graph, "right", 2);
  auto output = graph->pointwise(
      left,
      right,
      fe::graph::Pointwise_attributes()
          .set_name("add")
          .set_mode(fe::PointwiseMode_t::ADD)
          .set_compute_data_type(fe::DataType_t::FLOAT));
  output->set_name("output")
      .set_uid(3)
      .set_data_type(fe::DataType_t::FLOAT)
      .set_dim({256})
      .set_stride({1})
      .set_output(true);
  check_frontend(
      graph->build(handle, {fe::HeurMode_t::A}),
      "installed mthreads Add graph build");
  require(
      graph->get_workspace_size() >= 0,
      "installed mthreads Add returned negative workspace");
  return graph;
}

struct CacheEvidence {
  std::size_t manifests = 0;
  std::size_t selections = 0;
  std::size_t sources = 0;
  std::filesystem::file_time_type selection_timestamp{};
};

CacheEvidence inspect_cache(const std::filesystem::path& cache) {
  CacheEvidence evidence;
  for (const auto& entry :
       std::filesystem::recursive_directory_iterator(cache)) {
    if (!entry.is_regular_file()) {
      continue;
    }
    const std::string filename = entry.path().filename().string();
    if (filename == "manifest.json") {
      ++evidence.manifests;
    }
    if (entry.path().parent_path().filename() == "tuning" &&
        filename.starts_with("stage-") &&
        entry.path().extension() == ".json") {
      ++evidence.selections;
      evidence.selection_timestamp = entry.last_write_time();
    }
    if (entry.path().extension() == ".py") {
      ++evidence.sources;
    }
    require(
        entry.path().extension() != ".cubin",
        "MThreads installed cache unexpectedly contains a CUDA cubin");
  }
  require(
      evidence.manifests == 1 && evidence.selections == 1 &&
          evidence.sources >= 1,
      "installed Add did not publish one manifest, source, and autotune "
      "selection");
  return evidence;
}

void execute_graph(
    fe::graph::Graph& graph,
    flagdnn::Handle& handle,
    std::span<const flagdnnBinding_t> bindings,
    DeviceBuffer& workspace,
    Stream& stream) {
  check_frontend(
      graph.execute(
          handle,
          bindings,
          workspace.get(),
          static_cast<std::size_t>(graph.get_workspace_size()),
          stream.opaque()),
      "installed mthreads Add graph execute");
}

void run_mudnn_add(
    void* left_pointer,
    void* right_pointer,
    void* output_pointer,
    musaStream_t stream) {
  musa::dnn::Handle handle(0);
  check_mudnn(handle.SetStream(stream), "muDNN Handle::SetStream");
  require(handle.GetStream() == stream, "muDNN did not retain caller stream");
  constexpr std::array<std::int64_t, 1> dimensions = {256};
  constexpr std::array<std::int64_t, 1> strides = {1};
  const auto configure = [&](musa::dnn::Tensor& tensor, void* pointer) {
    check_mudnn(tensor.SetAddr(pointer), "muDNN Tensor::SetAddr");
    check_mudnn(
        tensor.SetType(musa::dnn::Tensor::Type::FLOAT),
        "muDNN Tensor::SetType");
    check_mudnn(
        tensor.SetNdInfo(1, dimensions.data(), strides.data()),
        "muDNN Tensor::SetNdInfo");
  };
  musa::dnn::Tensor left;
  musa::dnn::Tensor right;
  musa::dnn::Tensor output;
  configure(left, left_pointer);
  configure(right, right_pointer);
  configure(output, output_pointer);
  musa::dnn::Binary binary;
  check_mudnn(
      binary.SetMode(musa::dnn::Binary::Mode::ADD),
      "muDNN Binary::SetMode(Add)");
  check_mudnn(
      binary.Run(handle, output, left, right),
      "muDNN Binary::Run(Add)");
}

}  // namespace

int main() {
  try {
    require_clean_environment();
    const std::filesystem::path sdk =
        required_environment_path("FLAGDNN_INSTALLED_EXPECTED_SDK");
    const std::filesystem::path expected_plugin =
        required_environment_path("FLAGDNN_INSTALLED_EXPECTED_PLUGIN");
    const std::filesystem::path expected_jit =
        required_environment_path("FLAGDNN_INSTALLED_EXPECTED_JIT");
    const std::filesystem::path expected_scripts =
        required_environment_path(
            "FLAGDNN_INSTALLED_EXPECTED_JIT_SCRIPT_DIR");
    const std::filesystem::path backend_path =
        required_environment_path("FLAGDNN_BACKEND_PATH");
    const std::filesystem::path cache =
        required_environment_path("FLAGDNN_INSTALLED_CACHE");
    require(
        is_within(sdk, expected_plugin) && is_within(sdk, expected_jit) &&
            is_within(sdk, expected_scripts) &&
            backend_path == expected_plugin.parent_path(),
        "installed MThreads resources escape the isolated SDK");
    require(
        std::filesystem::directory_iterator(cache) ==
            std::filesystem::directory_iterator(),
        "installed MThreads cache was not empty at test start");
    if (setenv("FLAGDNN_CACHE_DIRECTORY", cache.c_str(), 1) != 0) {
      throw std::runtime_error("cannot select installed MThreads cache");
    }

    check_musa(musaSetDevice(0), "musaSetDevice");
    flagdnn::Handle explicit_handle("mthreads", 0);
    flagdnn::Handle default_handle;
    require(
        explicit_handle.backend_name() == "mthreads" &&
            default_handle.backend_name() == "mthreads" &&
            explicit_handle.target_fingerprint() == "musa-mtgpu-cc31-w32" &&
            default_handle.target_fingerprint() == "musa-mtgpu-cc31-w32",
        "installed explicit/default MThreads Handle identity differs");

    auto explicit_graph = build_add_graph(explicit_handle);
    require_no_python_bytecode(cache, "installed MThreads artifact");
    require_no_python_bytecode(sdk, "installed MThreads SDK");
    const CacheEvidence cold_cache = inspect_cache(cache);
    auto default_graph = build_add_graph(default_handle);
    require_no_python_bytecode(cache, "installed MThreads artifact");
    require_no_python_bytecode(sdk, "installed MThreads SDK");
    require(
        std::getenv("PYTHONDONTWRITEBYTECODE") == nullptr,
        "MThreads JIT leaked its temporary Python bytecode setting");
    const CacheEvidence warm_cache = inspect_cache(cache);
    require(
        cold_cache.selection_timestamp == warm_cache.selection_timestamp,
        "installed MThreads autotune cache hit rewrote its selection");

    require_mapped_library(expected_plugin, "MThreads backend plugin");
    require_mapped_library(expected_jit, "MThreads private libtriton_jit");

    constexpr std::size_t element_count = 256;
    constexpr std::size_t tensor_bytes = element_count * sizeof(float);
    std::array<float, element_count> host_left{};
    std::array<float, element_count> host_right{};
    std::array<float, element_count> explicit_output{};
    std::array<float, element_count> default_output{};
    std::array<float, element_count> reference_output{};
    for (std::size_t index = 0; index < element_count; ++index) {
      host_left[index] =
          static_cast<float>(static_cast<int>(index % 29) - 14) * 0.25F;
      host_right[index] =
          static_cast<float>(static_cast<int>(index % 17) - 8) * 0.5F;
    }

    Stream stream;
    DeviceBuffer device_left(tensor_bytes);
    DeviceBuffer device_right(tensor_bytes);
    DeviceBuffer device_explicit_output(tensor_bytes);
    DeviceBuffer device_default_output(tensor_bytes);
    DeviceBuffer device_reference_output(tensor_bytes);
    DeviceBuffer explicit_workspace(
        static_cast<std::size_t>(explicit_graph->get_workspace_size()));
    DeviceBuffer default_workspace(
        static_cast<std::size_t>(default_graph->get_workspace_size()));
    device_left.copy_from(host_left.data(), tensor_bytes, stream.get());
    device_right.copy_from(host_right.data(), tensor_bytes, stream.get());
    device_explicit_output.clear(stream.get());
    device_default_output.clear(stream.get());
    device_reference_output.clear(stream.get());
    check_musa(
        musaStreamSynchronize(stream.get()),
        "musaStreamSynchronize(installed inputs)");

    const std::array<flagdnnBinding_t, 3> explicit_bindings = {{
        {1, device_left.get()},
        {2, device_right.get()},
        {3, device_explicit_output.get()},
    }};
    const std::array<flagdnnBinding_t, 3> default_bindings = {{
        {1, device_left.get()},
        {2, device_right.get()},
        {3, device_default_output.get()},
    }};
    CapturedGraph captured(
        stream.get(),
        [&] {
          execute_graph(
              *explicit_graph,
              explicit_handle,
              explicit_bindings,
              explicit_workspace,
              stream);
        });
    captured.launch(stream.get());
    execute_graph(
        *default_graph,
        default_handle,
        default_bindings,
        default_workspace,
        stream);
    run_mudnn_add(
        device_left.get(),
        device_right.get(),
        device_reference_output.get(),
        stream.get());

    device_explicit_output.copy_to(
        explicit_output.data(), tensor_bytes, stream.get());
    device_default_output.copy_to(
        default_output.data(), tensor_bytes, stream.get());
    device_reference_output.copy_to(
        reference_output.data(), tensor_bytes, stream.get());
    check_musa(
        musaStreamSynchronize(stream.get()),
        "musaStreamSynchronize(installed outputs)");
    for (std::size_t index = 0; index < element_count; ++index) {
      const float expected = host_left[index] + host_right[index];
      require(
          std::bit_cast<std::uint32_t>(explicit_output[index]) ==
                  std::bit_cast<std::uint32_t>(reference_output[index]) &&
              std::bit_cast<std::uint32_t>(default_output[index]) ==
                  std::bit_cast<std::uint32_t>(reference_output[index]) &&
              std::fabs(reference_output[index] - expected) <= 1.0e-6F,
          "installed MThreads Graph/muDNN Add mismatch at element " +
              std::to_string(index));
    }

    std::cout
        << "PASS installed MThreads explicit/default Graph Add == direct "
           "muDNN Binary Add; musa_graph_nodes="
        << captured.node_count()
        << ";manifests=" << warm_cache.manifests
        << ";autotune_selections=" << warm_cache.selections << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
