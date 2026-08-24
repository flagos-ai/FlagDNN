/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include <flagdnn/flagdnn.hpp>
#include <flagdnn_frontend.h>

#include <triton_jit/backend_config.h>
#include <triton_jit/backends/ix_backend.h>
#include <triton_jit/triton_kernel.h>

#include <cuda.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#ifndef BACKEND_IX
#error "The Iluvatar Graph contract must compile with BACKEND_IX"
#endif

static_assert(triton_jit::IxBackend::WARP_SIZE == 64);
static_assert(
    std::is_same_v<triton_jit::DefaultBackend, triton_jit::IxBackend>);

namespace {

namespace fe = ::flagdnn_frontend;

void expect(bool condition, std::string_view message) {
  if (!condition) {
    throw std::runtime_error(std::string(message));
  }
}

void check_driver(CUresult result, const char *operation) {
  if (result == CUDA_SUCCESS) {
    return;
  }
  const char *detail = nullptr;
  (void)cuGetErrorString(result, &detail);
  throw std::runtime_error(
      std::string(operation) + " failed: " +
      (detail == nullptr ? "unknown CoreX Driver error" : detail));
}

void check_frontend(fe::error_t status, const char *operation) {
  if (status.is_bad()) {
    throw std::runtime_error(std::string(operation) +
                             " failed: " + status.get_message());
  }
}

class TemporaryCache final {
public:
  TemporaryCache() {
    std::string pattern = (std::filesystem::temp_directory_path() /
                           "flagdnn-iluvatar-graph-XXXXXX")
                              .string();
    std::vector<char> writable(pattern.begin(), pattern.end());
    writable.push_back('\0');
    const char *created = mkdtemp(writable.data());
    if (created == nullptr) {
      throw std::runtime_error("mkdtemp failed");
    }
    path_ = created;
  }

  ~TemporaryCache() {
    std::error_code ignored;
    std::filesystem::remove_all(path_, ignored);
  }

  TemporaryCache(const TemporaryCache &) = delete;
  TemporaryCache &operator=(const TemporaryCache &) = delete;

  [[nodiscard]] const std::filesystem::path &path() const noexcept {
    return path_;
  }

  [[nodiscard]] std::filesystem::path selection() const {
    std::filesystem::path result;
    std::size_t count = 0;
    for (const auto &entry :
         std::filesystem::recursive_directory_iterator(path_)) {
      const std::string name = entry.path().filename().string();
      if (entry.is_regular_file() &&
          name.starts_with(".flagdnn-autotune-v1-stage-") &&
          entry.path().extension() == ".json") {
        result = entry.path();
        ++count;
      }
    }
    expect(count == 1,
           "Graph build did not publish exactly one autotune selection");
    return result;
  }

private:
  std::filesystem::path path_;
};

std::string read_file(const std::filesystem::path &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("cannot read " + path.string());
  }
  return std::string(std::istreambuf_iterator<char>(input),
                     std::istreambuf_iterator<char>());
}

class CurrentPrimaryContext final {
public:
  CurrentPrimaryContext() {
    check_driver(cuInit(0), "cuInit");
    check_driver(cuDeviceGet(&device_, 0), "cuDeviceGet");
    check_driver(cuDevicePrimaryCtxRetain(&context_, device_),
                 "cuDevicePrimaryCtxRetain");
    CUcontext current = nullptr;
    check_driver(cuCtxGetCurrent(&current), "cuCtxGetCurrent");
    if (current != context_) {
      check_driver(cuCtxPushCurrent(context_), "cuCtxPushCurrent");
      pushed_ = true;
    }
  }

  ~CurrentPrimaryContext() {
    if (pushed_) {
      CUcontext ignored = nullptr;
      (void)cuCtxPopCurrent(&ignored);
    }
    if (context_ != nullptr) {
      (void)cuDevicePrimaryCtxRelease(device_);
    }
  }

  CurrentPrimaryContext(const CurrentPrimaryContext &) = delete;
  CurrentPrimaryContext &operator=(const CurrentPrimaryContext &) = delete;

private:
  CUdevice device_ = 0;
  CUcontext context_ = nullptr;
  bool pushed_ = false;
};

class Stream final {
public:
  Stream() {
    check_driver(cuStreamCreate(&value_, CU_STREAM_NON_BLOCKING),
                 "cuStreamCreate");
  }
  ~Stream() {
    if (value_ != nullptr) {
      (void)cuStreamDestroy(value_);
    }
  }
  Stream(const Stream &) = delete;
  Stream &operator=(const Stream &) = delete;
  [[nodiscard]] CUstream get() const noexcept { return value_; }

private:
  CUstream value_ = nullptr;
};

class DeviceAllocation final {
public:
  explicit DeviceAllocation(std::size_t bytes) {
    if (bytes != 0) {
      check_driver(cuMemAlloc(&value_, bytes), "cuMemAlloc");
    }
  }
  ~DeviceAllocation() {
    if (value_ != 0) {
      (void)cuMemFree(value_);
    }
  }
  DeviceAllocation(const DeviceAllocation &) = delete;
  DeviceAllocation &operator=(const DeviceAllocation &) = delete;
  [[nodiscard]] CUdeviceptr get() const noexcept { return value_; }
  [[nodiscard]] void *opaque() const noexcept {
    return reinterpret_cast<void *>(static_cast<std::uintptr_t>(value_));
  }

private:
  CUdeviceptr value_ = 0;
};

class CapturedExecution final {
public:
  template <typename Function>
  CapturedExecution(CUstream stream, Function &&execute) {
    check_driver(cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_RELAXED),
                 "cuStreamBeginCapture(FlagDNN Graph)");
    bool capture_active = true;
    try {
      execute();
      check_driver(cuStreamEndCapture(stream, &graph_),
                   "cuStreamEndCapture(FlagDNN Graph)");
      capture_active = false;
      expect(graph_ != nullptr, "CoreX returned a null captured graph");

      check_driver(cuGraphGetNodes(graph_, nullptr, &node_count_),
                   "cuGraphGetNodes(FlagDNN Graph count)");
      expect(node_count_ != 0, "FlagDNN capture produced an empty graph");
      std::vector<CUgraphNode> nodes(node_count_);
      check_driver(cuGraphGetNodes(graph_, nodes.data(), &node_count_),
                   "cuGraphGetNodes(FlagDNN Graph)");
      for (CUgraphNode node : nodes) {
        CUgraphNodeType type = CU_GRAPH_NODE_TYPE_EMPTY;
        check_driver(cuGraphNodeGetType(node, &type),
                     "cuGraphNodeGetType(FlagDNN Graph)");
        expect(type == CU_GRAPH_NODE_TYPE_KERNEL,
               "FlagDNN capture contains a non-kernel node");
      }
      check_driver(
          cuGraphInstantiate(&executable_, graph_, nullptr, nullptr, 0),
          "cuGraphInstantiate(FlagDNN Graph)");
    } catch (...) {
      if (capture_active) {
        CUgraph abandoned = nullptr;
        if (cuStreamEndCapture(stream, &abandoned) == CUDA_SUCCESS &&
            abandoned != nullptr) {
          (void)cuGraphDestroy(abandoned);
        }
      }
      cleanup();
      throw;
    }
  }

  ~CapturedExecution() { cleanup(); }

  CapturedExecution(const CapturedExecution &) = delete;
  CapturedExecution &operator=(const CapturedExecution &) = delete;

  void launch(CUstream stream) const {
    check_driver(cuGraphLaunch(executable_, stream),
                 "cuGraphLaunch(FlagDNN Graph)");
  }

  [[nodiscard]] std::size_t node_count() const noexcept { return node_count_; }

private:
  void cleanup() noexcept {
    if (executable_ != nullptr) {
      (void)cuGraphExecDestroy(executable_);
      executable_ = nullptr;
    }
    if (graph_ != nullptr) {
      (void)cuGraphDestroy(graph_);
      graph_ = nullptr;
    }
  }

  CUgraph graph_ = nullptr;
  CUgraphExec executable_ = nullptr;
  std::size_t node_count_ = 0;
};

void configure_graph(fe::graph::Graph &graph) {
  graph.set_name("iluvatar_public_cuda_graph_contract")
      .set_io_data_type(fe::DataType_t::FLOAT)
      .set_intermediate_data_type(fe::DataType_t::FLOAT)
      .set_compute_data_type(fe::DataType_t::FLOAT)
      .set_autotune(true);
  const auto left = graph.tensor(fe::graph::Tensor_attributes()
                                     .set_name("left")
                                     .set_uid(1)
                                     .set_data_type(fe::DataType_t::FLOAT)
                                     .set_dim({4096})
                                     .set_stride({1}));
  const auto right = graph.tensor(fe::graph::Tensor_attributes()
                                      .set_name("right")
                                      .set_uid(2)
                                      .set_data_type(fe::DataType_t::FLOAT)
                                      .set_dim({4096})
                                      .set_stride({1}));
  auto output =
      graph.pointwise(left, right,
                      fe::graph::Pointwise_attributes()
                          .set_name("add")
                          .set_mode(fe::PointwiseMode_t::ADD)
                          .set_compute_data_type(fe::DataType_t::FLOAT)
                          .set_alpha(1.0));
  output->set_name("output")
      .set_uid(3)
      .set_data_type(fe::DataType_t::FLOAT)
      .set_dim({4096})
      .set_stride({1})
      .set_output(true);
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 4) {
      std::cerr << "usage: flagdnn_test_iluvatar_graph "
                   "PLUGIN COMPILER_EXECUTABLE COMPILER_ENTRY\n";
      return 2;
    }
    const std::filesystem::path plugin = std::filesystem::canonical(argv[1]);
    if (setenv("FLAGDNN_BACKEND_PATH", plugin.parent_path().c_str(), 1) != 0 ||
        setenv("FLAGDNN_EXECUTION_ENGINE", "libtriton_jit", 1) != 0) {
      throw std::runtime_error("cannot configure Graph test environment");
    }

    CurrentPrimaryContext current_context;
    TemporaryCache cache;
    flagdnn::Handle handle("iluvatar", 0);
    handle.set_compiler(argv[2], argv[3], cache.path().string());

    std::atomic<std::size_t> jit_launches{0};
    triton_jit::set_launch_enter_hook([&](const triton_jit::LaunchMetadata &) {
      jit_launches.fetch_add(1, std::memory_order_relaxed);
    });

    fe::graph::Graph graph;
    configure_graph(graph);
    check_frontend(graph.build(handle, {fe::HeurMode_t::A}),
                   "FlagDNN Iluvatar Graph build");
    const std::size_t build_jit_launches =
        jit_launches.load(std::memory_order_relaxed);
    expect(build_jit_launches > 2,
           "Graph build did not finish autotune before capture");
    const std::filesystem::path selection = cache.selection();
    const std::string selection_before_capture = read_file(selection);

    std::array<float, 4096> host_left{};
    std::array<float, 4096> host_right{};
    std::array<float, 4096> host_output{};
    for (std::size_t index = 0; index < host_left.size(); ++index) {
      host_left[index] = static_cast<float>(index % 37) - 9.0F;
      host_right[index] = static_cast<float>(index % 23) * 0.125F;
    }

    Stream stream;
    DeviceAllocation device_left(sizeof(host_left));
    DeviceAllocation device_right(sizeof(host_right));
    DeviceAllocation device_output(sizeof(host_output));
    DeviceAllocation workspace(graph.get_workspace_size());
    check_driver(cuMemcpyHtoDAsync(device_left.get(), host_left.data(),
                                   sizeof(host_left), stream.get()),
                 "cuMemcpyHtoDAsync(Graph left)");
    check_driver(cuMemcpyHtoDAsync(device_right.get(), host_right.data(),
                                   sizeof(host_right), stream.get()),
                 "cuMemcpyHtoDAsync(Graph right)");
    const std::array<flagdnnBinding_t, 3> bindings = {{
        {1, device_left.opaque()},
        {2, device_right.opaque()},
        {3, device_output.opaque()},
    }};
    const auto execute = [&] {
      check_frontend(
          graph.execute(handle, std::span<const flagdnnBinding_t>(bindings),
                        workspace.opaque(), graph.get_workspace_size(),
                        reinterpret_cast<void *>(stream.get())),
          "FlagDNN execute during CoreX capture");
    };

    execute();
    check_driver(cuStreamSynchronize(stream.get()),
                 "cuStreamSynchronize(Graph direct warmup)");
    const std::size_t before_capture =
        jit_launches.load(std::memory_order_relaxed);
    CapturedExecution captured(stream.get(), execute);
    expect(captured.node_count() == 1,
           "single-stage Add capture did not contain one kernel node");
    expect(jit_launches.load(std::memory_order_relaxed) == before_capture,
           "Graph capture re-entered libtriton_jit");

    check_driver(cuMemsetD8Async(device_output.get(), 0, sizeof(host_output),
                                 stream.get()),
                 "cuMemsetD8Async(Graph output)");
    for (int replay = 0; replay < 3; ++replay) {
      captured.launch(stream.get());
    }
    check_driver(cuMemcpyDtoHAsync(host_output.data(), device_output.get(),
                                   sizeof(host_output), stream.get()),
                 "cuMemcpyDtoHAsync(Graph output)");
    check_driver(cuStreamSynchronize(stream.get()),
                 "cuStreamSynchronize(Graph replay)");
    expect(jit_launches.load(std::memory_order_relaxed) == before_capture,
           "Graph replay re-entered libtriton_jit");
    expect(read_file(selection) == selection_before_capture,
           "Graph capture/replay changed the autotune selection cache");

    for (std::size_t index = 0; index < host_output.size(); ++index) {
      if (std::fabs(host_output[index] -
                    (host_left[index] + host_right[index])) > 1.0e-6F) {
        throw std::runtime_error("Graph replay result mismatch at element " +
                                 std::to_string(index));
      }
    }

    triton_jit::clear_launch_hooks();
    std::cout
        << "PASS Iluvatar public execute CoreX CUDA Graph capture/replay; "
        << "nodes=" << captured.node_count()
        << ", build_jit_launches=" << build_jit_launches << '\n';
    return 0;
  } catch (const std::exception &error) {
    triton_jit::clear_launch_hooks();
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
