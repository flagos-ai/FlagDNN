/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "backends/backend_api.h"
#include "backends/iluvatar/engines/libtriton_jit.hpp"

#include <flagdnn/flagdnn.hpp>
#include <flagdnn_frontend.h>

#include <triton_jit/backend_config.h>
#include <triton_jit/backends/ix_backend.h>
#include <triton_jit/triton_kernel.h>

#include <cuda.h>
#include <dlfcn.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#ifndef BACKEND_IX
#error "The Iluvatar JIT contract must compile with BACKEND_IX"
#endif

static_assert(triton_jit::IxBackend::WARP_SIZE == 64);
static_assert(
    std::is_same_v<triton_jit::DefaultBackend, triton_jit::IxBackend>);

namespace {

namespace fe = ::flagdnn_frontend;

void expect(bool condition, const char *message) {
  if (!condition) {
    throw std::runtime_error(message);
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
    std::string pattern =
        (std::filesystem::temp_directory_path() / "flagdnn-iluvatar-jit-XXXXXX")
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

  [[nodiscard]] const std::filesystem::path &path() const noexcept {
    return path_;
  }

private:
  std::filesystem::path path_;
};

class PluginProbe final {
public:
  explicit PluginProbe(const char *path) {
    handle_ = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (handle_ == nullptr) {
      const char *detail = dlerror();
      throw std::runtime_error(
          "cannot load Iluvatar plugin: " +
          std::string(detail == nullptr ? "unknown dlopen error" : detail));
    }
    dlerror();
    auto *symbol = dlsym(handle_, FLAGDNN_BACKEND_GET_API_SYMBOL);
    const char *detail = dlerror();
    if (detail != nullptr || symbol == nullptr) {
      throw std::runtime_error("Iluvatar plugin has no v2 entry point");
    }
    const auto get_api =
        reinterpret_cast<flagdnnBackendGetApiV2Function>(symbol);
    api_ = get_api();
  }

  ~PluginProbe() {
    if (handle_ != nullptr) {
      (void)dlclose(handle_);
    }
  }

  PluginProbe(const PluginProbe &) = delete;
  PluginProbe &operator=(const PluginProbe &) = delete;

  [[nodiscard]] const flagdnnBackendApiV2 &api() const {
    expect(api_ != nullptr, "Iluvatar plugin returned a null API");
    return *api_;
  }

private:
  void *handle_ = nullptr;
  const flagdnnBackendApiV2 *api_ = nullptr;
};

class DeviceAllocation final {
public:
  explicit DeviceAllocation(std::size_t bytes) : bytes_(bytes) {
    if (bytes_ != 0) {
      check_driver(cuMemAlloc(&pointer_, bytes_), "cuMemAlloc");
    }
  }

  ~DeviceAllocation() {
    if (pointer_ != 0) {
      (void)cuMemFree(pointer_);
    }
  }

  DeviceAllocation(const DeviceAllocation &) = delete;
  DeviceAllocation &operator=(const DeviceAllocation &) = delete;

  [[nodiscard]] CUdeviceptr get() const noexcept { return pointer_; }
  [[nodiscard]] void *as_void() const noexcept {
    return reinterpret_cast<void *>(static_cast<std::uintptr_t>(pointer_));
  }

private:
  std::size_t bytes_ = 0;
  CUdeviceptr pointer_ = 0;
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

fe::graph::Graph::Tensor make_tensor(fe::graph::Graph &graph, const char *name,
                                     std::int64_t uid) {
  return graph.tensor(fe::graph::Tensor_attributes()
                          .set_name(name)
                          .set_uid(uid)
                          .set_data_type(fe::DataType_t::FLOAT)
                          .set_dim({1024})
                          .set_stride({1}));
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 4) {
      std::cerr << "usage: flagdnn_test_iluvatar_jit "
                   "PLUGIN COMPILER_EXECUTABLE COMPILER_ENTRY\n";
      return 2;
    }

    const std::filesystem::path plugin = std::filesystem::canonical(argv[1]);
    if (setenv("FLAGDNN_BACKEND_PATH", plugin.parent_path().c_str(), 1) != 0 ||
        setenv("FLAGDNN_EXECUTION_ENGINE", "libtriton_jit", 1) != 0) {
      throw std::runtime_error("cannot configure Iluvatar test environment");
    }

    PluginProbe probe(plugin.c_str());
    const flagdnnBackendApiV2 &api = probe.api();
    expect(api.struct_size >= sizeof(flagdnnBackendApiV2),
           "Iluvatar plugin API structure is too small");
    expect(api.abi_version == FLAGDNN_BACKEND_ABI_VERSION,
           "Iluvatar plugin does not expose ABI v2");
    expect(api.backend_name != nullptr &&
               std::string(api.backend_name) == "iluvatar",
           "Iluvatar plugin reports the wrong backend name");

    std::atomic<std::size_t> jit_launches{0};
    triton_jit::set_launch_enter_hook([&](const triton_jit::LaunchMetadata &) {
      jit_launches.fetch_add(1, std::memory_order_relaxed);
    });

    CurrentPrimaryContext current_context;
    TemporaryCache cache;
    flagdnn::Handle handle("iluvatar", 0);
    expect(handle.backend_name() == "iluvatar",
           "public Handle reports the wrong backend");
    expect(handle.target_fingerprint() == "corex_71",
           "public Handle reports a target other than corex_71");
    handle.set_compiler(argv[2], argv[3], cache.path().string());

    fe::graph::Graph graph;
    graph.set_name("iluvatar_add_jit_contract")
        .set_io_data_type(fe::DataType_t::FLOAT)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT)
        .set_autotune(false);
    const auto left = make_tensor(graph, "left", 1);
    const auto right = make_tensor(graph, "right", 2);
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
        .set_dim({1024})
        .set_stride({1})
        .set_output(true);

    check_frontend(graph.build(handle, {fe::HeurMode_t::A}),
                   "FlagDNN Iluvatar Add graph build");
    const std::size_t build_jit_launches =
        jit_launches.load(std::memory_order_relaxed);
    expect(build_jit_launches != 0,
           "Iluvatar executable build did not prepare an IX JIT kernel");

    std::array<float, 1024> host_left{};
    std::array<float, 1024> host_right{};
    std::array<float, 1024> host_output{};
    for (std::size_t index = 0; index < host_left.size(); ++index) {
      host_left[index] = static_cast<float>(index % 29) - 7.0F;
      host_right[index] = static_cast<float>(index % 17) * 0.5F;
    }

    Stream stream;
    DeviceAllocation device_left(sizeof(host_left));
    DeviceAllocation device_right(sizeof(host_right));
    DeviceAllocation device_output(sizeof(host_output));
    DeviceAllocation workspace(graph.get_workspace_size());
    check_driver(cuMemcpyHtoDAsync(device_left.get(), host_left.data(),
                                   sizeof(host_left), stream.get()),
                 "cuMemcpyHtoDAsync(left)");
    check_driver(cuMemcpyHtoDAsync(device_right.get(), host_right.data(),
                                   sizeof(host_right), stream.get()),
                 "cuMemcpyHtoDAsync(right)");

    const std::array<flagdnnBinding_t, 3> bindings = {{
        {1, device_left.as_void()},
        {2, device_right.as_void()},
        {3, device_output.as_void()},
    }};
    for (int repetition = 0; repetition < 3; ++repetition) {
      check_frontend(
          graph.execute(handle, std::span<const flagdnnBinding_t>(bindings),
                        workspace.as_void(), graph.get_workspace_size(),
                        reinterpret_cast<void *>(stream.get())),
          "FlagDNN Iluvatar Add execute");
    }
    expect(jit_launches.load(std::memory_order_relaxed) == build_jit_launches,
           "steady-state execute re-entered libtriton_jit");

    check_driver(cuMemcpyDtoHAsync(host_output.data(), device_output.get(),
                                   sizeof(host_output), stream.get()),
                 "cuMemcpyDtoHAsync(output)");
    check_driver(cuStreamSynchronize(stream.get()), "cuStreamSynchronize");
    triton_jit::clear_launch_hooks();

    for (std::size_t index = 0; index < host_output.size(); ++index) {
      const float expected = host_left[index] + host_right[index];
      if (std::fabs(host_output[index] - expected) > 1.0e-6F) {
        throw std::runtime_error("Iluvatar Add result mismatch at element " +
                                 std::to_string(index));
      }
    }

    std::cout << "PASS Iluvatar public Graph -> IX JIT -> CoreX Add; "
              << "workspace=" << graph.get_workspace_size()
              << ", build_jit_launches=" << build_jit_launches << '\n';
    return 0;
  } catch (const std::exception &error) {
    triton_jit::clear_launch_hooks();
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
