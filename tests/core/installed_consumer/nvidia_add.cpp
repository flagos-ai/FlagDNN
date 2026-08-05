/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include <flagdnn/flagdnn.hpp>
#include <flagdnn_frontend.h>

#include <cuda.h>

#include <unistd.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace fe = ::flagdnn_frontend;

void check_cuda(CUresult status, std::string_view operation) {
  if (status == CUDA_SUCCESS) {
    return;
  }
  const char* detail = nullptr;
  (void)cuGetErrorString(status, &detail);
  throw std::runtime_error(
      std::string(operation) + " failed: " +
      (detail == nullptr ? "unknown CUDA Driver error" : detail));
}

void check_frontend(fe::error_t status, std::string_view operation) {
  if (status.is_bad()) {
    throw std::runtime_error(
        std::string(operation) + " failed: " + status.get_message());
  }
}

class TemporaryCache {
 public:
  TemporaryCache() {
    std::string pattern =
        (std::filesystem::temp_directory_path() /
         "flagdnn-installed-add-XXXXXX")
            .string();
    std::vector<char> writable(pattern.begin(), pattern.end());
    writable.push_back('\0');
    char* created = mkdtemp(writable.data());
    if (created == nullptr) {
      throw std::runtime_error("mkdtemp failed");
    }
    path_ = created;
    if (setenv("FLAGDNN_CACHE_DIRECTORY", path_.c_str(), 1) != 0 ||
        setenv("FLAGDNN_EXECUTION_ENGINE", "libtriton_jit", 1) != 0) {
      throw std::runtime_error("cannot configure installed Add environment");
    }
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
    check_cuda(cuStreamCreate(&stream_, CU_STREAM_NON_BLOCKING),
               "cuStreamCreate");
  }

  ~Stream() {
    if (stream_ != nullptr) {
      (void)cuStreamDestroy(stream_);
    }
  }

  [[nodiscard]] CUstream get() const noexcept { return stream_; }
  [[nodiscard]] flagdnnStream_t opaque() const noexcept {
    return reinterpret_cast<flagdnnStream_t>(stream_);
  }

 private:
  CUstream stream_ = nullptr;
};

class DeviceBuffer {
 public:
  explicit DeviceBuffer(std::size_t bytes) : bytes_(bytes) {
    if (bytes_ != 0) {
      check_cuda(cuMemAlloc(&pointer_, bytes_), "cuMemAlloc");
    }
  }

  ~DeviceBuffer() {
    if (pointer_ != 0) {
      (void)cuMemFree(pointer_);
    }
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  [[nodiscard]] void* opaque() const noexcept {
    return reinterpret_cast<void*>(static_cast<std::uintptr_t>(pointer_));
  }

  void copy_from(const void* source, CUstream stream) const {
    check_cuda(cuMemcpyHtoDAsync(pointer_, source, bytes_, stream),
               "cuMemcpyHtoDAsync");
  }

  void copy_to(void* destination, CUstream stream) const {
    check_cuda(cuMemcpyDtoHAsync(destination, pointer_, bytes_, stream),
               "cuMemcpyDtoHAsync");
  }

 private:
  CUdeviceptr pointer_ = 0;
  std::size_t bytes_ = 0;
};

fe::graph::Graph::Tensor make_tensor(fe::graph::Graph& graph,
                                     const char* name,
                                     std::int64_t uid) {
  return graph.tensor(fe::graph::Tensor_attributes()
                          .set_name(name)
                          .set_uid(uid)
                          .set_data_type(fe::DataType_t::FLOAT)
                          .set_dim({256})
                          .set_stride({1}));
}

void require_jit_autotune_artifacts(const std::filesystem::path& cache) {
  std::size_t manifests = 0;
  std::size_t selections = 0;
  std::size_t cubins = 0;
  for (const auto& entry :
       std::filesystem::recursive_directory_iterator(cache)) {
    if (!entry.is_regular_file()) {
      continue;
    }
    const std::string filename = entry.path().filename().string();
    manifests += filename == "manifest.json" ? 1U : 0U;
    selections += filename.starts_with(".flagdnn-autotune-v1-stage-")
                      ? 1U
                      : 0U;
    cubins += entry.path().extension() == ".cubin" ? 1U : 0U;
  }
  if (manifests != 1 || selections != 1 || cubins != 0) {
    throw std::runtime_error(
        "installed Add did not use one libtriton_jit artifact and autotune selection");
  }
}

}  // namespace

int main() {
  try {
    TemporaryCache cache;
    DriverContext driver;
    flagdnn::Handle handle(FLAGDNN_BACKEND_NVIDIA, 0);
    Stream stream;

    fe::graph::Graph graph;
    graph.set_name("installed_nvidia_add")
        .set_io_data_type(fe::DataType_t::FLOAT)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT)
        .set_autotune(true);
    const auto left = make_tensor(graph, "left", 1);
    const auto right = make_tensor(graph, "right", 2);
    auto output = graph.pointwise(
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
    check_frontend(graph.build(handle, {fe::HeurMode_t::A}),
                   "installed Add graph build");
    require_jit_autotune_artifacts(cache.path());

    std::array<float, 256> host_left{};
    std::array<float, 256> host_right{};
    std::array<float, 256> host_output{};
    for (std::size_t index = 0; index < host_left.size(); ++index) {
      host_left[index] = static_cast<float>(index) * 0.25F;
      host_right[index] = 7.0F - static_cast<float>(index) * 0.125F;
    }
    constexpr std::size_t tensor_bytes = sizeof(host_left);
    DeviceBuffer device_left(tensor_bytes);
    DeviceBuffer device_right(tensor_bytes);
    DeviceBuffer device_output(tensor_bytes);
    const std::int64_t workspace_size = graph.get_workspace_size();
    if (workspace_size < 0) {
      throw std::runtime_error("installed Add returned negative workspace");
    }
    DeviceBuffer workspace(static_cast<std::size_t>(workspace_size));

    device_left.copy_from(host_left.data(), stream.get());
    device_right.copy_from(host_right.data(), stream.get());
    const std::array<flagdnnBinding_t, 3> bindings = {
        flagdnnBinding_t{1, device_left.opaque()},
        flagdnnBinding_t{2, device_right.opaque()},
        flagdnnBinding_t{3, device_output.opaque()}};
    check_frontend(graph.execute(handle,
                                 bindings,
                                 workspace.opaque(),
                                 static_cast<std::size_t>(workspace_size),
                                 stream.opaque()),
                   "installed Add execute");
    device_output.copy_to(host_output.data(), stream.get());
    check_cuda(cuStreamSynchronize(stream.get()), "cuStreamSynchronize");

    for (std::size_t index = 0; index < host_output.size(); ++index) {
      const float expected = host_left[index] + host_right[index];
      if (std::abs(host_output[index] - expected) > 1.0e-6F) {
        throw std::runtime_error(
            "installed Add output differs at index " +
            std::to_string(index));
      }
    }

    std::cout << "PASS installed FlagDNN C++ Graph Add -> libtriton_jit -> "
                 "autotune -> NVIDIA GPU\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
