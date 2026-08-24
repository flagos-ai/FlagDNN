/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include <flagdnn/flagdnn.hpp>

#include <dlfcn.h>
#include <link.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

void expect(bool condition, std::string_view message) {
  if (!condition) {
    throw std::runtime_error(std::string(message));
  }
}

std::filesystem::path required_path(const char *name) {
  const char *value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    throw std::runtime_error(std::string(name) + " is missing");
  }
  const std::filesystem::path path(value);
  if (!path.is_absolute() || !std::filesystem::exists(path)) {
    throw std::runtime_error(std::string(name) +
                             " is not an existing absolute path");
  }
  return std::filesystem::canonical(path);
}

bool is_within(const std::filesystem::path &root,
               const std::filesystem::path &candidate) {
  const std::filesystem::path relative = candidate.lexically_relative(root);
  return !relative.empty() && !relative.is_absolute() &&
         *relative.begin() != "..";
}

void require_clean_resource_environment() {
  constexpr std::array<const char *, 8> forbidden = {
      "FLAGDNN_BACKEND_ROOT", "FLAGDNN_KERNEL_SOURCE_ROOT",
      "FLAGDNN_TUNING_ROOT",  "FLAGDNN_CODEGEN_COMPILER",
      "FLAGDNN_COMPILER",     "PYTHONPATH",
      "PYTHONHOME",           "LIBTRITON_JIT_ROOT",
  };
  for (const char *name : forbidden) {
    if (std::getenv(name) != nullptr) {
      throw std::runtime_error(
          std::string("inherited resource override was not unset: ") + name);
    }
  }
}

using DriverResult = int;
using DriverDevice = int;
struct DriverContextStorage;
struct DriverStreamStorage;
using DriverContext = DriverContextStorage *;
using DriverStream = DriverStreamStorage *;
using DriverPointer = std::uint64_t;

constexpr DriverResult kDriverSuccess = 0;
constexpr unsigned int kNonBlockingStream = 1;

class Driver final {
public:
  explicit Driver(const std::filesystem::path &path) {
    library_ = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (library_ == nullptr) {
      const char *detail = dlerror();
      throw std::runtime_error(
          "cannot load the selected CoreX Driver: " +
          std::string(detail == nullptr ? "unknown dlopen error" : detail));
    }
    init = symbol<Init>("cuInit");
    device_get = symbol<DeviceGet>("cuDeviceGet");
    primary_retain = symbol<PrimaryRetain>("cuDevicePrimaryCtxRetain");
    primary_release = symbol<PrimaryRelease>("cuDevicePrimaryCtxRelease");
    context_get_current = symbol<ContextGetCurrent>("cuCtxGetCurrent");
    context_push = symbol<ContextPush>("cuCtxPushCurrent_v2");
    context_pop = symbol<ContextPop>("cuCtxPopCurrent_v2");
    stream_create = symbol<StreamCreate>("cuStreamCreate");
    stream_destroy = symbol<StreamDestroy>("cuStreamDestroy_v2");
    stream_synchronize = symbol<StreamSynchronize>("cuStreamSynchronize");
    memory_allocate = symbol<MemoryAllocate>("cuMemAlloc_v2");
    memory_free = symbol<MemoryFree>("cuMemFree_v2");
    copy_host_to_device_async =
        symbol<CopyHostToDeviceAsync>("cuMemcpyHtoDAsync_v2");
    copy_device_to_host_async =
        symbol<CopyDeviceToHostAsync>("cuMemcpyDtoHAsync_v2");
    error_string = symbol<ErrorString>("cuGetErrorString");
  }

  ~Driver() {
    if (library_ != nullptr) {
      (void)dlclose(library_);
    }
  }

  Driver(const Driver &) = delete;
  Driver &operator=(const Driver &) = delete;

  using Init = DriverResult (*)(unsigned int);
  using DeviceGet = DriverResult (*)(DriverDevice *, int);
  using PrimaryRetain = DriverResult (*)(DriverContext *, DriverDevice);
  using PrimaryRelease = DriverResult (*)(DriverDevice);
  using ContextGetCurrent = DriverResult (*)(DriverContext *);
  using ContextPush = DriverResult (*)(DriverContext);
  using ContextPop = DriverResult (*)(DriverContext *);
  using StreamCreate = DriverResult (*)(DriverStream *, unsigned int);
  using StreamDestroy = DriverResult (*)(DriverStream);
  using StreamSynchronize = DriverResult (*)(DriverStream);
  using MemoryAllocate = DriverResult (*)(DriverPointer *, std::size_t);
  using MemoryFree = DriverResult (*)(DriverPointer);
  using CopyHostToDeviceAsync = DriverResult (*)(DriverPointer, const void *,
                                                 std::size_t, DriverStream);
  using CopyDeviceToHostAsync = DriverResult (*)(void *, DriverPointer,
                                                 std::size_t, DriverStream);
  using ErrorString = DriverResult (*)(DriverResult, const char **);

  Init init = nullptr;
  DeviceGet device_get = nullptr;
  PrimaryRetain primary_retain = nullptr;
  PrimaryRelease primary_release = nullptr;
  ContextGetCurrent context_get_current = nullptr;
  ContextPush context_push = nullptr;
  ContextPop context_pop = nullptr;
  StreamCreate stream_create = nullptr;
  StreamDestroy stream_destroy = nullptr;
  StreamSynchronize stream_synchronize = nullptr;
  MemoryAllocate memory_allocate = nullptr;
  MemoryFree memory_free = nullptr;
  CopyHostToDeviceAsync copy_host_to_device_async = nullptr;
  CopyDeviceToHostAsync copy_device_to_host_async = nullptr;
  ErrorString error_string = nullptr;

  void check(DriverResult result, const char *operation) const {
    if (result == kDriverSuccess) {
      return;
    }
    const char *detail = nullptr;
    if (error_string != nullptr) {
      (void)error_string(result, &detail);
    }
    throw std::runtime_error(
        std::string(operation) + " failed: " +
        (detail == nullptr ? "unknown CoreX Driver error" : detail));
  }

private:
  template <typename Function> Function symbol(const char *name) {
    dlerror();
    void *value = dlsym(library_, name);
    const char *detail = dlerror();
    if (value == nullptr || detail != nullptr) {
      throw std::runtime_error(std::string("CoreX Driver symbol is missing: ") +
                               name);
    }
    return reinterpret_cast<Function>(value);
  }

  void *library_ = nullptr;
};

class PrimaryContext final {
public:
  explicit PrimaryContext(Driver &driver) : driver_(driver) {
    driver_.check(driver_.init(0), "cuInit");
    driver_.check(driver_.device_get(&device_, 0), "cuDeviceGet");
    driver_.check(driver_.primary_retain(&context_, device_),
                  "cuDevicePrimaryCtxRetain");
    DriverContext current = nullptr;
    driver_.check(driver_.context_get_current(&current), "cuCtxGetCurrent");
    if (current != context_) {
      driver_.check(driver_.context_push(context_), "cuCtxPushCurrent");
      pushed_ = true;
    }
  }

  ~PrimaryContext() {
    if (pushed_) {
      DriverContext ignored = nullptr;
      (void)driver_.context_pop(&ignored);
    }
    if (context_ != nullptr) {
      (void)driver_.primary_release(device_);
    }
  }

  PrimaryContext(const PrimaryContext &) = delete;
  PrimaryContext &operator=(const PrimaryContext &) = delete;

private:
  Driver &driver_;
  DriverDevice device_ = 0;
  DriverContext context_ = nullptr;
  bool pushed_ = false;
};

class Stream final {
public:
  explicit Stream(Driver &driver) : driver_(driver) {
    driver_.check(driver_.stream_create(&value_, kNonBlockingStream),
                  "cuStreamCreate");
  }
  ~Stream() {
    if (value_ != nullptr) {
      (void)driver_.stream_destroy(value_);
    }
  }
  Stream(const Stream &) = delete;
  Stream &operator=(const Stream &) = delete;

  [[nodiscard]] DriverStream get() const noexcept { return value_; }
  [[nodiscard]] flagdnnStream_t opaque() const noexcept {
    return reinterpret_cast<flagdnnStream_t>(value_);
  }

private:
  Driver &driver_;
  DriverStream value_ = nullptr;
};

class DeviceBuffer final {
public:
  DeviceBuffer(Driver &driver, std::size_t bytes) : driver_(driver) {
    driver_.check(driver_.memory_allocate(&value_, bytes), "cuMemAlloc");
  }
  ~DeviceBuffer() {
    if (value_ != 0) {
      (void)driver_.memory_free(value_);
    }
  }
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;

  [[nodiscard]] DriverPointer get() const noexcept { return value_; }
  [[nodiscard]] void *opaque() const noexcept {
    return reinterpret_cast<void *>(static_cast<std::uintptr_t>(value_));
  }

private:
  Driver &driver_;
  DriverPointer value_ = 0;
};

flagdnn::Executable build_graph(const flagdnn::Handle &handle) {
  const std::array<std::int64_t, 2> dimensions = {32, 32};
  const std::array<std::int64_t, 2> strides = {32, 1};
  flagdnn::TensorDescriptor left(1, FLAGDNN_DATA_FLOAT32, dimensions, strides);
  flagdnn::TensorDescriptor right(2, FLAGDNN_DATA_FLOAT32, dimensions, strides);
  flagdnn::TensorDescriptor added(3, FLAGDNN_DATA_FLOAT32, dimensions, strides);
  added.set_virtual();
  flagdnn::TensorDescriptor activated(4, FLAGDNN_DATA_FLOAT32, dimensions,
                                      strides);
  activated.set_virtual();
  flagdnn::TensorDescriptor output(5, FLAGDNN_DATA_FLOAT32, dimensions,
                                   strides);
  flagdnn::Graph graph;
  graph.pointwise(left, right, FLAGDNN_POINTWISE_ADD, added, 1.0);
  graph.relu(added, activated);
  flagdnn::OperationDescriptor transpose("transpose");
  transpose.set_input("input", activated);
  transpose.set_output("output", output);
  constexpr std::array<std::int64_t, 2> permutation = {1, 0};
  transpose.set_attribute("permutation", permutation);
  transpose.finalize();
  graph.add(transpose);
  graph.finalize();
  return flagdnn::Executable(handle, graph);
}

void require_installed_jit(const std::filesystem::path &expected) {
  void *handle = dlopen(expected.c_str(), RTLD_NOW | RTLD_NOLOAD);
  if (handle == nullptr) {
    const char *detail = dlerror();
    throw std::runtime_error(
        "installed consumer did not map the /usr/local IX JIT: " +
        std::string(detail == nullptr ? "unknown dlopen error" : detail));
  }
  link_map *mapping = nullptr;
  if (dlinfo(handle, RTLD_DI_LINKMAP, &mapping) != 0 || mapping == nullptr ||
      mapping->l_name == nullptr || mapping->l_name[0] == '\0') {
    (void)dlclose(handle);
    throw std::runtime_error("cannot identify the mapped IX JIT image");
  }
  const std::filesystem::path loaded =
      std::filesystem::canonical(mapping->l_name);
  (void)dlclose(handle);
  expect(loaded == expected,
         "installed consumer mapped IX JIT outside /usr/local");
}

} // namespace

int main() {
  try {
    require_clean_resource_environment();
    const std::filesystem::path sdk = required_path("FLAGDNN_INSTALLED_SDK");
    const std::filesystem::path provider =
        required_path("FLAGDNN_INSTALLED_PROVIDER");
    const std::filesystem::path platform_registry =
        required_path("FLAGDNN_INSTALLED_PLATFORM_REGISTRY");
    const std::array<std::filesystem::path, 4> platform_resources = {
        required_path("FLAGDNN_INSTALLED_PLATFORM_BINARY_KERNEL"),
        required_path("FLAGDNN_INSTALLED_PLATFORM_UNARY_KERNEL"),
        required_path("FLAGDNN_INSTALLED_PLATFORM_LAYOUT_KERNEL"),
        required_path("FLAGDNN_INSTALLED_PLATFORM_TUNING"),
    };
    const std::filesystem::path cache =
        required_path("FLAGDNN_CACHE_DIRECTORY");
    const std::filesystem::path expected_jit =
        required_path("FLAGDNN_INSTALLED_EXPECTED_JIT");
    expect(is_within(sdk, provider),
           "installed provider escapes the FlagDNN SDK");
    expect(is_within(sdk, platform_registry),
           "installed platform registry escapes the FlagDNN SDK");
    for (const auto &resource : platform_resources) {
      expect(is_within(sdk, resource),
             "installed platform resource escapes the FlagDNN SDK");
    }
    expect(!is_within(sdk, expected_jit),
           "external IX JIT was copied into the FlagDNN SDK");

    Driver driver(required_path("FLAGDNN_INSTALLED_COREX_DRIVER"));
    PrimaryContext context(driver);
    flagdnn::Handle handle("iluvatar", 0);
    auto first = build_graph(handle);
    auto cached = build_graph(handle);
    expect(first.operation_count() == 3 && cached.operation_count() == 3,
           "installed Graph must contain Add, Relu and Transpose");
    expect(first.workspace_size() > 0 &&
               first.workspace_size() == cached.workspace_size(),
           "installed multi-node Graph workspace is invalid");

    std::array<float, 1024> left{};
    std::array<float, 1024> right{};
    std::array<float, 1024> output{};
    for (std::size_t index = 0; index < left.size(); ++index) {
      left[index] = static_cast<float>(index % 31) - 7.0F;
      right[index] = static_cast<float>(index % 13) * 0.125F;
    }
    Stream stream(driver);
    DeviceBuffer device_left(driver, sizeof(left));
    DeviceBuffer device_right(driver, sizeof(right));
    DeviceBuffer device_output(driver, sizeof(output));
    DeviceBuffer workspace(driver, first.workspace_size());
    driver.check(driver.copy_host_to_device_async(device_left.get(),
                                                  left.data(), sizeof(left),
                                                  stream.get()),
                 "cuMemcpyHtoDAsync(left)");
    driver.check(driver.copy_host_to_device_async(device_right.get(),
                                                  right.data(), sizeof(right),
                                                  stream.get()),
                 "cuMemcpyHtoDAsync(right)");
    const std::array<flagdnnBinding_t, 3> bindings = {{
        {1, device_left.opaque()},
        {2, device_right.opaque()},
        {5, device_output.opaque()},
    }};
    first.execute(bindings, workspace.opaque(), first.workspace_size(),
                  stream.opaque());
    cached.execute(bindings, workspace.opaque(), cached.workspace_size(),
                   stream.opaque());
    driver.check(driver.copy_device_to_host_async(output.data(),
                                                  device_output.get(),
                                                  sizeof(output), stream.get()),
                 "cuMemcpyDtoHAsync(output)");
    driver.check(driver.stream_synchronize(stream.get()),
                 "cuStreamSynchronize");
    constexpr std::size_t extent = 32;
    for (std::size_t row = 0; row < extent; ++row) {
      for (std::size_t column = 0; column < extent; ++column) {
        const std::size_t input_index = row * extent + column;
        const std::size_t output_index = column * extent + row;
        const float expected =
            std::max(left[input_index] + right[input_index], 0.0F);
        if (std::fabs(output[output_index] - expected) > 1.0e-6F) {
          throw std::runtime_error(
              "installed Graph result mismatch at element " +
              std::to_string(output_index));
        }
      }
    }

    require_installed_jit(expected_jit);
    std::size_t manifest_count = 0;
    std::size_t source_count = 0;
    for (const auto &entry :
         std::filesystem::recursive_directory_iterator(cache)) {
      if (!entry.is_regular_file()) {
        continue;
      }
      manifest_count += entry.path().filename() == "manifest.json";
      const std::string filename = entry.path().filename().string();
      source_count += filename.rfind("generated_stage_", 0) == 0 &&
                      entry.path().extension() == ".py";
    }
    expect(manifest_count == 1 && source_count == 3,
           "installed consumer cache is incomplete or ambiguous");
    std::cout << "PASS installed Iluvatar Add/Relu/Transpose consumer; cache="
              << cache << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
