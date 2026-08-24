/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include <flagdnn/flagdnn.hpp>

#include <cuda.h>
#include <unistd.h>

#include <array>
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
#include <utility>
#include <vector>

namespace {

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

void expect_status(std::string_view name, flagdnnStatus_t expected,
                   flagdnnStatus_t actual) {
  if (actual != expected) {
    const char *detail = flagdnnGetLastErrorString();
    throw std::runtime_error(std::string(name) + " returned status " +
                             std::to_string(static_cast<int>(actual)) +
                             " instead of " +
                             std::to_string(static_cast<int>(expected)) + ": " +
                             (detail == nullptr ? "no diagnostic" : detail));
  }
}

std::string read_file(const std::filesystem::path &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("cannot read " + path.string());
  }
  return std::string(std::istreambuf_iterator<char>(input),
                     std::istreambuf_iterator<char>());
}

void write_file(const std::filesystem::path &path, std::string_view bytes) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output) {
    throw std::runtime_error("cannot write " + path.string());
  }
  output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  if (!output) {
    throw std::runtime_error("short write for " + path.string());
  }
}

std::string replace_once(std::string value, std::string_view before,
                         std::string_view after) {
  const std::size_t position = value.find(before);
  if (position == std::string::npos) {
    throw std::runtime_error("manifest mutation pattern was not found");
  }
  value.replace(position, before.size(), after);
  return value;
}

bool is_sha256_name(std::string_view value) {
  if (value.size() != 64) {
    return false;
  }
  for (const char character : value) {
    if (!((character >= '0' && character <= '9') ||
          (character >= 'a' && character <= 'f'))) {
      return false;
    }
  }
  return true;
}

class TemporaryRuntime final {
public:
  TemporaryRuntime(const char *python) {
    std::string pattern = (std::filesystem::temp_directory_path() /
                           "flagdnn-iluvatar-runtime-XXXXXX")
                              .string();
    std::vector<char> writable(pattern.begin(), pattern.end());
    writable.push_back('\0');
    const char *created = mkdtemp(writable.data());
    if (created == nullptr) {
      throw std::runtime_error("mkdtemp failed");
    }
    root_ = created;
    cache_ = root_ / "cache";
    log_ = root_ / "compiler.log";
    wrapper_ = root_ / "compiler-wrapper.sh";
    write_file(wrapper_,
               "#!/bin/sh\n"
               "for arg in \"$@\"; do\n"
               "  if [ \"$arg\" = \"--request\" ]; then\n"
               "    printf 'compile\\n' >> \"$FLAGDNN_TEST_COMPILER_LOG\"\n"
               "  fi\n"
               "done\n"
               "exec \"$FLAGDNN_TEST_REAL_PYTHON\" \"$@\"\n");
    std::filesystem::permissions(wrapper_,
                                 std::filesystem::perms::owner_read |
                                     std::filesystem::perms::owner_write |
                                     std::filesystem::perms::owner_exec,
                                 std::filesystem::perm_options::replace);
    if (setenv("FLAGDNN_TEST_COMPILER_LOG", log_.c_str(), 1) != 0 ||
        setenv("FLAGDNN_TEST_REAL_PYTHON", python, 1) != 0) {
      throw std::runtime_error("cannot configure compiler wrapper");
    }
  }

  ~TemporaryRuntime() {
    std::error_code ignored;
    std::filesystem::remove_all(root_, ignored);
  }

  TemporaryRuntime(const TemporaryRuntime &) = delete;
  TemporaryRuntime &operator=(const TemporaryRuntime &) = delete;

  [[nodiscard]] const std::filesystem::path &cache() const noexcept {
    return cache_;
  }
  [[nodiscard]] const std::filesystem::path &wrapper() const noexcept {
    return wrapper_;
  }

  [[nodiscard]] std::size_t compile_count() const {
    std::ifstream input(log_);
    std::size_t result = 0;
    std::string line;
    while (std::getline(input, line)) {
      if (line == "compile") {
        ++result;
      }
    }
    return result;
  }

  [[nodiscard]] std::filesystem::path manifest() const {
    std::filesystem::path result;
    std::size_t count = 0;
    if (!std::filesystem::exists(cache_)) {
      throw std::runtime_error("runtime cache does not exist");
    }
    for (const auto &entry :
         std::filesystem::recursive_directory_iterator(cache_)) {
      if (entry.is_regular_file() &&
          entry.path().filename() == "manifest.json") {
        result = entry.path();
        ++count;
      }
    }
    if (count != 1) {
      throw std::runtime_error(
          "runtime cache must contain exactly one manifest");
    }
    return result;
  }

  void validate_layout(const std::filesystem::path &manifest) const {
    const std::filesystem::path identity = manifest.parent_path();
    const std::filesystem::path graph = identity.parent_path();
    const std::filesystem::path engine = graph.parent_path();
    const std::filesystem::path target = engine.parent_path();
    const std::filesystem::path backend = target.parent_path();
    expect(identity.filename() != "" &&
               is_sha256_name(identity.filename().string()),
           "cache identity directory is not a SHA-256");
    expect(is_sha256_name(graph.filename().string()),
           "cache graph directory is not a SHA-256");
    expect(engine.filename() == "libtriton_jit",
           "cache engine directory is wrong");
    expect(target.filename() == "corex_71", "cache target directory is wrong");
    expect(backend.filename() == "iluvatar",
           "cache backend directory is wrong");
    expect(backend.parent_path() == cache_,
           "cache layout has an unexpected prefix");
  }

private:
  std::filesystem::path root_;
  std::filesystem::path cache_;
  std::filesystem::path log_;
  std::filesystem::path wrapper_;
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
  [[nodiscard]] flagdnnStream_t opaque() const noexcept {
    return reinterpret_cast<flagdnnStream_t>(value_);
  }

private:
  CUstream value_ = nullptr;
};

class DeviceBuffer final {
public:
  explicit DeviceBuffer(std::size_t bytes) : bytes_(bytes) {
    if (bytes_ != 0) {
      check_driver(cuMemAlloc(&value_, bytes_), "cuMemAlloc");
    }
  }
  ~DeviceBuffer() {
    if (value_ != 0) {
      (void)cuMemFree(value_);
    }
  }
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;

  [[nodiscard]] CUdeviceptr get() const noexcept { return value_; }
  [[nodiscard]] void *opaque(std::size_t offset = 0) const noexcept {
    return reinterpret_cast<void *>(static_cast<std::uintptr_t>(value_) +
                                    offset);
  }

private:
  std::size_t bytes_ = 0;
  CUdeviceptr value_ = 0;
};

flagdnn::Executable build_add(const flagdnn::Handle &handle) {
  const std::array<std::int64_t, 1> dimensions = {1024};
  const std::array<std::int64_t, 1> strides = {1};
  flagdnn::TensorDescriptor left(1, FLAGDNN_DATA_FLOAT32, dimensions, strides);
  flagdnn::TensorDescriptor right(2, FLAGDNN_DATA_FLOAT32, dimensions, strides);
  flagdnn::TensorDescriptor output(3, FLAGDNN_DATA_FLOAT32, dimensions,
                                   strides);
  flagdnn::Graph graph;
  graph.pointwise(left, right, FLAGDNN_POINTWISE_ADD, output, 1.0);
  graph.finalize();
  return flagdnn::Executable(handle, graph);
}

void expect_compilation_failure(const flagdnn::Handle &handle) {
  try {
    (void)build_add(handle);
  } catch (const flagdnn::Error &error) {
    expect(error.status() == FLAGDNN_STATUS_COMPILATION_FAILED,
           "invalid offline cache returned the wrong status");
    return;
  }
  throw std::runtime_error("invalid offline cache unexpectedly built");
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 4) {
      std::cerr << "usage: flagdnn_test_iluvatar_runtime "
                   "PLUGIN PYTHON COMPILER_ENTRY\n";
      return 2;
    }
    const std::filesystem::path plugin = std::filesystem::canonical(argv[1]);
    if (setenv("FLAGDNN_BACKEND_PATH", plugin.parent_path().c_str(), 1) != 0 ||
        setenv("FLAGDNN_EXECUTION_ENGINE", "libtriton_jit", 1) != 0) {
      throw std::runtime_error("cannot configure runtime test environment");
    }

    CurrentPrimaryContext current_context;
    TemporaryRuntime runtime(argv[2]);
    flagdnn::Handle handle("iluvatar", 0);
    handle.set_compiler(runtime.wrapper().string(), argv[3],
                        runtime.cache().string());

    auto initial = build_add(handle);
    expect(runtime.compile_count() == 1,
           "initial cache miss did not compile exactly once");
    std::filesystem::path manifest = runtime.manifest();
    runtime.validate_layout(manifest);

    auto cache_hit = build_add(handle);
    (void)cache_hit;
    expect(runtime.compile_count() == 1,
           "valid cache hit invoked the provider compiler");

    write_file(manifest, "{}\n");
    auto rebuilt = build_add(handle);
    (void)rebuilt;
    expect(runtime.compile_count() == 2,
           "corrupt cache was not quarantined and recompiled exactly once");
    manifest = runtime.manifest();
    runtime.validate_layout(manifest);

    handle.set_compiler("/definitely/missing/flagdnn-python", argv[3],
                        runtime.cache().string());
    auto offline_hit = build_add(handle);
    (void)offline_hit;
    expect(runtime.compile_count() == 2,
           "offline valid-cache build unexpectedly invoked the compiler");

    write_file(manifest, "{}\n");
    expect_compilation_failure(handle);
    expect(runtime.compile_count() == 2,
           "offline invalid-cache failure invoked the compiler");

    handle.set_compiler(runtime.wrapper().string(), argv[3],
                        runtime.cache().string());
    auto executable = build_add(handle);
    expect(runtime.compile_count() == 3,
           "restoring the compiler did not rebuild the quarantined cache");
    manifest = runtime.manifest();
    runtime.validate_layout(manifest);

    std::array<float, 1024> left{};
    std::array<float, 1024> right{};
    std::array<float, 1024> output{};
    for (std::size_t index = 0; index < left.size(); ++index) {
      left[index] = static_cast<float>(index % 23) - 5.0F;
      right[index] = static_cast<float>(index % 11) * 0.25F;
    }
    Stream stream;
    DeviceBuffer device_left(sizeof(left));
    DeviceBuffer device_right(sizeof(right));
    DeviceBuffer device_output(sizeof(output));
    check_driver(cuMemcpyHtoDAsync(device_left.get(), left.data(), sizeof(left),
                                   stream.get()),
                 "cuMemcpyHtoDAsync(left)");
    check_driver(cuMemcpyHtoDAsync(device_right.get(), right.data(),
                                   sizeof(right), stream.get()),
                 "cuMemcpyHtoDAsync(right)");
    const std::array<flagdnnBinding_t, 3> bindings = {{
        {1, device_left.opaque()},
        {2, device_right.opaque()},
        {3, device_output.opaque()},
    }};
    for (int repetition = 0; repetition < 3; ++repetition) {
      executable.execute(bindings, nullptr, 0, stream.opaque());
    }
    check_driver(cuMemcpyDtoHAsync(output.data(), device_output.get(),
                                   sizeof(output), stream.get()),
                 "cuMemcpyDtoHAsync(output)");
    check_driver(cuStreamSynchronize(stream.get()),
                 "cuStreamSynchronize(caller stream)");
    for (std::size_t index = 0; index < output.size(); ++index) {
      const float expected = left[index] + right[index];
      if (std::fabs(output[index] - expected) > 1.0e-6F) {
        throw std::runtime_error(
            "caller-stream Add ordering failed at element " +
            std::to_string(index));
      }
    }

    const std::array<flagdnnBinding_t, 2> missing = {{
        {1, device_left.opaque()},
        {3, device_output.opaque()},
    }};
    expect_status("missing binding", FLAGDNN_STATUS_INVALID_VALUE,
                  flagdnnExecuteAsync(executable.get(), missing.data(),
                                      missing.size(), nullptr, 0,
                                      stream.opaque()));
    const std::array<flagdnnBinding_t, 3> duplicate = {{
        {1, device_left.opaque()},
        {1, device_right.opaque()},
        {3, device_output.opaque()},
    }};
    expect_status("duplicate binding", FLAGDNN_STATUS_INVALID_VALUE,
                  flagdnnExecuteAsync(executable.get(), duplicate.data(),
                                      duplicate.size(), nullptr, 0,
                                      stream.opaque()));
    const std::array<flagdnnBinding_t, 3> null_pointer = {{
        {1, device_left.opaque()},
        {2, nullptr},
        {3, device_output.opaque()},
    }};
    expect_status("null binding pointer", FLAGDNN_STATUS_INVALID_VALUE,
                  flagdnnExecuteAsync(executable.get(), null_pointer.data(),
                                      null_pointer.size(), nullptr, 0,
                                      stream.opaque()));
    expect_status("null executable", FLAGDNN_STATUS_INVALID_VALUE,
                  flagdnnExecuteAsync(nullptr, bindings.data(), bindings.size(),
                                      nullptr, 0, stream.opaque()));

    const std::string with_workspace = replace_once(
        read_file(manifest),
        "\"workspace\": {\n    \"alignment\": 1,\n    \"size\": 0\n  }",
        "\"workspace\": {\n    \"alignment\": 256,\n    \"size\": 256\n  }");
    write_file(manifest, with_workspace);
    auto workspace_executable = build_add(handle);
    expect(workspace_executable.workspace_size() == 256,
           "workspace-mutated artifact did not retain its contract");
    DeviceBuffer workspace(257);
    expect_status("undersized workspace", FLAGDNN_STATUS_INVALID_VALUE,
                  flagdnnExecuteAsync(workspace_executable.get(),
                                      bindings.data(), bindings.size(),
                                      workspace.opaque(), 255,
                                      stream.opaque()));
    expect_status("misaligned workspace", FLAGDNN_STATUS_INVALID_VALUE,
                  flagdnnExecuteAsync(workspace_executable.get(),
                                      bindings.data(), bindings.size(),
                                      workspace.opaque(1), 256,
                                      stream.opaque()));
    expect_status("valid workspace", FLAGDNN_STATUS_SUCCESS,
                  flagdnnExecuteAsync(workspace_executable.get(),
                                      bindings.data(), bindings.size(),
                                      workspace.opaque(), 256,
                                      stream.opaque()));
    check_driver(cuStreamSynchronize(stream.get()),
                 "cuStreamSynchronize(workspace contract)");

    expect(runtime.compile_count() == 3,
           "runtime edge cases unexpectedly invoked the compiler");
    std::cout << "PASS Iluvatar runtime/cache/caller-stream contract; "
              << "compile_count=" << runtime.compile_count() << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
