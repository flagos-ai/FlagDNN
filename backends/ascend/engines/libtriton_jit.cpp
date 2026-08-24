/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "backends/ascend/engines/libtriton_jit.hpp"

#include "backends/ascend/engines/python_stdout_containment.hpp"

#include "backends/ascend/error.hpp"
#include "src/runtime/json.hpp"
#include "src/runtime/sha256.hpp"

#include <Python.h>
#include <runtime/runtime/rt.h>
#include <triton_jit/kernel_metadata.h>
#include <triton_jit/triton_jit_function.h>
#include <triton_jit/triton_kernel.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <exception>
#include <fcntl.h>
#include <filesystem>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>
#include <variant>
#include <vector>

/* libtriton_jit explicitly instantiates and exports its NPU function class.
 * Suppress this plugin TU's implicit instantiation of the inline definition;
 * otherwise NpuBackend::ensure_context() is copied into the plugin and creates
 * forbidden direct aclrtSetDevice/aclrtCreateContext references even though the
 * outer ContextGuard guarantees that branch is never taken. */
namespace triton_jit {
extern template class TritonKernelImpl<NpuBackend>;
extern template class TritonJITFunctionImpl<NpuBackend>;
}  // namespace triton_jit

namespace flagdnn::ascend {
namespace {

namespace fs = std::filesystem;
using JsonValue = flagdnn::native::json::Value;
using LtjFunction = triton_jit::TritonJITFunction;

#ifndef FLAGDNN_ASCEND_STANDALONE_PATH
#define FLAGDNN_ASCEND_STANDALONE_PATH ""
#endif

#ifndef FLAGDNN_ASCEND_STANDALONE_SHA256
#define FLAGDNN_ASCEND_STANDALONE_SHA256 ""
#endif

constexpr std::size_t kMaximumMetadataBytes = 1U << 20U;
constexpr std::size_t kMaximumCacheFiles = 4096;
constexpr std::uintmax_t kMaximumCacheBytes = 1ULL << 30U;
constexpr std::size_t kGraphWorkspaceAlignment = 256;
constexpr std::size_t kMaximumKernelWorkspacePerBlock = 1U << 20U;
constexpr std::size_t kMaximumKernelWorkspaceBytes = 1U << 30U;
constexpr std::size_t kNpuSystemArgumentBytes = 3U * sizeof(void*);
constexpr std::size_t kMaximumPreparedArgumentBytes =
    kNpuSystemArgumentBytes +
    FLAGDNN_BACKEND_MAX_KERNEL_ARGUMENTS * sizeof(std::uint64_t) +
    3U * sizeof(std::int32_t);

using LtjRawLaunchMethod = void (LtjFunction::*)(
    triton_jit::NpuBackend::StreamType,
    unsigned int,
    unsigned int,
    unsigned int,
    unsigned int,
    unsigned int,
    std::string,
    void**,
    std::size_t) const;

[[nodiscard]] LtjRawLaunchMethod exported_raw_launch_method() noexcept {
  /* The installed LTJ explicitly instantiates and exports this member.  Keep
   * the typed pointer volatile so this TU emits an indirect call to that
   * exported instantiation instead of inlining NpuBackend::ensure_context().
   * Inlining the header implementation would give this caller-owned plugin
   * forbidden direct aclrtSetDevice/aclrtCreateContext references. */
  static const volatile LtjRawLaunchMethod method =
      &LtjFunction::launch_with_raw_args;
  return method;
}

void launch_with_exported_raw_api(LtjFunction& function,
                                  aclrtStream stream,
                                  const LtjNpuRawCandidate& candidate,
                                  void** arguments,
                                  std::size_t argument_count) {
  const LtjRawLaunchMethod method = exported_raw_launch_method();
  (function.*method)(stream,
                     candidate.grid[0],
                     candidate.grid[1],
                     candidate.grid[2],
                     candidate.num_warps,
                     candidate.num_stages,
                     candidate.full_signature,
                     arguments,
                     argument_count);
}

using LtjLoadKernelMethod = void* (*)(const std::string&,
                                     const std::string&);

[[nodiscard]] LtjLoadKernelMethod exported_load_kernel_method() {
  /* load_kernel is inline in LTJ's public header but is also exported by the
   * pinned DSO. Resolve that exported copy explicitly: compiling the inline
   * body into this plugin would duplicate LTJ's module registry and its device
   * lifecycle fallback. */
  static const LtjLoadKernelMethod method = [] {
    constexpr const char* symbol_name =
        "_ZN10triton_jit10NpuBackend11load_kernelERKNSt7__cxx1112basic_"
        "stringIcSt11char_traitsIcESaIcEEES8_";
    (void)::dlerror();
    void* symbol = ::dlsym(RTLD_DEFAULT, symbol_name);
    const char* error = ::dlerror();
    if (symbol == nullptr || error != nullptr) {
      throw AscendError(
          FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
          "pinned libtriton_jit does not export its NPU kernel loader");
    }
    static_assert(sizeof(LtjLoadKernelMethod) == sizeof(symbol));
    LtjLoadKernelMethod result = nullptr;
    std::memcpy(&result, &symbol, sizeof(result));
    return result;
  }();
  return method;
}

struct ContainedRawLaunch {
  LtjFunction* function = nullptr;
  aclrtStream stream = nullptr;
  const LtjNpuRawCandidate* candidate = nullptr;
  void** arguments = nullptr;
  std::size_t argument_count = 0;
  bool* raw_started = nullptr;
};

void launch_create_raw_with_contained_stdout(void* opaque) {
  auto* launch = static_cast<ContainedRawLaunch*>(opaque);
  if (launch == nullptr || launch->function == nullptr ||
      launch->candidate == nullptr || launch->raw_started == nullptr) {
    throw std::invalid_argument("invalid contained Ascend raw launch");
  }
  *launch->raw_started = true;
  launch_with_exported_raw_api(*launch->function,
                               launch->stream,
                               *launch->candidate,
                               launch->arguments,
                               launch->argument_count);
}

[[noreturn]] void compilation_failure(std::string message) {
  throw AscendError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                    std::move(message));
}

struct PyObjectDeleter {
  void operator()(PyObject* object) const noexcept { Py_XDECREF(object); }
};

using OwnedPyObject = std::unique_ptr<PyObject, PyObjectDeleter>;

[[nodiscard]] std::string consume_python_error() {
  if (PyErr_Occurred() == nullptr) {
    return "unknown Python error";
  }
  PyObject* type_raw = nullptr;
  PyObject* value_raw = nullptr;
  PyObject* traceback_raw = nullptr;
  PyErr_Fetch(&type_raw, &value_raw, &traceback_raw);
  PyErr_NormalizeException(&type_raw, &value_raw, &traceback_raw);
  OwnedPyObject type(type_raw);
  OwnedPyObject value(value_raw);
  OwnedPyObject traceback(traceback_raw);

  PyObject* source = value != nullptr ? value.get() : type.get();
  if (source == nullptr) {
    PyErr_Clear();
    return "unknown Python exception";
  }
  OwnedPyObject rendered(PyObject_Str(source));
  if (rendered == nullptr) {
    PyErr_Clear();
    return "unprintable Python exception";
  }
  const char* text = PyUnicode_AsUTF8(rendered.get());
  if (text == nullptr) {
    PyErr_Clear();
    return "non-UTF-8 Python exception";
  }
  std::string result(text);
  PyErr_Clear();
  return result;
}

[[noreturn]] void python_compilation_failure(std::string_view operation) {
  compilation_failure(std::string(operation) + ": " + consume_python_error());
}

[[nodiscard]] bool same_time(const timespec& left,
                             const timespec& right) noexcept {
  return left.tv_sec == right.tv_sec && left.tv_nsec == right.tv_nsec;
}

class FileDescriptor {
 public:
  explicit FileDescriptor(int value) noexcept : value_(value) {}
  ~FileDescriptor() {
    if (value_ >= 0) {
      (void)::close(value_);
    }
  }

  FileDescriptor(const FileDescriptor&) = delete;
  FileDescriptor& operator=(const FileDescriptor&) = delete;

  [[nodiscard]] int get() const noexcept { return value_; }

 private:
  int value_ = -1;
};

struct RegularFile {
  std::string contents;
  std::string sha256;
  std::uint64_t device = 0;
  std::uint64_t inode = 0;
  std::uintmax_t size = 0;
  timespec modification_time{};
  timespec change_time{};
};

[[nodiscard]] RegularFile read_regular_file(const fs::path& path,
                                            std::size_t maximum_size,
                                            bool retain_contents = true) {
  const int descriptor =
      ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
  if (descriptor < 0) {
    compilation_failure("cannot open Ascend cache file without following links: " +
                        path.string());
  }
  FileDescriptor owner(descriptor);
  struct stat before {};
  if (::fstat(owner.get(), &before) != 0 || !S_ISREG(before.st_mode) ||
      before.st_nlink != 1 || before.st_size < 0 ||
      static_cast<std::uintmax_t>(before.st_size) > maximum_size) {
    compilation_failure("Ascend cache file is not a bounded private regular file: " +
                        path.string());
  }

  RegularFile result;
  result.contents.resize(static_cast<std::size_t>(before.st_size));
  std::size_t offset = 0;
  while (offset < result.contents.size()) {
    const ssize_t count = ::read(owner.get(), result.contents.data() + offset,
                                 result.contents.size() - offset);
    if (count < 0 && errno == EINTR) {
      continue;
    }
    if (count <= 0) {
      compilation_failure("cannot read complete Ascend cache file: " +
                          path.string());
    }
    offset += static_cast<std::size_t>(count);
  }

  struct stat after {};
  if (::fstat(owner.get(), &after) != 0 || before.st_dev != after.st_dev ||
      before.st_ino != after.st_ino || before.st_size != after.st_size ||
      !same_time(before.st_mtim, after.st_mtim) ||
      !same_time(before.st_ctim, after.st_ctim)) {
    compilation_failure("Ascend cache file changed while it was inspected: " +
                        path.string());
  }
  result.sha256 = flagdnn::native::sha256(result.contents);
  if (!retain_contents) {
    std::string{}.swap(result.contents);
  }
  result.device = static_cast<std::uint64_t>(after.st_dev);
  result.inode = static_cast<std::uint64_t>(after.st_ino);
  result.size = static_cast<std::uintmax_t>(after.st_size);
  result.modification_time = after.st_mtim;
  result.change_time = after.st_ctim;
  return result;
}

[[nodiscard]] bool same_file(const RegularFile& left,
                             const RegularFile& right) noexcept {
  return left.device == right.device && left.inode == right.inode &&
         left.size == right.size && left.sha256 == right.sha256 &&
         same_time(left.modification_time, right.modification_time) &&
         same_time(left.change_time, right.change_time);
}

[[nodiscard]] bool same_file_contents(const RegularFile& left,
                                      const RegularFile& right) noexcept {
  return left.size == right.size && left.sha256 == right.sha256;
}

[[nodiscard]] bool path_is_within(const fs::path& child,
                                  const fs::path& parent) noexcept {
  auto child_iterator = child.begin();
  for (auto parent_iterator = parent.begin();
       parent_iterator != parent.end();
       ++parent_iterator, ++child_iterator) {
    if (child_iterator == child.end() ||
        *child_iterator != *parent_iterator) {
      return false;
    }
  }
  return true;
}

using CacheSnapshot = std::map<fs::path, RegularFile>;

[[nodiscard]] fs::path validate_private_cache_root(const fs::path& configured) {
  if (configured.empty() || !configured.is_absolute()) {
    compilation_failure("Ascend Triton cache root must be absolute");
  }
  std::error_code error;
  const fs::path canonical = fs::canonical(configured, error);
  if (error || canonical != configured.lexically_normal()) {
    compilation_failure("Ascend Triton cache root must exist and be canonical");
  }
  struct stat status {};
  if (::lstat(canonical.c_str(), &status) != 0 ||
      !S_ISDIR(status.st_mode) || S_ISLNK(status.st_mode) ||
      status.st_uid != ::geteuid() || (status.st_mode & 0077) != 0) {
    compilation_failure(
        "Ascend Triton cache root must be a private owner-only directory");
  }
  const char* environment = std::getenv("TRITON_CACHE_DIR");
  if (environment == nullptr || fs::path(environment) != canonical) {
    compilation_failure(
        "TRITON_CACHE_DIR differs from the process-bound Ascend cache root");
  }
  return canonical;
}

[[nodiscard]] CacheSnapshot scan_cache(const fs::path& root,
                                       std::string_view entry_point) {
  const std::string filename = std::string(entry_point) + ".json";
  CacheSnapshot result;
  std::size_t file_count = 0;
  std::uintmax_t total_bytes = 0;
  std::error_code error;
  fs::recursive_directory_iterator iterator(
      root, fs::directory_options::none, error);
  const fs::recursive_directory_iterator end;
  if (error) {
    compilation_failure("cannot enter the private Ascend cache root");
  }
  while (iterator != end) {
    const fs::directory_entry entry = *iterator;
    const fs::file_status status = entry.symlink_status(error);
    if (error) {
      compilation_failure("cannot stat an Ascend cache entry");
    }
    if (fs::is_symlink(status) ||
        (!fs::is_directory(status) && !fs::is_regular_file(status))) {
      compilation_failure(
          "private Ascend cache contains a link or non-regular entry");
    }
    if (fs::is_regular_file(status)) {
      if (++file_count > kMaximumCacheFiles ||
          total_bytes > kMaximumCacheBytes) {
        compilation_failure("private Ascend cache exceeds its scan budget");
      }
      const std::uintmax_t remaining = kMaximumCacheBytes - total_bytes;
      const bool is_target_metadata = entry.path().filename() == filename;
      const std::uintmax_t per_file_limit =
          is_target_metadata
              ? std::min<std::uintmax_t>(remaining, kMaximumMetadataBytes)
              : remaining;
      RegularFile file = read_regular_file(
          entry.path(), static_cast<std::size_t>(per_file_limit),
          is_target_metadata);
      if (file.size > remaining) {
        compilation_failure("private Ascend cache exceeds its scan budget");
      }
      total_bytes += file.size;
      const auto [unused, inserted] = result.emplace(
          entry.path().lexically_normal(), std::move(file));
      (void)unused;
      if (!inserted) {
        compilation_failure("private Ascend cache contains a duplicate path");
      }
    }
    iterator.increment(error);
    if (error) {
      compilation_failure("cannot complete the private Ascend cache scan");
    }
  }
  return result;
}

[[nodiscard]] bool same_cache_snapshot(const CacheSnapshot& left,
                                       const CacheSnapshot& right) noexcept {
  if (left.size() != right.size()) {
    return false;
  }
  for (const auto& [path, state] : left) {
    const auto found = right.find(path);
    if (found == right.end() || !same_file(state, found->second)) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] const JsonValue* member(const JsonValue::Object& object,
                                      std::string_view name) noexcept {
  const auto found = object.find(name);
  return found == object.end() ? nullptr : &found->second;
}

[[nodiscard]] RawArgumentType metadata_argument_type(
    const JsonValue& value) {
  const auto& object = value.as_object();
  const JsonValue* encoded = member(object, "type");
  if (encoded == nullptr) {
    compilation_failure("NPU metadata argument has no type");
  }
  const std::string& type = encoded->as_string();
  if (type == "ptr" || type == "pointer") {
    return RawArgumentType::kPointer;
  }
  if (type == "i32" || type == "u32") {
    return RawArgumentType::kI32;
  }
  if (type == "i64" || type == "u64") {
    return RawArgumentType::kI64;
  }
  if (type == "fp32" || type == "f32") {
    return RawArgumentType::kF32;
  }
  if (type == "fp64" || type == "f64") {
    return RawArgumentType::kF64;
  }
  compilation_failure("NPU metadata contains an unsupported runtime type");
}

[[nodiscard]] triton_jit::NpuArgType ltj_argument_type(
    RawArgumentType type) {
  switch (type) {
    case RawArgumentType::kPointer:
      return triton_jit::NpuArgType::POINTER;
    case RawArgumentType::kI32:
      return triton_jit::NpuArgType::I32;
    case RawArgumentType::kI64:
      return triton_jit::NpuArgType::I64;
    case RawArgumentType::kF32:
      return triton_jit::NpuArgType::F32;
    case RawArgumentType::kF64:
      return triton_jit::NpuArgType::F64;
  }
  compilation_failure("unknown Ascend raw argument type");
}

[[nodiscard]] std::size_t validate_metadata(
    const fs::path& path,
    const RegularFile& file,
    const fs::path& cache_root,
    const CacheSnapshot& snapshot,
    const LtjNpuRawCandidate& candidate,
    std::optional<unsigned int> observed_shared_memory) {
  try {
    const JsonValue document = flagdnn::native::json::parse(file.contents);
    const auto& object = document.as_object();
    const JsonValue* layout = member(object, "arg_layout");
    if (layout == nullptr) {
      compilation_failure("NPU metadata is missing its runtime argument layout");
    }

    std::uint64_t workspace_size = 0;
    if (const JsonValue* workspace = member(object, "workspace_size")) {
      const std::int64_t encoded = workspace->as_int();
      if (encoded < 0) {
        compilation_failure("NPU metadata workspace size is negative");
      }
      workspace_size = static_cast<std::uint64_t>(encoded);
    }
    if (workspace_size != 0 &&
        candidate.entry_point != "convolution_fprop_persistent_kernel" &&
        candidate.entry_point != "matmul_strided_kernel") {
      compilation_failure(
          "only the Ascend convolution and MatMul kernels may use compiler "
          "workspace "
          "(entry_point=" +
          candidate.entry_point + ", bytes=" +
          std::to_string(workspace_size) + ")");
    }
    if (workspace_size > kMaximumKernelWorkspacePerBlock ||
        workspace_size > std::numeric_limits<std::size_t>::max()) {
      compilation_failure("NPU metadata workspace size exceeds its limit");
    }

    unsigned int shared_memory = 0;
    if (const JsonValue* shared = member(object, "shared")) {
      const std::int64_t encoded = shared->as_int();
      if (encoded < 0 ||
          static_cast<std::uint64_t>(encoded) >
              std::numeric_limits<unsigned int>::max()) {
        compilation_failure("NPU metadata shared memory is out of range");
      }
      shared_memory = static_cast<unsigned int>(encoded);
    }
    if (observed_shared_memory.has_value() &&
        shared_memory != *observed_shared_memory) {
      compilation_failure(
          "NPU metadata shared memory differs from launch-enter metadata");
    }

    const auto& arguments = layout->as_array();
    if (arguments.size() != candidate.argument_types.size()) {
      compilation_failure("NPU metadata runtime ABI count differs");
    }
    for (std::size_t index = 0; index < arguments.size(); ++index) {
      if (metadata_argument_type(arguments[index]) !=
          candidate.argument_types[index]) {
        compilation_failure("NPU metadata runtime ABI type differs");
      }
    }

    const fs::path binary = path.parent_path() /
                            (candidate.entry_point + std::string(".npubin"));
    const auto binary_state = snapshot.find(binary.lexically_normal());
    if (binary_state == snapshot.end() || binary_state->second.size == 0 ||
        binary_state->second.sha256.empty()) {
      compilation_failure(
          "NPU metadata has no matching no-follow attested npubin");
    }

    const triton_jit::NpuKernelMetadata normalized =
        triton_jit::load_npu_metadata(path.parent_path().string(),
                                      candidate.entry_point);
    if (normalized.workspace_size != workspace_size ||
        normalized.shared != shared_memory ||
        normalized.arg_layout.size() != candidate.argument_types.size()) {
      compilation_failure("public NPU metadata loader disagrees with cache proof");
    }
    for (std::size_t index = 0; index < normalized.arg_layout.size(); ++index) {
      if (normalized.arg_layout[index].type !=
          ltj_argument_type(candidate.argument_types[index])) {
        compilation_failure("public NPU metadata ABI disagrees with artifact");
      }
    }
    const CacheSnapshot unchanged = scan_cache(cache_root, candidate.entry_point);
    if (!same_cache_snapshot(snapshot, unchanged)) {
      compilation_failure(
          "public NPU metadata loading changed the private cache tree");
    }
    return static_cast<std::size_t>(workspace_size);
  } catch (const AscendError&) {
    throw;
  } catch (const std::exception& error) {
    compilation_failure("cannot validate NPU metadata " + path.string() +
                        ": " + error.what());
  }
}

[[nodiscard]] std::string candidate_key(const fs::path& cache_root,
                                        const EngineBuildContext& context,
                                        const LtjNpuRawCandidate& candidate) {
  std::ostringstream output;
  output << cache_root.string() << '\n' << context.configuration_identity << '\n'
         << candidate.source.string() << '\n' << candidate.source_sha256 << '\n'
         << candidate.entry_point << '\n' << candidate.full_signature << '\n'
         << context.device_ordinal << '\n' << candidate.num_warps << '\n'
         << candidate.num_stages;
  return output.str();
}

struct CacheAttestation {
  fs::path metadata_path;
  CacheSnapshot published_files;
};

[[nodiscard]] std::map<std::string, CacheAttestation>& cache_attestations() {
  static std::map<std::string, CacheAttestation> attestations;
  return attestations;
}

[[nodiscard]] const RegularFile& select_metadata(
    const CacheSnapshot& before,
    const CacheSnapshot& after,
    const std::string& key,
    std::string_view entry_point,
    fs::path* selected_path) {
  auto& attestations = cache_attestations();
  const auto known = attestations.find(key);
  if (known != attestations.end()) {
    if (!same_cache_snapshot(before, after)) {
      compilation_failure(
          "prewarmed raw launch changed the private NPU cache tree");
    }
    for (const auto& [path, state] : known->second.published_files) {
      const auto current = before.find(path);
      if (current == before.end() ||
          !same_file_contents(state, current->second)) {
        compilation_failure(
            "an attested NPU cache file changed between raw launches");
      }
    }
    const auto current = after.find(known->second.metadata_path);
    if (current == after.end()) {
      compilation_failure("prewarmed NPU metadata disappeared");
    }
    *selected_path = current->first;
    return current->second;
  }

  CacheSnapshot published_files;
  for (const auto& [path, state] : before) {
    const auto current = after.find(path);
    if (current == after.end()) {
      compilation_failure(
          "first raw compilation removed existing NPU cache file: " +
          path.string());
    }
    if (!same_file(state, current->second)) {
      if (!same_file_contents(state, current->second)) {
        compilation_failure(
            "first raw compilation changed existing NPU cache contents: " +
            path.string());
      }
      RegularFile refreshed = current->second;
      std::string{}.swap(refreshed.contents);
      published_files.emplace(path, std::move(refreshed));
    }
  }

  const std::string metadata_filename = std::string(entry_point) + ".json";
  std::size_t target_metadata_count = 0;
  for (const auto& [path, state] : after) {
    const auto previous = before.find(path);
    if (previous == before.end()) {
      RegularFile published = state;
      std::string{}.swap(published.contents);
      published_files.emplace(path, std::move(published));
    }
    if (published_files.find(path) != published_files.end() &&
        path.filename() == metadata_filename) {
      ++target_metadata_count;
      *selected_path = path;
    }
  }
  if (target_metadata_count != 1) {
    compilation_failure(
        "first raw compilation must publish or identity-refresh exactly one "
        "target entry metadata file");
  }
  const fs::path binary = selected_path->parent_path() /
                          (std::string(entry_point) + ".npubin");
  const auto binary_state = after.find(binary.lexically_normal());
  if (binary_state == after.end() || binary_state->second.size == 0 ||
      binary_state->second.sha256.empty()) {
    compilation_failure(
        "first raw compilation has no matching attested npubin");
  }
  RegularFile published_binary = binary_state->second;
  std::string{}.swap(published_binary.contents);
  published_files.insert_or_assign(binary.lexically_normal(),
                                   std::move(published_binary));
  const auto selected = after.find(*selected_path);
  if (selected == after.end()) {
    compilation_failure("selected NPU metadata is absent from the cache snapshot");
  }
  const auto [attestation, inserted] = attestations.emplace(
      key, CacheAttestation{*selected_path, std::move(published_files)});
  (void)attestation;
  if (!inserted) {
    compilation_failure("NPU cache attestation key was inserted concurrently");
  }
  return selected->second;
}

void validate_source(const LtjNpuRawCandidate& candidate) {
  std::error_code error;
  const fs::path canonical = fs::canonical(candidate.source, error);
  const fs::file_status status = fs::symlink_status(candidate.source, error);
  if (error || canonical != candidate.source.lexically_normal() ||
      !fs::is_regular_file(status) || fs::is_symlink(status)) {
    compilation_failure("materialized Ascend source is not canonical and regular");
  }
  const RegularFile source =
      read_regular_file(candidate.source, kMaximumMetadataBytes);
  if (source.sha256 != candidate.source_sha256) {
    compilation_failure("materialized Ascend source hash changed before JIT");
  }
}

[[nodiscard]] fs::path validate_standalone_compiler() {
  const fs::path configured(FLAGDNN_ASCEND_STANDALONE_PATH);
  if (configured.empty() || !configured.is_absolute()) {
    compilation_failure("configured Ascend standalone compiler is not absolute");
  }
  std::error_code error;
  const fs::path canonical = fs::canonical(configured, error);
  const fs::file_status status = fs::symlink_status(configured, error);
  if (error || canonical != configured.lexically_normal() ||
      !fs::is_regular_file(status) || fs::is_symlink(status)) {
    compilation_failure(
        "configured Ascend standalone compiler is not canonical and regular");
  }
  const RegularFile source =
      read_regular_file(canonical, kMaximumMetadataBytes);
  if (std::string_view(FLAGDNN_ASCEND_STANDALONE_SHA256).empty() ||
      source.sha256 != FLAGDNN_ASCEND_STANDALONE_SHA256) {
    compilation_failure("configured Ascend standalone compiler hash changed");
  }
  return canonical;
}

struct ContainedStandaloneCompile {
  const LtjNpuRawCandidate* candidate = nullptr;
  fs::path compiler;
  std::int32_t device_ordinal = 0;
  std::string cache_directory;
};

void compile_with_contained_stdout(void* opaque) {
  auto* request = static_cast<ContainedStandaloneCompile*>(opaque);
  if (request == nullptr || request->candidate == nullptr ||
      request->compiler.empty()) {
    throw std::invalid_argument("invalid contained Ascend compilation");
  }
  const LtjNpuRawCandidate& candidate = *request->candidate;

  OwnedPyObject importlib_util(PyImport_ImportModule("importlib.util"));
  if (importlib_util == nullptr) {
    python_compilation_failure("cannot import embedded importlib.util");
  }
  OwnedPyObject spec_from_file_location(
      PyObject_GetAttrString(importlib_util.get(), "spec_from_file_location"));
  OwnedPyObject module_from_spec(
      PyObject_GetAttrString(importlib_util.get(), "module_from_spec"));
  if (spec_from_file_location == nullptr || module_from_spec == nullptr) {
    python_compilation_failure(
        "embedded importlib.util has an incomplete module-loading API");
  }
  if (PyCallable_Check(spec_from_file_location.get()) == 0 ||
      PyCallable_Check(module_from_spec.get()) == 0) {
    compilation_failure(
        "embedded importlib.util module-loading API is not callable");
  }

  OwnedPyObject module_name(
      PyUnicode_FromString("_flagdnn_ascend_standalone_compile"));
  const std::string compiler_string = request->compiler.string();
  OwnedPyObject compiler_path(PyUnicode_DecodeFSDefaultAndSize(
      compiler_string.c_str(),
      static_cast<Py_ssize_t>(compiler_string.size())));
  if (module_name == nullptr || compiler_path == nullptr) {
    python_compilation_failure(
        "cannot encode the pinned Ascend standalone compiler path");
  }
  OwnedPyObject spec(PyObject_CallFunctionObjArgs(spec_from_file_location.get(),
                                                  module_name.get(),
                                                  compiler_path.get(),
                                                  nullptr));
  if (spec == nullptr) {
    python_compilation_failure(
        "cannot create the Ascend standalone compiler module specification");
  }
  OwnedPyObject module(
      PyObject_CallFunctionObjArgs(module_from_spec.get(), spec.get(), nullptr));
  if (module == nullptr) {
    python_compilation_failure(
        "cannot create the Ascend standalone compiler module");
  }
  OwnedPyObject loader(PyObject_GetAttrString(spec.get(), "loader"));
  OwnedPyObject exec_module(
      loader == nullptr ? nullptr
                        : PyObject_GetAttrString(loader.get(), "exec_module"));
  if (loader == nullptr || exec_module == nullptr) {
    python_compilation_failure(
        "Ascend standalone compiler module has no executable loader");
  }
  if (PyCallable_Check(exec_module.get()) == 0) {
    compilation_failure(
        "Ascend standalone compiler module loader is not callable");
  }
  OwnedPyObject executed(
      PyObject_CallFunctionObjArgs(exec_module.get(), module.get(), nullptr));
  if (executed == nullptr) {
    python_compilation_failure(
        "cannot execute the pinned Ascend standalone compiler module");
  }

  OwnedPyObject compile(
      PyObject_GetAttrString(module.get(), "compile_a_kernel"));
  if (compile == nullptr) {
    python_compilation_failure(
        "pinned Ascend standalone compiler has no compile_a_kernel API");
  }
  if (PyCallable_Check(compile.get()) == 0) {
    compilation_failure(
        "pinned Ascend standalone compile_a_kernel API is not callable");
  }

  const std::string source_string = candidate.source.string();
  OwnedPyObject source_path(PyUnicode_DecodeFSDefaultAndSize(
      source_string.c_str(), static_cast<Py_ssize_t>(source_string.size())));
  OwnedPyObject entry_point(PyUnicode_FromStringAndSize(
      candidate.entry_point.data(),
      static_cast<Py_ssize_t>(candidate.entry_point.size())));
  OwnedPyObject signature(PyUnicode_FromStringAndSize(
      candidate.full_signature.data(),
      static_cast<Py_ssize_t>(candidate.full_signature.size())));
  OwnedPyObject num_warps(PyLong_FromUnsignedLong(candidate.num_warps));
  OwnedPyObject num_stages(PyLong_FromUnsignedLong(candidate.num_stages));
  OwnedPyObject device(PyLong_FromLong(request->device_ordinal));
  OwnedPyObject extra_options(PyDict_New());
  if (source_path == nullptr || entry_point == nullptr ||
      signature == nullptr || num_warps == nullptr || num_stages == nullptr ||
      device == nullptr || extra_options == nullptr) {
    python_compilation_failure(
        "cannot encode the Ascend standalone compilation request");
  }
  OwnedPyObject result(PyObject_CallFunctionObjArgs(compile.get(),
                                                    source_path.get(),
                                                    entry_point.get(),
                                                    signature.get(),
                                                    num_warps.get(),
                                                    num_stages.get(),
                                                    device.get(),
                                                    extra_options.get(),
                                                    nullptr));
  if (result == nullptr) {
    python_compilation_failure("Ascend standalone kernel compilation failed");
  }
  OwnedPyObject path(PyOS_FSPath(result.get()));
  if (path == nullptr) {
    python_compilation_failure(
        "Ascend standalone compiler returned a non-path cache directory");
  }

  const char* data = nullptr;
  Py_ssize_t size = 0;
  if (PyUnicode_Check(path.get()) != 0) {
    data = PyUnicode_AsUTF8AndSize(path.get(), &size);
  } else if (PyBytes_Check(path.get()) != 0) {
    char* bytes = nullptr;
    if (PyBytes_AsStringAndSize(path.get(), &bytes, &size) != 0) {
      data = nullptr;
    } else {
      data = bytes;
    }
  } else {
    compilation_failure(
        "Ascend standalone compiler returned an unsupported path type");
  }
  if (data == nullptr || size <= 0) {
    python_compilation_failure(
        "cannot decode the Ascend standalone compiler cache directory");
  }
  if (std::memchr(data, '\0', static_cast<std::size_t>(size)) != nullptr) {
    compilation_failure(
        "Ascend standalone compiler returned a cache path containing NUL");
  }
  request->cache_directory.assign(data, static_cast<std::size_t>(size));
}

[[nodiscard]] fs::path compile_candidate_without_launch(
    const EngineBuildContext& context,
    const fs::path& cache_root,
    const LtjNpuRawCandidate& candidate) {
  ContainedStandaloneCompile request;
  request.candidate = &candidate;
  request.compiler = validate_standalone_compiler();
  request.device_ordinal = context.device_ordinal;
  detail::run_with_contained_python_stdout(compile_with_contained_stdout,
                                            &request);
  if (request.cache_directory.empty()) {
    compilation_failure(
        "Ascend standalone compiler returned an empty cache directory");
  }

  const fs::path returned(request.cache_directory);
  if (!returned.is_absolute()) {
    compilation_failure(
        "Ascend standalone compiler returned a relative cache directory");
  }
  std::error_code error;
  const fs::path canonical = fs::canonical(returned, error);
  const fs::file_status status = fs::symlink_status(returned, error);
  if (error || canonical != returned.lexically_normal() ||
      !fs::is_directory(status) || fs::is_symlink(status) ||
      !path_is_within(canonical, cache_root)) {
    compilation_failure(
        "Ascend standalone compiler returned an untrusted cache directory");
  }
  return canonical;
}

void* find_binding(const flagdnnBackendBindingV2 bindings[],
                   std::size_t binding_count,
                   std::int64_t uid) {
  void* result = nullptr;
  std::size_t matches = 0;
  for (std::size_t index = 0; index < binding_count; ++index) {
    if (bindings[index].uid == uid) {
      result = bindings[index].device_pointer;
      ++matches;
    }
  }
  require(matches == 1 && result != nullptr,
          "a required Ascend binding is missing or duplicated");
  return result;
}

void validate_execution_inputs(const AscendArtifact& artifact,
                               const flagdnnBackendBindingV2 bindings[],
                               std::size_t binding_count,
                               void* workspace,
                               std::size_t workspace_size) {
  require(binding_count == artifact.binding_uids.size() &&
              (binding_count == 0 || bindings != nullptr),
          "Ascend binding count does not match the executable");
  for (const std::int64_t uid : artifact.binding_uids) {
    (void)find_binding(bindings, binding_count, uid);
  }
  require(workspace_size >= artifact.workspace_size,
          "Ascend Graph workspace is smaller than the executable requirement");
  if (artifact.workspace_size != 0) {
    require(workspace != nullptr &&
                reinterpret_cast<std::uintptr_t>(workspace) %
                        kGraphWorkspaceAlignment ==
                    0,
            "Ascend Graph workspace must be non-null and 256-byte aligned");
  }
}

class DeviceAllocation {
 public:
  DeviceAllocation() = default;
  explicit DeviceAllocation(std::size_t size) : size_(size) {
    if (size_ != 0) {
      check_acl(aclrtMalloc(&value_, size_, ACL_MEM_MALLOC_HUGE_FIRST),
                "aclrtMalloc(Ascend build-time resource)");
    }
  }
  ~DeviceAllocation() {
    if (!release_noexcept()) {
      latch_process_terminal();
    }
  }

  DeviceAllocation(DeviceAllocation&& other) noexcept
      : value_(std::exchange(other.value_, nullptr)),
        size_(std::exchange(other.size_, 0)) {}
  DeviceAllocation& operator=(DeviceAllocation&& other) noexcept {
    if (this != &other) {
      if (!release_noexcept()) {
        latch_process_terminal();
      }
      value_ = std::exchange(other.value_, nullptr);
      size_ = std::exchange(other.size_, 0);
    }
    return *this;
  }
  DeviceAllocation(const DeviceAllocation&) = delete;
  DeviceAllocation& operator=(const DeviceAllocation&) = delete;

  [[nodiscard]] void* get() const noexcept { return value_; }
  [[nodiscard]] std::size_t size() const noexcept { return size_; }

  [[nodiscard]] bool release_noexcept() noexcept {
    if (value_ == nullptr) {
      return true;
    }
    void* value = std::exchange(value_, nullptr);
    size_ = 0;
    return aclrtFree(value) == ACL_SUCCESS;
  }

  void abandon() noexcept {
    value_ = nullptr;
    size_ = 0;
  }

 private:
  void* value_ = nullptr;
  std::size_t size_ = 0;
};

class BuildResources {
 public:
  BuildResources() = default;
  ~BuildResources() {
    if (!release_noexcept()) {
      latch_process_terminal();
    }
  }

  BuildResources(const BuildResources&) = delete;
  BuildResources& operator=(const BuildResources&) = delete;

  void initialize(const AscendArtifact& artifact) {
    require(stream_ == nullptr && allocations_.empty() && bindings_.empty() &&
                workspace_.get() == nullptr &&
                kernel_workspace_.get() == nullptr,
            "Ascend build-time resources were initialized twice",
            FLAGDNN_BACKEND_RESULT_INTERNAL_ERROR);
    check_acl(aclrtCreateStream(&stream_), "aclrtCreateStream(Ascend build)");

    std::map<std::int64_t, std::pair<std::size_t, std::size_t>> requirements;
    for (const AscendStageArtifact& stage : artifact.stages) {
      for (const ArgumentSource& argument : stage.arguments) {
        if (argument.source != ArgumentSourceKind::kBinding) {
          continue;
        }
        auto& requirement = requirements[argument.uid];
        requirement.first = std::max(requirement.first, argument.size);
        requirement.second = std::max(requirement.second, argument.alignment);
      }
    }

    allocations_.reserve(artifact.binding_uids.size());
    bindings_.reserve(artifact.binding_uids.size());
    for (const std::int64_t uid : artifact.binding_uids) {
      const auto found = requirements.find(uid);
      require(found != requirements.end() && found->second.first != 0 &&
                  found->second.second != 0,
              "Ascend build-time binding has no valid allocation description",
              FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
      allocations_.emplace_back(found->second.first);
      void* pointer = allocations_.back().get();
      require(pointer != nullptr &&
                  reinterpret_cast<std::uintptr_t>(pointer) %
                          found->second.second ==
                      0,
              "Ascend build-time allocation does not satisfy artifact alignment",
              FLAGDNN_BACKEND_RESULT_RUNTIME_ERROR);
      bindings_.push_back({uid, pointer});
      check_acl(aclrtMemsetAsync(pointer,
                                 allocations_.back().size(),
                                 0,
                                 allocations_.back().size(),
                                 stream_),
                "aclrtMemsetAsync(Ascend build binding)");
      synchronized_ = false;
    }

    if (artifact.workspace_size != 0) {
      workspace_ = DeviceAllocation(artifact.workspace_size);
      require(reinterpret_cast<std::uintptr_t>(workspace_.get()) %
                      kGraphWorkspaceAlignment ==
                  0,
              "Ascend build-time Graph workspace is not 256-byte aligned",
              FLAGDNN_BACKEND_RESULT_RUNTIME_ERROR);
      check_acl(aclrtMemsetAsync(workspace_.get(),
                                 workspace_.size(),
                                 0,
                                 workspace_.size(),
                                 stream_),
                "aclrtMemsetAsync(Ascend build workspace)");
      synchronized_ = false;
    }
    synchronize();
  }

  void synchronize() {
    check_acl(aclrtSynchronizeStream(stream_),
              "aclrtSynchronizeStream(Ascend build)");
    synchronized_ = true;
  }

  [[nodiscard]] bool synchronize_noexcept() noexcept {
    if (stream_ != nullptr && !synchronized_) {
      if (aclrtSynchronizeStream(stream_) == ACL_SUCCESS) {
        synchronized_ = true;
      } else {
        return false;
      }
    }
    return true;
  }

  void mark_pending() noexcept { synchronized_ = false; }

  void release() {
    if (!synchronized_) {
      synchronize();
    }
    if (!release_synchronized_noexcept()) {
      throw AscendError(FLAGDNN_BACKEND_RESULT_RUNTIME_ERROR,
                        "Ascend build-time resources could not be released");
    }
  }

  [[nodiscard]] bool release_noexcept() noexcept {
    if (stream_ != nullptr && !synchronized_) {
      if (aclrtSynchronizeStream(stream_) != ACL_SUCCESS) {
        abandon_noexcept();
        return false;
      }
      synchronized_ = true;
    }
    return release_synchronized_noexcept();
  }

  [[nodiscard]] bool release_synchronized_noexcept() noexcept {
    bool success = !cleanup_failed_;
    for (DeviceAllocation& allocation : allocations_) {
      success = allocation.release_noexcept() && success;
    }
    allocations_.clear();
    success = workspace_.release_noexcept() && success;
    success = kernel_workspace_.release_noexcept() && success;
    bindings_.clear();
    if (stream_ != nullptr) {
      aclrtStream stream = stream_;
      const aclError status = aclrtDestroyStream(stream);
      stream_ = nullptr;
      success = status == ACL_SUCCESS && success;
    }
    cleanup_failed_ = !success;
    return success;
  }

  [[nodiscard]] aclrtStream stream() const noexcept { return stream_; }
  [[nodiscard]] const flagdnnBackendBindingV2* bindings() const noexcept {
    return bindings_.data();
  }
  [[nodiscard]] std::size_t binding_count() const noexcept {
    return bindings_.size();
  }
  [[nodiscard]] void* workspace() const noexcept { return workspace_.get(); }
  [[nodiscard]] std::size_t workspace_size() const noexcept {
    return workspace_.size();
  }

  void ensure_kernel_workspace(std::size_t size) {
    if (size <= kernel_workspace_.size()) {
      return;
    }
    require(synchronized_,
            "Ascend kernel workspace cannot grow while work is pending",
            FLAGDNN_BACKEND_RESULT_INTERNAL_ERROR);
    kernel_workspace_ = DeviceAllocation(size);
    require(kernel_workspace_.get() != nullptr,
            "Ascend kernel workspace allocation returned null",
            FLAGDNN_BACKEND_RESULT_ALLOC_FAILED);
  }

  [[nodiscard]] void* kernel_workspace() const noexcept {
    return kernel_workspace_.get();
  }
  [[nodiscard]] std::size_t kernel_workspace_size() const noexcept {
    return kernel_workspace_.size();
  }

 private:
  void abandon_noexcept() noexcept {
    for (DeviceAllocation& allocation : allocations_) {
      allocation.abandon();
    }
    allocations_.clear();
    workspace_.abandon();
    kernel_workspace_.abandon();
    bindings_.clear();
    stream_ = nullptr;
    synchronized_ = false;
    cleanup_failed_ = true;
  }

  aclrtStream stream_ = nullptr;
  std::vector<DeviceAllocation> allocations_;
  std::vector<flagdnnBackendBindingV2> bindings_;
  DeviceAllocation workspace_;
  DeviceAllocation kernel_workspace_;
  bool synchronized_ = false;
  bool cleanup_failed_ = false;
};
struct HookCapture {
  bool entered = false;
  unsigned int shared_memory = 0;
};

void require_locked_configuration(const EngineBuildContext& context,
                                  bool* terminal_failure) {
  if (detail::process_domain().terminal_failure_latched() ||
      !process_configuration_matches(context)) {
    *terminal_failure = true;
    latch_process_terminal();
    throw AscendError(
        FLAGDNN_BACKEND_RESULT_NOT_SUPPORTED,
        "Ascend process configuration changed while waiting for or "
        "executing a raw launch");
  }
}

void clear_launch_hooks_noexcept() noexcept {
  try {
    triton_jit::clear_launch_hooks();
  } catch (...) {
    latch_process_terminal();
  }
}

class LaunchHookOwner {
 public:
  LaunchHookOwner(const EngineBuildContext& context,
                  const LtjNpuRawCandidate& candidate,
                  aclrtStream stream,
                  HookCapture* capture) {
    triton_jit::set_launch_enter_hook(
        [&context, &candidate, stream, capture](
            const triton_jit::LaunchMetadata& metadata) {
          aclrtContext current = nullptr;
          aclError status = aclrtGetCurrentContext(&current);
          if (status == ACL_SUCCESS && current == nullptr) {
            status = aclrtSetCurrentContext(context.context);
            current = status == ACL_SUCCESS ? context.context : nullptr;
          }
          const bool matches =
              !capture->entered &&
              !detail::process_domain().terminal_failure_latched() &&
              process_configuration_matches(context) &&
              status == ACL_SUCCESS &&
              current == context.context &&
              metadata.kernel_name == candidate.entry_point &&
              metadata.grid_x == candidate.grid[0] &&
              metadata.grid_y == candidate.grid[1] &&
              metadata.grid_z == candidate.grid[2] &&
              metadata.num_warps == static_cast<int>(candidate.num_warps) &&
              metadata.signature == candidate.full_signature &&
              metadata.stream == reinterpret_cast<void*>(stream);
          if (!matches) {
            latch_process_terminal();
            throw std::runtime_error(
                "libtriton_jit launch-enter metadata/context mismatch");
          }
          capture->entered = true;
          capture->shared_memory = metadata.shared_memory;
        });
    active_ = true;
  }

  ~LaunchHookOwner() { clear_noexcept(); }
  LaunchHookOwner(const LaunchHookOwner&) = delete;
  LaunchHookOwner& operator=(const LaunchHookOwner&) = delete;

  void clear() {
    if (active_) {
      triton_jit::clear_launch_hooks();
      active_ = false;
    }
  }

  void clear_noexcept() noexcept {
    if (active_) {
      clear_launch_hooks_noexcept();
      active_ = false;
    }
  }

 private:
  bool active_ = false;
};

[[nodiscard]] constexpr std::size_t raw_argument_size(
    RawArgumentType type) noexcept {
  switch (type) {
    case RawArgumentType::kPointer:
      return sizeof(void*);
    case RawArgumentType::kI32:
      return sizeof(std::int32_t);
    case RawArgumentType::kI64:
      return sizeof(std::int64_t);
    case RawArgumentType::kF32:
      return sizeof(float);
    case RawArgumentType::kF64:
      return sizeof(double);
  }
  return 0;
}

[[nodiscard]] constexpr std::size_t align_up(std::size_t value,
                                             std::size_t alignment) noexcept {
  return (value + alignment - 1U) & ~(alignment - 1U);
}

class PreparedNpuLaunch {
 public:
  [[nodiscard]] bool is_prepared() const noexcept {
    return kernel_handle_ != nullptr;
  }

  void prepare(const AscendStageArtifact& stage,
               const LtjNpuRawCandidate& candidate,
               const fs::path& metadata_directory,
               std::size_t kernel_workspace_size) {
    require(!metadata_directory.empty(),
            "Ascend selected candidate has no attested cache directory",
            FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
    require(stage.arguments.size() == candidate.argument_types.size() &&
                !stage.arguments.empty() &&
                stage.arguments.size() <=
                    FLAGDNN_BACKEND_MAX_KERNEL_ARGUMENTS,
            "Ascend prepared argument metadata is inconsistent",
            FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);

    const LtjLoadKernelMethod load_kernel = exported_load_kernel_method();
    kernel_handle_ =
        load_kernel(metadata_directory.string(), candidate.entry_point);
    require(kernel_handle_ != nullptr,
            "libtriton_jit returned a null prepared NPU kernel",
            FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);

    std::uint64_t ffts_address = 0;
    std::uint32_t ffts_size = 0;
    const rtError_t ffts_status =
        rtGetC2cCtrlAddr(&ffts_address, &ffts_size);
    if (ffts_status != RT_ERROR_NONE) {
      throw AscendError(
          FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
          "rtGetC2cCtrlAddr failed while preparing an Ascend launch with "
          "runtime status " +
              std::to_string(static_cast<int>(ffts_status)));
    }
    ffts_address_ = reinterpret_cast<void*>(ffts_address);

    std::uint64_t block_count = 1;
    for (const unsigned int dimension : candidate.grid) {
      require(dimension != 0 &&
                  block_count <=
                      std::numeric_limits<std::uint32_t>::max() / dimension,
              "Ascend prepared launch grid is invalid",
              FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
      block_count *= dimension;
    }
    block_count_ = static_cast<std::uint32_t>(block_count);
    candidate_grid_ = candidate.grid;
    for (const unsigned int dimension : candidate_grid_) {
      require(dimension <=
                  static_cast<unsigned int>(
                      std::numeric_limits<std::int32_t>::max()),
              "Ascend prepared grid exceeds the kernel argument ABI",
              FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
    }

    argument_offsets_.clear();
    argument_offsets_.reserve(candidate.argument_types.size());
    std::size_t cursor = kNpuSystemArgumentBytes;
    for (std::size_t index = 0; index < candidate.argument_types.size();
         ++index) {
      const RawArgumentType type = candidate.argument_types[index];
      require(stage.arguments[index].type == type,
              "Ascend prepared argument type differs from its ABI",
              FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
      const std::size_t size = raw_argument_size(type);
      require(size != 0 && (size & (size - 1U)) == 0,
              "Ascend prepared argument type has an invalid size",
              FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
      cursor = align_up(cursor, size);
      argument_offsets_.push_back(cursor);
      require(cursor <= kMaximumPreparedArgumentBytes - size,
              "Ascend prepared argument buffer exceeds its ABI limit",
              FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
      cursor += size;
    }
    grid_offset_ = align_up(cursor, alignof(std::int32_t));
    require(grid_offset_ <= kMaximumPreparedArgumentBytes -
                                3U * sizeof(std::int32_t),
            "Ascend prepared grid buffer exceeds its ABI limit",
            FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
    packed_size_ = grid_offset_ + 3U * sizeof(std::int32_t);
    require(packed_size_ <= std::numeric_limits<std::uint32_t>::max(),
            "Ascend prepared argument size exceeds the runtime ABI",
            FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
    kernel_workspace_size_ = kernel_workspace_size;
  }

  void launch(aclrtStream stream,
              const AscendStageArtifact& stage,
              const flagdnnBackendBindingV2 bindings[],
              std::size_t binding_count,
              void* workspace,
              std::size_t workspace_size,
              void* kernel_workspace,
              std::size_t kernel_workspace_size) const {
    require(stream != nullptr && kernel_handle_ != nullptr &&
                stage.arguments.size() == argument_offsets_.size(),
            "Ascend prepared launch is incomplete",
            FLAGDNN_BACKEND_RESULT_INTERNAL_ERROR);

    alignas(std::uint64_t)
        std::array<std::byte, kMaximumPreparedArgumentBytes> buffer{};
    void* sync_lock = nullptr;
    require(kernel_workspace_size >= kernel_workspace_size_ &&
                (kernel_workspace_size_ == 0 || kernel_workspace != nullptr),
            "Ascend compiler workspace is smaller than the kernel requirement",
            FLAGDNN_BACKEND_RESULT_INVALID_VALUE);
    std::memcpy(buffer.data(), &ffts_address_, sizeof(ffts_address_));
    std::memcpy(buffer.data() + sizeof(void*),
                &sync_lock,
                sizeof(sync_lock));
    std::memcpy(buffer.data() + 2U * sizeof(void*),
                &kernel_workspace,
                sizeof(kernel_workspace));

    for (std::size_t index = 0; index < stage.arguments.size(); ++index) {
      const ArgumentSource& source = stage.arguments[index];
      require(source.index == index,
              "Ascend prepared argument index differs from its ABI",
              FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
      std::byte* destination = buffer.data() + argument_offsets_[index];
      if (source.source == ArgumentSourceKind::kBinding) {
        void* pointer = find_binding(bindings, binding_count, source.uid);
        require(reinterpret_cast<std::uintptr_t>(pointer) %
                        source.alignment ==
                    0,
                "Ascend binding does not satisfy artifact alignment");
        std::memcpy(destination, &pointer, sizeof(pointer));
      } else if (source.source == ArgumentSourceKind::kGraphWorkspace) {
        require(workspace != nullptr &&
                    source.workspace_offset <= workspace_size &&
                    source.size <= workspace_size - source.workspace_offset,
                "Ascend Graph workspace argument is out of range");
        const std::uintptr_t address =
            reinterpret_cast<std::uintptr_t>(workspace) +
            source.workspace_offset;
        require(address % source.alignment == 0,
                "Ascend Graph workspace argument is misaligned");
        void* pointer = reinterpret_cast<void*>(address);
        std::memcpy(destination, &pointer, sizeof(pointer));
      } else {
        switch (source.type) {
          case RawArgumentType::kI32: {
            const std::int32_t value = std::get<std::int32_t>(source.scalar);
            std::memcpy(destination, &value, sizeof(value));
            break;
          }
          case RawArgumentType::kI64: {
            const std::int64_t value = std::get<std::int64_t>(source.scalar);
            std::memcpy(destination, &value, sizeof(value));
            break;
          }
          case RawArgumentType::kF32: {
            const float value = std::get<float>(source.scalar);
            std::memcpy(destination, &value, sizeof(value));
            break;
          }
          case RawArgumentType::kF64: {
            const double value = std::get<double>(source.scalar);
            std::memcpy(destination, &value, sizeof(value));
            break;
          }
          case RawArgumentType::kPointer:
            throw AscendError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                              "Ascend scalar source has pointer ABI");
        }
      }
    }

    for (std::size_t axis = 0; axis < 3; ++axis) {
      const std::int32_t dimension =
          static_cast<std::int32_t>(candidate_grid_[axis]);
      std::memcpy(buffer.data() + grid_offset_ +
                      axis * sizeof(std::int32_t),
                  &dimension,
                  sizeof(dimension));
    }

    const rtError_t status =
        rtKernelLaunch(kernel_handle_,
                       block_count_,
                       buffer.data(),
                       static_cast<std::uint32_t>(packed_size_),
                       nullptr,
                       stream);
    if (status != RT_ERROR_NONE) {
      throw AscendError(
          FLAGDNN_BACKEND_RESULT_RUNTIME_ERROR,
          "rtKernelLaunch(prepared) failed with runtime status " +
              std::to_string(static_cast<int>(status)));
    }
  }

 private:
  void* kernel_handle_ = nullptr;
  void* ffts_address_ = nullptr;
  std::uint32_t block_count_ = 0;
  std::array<unsigned int, 3> candidate_grid_ = {1, 1, 1};
  std::vector<std::size_t> argument_offsets_;
  std::size_t grid_offset_ = 0;
  std::size_t packed_size_ = 0;
  std::size_t kernel_workspace_size_ = 0;
};

struct StageLaunch {
  std::size_t stage_index = 0;
  LtjNpuRawCandidate candidate;
  LtjFunction* function = nullptr;
  fs::path metadata_directory;
  std::size_t kernel_workspace_size = 0;
  PreparedNpuLaunch prepared;
};

[[nodiscard]] std::size_t total_kernel_workspace_size(
    const LtjNpuRawCandidate& candidate,
    std::size_t per_block_workspace) {
  std::size_t block_count = 1;
  for (const unsigned int dimension : candidate.grid) {
    require(dimension != 0 &&
                block_count <= kMaximumKernelWorkspaceBytes / dimension,
            "Ascend kernel workspace grid exceeds its limit",
            FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
    block_count *= dimension;
  }
  require(per_block_workspace == 0 ||
              block_count <=
                  kMaximumKernelWorkspaceBytes / per_block_workspace,
          "Ascend kernel workspace allocation exceeds its limit",
          FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
  return block_count * per_block_workspace;
}

[[nodiscard]] bool uses_standalone_compilation(
    const LtjNpuRawCandidate& candidate) noexcept {
  return candidate.entry_point == "convolution_fprop_persistent_kernel";
}

void compile_and_attest(const EngineBuildContext& context,
                        const fs::path& cache_root,
                        const AscendStageArtifact& stage,
                        StageLaunch& launch,
                        bool* terminal_failure) {
  const CacheSnapshot before =
      scan_cache(cache_root, launch.candidate.entry_point);
  require_locked_configuration(context, terminal_failure);
  const fs::path returned_directory =
      compile_candidate_without_launch(context, cache_root, launch.candidate);
  require_locked_configuration(context, terminal_failure);
  const CacheSnapshot after =
      scan_cache(cache_root, launch.candidate.entry_point);
  fs::path metadata_path;
  const std::string key = candidate_key(cache_root, context, launch.candidate);
  const RegularFile& metadata =
      select_metadata(before,
                      after,
                      key,
                      launch.candidate.entry_point,
                      &metadata_path);
  const std::size_t per_block_workspace =
      validate_metadata(metadata_path,
                        metadata,
                        cache_root,
                        after,
                        launch.candidate,
                        std::nullopt);
  const std::size_t kernel_workspace =
      total_kernel_workspace_size(launch.candidate, per_block_workspace);
  const fs::path metadata_directory = metadata_path.parent_path();
  require(metadata_directory == returned_directory,
          "standalone compiler cache result differs from attested metadata",
          FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
  require(launch.metadata_directory.empty() ||
              launch.metadata_directory == metadata_directory,
          "selected NPU cache directory changed between compilation steps",
          FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
  require(launch.metadata_directory.empty() ||
              launch.kernel_workspace_size == kernel_workspace,
          "selected NPU kernel workspace changed between compilation steps",
          FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
  launch.metadata_directory = metadata_directory;
  launch.kernel_workspace_size = kernel_workspace;

  const CacheSnapshot before_prepare =
      scan_cache(cache_root, launch.candidate.entry_point);
  launch.prepared.prepare(stage,
                          launch.candidate,
                          launch.metadata_directory,
                          launch.kernel_workspace_size);
  const CacheSnapshot after_prepare =
      scan_cache(cache_root, launch.candidate.entry_point);
  if (!same_cache_snapshot(before_prepare, after_prepare)) {
    compilation_failure(
        "preparing an Ascend launch changed the attested cache tree");
  }
  require_locked_configuration(context, terminal_failure);
}

class AutotuneEvents {
 public:
  AutotuneEvents() {
    check_acl(aclrtCreateEventExWithFlag(&start_, ACL_EVENT_TIME_LINE),
              "aclrtCreateEventExWithFlag(autotune start)");
    try {
      check_acl(aclrtCreateEventExWithFlag(&end_, ACL_EVENT_TIME_LINE),
                "aclrtCreateEventExWithFlag(autotune end)");
    } catch (...) {
      (void)aclrtDestroyEvent(start_);
      start_ = nullptr;
      throw;
    }
  }

  ~AutotuneEvents() {
    if (!release_noexcept()) {
      latch_process_terminal();
    }
  }

  AutotuneEvents(const AutotuneEvents&) = delete;
  AutotuneEvents& operator=(const AutotuneEvents&) = delete;

  [[nodiscard]] aclrtEvent start() const noexcept { return start_; }
  [[nodiscard]] aclrtEvent end() const noexcept { return end_; }

  void abandon() noexcept {
    start_ = nullptr;
    end_ = nullptr;
  }

  void release() {
    bool success = true;
    if (end_ != nullptr) {
      const aclrtEvent event = end_;
      end_ = nullptr;
      success = aclrtDestroyEvent(event) == ACL_SUCCESS && success;
    }
    if (start_ != nullptr) {
      const aclrtEvent event = start_;
      start_ = nullptr;
      success = aclrtDestroyEvent(event) == ACL_SUCCESS && success;
    }
    if (!success) {
      throw AscendError(FLAGDNN_BACKEND_RESULT_RUNTIME_ERROR,
                        "Ascend autotune events could not be destroyed");
    }
  }

 private:
  [[nodiscard]] bool release_noexcept() noexcept {
    bool success = true;
    if (end_ != nullptr) {
      const aclrtEvent event = end_;
      end_ = nullptr;
      success = aclrtDestroyEvent(event) == ACL_SUCCESS && success;
    }
    if (start_ != nullptr) {
      const aclrtEvent event = start_;
      start_ = nullptr;
      success = aclrtDestroyEvent(event) == ACL_SUCCESS && success;
    }
    return success;
  }

  aclrtEvent start_ = nullptr;
  aclrtEvent end_ = nullptr;
};

void launch_and_attest(const EngineBuildContext& context,
                       const fs::path& cache_root,
                       const AscendStageArtifact& stage,
                       StageLaunch& launch,
                       aclrtStream stream,
                       const flagdnnBackendBindingV2 bindings[],
                       std::size_t binding_count,
                       void* workspace,
                       std::size_t workspace_size,
                       bool* raw_started,
                       bool* terminal_failure) {
  const CacheSnapshot before =
      scan_cache(cache_root, launch.candidate.entry_point);
  HookCapture capture;
  LaunchHookOwner hook(context, launch.candidate, stream, &capture);
  try {
    RawArgumentPack arguments(stage,
                              launch.candidate,
                              bindings,
                              binding_count,
                              workspace,
                              workspace_size);
    require_locked_configuration(context, terminal_failure);
    ContainedRawLaunch contained_launch{launch.function,
                                        stream,
                                        &launch.candidate,
                                        arguments.data(),
                                        arguments.size(),
                                        raw_started};
    detail::run_with_contained_python_stdout(
        launch_create_raw_with_contained_stdout, &contained_launch);
    require_locked_configuration(context, terminal_failure);
    if (!capture.entered) {
      compilation_failure("libtriton_jit did not invoke the launch-enter hook");
    }
    const CacheSnapshot after =
        scan_cache(cache_root, launch.candidate.entry_point);
    fs::path metadata_path;
    const std::string key = candidate_key(cache_root, context, launch.candidate);
    const RegularFile& metadata =
        select_metadata(before,
                        after,
                        key,
                        launch.candidate.entry_point,
                        &metadata_path);
    const std::size_t per_block_workspace =
        validate_metadata(metadata_path,
                          metadata,
                          cache_root,
                          after,
                          launch.candidate,
                          capture.shared_memory);
    const std::size_t kernel_workspace =
        total_kernel_workspace_size(launch.candidate, per_block_workspace);
    const fs::path metadata_directory = metadata_path.parent_path();
    require(launch.metadata_directory.empty() ||
                launch.metadata_directory == metadata_directory,
            "selected NPU cache directory changed between prewarm launches",
            FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
    require(launch.metadata_directory.empty() ||
                launch.kernel_workspace_size == kernel_workspace,
            "selected NPU kernel workspace changed between prewarm launches",
            FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
    launch.metadata_directory = metadata_directory;
    launch.kernel_workspace_size = kernel_workspace;
    require_locked_configuration(context, terminal_failure);
    hook.clear();
  } catch (...) {
    if (*raw_started) {
      latch_process_terminal();
    }
    hook.clear_noexcept();
    throw;
  }
}

void launch_build_candidate(const EngineBuildContext& context,
                            const fs::path& cache_root,
                            const AscendStageArtifact& stage,
                            StageLaunch& launch,
                            BuildResources& resources,
                            bool* raw_started,
                            bool* terminal_failure) {
  resources.mark_pending();
  if (!uses_standalone_compilation(launch.candidate)) {
    launch_and_attest(context,
                      cache_root,
                      stage,
                      launch,
                      resources.stream(),
                      resources.bindings(),
                      resources.binding_count(),
                      resources.workspace(),
                      resources.workspace_size(),
                      raw_started,
                      terminal_failure);
    return;
  }

  require(launch.prepared.is_prepared(),
          "standalone-compiled Ascend candidate is not prepared",
          FLAGDNN_BACKEND_RESULT_INTERNAL_ERROR);
  require(resources.kernel_workspace_size() >=
              launch.kernel_workspace_size,
          "Ascend build-time compiler workspace is too small",
          FLAGDNN_BACKEND_RESULT_INTERNAL_ERROR);
  require_locked_configuration(context, terminal_failure);
  *raw_started = true;
  launch.prepared.launch(resources.stream(),
                         stage,
                         resources.bindings(),
                         resources.binding_count(),
                         resources.workspace(),
                         resources.workspace_size(),
                         resources.kernel_workspace(),
                         resources.kernel_workspace_size());
  require_locked_configuration(context, terminal_failure);
}

[[nodiscard]] double measure_candidate(
    const EngineBuildContext& context,
    const fs::path& cache_root,
    const AscendStageArtifact& stage,
    const StageLaunch& launch,
    BuildResources& resources,
    bool* raw_started,
    bool* terminal_failure) {
  const CacheSnapshot before =
      scan_cache(cache_root, launch.candidate.entry_point);
  resources.ensure_kernel_workspace(launch.kernel_workspace_size);
  PreparedNpuLaunch prepared;
  prepared.prepare(stage,
                   launch.candidate,
                   launch.metadata_directory,
                   launch.kernel_workspace_size);
  for (unsigned int index = 0; index < stage.warmup; ++index) {
    require_locked_configuration(context, terminal_failure);
    resources.mark_pending();
    *raw_started = true;
    prepared.launch(resources.stream(),
                    stage,
                    resources.bindings(),
                    resources.binding_count(),
                    resources.workspace(),
                    resources.workspace_size(),
                    resources.kernel_workspace(),
                    resources.kernel_workspace_size());
    require_locked_configuration(context, terminal_failure);
  }
  resources.synchronize();

  AutotuneEvents events;
  std::vector<double> samples;
  try {
    samples.reserve(stage.repetitions);
    for (unsigned int index = 0; index < stage.repetitions; ++index) {
      require_locked_configuration(context, terminal_failure);
      resources.mark_pending();
      check_acl(aclrtRecordEvent(events.start(), resources.stream()),
                "aclrtRecordEvent(autotune start)");
      *raw_started = true;
      prepared.launch(resources.stream(),
                      stage,
                      resources.bindings(),
                      resources.binding_count(),
                      resources.workspace(),
                      resources.workspace_size(),
                      resources.kernel_workspace(),
                      resources.kernel_workspace_size());
      check_acl(aclrtRecordEvent(events.end(), resources.stream()),
                "aclrtRecordEvent(autotune end)");
      require_locked_configuration(context, terminal_failure);
      resources.synchronize();
      float milliseconds = 0.0F;
      check_acl(aclrtEventElapsedTime(
                    &milliseconds, events.start(), events.end()),
                "aclrtEventElapsedTime(autotune)");
      const double microseconds = static_cast<double>(milliseconds) * 1000.0;
      if (!std::isfinite(microseconds) || microseconds < 0.0) {
        compilation_failure(
            "Ascend autotune produced an invalid event sample");
      }
      samples.push_back(microseconds);
    }
    events.release();
  } catch (...) {
    if (resources.synchronize_noexcept()) {
      try {
        events.release();
      } catch (...) {
        latch_process_terminal();
        events.abandon();
      }
    } else {
      events.abandon();
    }
    throw;
  }
  require_locked_configuration(context, terminal_failure);
  const CacheSnapshot after =
      scan_cache(cache_root, launch.candidate.entry_point);
  if (!same_cache_snapshot(before, after)) {
    compilation_failure(
        "Ascend autotune changed the attested private NPU cache tree");
  }
  if (samples.empty()) {
    compilation_failure("Ascend autotune produced no timing samples");
  }
  std::sort(samples.begin(), samples.end());
  const std::size_t middle = samples.size() / 2;
  if (samples.size() % 2 != 0) {
    return samples[middle];
  }
  return (samples[middle - 1] + samples[middle]) * 0.5;
}

void prewarm_candidate(const EngineBuildContext& context,
                       const fs::path& cache_root,
                       const AscendStageArtifact& stage,
                       StageLaunch& launch,
                       BuildResources& resources,
                       bool* raw_started,
                       bool* terminal_failure) {
  launch_build_candidate(context,
                         cache_root,
                         stage,
                         launch,
                         resources,
                         raw_started,
                         terminal_failure);
  resources.synchronize();
}

void smoke_candidate(const EngineBuildContext& context,
                     const fs::path& cache_root,
                     const AscendStageArtifact& stage,
                     StageLaunch& launch,
                     BuildResources& resources,
                     bool* raw_started,
                     bool* terminal_failure) {
  if (uses_standalone_compilation(launch.candidate)) {
    compile_and_attest(context,
                       cache_root,
                       stage,
                       launch,
                       terminal_failure);
    resources.synchronize();
    resources.ensure_kernel_workspace(launch.kernel_workspace_size);
  }
  prewarm_candidate(context,
                    cache_root,
                    stage,
                    launch,
                    resources,
                    raw_started,
                    terminal_failure);
}
[[nodiscard]] std::pair<flagdnnBackendResult_t, std::string>
current_failure(flagdnnBackendResult_t fallback, const char* prefix) {
  try {
    throw;
  } catch (const AscendError& error) {
    return {error.result(), std::string(prefix) + error.what()};
  } catch (const std::bad_alloc&) {
    return {FLAGDNN_BACKEND_RESULT_ALLOC_FAILED,
            std::string(prefix) + "host allocation failed"};
  } catch (const std::exception& error) {
    return {fallback, std::string(prefix) + error.what()};
  } catch (...) {
    return {fallback, std::string(prefix) + "unknown failure"};
  }
}

class LtjExecutionEngine final : public ExecutionEngine {
 public:
  LtjExecutionEngine(EngineBuildContext context,
                     AscendArtifact artifact,
                     std::vector<StageLaunch> launches)
      : context_(std::move(context)),
        artifact_(std::move(artifact)),
        launches_(std::move(launches)) {
    for (const StageLaunch& launch : launches_) {
      kernel_workspace_size_ =
          std::max(kernel_workspace_size_, launch.kernel_workspace_size);
    }
    if (kernel_workspace_size_ == 0) {
      workspace_size_ = artifact_.workspace_size;
      return;
    }
    kernel_workspace_offset_ =
        align_up(artifact_.workspace_size, kGraphWorkspaceAlignment);
    require(kernel_workspace_offset_ <=
                std::numeric_limits<std::size_t>::max() -
                    kernel_workspace_size_,
            "Ascend executable workspace size overflows",
            FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
    workspace_size_ = kernel_workspace_offset_ + kernel_workspace_size_;
  }

  [[nodiscard]] std::size_t workspace_size() const noexcept override {
    return workspace_size_;
  }

  void execute(void* native_stream,
               const flagdnnBackendBindingV2 bindings[],
               std::size_t binding_count,
               void* workspace,
               std::size_t workspace_size) const override {
    ContextGuard context_guard(context_.context);
    aclrtStream stream = static_cast<aclrtStream>(native_stream);
    if (stream == nullptr) {
      check_acl(aclrtCtxGetCurrentDefaultStream(&stream),
                "aclrtCtxGetCurrentDefaultStream");
      require(stream != nullptr,
              "aclrtCtxGetCurrentDefaultStream returned a null stream",
              FLAGDNN_BACKEND_RESULT_RUNTIME_ERROR);
    }
    std::unique_lock<std::mutex> lock(process_ltj_mutex());
    if (detail::process_domain().terminal_failure_latched()) {
      lock.unlock();
      ensure_process_healthy();
    }

    bool raw_started = false;
    try {
      // The compiler environment, cache directories and toolchain identity are
      // fully validated while constructing this immutable engine. Raw launch
      // does not consult them, so repeating getenv/stat/access scans here
      // would turn each steady-state stage into filesystem control-plane work.
      validate_execution_inputs(
          artifact_, bindings, binding_count, workspace, workspace_size);
      require(workspace_size >= workspace_size_ &&
                  (workspace_size_ == 0 || workspace != nullptr),
              "Ascend workspace is smaller than the executable requirement");
      for (const StageLaunch& launch : launches_) {
        require(launch.stage_index < artifact_.stages.size() &&
                    launch.function != nullptr,
                "Ascend executable has an invalid stage launch",
                FLAGDNN_BACKEND_RESULT_INTERNAL_ERROR);
        const AscendStageArtifact& stage = artifact_.stages[launch.stage_index];
        void* kernel_workspace = nullptr;
        if (launch.kernel_workspace_size != 0) {
          kernel_workspace = static_cast<std::byte*>(workspace) +
                             kernel_workspace_offset_;
        }
        raw_started = true;
        launch.prepared.launch(stream,
                               stage,
                               bindings,
                               binding_count,
                               workspace,
                               workspace_size,
                               kernel_workspace,
                               kernel_workspace_size_);
      }
      lock.unlock();
    } catch (...) {
      if (!raw_started) {
        throw;
      }
      latch_process_terminal();
      clear_launch_hooks_noexcept();
      const auto [result, message] = current_failure(
          FLAGDNN_BACKEND_RESULT_RUNTIME_ERROR,
          "Ascend raw execute failed terminally: ");
      lock.unlock();
      mark_process_terminal(result, message);
      throw AscendError(result, message);
    }
  }

 private:
  EngineBuildContext context_;
  AscendArtifact artifact_;
  std::vector<StageLaunch> launches_;
  std::size_t kernel_workspace_offset_ = 0;
  std::size_t kernel_workspace_size_ = 0;
  std::size_t workspace_size_ = 0;
};

}  // namespace

RawArgumentPack::RawArgumentPack(
    const AscendStageArtifact& stage,
    const LtjNpuRawCandidate& candidate,
    const flagdnnBackendBindingV2 bindings[],
    std::size_t binding_count,
    void* workspace,
    std::size_t workspace_size) {
  require(stage.arguments.size() == candidate.argument_types.size() &&
              !stage.arguments.empty() &&
              stage.arguments.size() <= FLAGDNN_BACKEND_MAX_KERNEL_ARGUMENTS,
          "Ascend raw argument metadata is inconsistent",
          FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
  require(binding_count == 0 || bindings != nullptr,
          "Ascend binding array is null");

  size_ = stage.arguments.size();
  for (std::size_t index = 0; index < stage.arguments.size(); ++index) {
    const ArgumentSource& source = stage.arguments[index];
    require(source.index == index && source.type == candidate.argument_types[index],
            "Ascend raw argument type differs from the selected candidate",
            FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
    Slot& slot = slots_[index];
    if (source.source == ArgumentSourceKind::kBinding) {
      slot.pointer = find_binding(bindings, binding_count, source.uid);
      require(reinterpret_cast<std::uintptr_t>(slot.pointer) %
                      source.alignment ==
                  0,
              "Ascend binding does not satisfy artifact alignment");
    } else if (source.source == ArgumentSourceKind::kGraphWorkspace) {
      require(workspace != nullptr && source.workspace_offset <= workspace_size &&
                  source.size <= workspace_size - source.workspace_offset,
              "Ascend Graph workspace argument is out of range");
      const std::uintptr_t address =
          reinterpret_cast<std::uintptr_t>(workspace) +
          source.workspace_offset;
      require(address % source.alignment == 0,
              "Ascend Graph workspace argument is misaligned");
      slot.pointer = reinterpret_cast<void*>(address);
    } else {
      switch (source.type) {
        case RawArgumentType::kI32:
          slot.i32 = std::get<std::int32_t>(source.scalar);
          break;
        case RawArgumentType::kI64:
          slot.i64 = std::get<std::int64_t>(source.scalar);
          break;
        case RawArgumentType::kF32:
          slot.f32 = std::get<float>(source.scalar);
          break;
        case RawArgumentType::kF64:
          slot.f64 = std::get<double>(source.scalar);
          break;
        case RawArgumentType::kPointer:
          throw AscendError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                            "Ascend scalar source has pointer ABI");
      }
    }
  }

  for (std::size_t index = 0; index < stage.arguments.size(); ++index) {
    const ArgumentSource& source = stage.arguments[index];
    Slot& slot = slots_[index];
    switch (source.type) {
      case RawArgumentType::kPointer:
        pointers_[index] = &slot.pointer;
        break;
      case RawArgumentType::kI32:
        pointers_[index] = &slot.i32;
        break;
      case RawArgumentType::kI64:
        pointers_[index] = &slot.i64;
        break;
      case RawArgumentType::kF32:
        pointers_[index] = &slot.f32;
        break;
      case RawArgumentType::kF64:
        pointers_[index] = &slot.f64;
        break;
    }
  }
}

std::unique_ptr<ExecutionEngine> create_libtriton_jit_engine(
    const EngineBuildContext& context, AscendArtifact artifact) {
  require(Py_IsInitialized() != 0,
          "Ascend domain has no initialized embedded Python",
          FLAGDNN_BACKEND_RESULT_NOT_SUPPORTED);
  require(!context.configuration_identity.empty(),
          "Ascend compiler environment identity is empty",
          FLAGDNN_BACKEND_RESULT_NOT_SUPPORTED);
  const fs::path cache_root =
      validate_private_cache_root(fs::path(context.cache_root));

  ContextGuard context_guard(context.context);
  ensure_process_configuration(context);
  std::unique_lock<std::mutex> lock(process_ltj_mutex());
  if (detail::process_domain().terminal_failure_latched()) {
    lock.unlock();
    ensure_process_healthy();
  }

  BuildResources resources;
  bool raw_started = false;
  bool terminal_failure = false;
  try {
    require_locked_configuration(context, &terminal_failure);
    resources.initialize(artifact);
    (void)validate_private_cache_root(cache_root);
    require_locked_configuration(context, &terminal_failure);
    std::vector<StageLaunch> launches;
    launches.reserve(artifact.stages.size());
    for (std::size_t stage_index = 0; stage_index < artifact.stages.size();
         ++stage_index) {
      AscendStageArtifact& stage = artifact.stages[stage_index];
      if (!stage.autotune) {
        const auto selected = std::find_if(
            stage.candidates.begin(),
            stage.candidates.end(),
            [&stage](const LtjNpuRawCandidate& candidate) {
              return candidate.candidate_id == stage.selected_candidate;
            });
        if (stage.candidates.size() != 1 ||
            selected == stage.candidates.end()) {
          compilation_failure(
              "Ascend fixed stage does not have exactly one selected candidate");
        }
        validate_source(*selected);
        require_locked_configuration(context, &terminal_failure);
        LtjFunction& function = LtjFunction::get_instance(
            selected->source.string(), selected->entry_point);
        require_locked_configuration(context, &terminal_failure);
        launches.push_back(
            {stage_index, *selected, &function, fs::path{}, 0, {}});
        smoke_candidate(context,
                        cache_root,
                        stage,
                        launches.back(),
                        resources,
                        &raw_started,
                        &terminal_failure);
        /* Match NVIDIA's create-time policy: a second identical device launch
         * establishes the selected in-memory/cache-hit path. Correctness stays
         * in the functional suite instead of a per-operator host oracle here. */
        prewarm_candidate(context,
                          cache_root,
                          stage,
                          launches.back(),
                          resources,
                          &raw_started,
                          &terminal_failure);
        continue;
      }

      struct MeasuredCandidate {
        StageLaunch launch;
        double median_microseconds = 0.0;
      };
      std::vector<MeasuredCandidate> measured;
      measured.reserve(stage.candidates.size());
      for (const LtjNpuRawCandidate& candidate : stage.candidates) {
        validate_source(candidate);
        require_locked_configuration(context, &terminal_failure);
        LtjFunction& function = LtjFunction::get_instance(
            candidate.source.string(), candidate.entry_point);
        require_locked_configuration(context, &terminal_failure);
        StageLaunch launch{
            stage_index, candidate, &function, fs::path{}, 0, {}};
        smoke_candidate(context,
                        cache_root,
                        stage,
                        launch,
                        resources,
                        &raw_started,
                        &terminal_failure);
        /* Establish an exact cache-hit launch before collecting event samples. */
        prewarm_candidate(context,
                          cache_root,
                          stage,
                          launch,
                          resources,
                          &raw_started,
                          &terminal_failure);
        const double median = measure_candidate(context,
                                                cache_root,
                                                stage,
                                                launch,
                                                resources,
                                                &raw_started,
                                                &terminal_failure);
        measured.push_back({std::move(launch), median});
      }
      if (measured.size() < 2) {
        compilation_failure(
            "Ascend autotune did not evaluate at least two candidates");
      }
      const auto best = std::min_element(
          measured.begin(),
          measured.end(),
          [](const MeasuredCandidate& left, const MeasuredCandidate& right) {
            return left.median_microseconds < right.median_microseconds;
          });
      require(best != measured.end(),
              "Ascend autotune did not select a candidate",
              FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
      StageLaunch selected = best->launch;
      stage.selected_candidate = selected.candidate.candidate_id;

      /* Re-run and attest the selected immutable launch description.  This is
       * the final create-time prewarm and leaves the stage output ready for a
       * dependent stage without putting compilation or tuning in execute(). */
      prewarm_candidate(context,
                        cache_root,
                        stage,
                        selected,
                        resources,
                        &raw_started,
                        &terminal_failure);
      std::cerr << "[FLAGDNN_ASCEND_AUTOTUNE] stage=" << stage.stage_id
                << " candidate=" << stage.selected_candidate
                << " median_us=" << best->median_microseconds << '\n';
      launches.push_back(std::move(selected));
    }
    for (StageLaunch& launch : launches) {
      require(launch.stage_index < artifact.stages.size(),
              "selected Ascend launch has an invalid stage index",
              FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
      const CacheSnapshot before =
          scan_cache(cache_root, launch.candidate.entry_point);
      if (!launch.prepared.is_prepared()) {
        launch.prepared.prepare(artifact.stages[launch.stage_index],
                                launch.candidate,
                                launch.metadata_directory,
                                launch.kernel_workspace_size);
      }
      const CacheSnapshot after =
          scan_cache(cache_root, launch.candidate.entry_point);
      if (!same_cache_snapshot(before, after)) {
        compilation_failure(
            "preparing an Ascend launch changed the attested cache tree");
      }
    }
    resources.synchronize();
    resources.release();
    require_locked_configuration(context, &terminal_failure);
    auto result = std::make_unique<LtjExecutionEngine>(
        context, std::move(artifact), std::move(launches));
    require_locked_configuration(context, &terminal_failure);
    lock.unlock();
    return result;
  } catch (...) {
    clear_launch_hooks_noexcept();
    const bool resources_released = resources.release_noexcept();
    if (!raw_started && !terminal_failure && resources_released) {
      throw;
    }
    latch_process_terminal();
    auto [result, message] = current_failure(
        FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
        "Ascend JIT/prewarm failed terminally: ");
    if (!resources_released) {
      result = FLAGDNN_BACKEND_RESULT_RUNTIME_ERROR;
      message += "; prewarm resources could not be released";
    }
    lock.unlock();
    mark_process_terminal(result, message);
    throw AscendError(result, message);
  }
}

}  // namespace flagdnn::ascend
