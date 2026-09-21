/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include <flagdnn/flagdnn.hpp>
#include <flagdnn_frontend.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <sys/inotify.h>
#include <unistd.h>

#include <cerrno>
#endif

namespace {

class TemporaryCache {
public:
  TemporaryCache() {
    std::string pattern = (std::filesystem::temp_directory_path() /
                           "flagdnn-contract-backend-XXXXXX")
                              .string();
    std::vector<char> writable(pattern.begin(), pattern.end());
    writable.push_back('\0');
    char *created = mkdtemp(writable.data());
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

class ScopedEnvironment {
public:
  ScopedEnvironment(std::string name, std::string value)
      : name_(std::move(name)) {
    if (const char *previous = std::getenv(name_.c_str())) {
      had_previous_ = true;
      previous_ = previous;
    }
    if (setenv(name_.c_str(), value.c_str(), 1) != 0) {
      throw std::runtime_error("setenv failed for " + name_);
    }
  }

  ~ScopedEnvironment() {
    if (had_previous_) {
      (void)setenv(name_.c_str(), previous_.c_str(), 1);
    } else {
      (void)unsetenv(name_.c_str());
    }
  }

  ScopedEnvironment(const ScopedEnvironment &) = delete;
  ScopedEnvironment &operator=(const ScopedEnvironment &) = delete;

private:
  std::string name_;
  std::string previous_;
  bool had_previous_ = false;
};

std::filesystem::path
find_single_manifest(const std::filesystem::path &cache_root) {
  std::filesystem::path result;
  std::size_t count = 0;
  for (const auto &entry :
       std::filesystem::recursive_directory_iterator(cache_root)) {
    if (entry.is_regular_file() && entry.path().filename() == "manifest.json") {
      result = entry.path();
      ++count;
    }
  }
  if (count != 1) {
    throw std::runtime_error("expected one cached manifest, found " +
                             std::to_string(count));
  }
  return result;
}

std::size_t count_manifests(const std::filesystem::path &cache_root) {
  std::size_t count = 0;
  for (const auto &entry :
       std::filesystem::recursive_directory_iterator(cache_root)) {
    if (entry.is_regular_file() && entry.path().filename() == "manifest.json") {
      ++count;
    }
  }
  return count;
}

void corrupt_manifest(const std::filesystem::path &path) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  output << "{invalid cached manifest\n";
  output.close();
  if (!output) {
    throw std::runtime_error("cannot corrupt cached manifest");
  }
}

void write_identity_dependency(const std::filesystem::path &path,
                               std::string_view value) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  output.write(value.data(), static_cast<std::streamsize>(value.size()));
  if (!output) {
    throw std::runtime_error("cannot write compiler identity dependency");
  }
}

std::string read_text_file(const std::filesystem::path &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("cannot open contract test file: " +
                             path.string());
  }
  std::string value((std::istreambuf_iterator<char>(input)),
                    std::istreambuf_iterator<char>());
  if (input.bad()) {
    throw std::runtime_error("cannot read contract test file: " +
                             path.string());
  }
  return value;
}

void wait_for_regular_file(const std::filesystem::path &path,
                           std::chrono::seconds timeout) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  for (;;) {
    std::error_code error;
    if (std::filesystem::is_regular_file(path, error) && !error) {
      return;
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      throw std::runtime_error("timed out waiting for contract marker: " +
                               path.string());
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}

std::string
require_cache_identity_consistent(const std::filesystem::path &manifest) {
  const std::filesystem::path artifact_directory = manifest.parent_path();
  const std::filesystem::path graph_cache_directory =
      artifact_directory.parent_path();
  const std::string identity = artifact_directory.filename().string();
  if (identity.size() != 64) {
    throw std::runtime_error("cached artifact directory is not an identity");
  }

  std::string active_identity =
      read_text_file(graph_cache_directory / "active_identity");
  while (!active_identity.empty() &&
         (active_identity.back() == '\n' || active_identity.back() == '\r')) {
    active_identity.pop_back();
  }
  if (active_identity != identity) {
    throw std::runtime_error(
        "active compiler identity differs from artifact directory");
  }

  const std::string manifest_source = read_text_file(manifest);
  const std::string request_source =
      read_text_file(artifact_directory / "request.json");
  if (manifest_source.find("\"identity_sha256\":\"" + identity + "\"") ==
          std::string::npos ||
      request_source.find("\"compiler_identity\":\"" + identity + "\"") ==
          std::string::npos) {
    throw std::runtime_error(
        "request, manifest, and artifact identity are inconsistent");
  }
  return identity;
}

std::uint64_t read_identity_query_count(const std::filesystem::path &path) {
  std::ifstream input(path);
  std::uint64_t count = 0;
  input >> count;
  if (!input || input.peek() != std::ifstream::traits_type::eof()) {
    throw std::runtime_error("compiler identity query counter is invalid");
  }
  return count;
}

void require_native_process_clean(const char *stage) {
  std::ifstream maps("/proc/self/maps");
  if (!maps) {
    throw std::runtime_error("cannot inspect /proc/self/maps");
  }
  std::string line;
  while (std::getline(maps, line)) {
    if (line.find("libpython") != std::string::npos ||
        line.find("site-packages/torch") != std::string::npos ||
        line.find("/torch/lib/") != std::string::npos) {
      throw std::runtime_error(std::string(stage) +
                               " unexpectedly mapped Python or Torch: " + line);
    }
  }
}

template <typename Function>
void require_status(Function &&function, flagdnnStatus_t expected,
                    std::string_view context) {
  try {
    function();
  } catch (const flagdnn::Error &error) {
    if (error.status() != expected) {
      throw std::runtime_error(std::string(context) +
                               " returned unexpected status");
    }
    return;
  }
  throw std::runtime_error(std::string(context) + " unexpectedly succeeded");
}

void require_frontend_status(const flagdnn_frontend::error_t &status,
                             flagdnnStatus_t expected,
                             std::string_view context) {
  if (status.get_status() != expected) {
    throw std::runtime_error(
        std::string(context) +
        " returned unexpected frontend status: " + status.get_message());
  }
}

void require_relu(std::span<const float> input, std::span<const float> output) {
  if (input.size() != output.size()) {
    throw std::runtime_error("contract output size differs");
  }
  for (std::size_t index = 0; index < input.size(); ++index) {
    const float expected = std::max(input[index], 0.0F);
    if (output[index] != expected) {
      throw std::runtime_error("contract ReLU output differs at index " +
                               std::to_string(index));
    }
  }
}

void require_legacy_identity_protocol(const char *compiler_executable,
                                      const char *compiler_entry,
                                      const flagdnn::Graph &graph,
                                      std::string_view response_mode) {
  TemporaryCache protocol_cache;
  const std::filesystem::path query_counter =
      protocol_cache.path() / "identity-query-count";
  ScopedEnvironment counter_environment(
      "FLAGDNN_CONTRACT_COMPILER_IDENTITY_COUNTER", query_counter.string());
  ScopedEnvironment response_environment(
      "FLAGDNN_CONTRACT_COMPILER_IDENTITY_RESPONSE_MODE",
      std::string(response_mode));

  flagdnn::Handle protocol_handle("contract", 0);
  protocol_handle.set_compiler(compiler_executable, compiler_entry,
                               protocol_cache.path().string());
  flagdnn::Executable first(protocol_handle, graph);
  flagdnn::Executable second(protocol_handle, graph);
  // Legacy/incomplete responses are intentionally not memoized. A cache miss
  // queries once before compilation and once again before publication; the
  // following cache hit performs its own identity query.
  if (read_identity_query_count(query_counter) != 3) {
    throw std::runtime_error(std::string(response_mode) +
                             " identity response was incorrectly memoized");
  }
  if (count_manifests(protocol_cache.path()) != 1) {
    throw std::runtime_error(std::string(response_mode) +
                             " identity response changed the cache slot");
  }
  (void)require_cache_identity_consistent(
      find_single_manifest(protocol_cache.path()));
}


#if defined(__linux__)
class DependencyReadEvents {
public:
  explicit DependencyReadEvents(const std::filesystem::path &path) {
    descriptor_ = inotify_init1(IN_NONBLOCK | IN_CLOEXEC);
    if (descriptor_ < 0) {
      throw std::runtime_error("cannot observe compiler dependency reads");
    }
    // Watching closes as well keeps consecutive opens from coalescing.
    if (inotify_add_watch(descriptor_, path.c_str(),
                         IN_OPEN | IN_CLOSE_NOWRITE) < 0) {
      close(descriptor_);
      throw std::runtime_error("cannot watch compiler dependency reads");
    }
  }

  ~DependencyReadEvents() { close(descriptor_); }

  std::size_t take_open_count() {
    std::size_t count = 0;
    alignas(inotify_event) std::array<char, 4096> buffer{};
    for (;;) {
      const ssize_t size = read(descriptor_, buffer.data(), buffer.size());
      if (size < 0 && errno == EINTR) {
        continue;
      }
      if (size < 0 && errno == EAGAIN) {
        return count;
      }
      if (size <= 0) {
        throw std::runtime_error("cannot read dependency filesystem events");
      }
      for (std::size_t offset = 0; offset < static_cast<std::size_t>(size);) {
        const auto &event =
            *reinterpret_cast<const inotify_event *>(buffer.data() + offset);
        if ((event.mask & IN_Q_OVERFLOW) != 0) {
          throw std::runtime_error("dependency filesystem events overflowed");
        }
        if ((event.mask & IN_OPEN) != 0) {
          ++count;
        }
        offset += sizeof(inotify_event) + event.len;
      }
    }
  }

private:
  int descriptor_ = -1;
};
#endif

void require_dependency_content_memo(const char *compiler_executable,
                                     const char *compiler_entry,
                                     const flagdnn::Graph &graph) {
  TemporaryCache memo_cache;
  const auto dependency = memo_cache.path() / "dependency";
  const auto counter = memo_cache.path() / "identity-query-count";
  write_identity_dependency(dependency, "resource-v1");
  ScopedEnvironment dependency_environment(
      "FLAGDNN_CONTRACT_COMPILER_IDENTITY_DEPENDENCY", dependency.string());
  ScopedEnvironment counter_environment(
      "FLAGDNN_CONTRACT_COMPILER_IDENTITY_COUNTER", counter.string());
  flagdnn::Handle memo_handle("contract", 0);
  memo_handle.set_compiler(compiler_executable, compiler_entry,
                           memo_cache.path().string());

#if defined(__linux__)
  DependencyReadEvents reads(dependency);
#endif
  flagdnn::Executable first(memo_handle, graph);
#if defined(__linux__)
  (void)reads.take_open_count();
#endif
  {
    ScopedEnvironment recent_environment("FLAGDNN_CONTRACT_MEMO_PROBE", "recent");
    flagdnn::Executable recent(memo_handle, graph);
#if defined(__linux__)
    if (reads.take_open_count() != 4) {
      throw std::runtime_error("recent dependency incorrectly reused digest");
    }
#endif
  }
  // Even nanosecond stat fields can have coarse filesystem precision. Only
  // settled metadata is eligible for reuse, so immediate writes cannot alias.
  std::this_thread::sleep_for(std::chrono::milliseconds(2100));
  {
    ScopedEnvironment mature_environment("FLAGDNN_CONTRACT_MEMO_PROBE", "mature");
    flagdnn::Executable mature(memo_handle, graph);
#if defined(__linux__)
    (void)reads.take_open_count();
#endif
  }
  {
    // Changing an inherited environment value forces a new identity query.
    ScopedEnvironment refresh_environment("FLAGDNN_CONTRACT_MEMO_PROBE", "1");
    const auto queries_before = read_identity_query_count(counter);
    flagdnn::Executable refreshed(memo_handle, graph);
    if (read_identity_query_count(counter) != queries_before + 1) {
      throw std::runtime_error("content memo suppressed identity refresh");
    }
#if defined(__linux__)
    // The compiler reads the dependency three times: both snapshots and its
    // identity. The native verifier must reuse its independently hashed value.
    const auto opens = reads.take_open_count();
    if (opens != 3) {
      throw std::runtime_error("unchanged dependency was rehashed by native "
                               "verifier; file opens=" +
                               std::to_string(opens));
    }
#endif
  }
  {
    ScopedEnvironment forged_content_environment(
        "FLAGDNN_CONTRACT_COMPILER_FORGED_CONTENT_SHA256", "1");
    const auto queries_before = read_identity_query_count(counter);
    require_status(
        [&] { flagdnn::Executable forged(memo_handle, graph); },
        FLAGDNN_STATUS_COMPILATION_FAILED, "forged cached dependency digest");
    if (read_identity_query_count(counter) != queries_before + 3 ||
        count_manifests(memo_cache.path()) != 1) {
      throw std::runtime_error("forged dependency digest did not fail closed");
    }
  }
  flagdnn::Executable recovered(memo_handle, graph);
#if defined(__linux__)
  (void)reads.take_open_count();
#endif
  memo_handle.set_compiler(compiler_executable, compiler_entry,
                           memo_cache.path().string());
  flagdnn::Executable reconfigured(memo_handle, graph);
#if defined(__linux__)
  if (reads.take_open_count() != 4) {
    throw std::runtime_error("set_compiler retained dependency content memo");
  }
#endif

  const auto original_time = std::filesystem::last_write_time(dependency);
  const auto replacement = memo_cache.path() / "replacement";
  write_identity_dependency(replacement, "resource-v2");
  std::filesystem::last_write_time(replacement, original_time);
  std::filesystem::rename(replacement, dependency);
  flagdnn::Executable replaced(memo_handle, graph);
  if (count_manifests(memo_cache.path()) != 2) {
    throw std::runtime_error("same-size dependency replacement reused digest");
  }

  std::filesystem::last_write_time(
      dependency, std::filesystem::file_time_type::clock::now() +
                      std::chrono::hours(1));
  std::this_thread::sleep_for(std::chrono::milliseconds(2100));
  flagdnn::Executable future_first(memo_handle, graph);
#if defined(__linux__)
  DependencyReadEvents future_reads(dependency);
#endif
  {
    ScopedEnvironment future_environment("FLAGDNN_CONTRACT_MEMO_PROBE", "future");
    flagdnn::Executable future_refreshed(memo_handle, graph);
#if defined(__linux__)
    if (future_reads.take_open_count() != 4) {
      throw std::runtime_error("future-dated dependency incorrectly reused digest");
    }
#endif
  }
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 3) {
      throw std::invalid_argument("usage: native_backend_contract_smoke "
                                  "COMPILER_EXECUTABLE COMPILER_ENTRY");
    }
    require_native_process_clean("startup");

    require_status([] { flagdnn::Handle invalid("contract", 1); },
                   FLAGDNN_STATUS_INVALID_VALUE,
                   "nonzero contract device ordinal");

    TemporaryCache cache;
    const std::filesystem::path identity_query_counter =
        cache.path() / "identity-query-count";
    const std::filesystem::path identity_retry_marker =
        cache.path() / "identity-retry-marker";
    const std::filesystem::path identity_dependency_target_a =
        cache.path() / "identity-dependency-target-a";
    const std::filesystem::path identity_dependency_target_b =
        cache.path() / "identity-dependency-target-b";
    const std::filesystem::path identity_dependency =
        cache.path() / "identity-dependency";
    write_identity_dependency(identity_dependency_target_a, "resource-v1");
    std::filesystem::create_hard_link(identity_dependency_target_a,
                                      identity_dependency_target_b);
    std::filesystem::create_symlink(identity_dependency_target_a.filename(),
                                    identity_dependency);
    ScopedEnvironment identity_counter_environment(
        "FLAGDNN_CONTRACT_COMPILER_IDENTITY_COUNTER",
        identity_query_counter.string());
    ScopedEnvironment identity_retry_environment(
        "FLAGDNN_CONTRACT_COMPILER_TEMPFAIL_MARKER",
        identity_retry_marker.string());
    ScopedEnvironment identity_dependency_environment(
        "FLAGDNN_CONTRACT_COMPILER_IDENTITY_DEPENDENCY",
        identity_dependency.string());
    flagdnn::Handle handle("contract", 0);
    if (handle.backend_name() != "contract" ||
        handle.target_fingerprint() != "host_contract_v1") {
      throw std::runtime_error("contract backend identity differs");
    }
    handle.set_compiler(argv[1], argv[2], cache.path().string());

    constexpr std::array<std::int64_t, 1> dimensions = {8};
    constexpr std::array<std::int64_t, 1> strides = {1};
    flagdnn::TensorDescriptor input_descriptor(1, FLAGDNN_DATA_FLOAT32,
                                               dimensions, strides);
    flagdnn::TensorDescriptor output_descriptor(2, FLAGDNN_DATA_FLOAT32,
                                                dimensions, strides);
    flagdnn::Graph graph;
    graph.relu(input_descriptor, output_descriptor);
    graph.finalize();

    flagdnn::Executable executable(handle, graph);
    if (executable.operation_count() != 1 ||
        executable.workspace_size() != 64) {
      throw std::runtime_error("contract executable metadata differs");
    }
    require_native_process_clean("after cache miss build");

    std::array<float, 8> input = {-4.0F, -1.5F, -0.0F, 0.25F,
                                  1.0F,  3.5F,  -9.0F, 8.0F};
    std::array<float, 8> output;
    output.fill(-123.0F);
    std::array<flagdnnBinding_t, 2> bindings = {
        flagdnnBinding_t{1, input.data()},
        flagdnnBinding_t{2, output.data()},
    };
    alignas(256) std::array<unsigned char, 64> workspace{};
    int stream_cookie = 7;

    require_status(
        [&] { executable.execute(bindings, nullptr, 0, &stream_cookie); },
        FLAGDNN_STATUS_INVALID_VALUE, "missing contract workspace");
    require_status(
        [&] {
          executable.execute(
              bindings, workspace.data() + 1, workspace.size(), &stream_cookie);
        },
        FLAGDNN_STATUS_INVALID_VALUE,
        "misaligned contract workspace");
    require_status(
        [&] {
          executable.execute(
              bindings, workspace.data(), workspace.size(), nullptr);
        },
        FLAGDNN_STATUS_INVALID_VALUE, "missing contract stream");

    executable.execute(bindings, workspace.data(), workspace.size(),
                       &stream_cookie);
    require_relu(input, output);
    if (workspace.front() != 0x5aU) {
      throw std::runtime_error("contract backend did not use workspace");
    }
    require_native_process_clean("after execute");

    corrupt_manifest(find_single_manifest(cache.path()));
    flagdnn::Executable recovered_executable(handle, graph);
    output.fill(-234.0F);
    workspace.fill(0);
    recovered_executable.execute(bindings, workspace.data(), workspace.size(),
                                 &stream_cookie);
    require_relu(input, output);
    if (workspace.front() != 0x5aU) {
      throw std::runtime_error(
          "rebuilt contract artifact did not use workspace");
    }

    flagdnn::Graph timeout_graph;
    timeout_graph.set_name("contract compiler timeout");
    timeout_graph.relu(input_descriptor, output_descriptor);
    timeout_graph.finalize();
    {
      ScopedEnvironment timeout("FLAGDNN_COMPILER_TIMEOUT_SECONDS", "1");
      ScopedEnvironment delay("FLAGDNN_CONTRACT_COMPILER_DELAY_SECONDS", "2");
      require_status(
          [&] { flagdnn::Executable timed_out(handle, timeout_graph); },
          FLAGDNN_STATUS_COMPILATION_FAILED, "contract compiler timeout");
    }
    require_native_process_clean("after compiler timeout");

    namespace fe = flagdnn_frontend;
    fe::graph::Graph frontend_graph;
    frontend_graph.set_name("contract frontend \"relu\"\n")
        .set_io_data_type(fe::DataType_t::FLOAT)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);
    const auto frontend_input =
        frontend_graph.tensor(fe::graph::Tensor_attributes()
                                  .set_name("x")
                                  .set_uid(21)
                                  .set_dim({8})
                                  .set_stride({1}));
    const auto frontend_output = frontend_graph.pointwise(
        frontend_input,
        fe::graph::Pointwise_attributes().set_name("relu").set_mode(
            fe::PointwiseMode_t::RELU_FWD));
    frontend_output->set_name("y").set_uid(22).set_output(true);

    require_frontend_status(
        frontend_graph.create_execution_plans({fe::HeurMode_t::A}),
        FLAGDNN_STATUS_NOT_INITIALIZED, "out-of-order create_execution_plans");
    require_frontend_status(frontend_graph.validate(), FLAGDNN_STATUS_SUCCESS,
                            "frontend validate");
    require_frontend_status(frontend_graph.build_operation_graph(handle),
                            FLAGDNN_STATUS_SUCCESS,
                            "frontend build_operation_graph");
    require_frontend_status(frontend_graph.create_execution_plans(
                                {fe::HeurMode_t::A, fe::HeurMode_t::FALLBACK}),
                            FLAGDNN_STATUS_SUCCESS,
                            "frontend create_execution_plans");
    require_frontend_status(frontend_graph.check_support(handle),
                            FLAGDNN_STATUS_SUCCESS, "frontend check_support");
    require_frontend_status(frontend_graph.build_plans(handle),
                            FLAGDNN_STATUS_SUCCESS, "frontend build_plans");
    if (!frontend_graph.is_built() ||
        frontend_graph.get_workspace_size() != 64) {
      throw std::runtime_error("frontend staged graph metadata differs");
    }
    std::array<float, 8> frontend_output_values;
    frontend_output_values.fill(-789.0F);
    const std::array<flagdnnBinding_t, 2> frontend_bindings = {
        flagdnnBinding_t{21, input.data()},
        flagdnnBinding_t{22, frontend_output_values.data()}};
    workspace.fill(0);
    require_frontend_status(
        frontend_graph.execute(handle, frontend_bindings, workspace.data(),
                               workspace.size(), &stream_cookie),
        FLAGDNN_STATUS_SUCCESS, "frontend execute");
    require_relu(input, frontend_output_values);

    flagdnn::TensorDescriptor generic_input_descriptor(31, FLAGDNN_DATA_FLOAT32,
                                                       dimensions, strides);
    flagdnn::TensorDescriptor generic_output_descriptor(
        32, FLAGDNN_DATA_FLOAT32, dimensions, strides);
    flagdnn::OperationDescriptor generic_relu("relu");
    generic_relu.set_input("input", generic_input_descriptor);
    generic_relu.set_output("output", generic_output_descriptor);
    generic_relu.set_attribute("integer_value", std::int64_t{7});
    generic_relu.set_attribute("double_value", 0.5);
    generic_relu.set_attribute("boolean_value", true);
    generic_relu.set_attribute("string_value", "generic \"relu\"");
    const std::array<std::int64_t, 2> generic_axes = {0, 1};
    generic_relu.set_attribute("array_value", generic_axes);
    generic_relu.finalize();
    generic_relu.set_name("generic_relu");
    generic_relu.set_compute_data_type(FLAGDNN_DATA_FLOAT32);
    flagdnn::Graph generic_graph;
    generic_graph.set_name("generic_descriptor_contract");
    generic_graph.add(generic_relu);
    generic_graph.finalize();
    flagdnn::Executable generic_executable(handle, generic_graph);
    std::array<float, 8> generic_output;
    generic_output.fill(-654.0F);
    const std::array<flagdnnBinding_t, 2> generic_bindings = {
        flagdnnBinding_t{31, input.data()},
        flagdnnBinding_t{32, generic_output.data()}};
    workspace.fill(0);
    generic_executable.execute(generic_bindings, workspace.data(),
                               workspace.size(), &stream_cookie);
    require_relu(input, generic_output);

    constexpr std::array<std::int64_t, 2> layout_dimensions = {2, 4};
    constexpr std::array<std::int64_t, 2> layout_strides = {4, 1};
    flagdnn::TensorDescriptor layout_input_descriptor(
        41, FLAGDNN_DATA_FLOAT32, layout_dimensions, layout_strides);
    flagdnn::TensorDescriptor layout_output_descriptor(
        42, FLAGDNN_DATA_FLOAT32, layout_dimensions, layout_strides);

    flagdnn::OperationDescriptor view_only_reshape("reshape");
    view_only_reshape.set_input("input", layout_input_descriptor);
    view_only_reshape.set_output("output", layout_output_descriptor);
    view_only_reshape.set_attribute("reshape_mode", std::int64_t{1});
    view_only_reshape.finalize();
    flagdnn::Graph view_only_graph;
    view_only_graph.add(view_only_reshape);
    require_status([&] { view_only_graph.finalize(); },
                   FLAGDNN_STATUS_NOT_SUPPORTED,
                   "VIEW_ONLY reshape without tensor aliasing");

    flagdnn::OperationDescriptor invalid_transpose("transpose");
    invalid_transpose.set_input("input", layout_input_descriptor);
    invalid_transpose.set_output("output", layout_output_descriptor);
    constexpr std::array<std::int64_t, 2> duplicate_permutation = {0, 0};
    invalid_transpose.set_attribute("permutation", duplicate_permutation);
    invalid_transpose.finalize();
    flagdnn::Graph invalid_transpose_graph;
    invalid_transpose_graph.add(invalid_transpose);
    require_status([&] { invalid_transpose_graph.finalize(); },
                   FLAGDNN_STATUS_INVALID_VALUE,
                   "duplicate transpose permutation");

    flagdnn::OperationDescriptor invalid_slice("slice");
    invalid_slice.set_input("input", layout_input_descriptor);
    invalid_slice.set_output("output", layout_output_descriptor);
    constexpr std::array<std::int64_t, 2> slice_starts = {0, 0};
    constexpr std::array<std::int64_t, 2> slice_limits = {2, 4};
    constexpr std::array<std::int64_t, 2> invalid_slice_strides = {1, 0};
    invalid_slice.set_attribute("starts", slice_starts);
    invalid_slice.set_attribute("limits", slice_limits);
    invalid_slice.set_attribute("slice_strides", invalid_slice_strides);
    invalid_slice.finalize();
    flagdnn::Graph invalid_slice_graph;
    invalid_slice_graph.add(invalid_slice);
    require_status([&] { invalid_slice_graph.finalize(); },
                   FLAGDNN_STATUS_INVALID_VALUE, "non-positive slice stride");

    constexpr std::array<std::int64_t, 1> extreme_slice_input_dimensions = {
        2};
    constexpr std::array<std::int64_t, 1> extreme_slice_input_strides = {1};
    constexpr std::array<std::int64_t, 1> extreme_slice_output_dimensions = {
        1};
    constexpr std::array<std::int64_t, 1> extreme_slice_output_strides = {
        std::numeric_limits<std::int64_t>::max()};
    flagdnn::TensorDescriptor extreme_slice_input_descriptor(
        43,
        FLAGDNN_DATA_FLOAT32,
        extreme_slice_input_dimensions,
        extreme_slice_input_strides);
    flagdnn::TensorDescriptor extreme_slice_output_descriptor(
        44,
        FLAGDNN_DATA_FLOAT32,
        extreme_slice_output_dimensions,
        extreme_slice_output_strides);
    flagdnn::OperationDescriptor extreme_slice("slice");
    extreme_slice.set_input("input", extreme_slice_input_descriptor);
    extreme_slice.set_output("output", extreme_slice_output_descriptor);
    constexpr std::array<std::int64_t, 1> extreme_slice_starts = {0};
    constexpr std::array<std::int64_t, 1> extreme_slice_limits = {2};
    constexpr std::array<std::int64_t, 1> extreme_slice_steps = {
        std::numeric_limits<std::int64_t>::max()};
    extreme_slice.set_attribute("starts", extreme_slice_starts);
    extreme_slice.set_attribute("limits", extreme_slice_limits);
    extreme_slice.set_attribute("slice_strides", extreme_slice_steps);
    extreme_slice.finalize();
    flagdnn::Graph extreme_slice_native_graph;
    extreme_slice_native_graph.add(extreme_slice);
    extreme_slice_native_graph.finalize();

    flagdnn::TensorDescriptor chain_input_descriptor(
        11, FLAGDNN_DATA_FLOAT32, dimensions, strides);
    flagdnn::TensorDescriptor chain_intermediate_descriptor(
        12, FLAGDNN_DATA_FLOAT32, dimensions, strides);
    chain_intermediate_descriptor.set_virtual();
    flagdnn::TensorDescriptor chain_output_descriptor(13, FLAGDNN_DATA_FLOAT32,
                                                      dimensions, strides);
    flagdnn::OperationDescriptor chain_consumer(FLAGDNN_OPERATION_RELU);
    chain_consumer.set_relu(chain_intermediate_descriptor,
                            chain_output_descriptor);
    flagdnn::OperationDescriptor chain_producer(FLAGDNN_OPERATION_RELU);
    chain_producer.set_relu(chain_input_descriptor,
                            chain_intermediate_descriptor);
    flagdnn::Graph out_of_order_graph;
    out_of_order_graph.add(chain_consumer);
    out_of_order_graph.add(chain_producer);
    out_of_order_graph.finalize();

    flagdnn::Executable fused_chain(handle, out_of_order_graph);
    if (fused_chain.operation_count() != 2 ||
        fused_chain.workspace_size() != 64) {
      throw std::runtime_error("out-of-order graph metadata differs");
    }
    std::array<float, 8> chain_output;
    chain_output.fill(-456.0F);
    std::array<flagdnnBinding_t, 2> chain_bindings = {
        flagdnnBinding_t{11, input.data()},
        flagdnnBinding_t{13, chain_output.data()},
    };
    workspace.fill(0);
    fused_chain.execute(chain_bindings, workspace.data(), workspace.size(),
                        &stream_cookie);
    require_relu(input, chain_output);
    if (workspace.front() != 0x5aU) {
      throw std::runtime_error("fused contract stage did not use workspace");
    }
    require_native_process_clean("after out-of-order graph execute");
    const std::uint64_t initial_identity_queries =
        read_identity_query_count(identity_query_counter);
    if (initial_identity_queries != 7) {
      throw std::runtime_error(
          "compiler identity retry/publication refresh count differs: " +
          std::to_string(initial_identity_queries));
    }

    const std::size_t manifests_before_dependency_change =
        count_manifests(cache.path());
    const auto original_dependency_time =
        std::filesystem::last_write_time(identity_dependency_target_a);
    write_identity_dependency(identity_dependency_target_a, "resource-v2");
    std::filesystem::last_write_time(identity_dependency_target_a,
                                     original_dependency_time);
    flagdnn::Executable dependency_refreshed_identity(handle, graph);
    if (read_identity_query_count(identity_query_counter) != 9) {
      throw std::runtime_error("in-place compiler dependency change did not "
                               "invalidate identity memo");
    }
    if (count_manifests(cache.path()) !=
        manifests_before_dependency_change + 1) {
      throw std::runtime_error(
          "dependency content change did not create a new identity cache slot");
    }

    const std::size_t manifests_before_symlink_change =
        count_manifests(cache.path());
    const std::filesystem::path replacement_dependency =
        cache.path() / "identity-dependency-replacement";
    std::filesystem::create_symlink(identity_dependency_target_b.filename(),
                                    replacement_dependency);
    std::filesystem::rename(replacement_dependency, identity_dependency);
    flagdnn::Executable symlink_refreshed_identity(handle, graph);
    if (read_identity_query_count(identity_query_counter) != 11) {
      throw std::runtime_error("compiler dependency symlink change did not "
                               "invalidate identity memo");
    }
    if (count_manifests(cache.path()) != manifests_before_symlink_change + 1) {
      throw std::runtime_error(
          "dependency symlink change did not create a new identity cache slot");
    }

    const std::size_t manifests_before_set_compiler =
        count_manifests(cache.path());
    handle.set_compiler(argv[1], argv[2], cache.path().string());
    flagdnn::Executable refreshed_identity(handle, graph);
    if (read_identity_query_count(identity_query_counter) != 12) {
      throw std::runtime_error(
          "set_compiler did not invalidate the compiler identity memo");
    }
    if (count_manifests(cache.path()) != manifests_before_set_compiler) {
      throw std::runtime_error(
          "unchanged compiler identity created a duplicate cache slot");
    }

    const std::size_t manifests_before_incomplete_dependencies =
        count_manifests(cache.path());
    {
      ScopedEnvironment incomplete_dependencies(
          "FLAGDNN_CONTRACT_COMPILER_DEPENDENCIES_INCOMPLETE", "1");
      flagdnn::Executable incomplete_identity_first(handle, graph);
      flagdnn::Executable incomplete_identity_second(handle, graph);
    }
    if (read_identity_query_count(identity_query_counter) != 14) {
      throw std::runtime_error(
          "incomplete compiler dependencies were incorrectly memoized");
    }
    if (count_manifests(cache.path()) !=
        manifests_before_incomplete_dependencies) {
      throw std::runtime_error(
          "incomplete dependency reports changed an unchanged cache identity");
    }

    TemporaryCache reconfigured_cache;
    flagdnn::Graph compiler_transaction_graph;
    compiler_transaction_graph.set_name(
        "contract compiler configuration transaction");
    compiler_transaction_graph.relu(input_descriptor, output_descriptor);
    compiler_transaction_graph.finalize();
    const std::filesystem::path compile_started_marker =
        cache.path() / "compiler-transaction-started";
    const std::filesystem::path compile_finished_marker =
        cache.path() / "compiler-transaction-finished";
    const std::size_t manifests_before_compiler_transaction =
        count_manifests(cache.path());
    std::exception_ptr build_failure;
    {
      ScopedEnvironment started_marker_environment(
          "FLAGDNN_CONTRACT_COMPILER_COMPILE_STARTED_MARKER",
          compile_started_marker.string());
      ScopedEnvironment finished_marker_environment(
          "FLAGDNN_CONTRACT_COMPILER_COMPILE_FINISHED_MARKER",
          compile_finished_marker.string());
      ScopedEnvironment compile_only_delay_environment(
          "FLAGDNN_CONTRACT_COMPILER_DELAY_COMPILE_ONLY", "1");
      ScopedEnvironment delay_environment(
          "FLAGDNN_CONTRACT_COMPILER_DELAY_SECONDS", "2");

      std::thread build_worker([&] {
        try {
          flagdnn::Executable concurrent_build(handle,
                                               compiler_transaction_graph);
        } catch (...) {
          build_failure = std::current_exception();
        }
      });
      try {
        wait_for_regular_file(compile_started_marker, std::chrono::seconds(5));
      } catch (...) {
        build_worker.join();
        if (build_failure) {
          std::rethrow_exception(build_failure);
        }
        throw;
      }

      std::exception_ptr set_compiler_failure;
      bool compile_finished_when_set_compiler_returned = false;
      try {
        handle.set_compiler(argv[1], argv[2],
                            reconfigured_cache.path().string());
        std::error_code marker_error;
        compile_finished_when_set_compiler_returned =
            std::filesystem::is_regular_file(compile_finished_marker,
                                             marker_error) &&
            !marker_error;
      } catch (...) {
        set_compiler_failure = std::current_exception();
      }
      build_worker.join();
      if (build_failure) {
        std::rethrow_exception(build_failure);
      }
      if (set_compiler_failure) {
        std::rethrow_exception(set_compiler_failure);
      }
      if (!compile_finished_when_set_compiler_returned) {
        throw std::runtime_error(
            "set_compiler returned before the compiler transaction finished");
      }
    }

    if (read_text_file(compile_started_marker) !=
        read_text_file(compile_finished_marker)) {
      throw std::runtime_error(
          "compiler transaction markers refer to different output paths");
    }
    const std::filesystem::path old_temporary_artifact =
        read_text_file(compile_started_marker);
    const std::filesystem::path old_relative_artifact =
        old_temporary_artifact.lexically_relative(cache.path());
    if (old_relative_artifact.empty() || old_relative_artifact.is_absolute() ||
        *old_relative_artifact.begin() == "..") {
      throw std::runtime_error(
          "in-flight compiler transaction escaped its original cache");
    }
    if (count_manifests(cache.path()) !=
        manifests_before_compiler_transaction + 1) {
      throw std::runtime_error("compiler transaction did not publish exactly "
                               "one old-cache artifact");
    }
    if (count_manifests(reconfigured_cache.path()) != 0) {
      throw std::runtime_error(
          "in-flight compiler transaction leaked into the new cache");
    }
    const std::filesystem::path old_graph_cache =
        old_temporary_artifact.parent_path();
    const std::string old_transaction_identity =
        require_cache_identity_consistent(
            find_single_manifest(old_graph_cache));

    write_identity_dependency(identity_dependency_target_b, "resource-v3");
    flagdnn::Executable reconfigured_build(handle, compiler_transaction_graph);
    if (count_manifests(reconfigured_cache.path()) != 1) {
      throw std::runtime_error(
          "reconfigured build did not publish exactly one new-cache artifact");
    }
    const std::string new_transaction_identity =
        require_cache_identity_consistent(
            find_single_manifest(reconfigured_cache.path()));
    if (new_transaction_identity == old_transaction_identity) {
      throw std::runtime_error(
          "compiler dependency change reused the old transaction identity");
    }

    require_legacy_identity_protocol(argv[1], argv[2], graph, "digest-only");
    require_legacy_identity_protocol(argv[1], argv[2], graph, "files-only");
    require_dependency_content_memo(argv[1], argv[2], graph);

    const std::size_t manifests_before_identity_failures =
        count_manifests(cache.path());
    struct IdentityFailureCase {
      const char *mode;
      const char *description;
      std::uint64_t expected_queries;
    };
    constexpr std::array<IdentityFailureCase, 3> identity_failure_cases = {{
        {"malformed", "malformed compiler identity", 1},
        {"nonzero", "nonzero compiler identity process", 1},
        {"temporary", "persistent temporary compiler identity failure", 3},
    }};
    for (const IdentityFailureCase &failure_case : identity_failure_cases) {
      handle.set_compiler(argv[1], argv[2], cache.path().string());
      const std::uint64_t queries_before =
          read_identity_query_count(identity_query_counter);
      {
        ScopedEnvironment failure_mode(
            "FLAGDNN_CONTRACT_COMPILER_IDENTITY_FAILURE_MODE",
            failure_case.mode);
        require_status(
            [&] {
              flagdnn::Executable must_not_use_stale_cache(handle, graph);
            },
            FLAGDNN_STATUS_COMPILATION_FAILED, failure_case.description);
      }
      const std::uint64_t queries_after =
          read_identity_query_count(identity_query_counter);
      if (queries_after - queries_before != failure_case.expected_queries) {
        throw std::runtime_error(
            std::string(failure_case.description) +
            " did not execute the expected number of identity queries");
      }
      if (count_manifests(cache.path()) != manifests_before_identity_failures) {
        throw std::runtime_error(
            std::string(failure_case.description) +
            " changed the artifact cache while failing closed");
      }
    }

    flagdnn::Graph identity_publish_race_graph;
    identity_publish_race_graph.set_name(
        "contract compiler identity publish race");
    identity_publish_race_graph.relu(input_descriptor, output_descriptor);
    identity_publish_race_graph.finalize();
    const std::filesystem::path identity_race_started =
        cache.path() / "identity-race-started";
    const std::filesystem::path identity_race_finished =
        cache.path() / "identity-race-finished";
    const std::filesystem::path identity_race_validated =
        cache.path() / "identity-race-validated";
    const std::size_t manifests_before_identity_race =
        count_manifests(cache.path());
    std::exception_ptr identity_race_failure;
    {
      ScopedEnvironment started_marker_environment(
          "FLAGDNN_CONTRACT_COMPILER_COMPILE_STARTED_MARKER",
          identity_race_started.string());
      ScopedEnvironment finished_marker_environment(
          "FLAGDNN_CONTRACT_COMPILER_COMPILE_FINISHED_MARKER",
          identity_race_finished.string());
      ScopedEnvironment validated_marker_environment(
          "FLAGDNN_CONTRACT_COMPILER_COMPILE_VALIDATED_MARKER",
          identity_race_validated.string());
      ScopedEnvironment post_validate_delay_environment(
          "FLAGDNN_CONTRACT_COMPILER_DELAY_AFTER_VALIDATE_SECONDS", "2");

      std::thread build_worker([&] {
        try {
          flagdnn::Executable raced_build(handle, identity_publish_race_graph);
        } catch (...) {
          identity_race_failure = std::current_exception();
        }
      });
      try {
        wait_for_regular_file(identity_race_validated, std::chrono::seconds(5));
        write_identity_dependency(identity_dependency_target_b,
                                  "resource-during-compile");
      } catch (...) {
        build_worker.join();
        throw;
      }
      build_worker.join();
    }
    if (!identity_race_failure) {
      throw std::runtime_error(
          "identity dependency race unexpectedly published an artifact");
    }
    try {
      std::rethrow_exception(identity_race_failure);
    } catch (const flagdnn::Error &error) {
      if (error.status() != FLAGDNN_STATUS_COMPILATION_FAILED ||
          std::string_view(error.what())
                  .find("identity changed during artifact compilation") ==
              std::string_view::npos) {
        throw std::runtime_error(
            "identity dependency race returned the wrong failure");
      }
    }
    if (read_text_file(identity_race_started) !=
            read_text_file(identity_race_validated) ||
        read_text_file(identity_race_started) !=
            read_text_file(identity_race_finished)) {
      throw std::runtime_error(
          "identity dependency race markers refer to different artifacts");
    }
    const std::filesystem::path raced_temporary_artifact =
        read_text_file(identity_race_started);
    if (std::filesystem::exists(raced_temporary_artifact) ||
        std::filesystem::exists(raced_temporary_artifact.parent_path() /
                                "active_identity") ||
        count_manifests(cache.path()) != manifests_before_identity_race) {
      throw std::runtime_error(
          "identity dependency race changed the artifact cache while "
          "failing closed");
    }

    // A backend JIT may temporarily publish process environment while it
    // creates an executable. A compiler subprocess for another handle must not
    // inherit that private build-time value. The release worker deliberately
    // does not call FlagDNN, so it cannot be blocked by the build guard.
    TemporaryCache environment_guard_cache_a;
    TemporaryCache environment_guard_cache_b;
    flagdnn::Handle environment_guard_handle_a("contract", 0);
    flagdnn::Handle environment_guard_handle_b("contract", 0);
    environment_guard_handle_a.set_compiler(
        argv[1], argv[2], environment_guard_cache_a.path().string());
    environment_guard_handle_b.set_compiler(
        argv[1], argv[2], environment_guard_cache_b.path().string());
    flagdnn::Graph environment_guard_graph_a;
    environment_guard_graph_a.set_name("contract backend environment guard");
    environment_guard_graph_a.relu(input_descriptor, output_descriptor);
    environment_guard_graph_a.finalize();
    flagdnn::Graph environment_guard_graph_b;
    environment_guard_graph_b.set_name("contract compiler environment probe");
    environment_guard_graph_b.relu(input_descriptor, output_descriptor);
    environment_guard_graph_b.finalize();
    const std::filesystem::path backend_entered =
        environment_guard_cache_a.path() / "backend-entered";
    const std::filesystem::path backend_release =
        environment_guard_cache_a.path() / "backend-release";
    const std::filesystem::path compiler_observed =
        environment_guard_cache_b.path() / "compiler-observed";
    constexpr const char *temporary_variable =
        "FLAGDNN_CONTRACT_BACKEND_PRIVATE_BUILD_VALUE";
    constexpr const char *temporary_value = "temporary-backend-value";
    std::exception_ptr environment_guard_failure_a;
    std::exception_ptr environment_guard_failure_b;
    std::exception_ptr environment_guard_handle_failure;
    std::atomic<bool> environment_guard_build_b_completed = false;
    std::atomic<bool> environment_guard_handle_completed = false;
    std::barrier environment_guard_workers_ready(3);
    {
      ScopedEnvironment backend_entered_environment(
          "FLAGDNN_CONTRACT_BACKEND_ENVIRONMENT_ENTERED_MARKER",
          backend_entered.string());
      ScopedEnvironment backend_release_environment(
          "FLAGDNN_CONTRACT_BACKEND_ENVIRONMENT_RELEASE_MARKER",
          backend_release.string());
      ScopedEnvironment backend_variable_environment(
          "FLAGDNN_CONTRACT_BACKEND_ENVIRONMENT_VARIABLE", temporary_variable);
      ScopedEnvironment backend_value_environment(
          "FLAGDNN_CONTRACT_BACKEND_ENVIRONMENT_VALUE", temporary_value);
      ScopedEnvironment backend_path_environment(
          "FLAGDNN_CONTRACT_BACKEND_ENVIRONMENT_BACKEND_PATH",
          "/flagdnn/backend-private-missing-path");
      ScopedEnvironment compiler_marker_environment(
          "FLAGDNN_CONTRACT_COMPILER_INHERITED_ENVIRONMENT_MARKER",
          compiler_observed.string());
      ScopedEnvironment compiler_variable_environment(
          "FLAGDNN_CONTRACT_COMPILER_INHERITED_ENVIRONMENT_VARIABLE",
          temporary_variable);
      ScopedEnvironment compiler_value_environment(
          "FLAGDNN_CONTRACT_COMPILER_INHERITED_ENVIRONMENT_VALUE",
          temporary_value);

      std::thread build_a([&] {
        try {
          flagdnn::Executable executable_a(environment_guard_handle_a,
                                           environment_guard_graph_a);
        } catch (...) {
          environment_guard_failure_a = std::current_exception();
        }
      });
      try {
        wait_for_regular_file(backend_entered, std::chrono::seconds(5));
      } catch (...) {
        write_identity_dependency(backend_release, "release\n");
        build_a.join();
        throw;
      }

      std::thread build_b([&] {
        environment_guard_workers_ready.arrive_and_wait();
        try {
          flagdnn::Executable executable_b(environment_guard_handle_b,
                                           environment_guard_graph_b);
        } catch (...) {
          environment_guard_failure_b = std::current_exception();
        }
        environment_guard_build_b_completed.store(true,
                                                  std::memory_order_release);
      });
      std::thread create_handle([&] {
        environment_guard_workers_ready.arrive_and_wait();
        try {
          flagdnn::Handle guarded_handle("contract", 0);
        } catch (...) {
          environment_guard_handle_failure = std::current_exception();
        }
        environment_guard_handle_completed.store(true,
                                                 std::memory_order_release);
      });
      // Do not start the observation window until both workers have reached
      // their FlagDNN API calls.  The backend temporarily publishes an invalid
      // execution engine as well as a compiler-observable private value, so an
      // unguarded constructor or build must become observable before release.
      environment_guard_workers_ready.arrive_and_wait();
      const auto observation_deadline =
          std::chrono::steady_clock::now() + std::chrono::seconds(2);
      while (
          std::chrono::steady_clock::now() < observation_deadline &&
          !environment_guard_build_b_completed.load(
              std::memory_order_acquire) &&
          !environment_guard_handle_completed.load(std::memory_order_acquire) &&
          !std::filesystem::exists(compiler_observed)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
      const bool worker_completed_before_release =
          environment_guard_build_b_completed.load(std::memory_order_acquire) ||
          environment_guard_handle_completed.load(std::memory_order_acquire);
      write_identity_dependency(backend_release, "release\n");
      build_a.join();
      build_b.join();
      create_handle.join();
      if (worker_completed_before_release) {
        throw std::runtime_error(
            "FlagDNN API call completed while a backend-private build "
            "environment was active");
      }
    }
    if (environment_guard_failure_a) {
      std::rethrow_exception(environment_guard_failure_a);
    }
    if (environment_guard_failure_b) {
      std::rethrow_exception(environment_guard_failure_b);
    }
    if (environment_guard_handle_failure) {
      std::rethrow_exception(environment_guard_handle_failure);
    }
    if (std::filesystem::exists(compiler_observed)) {
      throw std::runtime_error(
          "compiler subprocess inherited a backend-private build value");
    }

    // A graph compiled for one execution engine must never be treated as an
    // offline cache hit for another engine. The contract compiler intentionally
    // supports external_artifact only, so the missing compiler below proves
    // that the libtriton_jit cache namespace starts empty.
    {
      ScopedEnvironment alternate_engine("FLAGDNN_EXECUTION_ENGINE",
                                         "libtriton_jit");
      flagdnn::Handle alternate_engine_handle("contract", 0);
      alternate_engine_handle.set_compiler(
          "/flagdnn/missing-compiler-executable", "/flagdnn/missing-compiler",
          cache.path().string());
      require_status(
          [&] {
            flagdnn::Executable must_not_cross_engine_cache(
                alternate_engine_handle, graph);
          },
          FLAGDNN_STATUS_COMPILATION_FAILED,
          "cross-execution-engine offline cache reuse");
    }

    handle.set_compiler("/flagdnn/missing-compiler-executable",
                        "/flagdnn/missing-compiler", cache.path().string());
    flagdnn::Executable cache_hit(handle, graph);
    output.fill(-321.0F);
    cache_hit.execute(bindings, workspace.data(), workspace.size(),
                      &stream_cookie);
    require_relu(input, output);
    require_native_process_clean("after cache hit execute");

    std::cout << "PASS backend=contract target=host_contract_v1 "
                 "operation=relu graph=toposort stage_fusion "
                 "cache=miss+recovery+hit compiler_timeout=pass "
                 "compiler_config_transaction=pass "
                 "identity_protocol=digest-only+files-only "
                 "identity_failure=fail-closed "
                 "identity_publish_race=fail-closed "
                 "build_environment_guard=pass "
                 "engine_cache_isolation=pass offline_cache=pass\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "CONTRACT_BACKEND_FAILED: " << error.what() << '\n';
    return 1;
  }
}
