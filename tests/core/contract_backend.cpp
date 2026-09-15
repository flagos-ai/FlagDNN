/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

/*
 * Test-only backend used to prove that the native Core, artifact cache and
 * private plugin ABI can host a non-CUDA platform. It is never installed and
 * intentionally supports only one FP32 contiguous ReLU contract case.
 */

#include "backends/backend_api.h"
#include "runtime/json.hpp"
#include "runtime/sha256.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <new>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#if defined(_WIN32)
#define FLAGDNN_BACKEND_EXPORT __declspec(dllexport)
#else
#define FLAGDNN_BACKEND_EXPORT __attribute__((visibility("default")))
#endif

namespace {

constexpr std::string_view kBackendName = "contract";
constexpr std::string_view kTargetFingerprint = "host_contract_v1";
constexpr std::size_t kWorkspaceSize = 64;

thread_local std::string last_error;

#if defined(FLAGDNN_CONTRACT_PREPARED_ENVIRONMENT)
std::atomic<bool> environment_prepared{false};
std::atomic<unsigned int> concurrent_builds{0};
#endif

void write_marker(const std::filesystem::path &path, std::string_view value) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  output.write(value.data(), static_cast<std::streamsize>(value.size()));
  if (!output) {
    throw std::runtime_error("cannot write contract backend marker: " +
                             path.string());
  }
}

void exercise_build_environment_guard() {
  const char *entered =
      std::getenv("FLAGDNN_CONTRACT_BACKEND_ENVIRONMENT_ENTERED_MARKER");
  const char *release =
      std::getenv("FLAGDNN_CONTRACT_BACKEND_ENVIRONMENT_RELEASE_MARKER");
  const char *variable =
      std::getenv("FLAGDNN_CONTRACT_BACKEND_ENVIRONMENT_VARIABLE");
  const char *value = std::getenv("FLAGDNN_CONTRACT_BACKEND_ENVIRONMENT_VALUE");
  const char *forbidden_backend_path =
      std::getenv("FLAGDNN_CONTRACT_BACKEND_ENVIRONMENT_BACKEND_PATH");
  if (entered == nullptr && release == nullptr && variable == nullptr &&
      value == nullptr && forbidden_backend_path == nullptr) {
    return;
  }
  if (entered == nullptr || entered[0] == '\0' || release == nullptr ||
      release[0] == '\0' || variable == nullptr || variable[0] == '\0' ||
      value == nullptr || value[0] == '\0' ||
      forbidden_backend_path == nullptr || forbidden_backend_path[0] == '\0') {
    throw std::runtime_error(
        "contract backend environment-guard hook is incomplete");
  }

  // setenv may invalidate every pointer previously returned by getenv, even
  // for a different key. Own all hook values before publishing any temporary
  // process environment below.
  const std::string entered_value(entered);
  const std::string release_value(release);
  const std::string variable_value(variable);
  const std::string replacement_value(value);
  const std::string forbidden_backend_path_value(forbidden_backend_path);

  const std::filesystem::path entered_path(entered_value);
  const std::filesystem::path release_path(release_value);
  const char *previous = std::getenv(variable_value.c_str());
  const char *previous_backend_path = std::getenv("FLAGDNN_BACKEND_PATH");
  const char *previous_execution_engine =
      std::getenv("FLAGDNN_EXECUTION_ENGINE");
  const bool had_previous = previous != nullptr;
  const bool had_previous_backend_path = previous_backend_path != nullptr;
  const bool had_previous_execution_engine =
      previous_execution_engine != nullptr;
  const std::string previous_value = had_previous ? previous : "";
  const std::string previous_backend_path_value =
      had_previous_backend_path ? previous_backend_path : "";
  const std::string previous_execution_engine_value =
      had_previous_execution_engine ? previous_execution_engine : "";
  const auto restore_environment = [&] {
    if (had_previous) {
      (void)setenv(variable_value.c_str(), previous_value.c_str(), 1);
    } else {
      (void)unsetenv(variable_value.c_str());
    }
    if (had_previous_backend_path) {
      (void)setenv("FLAGDNN_BACKEND_PATH", previous_backend_path_value.c_str(),
                   1);
    } else {
      (void)unsetenv("FLAGDNN_BACKEND_PATH");
    }
    if (had_previous_execution_engine) {
      (void)setenv("FLAGDNN_EXECUTION_ENGINE",
                   previous_execution_engine_value.c_str(), 1);
    } else {
      (void)unsetenv("FLAGDNN_EXECUTION_ENGINE");
    }
  };
  if (setenv(variable_value.c_str(), replacement_value.c_str(), 1) != 0 ||
      setenv("FLAGDNN_BACKEND_PATH", forbidden_backend_path_value.c_str(), 1) !=
          0 ||
      setenv("FLAGDNN_EXECUTION_ENGINE",
             "invalid-contract-backend-private-engine", 1) != 0) {
    restore_environment();
    throw std::runtime_error(
        "contract backend cannot publish temporary build environment");
  }
  try {
    write_marker(entered_path, "entered\n");
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!std::filesystem::is_regular_file(release_path)) {
      if (std::chrono::steady_clock::now() >= deadline) {
        throw std::runtime_error(
            "timed out waiting for contract backend environment release");
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  } catch (...) {
    restore_environment();
    throw;
  }
  restore_environment();
}

class PluginFailure final : public std::runtime_error {
public:
  PluginFailure(flagdnnBackendResult_t result, std::string message)
      : std::runtime_error(std::move(message)), result_(result) {}

  [[nodiscard]] flagdnnBackendResult_t result() const noexcept {
    return result_;
  }

private:
  flagdnnBackendResult_t result_;
};

[[noreturn]] void fail(flagdnnBackendResult_t result, std::string message) {
  throw PluginFailure(result, std::move(message));
}

void require(bool condition, const char *message) {
  if (!condition) {
    fail(FLAGDNN_BACKEND_RESULT_INVALID_VALUE, message);
  }
}

template <typename Function>
flagdnnBackendResult_t plugin_call(Function &&function) noexcept {
  last_error.clear();
  try {
    function();
    return FLAGDNN_BACKEND_RESULT_SUCCESS;
  } catch (const PluginFailure &error) {
    last_error = error.what();
    return error.result();
  } catch (const std::bad_alloc &) {
    last_error = "allocation failed";
    return FLAGDNN_BACKEND_RESULT_ALLOC_FAILED;
  } catch (const std::exception &error) {
    last_error = error.what();
    return FLAGDNN_BACKEND_RESULT_INTERNAL_ERROR;
  } catch (...) {
    last_error = "unknown contract backend failure";
    return FLAGDNN_BACKEND_RESULT_INTERNAL_ERROR;
  }
}

#if defined(FLAGDNN_CONTRACT_PREPARED_ENVIRONMENT)
int is_environment_prepared(const char*) noexcept {
  return environment_prepared.load(std::memory_order_acquire) ? 1 : 0;
}

flagdnnBackendResult_t prepare_environment(
    void*, const char*, const flagdnnBackendBuildInputV2*) noexcept {
  return plugin_call([] {
    const char *mode = std::getenv("FLAGDNN_CONTRACT_PREPARATION");
    if (mode != nullptr && std::string_view(mode) == "fail") {
      fail(FLAGDNN_BACKEND_RESULT_INTERNAL_ERROR,
           "requested preparation failure");
    }
    if (mode == nullptr || std::string_view(mode) != "not_ready") {
      environment_prepared.store(true, std::memory_order_release);
    }
  });
}

void require_concurrent_builds() {
  require(is_environment_prepared(nullptr) != 0, "environment was not prepared");
  if (std::getenv("FLAGDNN_CONTRACT_PARALLEL_BUILDS") == nullptr) {
    return;
  }
  concurrent_builds.fetch_add(1, std::memory_order_acq_rel);
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (concurrent_builds.load(std::memory_order_acquire) < 2) {
    if (std::chrono::steady_clock::now() >= deadline) {
      fail(FLAGDNN_BACKEND_RESULT_INTERNAL_ERROR,
           "prepared builds were serialized");
    }
    std::this_thread::yield();
  }
}
#endif

std::string read_file(const std::filesystem::path &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    fail(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
         "cannot open artifact file: " + path.string());
  }
  std::ostringstream output;
  output << input.rdbuf();
  if (!input.good() && !input.eof()) {
    fail(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
         "cannot read artifact file: " + path.string());
  }
  return output.str();
}

std::size_t checked_size(std::int64_t value, const char *name) {
  if (value < 0 || static_cast<std::uint64_t>(value) >
                       std::numeric_limits<std::size_t>::max()) {
    fail(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
         std::string(name) + " is outside size_t range");
  }
  return static_cast<std::size_t>(value);
}

struct TensorMetadata {
  std::int64_t uid = 0;
  std::size_t element_count = 0;
  bool is_virtual = false;
};

TensorMetadata parse_tensor(const flagdnn::native::json::Value &value) {
  const auto &object = value.as_object();
  (void)object;
  if (value.at("data_type").as_string() != "float32") {
    fail(FLAGDNN_BACKEND_RESULT_NOT_SUPPORTED,
         "contract backend only supports float32");
  }
  const bool is_virtual = value.at("virtual").as_bool();
  const std::int64_t uid = value.at("uid").as_int();
  if (uid <= 0) {
    fail(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
         "contract tensor UID must be positive");
  }
  const auto &dimensions = value.at("dimensions").as_array();
  const auto &strides = value.at("strides").as_array();
  if (dimensions.empty() || dimensions.size() != strides.size()) {
    fail(FLAGDNN_BACKEND_RESULT_NOT_SUPPORTED,
         "contract backend requires a positive-rank tensor");
  }

  std::size_t element_count = 1;
  std::size_t expected_stride = 1;
  for (std::size_t index = dimensions.size(); index != 0; --index) {
    const std::size_t current = index - 1;
    const std::size_t dimension =
        checked_size(dimensions[current].as_int(), "tensor dimension");
    const std::size_t stride =
        checked_size(strides[current].as_int(), "tensor stride");
    if (dimension == 0 || stride != expected_stride ||
        element_count > std::numeric_limits<std::size_t>::max() / dimension) {
      fail(FLAGDNN_BACKEND_RESULT_NOT_SUPPORTED,
           "contract backend requires a contiguous nonempty tensor");
    }
    element_count *= dimension;
    expected_stride = element_count;
  }
  return {uid, element_count, is_virtual};
}

std::int64_t parse_port_uid(const flagdnn::native::json::Value &value,
                            std::string_view expected_name) {
  if (value.at("name").as_string() != expected_name) {
    fail(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
         "contract graph port name is invalid");
  }
  const std::int64_t uid = value.at("uid").as_int();
  if (uid <= 0) {
    fail(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
         "contract graph port UID is invalid");
  }
  return uid;
}

const flagdnn::native::json::Value &
find_tensor(const flagdnn::native::json::Value::Array &tensors,
            std::int64_t uid) {
  for (const auto &tensor : tensors) {
    if (tensor.at("uid").as_int() == uid) {
      return tensor;
    }
  }
  fail(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
       "contract graph port references an unknown tensor UID");
}

struct ContractContext {
  explicit ContractContext(std::int32_t ordinal) : device_ordinal(ordinal) {}
  std::int32_t device_ordinal = 0;
};

class ContractExecutable {
public:
  explicit ContractExecutable(const flagdnnBackendBuildInputV2 &input) {
    require(input.graph_ir != nullptr && input.graph_ir_size != 0,
            "build request is empty");
    require(input.artifact_directory != nullptr, "artifact directory is null");
    require(input.request_sha256 != nullptr, "request SHA-256 is null");

    const std::string_view build_request(
        static_cast<const char *>(input.graph_ir), input.graph_ir_size);
    if (flagdnn::native::sha256(build_request) != input.request_sha256) {
      fail(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
           "Core request SHA-256 does not match build request");
    }

    const auto root = flagdnn::native::json::parse(build_request);
    if (root.at("schema_version").as_int() != 3 ||
        root.at("backend").as_string() != kBackendName ||
        root.at("target").as_string() != kTargetFingerprint) {
      fail(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
           "graph IR backend identity is invalid");
    }
    const std::string compiler_identity =
        root.at("compiler_identity").as_string();
    if (compiler_identity.size() != 64) {
      fail(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
           "build request compiler identity is invalid");
    }

    const auto &graph = root.at("graph");
    if (graph.at("name").as_string() == "contract backend environment guard") {
      exercise_build_environment_guard();
    }
    const auto &graph_nodes = graph.at("nodes").as_array();
    const auto &tensors = graph.at("tensors").as_array();
    if (graph_nodes.empty() || graph_nodes.size() > 2 ||
        graph.at("node_count").as_int() !=
            static_cast<std::int64_t>(graph_nodes.size()) ||
        graph.at("tensor_count").as_int() !=
            static_cast<std::int64_t>(tensors.size())) {
      fail(FLAGDNN_BACKEND_RESULT_NOT_SUPPORTED,
           "contract backend supports one or two graph nodes");
    }

    std::vector<std::int64_t> produced_uids;
    for (const auto &node : graph_nodes) {
      if (node.at("type").as_string() != "relu") {
        fail(FLAGDNN_BACKEND_RESULT_NOT_SUPPORTED,
             "contract backend supports only ReLU");
      }
      const auto &inputs = node.at("inputs").as_array();
      const auto &outputs = node.at("outputs").as_array();
      if (inputs.size() != 1 || outputs.size() != 1) {
        fail(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
             "contract ReLU port count is invalid");
      }
      const std::int64_t input_uid = parse_port_uid(inputs[0], "input");
      const std::int64_t output_uid = parse_port_uid(outputs[0], "output");
      const TensorMetadata input_tensor =
          parse_tensor(find_tensor(tensors, input_uid));
      const TensorMetadata output_tensor =
          parse_tensor(find_tensor(tensors, output_uid));
      if (input_tensor.element_count != output_tensor.element_count) {
        fail(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
             "contract ReLU tensor sizes differ");
      }
      if (input_tensor.is_virtual &&
          std::find(produced_uids.begin(), produced_uids.end(), input_uid) ==
              produced_uids.end()) {
        fail(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
             "contract graph nodes are not topologically ordered");
      }
      if (std::find(produced_uids.begin(), produced_uids.end(), output_uid) !=
          produced_uids.end()) {
        fail(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
             "contract graph tensor has duplicate producers");
      }
      produced_uids.push_back(output_uid);
      nodes_.push_back({input_uid, output_uid, input_tensor.element_count});
      append_binding_uid(input_tensor);
      append_binding_uid(output_tensor);
      if (output_tensor.is_virtual) {
        if (!virtual_offsets_.empty() ||
            output_tensor.element_count * sizeof(float) >
                kWorkspaceSize - kVirtualTensorOffset) {
          fail(FLAGDNN_BACKEND_RESULT_NOT_SUPPORTED,
               "contract virtual tensor workspace is unsupported");
        }
        virtual_offsets_.push_back({output_uid, kVirtualTensorOffset});
      }
    }
    if (binding_uids_.size() != 2) {
      fail(FLAGDNN_BACKEND_RESULT_NOT_SUPPORTED,
           "contract graph must expose exactly two bindings");
    }

    const std::filesystem::path manifest_path =
        std::filesystem::path(input.artifact_directory) / "manifest.json";
    const auto manifest =
        flagdnn::native::json::parse(read_file(manifest_path));
    const auto &program = manifest.at("program");
    const auto &stages = program.at("stages").as_array();
    const std::string expected_kind =
        nodes_.size() == 1 ? "contract_relu" : "contract_relu_chain";
    const auto &source_nodes = stages[0].at("source_node_ids").as_array();
    if (manifest.at("schema_version").as_int() != 3 ||
        manifest.at("artifact_kind").as_string() !=
            "flagdnn_execution_program" ||
        manifest.at("backend").as_string() != kBackendName ||
        manifest.at("target").as_string() != kTargetFingerprint ||
        manifest.at("request_sha256").as_string() != input.request_sha256 ||
        manifest.at("compiler").at("identity_sha256").as_string() !=
            compiler_identity ||
        manifest.at("graph_node_count").as_int() !=
            static_cast<std::int64_t>(nodes_.size()) ||
        checked_size(manifest.at("workspace_size").as_int(),
                     "workspace size") != kWorkspaceSize ||
        program.at("schema_version").as_int() != 1 ||
        program.at("stage_count").as_int() != 1 || stages.size() != 1 ||
        stages[0].at("stage_id").as_int() != 0 ||
        stages[0].at("kind").as_string() != expected_kind ||
        source_nodes.size() != nodes_.size()) {
      fail(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
           "contract execution program manifest is invalid");
    }
  }

  void execute(void *native_stream, const flagdnnBackendBindingV2 bindings[],
               std::size_t binding_count, void *workspace,
               std::size_t workspace_size) const {
    require(native_stream != nullptr, "native stream is null");
    require(bindings != nullptr && binding_count == binding_uids_.size(),
            "contract binding count is invalid");
    require(workspace != nullptr && workspace_size >= kWorkspaceSize,
            "contract workspace is too small");

    for (std::size_t index = 0; index < binding_count; ++index) {
      require(bindings[index].device_pointer != nullptr,
              "binding device pointer is null");
      require(std::find(binding_uids_.begin(), binding_uids_.end(),
                        bindings[index].uid) != binding_uids_.end(),
              "unexpected binding UID");
      for (std::size_t previous = 0; previous < index; ++previous) {
        require(bindings[previous].uid != bindings[index].uid,
                "binding UID is duplicated");
      }
    }

    static_cast<unsigned char *>(workspace)[0] = 0x5aU;
    for (const NodeSpec &node : nodes_) {
      const float *input_pointer = static_cast<const float *>(
          resolve_pointer(node.input_uid, bindings, binding_count, workspace));
      float *output_pointer = static_cast<float *>(
          resolve_pointer(node.output_uid, bindings, binding_count, workspace));
      for (std::size_t index = 0; index < node.element_count; ++index) {
        output_pointer[index] = std::max(input_pointer[index], 0.0F);
      }
    }
  }

private:
  static constexpr std::size_t kVirtualTensorOffset = 16;

  struct NodeSpec {
    std::int64_t input_uid = 0;
    std::int64_t output_uid = 0;
    std::size_t element_count = 0;
  };

  void append_binding_uid(const TensorMetadata &tensor) {
    if (!tensor.is_virtual &&
        std::find(binding_uids_.begin(), binding_uids_.end(), tensor.uid) ==
            binding_uids_.end()) {
      binding_uids_.push_back(tensor.uid);
    }
  }

  void *resolve_pointer(std::int64_t uid,
                        const flagdnnBackendBindingV2 bindings[],
                        std::size_t binding_count, void *workspace) const {
    for (std::size_t index = 0; index < binding_count; ++index) {
      if (bindings[index].uid == uid) {
        return bindings[index].device_pointer;
      }
    }
    for (const auto &[virtual_uid, offset] : virtual_offsets_) {
      if (virtual_uid == uid) {
        return static_cast<unsigned char *>(workspace) + offset;
      }
    }
    fail(FLAGDNN_BACKEND_RESULT_INVALID_VALUE,
         "required contract tensor pointer is unavailable");
  }

  std::vector<NodeSpec> nodes_;
  std::vector<std::int64_t> binding_uids_;
  std::vector<std::pair<std::int64_t, std::size_t>> virtual_offsets_;
};

const char *get_last_error() noexcept { return last_error.c_str(); }

flagdnnBackendResult_t create_context(std::int32_t device_ordinal,
                                      void **context) noexcept {
  return plugin_call([&] {
    require(context != nullptr, "context output pointer is null");
    *context = nullptr;
    require(device_ordinal == 0,
            "contract backend only exposes device ordinal zero");
    std::unique_ptr<ContractContext> result =
        std::make_unique<ContractContext>(device_ordinal);
    *context = result.release();
  });
}

void destroy_context(void *context) noexcept {
  delete static_cast<ContractContext *>(context);
}

flagdnnBackendResult_t
get_target_fingerprint(void *context, char *buffer, std::size_t buffer_size,
                       std::size_t *required_size) noexcept {
  return plugin_call([&] {
    require(context != nullptr, "context is null");
    require(required_size != nullptr, "required size output is null");
    *required_size = kTargetFingerprint.size() + 1;
    require(buffer != nullptr && buffer_size >= *required_size,
            "target fingerprint buffer is too small");
    std::memcpy(buffer, kTargetFingerprint.data(), kTargetFingerprint.size());
    buffer[kTargetFingerprint.size()] = '\0';
  });
}

flagdnnBackendResult_t
create_executable(void *context, const flagdnnBackendBuildInputV2 *input,
                  void **executable, std::size_t *workspace_size) noexcept {
  return plugin_call([&] {
    require(context != nullptr, "context is null");
    require(input != nullptr, "build input is null");
    require(input->struct_size >= sizeof(flagdnnBackendBuildInputV2),
            "build input structure is too small");
    require(executable != nullptr, "executable output pointer is null");
    require(workspace_size != nullptr, "workspace size output pointer is null");
    *executable = nullptr;
    *workspace_size = 0;
#if defined(FLAGDNN_CONTRACT_PREPARED_ENVIRONMENT)
    require_concurrent_builds();
#endif
    std::unique_ptr<ContractExecutable> result;
    try {
      result = std::make_unique<ContractExecutable>(*input);
    } catch (const PluginFailure &) {
      throw;
    } catch (const std::bad_alloc &) {
      throw;
    } catch (const std::exception &error) {
      fail(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
           "invalid contract artifact: " + std::string(error.what()));
    }
    *workspace_size = kWorkspaceSize;
    *executable = result.release();
  });
}

void destroy_executable(void *executable) noexcept {
  delete static_cast<ContractExecutable *>(executable);
}

flagdnnBackendResult_t execute(void *executable, void *native_stream,
                               const flagdnnBackendBindingV2 bindings[],
                               std::size_t binding_count, void *workspace,
                               std::size_t workspace_size) noexcept {
  return plugin_call([&] {
    require(executable != nullptr, "executable is null");
    static_cast<ContractExecutable *>(executable)
        ->execute(native_stream, bindings, binding_count, workspace,
                  workspace_size);
  });
}

const flagdnnBackendApiV2 api = {sizeof(flagdnnBackendApiV2),
                                 FLAGDNN_BACKEND_ABI_VERSION,
                                 "contract",
                                 &get_last_error,
                                 &create_context,
                                 &destroy_context,
                                 &get_target_fingerprint,
                                 &create_executable,
                                 &destroy_executable,
                                 &execute};

} // namespace

extern "C" FLAGDNN_BACKEND_EXPORT const flagdnnBackendApiV2 *
flagdnnBackendGetApiV2(void) {
  return &api;
}

#if defined(FLAGDNN_CONTRACT_PREPARED_ENVIRONMENT)
extern "C" FLAGDNN_BACKEND_EXPORT const flagdnnBackendBuildApiV1 *
flagdnnBackendGetBuildApiV1(void) {
  static const flagdnnBackendBuildApiV1 build_api = {
      sizeof(flagdnnBackendBuildApiV1), 1,
      &is_environment_prepared, &prepare_environment};
  return &build_api;
}
#endif
