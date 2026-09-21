/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

/*
 * Native compiler provider for the test-only contract backend.  It exercises
 * the same external compiler protocol as the Triton provider without making
 * the platform-neutral Core tests depend on a Python interpreter.
 */

#include "runtime/json.hpp"
#include "runtime/sha256.hpp"

#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <vector>

namespace {

constexpr std::string_view kBackend = "contract";
constexpr std::string_view kTarget = "host_contract_v1";
constexpr std::string_view kExecutionEngine = "external_artifact";
constexpr std::string_view kProvider = "contract_reference";
constexpr std::string_view kProviderVersion = "4";

class TemporaryIdentityFailure final : public std::exception {};

struct Arguments {
  bool identify = false;
  bool quiet = false;
  std::filesystem::path request;
  std::filesystem::path output_directory;
  std::filesystem::path identity_output;
  std::string backend;
  std::string target;
  std::string execution_engine = std::string(kExecutionEngine);
};

std::string read_file(const std::filesystem::path &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("cannot open input file: " + path.string());
  }
  std::ostringstream output;
  output << input.rdbuf();
  if (!input.good() && !input.eof()) {
    throw std::runtime_error("cannot read input file: " + path.string());
  }
  return output.str();
}

void write_file(const std::filesystem::path &path, std::string_view value) {
  if (!path.parent_path().empty()) {
    std::filesystem::create_directories(path.parent_path());
  }
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output) {
    throw std::runtime_error("cannot open output file: " + path.string());
  }
  output.write(value.data(), static_cast<std::streamsize>(value.size()));
  if (!output) {
    throw std::runtime_error("cannot write output file: " + path.string());
  }
}

void append_stat_record(std::string &state, const struct stat &status) {
  state += std::to_string(static_cast<std::uintmax_t>(status.st_dev));
  state.push_back(':');
  state += std::to_string(static_cast<std::uintmax_t>(status.st_ino));
  state.push_back(':');
  state += std::to_string(static_cast<std::uintmax_t>(status.st_mode));
  state.push_back(':');
  state += std::to_string(static_cast<std::uintmax_t>(status.st_size));
  state.push_back(':');
#if defined(__APPLE__)
  const auto &modified = status.st_mtimespec;
  const auto &changed = status.st_ctimespec;
#else
  const auto &modified = status.st_mtim;
  const auto &changed = status.st_ctim;
#endif
  state += std::to_string(modified.tv_sec);
  state.push_back(':');
  state += std::to_string(modified.tv_nsec);
  state.push_back(':');
  state += std::to_string(changed.tv_sec);
  state.push_back(':');
  state += std::to_string(changed.tv_nsec);
}

std::string dependency_fingerprint(const std::filesystem::path &path) {
  std::string state("flagdnn-dependency-state-v1\0", 28);
  struct stat link_status {};
  if (::lstat(path.c_str(), &link_status) != 0) {
    const int error = errno;
    state += "lstat-error:";
    state += std::to_string(error);
    state.push_back('\0');
    return flagdnn::native::sha256(state);
  }
  state += "lstat:";
  append_stat_record(state, link_status);
  state.push_back('\0');

  if (S_ISLNK(link_status.st_mode)) {
    std::vector<char> link_value(256);
    for (;;) {
      const ssize_t size =
          ::readlink(path.c_str(), link_value.data(), link_value.size());
      if (size < 0) {
        const int error = errno;
        state += "link-error:";
        state += std::to_string(error);
        state.push_back('\0');
        break;
      }
      if (static_cast<std::size_t>(size) < link_value.size()) {
        state += "link:";
        state.append(link_value.data(), static_cast<std::size_t>(size));
        state.push_back('\0');
        break;
      }
      if (link_value.size() >= 65536U) {
        state += "link-error:";
        state += std::to_string(ENAMETOOLONG);
        state.push_back('\0');
        break;
      }
      link_value.resize(link_value.size() * 2U);
    }
  } else {
    state.append("link:\0", 6);
  }

  struct stat status {};
  if (::stat(path.c_str(), &status) != 0) {
    const int error = errno;
    state += "stat-error:";
    state += std::to_string(error);
    state.push_back('\0');
  } else {
    state += "stat:";
    append_stat_record(state, status);
    state.push_back('\0');
  }
  return flagdnn::native::sha256(state);
}

std::filesystem::path identity_dependency_path() {
  const char *configured =
      std::getenv("FLAGDNN_CONTRACT_COMPILER_IDENTITY_DEPENDENCY");
  if (configured == nullptr || configured[0] == '\0') {
    return {};
  }
  const std::filesystem::path path(configured);
  if (!path.is_absolute()) {
    throw std::invalid_argument("identity dependency must be absolute");
  }
  return path.lexically_normal();
}

std::string compiler_identity() {
  std::string material("flagdnn-contract-compiler:v4\0", 29);
  const std::filesystem::path dependency = identity_dependency_path();
  if (dependency.empty()) {
    material += "no-dependency";
  } else {
    material += dependency.string();
    material.push_back('\0');
    material += dependency_fingerprint(dependency);
    material.push_back('\0');
    material += read_file(dependency);
  }
  return flagdnn::native::sha256(material);
}

std::string json_string(std::string_view value) {
  constexpr char hexadecimal[] = "0123456789abcdef";
  std::string result = "\"";
  for (const unsigned char character : value) {
    switch (character) {
    case '\"':
      result += "\\\"";
      break;
    case '\\':
      result += "\\\\";
      break;
    case '\b':
      result += "\\b";
      break;
    case '\f':
      result += "\\f";
      break;
    case '\n':
      result += "\\n";
      break;
    case '\r':
      result += "\\r";
      break;
    case '\t':
      result += "\\t";
      break;
    default:
      if (character < 0x20U) {
        result += "\\u00";
        result.push_back(hexadecimal[(character >> 4U) & 0x0fU]);
        result.push_back(hexadecimal[character & 0x0fU]);
      } else {
        result.push_back(static_cast<char>(character));
      }
    }
  }
  result.push_back('\"');
  return result;
}

void apply_delay_from_environment(const char *environment_name) {
  const char *configured = std::getenv(environment_name);
  if (configured == nullptr || configured[0] == '\0') {
    return;
  }
  const std::string_view value(configured);
  std::uint64_t seconds = 0;
  const auto [end, error] =
      std::from_chars(value.data(), value.data() + value.size(), seconds);
  if (error != std::errc{} || end != value.data() + value.size() ||
      seconds == 0 || seconds > 30) {
    throw std::invalid_argument(std::string(environment_name) +
                                " must be in [1, 30]");
  }
  std::this_thread::sleep_for(std::chrono::seconds(seconds));
}

void apply_test_delay(bool identify) {
  const char *compile_only =
      std::getenv("FLAGDNN_CONTRACT_COMPILER_DELAY_COMPILE_ONLY");
  if (compile_only != nullptr && compile_only[0] != '\0') {
    if (std::string_view(compile_only) != "1") {
      throw std::invalid_argument(
          "FLAGDNN_CONTRACT_COMPILER_DELAY_COMPILE_ONLY must be 1");
    }
    if (identify) {
      return;
    }
  }
  apply_delay_from_environment("FLAGDNN_CONTRACT_COMPILER_DELAY_SECONDS");
}

void record_compile_marker(const char *environment_name,
                           const Arguments &arguments) {
  const char *configured = std::getenv(environment_name);
  if (configured == nullptr || configured[0] == '\0') {
    return;
  }
  const std::filesystem::path marker(configured);
  if (!marker.is_absolute()) {
    throw std::invalid_argument(std::string(environment_name) +
                                " must be an absolute path");
  }
  write_file(marker, arguments.output_directory.string());
}

void record_inherited_environment() {
  const char *marker =
      std::getenv("FLAGDNN_CONTRACT_COMPILER_INHERITED_ENVIRONMENT_MARKER");
  const char *variable =
      std::getenv("FLAGDNN_CONTRACT_COMPILER_INHERITED_ENVIRONMENT_VARIABLE");
  const char *expected =
      std::getenv("FLAGDNN_CONTRACT_COMPILER_INHERITED_ENVIRONMENT_VALUE");
  if (marker == nullptr && variable == nullptr && expected == nullptr) {
    return;
  }
  if (marker == nullptr || marker[0] == '\0' || variable == nullptr ||
      variable[0] == '\0' || expected == nullptr || expected[0] == '\0') {
    throw std::invalid_argument(
        "inherited-environment contract configuration is incomplete");
  }
  const char *actual = std::getenv(variable);
  if (actual != nullptr && std::string_view(actual) == expected) {
    write_file(marker, "observed\n");
  }
}

std::string identity_response_mode() {
  const char *configured =
      std::getenv("FLAGDNN_CONTRACT_COMPILER_IDENTITY_RESPONSE_MODE");
  const std::string result =
      configured == nullptr || configured[0] == '\0' ? "full" : configured;
  if (result != "full" && result != "digest-only" && result != "files-only") {
    throw std::invalid_argument(
        "FLAGDNN_CONTRACT_COMPILER_IDENTITY_RESPONSE_MODE must be full, "
        "digest-only, or files-only");
  }
  return result;
}

void record_identity_query() {
  const char *configured =
      std::getenv("FLAGDNN_CONTRACT_COMPILER_IDENTITY_COUNTER");
  if (configured == nullptr || configured[0] == '\0') {
    return;
  }
  const std::filesystem::path path(configured);
  std::uint64_t count = 0;
  if (std::filesystem::is_regular_file(path)) {
    const std::string current = read_file(path);
    const auto [end, error] =
        std::from_chars(current.data(), current.data() + current.size(), count);
    if (error != std::errc{} || end != current.data() + current.size()) {
      throw std::runtime_error("identity query counter is invalid");
    }
  }
  write_file(path, std::to_string(count + 1));
}

void apply_identity_temporary_failure_once() {
  const char *configured =
      std::getenv("FLAGDNN_CONTRACT_COMPILER_TEMPFAIL_MARKER");
  if (configured == nullptr || configured[0] == '\0') {
    return;
  }
  const std::filesystem::path marker(configured);
  if (!marker.is_absolute()) {
    throw std::invalid_argument(
        "identity temporary-failure marker must be absolute");
  }
  if (!std::filesystem::exists(marker)) {
    write_file(marker, "retry");
    throw TemporaryIdentityFailure();
  }
}

bool apply_identity_failure_mode(const Arguments &arguments) {
  const char *configured =
      std::getenv("FLAGDNN_CONTRACT_COMPILER_IDENTITY_FAILURE_MODE");
  if (configured == nullptr || configured[0] == '\0') {
    return false;
  }
  const std::string_view mode(configured);
  if (mode == "malformed") {
    write_file(arguments.identity_output, "malformed-identity\n");
    return true;
  }
  if (mode == "nonzero") {
    throw std::runtime_error("requested nonzero identity failure");
  }
  if (mode == "temporary") {
    throw TemporaryIdentityFailure();
  }
  throw std::invalid_argument(
      "FLAGDNN_CONTRACT_COMPILER_IDENTITY_FAILURE_MODE must be malformed, "
      "nonzero, or temporary");
}

Arguments parse_arguments(int argc, char **argv) {
  if (argc < 2) {
    throw std::invalid_argument(
        "usage: contract_compiler ENTRY [compiler protocol options]");
  }

  // argv[1] is the compiler entry selected by RuntimeContext.  Native test
  // providers are single-binary, so the value is only an explicit protocol
  // slot and does not select another script.
  Arguments result;
  for (int index = 2; index < argc; ++index) {
    const std::string_view option = argv[index];
    if (option == "--identify") {
      result.identify = true;
    } else if (option == "--quiet") {
      result.quiet = true;
    } else {
      if (index + 1 >= argc) {
        throw std::invalid_argument("compiler option has no value: " +
                                    std::string(option));
      }
      const std::string value = argv[++index];
      if (option == "--request") {
        result.request = value;
      } else if (option == "--output-dir") {
        result.output_directory = value;
      } else if (option == "--identity-output") {
        result.identity_output = value;
      } else if (option == "--backend") {
        result.backend = value;
      } else if (option == "--target") {
        result.target = value;
      } else if (option == "--execution-engine") {
        result.execution_engine = value;
      } else {
        throw std::invalid_argument("unknown compiler option: " +
                                    std::string(option));
      }
    }
  }
  return result;
}

void require(bool condition, std::string_view message) {
  if (!condition) {
    throw std::invalid_argument(std::string(message));
  }
}

std::vector<std::int64_t>
validate_graph(const flagdnn::native::json::Value &root) {
  using flagdnn::native::json::Value;
  require(root.at("schema_version").as_int() == 3,
          "contract compiler requires Graph IR schema v3");
  require(root.at("backend").as_string() == kBackend,
          "contract compiler received another backend");
  require(root.at("target").as_string() == kTarget,
          "contract target fingerprint is invalid");
  require(root.at("compiler_identity").as_string() == compiler_identity(),
          "contract compiler identity does not match request");

  const Value &build_options = root.at("build_options");
  const auto &modes = build_options.at("heuristic_modes").as_array();
  require(!modes.empty(), "contract heuristic modes are empty");
  std::map<std::string, bool, std::less<>> unique_modes;
  for (const Value &mode : modes) {
    const std::string &name = mode.as_string();
    require(name == "A" || name == "FALLBACK",
            "contract heuristic mode is invalid");
    require(unique_modes.emplace(name, true).second,
            "contract heuristic modes contain duplicates");
  }

  const Value &graph = root.at("graph");
  (void)graph.at("name").as_string();
  const auto &nodes = graph.at("nodes").as_array();
  const auto &tensors = graph.at("tensors").as_array();
  require(nodes.size() == 1 || nodes.size() == 2,
          "contract compiler supports one or two graph nodes");
  require(graph.at("node_count").as_int() ==
              static_cast<std::int64_t>(nodes.size()),
          "contract graph node count is invalid");
  require(graph.at("tensor_count").as_int() ==
              static_cast<std::int64_t>(tensors.size()),
          "contract graph tensor count is invalid");

  std::vector<std::int64_t> node_ids;
  node_ids.reserve(nodes.size());
  for (const Value &node : nodes) {
    require(node.at("type").as_string() == "relu",
            "contract compiler supports only ReLU");
    require(node.at("compute_data_type").as_string() == "float32",
            "contract ReLU compute type is invalid");
    const auto &inputs = node.at("inputs").as_array();
    const auto &outputs = node.at("outputs").as_array();
    require(inputs.size() == 1 && outputs.size() == 1,
            "contract ReLU port count is invalid");
    require(inputs[0].at("name").as_string() == "input" &&
                outputs[0].at("name").as_string() == "output",
            "contract ReLU port name is invalid");
    const std::int64_t node_id = node.at("id").as_int();
    require(node_id >= 0, "contract graph node ID is invalid");
    for (const std::int64_t previous : node_ids) {
      require(previous != node_id,
              "contract graph node IDs contain duplicates");
    }
    node_ids.push_back(node_id);
  }
  return node_ids;
}

std::string integer_array(const std::vector<std::int64_t> &values) {
  std::string result = "[";
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0) {
      result += ',';
    }
    result += std::to_string(values[index]);
  }
  result += ']';
  return result;
}

std::string manifest(std::string_view request_sha256,
                     std::string_view flagdnn_version,
                     std::string_view compiler_identity,
                     const std::vector<std::int64_t> &node_ids) {
  const std::string stage_kind =
      node_ids.size() == 1 ? "contract_relu" : "contract_relu_chain";
  std::ostringstream output;
  output << "{\"artifact_kind\":\"flagdnn_execution_program\","
         << "\"backend\":\"" << kBackend << "\","
         << "\"compiler\":{\"identity_sha256\":\"" << compiler_identity
         << "\",\"provider\":\"" << kProvider << "\",\"provider_version\":\""
         << kProviderVersion << "\"},"
         << "\"flagdnn_version\":\"" << flagdnn_version << "\","
         << "\"graph_node_count\":" << node_ids.size() << ','
         << "\"program\":{\"schema_version\":1,\"stage_count\":1,"
         << "\"stages\":[{\"dependencies\":[],\"kind\":\"" << stage_kind
         << "\",\"source_node_ids\":" << integer_array(node_ids)
         << ",\"stage_id\":0}]},"
         << "\"request_sha256\":\"" << request_sha256 << "\","
         << "\"schema_version\":3,\"target\":\"" << kTarget << "\","
         << "\"workspace_size\":64}";
  return output.str();
}

void identify(const Arguments &arguments) {
  require(arguments.backend == kBackend,
          "invalid contract compiler backend identity request");
  require(arguments.target == kTarget,
          "invalid contract compiler target identity request");
  require(arguments.execution_engine == kExecutionEngine,
          "invalid contract compiler execution engine");
  require(!arguments.identity_output.empty(),
          "contract compiler identity output is missing");
  require(arguments.request.empty() && arguments.output_directory.empty(),
          "identity request contains build paths");
  record_identity_query();
  if (apply_identity_failure_mode(arguments)) {
    return;
  }
  apply_identity_temporary_failure_once();
  const std::filesystem::path dependency = identity_dependency_path();
  const std::string fingerprint_before =
      dependency.empty() ? std::string() : dependency_fingerprint(dependency);
  const std::string content_before =
      dependency.empty() ? std::string()
                         : flagdnn::native::sha256_file(dependency);
  const std::string identity = compiler_identity();
  const std::string fingerprint_after =
      dependency.empty() ? std::string() : dependency_fingerprint(dependency);
  const std::string content_after =
      dependency.empty() ? std::string()
                         : flagdnn::native::sha256_file(dependency);
  if (fingerprint_before != fingerprint_after ||
      content_before != content_after) {
    throw TemporaryIdentityFailure();
  }
  const std::string response_mode = identity_response_mode();
  if (response_mode == "digest-only") {
    write_file(arguments.identity_output, identity + "\n");
    return;
  }

  if (response_mode == "files-only") {
    std::string metadata = "{\"files\":[";
    if (!dependency.empty()) {
      metadata += json_string(dependency.string());
    }
    metadata += "],\"schema_version\":1}\n";
    write_file(arguments.identity_output, identity + "\n" + metadata);
    return;
  }

  const char *incomplete =
      std::getenv("FLAGDNN_CONTRACT_COMPILER_DEPENDENCIES_INCOMPLETE");
  const bool dependencies_complete =
      incomplete == nullptr || incomplete[0] == '\0';
  std::string metadata = "{\"dependencies_complete\":";
  metadata += dependencies_complete ? "true" : "false";
  metadata += ",\"files\":[";
  if (!dependency.empty()) {
    metadata += json_string(dependency.string());
  }
  metadata += "],\"schema_version\":1,\"snapshot_schema_version\":1,";
  metadata += "\"snapshots\":[";
  if (!dependency.empty()) {
    const char *forged_content =
        std::getenv("FLAGDNN_CONTRACT_COMPILER_FORGED_CONTENT_SHA256");
    const std::string reported_content =
        forged_content != nullptr && forged_content[0] != '\0'
            ? std::string(64, '0')
            : content_after;
    metadata += "{\"content_sha256\":" + json_string(reported_content) +
                ",\"fingerprint\":" + json_string(fingerprint_after) +
                ",\"path\":" + json_string(dependency.string()) + "}";
  }
  metadata += "]}\n";
  write_file(arguments.identity_output, identity + "\n" + metadata);
  if (!arguments.quiet) {
    std::cout << "{\"backend\":\"contract\",\"provider\":\"" << kProvider
              << "\",\"status\":\"success\",\"target\":\"" << kTarget
              << "\"}\n";
  }
}

void compile(const Arguments &arguments) {
  require(arguments.execution_engine == kExecutionEngine,
          "invalid contract compiler execution engine");
  require(!arguments.request.empty() && !arguments.output_directory.empty(),
          "contract compiler build paths are missing");
  require(arguments.backend.empty() && arguments.target.empty() &&
              arguments.identity_output.empty(),
          "build request contains identity arguments");

  const std::string request = read_file(arguments.request);
  const auto root = flagdnn::native::json::parse(request);
  const std::vector<std::int64_t> node_ids = validate_graph(root);
  record_compile_marker("FLAGDNN_CONTRACT_COMPILER_COMPILE_VALIDATED_MARKER",
                        arguments);
  apply_delay_from_environment(
      "FLAGDNN_CONTRACT_COMPILER_DELAY_AFTER_VALIDATE_SECONDS");
  const std::string &flagdnn_version = root.at("flagdnn_version").as_string();
  const std::string &requested_identity =
      root.at("compiler_identity").as_string();
  std::filesystem::create_directories(arguments.output_directory);
  write_file(arguments.output_directory / "manifest.json",
             manifest(flagdnn::native::sha256(request), flagdnn_version,
                      requested_identity, node_ids));
  if (!arguments.quiet) {
    std::cout << "{\"backend\":\"contract\",\"node_count\":" << node_ids.size()
              << ",\"provider\":\"" << kProvider
              << "\",\"stage_count\":1,\"status\":\"success\","
              << "\"target\":\"" << kTarget << "\"}\n";
  }
}

} // namespace

int main(int argc, char **argv) {
  try {
    record_inherited_environment();
    const Arguments arguments = parse_arguments(argc, argv);
    if (!arguments.identify) {
      record_compile_marker("FLAGDNN_CONTRACT_COMPILER_COMPILE_STARTED_MARKER",
                            arguments);
    }
    apply_test_delay(arguments.identify);
    if (arguments.identify) {
      identify(arguments);
    } else {
      compile(arguments);
      record_compile_marker("FLAGDNN_CONTRACT_COMPILER_COMPILE_FINISHED_MARKER",
                            arguments);
    }
    return 0;
  } catch (const TemporaryIdentityFailure &) {
    return 75;
  } catch (const std::exception &error) {
    std::cerr << "CONTRACT_COMPILER_FAILED: " << error.what() << '\n';
    return 1;
  }
}
