/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

/*
 * Native compiler provider for the test-only contract backend.  It exercises
 * the same external compiler protocol as the Triton provider without making
 * the platform-neutral Core tests depend on a Python interpreter.
 */

#include "runtime/json.hpp"
#include "runtime/sha256.hpp"

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
constexpr std::string_view kProviderVersion = "3";

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

std::string read_file(const std::filesystem::path& path) {
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

void write_file(const std::filesystem::path& path, std::string_view value) {
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

std::string compiler_identity() {
  return flagdnn::native::sha256("flagdnn-contract-compiler:v3");
}

void apply_test_delay() {
  const char* configured =
      std::getenv("FLAGDNN_CONTRACT_COMPILER_DELAY_SECONDS");
  if (configured == nullptr || configured[0] == '\0') {
    return;
  }
  const std::string_view value(configured);
  std::uint64_t seconds = 0;
  const auto [end, error] =
      std::from_chars(value.data(), value.data() + value.size(), seconds);
  if (error != std::errc{} || end != value.data() + value.size() ||
      seconds == 0 || seconds > 30) {
    throw std::invalid_argument(
        "FLAGDNN_CONTRACT_COMPILER_DELAY_SECONDS must be in [1, 30]");
  }
  std::this_thread::sleep_for(std::chrono::seconds(seconds));
}

Arguments parse_arguments(int argc, char** argv) {
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

std::vector<std::int64_t> validate_graph(
    const flagdnn::native::json::Value& root) {
  using flagdnn::native::json::Value;
  require(root.at("schema_version").as_int() == 3,
          "contract compiler requires Graph IR schema v3");
  require(root.at("backend").as_string() == kBackend,
          "contract compiler received another backend");
  require(root.at("target").as_string() == kTarget,
          "contract target fingerprint is invalid");
  require(root.at("compiler_identity").as_string() == compiler_identity(),
          "contract compiler identity does not match request");

  const Value& build_options = root.at("build_options");
  const auto& modes = build_options.at("heuristic_modes").as_array();
  require(!modes.empty(), "contract heuristic modes are empty");
  std::map<std::string, bool, std::less<>> unique_modes;
  for (const Value& mode : modes) {
    const std::string& name = mode.as_string();
    require(name == "A" || name == "FALLBACK",
            "contract heuristic mode is invalid");
    require(unique_modes.emplace(name, true).second,
            "contract heuristic modes contain duplicates");
  }

  const Value& graph = root.at("graph");
  (void)graph.at("name").as_string();
  const auto& nodes = graph.at("nodes").as_array();
  const auto& tensors = graph.at("tensors").as_array();
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
  for (const Value& node : nodes) {
    require(node.at("type").as_string() == "relu",
            "contract compiler supports only ReLU");
    require(node.at("compute_data_type").as_string() == "float32",
            "contract ReLU compute type is invalid");
    const auto& inputs = node.at("inputs").as_array();
    const auto& outputs = node.at("outputs").as_array();
    require(inputs.size() == 1 && outputs.size() == 1,
            "contract ReLU port count is invalid");
    require(inputs[0].at("name").as_string() == "input" &&
                outputs[0].at("name").as_string() == "output",
            "contract ReLU port name is invalid");
    const std::int64_t node_id = node.at("id").as_int();
    require(node_id >= 0,
            "contract graph node ID is invalid");
    for (const std::int64_t previous : node_ids) {
      require(previous != node_id,
              "contract graph node IDs contain duplicates");
    }
    node_ids.push_back(node_id);
  }
  return node_ids;
}

std::string integer_array(const std::vector<std::int64_t>& values) {
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
                     const std::vector<std::int64_t>& node_ids) {
  const std::string stage_kind =
      node_ids.size() == 1 ? "contract_relu" : "contract_relu_chain";
  std::ostringstream output;
  output << "{\"artifact_kind\":\"flagdnn_execution_program\","
         << "\"backend\":\"" << kBackend << "\","
         << "\"compiler\":{\"identity_sha256\":\""
         << compiler_identity() << "\",\"provider\":\"" << kProvider
         << "\",\"provider_version\":\"" << kProviderVersion << "\"},"
         << "\"flagdnn_version\":\"" << flagdnn_version << "\","
         << "\"graph_node_count\":" << node_ids.size() << ','
         << "\"program\":{\"schema_version\":1,\"stage_count\":1,"
         << "\"stages\":[{\"dependencies\":[],\"kind\":\""
         << stage_kind << "\",\"source_node_ids\":"
         << integer_array(node_ids) << ",\"stage_id\":0}]},"
         << "\"request_sha256\":\"" << request_sha256 << "\","
         << "\"schema_version\":3,\"target\":\"" << kTarget << "\","
         << "\"workspace_size\":64}";
  return output.str();
}

void identify(const Arguments& arguments) {
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
  write_file(arguments.identity_output, compiler_identity() + "\n");
  if (!arguments.quiet) {
    std::cout << "{\"backend\":\"contract\",\"provider\":\""
              << kProvider << "\",\"status\":\"success\",\"target\":\""
              << kTarget << "\"}\n";
  }
}

void compile(const Arguments& arguments) {
  require(arguments.execution_engine == kExecutionEngine,
          "invalid contract compiler execution engine");
  require(!arguments.request.empty() &&
              !arguments.output_directory.empty(),
          "contract compiler build paths are missing");
  require(arguments.backend.empty() && arguments.target.empty() &&
              arguments.identity_output.empty(),
          "build request contains identity arguments");

  const std::string request = read_file(arguments.request);
  const auto root = flagdnn::native::json::parse(request);
  const std::vector<std::int64_t> node_ids = validate_graph(root);
  const std::string& flagdnn_version =
      root.at("flagdnn_version").as_string();
  std::filesystem::create_directories(arguments.output_directory);
  write_file(arguments.output_directory / "manifest.json",
             manifest(flagdnn::native::sha256(request),
                      flagdnn_version,
                      node_ids));
  if (!arguments.quiet) {
    std::cout << "{\"backend\":\"contract\",\"node_count\":"
              << node_ids.size()
              << ",\"provider\":\"" << kProvider
              << "\",\"stage_count\":1,\"status\":\"success\","
              << "\"target\":\"" << kTarget << "\"}\n";
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Arguments arguments = parse_arguments(argc, argv);
    apply_test_delay();
    if (arguments.identify) {
      identify(arguments);
    } else {
      compile(arguments);
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "CONTRACT_COMPILER_FAILED: " << error.what() << '\n';
    return 1;
  }
}
