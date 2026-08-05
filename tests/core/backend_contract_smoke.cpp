/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include <flagdnn/flagdnn.hpp>
#include <flagdnn_frontend.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

class TemporaryCache {
 public:
  TemporaryCache() {
    std::string pattern =
        (std::filesystem::temp_directory_path() /
         "flagdnn-contract-backend-XXXXXX")
            .string();
    std::vector<char> writable(pattern.begin(), pattern.end());
    writable.push_back('\0');
    char* created = mkdtemp(writable.data());
    if (created == nullptr) {
      throw std::runtime_error("mkdtemp failed");
    }
    path_ = created;
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

class ScopedEnvironment {
 public:
  ScopedEnvironment(std::string name, std::string value)
      : name_(std::move(name)) {
    if (const char* previous = std::getenv(name_.c_str())) {
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

  ScopedEnvironment(const ScopedEnvironment&) = delete;
  ScopedEnvironment& operator=(const ScopedEnvironment&) = delete;

 private:
  std::string name_;
  std::string previous_;
  bool had_previous_ = false;
};

std::filesystem::path find_single_manifest(
    const std::filesystem::path& cache_root) {
  std::filesystem::path result;
  std::size_t count = 0;
  for (const auto& entry :
       std::filesystem::recursive_directory_iterator(cache_root)) {
    if (entry.is_regular_file() &&
        entry.path().filename() == "manifest.json") {
      result = entry.path();
      ++count;
    }
  }
  if (count != 1) {
    throw std::runtime_error(
        "expected one cached manifest, found " + std::to_string(count));
  }
  return result;
}

void corrupt_manifest(const std::filesystem::path& path) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  output << "{invalid cached manifest\n";
  output.close();
  if (!output) {
    throw std::runtime_error("cannot corrupt cached manifest");
  }
}

void require_native_process_clean(const char* stage) {
  std::ifstream maps("/proc/self/maps");
  if (!maps) {
    throw std::runtime_error("cannot inspect /proc/self/maps");
  }
  std::string line;
  while (std::getline(maps, line)) {
    if (line.find("libpython") != std::string::npos ||
        line.find("site-packages/torch") != std::string::npos ||
        line.find("/torch/lib/") != std::string::npos) {
      throw std::runtime_error(
          std::string(stage) + " unexpectedly mapped Python or Torch: " +
          line);
    }
  }
}

template <typename Function>
void require_status(Function&& function,
                    flagdnnStatus_t expected,
                    std::string_view context) {
  try {
    function();
  } catch (const flagdnn::Error& error) {
    if (error.status() != expected) {
      throw std::runtime_error(
          std::string(context) + " returned unexpected status");
    }
    return;
  }
  throw std::runtime_error(
      std::string(context) + " unexpectedly succeeded");
}

void require_frontend_status(const flagdnn_frontend::error_t& status,
                             flagdnnStatus_t expected,
                             std::string_view context) {
  if (status.get_status() != expected) {
    throw std::runtime_error(
        std::string(context) + " returned unexpected frontend status: " +
        status.get_message());
  }
}

void require_relu(std::span<const float> input,
                  std::span<const float> output) {
  if (input.size() != output.size()) {
    throw std::runtime_error("contract output size differs");
  }
  for (std::size_t index = 0; index < input.size(); ++index) {
    const float expected = std::max(input[index], 0.0F);
    if (output[index] != expected) {
      throw std::runtime_error(
          "contract ReLU output differs at index " +
          std::to_string(index));
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 3) {
      throw std::invalid_argument(
          "usage: native_backend_contract_smoke COMPILER_EXECUTABLE COMPILER_ENTRY");
    }
    require_native_process_clean("startup");

    require_status(
        [] { flagdnn::Handle invalid("contract", 1); },
        FLAGDNN_STATUS_INVALID_VALUE,
        "nonzero contract device ordinal");

    TemporaryCache cache;
    flagdnn::Handle handle("contract", 0);
    if (handle.backend_name() != "contract" ||
        handle.target_fingerprint() != "host_contract_v1") {
      throw std::runtime_error("contract backend identity differs");
    }
    handle.set_compiler(argv[1], argv[2], cache.path().string());

    constexpr std::array<std::int64_t, 1> dimensions = {8};
    constexpr std::array<std::int64_t, 1> strides = {1};
    flagdnn::TensorDescriptor input_descriptor(
        1, FLAGDNN_DATA_FLOAT32, dimensions, strides);
    flagdnn::TensorDescriptor output_descriptor(
        2, FLAGDNN_DATA_FLOAT32, dimensions, strides);
    flagdnn::Graph graph;
    graph.relu(input_descriptor, output_descriptor);
    graph.finalize();

    flagdnn::Executable executable(handle, graph);
    if (executable.operation_count() != 1 ||
        executable.workspace_size() != 64) {
      throw std::runtime_error("contract executable metadata differs");
    }
    require_native_process_clean("after cache miss build");

    std::array<float, 8> input = {
        -4.0F, -1.5F, -0.0F, 0.25F, 1.0F, 3.5F, -9.0F, 8.0F};
    std::array<float, 8> output;
    output.fill(-123.0F);
    std::array<flagdnnBinding_t, 2> bindings = {
        flagdnnBinding_t{1, input.data()},
        flagdnnBinding_t{2, output.data()},
    };
    alignas(64) std::array<unsigned char, 64> workspace{};
    int stream_cookie = 7;

    require_status(
        [&] {
          executable.execute(bindings, nullptr, 0, &stream_cookie);
        },
        FLAGDNN_STATUS_INVALID_VALUE,
        "missing contract workspace");
    require_status(
        [&] {
          executable.execute(
              bindings, workspace.data(), workspace.size(), nullptr);
        },
        FLAGDNN_STATUS_INVALID_VALUE,
        "missing contract stream");

    executable.execute(bindings,
                       workspace.data(),
                       workspace.size(),
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
    recovered_executable.execute(bindings,
                                 workspace.data(),
                                 workspace.size(),
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
      ScopedEnvironment delay(
          "FLAGDNN_CONTRACT_COMPILER_DELAY_SECONDS", "2");
      require_status(
          [&] {
            flagdnn::Executable timed_out(handle, timeout_graph);
          },
          FLAGDNN_STATUS_COMPILATION_FAILED,
          "contract compiler timeout");
    }
    require_native_process_clean("after compiler timeout");

    namespace fe = flagdnn_frontend;
    fe::graph::Graph frontend_graph;
    frontend_graph.set_name("contract frontend \"relu\"\n")
        .set_io_data_type(fe::DataType_t::FLOAT)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);
    const auto frontend_input = frontend_graph.tensor(
        fe::graph::Tensor_attributes()
            .set_name("x")
            .set_uid(21)
            .set_dim({8})
            .set_stride({1}));
    const auto frontend_output = frontend_graph.pointwise(
        frontend_input,
        fe::graph::Pointwise_attributes()
            .set_name("relu")
            .set_mode(fe::PointwiseMode_t::RELU_FWD));
    frontend_output->set_name("y").set_uid(22).set_output(true);

    require_frontend_status(
        frontend_graph.create_execution_plans({fe::HeurMode_t::A}),
        FLAGDNN_STATUS_NOT_INITIALIZED,
        "out-of-order create_execution_plans");
    require_frontend_status(
        frontend_graph.validate(), FLAGDNN_STATUS_SUCCESS, "frontend validate");
    require_frontend_status(frontend_graph.build_operation_graph(handle),
                            FLAGDNN_STATUS_SUCCESS,
                            "frontend build_operation_graph");
    require_frontend_status(
        frontend_graph.create_execution_plans(
            {fe::HeurMode_t::A, fe::HeurMode_t::FALLBACK}),
        FLAGDNN_STATUS_SUCCESS,
        "frontend create_execution_plans");
    require_frontend_status(frontend_graph.check_support(handle),
                            FLAGDNN_STATUS_SUCCESS,
                            "frontend check_support");
    require_frontend_status(frontend_graph.build_plans(handle),
                            FLAGDNN_STATUS_SUCCESS,
                            "frontend build_plans");
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
        frontend_graph.execute(handle,
                               frontend_bindings,
                               workspace.data(),
                               workspace.size(),
                               &stream_cookie),
        FLAGDNN_STATUS_SUCCESS,
        "frontend execute");
    require_relu(input, frontend_output_values);

    flagdnn::TensorDescriptor generic_input_descriptor(
        31, FLAGDNN_DATA_FLOAT32, dimensions, strides);
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
    generic_executable.execute(generic_bindings,
                               workspace.data(),
                               workspace.size(),
                               &stream_cookie);
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
    require_status(
        [&] { view_only_graph.finalize(); },
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
    require_status(
        [&] { invalid_transpose_graph.finalize(); },
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
    require_status(
        [&] { invalid_slice_graph.finalize(); },
        FLAGDNN_STATUS_INVALID_VALUE,
        "non-positive slice stride");

    flagdnn::TensorDescriptor chain_input_descriptor(
        11, FLAGDNN_DATA_FLOAT32, dimensions, strides);
    flagdnn::TensorDescriptor chain_intermediate_descriptor(
        12, FLAGDNN_DATA_FLOAT32, dimensions, strides);
    chain_intermediate_descriptor.set_virtual();
    flagdnn::TensorDescriptor chain_output_descriptor(
        13, FLAGDNN_DATA_FLOAT32, dimensions, strides);
    flagdnn::OperationDescriptor chain_consumer(FLAGDNN_OPERATION_RELU);
    chain_consumer.set_relu(
        chain_intermediate_descriptor, chain_output_descriptor);
    flagdnn::OperationDescriptor chain_producer(FLAGDNN_OPERATION_RELU);
    chain_producer.set_relu(
        chain_input_descriptor, chain_intermediate_descriptor);
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
    fused_chain.execute(chain_bindings,
                        workspace.data(),
                        workspace.size(),
                        &stream_cookie);
    require_relu(input, chain_output);
    if (workspace.front() != 0x5aU) {
      throw std::runtime_error("fused contract stage did not use workspace");
    }
    require_native_process_clean("after out-of-order graph execute");

    handle.set_compiler("/flagdnn/missing-compiler-executable",
                        "/flagdnn/missing-compiler",
                        cache.path().string());
    flagdnn::Executable cache_hit(handle, graph);
    output.fill(-321.0F);
    cache_hit.execute(bindings,
                      workspace.data(),
                      workspace.size(),
                      &stream_cookie);
    require_relu(input, output);
    require_native_process_clean("after cache hit execute");

    std::cout << "PASS backend=contract target=host_contract_v1 "
                 "operation=relu graph=toposort stage_fusion "
                 "cache=miss+recovery+hit compiler_timeout=pass\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "CONTRACT_BACKEND_FAILED: " << error.what() << '\n';
    return 1;
  }
}
