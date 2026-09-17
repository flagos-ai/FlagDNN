/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include <unistd.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <flagdnn/frontend.hpp>
#include <fstream>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "hip_driver.hpp"
#include "pointwise_reference.hpp"
#include "tensor_io.hpp"

namespace {

namespace hv = flagdnn::validation::hygon;
namespace io = flagdnn::validation::hygon::tensor_io;
namespace fe = flagdnn_frontend;

void check_frontend(const fe::error_t &error, const char *operation) {
  if (error.is_bad()) {
    throw std::runtime_error(std::string(operation) +
                             " failed: " + error.get_message());
  }
}

bool process_maps_contains(const std::string &needle) {
  std::ifstream maps("/proc/self/maps");
  std::string line;
  while (std::getline(maps, line)) {
    if (line.find(needle) != std::string::npos) {
      return true;
    }
  }
  return false;
}

void require_native_process_clean(const char *stage) {
  const bool has_python = process_maps_contains("libpython");
  const bool has_torch = process_maps_contains("libtorch");
  std::cout << stage << "_has_libpython=" << std::boolalpha << has_python << " "
            << stage << "_has_libtorch=" << has_torch << '\n';
#if !defined(FLAGDNN_EXPECT_LIBTRITON_JIT)
  if (has_python || has_torch) {
    throw std::runtime_error(std::string(stage) +
                             " unexpectedly loaded Python or Torch");
  }
#endif
}

class TemporaryCache {
public:
  TemporaryCache() {
    std::string pattern = (std::filesystem::temp_directory_path() /
                           "flagdnn-hygon-native-graph-XXXXXX")
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

void require_invalid_graph(flagdnn::Graph &graph, const char *name) {
  const auto require_rejected = [&](auto &&operation, const char *stage) {
    try {
      operation();
    } catch (const flagdnn::Error &error) {
      if (error.status() == FLAGDNN_STATUS_INVALID_VALUE) {
        return;
      }
      throw std::runtime_error(std::string(name) +
                               " returned the wrong status from " + stage);
    }
    throw std::runtime_error(std::string(name) + " was not rejected by " +
                             stage);
  };
  require_rejected([&] { graph.validate(); }, "validate");
  require_rejected([&] { graph.finalize(); }, "finalize");
}

void test_invalid_graph_contracts() {
  const std::array<std::int64_t, 1> large_dimensions = {1024};
  const std::array<std::int64_t, 1> small_dimensions = {512};
  const std::array<std::int64_t, 1> strides = {1};

  flagdnn::TensorDescriptor virtual_input(40, FLAGDNN_DATA_FLOAT32,
                                          large_dimensions, strides);
  virtual_input.set_virtual();
  flagdnn::TensorDescriptor output(41, FLAGDNN_DATA_FLOAT32, large_dimensions,
                                   strides);
  flagdnn::Graph missing_producer;
  missing_producer.relu(virtual_input, output);
  require_invalid_graph(missing_producer, "missing virtual producer");

  flagdnn::TensorDescriptor first_input(42, FLAGDNN_DATA_FLOAT32,
                                        large_dimensions, strides);
  flagdnn::TensorDescriptor second_input(43, FLAGDNN_DATA_FLOAT32,
                                         large_dimensions, strides);
  flagdnn::TensorDescriptor shared_output(44, FLAGDNN_DATA_FLOAT32,
                                          large_dimensions, strides);
  shared_output.set_virtual();
  flagdnn::Graph missing_external_output;
  missing_external_output.relu(first_input, shared_output);
  require_invalid_graph(missing_external_output, "missing non-virtual output");

  flagdnn::Graph duplicate_producer;
  duplicate_producer.relu(first_input, shared_output);
  duplicate_producer.relu(second_input, shared_output);
  require_invalid_graph(duplicate_producer, "duplicate virtual producer");

  flagdnn::TensorDescriptor large_shared(45, FLAGDNN_DATA_FLOAT32,
                                         large_dimensions, strides);
  large_shared.set_virtual();
  flagdnn::TensorDescriptor small_shared(45, FLAGDNN_DATA_FLOAT32,
                                         small_dimensions, strides);
  small_shared.set_virtual();
  flagdnn::TensorDescriptor small_output(46, FLAGDNN_DATA_FLOAT32,
                                         small_dimensions, strides);
  flagdnn::Graph conflicting_metadata;
  conflicting_metadata.relu(first_input, large_shared);
  conflicting_metadata.relu(small_shared, small_output);
  require_invalid_graph(conflicting_metadata,
                        "conflicting shared UID metadata");
}

std::vector<std::uint8_t> encode(std::span<const float> values) {
  return io::encode(values, FLAGDNN_DATA_FLOAT32);
}

std::vector<float> read_output(const hv::DeviceBuffer &buffer,
                               std::size_t element_count, hv::Stream &stream) {
  std::vector<std::uint8_t> bytes(element_count * sizeof(float));
  buffer.copy_to_host(bytes.data(), bytes.size(), stream.get());
  stream.synchronize();
  return io::decode(bytes, FLAGDNN_DATA_FLOAT32, element_count);
}

double maximum_difference(std::span<const float> left,
                          std::span<const float> right) {
  if (left.size() != right.size()) {
    throw std::runtime_error("FlagDNN and hipDNN graph outputs differ in size");
  }
  double maximum = 0.0;
  for (std::size_t index = 0; index < left.size(); ++index) {
    const double difference = std::abs(static_cast<double>(left[index]) -
                                       static_cast<double>(right[index]));
    if (!std::isfinite(difference)) {
      throw std::runtime_error("graph output comparison is not finite");
    }
    maximum = std::max(maximum, difference);
  }
  return maximum;
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 3) {
      throw std::invalid_argument(
          "usage: native_hygon_graph_smoke COMPILER_EXECUTABLE "
          "COMPILER_ENTRY");
    }
    if (setenv("FLAGDNN_EXECUTION_ENGINE", "libtriton_jit", 1) != 0) {
      throw std::runtime_error("cannot select libtriton_jit engine");
    }
    require_native_process_clean("startup");
    hv::DeviceGuard device;
    hv::Stream stream;
    TemporaryCache cache;

    flagdnn::Handle handle("hygon", 0);
    handle.set_compiler(argv[1], argv[2], cache.path().string());
    test_invalid_graph_contracts();

    fe::graph::Graph graph;
    graph.set_name("hygon_mul_add_chain")
        .set_io_data_type(fe::DataType_t::FLOAT)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT)
        .set_autotune(true);
    auto left = graph.tensor(fe::graph::Tensor_attributes()
                                 .set_name("left")
                                 .set_uid(1)
                                 .set_data_type(fe::DataType_t::FLOAT)
                                 .set_dim({1, 1, 1024})
                                 .set_stride({1024, 1024, 1}));
    auto right = graph.tensor(fe::graph::Tensor_attributes()
                                  .set_name("right")
                                  .set_uid(2)
                                  .set_data_type(fe::DataType_t::FLOAT)
                                  .set_dim({1, 1, 1024})
                                  .set_stride({1024, 1024, 1}));
    auto squared =
        graph.pointwise(right, right,
                        fe::graph::Pointwise_attributes()
                            .set_name("square")
                            .set_mode(fe::PointwiseMode_t::MUL)
                            .set_compute_data_type(fe::DataType_t::FLOAT));
    constexpr std::int64_t kVirtualAlignment = 4096;
    squared->set_alignment(kVirtualAlignment);
    auto output =
        graph.pointwise(left, squared,
                        fe::graph::Pointwise_attributes()
                            .set_name("add")
                            .set_mode(fe::PointwiseMode_t::ADD)
                            .set_compute_data_type(fe::DataType_t::FLOAT));
    output->set_name("output")
        .set_uid(3)
        .set_data_type(fe::DataType_t::FLOAT)
        .set_dim({1, 1, 1024})
        .set_stride({1024, 1024, 1})
        .set_output(true);

    check_frontend(graph.build(handle), "graph.build");
    require_native_process_clean("after_build");
    if (!squared->get_is_virtual() || squared->get_uid() <= 0 ||
        squared->get_uid() == 1 || squared->get_uid() == 2 ||
        squared->get_uid() == 3) {
      throw std::runtime_error(
          "frontend did not assign a distinct virtual tensor UID");
    }

    std::int64_t workspace_size = 0;
    check_frontend(graph.get_workspace_size(workspace_size),
                   "graph.get_workspace_size");
    if (workspace_size < 1024 * static_cast<std::int64_t>(sizeof(float))) {
      throw std::runtime_error("virtual tensor workspace is too small");
    }

    std::vector<float> host_left(1024);
    std::vector<float> host_right(1024);
    for (std::size_t index = 0; index < host_left.size(); ++index) {
      host_left[index] =
          static_cast<float>(static_cast<int>(index % 37) - 18) / 7.0F;
      host_right[index] =
          static_cast<float>(static_cast<int>(index % 11) - 5) / 13.0F;
    }
    const std::vector<std::uint8_t> encoded_left = encode(host_left);
    const std::vector<std::uint8_t> encoded_right = encode(host_right);
    const std::size_t bytes = encoded_left.size();

    hv::DeviceBuffer flagdnn_left(bytes);
    hv::DeviceBuffer flagdnn_right(bytes);
    hv::DeviceBuffer flagdnn_output(bytes);
    hv::DeviceBuffer graph_workspace(static_cast<std::size_t>(workspace_size) +
                                     256);
    // Public execution requires 256-byte alignment; the backend must still
    // align this address to the larger virtual-tensor requirement.
    void *const deliberately_misaligned_workspace =
        graph_workspace.opaque_at(256);
    if (reinterpret_cast<std::uintptr_t>(deliberately_misaligned_workspace) %
            static_cast<std::uintptr_t>(kVirtualAlignment) ==
        0) {
      throw std::runtime_error(
          "graph alignment regression workspace is unexpectedly aligned");
    }
    flagdnn_left.copy_from_host(encoded_left.data(), bytes, stream.get());
    flagdnn_right.copy_from_host(encoded_right.data(), bytes, stream.get());
    const fe::VariantPack variant_pack = {
        {1, flagdnn_left.opaque()},
        {2, flagdnn_right.opaque()},
        {3, flagdnn_output.opaque()},
    };

    const auto missing_workspace =
        graph.execute(handle, variant_pack, nullptr, stream.opaque());
    if (missing_workspace.is_good()) {
      throw std::runtime_error("null graph workspace was not rejected");
    }
    check_frontend(graph.execute(handle, variant_pack,
                                 deliberately_misaligned_workspace,
                                 stream.opaque()),
                   "graph.execute");
    stream.synchronize();

    std::vector<hv::ReferenceTensor> reference_tensors = {
        {1, FLAGDNN_DATA_FLOAT32, {1, 1, 1024}, {1024, 1024, 1}, 0},
        {2, FLAGDNN_DATA_FLOAT32, {1, 1, 1024}, {1024, 1024, 1}, 0},
        {3, FLAGDNN_DATA_FLOAT32, {1, 1, 1024}, {1024, 1024, 1}, 0},
    };
    hv::HipdnnPointwiseOperation reference_operation;
    reference_operation.kind = hv::HipdnnPointwiseKind::kAddSquare;
    reference_operation.unavailable_reason.clear();
    const hv::HipdnnCapability capability = hv::hipdnn_pointwise_capability(
        reference_operation, reference_tensors, true);
    hv::require_valid_hipdnn_adapter_contract(capability,
                                              "hygon graph add_square");
    if (!capability.supported) {
      std::cout << "[SKIP][hipdnn] op=graph_add_square "
                   "case=hygon_mul_add_chain reason="
                << capability.reason << ' ' << hv::hipdnn_environment() << '\n';
      return 77;
    }

    hv::HipdnnPointwisePlan reference(reference_operation, reference_tensors);
    hv::DeviceBuffer reference_left(bytes);
    hv::DeviceBuffer reference_right(bytes);
    hv::DeviceBuffer reference_output(bytes);
    hv::DeviceBuffer reference_workspace(reference.workspace_size());
    reference_left.copy_from_host(encoded_left.data(), bytes, stream.get());
    reference_right.copy_from_host(encoded_right.data(), bytes, stream.get());
    const std::array<flagdnnBinding_t, 3> reference_bindings = {
        flagdnnBinding_t{1, reference_left.opaque()},
        flagdnnBinding_t{2, reference_right.opaque()},
        flagdnnBinding_t{3, reference_output.opaque()},
    };
    reference.execute(reference_bindings, reference_workspace.opaque(),
                      reference.workspace_size(), stream.opaque());
    stream.synchronize();

    const std::vector<float> actual =
        read_output(flagdnn_output, host_left.size(), stream);
    const std::vector<float> expected =
        read_output(reference_output, host_left.size(), stream);
    const double maximum_error = maximum_difference(actual, expected);
    if (maximum_error != 0.0) {
      throw std::runtime_error(
          "multi-operation Graph differs from hipDNN AddSquare sequence");
    }

    require_native_process_clean("after_execute");
    std::cout << "PASS Hygon multi_operation_graph operations=2 workspace="
              << workspace_size << " virtual_alignment=" << kVirtualAlignment
              << " hipdnn_max_abs_error=" << maximum_error << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
