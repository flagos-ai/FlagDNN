/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include <flagdnn/frontend.hpp>

#include "hip_driver.hpp"
#include "pointwise_reference.hpp"
#include "reference/cpu/pointwise.hpp"
#include "tensor_io.hpp"

#include <unistd.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace fe = flagdnn_frontend;
namespace hv = flagdnn::validation::hygon;
namespace io = flagdnn::validation::hygon::tensor_io;

constexpr std::size_t kElementCount = 1024;
constexpr std::array<std::int64_t, 3> kDimensions = {1, 1, 1024};
constexpr std::array<std::int64_t, 3> kStrides = {1024, 1024, 1};
constexpr std::uint8_t kOutputSentinel = 0xA5;

void check_frontend(const fe::error_t &status, const char *operation) {
  if (status.is_bad()) {
    throw std::runtime_error(std::string(operation) +
                             " failed: " + status.get_message());
  }
}

class TemporaryCache final {
public:
  explicit TemporaryCache(std::string_view operation) {
    std::string pattern =
        (std::filesystem::temp_directory_path() /
         ("flagdnn-hygon-" + std::string(operation) + "-smoke-XXXXXX"))
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

struct SmokeCase {
  std::string_view name;
  fe::PointwiseMode_t mode;
  bool boolean_output;
};

struct CacheInventory {
  std::size_t manifests = 0;
  std::size_t generated_sources = 0;
  std::size_t selections = 0;
};

template <std::size_t Rank>
std::size_t
storage_element_count(const std::array<std::int64_t, Rank> &dimensions,
                      const std::array<std::int64_t, Rank> &strides) {
  std::size_t maximum_offset = 0;
  for (std::size_t axis = 0; axis < Rank; ++axis) {
    maximum_offset += static_cast<std::size_t>(dimensions[axis] - 1) *
                      static_cast<std::size_t>(strides[axis]);
  }
  return maximum_offset + 1;
}

template <std::size_t Rank>
std::size_t logical_offset(std::size_t logical_index,
                           const std::array<std::int64_t, Rank> &dimensions,
                           const std::array<std::int64_t, Rank> &strides) {
  std::size_t offset = 0;
  for (std::size_t axis = Rank; axis > 0; --axis) {
    const std::size_t dimension =
        static_cast<std::size_t>(dimensions[axis - 1]);
    const std::size_t coordinate = logical_index % dimension;
    logical_index /= dimension;
    offset += coordinate * static_cast<std::size_t>(strides[axis - 1]);
  }
  return offset;
}

CacheInventory inspect_cache(const std::filesystem::path &path) {
  CacheInventory result;
  for (const auto &entry :
       std::filesystem::recursive_directory_iterator(path)) {
    if (!entry.is_regular_file()) {
      continue;
    }
    const std::string filename = entry.path().filename().string();
    if (filename == "manifest.json") {
      ++result.manifests;
    }
    if (filename.starts_with("generated_stage_") &&
        entry.path().extension() == ".py") {
      ++result.generated_sources;
    }
    if (filename.starts_with(".flagdnn-autotune-v1-stage-")) {
      ++result.selections;
    }
  }
  return result;
}

std::array<std::vector<float>, 2> make_inputs(std::string_view operation) {
  std::array<std::vector<float>, 2> result = {
      std::vector<float>(kElementCount),
      std::vector<float>(kElementCount),
  };
  constexpr std::array<float, 6> kModuloLeft = {-3.0F, -3.0F, 3.0F,
                                                3.0F,  -5.5F, 5.5F};
  constexpr std::array<float, 6> kModuloRight = {2.0F,  -2.0F, 2.0F,
                                                 -2.0F, 2.25F, -2.25F};
  for (std::size_t index = 0; index < kElementCount; ++index) {
    const int centered = static_cast<int>((index * 17) % 41) - 20;
    const float real = static_cast<float>(centered) / 13.0F;
    if (operation == "div") {
      result[0][index] = real;
      result[1][index] = std::abs(real) + 0.5F;
    } else if (operation == "pow") {
      if (index < 256) {
        result[0][index] = std::abs(real) + 0.5F;
        result[1][index] = static_cast<float>((index % 8) + 1) / 8.0F;
      } else if (index < 512) {
        result[0][index] =
            1.0F + static_cast<float>(static_cast<int>(index) - 384) * 0x1p-23F;
        result[1][index] = index % 2 == 0 ? 50000.0F : -50000.0F;
      } else {
        const std::array<float, 16> values = {
            0.0F,
            -0.0F,
            1.0F,
            -1.0F,
            2.0F,
            -2.0F,
            0.5F,
            -0.5F,
            std::numeric_limits<float>::infinity(),
            -std::numeric_limits<float>::infinity(),
            std::numeric_limits<float>::quiet_NaN(),
            std::numeric_limits<float>::min(),
            std::numeric_limits<float>::denorm_min(),
            std::numeric_limits<float>::max(),
            0x1p24F,
            -0x1p24F};
        result[0][index] = values[(index / values.size()) % values.size()];
        result[1][index] = values[index % values.size()];
      }
    } else if (operation == "mod") {
      result[0][index] = kModuloLeft[index % kModuloLeft.size()];
      result[1][index] = kModuloRight[index % kModuloRight.size()];
    } else if (operation == "cmp_eq") {
      result[0][index] = real;
      result[1][index] = index % 3 == 0 ? real : real + 0.25F;
    } else if (operation == "min" || operation == "max") {
      const float nan = std::numeric_limits<float>::quiet_NaN();
      constexpr std::array<float, 8> kLeft = {0.0F, -0.0F, 1.0F,  nan,
                                              3.0F, nan,   -4.0F, 7.0F};
      const std::array<float, 8> right = {-0.0F, 0.0F, -2.0F, 5.0F,
                                          nan,   nan,  9.0F,  -8.0F};
      result[0][index] = kLeft[index % kLeft.size()];
      result[1][index] = right[index % right.size()];
    } else {
      throw std::invalid_argument("unknown pointwise smoke operation");
    }
  }
  return result;
}

void compare_minmax_with_hipdnn(const SmokeCase &test_case,
                                const std::vector<std::uint8_t> &flagdnn_bytes,
                                flagdnnDataType_t data_type,
                                hv::DeviceBuffer &left_buffer,
                                hv::DeviceBuffer &right_buffer,
                                hv::Stream &stream) {
  if (test_case.name != "min" && test_case.name != "max") {
    return;
  }
  hv::HipdnnPointwiseOperation operation;
  operation.kind = test_case.name == "min" ? hv::HipdnnPointwiseKind::kMin
                                           : hv::HipdnnPointwiseKind::kMax;
  const std::vector<std::int64_t> dimensions(kDimensions.begin(),
                                             kDimensions.end());
  const std::vector<std::int64_t> strides(kStrides.begin(), kStrides.end());
  const std::vector<hv::ReferenceTensor> tensors = {
      {1, data_type, dimensions, strides, 0},
      {2, data_type, dimensions, strides, 0},
      {3, data_type, dimensions, strides, 0},
  };
  hv::HipdnnPointwisePlan reference(operation, tensors);
  std::vector<std::uint8_t> reference_bytes(kElementCount *
                                            io::data_type_size(data_type));
  hv::DeviceBuffer reference_buffer(reference_bytes.size());
  hv::DeviceBuffer reference_workspace(reference.workspace_size());
  const std::array<flagdnnBinding_t, 3> bindings = {
      flagdnnBinding_t{1, left_buffer.opaque()},
      flagdnnBinding_t{2, right_buffer.opaque()},
      flagdnnBinding_t{3, reference_buffer.opaque()},
  };
  reference.execute(bindings, reference_workspace.opaque(),
                    reference.workspace_size(), stream.opaque());
  reference_buffer.copy_to_host(reference_bytes.data(), reference_bytes.size(),
                                stream.get());
  stream.synchronize();

  const std::vector<float> flagdnn_output =
      io::decode(flagdnn_bytes, data_type, kElementCount);
  const std::vector<float> reference_output =
      io::decode(reference_bytes, data_type, kElementCount);
  for (std::size_t index = 0; index < kElementCount; ++index) {
    const float actual = flagdnn_output[index];
    const float expected = reference_output[index];
    bool equal = false;
    if (std::isnan(actual) || std::isnan(expected)) {
      equal = std::isnan(actual) && std::isnan(expected);
    } else if (actual == 0.0F || expected == 0.0F) {
      equal = actual == 0.0F && expected == 0.0F &&
              std::signbit(actual) == std::signbit(expected);
    } else {
      equal = actual == expected;
    }
    if (!equal) {
      throw std::runtime_error(
          std::string(test_case.name) +
          " differs from hipDNN NaN/signed-zero semantics at index " +
          std::to_string(index) +
          " (FlagDNN signbit=" + std::to_string(std::signbit(actual)) +
          " isnan=" + std::to_string(std::isnan(actual)) +
          " hipDNN signbit=" + std::to_string(std::signbit(expected)) +
          " isnan=" + std::to_string(std::isnan(expected)) + ")");
    }
  }
}

void run_case(const SmokeCase &test_case, const char *compiler_executable,
              const char *compiler_entry, hv::Stream &stream,
              fe::DataType_t frontend_data_type = fe::DataType_t::FLOAT,
              flagdnnDataType_t data_type = FLAGDNN_DATA_FLOAT32) {
  const std::string data_type_name =
      data_type == FLAGDNN_DATA_FLOAT16 ? "fp16" : "fp32";
  TemporaryCache cache(std::string(test_case.name) + "_" + data_type_name);
  // Fixed 256-element POW tiles exercise both the positive-finite path and
  // OCML fallback in the same execution. Other smoke cases cover autotuning.
  const bool autotune = test_case.name != "pow";
  flagdnn::Handle handle("hygon", 0);
  handle.set_compiler(compiler_executable, compiler_entry,
                      cache.path().string());

  fe::graph::Graph graph;
  graph
      .set_name("hygon_" + std::string(test_case.name) + "_" + data_type_name +
                "_launch_smoke")
      .set_io_data_type(frontend_data_type)
      .set_intermediate_data_type(fe::DataType_t::FLOAT)
      .set_compute_data_type(fe::DataType_t::FLOAT)
      .set_autotune(autotune);
  const auto left =
      graph.tensor(fe::graph::Tensor_attributes()
                       .set_name("left")
                       .set_uid(1)
                       .set_data_type(frontend_data_type)
                       .set_dim({kDimensions.begin(), kDimensions.end()})
                       .set_stride({kStrides.begin(), kStrides.end()}));
  const auto right =
      graph.tensor(fe::graph::Tensor_attributes()
                       .set_name("right")
                       .set_uid(2)
                       .set_data_type(frontend_data_type)
                       .set_dim({kDimensions.begin(), kDimensions.end()})
                       .set_stride({kStrides.begin(), kStrides.end()}));
  auto output =
      graph.pointwise(left, right,
                      fe::graph::Pointwise_attributes()
                          .set_name(std::string(test_case.name))
                          .set_mode(test_case.mode)
                          .set_compute_data_type(test_case.boolean_output
                                                     ? fe::DataType_t::BOOLEAN
                                                     : fe::DataType_t::FLOAT));
  output->set_name("output")
      .set_uid(3)
      .set_data_type(test_case.boolean_output ? fe::DataType_t::BOOLEAN
                                              : frontend_data_type)
      .set_dim({kDimensions.begin(), kDimensions.end()})
      .set_stride({kStrides.begin(), kStrides.end()})
      .set_output(true);

  check_frontend(graph.build(handle, {fe::HeurMode_t::A}), "graph.build");
  std::int64_t workspace_bytes = 0;
  check_frontend(graph.get_workspace_size(workspace_bytes),
                 "graph.get_workspace_size");
  if (workspace_bytes <= 0) {
    throw std::runtime_error("libtriton_jit smoke workspace is empty");
  }

  const auto logical_inputs = make_inputs(test_case.name);
  const std::array<std::vector<std::uint8_t>, 2> encoded_inputs = {
      io::encode(logical_inputs[0], data_type),
      io::encode(logical_inputs[1], data_type),
  };
  const std::size_t input_bytes = encoded_inputs[0].size();
  const std::size_t output_bytes =
      test_case.boolean_output ? kElementCount
                               : kElementCount * io::data_type_size(data_type);
  std::vector<std::uint8_t> output_host(output_bytes, kOutputSentinel);

  hv::DeviceBuffer left_buffer(input_bytes);
  hv::DeviceBuffer right_buffer(input_bytes);
  hv::DeviceBuffer output_buffer(output_bytes);
  hv::DeviceBuffer workspace(static_cast<std::size_t>(workspace_bytes));
  left_buffer.copy_from_host(encoded_inputs[0].data(), input_bytes,
                             stream.get());
  right_buffer.copy_from_host(encoded_inputs[1].data(), input_bytes,
                              stream.get());
  output_buffer.copy_from_host(output_host.data(), output_host.size(),
                               stream.get());
  const fe::VariantPack bindings = {
      {1, left_buffer.opaque()},
      {2, right_buffer.opaque()},
      {3, output_buffer.opaque()},
  };
  check_frontend(
      graph.execute(handle, bindings, workspace.opaque(), stream.opaque()),
      "graph.execute");
  output_buffer.copy_to_host(output_host.data(), output_host.size(),
                             stream.get());
  stream.synchronize();
  if (std::all_of(output_host.begin(), output_host.end(),
                  [](std::uint8_t byte) { return byte == kOutputSentinel; })) {
    throw std::runtime_error(std::string(test_case.name) +
                             " launch did not write its output");
  }
  compare_minmax_with_hipdnn(test_case, output_host, data_type, left_buffer,
                             right_buffer, stream);
  if (test_case.name == "pow") {
    const auto left_values =
        io::decode(encoded_inputs[0], data_type, kElementCount);
    const auto right_values =
        io::decode(encoded_inputs[1], data_type, kElementCount);
    const auto expected = flagdnn::reference::cpu::evaluate_binary_pointwise(
        FLAGDNN_POINTWISE_POW, left_values, kDimensions, right_values,
        kDimensions, kDimensions);
    const auto actual = io::decode(output_host, data_type, kElementCount);
    for (std::size_t index = 0; index < expected.size(); ++index) {
      bool equal = false;
      if (std::isnan(expected[index])) {
        equal = std::isnan(actual[index]);
      } else if (expected[index] == 0.0F || std::isinf(expected[index])) {
        equal = actual[index] == expected[index] &&
                std::signbit(actual[index]) == std::signbit(expected[index]);
      } else if (std::isfinite(actual[index])) {
        const auto actual_bits = std::bit_cast<std::uint32_t>(actual[index]);
        const auto expected_bits =
            std::bit_cast<std::uint32_t>(expected[index]);
        const auto distance = actual_bits > expected_bits
                                  ? actual_bits - expected_bits
                                  : expected_bits - actual_bits;
        equal = distance <= 2;
      }
      if (!equal) {
        throw std::runtime_error(
            "POW differs from validated CPU reference at index " +
            std::to_string(index));
      }
    }
  }

  const CacheInventory inventory = inspect_cache(cache.path());
  if (inventory.manifests != 1 || inventory.generated_sources != 1 ||
      inventory.selections != (autotune ? 1U : 0U)) {
    throw std::runtime_error(
        std::string(test_case.name) +
        " produced an unexpected manifest/source/autotune inventory");
  }
  std::cout << "PASS production_chain_smoke op=" << test_case.name
            << " dtype=" << data_type_name << " workspace=" << workspace_bytes
            << " manifests=" << inventory.manifests
            << " generated_sources=" << inventory.generated_sources
            << " selections=" << inventory.selections << " numeric_reference="
            << ((test_case.name == "min" || test_case.name == "max") ? "hipdnn"
                : test_case.name == "pow"                            ? "cpu"
                                                                     : "none")
            << '\n';
}

hv::HipdnnPointwiseKind hipdnn_kind(fe::PointwiseMode_t mode) {
  switch (mode) {
  case fe::PointwiseMode_t::ADD:
    return hv::HipdnnPointwiseKind::kAdd;
  case fe::PointwiseMode_t::SUB:
    return hv::HipdnnPointwiseKind::kSub;
  case fe::PointwiseMode_t::MUL:
    return hv::HipdnnPointwiseKind::kMul;
  case fe::PointwiseMode_t::MIN:
    return hv::HipdnnPointwiseKind::kMin;
  case fe::PointwiseMode_t::MAX:
    return hv::HipdnnPointwiseKind::kMax;
  default:
    throw std::invalid_argument("layout smoke mode has no hipDNN mapping");
  }
}

void run_strided_broadcast_case(const SmokeCase &test_case, double alpha,
                                const char *compiler_executable,
                                const char *compiler_entry,
                                hv::Stream &stream) {
  constexpr std::array<std::int64_t, 3> kLeftDimensions = {2, 3, 4};
  constexpr std::array<std::int64_t, 3> kLeftStrides = {31, 9, 2};
  constexpr std::array<std::int64_t, 2> kRightDimensions = {1, 4};
  constexpr std::array<std::int64_t, 2> kRightStrides = {13, 3};
  constexpr std::array<std::int64_t, 3> kOutputDimensions = {2, 3, 4};
  constexpr std::array<std::int64_t, 3> kOutputStrides = {37, 11, 2};
  constexpr std::size_t kLogicalElements = 24;
  constexpr float kPaddingSentinel = 12345.0F;

  TemporaryCache cache("layout_" + std::string(test_case.name));
  flagdnn::Handle handle("hygon", 0);
  handle.set_compiler(compiler_executable, compiler_entry,
                      cache.path().string());

  fe::graph::Graph graph;
  graph.set_name("hygon_" + std::string(test_case.name) + "_strided_broadcast")
      .set_io_data_type(fe::DataType_t::FLOAT)
      .set_intermediate_data_type(fe::DataType_t::FLOAT)
      .set_compute_data_type(fe::DataType_t::FLOAT)
      .set_autotune(true);
  const auto left = graph.tensor(
      fe::graph::Tensor_attributes()
          .set_name("left")
          .set_uid(11)
          .set_data_type(fe::DataType_t::FLOAT)
          .set_dim({kLeftDimensions.begin(), kLeftDimensions.end()})
          .set_stride({kLeftStrides.begin(), kLeftStrides.end()}));
  const auto right = graph.tensor(
      fe::graph::Tensor_attributes()
          .set_name("right")
          .set_uid(12)
          .set_data_type(fe::DataType_t::FLOAT)
          .set_dim({kRightDimensions.begin(), kRightDimensions.end()})
          .set_stride({kRightStrides.begin(), kRightStrides.end()}));
  auto output = graph.pointwise(
      left, right,
      fe::graph::Pointwise_attributes()
          .set_name(std::string(test_case.name) + "_strided_broadcast")
          .set_mode(test_case.mode)
          .set_compute_data_type(fe::DataType_t::FLOAT)
          .set_alpha(alpha));
  output->set_name("output")
      .set_uid(13)
      .set_data_type(fe::DataType_t::FLOAT)
      .set_dim({kOutputDimensions.begin(), kOutputDimensions.end()})
      .set_stride({kOutputStrides.begin(), kOutputStrides.end()})
      .set_output(true);
  check_frontend(graph.build(handle, {fe::HeurMode_t::A}),
                 "strided graph.build");
  std::int64_t workspace_bytes = 0;
  check_frontend(graph.get_workspace_size(workspace_bytes),
                 "strided graph.get_workspace_size");
  if (workspace_bytes <= 0) {
    throw std::runtime_error("strided libtriton_jit workspace is empty");
  }

  std::vector<float> left_host(
      storage_element_count(kLeftDimensions, kLeftStrides), -1000.0F);
  std::vector<float> right_host(
      storage_element_count(kRightDimensions, kRightStrides), -2000.0F);
  std::vector<float> reference_left_host(kLogicalElements);
  std::vector<float> reference_right_host(kLogicalElements);
  std::vector<float> output_host(
      storage_element_count(kOutputDimensions, kOutputStrides),
      kPaddingSentinel);
  std::vector<float> reference_host(output_host.size(), kPaddingSentinel);
  std::vector<bool> output_positions(output_host.size(), false);
  for (std::size_t index = 0; index < kLogicalElements; ++index) {
    left_host[logical_offset(index, kLeftDimensions, kLeftStrides)] =
        static_cast<float>(static_cast<int>(index) - 12) / 7.0F;
    reference_left_host[index] =
        left_host[logical_offset(index, kLeftDimensions, kLeftStrides)];
    const std::size_t output_offset =
        logical_offset(index, kOutputDimensions, kOutputStrides);
    output_positions[output_offset] = true;
  }
  for (std::size_t index = 0; index < 4; ++index) {
    const float value = static_cast<float>(static_cast<int>(index) - 2) / 5.0F;
    right_host[logical_offset(index, kRightDimensions, kRightStrides)] = value;
  }
  for (std::size_t index = 0; index < kLogicalElements; ++index) {
    reference_right_host[index] =
        right_host[logical_offset(index % 4, kRightDimensions, kRightStrides)];
  }

  hv::DeviceBuffer left_buffer(left_host.size() * sizeof(float));
  hv::DeviceBuffer reference_left_buffer(reference_left_host.size() *
                                         sizeof(float));
  hv::DeviceBuffer right_buffer(right_host.size() * sizeof(float));
  hv::DeviceBuffer reference_right_buffer(reference_right_host.size() *
                                          sizeof(float));
  hv::DeviceBuffer output_buffer(output_host.size() * sizeof(float));
  hv::DeviceBuffer workspace(static_cast<std::size_t>(workspace_bytes));
  left_buffer.copy_from_host(left_host.data(), left_host.size() * sizeof(float),
                             stream.get());
  reference_left_buffer.copy_from_host(
      reference_left_host.data(), reference_left_host.size() * sizeof(float),
      stream.get());
  right_buffer.copy_from_host(right_host.data(),
                              right_host.size() * sizeof(float), stream.get());
  reference_right_buffer.copy_from_host(
      reference_right_host.data(), reference_right_host.size() * sizeof(float),
      stream.get());
  output_buffer.copy_from_host(
      output_host.data(), output_host.size() * sizeof(float), stream.get());
  const fe::VariantPack bindings = {
      {11, left_buffer.opaque()},
      {12, right_buffer.opaque()},
      {13, output_buffer.opaque()},
  };
  check_frontend(
      graph.execute(handle, bindings, workspace.opaque(), stream.opaque()),
      "strided graph.execute");

  hv::HipdnnPointwiseOperation reference_operation;
  reference_operation.kind = hipdnn_kind(test_case.mode);
  reference_operation.alpha = alpha;
  reference_operation.unavailable_reason.clear();
  const std::vector<hv::ReferenceTensor> reference_tensors = {
      {11,
       FLAGDNN_DATA_FLOAT32,
       {kOutputDimensions.begin(), kOutputDimensions.end()},
       {12, 4, 1},
       0},
      {12,
       FLAGDNN_DATA_FLOAT32,
       {kOutputDimensions.begin(), kOutputDimensions.end()},
       {12, 4, 1},
       0},
      {13,
       FLAGDNN_DATA_FLOAT32,
       {kOutputDimensions.begin(), kOutputDimensions.end()},
       {12, 4, 1},
       0},
  };
  const hv::HipdnnCapability capability = hv::hipdnn_pointwise_capability(
      reference_operation, reference_tensors, true);
  hv::require_valid_hipdnn_adapter_contract(capability, test_case.name);
  if (!capability.supported) {
    throw std::runtime_error(
        std::string(test_case.name) +
        " strided hipDNN reference unexpectedly unsupported: " +
        capability.reason);
  }
  hv::HipdnnPointwisePlan reference(reference_operation, reference_tensors);
  reference_host.resize(kLogicalElements);
  hv::DeviceBuffer reference_buffer(reference_host.size() * sizeof(float));
  hv::DeviceBuffer reference_workspace(reference.workspace_size());
  reference_buffer.copy_from_host(reference_host.data(),
                                  reference_host.size() * sizeof(float),
                                  stream.get());
  const std::array<flagdnnBinding_t, 3> reference_bindings = {
      flagdnnBinding_t{11, reference_left_buffer.opaque()},
      flagdnnBinding_t{12, reference_right_buffer.opaque()},
      flagdnnBinding_t{13, reference_buffer.opaque()},
  };
  reference.execute(reference_bindings, reference_workspace.opaque(),
                    reference.workspace_size(), stream.opaque());
  output_buffer.copy_to_host(output_host.data(),
                             output_host.size() * sizeof(float), stream.get());
  reference_buffer.copy_to_host(reference_host.data(),
                                reference_host.size() * sizeof(float),
                                stream.get());
  stream.synchronize();

  for (std::size_t index = 0; index < kLogicalElements; ++index) {
    const std::size_t offset =
        logical_offset(index, kOutputDimensions, kOutputStrides);
    const float actual = output_host[offset];
    const float expected = reference_host[index];
    if (std::isnan(actual) || std::isnan(expected)) {
      if (!(std::isnan(actual) && std::isnan(expected))) {
        throw std::runtime_error(std::string(test_case.name) +
                                 " strided NaN mismatch");
      }
    } else if (std::abs(actual - expected) > 1.0e-6F) {
      const std::size_t left_offset =
          logical_offset(index, kLeftDimensions, kLeftStrides);
      const std::size_t right_index = index % 4;
      const std::size_t right_offset =
          logical_offset(right_index, kRightDimensions, kRightStrides);
      throw std::runtime_error(
          std::string(test_case.name) +
          " strided result differs from hipDNN at logical index " +
          std::to_string(index) + " output_offset=" + std::to_string(offset) +
          " left=" + std::to_string(left_host[left_offset]) +
          " right=" + std::to_string(right_host[right_offset]) + " alpha=" +
          std::to_string(alpha) + " FlagDNN=" + std::to_string(actual) +
          " hipDNN=" + std::to_string(expected));
    }
  }
  for (std::size_t index = 0; index < output_host.size(); ++index) {
    if (!output_positions[index] && output_host[index] != kPaddingSentinel) {
      throw std::runtime_error(std::string(test_case.name) +
                               " strided output overwrote padding");
    }
  }
  const CacheInventory inventory = inspect_cache(cache.path());
  if (inventory.manifests != 1 || inventory.generated_sources != 1 ||
      inventory.selections != 1) {
    throw std::runtime_error(std::string(test_case.name) +
                             " strided cache inventory is invalid");
  }
  std::cout << "PASS strided_broadcast_hipdnn op=" << test_case.name
            << " alpha=" << alpha << " padding_untouched=true"
            << " workspace=" << workspace_bytes
            << " selections=" << inventory.selections
            << " hipdnn_abc=logical_equivalent_packed" << '\n';
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 3) {
      std::cerr << "usage: native_hygon_pointwise_smoke "
                   "COMPILER_EXECUTABLE COMPILER_ENTRY\n";
      return 2;
    }
    if (setenv("FLAGDNN_EXECUTION_ENGINE", "libtriton_jit", 1) != 0) {
      throw std::runtime_error("cannot select libtriton_jit engine");
    }
    hv::DeviceGuard device;
    hv::Stream stream;
    constexpr std::array<SmokeCase, 6> kCases = {{
        {"max", fe::PointwiseMode_t::MAX, false},
        {"min", fe::PointwiseMode_t::MIN, false},
        {"div", fe::PointwiseMode_t::DIV, false},
        {"pow", fe::PointwiseMode_t::POW, false},
        {"mod", fe::PointwiseMode_t::MOD, false},
        {"cmp_eq", fe::PointwiseMode_t::CMP_EQ, true},
    }};
    constexpr std::array<SmokeCase, 5> kLayoutCases = {{
        {"add", fe::PointwiseMode_t::ADD, false},
        {"sub", fe::PointwiseMode_t::SUB, false},
        {"mul", fe::PointwiseMode_t::MUL, false},
        {"min", fe::PointwiseMode_t::MIN, false},
        {"max", fe::PointwiseMode_t::MAX, false},
    }};
    constexpr std::array<double, kLayoutCases.size()> kLayoutAlphas = {
        -0.75, 0.5, 1.0, 1.0, 1.0};
    for (std::size_t index = 0; index < kLayoutCases.size(); ++index) {
      run_strided_broadcast_case(kLayoutCases[index], kLayoutAlphas[index],
                                 argv[1], argv[2], stream);
    }
    for (const SmokeCase &test_case : kCases) {
      run_case(test_case, argv[1], argv[2], stream);
    }
    constexpr std::array<SmokeCase, 2> kFp16SpecialCases = {{
        {"min", fe::PointwiseMode_t::MIN, false},
        {"max", fe::PointwiseMode_t::MAX, false},
    }};
    for (const SmokeCase &test_case : kFp16SpecialCases) {
      run_case(test_case, argv[1], argv[2], stream, fe::DataType_t::HALF,
               FLAGDNN_DATA_FLOAT16);
    }
    std::cout << "ALL_HYGON_POINTWISE_PRODUCTION_CHAIN_SMOKES_PASSED "
                 "minmax_fp32_fp16_reference=hipdnn "
                 "strided_broadcast_reference=hipdnn\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
