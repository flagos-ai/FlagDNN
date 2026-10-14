// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "functional/pointwise_runner_support.hpp"
#include "cpu_validation.hpp"
#include "numeric_types.hpp"
#include "reference/cpu/index.hpp"

#include <flagdnn/flagdnn.hpp>
#include <flagdnn_frontend.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace fe = flagdnn_frontend;
namespace tv = flagdnn::validation::thead;
namespace functional = tv::functional;
namespace cpu = flagdnn::reference::cpu;
using flagdnn::testing::TestTensor;

void check(fe::error_t status) {
  if (status.is_bad()) throw std::runtime_error(status.get_message());
}

fe::DataType_t frontend_type(flagdnnDataType_t dtype) {
  switch (dtype) {
    case FLAGDNN_DATA_INT32: return fe::DataType_t::INT32;
    case FLAGDNN_DATA_FLOAT32: return fe::DataType_t::FLOAT;
    case FLAGDNN_DATA_FP8_E4M3: return fe::DataType_t::FP8_E4M3;
    default: throw std::invalid_argument("unexpected index contract dtype");
  }
}

// These helpers only bind packed reference values to a strided device buffer.
// Operator semantics live exclusively in reference/cpu/index.cpp.
std::size_t allocation_size(const TestTensor& tensor) {
  const auto width = tv::element_size(tensor.data_type);
  const auto count = functional::storage_element_count(tensor);
  constexpr std::size_t guard = 64;
  const auto maximum = std::numeric_limits<std::size_t>::max();
  if (tensor.binding_byte_offset > maximum - guard ||
      count > (maximum - guard - tensor.binding_byte_offset) / width)
    throw std::overflow_error("index contract allocation size overflow");
  return tensor.binding_byte_offset + count * width + guard;
}

std::vector<std::byte> scatter(std::span<const std::byte> logical,
                               const TestTensor& tensor, std::byte fill) {
  const auto width = tv::element_size(tensor.data_type);
  const auto count = functional::element_count(tensor);
  if (count > std::numeric_limits<std::size_t>::max() / width ||
      logical.size() != count * width)
    throw std::invalid_argument("index contract scatter shape mismatch");
  std::vector<std::byte> bytes(allocation_size(tensor), fill);
  for (std::size_t i = 0; i < count; ++i) {
    // allocation_size checked the full positive-stride address span.
    auto remaining = i;
    std::size_t physical = 0;
    for (auto axis = tensor.dimensions.size(); axis != 0; --axis) {
      const auto extent = static_cast<std::size_t>(tensor.dimensions[axis - 1]);
      physical += (remaining % extent) *
                  static_cast<std::size_t>(tensor.strides[axis - 1]);
      remaining /= extent;
    }
    const auto offset = tensor.binding_byte_offset + physical * width;
    std::memcpy(bytes.data() + offset, logical.data() + i * width, width);
  }
  return bytes;
}

std::vector<std::byte> fixture(const TestTensor& tensor, std::size_t input,
                               unsigned replay) {
  constexpr std::array<std::uint32_t, 6> patterns = {
      0x80000001U, 0x7fc00001U, 0x00000001U,
      0xff800000U, 0x7f800000U, 0xffffffffU};
  const auto width = tv::element_size(tensor.data_type);
  std::vector<std::byte> bytes(functional::element_count(tensor) * width);
  for (std::size_t i = 0; i < bytes.size() / width; ++i) {
    const auto value = patterns[(input * 3 + i) % patterns.size()] ^
                       (replay * 0x13579bdfU);
    std::memcpy(bytes.data() + i * width, &value, width);
  }
  return bytes;
}

void require_equal(std::span<const std::byte> actual,
                    std::span<const std::byte> expected,
                    const std::string& description) {
  if (actual.size() != expected.size())
    throw std::runtime_error(description + " buffer length differs");
  const auto mismatch = std::mismatch(actual.begin(), actual.end(),
                                      expected.begin());
  if (mismatch.first != actual.end())
    throw std::runtime_error(description + " differs at byte " +
                             std::to_string(mismatch.first - actual.begin()));
}

void run_case(flagdnn::Handle& handle, tv::DeviceStream& stream,
               flagdnnDataType_t dtype, bool repeated,
               const std::string& name) {
  const std::vector<TestTensor> tensors = {
      {1, dtype, {repeated ? 2 : 3}, {2}, 16},
      {2, dtype, {repeated ? 3 : 1},
       {repeated ? 3 : (std::int64_t{1} << 62)}, 32},
      {3, dtype, {repeated ? 7 : 4}, {3}, 48},
  };
  const std::vector<std::size_t> port_indices =
      repeated ? std::vector<std::size_t>{0, 1, 0}
               : std::vector<std::size_t>{0, 1};
  fe::graph::Graph graph;
  graph.set_name(name).set_compute_data_type(fe::DataType_t::FLOAT);
  std::vector<fe::graph::Graph::Tensor> unique_inputs;
  // Create each frontend input exactly once; repeated ports share its UID.
  for (std::size_t i = 0; i < 2; ++i)
    unique_inputs.push_back(graph.tensor(
        fe::graph::Tensor_attributes()
            .set_uid(tensors[i].uid)
            .set_data_type(frontend_type(dtype))
            .set_dim(tensors[i].dimensions)
            .set_stride(tensors[i].strides)));
  std::vector<fe::graph::Graph::Tensor> inputs;
  cpu::IndexParameters parameters;
  parameters.operation = "concatenate";
  parameters.axis = -1;
  parameters.output_shape = tensors.back().dimensions;
  for (const auto index : port_indices) {
    inputs.push_back(unique_inputs[index]);
    parameters.input_shapes.push_back(tensors[index].dimensions);
  }
  auto output = graph.concatenate(
      inputs, fe::graph::Concatenate_attributes().set_axis(-1));
  output->set_uid(tensors.back().uid)
      .set_data_type(frontend_type(dtype))
      .set_dim(tensors.back().dimensions)
      .set_stride(tensors.back().strides)
      .set_output(true);
  check(graph.build(handle, {fe::HeurMode_t::A}));
  std::int64_t workspace_bytes = 0;
  check(graph.get_workspace_size(workspace_bytes));
  if (workspace_bytes < 0)
    throw std::runtime_error("index contract workspace size is negative");
  tv::DeviceBuffer workspace(static_cast<std::size_t>(workspace_bytes));
  std::vector<std::unique_ptr<tv::DeviceBuffer>> buffers;
  std::vector<flagdnnBinding_t> bindings;
  for (const auto& tensor : tensors) {
    buffers.push_back(std::make_unique<tv::DeviceBuffer>(allocation_size(tensor)));
    bindings.push_back({tensor.uid,
                       buffers.back()->at(tensor.binding_byte_offset)});
  }
  // Replay the same graph and bindings with changed input bits and newly
  // poisoned output. Compare every byte, including binding guards and padding.
  for (unsigned replay = 0; replay < 2; ++replay) {
    const std::byte poison = replay ? std::byte{0x5a} : std::byte{0xa5};
    std::vector<std::vector<std::byte>> logical;
    std::vector<std::vector<std::byte>> expected;
    for (std::size_t i = 0; i < 2; ++i) {
      logical.push_back(fixture(tensors[i], i, replay));
      expected.push_back(scatter(logical.back(), tensors[i], poison));
    }
    std::vector<std::vector<std::byte>> oracle_inputs;
    for (const auto index : port_indices) oracle_inputs.push_back(logical[index]);
    const auto result = cpu::evaluate_index_bytes(parameters, oracle_inputs, dtype);
    expected.push_back(scatter(result, tensors.back(), poison));
    std::vector<std::byte> fresh_output(buffers.back()->size(), poison);
    for (std::size_t i = 0; i < 2; ++i)
      tv::copy_to_device_async(*buffers[i], expected[i], 0, stream.get());
    tv::copy_to_device_async(*buffers.back(), fresh_output, 0, stream.get());
    tv::check_driver(cuStreamSynchronize(stream.get()), "index fixture upload");
    check(graph.execute(handle, bindings, workspace.data(), workspace.size(),
                         stream.opaque()));
    std::vector<std::vector<std::byte>> actual;
    for (const auto& buffer : buffers) {
      actual.emplace_back(buffer->size());
      tv::copy_from_device_async(actual.back(), *buffer, 0, stream.get());
    }
    tv::check_driver(cuStreamSynchronize(stream.get()), "index contract replay");
    for (std::size_t i = 0; i < buffers.size(); ++i)
      require_equal(actual[i], expected[i], name + " replay=" +
                      std::to_string(replay) + " uid=" +
                      std::to_string(tensors[i].uid));
  }
  std::cout << name << ": PASS raw_bytes=1 input_immutable=1 "
               "padding_unchanged=1 fresh_replays=2\n";
}
// Deliberately omit the last output byte to verify that the validation
// harness catches a missed store even when its expected value is the old fill
// sentinel. This executable only copies device bytes; the CPU oracle remains
// in reference/cpu/index.cpp.
class PrefixCopyExecutable final : public flagdnn::benchmarking::BenchmarkExecutable {
 public:
  explicit PrefixCopyExecutable(std::size_t bytes) : bytes_(bytes) {}
  std::size_t workspace_size() const noexcept override { return 0; }
  void execute(std::span<const flagdnnBinding_t> bindings, void*, std::size_t,
               flagdnnStream_t stream) override {
    if (bindings.size() != 2)
      throw std::invalid_argument("prefix copy contract binding count differs");
    tv::check_driver(cuMemcpyDtoDAsync(
        reinterpret_cast<CUdeviceptr>(bindings[1].device_pointer),
        reinterpret_cast<CUdeviceptr>(bindings[0].device_pointer), bytes_,
        reinterpret_cast<CUstream>(stream)), "prefix copy contract execute");
  }
 private:
  std::size_t bytes_;
};

void reject_unwritten_index_byte(tv::DeviceStream& stream) {
  namespace bm = flagdnn::benchmarking;
  constexpr std::size_t count = 130;
  // The raw fixture's final byte is (129 * 37) % 256 == 0xa5.
  // With the previous fixed fill, skipping that store silently passed.
  static_assert(((count - 1) * 37) % 256 == 0xa5);
  for (const auto dtype : {FLAGDNN_DATA_FP8_E4M3, FLAGDNN_DATA_FP8_E5M2,
                           FLAGDNN_DATA_FP8_E8M0}) {
    bm::BenchmarkCase spec;
    spec.name = "cpu_index_reject_unwritten_byte_" + std::to_string(dtype);
    spec.operation = bm::Operation::kGraph;
    spec.tensors = {{1, dtype, {count}, {1}, 16},
                    {2, dtype, {count}, {1}, 32}};
    const bm::thead::NativeBaselineMetadata metadata{"concatenate", 0};
    PrefixCopyExecutable complete(count);
    (void)tv::run_cpu_validated_case(spec, complete, stream, &metadata);
    PrefixCopyExecutable incomplete(count - 1);
    bool rejected = false;
    try {
      (void)tv::run_cpu_validated_case(spec, incomplete, stream, &metadata);
    } catch (const std::runtime_error& error) {
      if (std::string(error.what()).find(": CPU byte oracle differs") ==
          std::string::npos) throw;
      rejected = true;
    }
    if (!rejected)
      throw std::runtime_error("unwritten index byte equal to 0xa5 passed");
  }
  std::cout << "cpu_index_validation: PASS rejected_partial=3\n";
}

void run_gen_index_case(flagdnn::Handle& handle, tv::DeviceStream& stream) {
  const TestTensor spec{1, FLAGDNN_DATA_INT32, {2, 2}, {2, 3}, 16};
  fe::graph::Graph graph;
  graph.set_name("gen_index_compact_stride")
      .set_compute_data_type(fe::DataType_t::FLOAT);
  auto output = graph.gen_index(fe::graph::Gen_index_attributes()
                                    .set_dim(spec.dimensions)
                                    .set_axis(-1)
                                    .set_data_type(fe::DataType_t::INT32));
  output->set_uid(spec.uid).set_stride(spec.strides).set_output(true);
  check(graph.build(handle, {fe::HeurMode_t::A}));
  std::int64_t workspace_bytes = 0;
  check(graph.get_workspace_size(workspace_bytes));
  if (workspace_bytes < 0)
    throw std::runtime_error("index contract workspace size is negative");
  tv::DeviceBuffer workspace(static_cast<std::size_t>(workspace_bytes));
  tv::DeviceBuffer buffer(allocation_size(spec));
  const std::array<flagdnnBinding_t, 1> bindings = {
      flagdnnBinding_t{spec.uid, buffer.at(spec.binding_byte_offset)}};
  const cpu::IndexParameters parameters{"gen_index", {}, spec.dimensions, -1};
  const auto logical = cpu::evaluate_index_bytes(parameters, {}, spec.data_type);
  for (unsigned replay = 0; replay < 2; ++replay) {
    const std::byte poison = replay ? std::byte{0x5a} : std::byte{0xa5};
    const auto expected = scatter(logical, spec, poison);
    std::vector<std::byte> fresh(buffer.size(), poison), actual(buffer.size());
    tv::copy_to_device_async(buffer, fresh, 0, stream.get());
    tv::check_driver(cuStreamSynchronize(stream.get()), "index fixture upload");
    check(graph.execute(handle, bindings, workspace.data(), workspace.size(),
                         stream.opaque()));
    tv::copy_from_device_async(actual, buffer, 0, stream.get());
    tv::check_driver(cuStreamSynchronize(stream.get()), "gen_index replay");
    require_equal(actual, expected, "gen_index_compact_stride replay=" +
                                    std::to_string(replay));
  }
  std::cout << "gen_index_compact_stride: PASS raw_bytes=1 "
               "padding_unchanged=1 fresh_replays=2\n";
}
}  // namespace

int main(int argc, char** argv) {
  constexpr const char* suite = "FLAGDNN_THEAD_INDEX_RUNTIME_CONTRACT";
  try {
    if (argc != 3)
      throw std::invalid_argument("requires COMPILER_EXECUTABLE COMPILER_ENTRY");
    tv::check_driver(cuInit(0), "cuInit");
    CUdevice device = 0;
    tv::check_driver(cuDeviceGet(&device, 0), "cuDeviceGet");
    tv::PrimaryContext primary(device);
    tv::ScopedCurrentContext current(primary.get());
    tv::DeviceStream stream;
    functional::TemporaryCache cache;
    flagdnn::Handle handle("thead", 0);
    handle.set_compiler(argv[1], argv[2], cache.path().string());
    run_case(handle, stream, FLAGDNN_DATA_INT32, true, "repeat_int32");
    run_case(handle, stream, FLAGDNN_DATA_FP8_E4M3, true, "repeat_fp8_e4m3");
    run_case(handle, stream, FLAGDNN_DATA_INT32, false, "singleton_stride_int32");
    run_case(handle, stream, FLAGDNN_DATA_FLOAT32, false, "singleton_stride_fp32");
    run_gen_index_case(handle, stream);
    reject_unwritten_index_byte(stream);
    std::cout << suite << ": PASS cases=5 rejected_partial=3\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << suite << ": FAIL reason=" << error.what() << '\n';
    return 1;
  }
}
