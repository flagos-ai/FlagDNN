// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#include "cpu_validation.hpp"

#include "numeric_types.hpp"
#include "reference/cpu/benchmark.hpp"
#include "reference/cpu/index.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>

namespace flagdnn::validation::thead {
namespace {
namespace bm = benchmarking;

std::size_t count(const bm::TensorSpec& t) {
  if (t.dimensions.empty() || t.dimensions.size() != t.strides.size())
    throw std::invalid_argument("invalid CPU validation tensor geometry");
  std::size_t n = 1;
  for (std::size_t i = 0; i < t.dimensions.size(); ++i) {
    if (t.dimensions[i] <= 0 || t.strides[i] <= 0 ||
        n > std::numeric_limits<std::size_t>::max() / t.dimensions[i])
      throw std::invalid_argument("invalid CPU validation tensor extent");
    n *= static_cast<std::size_t>(t.dimensions[i]);
  }
  return n;
}

std::size_t offset(std::size_t i, const bm::TensorSpec& t) {
  std::size_t result = 0;
  for (std::size_t axis = t.dimensions.size(); axis > 0; --axis) {
    const auto coordinate = i % static_cast<std::size_t>(t.dimensions[axis - 1]);
    const auto stride = static_cast<std::size_t>(t.strides[axis - 1]);
    if (coordinate && stride > (std::numeric_limits<std::size_t>::max() - result) / coordinate)
      throw std::overflow_error("CPU validation tensor offset overflows");
    result += coordinate * stride;
    i /= t.dimensions[axis - 1];
  }
  const auto width = element_size(t.data_type);
  if (result > (std::numeric_limits<std::size_t>::max() - t.binding_byte_offset) / width)
    throw std::overflow_error("CPU validation byte offset overflows");
  return t.binding_byte_offset + result * width;
}

struct Buffer {
  bm::TensorSpec spec;
  std::vector<std::byte> initial;
  std::unique_ptr<DeviceBuffer> device;
};

std::vector<std::byte> encode(flagdnnDataType_t type, std::span<const float> values) {
  if (type != FLAGDNN_DATA_INT32 && type != FLAGDNN_DATA_BOOLEAN)
    return encode_floating(type, values);
  std::vector<std::byte> bytes(values.size() * element_size(type));
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (type == FLAGDNN_DATA_BOOLEAN) bytes[i] = std::byte(values[i] != 0);
    else {
      const auto value = static_cast<std::int32_t>(values[i]);
      std::memcpy(bytes.data() + i * sizeof(value), &value, sizeof(value));
    }
  }
  return bytes;
}

std::vector<float> decode(flagdnnDataType_t type, std::span<const std::byte> bytes) {
  if (type != FLAGDNN_DATA_INT32 && type != FLAGDNN_DATA_BOOLEAN)
    return decode_floating(type, bytes);
  std::vector<float> result(bytes.size() / element_size(type));
  for (std::size_t i = 0; i < result.size(); ++i) {
    if (type == FLAGDNN_DATA_BOOLEAN) result[i] = std::to_integer<unsigned>(bytes[i]);
    else {
      std::int32_t value;
      std::memcpy(&value, bytes.data() + i * sizeof(value), sizeof(value));
      result[i] = static_cast<float>(value);
    }
  }
  return result;
}

std::vector<float> input_values(const bm::BenchmarkCase& spec, std::size_t input) {
  std::vector<float> values(count(spec.tensors.at(input)));
  const auto domain = spec.input_domains.empty() ? spec.input_domain
                                                : spec.input_domains.at(input);
  for (std::size_t i = 0; i < values.size(); ++i) {
    const auto v = (i * 17 + input * 11) % 41;
    switch (domain) {
      case bm::InputDomain::kPositive:
      case bm::InputDomain::kDivisor:
      case bm::InputDomain::kPower:
      case bm::InputDomain::kModulo:
        values[i] = static_cast<float>(v + 1) / static_cast<float>(13 + input); break;
      case bm::InputDomain::kLogical: values[i] = (i + input) % 3 != 0; break;
      case bm::InputDomain::kScaled: values[i] = (static_cast<int>(v) - 20) * .25F; break;
      case bm::InputDomain::kTan: values[i] = (static_cast<int>(v) - 20) / 32.F; break;
      default: values[i] = static_cast<float>(static_cast<int>(v) - 20) /
                           static_cast<float>(13 + input); break;
    }
  }
  return values;
}

class CapturedBatch {
 public:
  template<class Execute>
  CapturedBatch(DeviceStream& stream, int iterations, Execute execute) {
    check_driver(cuStreamBeginCapture(stream.get(), CU_STREAM_CAPTURE_MODE_RELAXED),
                 "equivalent begin capture");
    try {
      for (int i = 0; i < iterations; ++i) execute();
      check_driver(cuStreamEndCapture(stream.get(), &graph_), "equivalent end capture");
      check_driver(cuGraphInstantiate(&replay_, graph_, 0), "equivalent instantiate");
    } catch (...) {
      if (!graph_) (void)cuStreamEndCapture(stream.get(), &graph_);
      if (graph_) (void)cuGraphDestroy(graph_);
      throw;
    }
  }
  ~CapturedBatch() {
    if (replay_) (void)cuGraphExecDestroy(replay_);
    if (graph_) (void)cuGraphDestroy(graph_);
  }
  void launch(DeviceStream& stream) {
    check_driver(cuGraphLaunch(replay_, stream.get()), "equivalent replay");
  }
 private:
  CUgraph graph_ = nullptr;
  CUgraphExec replay_ = nullptr;
};
}  // namespace

std::vector<double> run_cpu_validated_case(
    const bm::BenchmarkCase& spec, bm::BenchmarkExecutable& executable,
    DeviceStream& stream, const bm::thead::NativeBaselineMetadata* metadata,
    bool benchmark) {
  const auto inputs = bm::input_tensor_count(spec);
  if (spec.output_count != 1) throw std::invalid_argument("CPU fallback requires one output");
  std::vector<Buffer> buffers;
  std::vector<std::vector<float>> logical_inputs;
  std::vector<flagdnnBinding_t> bindings;
  for (std::size_t k = 0; k < spec.tensors.size(); ++k) {
    const auto& t = spec.tensors[k];
    const auto n = count(t), width = element_size(t.data_type);
    if (n > std::numeric_limits<std::size_t>::max() / width)
      throw std::overflow_error("CPU validation logical byte count overflows");
    const auto last = offset(n - 1, t);
    if (last > std::numeric_limits<std::size_t>::max() - width - 64)
      throw std::overflow_error("CPU validation allocation overflows");
    const auto bytes = last + width;
    Buffer b{t, std::vector<std::byte>(bytes + 64, std::byte{0xa5}), nullptr};
    if (k < inputs) {
      std::vector<std::byte> packed;
      if (metadata && benchmark &&
          (t.data_type == FLAGDNN_DATA_FLOAT32 ||
           t.data_type == FLAGDNN_DATA_FLOAT16 ||
           t.data_type == FLAGDNN_DATA_BFLOAT16)) {
        // Match benchmark_nvidia_extended.cpp indexed_inputs exactly.
        std::vector<float> values(n);
        for (std::size_t i = 0; i < n; ++i)
          values[i] = static_cast<float>(static_cast<int>((i * 17 + k * 7) % 61) - 30) / 16.F;
        packed = encode(t.data_type, values);
      } else if (metadata) {
        // Raw concatenate fixtures cover all representation bits, with
        // canonical Boolean inputs. There is no float round trip for INT32/FP8.
        packed.resize(n * width);
        for (std::size_t i = 0; i < packed.size(); ++i)
          packed[i] = static_cast<std::byte>(t.data_type == FLAGDNN_DATA_BOOLEAN
              ? (i + k) % 2 : (i * 37 + k * 83) % 256);
      } else {
        packed = encode(t.data_type, input_values(spec, k));
        logical_inputs.push_back(decode(t.data_type, packed));
      }
      for (std::size_t i = 0; i < n; ++i)
        std::memcpy(b.initial.data() + offset(i, t), packed.data() + i * width, width);
    } else if (!metadata) {
      const auto sentinel = encode(t.data_type, std::vector<float>{-64.F});
      for (std::size_t i = 0; i < n; ++i)
        std::memcpy(b.initial.data() + offset(i, t), sentinel.data(), width);
    }
    b.device = std::make_unique<DeviceBuffer>(b.initial.size());
    copy_to_device_async(*b.device, b.initial, 0, stream.get());
    buffers.push_back(std::move(b));
    bindings.push_back({t.uid, buffers.back().device->at(t.binding_byte_offset)});
  }
  check_driver(cuStreamSynchronize(stream.get()), "CPU fallback upload");
  const auto& out = spec.tensors.back();
  const auto n = count(out), width = element_size(out.data_type);
  std::vector<std::byte> expected_raw(n * width);
  std::vector<float> expected;
  if (metadata) {
    reference::cpu::IndexParameters p{metadata->operation, {}, out.dimensions, metadata->axis};
    for (std::size_t k = 0; k < inputs; ++k) p.input_shapes.push_back(spec.tensors[k].dimensions);
    std::vector<std::vector<std::byte>> packed_inputs;
    packed_inputs.reserve(inputs);
    for (std::size_t k = 0; k < inputs; ++k) {
      const auto& b = buffers[k];
      if (b.spec.data_type != out.data_type)
        throw std::invalid_argument("index input/output data types differ");
      packed_inputs.emplace_back(count(b.spec) * width);
      auto& packed = packed_inputs.back();
      for (std::size_t i = 0; i < count(b.spec); ++i)
        std::memcpy(packed.data() + i * width,
                    b.initial.data() + offset(i, b.spec), width);
    }
    expected_raw = reference::cpu::evaluate_index_bytes(p, packed_inputs, out.data_type);
    // Raw input fixtures span all byte representations, including the padding
    // sentinel 0xa5. Poison each logical output byte with the complement of its
    // expected value so even a single omitted store fails the first execution.
    // Keep offsets, padding and the trailing guard unchanged.
    auto& output = buffers.back();
    for (std::size_t i = 0; i < n; ++i) {
      const auto pos = offset(i, out);
      for (std::size_t byte = 0; byte < width; ++byte)
        output.initial[pos + byte] = expected_raw[i * width + byte] ^ std::byte{0xff};
    }
    copy_to_device_async(*output.device, output.initial, 0, stream.get());
    check_driver(cuStreamSynchronize(stream.get()), "CPU fallback poison output");
  } else {
    expected = reference::cpu::evaluate_benchmark(spec, logical_inputs,
        [](std::span<const float> v, flagdnnDataType_t t) { return decode(t, encode(t, v)); }).at(0);
    if (expected.size() != n) throw std::runtime_error("CPU oracle size differs");
  }
  DeviceBuffer workspace(executable.workspace_size());
  executable.prepare(bindings, stream.opaque());
  const auto execute = [&] { executable.execute(bindings, workspace.data(), workspace.size(), stream.opaque()); };
  const auto validate = [&] {
    for (std::size_t k = 0; k < buffers.size(); ++k) {
      const auto& b = buffers[k];
      std::vector<std::byte> actual(b.initial.size());
      copy_from_device_async(actual, *b.device, 0, stream.get());
      check_driver(cuStreamSynchronize(stream.get()), "CPU fallback validation");
      if (k < inputs) {
        if (actual != b.initial) throw std::runtime_error(spec.name + ": input modified");
        continue;
      }
      std::vector<bool> occupied(actual.size(), false);
      std::vector<std::byte> packed(n * width);
      for (std::size_t i = 0; i < n; ++i) {
        const auto pos = offset(i, out);
        std::memcpy(packed.data() + i * width, actual.data() + pos, width);
        for (std::size_t byte = 0; byte < width; ++byte) occupied.at(pos + byte) = true;
      }
      for (std::size_t i = 0; i < actual.size(); ++i)
        if (!occupied[i] && actual[i] != b.initial[i])
          throw std::runtime_error(spec.name + ": output padding modified");
      if (metadata) {
        if (packed != expected_raw) throw std::runtime_error(spec.name + ": CPU byte oracle differs");
      } else {
        const auto values = decode(out.data_type, packed);
        for (std::size_t i = 0; i < n; ++i) {
          const double a = values[i], e = expected[i];
          if (a == e || (std::isnan(a) && std::isnan(e))) continue;
          const auto diff = std::abs(a - e);
          if (!std::isfinite(diff) ||
              (diff > spec.absolute_tolerance &&
               diff > spec.relative_tolerance * std::max({std::abs(a), std::abs(e), 1e-30}))) {
            std::ostringstream message;
            message << spec.name << ": CPU oracle differs at " << i << " actual=" << a << " expected=" << e;
            throw std::runtime_error(message.str());
          }
        }
      }
    }
  };
  execute();
  validate();
  std::cout << spec.name << ": FlagDNN Graph vs CPU reference PASS\n";
  if (!benchmark) return {};
  const auto config = spec.benchmark;
  if (config.warmup_iterations < 0 || config.sample_count <= 0 || config.iterations_per_sample <= 0)
    throw std::invalid_argument("invalid equivalent timing configuration");
  for (int i = 0; i < config.warmup_iterations; ++i) execute();
  check_driver(cuStreamSynchronize(stream.get()), "equivalent warmup");
  CapturedBatch batch(stream, config.iterations_per_sample, execute);
  batch.launch(stream);
  check_driver(cuStreamSynchronize(stream.get()), "equivalent graph warmup");
  std::vector<double> samples;
  for (int i = 0; i < config.sample_count; ++i) {
    DeviceEvent start, stop;
    start.record(stream.get());
    batch.launch(stream);
    stop.record(stream.get());
    stop.synchronize();
    const auto us = start.elapsed_microseconds_to(stop) / config.iterations_per_sample;
    if (!std::isfinite(us) || us <= 0) throw std::runtime_error("invalid equivalent timing sample");
    samples.push_back(us);
  }
  validate();
  // Validate a fresh replay after poisoning every output, so an empty or
  // partially captured graph cannot reuse the successful direct execution.
  auto fresh = buffers.back().initial;
  for (std::size_t i = 0; i < n; ++i) {
    const auto pos = offset(i, out);
    if (metadata) {
      for (std::size_t b = 0; b < width; ++b)
        fresh[pos + b] = expected_raw[i * width + b] ^ std::byte{0xff};
    } else {
      const auto sentinel = encode(out.data_type, std::vector<float>{-64.F});
      std::memcpy(fresh.data() + pos, sentinel.data(), width);
    }
  }
  copy_to_device_async(*buffers.back().device, fresh, 0, stream.get());
  check_driver(cuStreamSynchronize(stream.get()), "equivalent reset output");
  batch.launch(stream);
  validate();
  std::cout << "[postcheck] case=" << spec.name << " cpu_full=true fresh_replay=true\n";
  return samples;
}
}  // namespace flagdnn::validation::thead
