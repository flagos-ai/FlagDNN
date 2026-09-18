// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef FLAGDNN_THEAD_VALIDATION_PAIRED_HPP_
#define FLAGDNN_THEAD_VALIDATION_PAIRED_HPP_
#include <acdnn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <type_traits>

#include "acdnn_copy_reference.hpp"
#include "acdnn_extended_reference.hpp"
#include "benchmark_mode.hpp"
#include "functional/pointwise_runner_support.hpp"
#include "numeric_types.hpp"

namespace flagdnn::validation::thead::functional {
inline constexpr char kIsolatedPairedChild[] =
    "FLAGDNN_THEAD_ISOLATED_PAIRED_BENCHMARK_CHILD";

template <class Case>
int isolated_backward_pairs(char** argv, std::span<const Case> cases,
                            const std::string& operation,
                            const std::string& suite,
                            const CapabilityCatalog& catalog,
                            const std::string& filter_name) {
  // SDK 1400 retains backward-convolution geometry across plans. Match the
  // isolation already used by the functional and base benchmark runners.
  const char* selected = std::getenv(filter_name.c_str());
  const std::string filter = selected ? selected : "";
  TemporaryCache cache;
  std::size_t executed = 0, skipped = 0;
  for (const auto& test : cases) {
    if (!filter.empty() && test.name.find(filter) == std::string::npos)
      continue;
    const bool unsupported = catalog.lookup(operation, test.name).status ==
                             CapabilityStatus::kUnsupported;
    std::cout.flush();
    std::cerr.flush();
    const auto child = ::fork();
    if (child < 0) throw std::runtime_error("cannot fork paired convolution");
    if (child == 0) {
      if (::setenv("FLAGDNN_CACHE_PATH", cache.path().c_str(), 1) != 0 ||
          ::setenv(kIsolatedPairedChild, "1", 1) != 0 ||
          ::setenv(filter_name.c_str(), test.name.c_str(), 1) != 0)
        ::_exit(126);
      ::execv(argv[0], argv);
      ::_exit(126);
    }
    int status = 0;
    pid_t waited;
    do {
      waited = ::waitpid(child, &status, 0);
    } while (waited < 0 && errno == EINTR);
    if (waited != child || !WIFEXITED(status) ||
        WEXITSTATUS(status) != (unsupported ? 77 : 0)) {
      throw std::runtime_error("isolated paired convolution failed: " +
                               test.name);
    }
    if (unsupported)
      ++skipped;
    else
      ++executed;
  }
  if (executed + skipped == 0)
    throw std::runtime_error("case filter matched no cases");
  std::cout << suite << ": " << (executed ? "PASS" : "SKIP")
            << " cases=" << executed + skipped
            << " comparable_executed=" << executed
            << " reference_skipped=" << skipped << '\n';
  return executed ? 0 : 77;
}

inline void emit_pair_samples(std::string_view provider, std::string_view name,
                              std::vector<double> samples) {
  auto ordered = samples;
  std::sort(ordered.begin(), ordered.end());
  std::cout << "{\"schema_version\":1,\"kind\":\"steady_state\",\"provider\":\""
            << provider << "\",\"case\":\"" << name
            << "\",\"unit\":\"us\",\"median\":"
            << ordered[(ordered.size() + 1) / 2 - 1]
            << ",\"p90\":" << ordered[(ordered.size() * 9 + 9) / 10 - 1]
            << ",\"samples\":[";
  for (std::size_t i = 0; i < samples.size(); ++i)
    std::cout << (i ? "," : "") << samples[i];
  std::cout << "]}\n";
}
class CapturedExecutable {
 public:
  CapturedExecutable(flagdnn::testing::TestExecutable& executable,
                     std::span<const flagdnnBinding_t> bindings,
                     DeviceBuffer& workspace, DeviceStream& stream) {
    executable.prepare(bindings, stream.opaque());
    executable.execute(bindings, workspace.data(), workspace.size(),
                       stream.opaque());
    check_driver(cuStreamSynchronize(stream.get()),
                 "paired warmup synchronize");
    check_driver(
        cuStreamBeginCapture(stream.get(), CU_STREAM_CAPTURE_MODE_RELAXED),
        "paired begin capture");
    try {
      executable.execute(bindings, workspace.data(), workspace.size(),
                         stream.opaque());
      check_driver(cuStreamEndCapture(stream.get(), &graph_),
                   "paired end capture");
      check_driver(cuGraphInstantiate(&replay_, graph_, 0),
                   "paired instantiate");
    } catch (...) {
      if (!graph_) (void)cuStreamEndCapture(stream.get(), &graph_);
      if (graph_) (void)cuGraphDestroy(graph_);
      throw;
    }
  }
  ~CapturedExecutable() {
    if (replay_) (void)cuGraphExecDestroy(replay_);
    if (graph_) (void)cuGraphDestroy(graph_);
  }
  CapturedExecutable(const CapturedExecutable&) = delete;
  CapturedExecutable& operator=(const CapturedExecutable&) = delete;
  double measure(DeviceStream& stream) {
    DeviceEvent start, stop;
    start.record(stream.get());
    for (int i = 0; i < 50; ++i)
      check_driver(cuGraphLaunch(replay_, stream.get()), "paired replay");
    stop.record(stream.get());
    stop.synchronize();
    return start.elapsed_microseconds_to(stop) / 50;
  }

 private:
  CUgraph graph_ = nullptr;
  CUgraphExec replay_ = nullptr;
};
inline void measure_pair(const std::string& name,
                         flagdnn::testing::TestExecutable& production,
                         flagdnn::testing::TestExecutable& reference,
                         std::span<const flagdnnBinding_t> production_bindings,
                         std::span<const flagdnnBinding_t> reference_bindings,
                         DeviceBuffer& production_workspace,
                         DeviceBuffer& reference_workspace,
                         DeviceStream& stream) {
  if (benchmark_validation_only(name)) return;
  CapturedExecutable a(production, production_bindings, production_workspace,
                       stream),
      b(reference, reference_bindings, reference_workspace, stream);
  for (int i = 0; i < 5; ++i) {
    (void)a.measure(stream);
    (void)b.measure(stream);
  }
  std::vector<double> actual, expected;
  for (int i = 0; i < 20; ++i) {
    if (i % 2 == 0) {
      actual.push_back(a.measure(stream));
      expected.push_back(b.measure(stream));
    } else {
      expected.push_back(b.measure(stream));
      actual.push_back(a.measure(stream));
    }
  }
  emit_pair_samples("flagdnn", name, actual);
  emit_pair_samples("acdnn", name, expected);
}
inline std::size_t copy_offset(std::size_t index,
                               const flagdnn::testing::TestTensor& tensor) {
  std::size_t offset = 0;
  for (std::size_t axis = tensor.dimensions.size(); axis > 0; --axis) {
    offset += (index % tensor.dimensions[axis - 1]) * tensor.strides[axis - 1];
    index /= tensor.dimensions[axis - 1];
  }
  return offset;
}
inline void run_raw_copy(const std::string& name,
                         const flagdnn::testing::TestTensor& input,
                         const flagdnn::testing::TestTensor& output,
                         const flagdnn::testing::TestTensor& reference_output,
                         flagdnn::testing::TestExecutable& production,
                         flagdnn::testing::TestExecutable& reference,
                         DeviceStream& stream, bool benchmark) {
  const auto width = element_size(input.data_type);
  std::vector<std::byte> source(storage_element_count(input) * width,
                                std::byte{0xA5});
  for (std::size_t i = 0; i < element_count(input); ++i)
    for (std::size_t byte = 0; byte < width; ++byte)
      source[copy_offset(i, input) * width + byte] = static_cast<std::byte>(
          input.data_type == FLAGDNN_DATA_BOOLEAN ? i % 2
                                                  : (i * 37 + byte * 83) % 256);
  std::vector<std::byte> actual(storage_element_count(output) * width,
                                std::byte{0xA5});
  std::vector<std::byte> expected(
      storage_element_count(reference_output) * width, std::byte{0xA5});
  DeviceBuffer x(input.binding_byte_offset + source.size()),
      y(output.binding_byte_offset + actual.size()),
      z(reference_output.binding_byte_offset + expected.size());
  copy_to_device_async(x, source, input.binding_byte_offset, stream.get());
  copy_to_device_async(y, actual, output.binding_byte_offset, stream.get());
  copy_to_device_async(z, expected, reference_output.binding_byte_offset,
                       stream.get());
  const std::array<flagdnnBinding_t, 2> pb{
      {{input.uid,
        static_cast<std::byte*>(x.data()) + input.binding_byte_offset},
       {output.uid,
        static_cast<std::byte*>(y.data()) + output.binding_byte_offset}}};
  auto rb = pb;
  rb[1] = {reference_output.uid, static_cast<std::byte*>(z.data()) +
                                     reference_output.binding_byte_offset};
  DeviceBuffer pw(production.workspace_size()), rw(reference.workspace_size());
  const auto compare = [&] {
    copy_from_device_async(actual, y, output.binding_byte_offset, stream.get());
    copy_from_device_async(expected, z, reference_output.binding_byte_offset,
                           stream.get());
    check_driver(cuStreamSynchronize(stream.get()), "raw copy synchronize");
    if (element_count(output) != element_count(reference_output))
      throw std::runtime_error("raw copy count mismatch");
    std::vector<bool> written(actual.size()),
        reference_written(expected.size());
    for (std::size_t i = 0; i < element_count(output); ++i)
      for (std::size_t byte = 0; byte < width; ++byte) {
        const auto ai = copy_offset(i, output) * width + byte;
        const auto bi = copy_offset(i, reference_output) * width + byte;
        if (actual[ai] != expected[bi])
          throw std::runtime_error(name + " raw copy differs at byte " +
                                   std::to_string(i * width + byte));
        written[ai] = true;
        reference_written[bi] = true;
      }
    for (std::size_t i = 0; i < actual.size(); ++i)
      if (!written[i] && actual[i] != std::byte{0xA5})
        throw std::runtime_error(name + " production changed copy padding");
    for (std::size_t i = 0; i < expected.size(); ++i)
      if (!reference_written[i] && expected[i] != std::byte{0xA5})
        throw std::runtime_error(name + " reference changed copy padding");
  };
  execute(production, pb, pw, stream);
  execute(reference, rb, rw, stream);
  compare();
  if (benchmark) {
    measure_pair(name, production, reference, pb, rb, pw, rw, stream);
    compare();
  }
  std::cout << name << ": FlagDNN Graph vs acDNN bit-exact PASS\n";
}
inline BoundTensor paired_input(const flagdnn::testing::TestTensor& tensor,
                                std::span<const float> logical,
                                CUstream stream) {
  if (tensor.data_type == FLAGDNN_DATA_INT32) {
    if (logical.size() != element_count(tensor))
      throw std::invalid_argument("INT32 input count mismatch");
    std::vector<std::int32_t> physical(storage_element_count(tensor), -64);
    for (std::size_t i = 0; i < logical.size(); ++i) {
      if (!std::isfinite(logical[i]) || std::trunc(logical[i]) != logical[i] ||
          static_cast<double>(logical[i]) < -2147483648.0 ||
          static_cast<double>(logical[i]) > 2147483647.0)
        throw std::invalid_argument(
            "INT32 test input is not exactly representable");
      physical[copy_offset(i, tensor)] = static_cast<std::int32_t>(logical[i]);
    }
    auto buffer = std::make_unique<DeviceBuffer>(
        tensor.binding_byte_offset + physical.size() * sizeof(std::int32_t));
    copy_to_device_async(*buffer,
                         std::as_bytes(std::span<const std::int32_t>(physical)),
                         tensor.binding_byte_offset, stream);
    return {tensor, std::move(buffer)};
  }
  auto result = make_output_buffer(tensor, stream);
  std::vector<float> physical(storage_element_count(tensor), kPaddingSentinel);
  for (std::size_t i = 0; i < logical.size(); ++i) {
    auto remaining = i;
    std::size_t offset = 0;
    for (std::size_t axis = tensor.dimensions.size(); axis > 0; --axis) {
      offset +=
          (remaining % tensor.dimensions[axis - 1]) * tensor.strides[axis - 1];
      remaining /= tensor.dimensions[axis - 1];
    }
    physical[offset] = logical[i];
  }
  if (tensor.data_type == FLAGDNN_DATA_BOOLEAN) {
    std::vector<std::byte> bytes;
    bytes.reserve(physical.size());
    for (float value : physical)
      bytes.push_back(static_cast<std::byte>(
          value == kPaddingSentinel ? 0x7f : (value > 0)));
    copy_to_device_async(*result.buffer, bytes, tensor.binding_byte_offset,
                         stream);
  } else {
    const auto encoded = encode_floating(tensor.data_type, physical);
    copy_to_device_async(*result.buffer, encoded, tensor.binding_byte_offset,
                         stream);
  }
  return result;
}
inline void compare_pair(std::span<const BoundTensor> actual,
                         std::span<const BoundTensor> expected,
                         DeviceStream& stream, const std::string& name,
                         double atol = -1, double rtol = -1) {
  for (std::size_t i = 0; i < actual.size(); ++i) {
    const auto& spec = actual[i].specification;
    const auto a = read_output(actual[i], stream.get()),
               b = read_output(expected[i], stream.get());
    require_padding_unchanged("FlagDNN", a, spec);
    require_padding_unchanged("acDNN", b, expected[i].specification);
    const auto x = gather(a, spec), y = gather(b, expected[i].specification);
    const double tolerance = spec.data_type == FLAGDNN_DATA_BFLOAT16  ? 8e-3
                             : spec.data_type == FLAGDNN_DATA_FLOAT16 ? 1e-3
                                                                      : 2e-4;
    if (x.size() != y.size())
      throw std::runtime_error("paired output sizes differ");
    for (std::size_t j = 0; j < x.size(); ++j) {
      const double diff = std::abs(double(x[j]) - double(y[j]));
      if (!std::isfinite(diff) ||
          (diff > (atol < 0 ? tolerance : atol) &&
           diff > (rtol < 0 ? tolerance : rtol) *
                      std::max(std::abs(x[j]), std::abs(y[j]))))
        throw std::runtime_error(name + " output " + std::to_string(i) +
                                 " differs at " + std::to_string(j) +
                                 ": FlagDNN=" + std::to_string(x[j]) +
                                 " acDNN=" + std::to_string(y[j]));
    }
  }
}
struct ExtendedReferenceFactory {
  template <class Case>
  auto operator()(const Case& value) const {
    return make_acdnn_extended_reference(value);
  }
};
template <class Case, class Build, class Inputs,
          class Reference = ExtendedReferenceFactory>
int run_paired_cases(int argc, char** argv, std::span<const Case> cases,
                     std::string operation, Build build, Inputs input_values,
                     bool benchmark, Reference reference_factory = {},
                     std::string category = "") {
  std::string label = operation;
  std::transform(
      label.begin(), label.end(), label.begin(),
      [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
  std::transform(
      category.begin(), category.end(), category.begin(),
      [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
  const std::string suite = "FLAGDNN_" + label +
                            (category.empty() ? "" : "_" + category) +
                            (benchmark ? "_BENCHMARK" : "_FUNCTIONAL");
  try {
    const bool probe =
        argc == 2 && std::string_view(argv[1]) == "--probe-reference";
    if (argc == 2 && std::string_view(argv[1]) == "--dump-cases") {
      for (const auto& t : cases) {
        std::cout << operation << '\t' << t.name << '\n';
      }
      return 0;
    }
    if (!probe && argc != 3)
      throw std::invalid_argument(
          "THEAD paired runner requires compiler arguments");
    const auto catalog =
        CapabilityCatalog::load(FLAGDNN_THEAD_ACDNN_CAPABILITY_CATALOG);
    catalog.validate_versions(FLAGDNN_THEAD_PPU_SDK_VERSION, ACDNN_VERSION,
                              static_cast<std::int64_t>(acdnnGetVersion()));
    if (!probe && benchmark &&
        (operation == "conv_dgrad" || operation == "conv_wgrad") &&
        std::getenv(kIsolatedPairedChild) == nullptr) {
      return isolated_backward_pairs(argv, cases, operation, suite, catalog,
                                     "FLAGDNN_" + label + "_CASE");
    }
    check_driver(cuInit(0), "cuInit");
    CUdevice device = 0;
    check_driver(cuDeviceGet(&device, 0), "cuDeviceGet");
    PrimaryContext primary(device);
    ScopedCurrentContext current(primary.get());
    DeviceStream stream;
    TemporaryCache cache;
    flagdnn::Handle handle("thead", 0);
    if (!probe) handle.set_compiler(argv[1], argv[2], cache.path().string());
    const char* filter = std::getenv(("FLAGDNN_" + label + "_CASE").c_str());
    std::size_t executed = 0, skipped = 0;
    std::cout << std::setprecision(9);
    for (const auto& t : cases) {
      if (filter && (std::getenv(kIsolatedPairedChild) != nullptr
                         ? t.name != filter
                         : t.name.find(filter) == std::string::npos))
        continue;
      if (!probe) {
        const auto& record = catalog.lookup(operation, t.name);
        if (record.status == CapabilityStatus::kUnsupported) {
          const auto& representative = t.inputs[0];
          std::cout << "[SKIP][acdnn] op=" << operation << " case=" << t.name
                    << " reason=" << record.reason_code
                    << " sdk=" << FLAGDNN_THEAD_PPU_SDK_VERSION
                    << " acdnn_header=" << ACDNN_VERSION
                    << " acdnn_runtime=" << acdnnGetVersion()
                    << " target=" << handle.target_fingerprint()
                    << " dtype=" << data_type_name(representative.data_type)
                    << " layout=" << layout_name(representative)
                    << " shape=" << shape_name(representative) << '\n';
          ++skipped;
          continue;
        }
      }
      try {
        auto reference = reference_factory(t);
        if (!probe && category == "COPY") {
          auto production = build(handle, t);
          if constexpr (requires { t.reference_outputs; }) {
            run_raw_copy(t.name, t.inputs.at(0), t.outputs.at(0),
                         t.reference_outputs.at(0), *production, *reference,
                         stream, benchmark);
          } else {
            run_raw_copy(t.name, t.inputs.at(0), t.outputs.at(0),
                         t.outputs.at(0), *production, *reference, stream,
                         benchmark);
          }
          ++executed;
          continue;
        }
        const auto values = input_values(t);
        std::vector<BoundTensor> inputs, actual, expected;
        for (std::size_t i = 0; i < t.inputs.size(); ++i)
          inputs.push_back(paired_input(t.inputs[i], values[i], stream.get()));
        for (const auto& out : t.outputs)
          actual.push_back(make_output_buffer(out, stream.get()));
        if constexpr (requires { t.reference_outputs; }) {
          for (const auto& out : t.reference_outputs)
            expected.push_back(make_output_buffer(out, stream.get()));
        } else {
          for (const auto& out : t.outputs)
            expected.push_back(make_output_buffer(out, stream.get()));
        }
        const auto compare = [&] {
          if constexpr (requires { t.specification.absolute_tolerance; }) {
            compare_pair(actual, expected, stream, t.name,
                         t.specification.absolute_tolerance,
                         t.specification.relative_tolerance);
          } else if constexpr (std::is_same_v<
                                   Case, flagdnn::testing::ResampleTestCase>) {
            for (std::size_t i = 0; i < actual.size(); ++i) {
              const auto type = t.outputs[i].data_type;
              const double tolerance = i != 0                          ? 0.0
                                       : type == FLAGDNN_DATA_BFLOAT16 ? 4.0e-3
                                       : type == FLAGDNN_DATA_FLOAT16  ? 5.0e-4
                                                                       : 1.0e-5;
              compare_pair(std::span<const BoundTensor>(actual).subspan(i, 1),
                           std::span<const BoundTensor>(expected).subspan(i, 1),
                           stream, t.name, tolerance, tolerance);
            }
          } else if constexpr (std::is_same_v<
                                   Case,
                                   flagdnn::testing::StatisticsTestCase>) {
            compare_pair(actual, expected, stream, t.name, 2.0e-5, 2.0e-4);
          } else {
            compare_pair(actual, expected, stream, t.name);
          }
        };
        DeviceBuffer rw(reference->workspace_size());
        auto rb = bindings(inputs, expected);
        execute(*reference, rb, rw, stream);
        check_driver(cuStreamSynchronize(stream.get()),
                     "acDNN reference synchronize");
        if (probe) {
          std::cout << "PROBE\t" << operation << '\t' << t.name
                    << "\tsupported\n";
          continue;
        }
        auto production = build(handle, t);
        DeviceBuffer pw(production->workspace_size());
        auto pb = bindings(inputs, actual);
        execute(*production, pb, pw, stream);
        compare();
        if (benchmark) {
          measure_pair(t.name, *production, *reference, pb, rb, pw, rw, stream);
          compare();
        }
        std::cout << t.name << ": FlagDNN Graph vs acDNN PASS\n";
        ++executed;
      } catch (const std::exception& error) {
        if (!probe) throw;
        std::cout << "PROBE\t" << operation << '\t' << t.name << "\trejected\t"
                  << error.what() << '\n';
      }
    }
    if (probe) return 0;
    if (!executed && !skipped)
      throw std::runtime_error("case filter matched no cases");
    if (std::getenv(kIsolatedPairedChild) == nullptr) {
      std::cout << suite << ": " << (executed ? "PASS" : "SKIP")
                << " cases=" << executed + skipped;
      if (benchmark)
        std::cout << " comparable_executed=" << executed
                  << " reference_skipped=" << skipped;
      else
        std::cout << " executed=" << executed << " skipped=" << skipped;
      std::cout << '\n';
    }
    return executed ? 0 : 77;
  } catch (const std::exception& e) {
    std::cerr << suite << ": FAIL reason=" << e.what() << '\n';
    return 1;
  }
}
}  // namespace flagdnn::validation::thead::functional
#endif
