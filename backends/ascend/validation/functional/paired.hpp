/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include "reference/cpu/pointwise.hpp"
#include "runtime/json.hpp"
#include "validation/development_environment.hpp"
#include "validation/functional/aclnn_plan.hpp"
#include "validation/functional/raw_io.hpp"
#include "validation/benchmark/equivalent_baseline.hpp"
#include "validation/benchmark/graph_replay.hpp"
#include <cmath>
#include <flagdnn/flagdnn.hpp>
#include <fstream>
#include <iomanip>
#include <optional>
#include <regex>
#include <sstream>

namespace flagdnn::testing::ascend {
struct PairedBenchmarkConfig {
  int warmup = 10, samples = 20, iterations = 50;
};
inline PairedBenchmarkConfig paired_benchmark_config;
struct PairedTolerance {
  double absolute, relative;
};
inline void metrics(std::string_view name, std::vector<double> values) {
  auto sorted = values;
  std::sort(sorted.begin(), sorted.end());
  std::cout
      << '\"' << name << "\":{\"median\":" << sorted[(sorted.size() - 1) / 2]
      << ",\"p90\":"
      << sorted[static_cast<std::size_t>(std::ceil(sorted.size() * 0.9)) - 1]
      << ",\"samples\":[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i)
      std::cout << ',';
    std::cout << values[i];
  }
  std::cout << "]}";
}
inline void measure(std::string_view name, std::string_view provider,
                    TestExecutable &executable,
                    std::span<const flagdnnBinding_t> bindings,
                    acl::DeviceBuffer &workspace, acl::Stream &stream,
                    const std::filesystem::path &cache,
                    const std::string &tuning_trace,
                    const EquivalentBaseline *equivalent = nullptr,
                    const std::function<void(bool)> &validate = {}) {
  auto execute = [&] {
    executable.execute(bindings, workspace.opaque(),
                       executable.workspace_size(), stream.opaque());
  };
  for (int i = 0; i < paired_benchmark_config.warmup; ++i) {
    execute();
  }
  stream.synchronize();
  acl::EventTimer timer;
  std::vector<double> device, submit, complete;
  std::unique_ptr<GraphReplay> graph;
  if (equivalent) {
    if (paired_benchmark_config.warmup != 10 ||
        paired_benchmark_config.samples != 20 ||
        paired_benchmark_config.iterations != 50 || !validate)
      throw std::runtime_error("equivalent baseline timing profile differs");
    graph = std::make_unique<GraphReplay>(
        stream.get(), paired_benchmark_config.iterations, execute);
    graph->replay();
    stream.synchronize();
  }
  for (int i = 0; i < paired_benchmark_config.samples; ++i) {
    auto t = graph ? timer.measure(stream.get(), 1, [&] { graph->replay(); })
                   : timer.measure(stream.get(), paired_benchmark_config.iterations,
                                   execute);
    if (graph) {
      t.stream_us /= paired_benchmark_config.iterations;
      t.submit_us /= paired_benchmark_config.iterations;
      t.end_to_end_us /= paired_benchmark_config.iterations;
    }
    device.push_back(t.stream_us);
    submit.push_back(t.submit_us);
    complete.push_back(t.end_to_end_us);
  }
  if (graph) {
    validate(false);
    validate(true);
    graph->replay();
    stream.synchronize();
    validate(false);
  }
  std::filesystem::path last;
  std::filesystem::file_time_type latest{};
  for (const auto &entry : std::filesystem::recursive_directory_iterator(cache))
    if (entry.path().filename() == "manifest.json") {
      auto time = entry.last_write_time();
      if (last.empty() || time > latest) {
        last = entry.path();
        latest = time;
      }
    }
  if (last.empty())
    throw std::runtime_error("benchmark compiled manifest missing");
  std::ifstream input(last);
  std::stringstream contents;
  contents << input.rdbuf();
  auto manifest = native::json::parse(contents.str());
  std::string selected;
  const auto &stages = manifest.at("program").at("stages").as_array();
  for (const auto &stage : stages) {
    const auto id = stage.at("stage_id").as_int();
    const auto &candidates = stage.at("candidates").as_array();
    std::string candidate;
    if (candidates.size() == 1) {
      candidate = candidates.front().at("candidate_id").as_string();
    } else {
      const std::regex pattern("\\[FLAGDNN_ASCEND_AUTOTUNE\\] stage=" +
                               std::to_string(id) + " candidate=([^ ]+)");
      std::smatch match;
      if (!std::regex_search(tuning_trace, match, pattern))
        throw std::runtime_error("selected autotune candidate is missing");
      candidate = match[1].str();
    }
    if (!selected.empty())
      selected += ';';
    selected +=
        stages.size() == 1 ? candidate : std::to_string(id) + ':' + candidate;
  }
  std::cout << std::setprecision(equivalent ? 17 : 10)
            << "{\"schema_version\":" << (equivalent ? 6 : 2)
            << ",\"kind\":\"steady_state\",\"provider\":\""
            << provider << "\",\"case\":\"" << name
            << "\",\"environment\":{\"soc_fingerprint\":\""
            << manifest.at("target").as_string()
            << "\",\"cann_package_version\":\""
            << acl::runtime_package_version() << "\",\"ascendcl_build_id\":\""
            << FLAGDNN_ASCEND_VALIDATION_ASCENDCL_BUILD_ID
            << "\",\"runtime_build_id\":\""
            << FLAGDNN_ASCEND_VALIDATION_RUNTIME_BUILD_ID
            << "\"},\"provider_identity\":{";
  if (provider == "flagdnn")
    std::cout << "\"libtriton_jit_sha256\":\""
              << FLAGDNN_ASCEND_VALIDATION_TRITON_JIT_SHA256
              << "\",\"compiler_identity_sha256\":\""
              << manifest.at("compiler").at("identity_sha256").as_string()
              << "\",\"artifact_request_sha256\":\""
              << manifest.at("request_sha256").as_string()
              << "\",\"launch_abi\":\"ltj_npu_raw_v1\",\"selected_candidate\":"
                 "\""
              << selected << '\"';
  else
    std::cout << "\"libnnopbase_sha256\":\""
              << FLAGDNN_ASCEND_VALIDATION_NNOPBASE_SHA256
              << "\",\"libopapi_math_sha256\":\""
              << FLAGDNN_ASCEND_VALIDATION_OPAPI_MATH_SHA256
              << "\",\"libopapi_nn_sha256\":\""
              << FLAGDNN_ASCEND_VALIDATION_OPAPI_NN_SHA256 << '"';
  std::cout << "},\"benchmark_config\":{\"warmup_iterations\":"
            << paired_benchmark_config.warmup
            << ",\"sample_count\":" << paired_benchmark_config.samples
            << ",\"iterations_per_sample\":"
            << paired_benchmark_config.iterations;
  if (equivalent)
    std::cout << ",\"timing_method\":\"device_events_over_ascend_graph\","
                 "\"graph_warmup_replays\":1,\"statistic\":\"nearest_rank_p50\"";
  std::cout << "},";
  metrics("stream_us", device);
  std::cout << ',';
  metrics("submit_us", submit);
  std::cout << ',';
  metrics("end_to_end_us", complete);
  if (equivalent) {
    auto sorted = device;
    std::sort(sorted.begin(), sorted.end());
    std::cout << ",\"equivalent_baseline\":"
              << equivalent->json(sorted[(sorted.size() - 1) / 2]);
  }
  std::cout << "}\n";
}

template <class Case, class Builder, class Inputs, class Reference,
          class Tolerance, class CpuReference = std::nullptr_t>
int run_paired_cases(int argc, char **argv, std::span<const Case> cases,
                     const char *filter_name, Builder build, Inputs make_inputs,
                     Reference reference, Tolerance tolerance,
                     bool benchmark = false,
                     CpuReference cpu_reference = nullptr) {
  if (argc != 3)
    return 2;
  try {
    acl::DevelopmentEnvironment development("paired");
    acl::AclRuntime runtime;
    development.prepare_target(acl::soc_name());
    acl::Stream stream;
    std::cout << "ACLNN_CAPABILITY_IDENTITY opapi_cv_sha256="
              << FLAGDNN_ASCEND_VALIDATION_OPAPI_CV_SHA256 << std::endl;
    flagdnn::Handle handle("ascend", 0);
    handle.set_compiler(argv[1], argv[2], development.graph_cache().string());
    const char *filter = std::getenv(filter_name);
    std::size_t passed = 0, skipped = 0;
    for (const auto &original : cases) {
      if (filter && original.name.find(filter) == std::string::npos)
        continue;
      try {
        auto test_case = original;
        // ACLNN may require dense outputs. Keep logical shape, dtype and input
        // layout identical and compare gathered results for both providers.
        for (auto &output : test_case.outputs)
          output.strides = dense_strides(output.dimensions);
        std::unique_ptr<TestExecutable> reference_executable;
        std::optional<std::string> native_unavailable;
        std::optional<EquivalentBaseline> equivalent;
        if constexpr (std::is_base_of_v<PointwiseTestCase, Case>) {
          if (benchmark && supports_equivalent_baseline(original)) {
            equivalent = PerformanceBaseline().lookup(
                original, acl::soc_name(), handle.target_fingerprint());
          }
        }
        std::optional<std::vector<std::vector<std::uint8_t>>> cpu_expected;
        try {
          if (!equivalent)
            reference_executable = reference(test_case);
        } catch (const Unsupported &error) {
          if constexpr (std::is_same_v<CpuReference, std::nullptr_t>)
            throw;
          if (benchmark)
            throw;
          native_unavailable = error.what();
        }
        for (const auto &tensor : original.inputs)
          (void)io::data_type_size(tensor.data_type);
        for (const auto &tensor : original.outputs)
          (void)io::data_type_size(tensor.data_type);
        auto inputs = make_inputs(original);
        if (inputs.size() != original.inputs.size())
          throw std::runtime_error("incorrect input count");
        std::optional<std::vector<std::vector<std::uint8_t>>> cpu_oracle;
        if constexpr (!std::is_same_v<CpuReference, std::nullptr_t>) {
          if (!equivalent)
            cpu_oracle = cpu_reference(original, inputs);
        }
        if (equivalent) {
          if constexpr (std::is_same_v<typename decltype(inputs)::value_type,
                                       std::vector<std::uint8_t>>) {
            const auto logical_input =
                logical_bytes(inputs.at(0), original.inputs.at(0));
            if (logical_input !=
                identity_fp8_logical_bytes(original.inputs.at(0)))
              throw std::runtime_error("FP8 identity baseline input domain differs");
            cpu_expected = std::vector<std::vector<std::uint8_t>>{
                flagdnn::reference::cpu::evaluate_identity_bytes(logical_input)};
          } else {
            throw std::runtime_error("FP8 identity requires raw CPU input bytes");
          }
          std::cout << "EQUIVALENT_REFERENCE case=" << original.name
                    << " reason=ACLNN_native_FP8_identity_unavailable"
                    << " validation=bit_exact_before_after_reset_replay\n";
        }
        if (native_unavailable) {
          cpu_expected = std::move(cpu_oracle);
          cpu_oracle.reset();
          if (!cpu_expected)
            throw Unsupported(*native_unavailable);
          std::cout << "CPU_REFERENCE case=" << original.name
                    << " reason=" << *native_unavailable << std::endl;
        }
        const char *reference_name = cpu_expected ? "CPU" : "ACLNN";
        if (const auto *plan = dynamic_cast<const Plan *>(reference_executable.get());
            plan && plan->uses_byte_storage_view())
          std::cout << "ACLNN_STORAGE_VIEW case=" << original.name
                    << " dtype=uint8 semantics=byte_preserving\n";
        std::vector<std::unique_ptr<acl::DeviceBuffer>> buffers,
            reference_buffers;
        std::vector<flagdnnBinding_t> bindings, reference_bindings;
        std::vector<std::vector<std::uint8_t>> original_input_bytes;
        std::unique_ptr<TestExecutable> executable;
        std::unique_ptr<acl::DeviceBuffer> workspace, reference_workspace;
        // All objects are drained before freeing allocations, including errors.
        struct Drain {
          acl::Stream &s;
          ~Drain() {
            try {
              s.synchronize();
            } catch (...) {
            }
          }
        } drain{stream};
        for (std::size_t i = 0; i < original.inputs.size(); ++i) {
          const auto &t = original.inputs[i];
          std::vector<std::uint8_t> bytes;
          if constexpr (std::is_same_v<typename decltype(inputs)::value_type,
                                       std::vector<std::uint8_t>>)
            bytes = inputs[i];
          else
            bytes = io::encode(io::scatter(inputs[i], t), t.data_type);
          if (bytes.size() != io::encoded_byte_count(t))
            throw std::runtime_error("incorrect encoded input size");
          original_input_bytes.push_back(bytes);
          auto b = std::make_unique<acl::DeviceBuffer>(io::allocation_byte_count(t));
          b->copy_from_host_at(bytes.data(), bytes.size(),
                               t.binding_byte_offset, stream.get());
          stream.synchronize();
          bindings.push_back({t.uid, b->opaque_at(t.binding_byte_offset)});
          buffers.push_back(std::move(b));
        }
        auto validate_inputs = [&] {
          for (std::size_t i = 0; i < original.inputs.size(); ++i) {
            const auto &t = original.inputs[i];
            std::vector<std::uint8_t> bytes(io::encoded_byte_count(t));
            buffers[i]->copy_to_host_at(bytes.data(), bytes.size(),
                                        t.binding_byte_offset, stream.get());
            stream.synchronize();
            if (bytes != original_input_bytes[i])
              throw std::runtime_error(original.name +
                                       " execution modified input bytes");
          }
        };
        reference_bindings = bindings;
        for (std::size_t i = 0; i < original.outputs.size(); ++i) {
          for (int ref = 0; ref < (cpu_expected ? 1 : 2); ++ref) {
            const auto &t = ref ? test_case.outputs[i] : original.outputs[i];
            auto bytes =
                io::encode(std::vector<float>(io::storage_element_count(t),
                                              io::kPaddingSentinel),
                           t.data_type);
            auto b = std::make_unique<acl::DeviceBuffer>(io::allocation_byte_count(t));
            b->copy_from_host_at(bytes.data(), bytes.size(),
                                 t.binding_byte_offset, stream.get());
            stream.synchronize();
            (ref ? reference_bindings : bindings)
                .push_back({t.uid, b->opaque_at(t.binding_byte_offset)});
            (ref ? reference_buffers : buffers).push_back(std::move(b));
          }
        }
        if (reference_executable) {
          reference_executable->prepare(reference_bindings, stream.opaque());
          reference_workspace = std::make_unique<acl::DeviceBuffer>(
              reference_executable->workspace_size());
          reference_executable->execute(
              reference_bindings, reference_workspace->opaque(),
              reference_executable->workspace_size(), stream.opaque());
          stream.synchronize();
          // Both providers share read-only inputs. Check the native reference
          // before FlagDNN can consume an accidentally modified input.
          validate_inputs();
        }
        std::ostringstream tuning_trace;
        {
          // Capture the runtime's selected candidate, then replay diagnostics.
          struct Capture {
            std::ostringstream &trace;
            std::streambuf *saved;
            explicit Capture(std::ostringstream &value)
                : trace(value), saved(std::cerr.rdbuf(value.rdbuf())) {}
            ~Capture() {
              std::cerr.rdbuf(saved);
              std::cerr << trace.str();
            }
          } capture(tuning_trace);
          executable = build(handle, original);
        }
        workspace =
            std::make_unique<acl::DeviceBuffer>(executable->workspace_size());
        std::vector<std::vector<float>> expected;
        std::vector<std::vector<std::uint8_t>> expected_bytes;
        if (cpu_expected && cpu_expected->size() != test_case.outputs.size())
          throw std::runtime_error("incorrect CPU reference output count");
        for (std::size_t i = 0; i < test_case.outputs.size(); ++i) {
          const auto &t = test_case.outputs[i];
          if (cpu_expected) {
            auto &bytes = (*cpu_expected)[i];
            if (bytes.size() !=
                io::element_count(t) * io::data_type_size(t.data_type))
              throw std::runtime_error("incorrect CPU reference output size");
            expected.emplace_back();
            if (!io::is_raw_storage(t.data_type))
              expected.back() =
                  io::decode(bytes, t.data_type, io::element_count(t));
            expected_bytes.push_back(std::move(bytes));
            continue;
          }
          std::vector<std::uint8_t> bytes(io::encoded_byte_count(t));
          reference_buffers[i]->copy_to_host_at(
              bytes.data(), bytes.size(), t.binding_byte_offset, stream.get());
          stream.synchronize();
          expected_bytes.push_back(logical_bytes(bytes, t));
          expected.push_back(io::is_raw_storage(t.data_type)
                                 ? std::vector<float>{}
                                 : io::gather(io::decode_storage("ACLNN", bytes, t), t));
        }
        if (cpu_oracle) {
          if (*cpu_oracle != expected_bytes)
            throw std::runtime_error(original.name +
                                     " ACLNN output differs from CPU reference");
          std::cout << "CPU_ORACLE case=" << original.name
                    << " provider=reference_cpu validation=bit_exact\n";
        }
        auto validate_outputs = [&](bool reset) {
          for (std::size_t i = 0; i < original.outputs.size(); ++i) {
            const auto &t = original.outputs[i];
            std::vector<std::uint8_t> bytes(io::encoded_byte_count(t));
            if (reset) {
              bytes = io::encode(std::vector<float>(io::storage_element_count(t),
                                                    io::kPaddingSentinel), t.data_type);
              buffers[original.inputs.size() + i]->copy_from_host_at(
                  bytes.data(), bytes.size(), t.binding_byte_offset, stream.get());
              stream.synchronize();
              continue;
            }
            buffers[original.inputs.size() + i]->copy_to_host_at(
                bytes.data(), bytes.size(), t.binding_byte_offset,
                stream.get());
            stream.synchronize();
            if (io::is_raw_storage(t.data_type)) {
              check_raw_padding(bytes, t);
              const auto actual = logical_bytes(bytes, t);
              if (actual != expected_bytes[i]) {
                std::size_t pos = 0;
                while (actual[pos] == expected_bytes[i][pos])
                  ++pos;
                std::int32_t av = actual[pos], ev = expected_bytes[i][pos];
                if (t.data_type == FLAGDNN_DATA_INT32) {
                  pos /= 4;
                  std::memcpy(&av, actual.data() + pos * 4, 4);
                  std::memcpy(&ev, expected_bytes[i].data() + pos * 4, 4);
                }
                throw std::runtime_error(
                    original.name + " exact output index=" +
                    std::to_string(pos) + " actual=" + std::to_string(av) +
                    " " + reference_name + "=" + std::to_string(ev));
              }
              continue;
            }
            const auto physical = io::decode_storage("FlagDNN", bytes, t);
            io::require_padding_unchanged("FlagDNN", physical, t);
            const auto actual = io::gather(physical, t);
            const auto limit = tolerance(original, i);
            for (std::size_t j = 0; j < actual.size(); ++j) {
              if (actual[j] == expected[i][j])
                continue;
              double error = std::abs(double(actual[j]) - expected[i][j]);
              if (!std::isfinite(error) ||
                  error > limit.absolute +
                              limit.relative * std::abs(expected[i][j]))
                throw std::runtime_error(
                    original.name + " output=" + std::to_string(i) + " index=" +
                    std::to_string(j) + " actual=" + std::to_string(actual[j]) +
                    " expected=" + std::to_string(expected[i][j]));
            }
          }
          if (!reset)
            validate_inputs();
        };
        for (int repeat = 0; repeat < 2; ++repeat) {
          executable->execute(bindings, workspace->opaque(),
                              executable->workspace_size(), stream.opaque());
          stream.synchronize();
          validate_outputs(false);
        }
        if (benchmark) {
          measure(original.name, "flagdnn", *executable, bindings, *workspace,
                  stream, development.graph_cache(), tuning_trace.str(),
                  equivalent ? &*equivalent : nullptr, validate_outputs);
          validate_outputs(false);
          if (reference_executable)
            measure(original.name, "aclnn", *reference_executable,
                  reference_bindings, *reference_workspace, stream,
                  development.graph_cache(), tuning_trace.str());
          validate_outputs(false);
        }
        ++passed;
        std::cout << original.name << ": FlagDNN vs " << reference_name
                  << " PASS" << std::endl;
      } catch (const Unsupported &error) {
        ++skipped;
        std::string reason = error.what();
        std::replace(reason.begin(), reason.end(), '\n', ' ');
        std::cout << "SKIP case=" << original.name << " reason=" << reason
                  << std::endl;
      }
    }
    if (!passed && !skipped)
      throw std::runtime_error("case filter matched no cases");
    std::cout << "FLAGDNN_PAIRED: " << (passed ? "PASS" : "SKIP")
              << " cases=" << passed + skipped << " executed=" << passed
              << " skipped=" << skipped << std::endl;
    return passed ? 0 : 77;
  } catch (const std::exception &error) {
    std::cerr << "FLAGDNN_PAIRED_FAILED: " << error.what() << std::endl;
    return 1;
  }
}
} // namespace flagdnn::testing::ascend
