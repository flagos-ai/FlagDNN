/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "benchmark/common/attention_runner.hpp"

#include <unistd.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <flagdnn/flagdnn.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "common/attention.hpp"
#include "validation/benchmark/timing.hpp"
#include "validation/cuda_driver.hpp"
#include "validation/tensor_io.hpp"

namespace flagdnn::testing {
namespace {

namespace timing = flagdnn::validation::nvidia::timing;

class TemporaryCache {
 public:
  explicit TemporaryCache(std::string_view operation) {
    std::string pattern =
        (std::filesystem::temp_directory_path() /
         ("flagdnn-" + std::string(operation) + "-functional-XXXXXX"))
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

std::vector<float> make_values(std::size_t count, std::size_t tensor_index,
                               float scale = 1.0F) {
  std::vector<float> result(count);
  for (std::size_t index = 0; index < count; ++index) {
    const int centered =
        static_cast<int>((index * 37 + tensor_index * 19) % 101) - 50;
    result[index] = scale * static_cast<float>(centered) /
                    static_cast<float>(53 + tensor_index);
  }
  return result;
}

class TensorAllocation {
 public:
  TensorAllocation(const TestTensor& specification,
                   std::span<const float> logical, Stream& stream)
      : specification_(specification),
        bytes_(cuda::encode(cuda::scatter(logical, specification),
                            specification.data_type,
                            cuda::BooleanEncoding::kByte)),
        buffer_(bytes_.size()) {
    buffer_.copy_from_host(bytes_.data(), bytes_.size(), stream.get());
    logical_ =
        cuda::gather(cuda::decode(bytes_, specification.data_type,
                                  cuda::storage_element_count(specification),
                                  cuda::BooleanEncoding::kByte),
                     specification);
  }

  static std::unique_ptr<TensorAllocation> input(
      const TestTensor& specification, std::size_t tensor_index, Stream& stream,
      float scale = 1.0F) {
    const std::vector<float> logical =
        make_values(cuda::element_count(specification), tensor_index, scale);
    return std::make_unique<TensorAllocation>(specification, logical, stream);
  }

  static std::unique_ptr<TensorAllocation> output(
      const TestTensor& specification, Stream& stream) {
    const std::vector<float> physical(
        cuda::storage_element_count(specification), cuda::padding_sentinel());
    std::vector<float> logical(cuda::element_count(specification));
    for (std::size_t index = 0; index < logical.size(); ++index) {
      logical[index] = physical[cuda::logical_offset(index, specification)];
    }
    return std::make_unique<TensorAllocation>(specification, logical, stream);
  }

  static std::unique_ptr<TensorAllocation> scalar(const Fp8Scalar& scalar_value,
                                                  Stream& stream) {
    const std::array<float, 1> logical{scalar_value.value};
    return std::make_unique<TensorAllocation>(scalar_value.tensor, logical,
                                              stream);
  }

  [[nodiscard]] const std::vector<float>& logical() const noexcept {
    return logical_;
  }

  [[nodiscard]] void* pointer() const noexcept { return buffer_.opaque(); }

  [[nodiscard]] std::vector<float> read(Stream& stream) const {
    std::vector<std::uint8_t> bytes(bytes_.size());
    buffer_.copy_to_host(bytes.data(), bytes.size(), stream.get());
    stream.synchronize();
    return cuda::decode(bytes, specification_.data_type,
                        cuda::storage_element_count(specification_),
                        cuda::BooleanEncoding::kByte);
  }

 private:
  TestTensor specification_;
  std::vector<std::uint8_t> bytes_;
  DeviceBuffer buffer_;
  std::vector<float> logical_;
};

struct Accuracy {
  double maximum_absolute = 0.0;
  double maximum_relative = 0.0;
};

Accuracy compare_tensor(std::string_view case_name,
                        std::string_view tensor_name,
                        const TestTensor& specification,
                        const TensorAllocation& actual,
                        const TensorAllocation& reference,
                        double absolute_tolerance, double relative_tolerance,
                        Stream& stream) {
  const std::vector<float> actual_physical = actual.read(stream);
  const std::vector<float> reference_physical = reference.read(stream);
  cuda::require_padding_unchanged("FlagDNN", actual_physical, specification);
  cuda::require_padding_unchanged("reference", reference_physical,
                                  specification);
  const std::vector<float> actual_logical =
      cuda::gather(actual_physical, specification);
  const std::vector<float> reference_logical =
      cuda::gather(reference_physical, specification);
  if (actual_logical.size() != reference_logical.size()) {
    throw std::runtime_error("SDPA output sizes differ");
  }
  Accuracy result;
  for (std::size_t index = 0; index < actual_logical.size(); ++index) {
    const double left = actual_logical[index];
    const double right = reference_logical[index];
    const double absolute = std::abs(left - right);
    const double relative =
        absolute / std::max({std::abs(left), std::abs(right), 1.0e-30});
    result.maximum_absolute = std::max(result.maximum_absolute, absolute);
    result.maximum_relative = std::max(result.maximum_relative, relative);
    if (!std::isfinite(absolute) ||
        (absolute > absolute_tolerance && relative > relative_tolerance)) {
      std::ostringstream message;
      message << case_name << " differs at " << tensor_name << " element "
              << index << ": FlagDNN=" << left << ", reference=" << right
              << ", abs=" << absolute << ", rel=" << relative
              << ", atol=" << absolute_tolerance
              << ", rtol=" << relative_tolerance;
      throw std::runtime_error(message.str());
    }
  }
  return result;
}

void execute(AttentionExecutable& executable,
             std::span<const flagdnnBinding_t> bindings,
             DeviceBuffer& workspace, Stream& stream) {
  executable.execute(bindings, workspace.opaque(), executable.workspace_size(),
                     stream.opaque());
}

void measure_attention_pair(
    std::string_view name, AttentionExecutable& flagdnn,
    AttentionExecutable& reference,
    std::span<const flagdnnBinding_t> flagdnn_bindings,
    std::span<const flagdnnBinding_t> reference_bindings,
    DeviceBuffer& flagdnn_workspace, DeviceBuffer& reference_workspace,
    Stream& stream, timing::RuntimeMeasurements& flagdnn_runtime,
    timing::RuntimeMeasurements& reference_runtime) {
  constexpr int iterations = 20;
  constexpr int sample_count = 10;
  const auto launch_flagdnn = [&] {
    execute(flagdnn, flagdnn_bindings, flagdnn_workspace, stream);
  };
  const auto launch_reference = [&] {
    execute(reference, reference_bindings, reference_workspace, stream);
  };
  for (int i = 0; i < 5; ++i) {
    launch_flagdnn();
    launch_reference();
  }
  stream.synchronize();
  timing::CapturedExecutionBatch flagdnn_batch(stream.get(), iterations,
                                               launch_flagdnn);
  timing::CapturedExecutionBatch reference_batch(stream.get(), iterations,
                                                 launch_reference);
  flagdnn_batch.launch(stream.get());
  reference_batch.launch(stream.get());
  stream.synchronize();
  EventTimer flagdnn_timer;
  EventTimer reference_timer;
  std::vector<double> flagdnn_samples;
  std::vector<double> reference_samples;
  for (int sample = 0; sample < sample_count; ++sample) {
    const auto measure_flagdnn = [&] {
      flagdnn_samples.push_back(
          flagdnn_timer.measure_microseconds(
              stream.get(), 1, [&] { flagdnn_batch.launch(stream.get()); }) /
          iterations);
    };
    const auto measure_reference = [&] {
      reference_samples.push_back(
          reference_timer.measure_microseconds(
              stream.get(), 1, [&] { reference_batch.launch(stream.get()); }) /
          iterations);
    };
    if (sample % 2 == 0) {
      measure_flagdnn();
      measure_reference();
    } else {
      measure_reference();
      measure_flagdnn();
    }
  }
  const auto host_samples = [&](auto&& launch) {
    std::vector<double> result;
    for (int sample = 0; sample < sample_count; ++sample) {
      stream.synchronize();
      const auto begin = timing::HostClock::now();
      for (int i = 0; i < iterations; ++i) launch();
      result.push_back(timing::host_microseconds_since(begin) / iterations);
      stream.synchronize();
    }
    return result;
  };
  flagdnn_runtime.host_submit_us = host_samples(launch_flagdnn);
  reference_runtime.host_submit_us = host_samples(launch_reference);
  timing::emit_samples("flagdnn", name, flagdnn_samples, flagdnn_runtime);
  timing::emit_samples("cudnn", name, reference_samples, reference_runtime);
  std::cout << name << ": speedup="
            << timing::percentile(reference_samples, 0.5) /
                   timing::percentile(flagdnn_samples, 0.5)
            << std::endl;
}

std::string read_file(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("cannot read SDPA generated artifact");
  }
  return {std::istreambuf_iterator<char>(input),
          std::istreambuf_iterator<char>()};
}

struct GeneratedArtifacts {
  std::vector<std::filesystem::path> manifests;
  std::vector<std::filesystem::path> selections;
  std::size_t cubins = 0;
};

GeneratedArtifacts generated_artifacts(const std::filesystem::path& cache) {
  GeneratedArtifacts result;
  for (const auto& entry :
       std::filesystem::recursive_directory_iterator(cache)) {
    if (!entry.is_regular_file()) {
      continue;
    }
    const std::string filename = entry.path().filename().string();
    if (filename == "manifest.json") {
      result.manifests.push_back(entry.path());
    } else if (filename.starts_with(".flagdnn-autotune-v1-stage-")) {
      result.selections.push_back(entry.path());
    } else if (entry.path().extension() == ".cubin") {
      ++result.cubins;
    }
  }
  std::sort(result.manifests.begin(), result.manifests.end());
  std::sort(result.selections.begin(), result.selections.end());
  return result;
}

std::size_t count_occurrences(std::string_view text, std::string_view needle) {
  std::size_t count = 0;
  std::size_t offset = 0;
  while ((offset = text.find(needle, offset)) != std::string_view::npos) {
    ++count;
    offset += needle.size();
  }
  return count;
}

std::vector<std::string> selection_contents(
    const GeneratedArtifacts& artifacts) {
  std::vector<std::string> result;
  result.reserve(artifacts.selections.size());
  for (const auto& selection : artifacts.selections) {
    result.push_back(read_file(selection));
  }
  return result;
}

std::filesystem::path new_manifest(
    const std::filesystem::path& cache,
    std::span<const std::filesystem::path> previous_manifests,
    std::string_view operation) {
  const GeneratedArtifacts artifacts = generated_artifacts(cache);
  std::vector<std::filesystem::path> added;
  for (const auto& manifest : artifacts.manifests) {
    if (std::find(previous_manifests.begin(), previous_manifests.end(),
                  manifest) == previous_manifests.end()) {
      added.push_back(manifest);
    }
  }
  if (added.size() != 1 || artifacts.cubins != 0) {
    throw std::runtime_error(std::string(operation) +
                             " must produce one new JIT manifest and no cubin");
  }
  return added.front();
}

void verify_forward_jit_artifact(const std::filesystem::path& cache,
                                 const SdpaTestCase& test_case,
                                 const GeneratedArtifacts& before) {
  const GeneratedArtifacts artifacts = generated_artifacts(cache);
  if (artifacts.selections.size() != before.selections.size() + 1 ||
      artifacts.cubins != 0) {
    throw std::runtime_error(
        "SDPA autotune must produce one JIT manifest, one selection, and no "
        "cubin");
  }
  const std::string manifest =
      read_file(new_manifest(cache, before.manifests, "SDPA"));
  for (const std::string_view token :
       {"\"engine\": \"libtriton_jit\"", "\"ownership\": \"platform\"",
        "\"provider\": \"nvidia_triton\"", "\"source\": \"attention.py\"",
        "\"function\": \"_sdpa_fwd_kernel\"", "\"table\": \"sdpa\""}) {
    if (manifest.find(token) == std::string::npos) {
      throw std::runtime_error(test_case.name + " manifest is missing " +
                               std::string(token));
    }
  }
  if (count_occurrences(manifest, "\"variant_id\":") != 8) {
    throw std::runtime_error(
        "SDPA did not consume all eight common.yaml tuning candidates");
  }
  const auto selected = std::find_if(
      artifacts.selections.begin(), artifacts.selections.end(),
      [&](const auto& path) {
        return std::find(before.selections.begin(), before.selections.end(),
                         path) == before.selections.end();
      });
  if (selected == artifacts.selections.end()) {
    throw std::runtime_error("SDPA autotune did not produce a new selection");
  }
  const std::string selection = read_file(*selected);
  if (selection.find("\"measurement_identity\":\"nvidia-libtriton-jit-"
                     "stage-cuda-graph-v3-build-") == std::string::npos ||
      selection.find("-stage-0\"") == std::string::npos ||
      selection.find("\"policy_identity\":") == std::string::npos) {
    throw std::runtime_error(
        "SDPA selection cache is not a JIT timing result: " + selection);
  }
}

void verify_backward_jit_artifact(
    const std::filesystem::path& cache,
    std::span<const std::filesystem::path> previous_manifests,
    bool different_value_dimension, bool autotune, bool fp32) {
  const GeneratedArtifacts artifacts = generated_artifacts(cache);
  std::vector<std::filesystem::path> new_manifests;
  for (const auto& manifest : artifacts.manifests) {
    if (std::find(previous_manifests.begin(), previous_manifests.end(),
                  manifest) == previous_manifests.end()) {
      new_manifests.push_back(manifest);
    }
  }
  if (new_manifests.size() != 1 || artifacts.cubins != 0) {
    throw std::runtime_error("SDPA backward did not produce a JIT manifest");
  }
  const std::string manifest = read_file(new_manifests.front());
  for (const std::string_view token :
       {"\"engine\": \"libtriton_jit\"", "\"ownership\": \"platform\"",
        "\"provider\": \"nvidia_triton\"", "\"source\": \"attention.py\"",
        "\"function\": \"_sdpa_bwd_dq_dbias_kernel\""}) {
    if (manifest.find(token) == std::string::npos) {
      throw std::runtime_error("SDPA backward manifest is missing " +
                               std::string(token));
    }
  }
  const std::string tuning_table =
      fp32 ? "sdpa_backward_dq_fp32" : "sdpa_backward_dq";
  if (autotune && manifest.find("\"table\": \"" + tuning_table + "\"") ==
                      std::string::npos) {
    throw std::runtime_error("SDPA backward autotune table is missing");
  }
  const std::string_view second_function =
      different_value_dimension ? "\"function\": \"_sdpa_bwd_dk_kernel\""
                                : "\"function\": \"_sdpa_bwd_dkdv_kernel\"";
  if (manifest.find(second_function) == std::string::npos) {
    throw std::runtime_error(
        "SDPA backward manifest is missing its dK/dV stage");
  }
  if (different_value_dimension &&
      manifest.find("\"function\": \"_sdpa_bwd_dv_kernel\"") ==
          std::string::npos) {
    throw std::runtime_error(
        "SDPA backward manifest is missing its separate dV stage");
  }
}

void require_manifest_tokens(std::string_view manifest,
                             std::string_view operation,
                             std::span<const std::string_view> tokens) {
  for (const std::string_view token : tokens) {
    if (manifest.find(token) == std::string_view::npos) {
      throw std::runtime_error(std::string(operation) +
                               " manifest is missing " + std::string(token));
    }
  }
}

void verify_fp8_forward_jit_artifact(
    const std::filesystem::path& cache,
    std::span<const std::filesystem::path> previous_manifests, bool autotune,
    std::size_t previous_selection_count) {
  const GeneratedArtifacts artifacts = generated_artifacts(cache);
  const std::string manifest =
      read_file(new_manifest(cache, previous_manifests, "FP8 SDPA"));
  const std::array<std::string_view, 7> tokens{
      "\"engine\": \"libtriton_jit\"",
      "\"ownership\": \"platform\"",
      "\"provider\": \"nvidia_triton\"",
      "\"source\": \"attention.py\"",
      "\"function\": \"_zero_sdpa_fp8_fwd_amax_kernel\"",
      "\"function\": \"_sdpa_fp8_fwd_kernel\"",
      "\"source_node_ids\":"};
  require_manifest_tokens(manifest, "FP8 SDPA", tokens);
  if (autotune) {
    require_manifest_tokens(
        manifest, "FP8 SDPA",
        std::array<std::string_view, 1>{"\"table\": \"sdpa_fp8\""});
    // The manifest also contains the single fixed zero-initialization stage;
    // the remaining 4 variants are the safe sdpa_fp8 YAML search space.
    if (count_occurrences(manifest, "\"variant_id\":") != 5 ||
        artifacts.selections.size() != previous_selection_count + 1) {
      throw std::runtime_error(
          "FP8 SDPA did not time all 4 YAML tuning candidates");
    }
  }
}

void verify_fp8_backward_jit_artifact(
    const std::filesystem::path& cache,
    std::span<const std::filesystem::path> previous_manifests, bool autotune,
    std::size_t previous_selection_count) {
  const GeneratedArtifacts artifacts = generated_artifacts(cache);
  const std::string manifest =
      read_file(new_manifest(cache, previous_manifests, "FP8 SDPA backward"));
  const std::array<std::string_view, 7> tokens{
      "\"engine\": \"libtriton_jit\"",
      "\"ownership\": \"platform\"",
      "\"provider\": \"nvidia_triton\"",
      "\"source\": \"attention.py\"",
      "\"function\": \"_zero_sdpa_fp8_bwd_amax_kernel\"",
      "\"function\": \"_sdpa_fp8_bwd_dq_kernel\"",
      "\"function\": \"_sdpa_fp8_bwd_dkdv_kernel\""};
  require_manifest_tokens(manifest, "FP8 SDPA backward", tokens);
  if (autotune) {
    require_manifest_tokens(manifest, "FP8 SDPA backward",
                            std::array<std::string_view, 2>{
                                "\"table\": \"sdpa_fp8_backward_dq\"",
                                "\"table\": \"sdpa_fp8_backward_dkdv\""});
    std::size_t tuning_stages = 0;
    std::size_t position = 0;
    while ((position = manifest.find("\"variants\"", position)) !=
           std::string::npos) {
      ++tuning_stages;
      ++position;
    }
    if (artifacts.selections.size() !=
        previous_selection_count + tuning_stages) {
      throw std::runtime_error(
          "FP8 SDPA backward did not persist all autotune selections");
    }
  }
}

void append_binding(std::vector<flagdnnBinding_t>& bindings,
                    const TestTensor& specification,
                    const TensorAllocation& allocation) {
  bindings.push_back({specification.uid, allocation.pointer()});
}

void append_binding(std::vector<flagdnnBinding_t>& bindings,
                    const Fp8Scalar& specification,
                    const TensorAllocation& allocation) {
  append_binding(bindings, specification.tensor, allocation);
}

void run_forward_case(const SdpaTestCase& test_case, flagdnn::Handle& handle,
                      const std::filesystem::path& cache, Stream& stream,
                      bool benchmark = false) {
  timing::RuntimeMeasurements flagdnn_runtime;
  timing::RuntimeMeasurements reference_runtime;
  flagdnn_runtime.build_cache = "fresh_artifact_cache";
  const GeneratedArtifacts before_build = generated_artifacts(cache);
  auto flagdnn = timing::profile_build(
      [&] { return build_flagdnn_sdpa(handle, test_case); },
      benchmark ? &flagdnn_runtime : nullptr);
  if (test_case.autotune) {
    verify_forward_jit_artifact(cache, test_case, before_build);
    const GeneratedArtifacts before = generated_artifacts(cache);
    const std::vector<std::string> cached_selection =
        selection_contents(before);
    auto cache_hit = build_flagdnn_sdpa(handle, test_case);
    (void)cache_hit;
    const GeneratedArtifacts after = generated_artifacts(cache);
    if (before.manifests != after.manifests ||
        before.selections != after.selections ||
        cached_selection != selection_contents(after)) {
      throw std::runtime_error(
          "SDPA repeated build did not reuse its autotune selection cache");
    }
  }
  auto reference =
      timing::profile_build([&] { return build_sdpa_reference(test_case); },
                            benchmark ? &reference_runtime : nullptr);

  auto q = TensorAllocation::input(test_case.q, 0, stream, 0.5F);
  auto k = TensorAllocation::input(test_case.k, 1, stream, 0.5F);
  auto v = TensorAllocation::input(test_case.v, 2, stream, 0.5F);
  std::unique_ptr<TensorAllocation> bias;
  if (test_case.bias.has_value()) {
    bias = TensorAllocation::input(*test_case.bias, 3, stream, 0.25F);
  }
  auto flagdnn_output = TensorAllocation::output(test_case.output, stream);
  auto reference_output = TensorAllocation::output(test_case.output, stream);
  std::unique_ptr<TensorAllocation> flagdnn_stats;
  std::unique_ptr<TensorAllocation> reference_stats;
  if (test_case.stats.has_value()) {
    flagdnn_stats = TensorAllocation::output(*test_case.stats, stream);
    reference_stats = TensorAllocation::output(*test_case.stats, stream);
  }

  std::vector<flagdnnBinding_t> flagdnn_bindings;
  std::vector<flagdnnBinding_t> reference_bindings;
  for (auto* bindings : {&flagdnn_bindings, &reference_bindings}) {
    append_binding(*bindings, test_case.q, *q);
    append_binding(*bindings, test_case.k, *k);
    append_binding(*bindings, test_case.v, *v);
    if (test_case.bias.has_value()) {
      append_binding(*bindings, *test_case.bias, *bias);
    }
  }
  append_binding(flagdnn_bindings, test_case.output, *flagdnn_output);
  append_binding(reference_bindings, test_case.output, *reference_output);
  if (test_case.stats.has_value()) {
    append_binding(flagdnn_bindings, *test_case.stats, *flagdnn_stats);
    append_binding(reference_bindings, *test_case.stats, *reference_stats);
  }
  DeviceBuffer flagdnn_workspace(flagdnn->workspace_size());
  DeviceBuffer reference_workspace(reference->workspace_size());
  stream.synchronize();
  try {
    execute(*flagdnn, flagdnn_bindings, flagdnn_workspace, stream);
    stream.synchronize();
  } catch (const std::exception& error) {
    throw std::runtime_error(std::string("FlagDNN SDPA execution: ") +
                             error.what());
  }
  try {
    execute(*reference, reference_bindings, reference_workspace, stream);
    stream.synchronize();
  } catch (const std::exception& error) {
    throw std::runtime_error(std::string("cuDNN SDPA execution: ") +
                             error.what());
  }

  const Accuracy output_accuracy = compare_tensor(
      test_case.name, "output", test_case.output, *flagdnn_output,
      *reference_output, test_case.output_absolute_tolerance,
      test_case.output_relative_tolerance, stream);
  Accuracy stats_accuracy;
  if (test_case.stats.has_value()) {
    stats_accuracy = compare_tensor(test_case.name, "stats", *test_case.stats,
                                    *flagdnn_stats, *reference_stats,
                                    test_case.stats_absolute_tolerance,
                                    test_case.stats_relative_tolerance, stream);
  }
  std::cout << test_case.name << ": FlagDNN Graph vs cuDNN Graph PASS"
            << " output_max_abs=" << output_accuracy.maximum_absolute
            << " output_max_rel=" << output_accuracy.maximum_relative;
  if (test_case.stats.has_value()) {
    std::cout << " stats_max_abs=" << stats_accuracy.maximum_absolute
              << " stats_max_rel=" << stats_accuracy.maximum_relative;
  }
  std::cout << std::endl;
  if (benchmark) {
    measure_attention_pair(test_case.name, *flagdnn, *reference,
                           flagdnn_bindings, reference_bindings,
                           flagdnn_workspace, reference_workspace, stream,
                           flagdnn_runtime, reference_runtime);
  }
}

void run_fp8_forward_case(const SdpaFp8TestCase& test_case,
                          flagdnn::Handle& handle,
                          const std::filesystem::path& cache, Stream& stream,
                          bool benchmark = false) {
  timing::RuntimeMeasurements flagdnn_runtime;
  timing::RuntimeMeasurements reference_runtime;
  flagdnn_runtime.build_cache = "fresh_artifact_cache";
  const GeneratedArtifacts before_build = generated_artifacts(cache);
  auto flagdnn = timing::profile_build(
      [&] { return build_flagdnn_sdpa_fp8(handle, test_case); },
      benchmark ? &flagdnn_runtime : nullptr);
  verify_fp8_forward_jit_artifact(cache, before_build.manifests,
                                  test_case.autotune,
                                  before_build.selections.size());
  if (test_case.autotune) {
    const GeneratedArtifacts before_cache_hit = generated_artifacts(cache);
    const std::vector<std::string> cached_selection =
        selection_contents(before_cache_hit);
    auto cache_hit = build_flagdnn_sdpa_fp8(handle, test_case);
    static_cast<void>(cache_hit);
    const GeneratedArtifacts after_cache_hit = generated_artifacts(cache);
    if (before_cache_hit.manifests != after_cache_hit.manifests ||
        before_cache_hit.selections != after_cache_hit.selections ||
        cached_selection != selection_contents(after_cache_hit)) {
      throw std::runtime_error(
          "FP8 SDPA repeated build did not reuse its autotune cache");
    }
  }
  auto reference =
      timing::profile_build([&] { return build_sdpa_fp8_reference(test_case); },
                            benchmark ? &reference_runtime : nullptr);

  auto q = TensorAllocation::input(test_case.q, 20, stream, 1.0F);
  auto k = TensorAllocation::input(test_case.k, 21, stream, 1.0F);
  auto v = TensorAllocation::input(test_case.v, 22, stream, 1.0F);
  auto descale_q = TensorAllocation::scalar(test_case.descale_q, stream);
  auto descale_k = TensorAllocation::scalar(test_case.descale_k, stream);
  auto descale_v = TensorAllocation::scalar(test_case.descale_v, stream);
  auto descale_s = TensorAllocation::scalar(test_case.descale_s, stream);
  auto scale_s = TensorAllocation::scalar(test_case.scale_s, stream);
  auto scale_o = TensorAllocation::scalar(test_case.scale_o, stream);
  std::unique_ptr<TensorAllocation> bias;
  if (test_case.bias.has_value()) {
    bias = TensorAllocation::input(*test_case.bias, 23, stream, 0.125F);
  }
  auto flagdnn_output = TensorAllocation::output(test_case.output, stream);
  auto reference_output = TensorAllocation::output(test_case.output, stream);
  std::unique_ptr<TensorAllocation> flagdnn_stats;
  std::unique_ptr<TensorAllocation> reference_stats;
  if (test_case.stats.has_value()) {
    flagdnn_stats = TensorAllocation::output(*test_case.stats, stream);
    reference_stats = TensorAllocation::output(*test_case.stats, stream);
  }
  auto flagdnn_amax_s = TensorAllocation::output(test_case.amax_s, stream);
  auto flagdnn_amax_o = TensorAllocation::output(test_case.amax_o, stream);
  auto reference_amax_s = TensorAllocation::output(test_case.amax_s, stream);
  auto reference_amax_o = TensorAllocation::output(test_case.amax_o, stream);

  std::vector<flagdnnBinding_t> flagdnn_bindings;
  std::vector<flagdnnBinding_t> reference_bindings;
  for (auto* bindings : {&flagdnn_bindings, &reference_bindings}) {
    append_binding(*bindings, test_case.q, *q);
    append_binding(*bindings, test_case.k, *k);
    append_binding(*bindings, test_case.v, *v);
    append_binding(*bindings, test_case.descale_q, *descale_q);
    append_binding(*bindings, test_case.descale_k, *descale_k);
    append_binding(*bindings, test_case.descale_v, *descale_v);
    append_binding(*bindings, test_case.descale_s, *descale_s);
    append_binding(*bindings, test_case.scale_s, *scale_s);
    append_binding(*bindings, test_case.scale_o, *scale_o);
    if (test_case.bias.has_value()) {
      append_binding(*bindings, *test_case.bias, *bias);
    }
  }
  append_binding(flagdnn_bindings, test_case.output, *flagdnn_output);
  append_binding(reference_bindings, test_case.output, *reference_output);
  if (test_case.stats.has_value()) {
    append_binding(flagdnn_bindings, *test_case.stats, *flagdnn_stats);
    append_binding(reference_bindings, *test_case.stats, *reference_stats);
  }
  append_binding(flagdnn_bindings, test_case.amax_s, *flagdnn_amax_s);
  append_binding(flagdnn_bindings, test_case.amax_o, *flagdnn_amax_o);
  append_binding(reference_bindings, test_case.amax_s, *reference_amax_s);
  append_binding(reference_bindings, test_case.amax_o, *reference_amax_o);

  DeviceBuffer flagdnn_workspace(flagdnn->workspace_size());
  DeviceBuffer reference_workspace(reference->workspace_size());
  stream.synchronize();
  execute(*flagdnn, flagdnn_bindings, flagdnn_workspace, stream);
  execute(*reference, reference_bindings, reference_workspace, stream);
  stream.synchronize();

  const Accuracy output_accuracy = compare_tensor(
      test_case.name, "output", test_case.output, *flagdnn_output,
      *reference_output, test_case.output_absolute_tolerance,
      test_case.output_relative_tolerance, stream);
  Accuracy stats_accuracy;
  if (test_case.stats.has_value()) {
    stats_accuracy = compare_tensor(test_case.name, "stats", *test_case.stats,
                                    *flagdnn_stats, *reference_stats,
                                    test_case.stats_absolute_tolerance,
                                    test_case.stats_relative_tolerance, stream);
  }
  const Accuracy amax_s_accuracy = compare_tensor(
      test_case.name, "amax_s", test_case.amax_s, *flagdnn_amax_s,
      *reference_amax_s, test_case.amax_absolute_tolerance,
      test_case.amax_relative_tolerance, stream);
  const Accuracy amax_o_accuracy = compare_tensor(
      test_case.name, "amax_o", test_case.amax_o, *flagdnn_amax_o,
      *reference_amax_o, test_case.amax_absolute_tolerance,
      test_case.amax_relative_tolerance, stream);
  std::cout << test_case.name << ": FlagDNN Graph vs cuDNN Graph PASS"
            << " output_max_abs=" << output_accuracy.maximum_absolute
            << " amax_s_abs=" << amax_s_accuracy.maximum_absolute
            << " amax_o_abs=" << amax_o_accuracy.maximum_absolute;
  if (test_case.stats.has_value()) {
    std::cout << " stats_max_abs=" << stats_accuracy.maximum_absolute;
  }
  std::cout << std::endl;
  if (benchmark) {
    measure_attention_pair(test_case.name, *flagdnn, *reference,
                           flagdnn_bindings, reference_bindings,
                           flagdnn_workspace, reference_workspace, stream,
                           flagdnn_runtime, reference_runtime);
  }
}

void run_backward_case(const SdpaBackwardTestCase& test_case,
                       flagdnn::Handle& handle,
                       const std::filesystem::path& cache, Stream& stream,
                       bool benchmark = false) {
  timing::RuntimeMeasurements flagdnn_runtime;
  timing::RuntimeMeasurements reference_runtime;
  flagdnn_runtime.build_cache = "fresh_artifact_cache";
  const std::vector<std::filesystem::path> manifests =
      generated_artifacts(cache).manifests;
  auto flagdnn = timing::profile_build(
      [&] { return build_flagdnn_sdpa_backward(handle, test_case); },
      benchmark ? &flagdnn_runtime : nullptr);
  verify_backward_jit_artifact(
      cache, manifests, test_case.q.dimensions[3] != test_case.v.dimensions[3],
      test_case.autotune, test_case.q.data_type == FLAGDNN_DATA_FLOAT32);
  auto reference = timing::profile_build(
      [&] { return build_sdpa_backward_reference(test_case); },
      benchmark ? &reference_runtime : nullptr);

  auto q = TensorAllocation::input(test_case.q, 10, stream, 0.5F);
  auto k = TensorAllocation::input(test_case.k, 11, stream, 0.5F);
  auto v = TensorAllocation::input(test_case.v, 12, stream, 0.5F);
  auto doutput = TensorAllocation::input(test_case.doutput, 13, stream, 0.25F);
  std::unique_ptr<TensorAllocation> bias;
  if (test_case.bias.has_value()) {
    bias = TensorAllocation::input(*test_case.bias, 14, stream, 0.25F);
  }
  SdpaTestCase primal;
  primal.name = test_case.name + "::primal";
  primal.q = test_case.q;
  primal.k = test_case.k;
  primal.v = test_case.v;
  primal.bias = test_case.bias;
  primal.output = test_case.output;
  primal.stats = test_case.stats;
  primal.options = test_case.options;
  auto primal_reference = build_sdpa_reference(primal);
  auto output = TensorAllocation::output(test_case.output, stream);
  auto stats = TensorAllocation::output(test_case.stats, stream);
  std::vector<flagdnnBinding_t> primal_bindings;
  append_binding(primal_bindings, primal.q, *q);
  append_binding(primal_bindings, primal.k, *k);
  append_binding(primal_bindings, primal.v, *v);
  append_binding(primal_bindings, primal.output, *output);
  append_binding(primal_bindings, *primal.stats, *stats);
  if (primal.bias.has_value())
    append_binding(primal_bindings, *primal.bias, *bias);
  DeviceBuffer primal_workspace(primal_reference->workspace_size());
  stream.synchronize();
  execute(*primal_reference, primal_bindings, primal_workspace, stream);
  stream.synchronize();

  auto flagdnn_dq = TensorAllocation::output(test_case.dq, stream);
  auto flagdnn_dk = TensorAllocation::output(test_case.dk, stream);
  auto flagdnn_dv = TensorAllocation::output(test_case.dv, stream);
  auto reference_dq = TensorAllocation::output(test_case.dq, stream);
  auto reference_dk = TensorAllocation::output(test_case.dk, stream);
  auto reference_dv = TensorAllocation::output(test_case.dv, stream);
  std::unique_ptr<TensorAllocation> flagdnn_dbias;
  std::unique_ptr<TensorAllocation> reference_dbias;
  if (test_case.dbias.has_value()) {
    flagdnn_dbias = TensorAllocation::output(*test_case.dbias, stream);
    reference_dbias = TensorAllocation::output(*test_case.dbias, stream);
  }

  std::vector<flagdnnBinding_t> flagdnn_bindings;
  std::vector<flagdnnBinding_t> reference_bindings;
  for (auto* bindings : {&flagdnn_bindings, &reference_bindings}) {
    append_binding(*bindings, test_case.q, *q);
    append_binding(*bindings, test_case.k, *k);
    append_binding(*bindings, test_case.v, *v);
    append_binding(*bindings, test_case.output, *output);
    append_binding(*bindings, test_case.doutput, *doutput);
    append_binding(*bindings, test_case.stats, *stats);
    if (test_case.bias.has_value()) {
      append_binding(*bindings, *test_case.bias, *bias);
    }
  }
  append_binding(flagdnn_bindings, test_case.dq, *flagdnn_dq);
  append_binding(flagdnn_bindings, test_case.dk, *flagdnn_dk);
  append_binding(flagdnn_bindings, test_case.dv, *flagdnn_dv);
  append_binding(reference_bindings, test_case.dq, *reference_dq);
  append_binding(reference_bindings, test_case.dk, *reference_dk);
  append_binding(reference_bindings, test_case.dv, *reference_dv);
  if (test_case.dbias.has_value()) {
    append_binding(flagdnn_bindings, *test_case.dbias, *flagdnn_dbias);
    append_binding(reference_bindings, *test_case.dbias, *reference_dbias);
  }
  DeviceBuffer flagdnn_workspace(flagdnn->workspace_size());
  DeviceBuffer reference_workspace(reference->workspace_size());
  stream.synchronize();
  try {
    execute(*flagdnn, flagdnn_bindings, flagdnn_workspace, stream);
    stream.synchronize();
  } catch (const std::exception& error) {
    throw std::runtime_error(test_case.name +
                             ": FlagDNN execution: " + error.what());
  }
  try {
    execute(*reference, reference_bindings, reference_workspace, stream);
    stream.synchronize();
  } catch (const std::exception& error) {
    throw std::runtime_error(test_case.name +
                             ": cuDNN execution: " + error.what());
  }

  Accuracy maximum;
  const auto compare_gradient =
      [&](std::string_view name, const TestTensor& specification,
          const TensorAllocation& actual, const TensorAllocation& expected) {
        const Accuracy accuracy = compare_tensor(
            test_case.name, name, specification, actual, expected,
            test_case.absolute_tolerance, test_case.relative_tolerance, stream);
        maximum.maximum_absolute =
            std::max(maximum.maximum_absolute, accuracy.maximum_absolute);
        maximum.maximum_relative =
            std::max(maximum.maximum_relative, accuracy.maximum_relative);
      };
  compare_gradient("dq", test_case.dq, *flagdnn_dq, *reference_dq);
  compare_gradient("dk", test_case.dk, *flagdnn_dk, *reference_dk);
  compare_gradient("dv", test_case.dv, *flagdnn_dv, *reference_dv);
  if (test_case.dbias.has_value()) {
    compare_gradient("dbias", *test_case.dbias, *flagdnn_dbias,
                     *reference_dbias);
  }
  std::cout << test_case.name << ": FlagDNN Graph vs cuDNN Graph PASS"
            << " max_abs=" << maximum.maximum_absolute
            << " max_rel=" << maximum.maximum_relative << std::endl;
  if (benchmark) {
    measure_attention_pair(test_case.name, *flagdnn, *reference,
                           flagdnn_bindings, reference_bindings,
                           flagdnn_workspace, reference_workspace, stream,
                           flagdnn_runtime, reference_runtime);
  }
}

void run_fp8_backward_case(const SdpaFp8BackwardTestCase& test_case,
                           flagdnn::Handle& handle,
                           const std::filesystem::path& cache, Stream& stream,
                           bool benchmark = false) {
  timing::RuntimeMeasurements flagdnn_runtime;
  timing::RuntimeMeasurements reference_runtime;
  flagdnn_runtime.build_cache = "fresh_artifact_cache";
  const GeneratedArtifacts before_build = generated_artifacts(cache);
  auto flagdnn = timing::profile_build(
      [&] { return build_flagdnn_sdpa_fp8_backward(handle, test_case); },
      benchmark ? &flagdnn_runtime : nullptr);
  verify_fp8_backward_jit_artifact(cache, before_build.manifests,
                                   test_case.autotune,
                                   before_build.selections.size());
  if (test_case.autotune) {
    const GeneratedArtifacts before_cache_hit = generated_artifacts(cache);
    const std::vector<std::string> cached_selection =
        selection_contents(before_cache_hit);
    auto cache_hit = build_flagdnn_sdpa_fp8_backward(handle, test_case);
    static_cast<void>(cache_hit);
    const GeneratedArtifacts after_cache_hit = generated_artifacts(cache);
    if (before_cache_hit.manifests != after_cache_hit.manifests ||
        before_cache_hit.selections != after_cache_hit.selections ||
        cached_selection != selection_contents(after_cache_hit)) {
      throw std::runtime_error(
          "FP8 SDPA backward repeated build did not reuse autotune cache");
    }
  }
  auto reference = timing::profile_build(
      [&] { return build_sdpa_fp8_backward_reference(test_case); },
      benchmark ? &reference_runtime : nullptr);

  auto q = TensorAllocation::input(test_case.q, 30, stream, 1.0F);
  auto k = TensorAllocation::input(test_case.k, 31, stream, 1.0F);
  auto v = TensorAllocation::input(test_case.v, 32, stream, 1.0F);
  auto doutput = TensorAllocation::input(test_case.doutput, 33, stream, 0.5F);
  const std::array<const Fp8Scalar*, 12> scale_specs{{
      &test_case.descale_q,
      &test_case.descale_k,
      &test_case.descale_v,
      &test_case.descale_o,
      &test_case.descale_doutput,
      &test_case.descale_s,
      &test_case.descale_dp,
      &test_case.scale_s,
      &test_case.scale_dq,
      &test_case.scale_dk,
      &test_case.scale_dv,
      &test_case.scale_dp,
  }};
  std::vector<std::unique_ptr<TensorAllocation>> scale_allocations;
  scale_allocations.reserve(scale_specs.size());
  for (const Fp8Scalar* scalar_value : scale_specs) {
    scale_allocations.push_back(
        TensorAllocation::scalar(*scalar_value, stream));
  }

  const std::int64_t auxiliary_uid = test_case.amax_dp.uid + 1;
  const Fp8Scalar primal_scale_o{
      {auxiliary_uid, FLAGDNN_DATA_FLOAT32, {1, 1, 1, 1}, {1, 1, 1, 1}},
      1.0F / test_case.descale_o.value};
  SdpaFp8TestCase primal;
  primal.name = test_case.name + "::primal";
  primal.q = test_case.q;
  primal.k = test_case.k;
  primal.v = test_case.v;
  primal.descale_q = test_case.descale_q;
  primal.descale_k = test_case.descale_k;
  primal.descale_v = test_case.descale_v;
  primal.descale_s = test_case.descale_s;
  primal.scale_s = test_case.scale_s;
  primal.scale_o = primal_scale_o;
  primal.output = test_case.output;
  primal.stats = test_case.stats;
  primal.amax_s = {
      auxiliary_uid + 1, FLAGDNN_DATA_FLOAT32, {1, 1, 1, 1}, {1, 1, 1, 1}};
  primal.amax_o = {
      auxiliary_uid + 2, FLAGDNN_DATA_FLOAT32, {1, 1, 1, 1}, {1, 1, 1, 1}};
  primal.options = test_case.options;
  auto primal_reference = build_sdpa_fp8_reference(primal);
  auto primal_scale_o_allocation =
      TensorAllocation::scalar(primal.scale_o, stream);
  auto output = TensorAllocation::output(test_case.output, stream);
  auto stats = TensorAllocation::output(test_case.stats, stream);
  auto primal_amax_s = TensorAllocation::output(primal.amax_s, stream);
  auto primal_amax_o = TensorAllocation::output(primal.amax_o, stream);
  std::vector<flagdnnBinding_t> primal_bindings;
  append_binding(primal_bindings, primal.q, *q);
  append_binding(primal_bindings, primal.k, *k);
  append_binding(primal_bindings, primal.v, *v);
  append_binding(primal_bindings, primal.descale_q, *scale_allocations[0]);
  append_binding(primal_bindings, primal.descale_k, *scale_allocations[1]);
  append_binding(primal_bindings, primal.descale_v, *scale_allocations[2]);
  append_binding(primal_bindings, primal.descale_s, *scale_allocations[5]);
  append_binding(primal_bindings, primal.scale_s, *scale_allocations[7]);
  append_binding(primal_bindings, primal.scale_o, *primal_scale_o_allocation);
  append_binding(primal_bindings, primal.output, *output);
  append_binding(primal_bindings, *primal.stats, *stats);
  append_binding(primal_bindings, primal.amax_s, *primal_amax_s);
  append_binding(primal_bindings, primal.amax_o, *primal_amax_o);
  DeviceBuffer primal_workspace(primal_reference->workspace_size());
  stream.synchronize();
  execute(*primal_reference, primal_bindings, primal_workspace, stream);
  stream.synchronize();

  auto flagdnn_dq = TensorAllocation::output(test_case.dq, stream);
  auto flagdnn_dk = TensorAllocation::output(test_case.dk, stream);
  auto flagdnn_dv = TensorAllocation::output(test_case.dv, stream);
  auto reference_dq = TensorAllocation::output(test_case.dq, stream);
  auto reference_dk = TensorAllocation::output(test_case.dk, stream);
  auto reference_dv = TensorAllocation::output(test_case.dv, stream);
  const std::array<const TestTensor*, 4> amax_specs{{
      &test_case.amax_dq,
      &test_case.amax_dk,
      &test_case.amax_dv,
      &test_case.amax_dp,
  }};
  std::vector<std::unique_ptr<TensorAllocation>> flagdnn_amaxes;
  std::vector<std::unique_ptr<TensorAllocation>> reference_amaxes;
  for (const TestTensor* amax : amax_specs) {
    flagdnn_amaxes.push_back(TensorAllocation::output(*amax, stream));
    reference_amaxes.push_back(TensorAllocation::output(*amax, stream));
  }

  std::vector<flagdnnBinding_t> flagdnn_bindings;
  std::vector<flagdnnBinding_t> reference_bindings;
  for (auto* bindings : {&flagdnn_bindings, &reference_bindings}) {
    append_binding(*bindings, test_case.q, *q);
    append_binding(*bindings, test_case.k, *k);
    append_binding(*bindings, test_case.v, *v);
    append_binding(*bindings, test_case.output, *output);
    append_binding(*bindings, test_case.doutput, *doutput);
    append_binding(*bindings, test_case.stats, *stats);
    for (std::size_t index = 0; index < scale_specs.size(); ++index) {
      append_binding(*bindings, *scale_specs[index], *scale_allocations[index]);
    }
  }
  append_binding(flagdnn_bindings, test_case.dq, *flagdnn_dq);
  append_binding(flagdnn_bindings, test_case.dk, *flagdnn_dk);
  append_binding(flagdnn_bindings, test_case.dv, *flagdnn_dv);
  append_binding(reference_bindings, test_case.dq, *reference_dq);
  append_binding(reference_bindings, test_case.dk, *reference_dk);
  append_binding(reference_bindings, test_case.dv, *reference_dv);
  for (std::size_t index = 0; index < amax_specs.size(); ++index) {
    append_binding(flagdnn_bindings, *amax_specs[index],
                   *flagdnn_amaxes[index]);
    append_binding(reference_bindings, *amax_specs[index],
                   *reference_amaxes[index]);
  }

  DeviceBuffer flagdnn_workspace(flagdnn->workspace_size());
  DeviceBuffer reference_workspace(reference->workspace_size());
  stream.synchronize();
  execute(*flagdnn, flagdnn_bindings, flagdnn_workspace, stream);
  execute(*reference, reference_bindings, reference_workspace, stream);
  stream.synchronize();

  Accuracy maximum_gradient;
  for (const auto& [name, specification, actual, expected] :
       std::array<std::tuple<std::string_view, const TestTensor*,
                             const TensorAllocation*, const TensorAllocation*>,
                  3>{{
           {"dq", &test_case.dq, flagdnn_dq.get(), reference_dq.get()},
           {"dk", &test_case.dk, flagdnn_dk.get(), reference_dk.get()},
           {"dv", &test_case.dv, flagdnn_dv.get(), reference_dv.get()},
       }}) {
    const Accuracy accuracy =
        compare_tensor(test_case.name, name, *specification, *actual, *expected,
                       test_case.gradient_absolute_tolerance,
                       test_case.gradient_relative_tolerance, stream);
    maximum_gradient.maximum_absolute =
        std::max(maximum_gradient.maximum_absolute, accuracy.maximum_absolute);
    maximum_gradient.maximum_relative =
        std::max(maximum_gradient.maximum_relative, accuracy.maximum_relative);
  }
  Accuracy maximum_amax;
  constexpr std::array<std::string_view, 4> amax_names{"amax_dq", "amax_dk",
                                                       "amax_dv", "amax_dp"};
  for (std::size_t index = 0; index < amax_specs.size(); ++index) {
    const Accuracy accuracy =
        compare_tensor(test_case.name, amax_names[index], *amax_specs[index],
                       *flagdnn_amaxes[index], *reference_amaxes[index],
                       test_case.amax_absolute_tolerance,
                       test_case.amax_relative_tolerance, stream);
    maximum_amax.maximum_absolute =
        std::max(maximum_amax.maximum_absolute, accuracy.maximum_absolute);
    maximum_amax.maximum_relative =
        std::max(maximum_amax.maximum_relative, accuracy.maximum_relative);
  }
  std::cout << test_case.name << ": FlagDNN Graph vs cuDNN Graph PASS"
            << " gradient_max_abs=" << maximum_gradient.maximum_absolute
            << " gradient_max_rel=" << maximum_gradient.maximum_relative
            << " amax_max_abs=" << maximum_amax.maximum_absolute
            << " amax_max_rel=" << maximum_amax.maximum_relative << std::endl;
  if (benchmark) {
    measure_attention_pair(test_case.name, *flagdnn, *reference,
                           flagdnn_bindings, reference_bindings,
                           flagdnn_workspace, reference_workspace, stream,
                           flagdnn_runtime, reference_runtime);
  }
}

void configure_jit() {
  if (setenv("FLAGDNN_EXECUTION_ENGINE", "libtriton_jit", 1) != 0) {
    throw std::runtime_error("cannot select libtriton_jit engine");
  }
}

template <typename Cases, typename RunCase>
int run_attention_benchmark_cases(int argc, char** argv, const Cases& cases,
                                  RunCase run_case) {
  if (argc != 3) return 2;
  try {
    std::cout << std::setprecision(9);
    configure_jit();
    DriverContext driver;
    Stream stream;
    TemporaryCache cache("attention-benchmark");
    flagdnn::Handle handle(FLAGDNN_BACKEND_NVIDIA, 0);
    handle.set_compiler(argv[1], argv[2], cache.path().string());
    const char* filter = std::getenv("FLAGDNN_BENCHMARK_CASE");
    std::size_t executed = 0;
    for (const auto& test_case : cases) {
      if (test_case.q.data_type == FLAGDNN_DATA_FLOAT32) continue;
      if (filter != nullptr && filter[0] != '\0' && test_case.name != filter)
        continue;
      run_case(test_case, handle, cache.path(), stream, true);
      ++executed;
    }
    if (executed == 0)
      throw std::runtime_error("Attention benchmark filter matched no cases");
    std::cout << "FLAGDNN_ATTENTION_BENCHMARK: PASS cases=" << executed
              << std::endl;
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FLAGDNN_ATTENTION_BENCHMARK_FAILED: " << error.what()
              << std::endl;
    return 1;
  }
}

}  // namespace

int run_sdpa_functional_test(int argc, char** argv,
                             std::span<const SdpaTestCase> cases) {
  if (argc != 3) {
    std::cerr << "usage: " << argv[0] << " COMPILER_EXECUTABLE COMPILER_ENTRY"
              << std::endl;
    return 2;
  }
  try {
    std::cout << std::setprecision(9);
    configure_jit();
    DriverContext driver;
    Stream stream;
    TemporaryCache cache("sdpa");
    flagdnn::Handle handle(FLAGDNN_BACKEND_NVIDIA, 0);
    handle.set_compiler(argv[1], argv[2], cache.path().string());
    const char* filter = std::getenv("FLAGDNN_SDPA_CASE");
    std::size_t executed = 0;
    for (const SdpaTestCase& test_case : cases) {
      if (test_case.q.data_type == FLAGDNN_DATA_FLOAT32) continue;
      if (filter != nullptr &&
          test_case.name.find(filter) == std::string::npos) {
        continue;
      }
      run_forward_case(test_case, handle, cache.path(), stream);
      ++executed;
    }
    if (executed == 0) {
      throw std::runtime_error("SDPA filter matched no test cases");
    }
    std::cout << "FLAGDNN_SDPA_FUNCTIONAL: PASS cases=" << executed
              << std::endl;
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FLAGDNN_SDPA_FUNCTIONAL_FAILED: " << error.what()
              << std::endl;
    return 1;
  }
}

int run_sdpa_backward_functional_test(
    int argc, char** argv, std::span<const SdpaBackwardTestCase> cases) {
  if (argc != 3) {
    std::cerr << "usage: " << argv[0] << " COMPILER_EXECUTABLE COMPILER_ENTRY"
              << std::endl;
    return 2;
  }
  try {
    std::cout << std::setprecision(9);
    configure_jit();
    DriverContext driver;
    Stream stream;
    TemporaryCache cache("sdpa-backward");
    flagdnn::Handle handle(FLAGDNN_BACKEND_NVIDIA, 0);
    handle.set_compiler(argv[1], argv[2], cache.path().string());
    const char* filter = std::getenv("FLAGDNN_SDPA_BACKWARD_CASE");
    std::size_t executed = 0;
    for (const SdpaBackwardTestCase& test_case : cases) {
      if (test_case.q.data_type == FLAGDNN_DATA_FLOAT32) continue;
      if (filter != nullptr &&
          test_case.name.find(filter) == std::string::npos) {
        continue;
      }
      run_backward_case(test_case, handle, cache.path(), stream);
      ++executed;
    }
    if (executed == 0) {
      throw std::runtime_error("SDPA backward filter matched no test cases");
    }
    std::cout << "FLAGDNN_SDPA_BACKWARD_FUNCTIONAL: PASS cases=" << executed
              << std::endl;
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FLAGDNN_SDPA_BACKWARD_FUNCTIONAL_FAILED: " << error.what()
              << std::endl;
    return 1;
  }
}

int run_sdpa_fp8_functional_test(int argc, char** argv,
                                 std::span<const SdpaFp8TestCase> cases) {
  if (argc != 3) {
    std::cerr << "usage: " << argv[0] << " COMPILER_EXECUTABLE COMPILER_ENTRY"
              << std::endl;
    return 2;
  }
  try {
    std::cout << std::setprecision(9);
    configure_jit();
    DriverContext driver;
    Stream stream;
    TemporaryCache cache("sdpa-fp8");
    flagdnn::Handle handle(FLAGDNN_BACKEND_NVIDIA, 0);
    handle.set_compiler(argv[1], argv[2], cache.path().string());
    if (cases.empty()) {
      throw std::runtime_error("FP8 SDPA test catalog is empty");
    }
    SdpaFp8TestCase unsupported_bias = cases.front();
    unsupported_bias.name = "sdpa_fp8_bias_not_supported";
    unsupported_bias.autotune = false;
    const std::int64_t sequence_q = unsupported_bias.q.dimensions[2];
    const std::int64_t sequence_kv = unsupported_bias.k.dimensions[2];
    unsupported_bias.bias = TestTensor{
        72999,
        FLAGDNN_DATA_FLOAT32,
        {1, 1, sequence_q, sequence_kv},
        {sequence_q * sequence_kv, sequence_q * sequence_kv, sequence_kv, 1}};
    const auto require_bias_rejection = [&](std::string_view implementation,
                                            auto&& build) {
      try {
        auto unexpected = build();
        static_cast<void>(unexpected);
      } catch (const std::exception& error) {
        if (std::string_view(error.what()).find("does not support bias") !=
            std::string_view::npos) {
          return;
        }
        throw std::runtime_error(
            std::string(implementation) +
            " rejected FP8 SDPA bias for an unexpected reason: " +
            error.what());
      }
      throw std::runtime_error(std::string(implementation) +
                               " unexpectedly accepted FP8 SDPA bias");
    };
    require_bias_rejection("FlagDNN", [&] {
      return build_flagdnn_sdpa_fp8(handle, unsupported_bias);
    });
    require_bias_rejection(
        "cuDNN", [&] { return build_sdpa_fp8_reference(unsupported_bias); });
    const char* filter = std::getenv("FLAGDNN_SDPA_FP8_CASE");
    std::size_t executed = 0;
    for (const SdpaFp8TestCase& test_case : cases) {
      if (filter != nullptr &&
          test_case.name.find(filter) == std::string::npos) {
        continue;
      }
      run_fp8_forward_case(test_case, handle, cache.path(), stream);
      ++executed;
    }
    if (executed == 0) {
      throw std::runtime_error("FP8 SDPA filter matched no test cases");
    }
    std::cout << "FLAGDNN_SDPA_FP8_FUNCTIONAL: PASS cases=" << executed
              << std::endl;
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FLAGDNN_SDPA_FP8_FUNCTIONAL_FAILED: " << error.what()
              << std::endl;
    return 1;
  }
}

int run_sdpa_fp8_backward_functional_test(
    int argc, char** argv, std::span<const SdpaFp8BackwardTestCase> cases) {
  if (argc != 3) {
    std::cerr << "usage: " << argv[0] << " COMPILER_EXECUTABLE COMPILER_ENTRY"
              << std::endl;
    return 2;
  }
  try {
    std::cout << std::setprecision(9);
    configure_jit();
    DriverContext driver;
    Stream stream;
    TemporaryCache cache("sdpa-fp8-backward");
    flagdnn::Handle handle(FLAGDNN_BACKEND_NVIDIA, 0);
    handle.set_compiler(argv[1], argv[2], cache.path().string());
    const char* filter = std::getenv("FLAGDNN_SDPA_FP8_BACKWARD_CASE");
    std::size_t executed = 0;
    for (const SdpaFp8BackwardTestCase& test_case : cases) {
      if (filter != nullptr &&
          test_case.name.find(filter) == std::string::npos) {
        continue;
      }
      run_fp8_backward_case(test_case, handle, cache.path(), stream);
      ++executed;
    }
    if (executed == 0) {
      throw std::runtime_error(
          "FP8 SDPA backward filter matched no test cases");
    }
    std::cout << "FLAGDNN_SDPA_FP8_BACKWARD_FUNCTIONAL: PASS cases=" << executed
              << std::endl;
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FLAGDNN_SDPA_FP8_BACKWARD_FUNCTIONAL_FAILED: " << error.what()
              << std::endl;
    return 1;
  }
}

int run_attention_benchmark_test(int argc, char** argv,
                                 AttentionBenchmarkOperation operation) {
  switch (operation) {
    case AttentionBenchmarkOperation::kForward:
      return run_attention_benchmark_cases(
          argc, argv, make_sdpa_benchmark_cases(), run_forward_case);
    case AttentionBenchmarkOperation::kBackward:
      return run_attention_benchmark_cases(
          argc, argv, make_sdpa_backward_benchmark_cases(), run_backward_case);
    case AttentionBenchmarkOperation::kFp8Forward:
      return run_attention_benchmark_cases(
          argc, argv, make_sdpa_fp8_benchmark_cases(), run_fp8_forward_case);
    case AttentionBenchmarkOperation::kFp8Backward:
      return run_attention_benchmark_cases(
          argc, argv, make_sdpa_fp8_backward_benchmark_cases(),
          run_fp8_backward_case);
  }
  return 2;
}

}  // namespace flagdnn::testing
