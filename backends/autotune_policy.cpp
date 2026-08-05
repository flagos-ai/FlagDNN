/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "backends/autotune_policy.hpp"

#include "runtime/json.hpp"

#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <limits>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <string_view>
#include <system_error>
#include <unordered_set>

namespace flagdnn::backend::autotune {
namespace {

std::atomic<std::uint64_t> cache_temporary_counter{0};
constexpr std::int64_t kSelectionCacheSchemaVersion = 2;
constexpr std::string_view kSelectionPolicyIdentity =
    "calibrated-batched-median-finalists-ms-v3";
constexpr unsigned int kCalibrationIterations = 32;
constexpr unsigned int kMeasurementSamples = 10;
constexpr std::size_t kMaximumFinalists = 3;
constexpr unsigned int kMaximumBatchIterations = 1U << 20U;

void append_json_string(std::ostream& output, std::string_view value) {
  output.put('"');
  for (const unsigned char character : value) {
    switch (character) {
      case '"':
        output << "\\\"";
        break;
      case '\\':
        output << "\\\\";
        break;
      case '\b':
        output << "\\b";
        break;
      case '\f':
        output << "\\f";
        break;
      case '\n':
        output << "\\n";
        break;
      case '\r':
        output << "\\r";
        break;
      case '\t':
        output << "\\t";
        break;
      default:
        if (character < 0x20U) {
          constexpr char digits[] = "0123456789abcdef";
          output << "\\u00" << digits[character >> 4U]
                 << digits[character & 0x0fU];
        } else {
          output.put(static_cast<char>(character));
        }
        break;
    }
  }
  output.put('"');
}

std::optional<std::size_t> read_cached_candidate(
    const SelectionRequest& request) noexcept {
  try {
    std::error_code error;
    if (!std::filesystem::is_regular_file(request.cache_path, error) ||
        error ||
        std::filesystem::file_size(request.cache_path, error) > 4096 ||
        error) {
      return std::nullopt;
    }
    std::ifstream input(request.cache_path, std::ios::binary);
    if (!input) {
      return std::nullopt;
    }
    const std::string contents{
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>()};
    if (input.bad()) {
      return std::nullopt;
    }
    const auto root = flagdnn::native::json::parse(contents);
    if (root.at("schema_version").as_int() !=
            kSelectionCacheSchemaVersion ||
        root.at("policy_identity").as_string() !=
            kSelectionPolicyIdentity ||
        root.at("measurement_identity").as_string() !=
            request.measurement_identity ||
        root.at("device_identity").as_string() != request.device_identity ||
        root.at("candidate_identity").as_string() !=
            request.candidate_identity) {
      return std::nullopt;
    }
    const std::string& variant_id = root.at("variant_id").as_string();
    const auto selected = std::find(
        request.candidate_ids.begin(), request.candidate_ids.end(), variant_id);
    if (selected == request.candidate_ids.end()) {
      return std::nullopt;
    }
    return static_cast<std::size_t>(
        std::distance(request.candidate_ids.begin(), selected));
  } catch (...) {
    return std::nullopt;
  }
}

void write_cached_candidate(const SelectionRequest& request,
                            std::string_view variant_id) noexcept {
  try {
    const std::uint64_t serial = cache_temporary_counter.fetch_add(1);
    std::filesystem::path temporary = request.cache_path;
    temporary += ".tmp." + std::to_string(getpid()) + "." +
                 std::to_string(serial);
    {
      std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
      if (!output) {
        return;
      }
      output << "{\"schema_version\":"
             << kSelectionCacheSchemaVersion
             << ",\"policy_identity\":";
      append_json_string(output, kSelectionPolicyIdentity);
      output << ",\"measurement_identity\":";
      append_json_string(output, request.measurement_identity);
      output << ",\"device_identity\":";
      append_json_string(output, request.device_identity);
      output << ",\"candidate_identity\":";
      append_json_string(output, request.candidate_identity);
      output << ",\"variant_id\":";
      append_json_string(output, variant_id);
      output << "}\n";
      output.close();
      if (!output) {
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        return;
      }
    }
    std::error_code error;
    std::filesystem::rename(temporary, request.cache_path, error);
    if (error) {
      std::filesystem::remove(temporary, error);
    }
  } catch (...) {
  }
}

float median(std::vector<float>& samples) {
  std::sort(samples.begin(), samples.end());
  const std::size_t middle = samples.size() / 2;
  if ((samples.size() % 2U) != 0U) {
    return samples[middle];
  }
  return (samples[middle - 1] + samples[middle]) * 0.5F;
}

void validate_request(const SelectionRequest& request,
                      const WarmupCallback& warmup,
                      const MeasureCallback& measure) {
  if (request.candidate_identity.empty() || request.device_identity.empty() ||
      request.measurement_identity.empty() || request.cache_path.empty()) {
    throw std::invalid_argument("autotune identity and cache path are required");
  }
  if (request.candidate_ids.size() < 2 ||
      request.benchmark_milliseconds == 0) {
    throw std::invalid_argument(
        "autotune requires at least two candidates and a benchmark budget");
  }
  if (!warmup || !measure) {
    throw std::invalid_argument("autotune callbacks are required");
  }
  std::unordered_set<std::string> unique_ids;
  unique_ids.reserve(request.candidate_ids.size());
  for (const std::string& candidate_id : request.candidate_ids) {
    if (candidate_id.empty() || !unique_ids.insert(candidate_id).second) {
      throw std::invalid_argument(
          "autotune candidate identifiers must be nonempty and unique");
    }
  }
}

void validate_timing(float milliseconds) {
  if (!std::isfinite(milliseconds) || milliseconds <= 0.0F) {
    throw std::runtime_error(
        "autotune backend returned an invalid timing sample");
  }
}

unsigned int iterations_for_budget(unsigned int budget_milliseconds,
                                   float estimated_milliseconds) {
  if (budget_milliseconds == 0) {
    return 0;
  }
  const double iterations = std::ceil(
      static_cast<double>(budget_milliseconds) /
      static_cast<double>(estimated_milliseconds));
  return static_cast<unsigned int>(std::clamp(
      iterations,
      1.0,
      static_cast<double>(kMaximumBatchIterations)));
}

}  // namespace

std::optional<std::size_t> find_cached_candidate(
    const SelectionRequest& request) noexcept {
  return read_cached_candidate(request);
}

void discard_cached_candidate(const SelectionRequest& request) noexcept {
  std::error_code ignored;
  std::filesystem::remove(request.cache_path, ignored);
}

SelectionResult select_best_candidate(const SelectionRequest& request,
                                      const WarmupCallback& warmup,
                                      const MeasureCallback& measure) {
  validate_request(request, warmup, measure);
  if (const auto cached = find_cached_candidate(request)) {
    return {*cached, true, {}};
  }

  SelectionResult result;
  result.median_milliseconds.reserve(request.candidate_ids.size());
  std::vector<unsigned int> candidate_measurement_iterations;
  candidate_measurement_iterations.reserve(request.candidate_ids.size());
  float best_time = std::numeric_limits<float>::infinity();
  for (std::size_t candidate_index = 0;
       candidate_index < request.candidate_ids.size();
       ++candidate_index) {
    const float estimated_milliseconds =
        measure(candidate_index, kCalibrationIterations);
    validate_timing(estimated_milliseconds);

    const unsigned int warmup_iterations = iterations_for_budget(
        request.warmup_milliseconds, estimated_milliseconds);
    if (warmup_iterations != 0) {
      warmup(candidate_index, warmup_iterations);
    }

    std::vector<float> timings;
    timings.reserve(kMeasurementSamples);
    const float sample_budget =
        static_cast<float>(request.benchmark_milliseconds) /
        static_cast<float>(kMeasurementSamples);
    const unsigned int measurement_iterations = iterations_for_budget(
        std::max(1U, static_cast<unsigned int>(std::ceil(sample_budget))),
        estimated_milliseconds);
    candidate_measurement_iterations.push_back(measurement_iterations);
    for (unsigned int sample = 0; sample < kMeasurementSamples; ++sample) {
      const float milliseconds =
          measure(candidate_index, measurement_iterations);
      validate_timing(milliseconds);
      timings.push_back(milliseconds);
    }
    const float candidate_median = median(timings);
    result.median_milliseconds.push_back(candidate_median);
    if (candidate_median < best_time) {
      best_time = candidate_median;
      result.candidate_index = candidate_index;
    }
  }

  // The first pass deliberately evaluates the complete candidate space. Its
  // sequential order, however, can bias otherwise close kernels as device
  // clocks and temperature drift during a long tuning session. Re-measure the
  // leading candidates sample-by-sample with a rotating order so that every
  // finalist observes comparable device conditions before publishing a
  // persistent winner.
  std::vector<std::size_t> finalists(request.candidate_ids.size());
  for (std::size_t index = 0; index < finalists.size(); ++index) {
    finalists[index] = index;
  }
  std::stable_sort(
      finalists.begin(), finalists.end(), [&](std::size_t left,
                                               std::size_t right) {
        return result.median_milliseconds[left] <
               result.median_milliseconds[right];
      });
  finalists.resize(std::min(kMaximumFinalists, finalists.size()));

  std::vector<std::vector<float>> confirmation_timings(finalists.size());
  for (std::vector<float>& timings : confirmation_timings) {
    timings.reserve(kMeasurementSamples);
  }
  for (unsigned int sample = 0; sample < kMeasurementSamples; ++sample) {
    for (std::size_t offset = 0; offset < finalists.size(); ++offset) {
      const std::size_t finalist =
          (static_cast<std::size_t>(sample) + offset) % finalists.size();
      const std::size_t candidate_index = finalists[finalist];
      const float milliseconds = measure(
          candidate_index,
          candidate_measurement_iterations[candidate_index]);
      validate_timing(milliseconds);
      confirmation_timings[finalist].push_back(milliseconds);
    }
  }

  best_time = std::numeric_limits<float>::infinity();
  for (std::size_t finalist = 0; finalist < finalists.size(); ++finalist) {
    const std::size_t candidate_index = finalists[finalist];
    const float candidate_median = median(confirmation_timings[finalist]);
    result.median_milliseconds[candidate_index] = candidate_median;
    if (candidate_median < best_time) {
      best_time = candidate_median;
      result.candidate_index = candidate_index;
    }
  }

  write_cached_candidate(
      request, request.candidate_ids[result.candidate_index]);
  return result;
}

}  // namespace flagdnn::backend::autotune
