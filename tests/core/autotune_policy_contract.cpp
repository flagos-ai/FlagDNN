/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "backends/autotune_policy.hpp"

#include <unistd.h>

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace tune = flagdnn::backend::autotune;

void expect(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

class TemporaryCache {
 public:
  TemporaryCache() {
    std::string pattern =
        (std::filesystem::temp_directory_path() /
         "flagdnn-autotune-contract-XXXXXX").string();
    std::vector<char> name(pattern.begin(), pattern.end());
    name.push_back('\0');
    const char* created = mkdtemp(name.data());
    if (created == nullptr) {
      throw std::runtime_error("mkdtemp failed");
    }
    path = created;
  }
  ~TemporaryCache() {
    std::error_code ignored;
    std::filesystem::remove_all(path, ignored);
  }
  TemporaryCache(const TemporaryCache&) = delete;
  TemporaryCache& operator=(const TemporaryCache&) = delete;
  std::filesystem::path path;
};
}  // namespace

int main() {
  try {
    TemporaryCache cache;
    tune::SelectionRequest request{
        "space", "device", "measurement", cache.path / "selection.json",
        {"runnable"}, 0, 0};
    auto result = tune::select_best_candidate(request, {}, {});
    expect(result.candidate_index == 0 && !result.cache_hit &&
               result.median_milliseconds.empty(),
           "singleton must select without timing");
    expect(tune::select_best_candidate(request, {}, {}).cache_hit,
           "singleton selection was not persisted");

    // A runnable subset is reusable in the original compiler candidate space.
    request.candidate_ids = {"incompatible", "runnable", "also-incompatible"};
    expect(tune::find_cached_candidate(request) == 1,
           "cached subset winner was not found in full candidate space");
    for (std::string* identity : {&request.candidate_identity,
                                 &request.device_identity,
                                 &request.measurement_identity}) {
      const auto original = *identity;
      *identity += "-changed";
      expect(!tune::find_cached_candidate(request),
             "incompatible identity reused selection");
      *identity = original;
    }
    request.candidate_ids = {"different"};
    expect(!tune::find_cached_candidate(request),
           "missing candidate reused selection");
    tune::discard_cached_candidate(request);
    request.candidate_ids = {"slow", "fast"};
    request.benchmark_milliseconds = 1;
    unsigned int measurements = 0;
    result = tune::select_best_candidate(
        request, [](std::size_t, unsigned int) {},
        [&](std::size_t index, unsigned int) {
          ++measurements;
          return index == 0 ? 2.0F : 1.0F;
        });
    expect(result.candidate_index == 1 && measurements != 0,
           "multi-candidate policy failed");
    const auto measured = measurements;
    result = tune::select_best_candidate(
        request, [](std::size_t, unsigned int) {},
        [&](std::size_t, unsigned int) {
          ++measurements;
          return 1.0F;
        });
    expect(result.cache_hit && measurements == measured,
           "warm selection repeated measurement");
    request.candidate_ids = {"duplicate", "duplicate"};
    bool rejected = false;
    try {
      (void)tune::select_best_candidate(
          request, [](std::size_t, unsigned int) {},
          [](std::size_t, unsigned int) { return 1.0F; });
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    expect(rejected, "duplicate candidate ids accepted");
    std::cout << "PASS autotune selection and cache contracts\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
