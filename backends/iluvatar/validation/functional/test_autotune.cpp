/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "backends/autotune_policy.hpp"

#include <flagdnn/flagdnn.hpp>
#include <flagdnn_frontend.h>

#include <triton_jit/backend_config.h>
#include <triton_jit/backends/ix_backend.h>
#include <triton_jit/triton_kernel.h>

#include <cuda.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#ifndef BACKEND_IX
#error "The Iluvatar autotune contract must compile with BACKEND_IX"
#endif

static_assert(triton_jit::IxBackend::WARP_SIZE == 64);
static_assert(
    std::is_same_v<triton_jit::DefaultBackend, triton_jit::IxBackend>);

namespace {

namespace autotune = flagdnn::backend::autotune;
namespace fe = ::flagdnn_frontend;

void expect(bool condition, std::string_view message) {
  if (!condition) {
    throw std::runtime_error(std::string(message));
  }
}

void check_driver(CUresult result, const char *operation) {
  if (result == CUDA_SUCCESS) {
    return;
  }
  const char *detail = nullptr;
  (void)cuGetErrorString(result, &detail);
  throw std::runtime_error(
      std::string(operation) + " failed: " +
      (detail == nullptr ? "unknown CoreX Driver error" : detail));
}

void check_frontend(fe::error_t status, const char *operation) {
  if (status.is_bad()) {
    throw std::runtime_error(std::string(operation) +
                             " failed: " + status.get_message());
  }
}

class TemporaryDirectory final {
public:
  explicit TemporaryDirectory(std::string_view prefix) {
    std::string pattern = (std::filesystem::temp_directory_path() /
                           (std::string(prefix) + "-XXXXXX"))
                              .string();
    std::vector<char> writable(pattern.begin(), pattern.end());
    writable.push_back('\0');
    const char *created = mkdtemp(writable.data());
    if (created == nullptr) {
      throw std::runtime_error("mkdtemp failed");
    }
    path_ = created;
  }

  ~TemporaryDirectory() {
    std::error_code ignored;
    std::filesystem::remove_all(path_, ignored);
  }

  TemporaryDirectory(const TemporaryDirectory &) = delete;
  TemporaryDirectory &operator=(const TemporaryDirectory &) = delete;

  [[nodiscard]] const std::filesystem::path &path() const noexcept {
    return path_;
  }

private:
  std::filesystem::path path_;
};

std::string read_file(const std::filesystem::path &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("cannot read " + path.string());
  }
  return std::string(std::istreambuf_iterator<char>(input),
                     std::istreambuf_iterator<char>());
}

void write_file(const std::filesystem::path &path, std::string_view contents) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output) {
    throw std::runtime_error("cannot write " + path.string());
  }
  output.write(contents.data(), static_cast<std::streamsize>(contents.size()));
  if (!output) {
    throw std::runtime_error("short write to " + path.string());
  }
}

void expect_no_temporary_cache_files(const std::filesystem::path &root) {
  for (const auto &entry :
       std::filesystem::recursive_directory_iterator(root)) {
    if (entry.path().filename().string().find(".tmp.") != std::string::npos) {
      throw std::runtime_error("autotune left a temporary cache file: " +
                               entry.path().string());
    }
  }
}

std::filesystem::path find_selection_cache(const std::filesystem::path &root) {
  std::filesystem::path result;
  std::size_t count = 0;
  for (const auto &entry :
       std::filesystem::recursive_directory_iterator(root)) {
    const std::string filename = entry.path().filename().string();
    if (entry.is_regular_file() &&
        filename.starts_with(".flagdnn-autotune-v1-stage-") &&
        entry.path().extension() == ".json") {
      result = entry.path();
      ++count;
    }
  }
  expect(count == 1, "autotune build must publish exactly one selection cache");
  return result;
}

autotune::SelectionRequest policy_request(const std::filesystem::path &cache) {
  autotune::SelectionRequest request;
  request.candidate_identity = std::string(64, '1');
  request.device_identity = "corex-device-contract";
  request.measurement_identity = "corex-event-contract-v1";
  request.cache_path = cache;
  request.candidate_ids = {"slow", "fast"};
  request.warmup_milliseconds = 1;
  request.benchmark_milliseconds = 1;
  return request;
}

void test_shared_policy_contract() {
  TemporaryDirectory temporary("flagdnn-iluvatar-autotune-policy");
  const std::filesystem::path cache = temporary.path() / "winner.json";
  autotune::SelectionRequest request = policy_request(cache);
  std::size_t warmup_calls = 0;
  std::size_t measure_calls = 0;
  const auto warmup = [&](std::size_t, unsigned int iterations) {
    expect(iterations != 0, "policy requested an empty warmup batch");
    ++warmup_calls;
  };
  const auto measure = [&](std::size_t index, unsigned int iterations) {
    expect(iterations != 0, "policy requested an empty measurement batch");
    ++measure_calls;
    return index == 0 ? 0.20F : 0.10F;
  };

  const autotune::SelectionResult miss =
      autotune::select_best_candidate(request, warmup, measure);
  expect(!miss.cache_hit && miss.candidate_index == 1,
         "policy miss did not select the fastest candidate");
  expect(warmup_calls != 0 && measure_calls != 0,
         "policy miss did not fully measure candidates");
  expect(std::filesystem::is_regular_file(cache),
         "policy miss did not atomically publish a winner");
  expect_no_temporary_cache_files(temporary.path());

  warmup_calls = 0;
  measure_calls = 0;
  const autotune::SelectionResult hit =
      autotune::select_best_candidate(request, warmup, measure);
  expect(hit.cache_hit && hit.candidate_index == 1,
         "policy hit did not return the cached winner");
  expect(warmup_calls == 0 && measure_calls == 0,
         "policy hit unexpectedly measured candidates");

  auto expect_invalidated = [&](autotune::SelectionRequest changed,
                                std::string_view name) {
    expect(!autotune::find_cached_candidate(changed).has_value(), name);
  };
  autotune::SelectionRequest changed = request;
  changed.candidate_ids = {"other", "slow"};
  expect_invalidated(changed,
                     "candidate-set change did not invalidate the winner");
  changed = request;
  changed.candidate_identity = std::string(64, '2');
  expect_invalidated(changed,
                     "tuning hash change did not invalidate the winner");
  changed = request;
  changed.device_identity = "another-corex-device";
  expect_invalidated(changed,
                     "device identity change did not invalidate the winner");
  changed = request;
  changed.measurement_identity = "corex-event-contract-v2";
  expect_invalidated(
      changed, "measurement identity change did not invalidate the winner");

  write_file(cache, "{}\n");
  expect(!autotune::find_cached_candidate(request).has_value(),
         "corrupt selection cache was accepted");
  measure_calls = 0;
  const autotune::SelectionResult retuned =
      autotune::select_best_candidate(request, warmup, measure);
  expect(!retuned.cache_hit && retuned.candidate_index == 1 &&
             measure_calls != 0,
         "corrupt selection cache did not trigger full retuning");
  expect_no_temporary_cache_files(temporary.path());

  const auto expect_measure_failure = [&](float timing, std::string_view name) {
    autotune::discard_cached_candidate(request);
    try {
      (void)autotune::select_best_candidate(
          request, warmup,
          [timing](std::size_t, unsigned int) { return timing; });
    } catch (const std::runtime_error &) {
      return;
    }
    throw std::runtime_error(std::string(name));
  };
  expect_measure_failure(0.0F, "nonpositive device timing was accepted");
  expect_measure_failure(std::numeric_limits<float>::infinity(),
                         "nonfinite device timing was accepted");

  for (const std::string_view failure :
       {"candidate compile failure", "candidate ABI mismatch",
        "candidate launch failure", "candidate device memory error"}) {
    autotune::discard_cached_candidate(request);
    try {
      (void)autotune::select_best_candidate(
          request, warmup, [failure](std::size_t, unsigned int) -> float {
            throw std::runtime_error(std::string(failure));
          });
    } catch (const std::runtime_error &error) {
      expect(error.what() == failure,
             "autotune rewrote a candidate failure diagnostic");
      continue;
    }
    throw std::runtime_error(
        "autotune discarded a failing candidate instead of failing build");
  }
}

class CurrentPrimaryContext final {
public:
  CurrentPrimaryContext() {
    check_driver(cuInit(0), "cuInit");
    check_driver(cuDeviceGet(&device_, 0), "cuDeviceGet");
    check_driver(cuDevicePrimaryCtxRetain(&context_, device_),
                 "cuDevicePrimaryCtxRetain");
    CUcontext current = nullptr;
    check_driver(cuCtxGetCurrent(&current), "cuCtxGetCurrent");
    if (current != context_) {
      check_driver(cuCtxPushCurrent(context_), "cuCtxPushCurrent");
      pushed_ = true;
    }
  }

  ~CurrentPrimaryContext() {
    if (pushed_) {
      CUcontext ignored = nullptr;
      (void)cuCtxPopCurrent(&ignored);
    }
    if (context_ != nullptr) {
      (void)cuDevicePrimaryCtxRelease(device_);
    }
  }

  CurrentPrimaryContext(const CurrentPrimaryContext &) = delete;
  CurrentPrimaryContext &operator=(const CurrentPrimaryContext &) = delete;

private:
  CUdevice device_ = 0;
  CUcontext context_ = nullptr;
  bool pushed_ = false;
};

class Stream final {
public:
  Stream() {
    check_driver(cuStreamCreate(&value_, CU_STREAM_NON_BLOCKING),
                 "cuStreamCreate");
  }
  ~Stream() {
    if (value_ != nullptr) {
      (void)cuStreamDestroy(value_);
    }
  }
  Stream(const Stream &) = delete;
  Stream &operator=(const Stream &) = delete;
  [[nodiscard]] CUstream get() const noexcept { return value_; }

private:
  CUstream value_ = nullptr;
};

class DeviceAllocation final {
public:
  explicit DeviceAllocation(std::size_t bytes) {
    if (bytes != 0) {
      check_driver(cuMemAlloc(&value_, bytes), "cuMemAlloc");
    }
  }
  ~DeviceAllocation() {
    if (value_ != 0) {
      (void)cuMemFree(value_);
    }
  }
  DeviceAllocation(const DeviceAllocation &) = delete;
  DeviceAllocation &operator=(const DeviceAllocation &) = delete;
  [[nodiscard]] CUdeviceptr get() const noexcept { return value_; }
  [[nodiscard]] void *opaque() const noexcept {
    return reinterpret_cast<void *>(static_cast<std::uintptr_t>(value_));
  }

private:
  CUdeviceptr value_ = 0;
};

void configure_add(fe::graph::Graph &graph) {
  graph.set_name("iluvatar_add_autotune_contract")
      .set_io_data_type(fe::DataType_t::FLOAT)
      .set_intermediate_data_type(fe::DataType_t::FLOAT)
      .set_compute_data_type(fe::DataType_t::FLOAT)
      .set_autotune(true);
  const auto left = graph.tensor(fe::graph::Tensor_attributes()
                                     .set_name("left")
                                     .set_uid(1)
                                     .set_data_type(fe::DataType_t::FLOAT)
                                     .set_dim({1024})
                                     .set_stride({1}));
  const auto right = graph.tensor(fe::graph::Tensor_attributes()
                                      .set_name("right")
                                      .set_uid(2)
                                      .set_data_type(fe::DataType_t::FLOAT)
                                      .set_dim({1024})
                                      .set_stride({1}));
  auto output =
      graph.pointwise(left, right,
                      fe::graph::Pointwise_attributes()
                          .set_name("add")
                          .set_mode(fe::PointwiseMode_t::ADD)
                          .set_compute_data_type(fe::DataType_t::FLOAT)
                          .set_alpha(1.0));
  output->set_name("output")
      .set_uid(3)
      .set_data_type(fe::DataType_t::FLOAT)
      .set_dim({1024})
      .set_stride({1})
      .set_output(true);
}

void build_add(fe::graph::Graph &graph, const flagdnn::Handle &handle) {
  configure_add(graph);
  check_frontend(graph.build(handle, {fe::HeurMode_t::A}),
                 "FlagDNN Iluvatar autotune Add build");
}

void execute_and_check(fe::graph::Graph &graph, const flagdnn::Handle &handle) {
  std::array<float, 1024> left{};
  std::array<float, 1024> right{};
  std::array<float, 1024> output{};
  for (std::size_t index = 0; index < left.size(); ++index) {
    left[index] = static_cast<float>(index % 31) - 4.0F;
    right[index] = static_cast<float>(index % 19) * 0.25F;
  }

  Stream stream;
  DeviceAllocation device_left(sizeof(left));
  DeviceAllocation device_right(sizeof(right));
  DeviceAllocation device_output(sizeof(output));
  DeviceAllocation workspace(graph.get_workspace_size());
  check_driver(cuMemcpyHtoDAsync(device_left.get(), left.data(), sizeof(left),
                                 stream.get()),
               "cuMemcpyHtoDAsync(autotune left)");
  check_driver(cuMemcpyHtoDAsync(device_right.get(), right.data(),
                                 sizeof(right), stream.get()),
               "cuMemcpyHtoDAsync(autotune right)");
  const std::array<flagdnnBinding_t, 3> bindings = {{
      {1, device_left.opaque()},
      {2, device_right.opaque()},
      {3, device_output.opaque()},
  }};
  for (int repetition = 0; repetition < 3; ++repetition) {
    check_frontend(graph.execute(handle,
                                 std::span<const flagdnnBinding_t>(bindings),
                                 workspace.opaque(), graph.get_workspace_size(),
                                 reinterpret_cast<void *>(stream.get())),
                   "FlagDNN Iluvatar autotune Add execute");
  }
  check_driver(cuMemcpyDtoHAsync(output.data(), device_output.get(),
                                 sizeof(output), stream.get()),
               "cuMemcpyDtoHAsync(autotune output)");
  check_driver(cuStreamSynchronize(stream.get()),
               "cuStreamSynchronize(autotune output)");
  for (std::size_t index = 0; index < output.size(); ++index) {
    if (std::fabs(output[index] - (left[index] + right[index])) > 1.0e-6F) {
      throw std::runtime_error("autotuned Add result mismatch at element " +
                               std::to_string(index));
    }
  }
}

void test_device_autotune_contract(int argc, char **argv) {
  expect(argc == 4, "usage: flagdnn_test_iluvatar_autotune "
                    "PLUGIN COMPILER_EXECUTABLE COMPILER_ENTRY");
  const std::filesystem::path plugin = std::filesystem::canonical(argv[1]);
  if (setenv("FLAGDNN_BACKEND_PATH", plugin.parent_path().c_str(), 1) != 0 ||
      setenv("FLAGDNN_EXECUTION_ENGINE", "libtriton_jit", 1) != 0) {
    throw std::runtime_error("cannot configure autotune test environment");
  }

  CurrentPrimaryContext current_context;
  TemporaryDirectory cache("flagdnn-iluvatar-autotune-device");
  flagdnn::Handle handle("iluvatar", 0);
  handle.set_compiler(argv[2], argv[3], cache.path().string());

  std::atomic<std::size_t> jit_launches{0};
  triton_jit::set_launch_enter_hook([&](const triton_jit::LaunchMetadata &) {
    jit_launches.fetch_add(1, std::memory_order_relaxed);
  });

  fe::graph::Graph miss_graph;
  build_add(miss_graph, handle);
  const std::size_t miss_launches =
      jit_launches.load(std::memory_order_relaxed);
  expect(miss_launches > 2,
         "autotune cache miss did not prepare the full candidate set");
  const std::filesystem::path selection = find_selection_cache(cache.path());
  const std::string winner = read_file(selection);
  expect_no_temporary_cache_files(cache.path());

  fe::graph::Graph hit_graph;
  build_add(hit_graph, handle);
  const std::size_t hit_launches =
      jit_launches.load(std::memory_order_relaxed) - miss_launches;
  expect(hit_launches == 2,
         "autotune cache hit did not prepare only the cached winner");
  expect(read_file(selection) == winner,
         "autotune cache hit rewrote the winner selection");

  const std::size_t before_execute =
      jit_launches.load(std::memory_order_relaxed);
  execute_and_check(hit_graph, handle);
  expect(jit_launches.load(std::memory_order_relaxed) == before_execute,
         "steady execute re-entered JIT/autotune");
  expect(read_file(selection) == winner,
         "steady execute read/wrote the selection cache");

  write_file(selection, "{}\n");
  const std::size_t before_retune =
      jit_launches.load(std::memory_order_relaxed);
  fe::graph::Graph retuned_graph;
  build_add(retuned_graph, handle);
  const std::size_t retune_launches =
      jit_launches.load(std::memory_order_relaxed) - before_retune;
  expect(retune_launches > hit_launches,
         "corrupt cache did not trigger full candidate retuning");
  expect(read_file(selection) != "{}\n",
         "corrupt cache was not replaced atomically");
  expect_no_temporary_cache_files(cache.path());
  execute_and_check(retuned_graph, handle);

  triton_jit::clear_launch_hooks();
  std::cout << "PASS Iluvatar CoreX event autotune; miss_jit_launches="
            << miss_launches << ", hit_jit_launches=" << hit_launches
            << ", retune_jit_launches=" << retune_launches << '\n';
}

} // namespace

int main(int argc, char **argv) {
  try {
    test_shared_policy_contract();
    test_device_autotune_contract(argc, argv);
    return 0;
  } catch (const std::exception &error) {
    triton_jit::clear_launch_hooks();
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
