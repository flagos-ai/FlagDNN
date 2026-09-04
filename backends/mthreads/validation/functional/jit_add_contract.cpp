/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include <flagdnn/flagdnn.hpp>
#include <flagdnn_frontend.h>

#include <musa_runtime_api.h>
#include <mudnn.h>
#include <triton_jit/triton_kernel.h>

#include <unistd.h>

#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

namespace fe = flagdnn_frontend;

void require(bool condition, std::string message) {
  if (!condition) {
    throw std::runtime_error(std::move(message));
  }
}

void check_musa(musaError_t status, std::string_view operation) {
  if (status == musaSuccess) {
    return;
  }
  const char* name = musaGetErrorName(status);
  const char* description = musaGetErrorString(status);
  throw std::runtime_error(
      std::string(operation) + " failed: " +
      (name == nullptr ? "unknown" : name) + ": " +
      (description == nullptr ? "unknown" : description));
}

void check_mudnn(
    musa::dnn::Status status, std::string_view operation) {
  if (status != musa::dnn::Status::SUCCESS) {
    throw std::runtime_error(
        std::string(operation) + " failed with muDNN status " +
        std::to_string(static_cast<int>(status)));
  }
}

void check_frontend(fe::error_t status, std::string_view operation) {
  if (status.is_bad()) {
    throw std::runtime_error(
        std::string(operation) + " failed: " + status.get_message());
  }
}

class TemporaryCache final {
 public:
  TemporaryCache() {
    std::string pattern =
        (std::filesystem::temp_directory_path() /
         "flagdnn-mthreads-jit-contract-XXXXXX")
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

class Stream final {
 public:
  Stream() {
    check_musa(
        musaStreamCreateWithFlags(&stream_, musaStreamNonBlocking),
        "musaStreamCreateWithFlags");
  }
  ~Stream() {
    if (stream_ != nullptr) {
      static_cast<void>(musaStreamDestroy(stream_));
    }
  }

  Stream(const Stream&) = delete;
  Stream& operator=(const Stream&) = delete;

  [[nodiscard]] musaStream_t get() const noexcept { return stream_; }

 private:
  musaStream_t stream_ = nullptr;
};

class DeviceBuffer final {
 public:
  explicit DeviceBuffer(std::size_t size) : size_(size) {
    require(size_ != 0, "device allocation size must be nonzero");
    check_musa(musaMalloc(&pointer_, size_), "musaMalloc");
  }
  ~DeviceBuffer() {
    if (pointer_ != nullptr) {
      static_cast<void>(musaFree(pointer_));
    }
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  [[nodiscard]] void* get() const noexcept { return pointer_; }
  [[nodiscard]] std::size_t size() const noexcept { return size_; }

  void copy_from(
      const void* source, std::size_t size, musaStream_t stream) {
    require(size <= size_, "host-to-device copy exceeds allocation");
    check_musa(
        musaMemcpyAsync(
            pointer_, source, size, musaMemcpyHostToDevice, stream),
        "musaMemcpyAsync(host-to-device)");
  }

  void copy_to(
      void* destination, std::size_t size, musaStream_t stream) const {
    require(size <= size_, "device-to-host copy exceeds allocation");
    check_musa(
        musaMemcpyAsync(
            destination, pointer_, size, musaMemcpyDeviceToHost, stream),
        "musaMemcpyAsync(device-to-host)");
  }

  void clear(musaStream_t stream) {
    check_musa(
        musaMemsetAsync(pointer_, 0, size_, stream),
        "musaMemsetAsync");
  }

 private:
  void* pointer_ = nullptr;
  std::size_t size_ = 0;
};

struct AddCase {
  std::string name;
  std::vector<std::int64_t> dimensions;
  std::vector<std::int64_t> strides;
  bool autotune = false;
};

std::size_t storage_elements(const AddCase& test_case) {
  require(
      test_case.dimensions.size() == test_case.strides.size() &&
          !test_case.dimensions.empty(),
      "test tensor dimensions/strides are invalid");
  std::size_t maximum_offset = 0;
  for (std::size_t index = 0;
       index < test_case.dimensions.size(); ++index) {
    maximum_offset +=
        static_cast<std::size_t>(test_case.dimensions[index] - 1) *
        static_cast<std::size_t>(test_case.strides[index]);
  }
  return maximum_offset + 1;
}

std::vector<std::size_t> logical_offsets(const AddCase& test_case) {
  std::size_t elements = 1;
  for (const std::int64_t dimension : test_case.dimensions) {
    elements *= static_cast<std::size_t>(dimension);
  }
  std::vector<std::size_t> result;
  result.reserve(elements);
  for (std::size_t linear = 0; linear < elements; ++linear) {
    std::size_t remaining = linear;
    std::size_t offset = 0;
    for (std::size_t trailing = 0;
         trailing < test_case.dimensions.size(); ++trailing) {
      const std::size_t axis =
          test_case.dimensions.size() - 1 - trailing;
      const std::size_t dimension =
          static_cast<std::size_t>(test_case.dimensions[axis]);
      const std::size_t coordinate = remaining % dimension;
      remaining /= dimension;
      offset += coordinate *
                static_cast<std::size_t>(test_case.strides[axis]);
    }
    result.push_back(offset);
  }
  return result;
}

fe::graph::Graph::Tensor make_tensor(
    fe::graph::Graph& graph,
    const AddCase& test_case,
    std::string name,
    std::int64_t uid) {
  return graph.tensor(
      fe::graph::Tensor_attributes()
          .set_name(std::move(name))
          .set_uid(uid)
          .set_data_type(fe::DataType_t::FLOAT)
          .set_dim(test_case.dimensions)
          .set_stride(test_case.strides)
          .set_alignment(16));
}

std::size_t selection_cache_count(
    const std::filesystem::path& cache,
    std::filesystem::file_time_type* timestamp = nullptr) {
  std::size_t count = 0;
  for (const auto& entry :
       std::filesystem::recursive_directory_iterator(cache)) {
    if (!entry.is_regular_file() ||
        entry.path().filename() != "stage-0.json" ||
        entry.path().parent_path().filename() != "tuning") {
      continue;
    }
    ++count;
    if (timestamp != nullptr) {
      *timestamp = entry.last_write_time();
    }
    std::ifstream input(entry.path(), std::ios::binary);
    const std::string contents{
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>()};
    require(
        contents.find("\"variant_id\"") != std::string::npos &&
            contents.find("\"device_identity\"") !=
                std::string::npos,
        "mthreads autotune cache content is incomplete");
  }
  return count;
}

struct CaseResult {
  std::size_t build_jit_launches = 0;
  std::filesystem::file_time_type selection_timestamp{};
};

CaseResult run_case(
    flagdnn::Handle& handle,
    const AddCase& test_case,
    const std::filesystem::path& cache,
    std::atomic<std::size_t>& jit_enters,
    std::atomic<std::size_t>& jit_exits,
    bool require_jit_build = true) {
  const std::size_t before_build =
      jit_enters.load(std::memory_order_relaxed);
  fe::graph::Graph graph;
  graph.set_name(test_case.name)
      .set_io_data_type(fe::DataType_t::FLOAT)
      .set_intermediate_data_type(fe::DataType_t::FLOAT)
      .set_compute_data_type(fe::DataType_t::FLOAT)
      .set_autotune(test_case.autotune);
  const auto left = make_tensor(graph, test_case, "left", 100);
  const auto right = make_tensor(graph, test_case, "right", 101);
  auto output = graph.pointwise(
      left,
      right,
      fe::graph::Pointwise_attributes()
          .set_name("add")
          .set_mode(fe::PointwiseMode_t::ADD)
          .set_compute_data_type(fe::DataType_t::FLOAT)
          .set_alpha(1.0));
  output->set_name("output")
      .set_uid(102)
      .set_data_type(fe::DataType_t::FLOAT)
      .set_dim(test_case.dimensions)
      .set_stride(test_case.strides)
      .set_alignment(16)
      .set_output(true);
  check_frontend(
      graph.build(handle, {fe::HeurMode_t::A}),
      "FlagDNN mthreads Add graph build");
  const std::size_t after_build =
      jit_enters.load(std::memory_order_relaxed);
  require(
      jit_exits.load(std::memory_order_relaxed) == after_build,
      "mthreads executable build left an incomplete JIT launch");
  if (require_jit_build) {
    require(
        after_build > before_build,
        "mthreads executable build did not perform JIT launch preparation");
  }
  require(
      graph.get_workspace_size() == 4096,
      "mthreads Graph workspace contract differs");

  const std::size_t storage = storage_elements(test_case);
  const std::size_t bytes = storage * sizeof(float);
  const std::vector<std::size_t> offsets =
      logical_offsets(test_case);
  std::vector<float> host_left(storage, -401.0F);
  std::vector<float> host_right(storage, -402.0F);
  std::vector<float> graph_output(storage, -403.0F);
  std::vector<float> reference_output(storage, -404.0F);
  for (std::size_t logical = 0; logical < offsets.size(); ++logical) {
    host_left[offsets[logical]] =
        static_cast<float>(static_cast<int>(logical % 17) - 8) * 0.5F;
    host_right[offsets[logical]] =
        static_cast<float>(static_cast<int>(logical % 11) - 5) * 0.25F;
  }

  Stream stream;
  DeviceBuffer device_left(bytes);
  DeviceBuffer device_right(bytes);
  DeviceBuffer device_graph_output(bytes);
  DeviceBuffer device_reference_output(bytes);
  DeviceBuffer workspace(
      static_cast<std::size_t>(graph.get_workspace_size()));
  require(
      reinterpret_cast<std::uintptr_t>(workspace.get()) % 256U == 0,
      "MUSA workspace allocation is not 256-byte aligned");
  device_left.copy_from(host_left.data(), bytes, stream.get());
  device_right.copy_from(host_right.data(), bytes, stream.get());
  device_graph_output.clear(stream.get());
  device_reference_output.clear(stream.get());

  const std::array<flagdnnBinding_t, 3> bindings = {{
      {100, device_left.get()},
      {101, device_right.get()},
      {102, device_graph_output.get()},
  }};
  for (unsigned int repetition = 0; repetition < 3; ++repetition) {
    check_frontend(
        graph.execute(
            handle,
            std::span<const flagdnnBinding_t>(bindings),
            workspace.get(),
            static_cast<std::size_t>(graph.get_workspace_size()),
            reinterpret_cast<void*>(stream.get())),
        "FlagDNN mthreads Add graph execute");
  }
  require(
      jit_enters.load(std::memory_order_relaxed) == after_build &&
          jit_exits.load(std::memory_order_relaxed) == after_build,
      "steady mthreads execute re-entered libtriton_jit");

  musa::dnn::Handle reference_handle(0);
  check_mudnn(
      reference_handle.SetStream(stream.get()),
      "muDNN Handle::SetStream");
  musa::dnn::Tensor reference_left;
  musa::dnn::Tensor reference_right;
  musa::dnn::Tensor reference_output_tensor;
  const auto configure_reference_tensor =
      [&](musa::dnn::Tensor& tensor, void* pointer) {
        check_mudnn(tensor.SetAddr(pointer), "muDNN Tensor::SetAddr");
        check_mudnn(
            tensor.SetType(musa::dnn::Tensor::Type::FLOAT),
            "muDNN Tensor::SetType");
        check_mudnn(
            tensor.SetNdInfo(
                static_cast<std::int64_t>(test_case.dimensions.size()),
                test_case.dimensions.data(),
                test_case.strides.data()),
            "muDNN Tensor::SetNdInfo");
      };
  configure_reference_tensor(reference_left, device_left.get());
  configure_reference_tensor(reference_right, device_right.get());
  configure_reference_tensor(
      reference_output_tensor, device_reference_output.get());
  musa::dnn::Binary binary;
  check_mudnn(
      binary.SetMode(musa::dnn::Binary::Mode::ADD),
      "muDNN Binary::SetMode");
  check_mudnn(
      binary.Run(
          reference_handle,
          reference_output_tensor,
          reference_left,
          reference_right),
      "muDNN Binary::Run(ADD)");

  device_graph_output.copy_to(
      graph_output.data(), bytes, stream.get());
  device_reference_output.copy_to(
      reference_output.data(), bytes, stream.get());
  check_musa(
      musaStreamSynchronize(stream.get()),
      "musaStreamSynchronize(mthreads Add outputs)");
  for (std::size_t logical = 0; logical < offsets.size(); ++logical) {
    const std::size_t offset = offsets[logical];
    const float cpu = host_left[offset] + host_right[offset];
    if (std::bit_cast<std::uint32_t>(graph_output[offset]) !=
            std::bit_cast<std::uint32_t>(reference_output[offset]) ||
        std::fabs(graph_output[offset] - cpu) > 1.0e-6F) {
      throw std::runtime_error(
          "mthreads Graph/muDNN Add mismatch for " + test_case.name +
          " at logical element " + std::to_string(logical));
    }
  }

  CaseResult result;
  result.build_jit_launches = after_build - before_build;
  if (test_case.autotune) {
    require(
        selection_cache_count(cache, &result.selection_timestamp) == 1,
        "mthreads autotune build did not publish exactly one selection");
  }
  return result;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 4) {
      throw std::invalid_argument(
          "usage: jit_add_contract "
          "<plugin> <compiler-python> <compiler-entry>");
    }
    const std::filesystem::path plugin =
        std::filesystem::canonical(argv[1]);
    if (setenv(
            "FLAGDNN_BACKEND_PATH",
            plugin.parent_path().c_str(),
            1) != 0 ||
        setenv("FLAGDNN_EXECUTION_ENGINE", "libtriton_jit", 1) != 0) {
      throw std::runtime_error(
          "cannot configure mthreads functional environment");
    }
    check_musa(musaSetDevice(0), "musaSetDevice");

    TemporaryCache cache;
    flagdnn::Handle handle("mthreads", 0);
    require(
        handle.backend_name() == "mthreads" &&
            handle.target_fingerprint() == "musa-mtgpu-cc31-w32",
        "public mthreads Handle identity differs");
    handle.set_compiler(argv[2], argv[3], cache.path().string());

    std::atomic<std::size_t> jit_enters{0};
    std::atomic<std::size_t> jit_exits{0};
    triton_jit::set_launch_enter_hook(
        [&](const triton_jit::LaunchMetadata&) {
          jit_enters.fetch_add(1, std::memory_order_relaxed);
        });
    triton_jit::set_launch_exit_hook(
        [&](const triton_jit::LaunchMetadata&) {
          jit_exits.fetch_add(1, std::memory_order_relaxed);
        });

    const CaseResult dense = run_case(
        handle,
        {"mthreads_dense_add", {2, 3, 5}, {15, 5, 1}, false},
        cache.path(),
        jit_enters,
        jit_exits);
    const AddCase autotune_case{
        "mthreads_strided_add_autotune",
        {2, 3, 5},
        {32, 8, 1},
        true};
    const CaseResult tuned = run_case(
        handle,
        autotune_case,
        cache.path(),
        jit_enters,
        jit_exits);
    require(
        tuned.build_jit_launches >= 8,
        "mthreads autotune miss did not prepare its four candidates");

    const std::size_t before_cache_hit =
        jit_enters.load(std::memory_order_relaxed);
    const CaseResult cached = run_case(
        handle,
        autotune_case,
        cache.path(),
        jit_enters,
        jit_exits,
        false);
    const std::size_t cache_hit_launches =
        jit_enters.load(std::memory_order_relaxed) - before_cache_hit;
    require(
        cache_hit_launches <= 2 &&
            cached.selection_timestamp == tuned.selection_timestamp,
        "mthreads autotune cache hit rebuilt the candidate space or cache");
    triton_jit::clear_launch_hooks();

    std::cout
        << "PASS mthreads Graph Add == muDNN Binary Add; "
        << "dense_build_jit=" << dense.build_jit_launches
        << ";autotune_miss_jit=" << tuned.build_jit_launches
        << ";autotune_hit_jit=" << cache_hit_launches
        << ";steady_execute_jit=0\n";
    return 0;
  } catch (const std::exception& error) {
    triton_jit::clear_launch_hooks();
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
