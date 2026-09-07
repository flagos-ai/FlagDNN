// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "acdnn_reference.hpp"
#include "acdnn_composite_reference.hpp"
#include "backend_pointwise_reference.hpp"
#include "capability.hpp"
#include "pointwise_reference.hpp"
#include "ppu_driver.hpp"
#include "tensor_io.hpp"

#include "common/common.hpp"

#include <acdnn.h>
#include <cuda.h>
#include <flagdnn/flagdnn.hpp>
#include <flagdnn_frontend.h>
#include <triton_jit/backend_config.h>
#include <triton_jit/backends/cuda_backend.h>
#include <triton_jit/triton_kernel.h>

#include <unistd.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <new>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#ifndef BACKEND_CUDA
#error "The THead Graph contract requires CUDA-backend libtriton_jit"
#endif

#ifndef FLAGDNN_THEAD_ACDNN_CAPABILITY_CATALOG
#define FLAGDNN_THEAD_ACDNN_CAPABILITY_CATALOG "capability.json"
#endif

#ifndef FLAGDNN_THEAD_PPU_SDK_VERSION
#define FLAGDNN_THEAD_PPU_SDK_VERSION "unknown"
#endif

static_assert(triton_jit::CudaBackend::WARP_SIZE == 32);
static_assert(
    std::is_same_v<triton_jit::DefaultBackend, triton_jit::CudaBackend>);

namespace flagdnn_thead_allocation_probe {

thread_local bool active = false;
thread_local std::size_t allocations = 0;
thread_local std::size_t deallocations = 0;

void record_allocation() noexcept {
  if (active) {
    ++allocations;
  }
}

void record_deallocation(void *pointer) noexcept {
  if (active && pointer != nullptr) {
    ++deallocations;
  }
}

void *allocate(std::size_t size) {
  record_allocation();
  void *const result = std::malloc(size == 0 ? 1 : size);
  if (result == nullptr) {
    throw std::bad_alloc();
  }
  return result;
}

void *allocate_aligned(std::size_t size, std::size_t alignment) {
  record_allocation();
  void *result = nullptr;
  if (::posix_memalign(&result, alignment, size == 0 ? 1 : size) != 0) {
    throw std::bad_alloc();
  }
  return result;
}

}  // namespace flagdnn_thead_allocation_probe

void *operator new(std::size_t size) {
  return flagdnn_thead_allocation_probe::allocate(size);
}

void *operator new[](std::size_t size) {
  return flagdnn_thead_allocation_probe::allocate(size);
}

void *operator new(std::size_t size, std::align_val_t alignment) {
  return flagdnn_thead_allocation_probe::allocate_aligned(
      size, static_cast<std::size_t>(alignment));
}

void *operator new[](std::size_t size, std::align_val_t alignment) {
  return flagdnn_thead_allocation_probe::allocate_aligned(
      size, static_cast<std::size_t>(alignment));
}

void *operator new(std::size_t size, const std::nothrow_t &) noexcept {
  try {
    return ::operator new(size);
  } catch (...) {
    return nullptr;
  }
}

void *operator new[](std::size_t size, const std::nothrow_t &) noexcept {
  try {
    return ::operator new[](size);
  } catch (...) {
    return nullptr;
  }
}

void *operator new(std::size_t size, std::align_val_t alignment,
                   const std::nothrow_t &) noexcept {
  try {
    return ::operator new(size, alignment);
  } catch (...) {
    return nullptr;
  }
}

void *operator new[](std::size_t size, std::align_val_t alignment,
                     const std::nothrow_t &) noexcept {
  try {
    return ::operator new[](size, alignment);
  } catch (...) {
    return nullptr;
  }
}

void operator delete(void *pointer) noexcept {
  flagdnn_thead_allocation_probe::record_deallocation(pointer);
  std::free(pointer);
}

void operator delete[](void *pointer) noexcept {
  flagdnn_thead_allocation_probe::record_deallocation(pointer);
  std::free(pointer);
}

void operator delete(void *pointer, std::size_t) noexcept {
  ::operator delete(pointer);
}

void operator delete[](void *pointer, std::size_t) noexcept {
  ::operator delete[](pointer);
}

void operator delete(void *pointer, std::align_val_t) noexcept {
  flagdnn_thead_allocation_probe::record_deallocation(pointer);
  std::free(pointer);
}

void operator delete[](void *pointer, std::align_val_t) noexcept {
  flagdnn_thead_allocation_probe::record_deallocation(pointer);
  std::free(pointer);
}

void operator delete(void *pointer, std::size_t,
                     std::align_val_t alignment) noexcept {
  ::operator delete(pointer, alignment);
}

void operator delete[](void *pointer, std::size_t,
                       std::align_val_t alignment) noexcept {
  ::operator delete[](pointer, alignment);
}

void operator delete(void *pointer, const std::nothrow_t &) noexcept {
  ::operator delete(pointer);
}

void operator delete[](void *pointer, const std::nothrow_t &) noexcept {
  ::operator delete[](pointer);
}

void operator delete(void *pointer, std::align_val_t alignment,
                     const std::nothrow_t &) noexcept {
  ::operator delete(pointer, alignment);
}

void operator delete[](void *pointer, std::align_val_t alignment,
                       const std::nothrow_t &) noexcept {
  ::operator delete[](pointer, alignment);
}

namespace {

namespace fe = ::flagdnn_frontend;
namespace tv = ::flagdnn::validation::thead;

constexpr std::size_t kElementCount = 2 * 4 * 8;
constexpr std::size_t kReplayCount = 3;
constexpr unsigned char kDependencyMarker = 0xA5;

struct PointwiseOperation {
  const char *name;
  fe::PointwiseMode_t frontend_mode;
  flagdnnPointwiseMode_t flagdnn_mode;
  const char *acdnn_primitive;
  bool unary;
  bool comparison = false;
};

PointwiseOperation requested_operation(int argc, char **argv) {
  if (argc == 4) {
    return {"add", fe::PointwiseMode_t::ADD, FLAGDNN_POINTWISE_ADD,
            "acdnnOpTensor(ADD)", false};
  }
  if (argc == 6 && std::string_view(argv[4]) == "--operation") {
    const std::string_view operation(argv[5]);
    if (operation == "add") {
      return {"add", fe::PointwiseMode_t::ADD, FLAGDNN_POINTWISE_ADD,
              "acdnnOpTensor(ADD)", false};
    }
    if (operation == "sub") {
      return {"sub", fe::PointwiseMode_t::SUB, FLAGDNN_POINTWISE_SUB,
              "acdnnOpTensor(ADD,alpha_right=-alpha)", false};
    }
    if (operation == "mul") {
      return {"mul", fe::PointwiseMode_t::MUL, FLAGDNN_POINTWISE_MUL,
              "acdnnOpTensor(MUL)", false};
    }
    if (operation == "min") {
      return {"min", fe::PointwiseMode_t::MIN, FLAGDNN_POINTWISE_MIN,
              "acdnnOpTensor(MIN)", false};
    }
    if (operation == "max") {
      return {"max", fe::PointwiseMode_t::MAX, FLAGDNN_POINTWISE_MAX,
              "acdnnOpTensor(MAX)", false};
    }
    if (operation == "scale") {
      return {"scale", fe::PointwiseMode_t::MUL, FLAGDNN_POINTWISE_MUL,
              "acdnnOpTensor(MUL)", false};
    }
    if (operation == "relu") {
      return {"relu", fe::PointwiseMode_t::RELU_FWD,
              FLAGDNN_POINTWISE_RELU_FWD,
              "acdnnActivationForward(RELU,NOT_PROPAGATE_NAN)", true};
    }
    if (operation == "leaky_relu") {
      return {"leaky_relu", fe::PointwiseMode_t::RELU_FWD,
              FLAGDNN_POINTWISE_RELU_FWD,
              "acdnnOpTensor(LeakyReLU-DAG)", true};
    }
    if (operation == "sigmoid") {
      return {"sigmoid", fe::PointwiseMode_t::SIGMOID_FWD,
              FLAGDNN_POINTWISE_SIGMOID_FWD,
              "acdnnActivationForward(SIGMOID,NOT_PROPAGATE_NAN)", true};
    }
    if (operation == "tanh") {
      return {"tanh", fe::PointwiseMode_t::TANH_FWD,
              FLAGDNN_POINTWISE_TANH_FWD,
              "acdnnActivationForward(TANH,NOT_PROPAGATE_NAN)", true};
    }
    if (operation == "elu") {
      return {"elu", fe::PointwiseMode_t::ELU_FWD,
              FLAGDNN_POINTWISE_ELU_FWD,
              "acdnnActivationForward(ELU,NOT_PROPAGATE_NAN,alpha=1)",
              true};
    }
    if (operation == "identity") {
      return {"identity", fe::PointwiseMode_t::IDENTITY,
              FLAGDNN_POINTWISE_IDENTITY,
              "acdnnTransformTensor(alpha=1,beta=0)", true};
    }
    if (operation == "gelu") {
      return {"gelu", fe::PointwiseMode_t::GELU_FWD,
              FLAGDNN_POINTWISE_GELU_FWD,
              "acdnnActivationForward(GELU,NOT_PROPAGATE_NAN)", true};
    }
    if (operation == "sqrt") {
      return {"sqrt", fe::PointwiseMode_t::SQRT,
              FLAGDNN_POINTWISE_SQRT, "acdnnOpTensor(SQRT)", true};
    }
    if (operation == "neg") {
      return {"neg", fe::PointwiseMode_t::NEG,
              FLAGDNN_POINTWISE_NEG,
              "acdnnTransformTensor(alpha=-1,beta=0)", true};
    }
    if (operation == "abs") {
      return {"abs", fe::PointwiseMode_t::ABS,
              FLAGDNN_POINTWISE_ABS,
              "acdnnBackendExecute(POINTWISE_ABS)", true};
    }
    if (operation == "ceil") {
      return {"ceil", fe::PointwiseMode_t::CEIL,
              FLAGDNN_POINTWISE_CEIL,
              "acdnnBackendExecute(POINTWISE_CEIL)", true};
    }
    if (operation == "floor") {
      return {"floor", fe::PointwiseMode_t::FLOOR,
              FLAGDNN_POINTWISE_FLOOR,
              "acdnnBackendExecute(POINTWISE_FLOOR)", true};
    }
    if (operation == "exp") {
      return {"exp", fe::PointwiseMode_t::EXP,
              FLAGDNN_POINTWISE_EXP,
              "acdnnBackendExecute(POINTWISE_EXP)", true};
    }
    if (operation == "log") {
      return {"log", fe::PointwiseMode_t::LOG,
              FLAGDNN_POINTWISE_LOG,
              "acdnnBackendExecute(POINTWISE_LOG)", true};
    }
    if (operation == "cos") {
      return {"cos", fe::PointwiseMode_t::COS,
              FLAGDNN_POINTWISE_COS,
              "acdnnBackendExecute(POINTWISE_COS)", true};
    }
    if (operation == "rsqrt") {
      return {"rsqrt", fe::PointwiseMode_t::RSQRT,
              FLAGDNN_POINTWISE_RSQRT,
              "acdnnBackendExecute(POINTWISE_RSQRT)", true};
    }
    if (operation == "sin") {
      return {"sin", fe::PointwiseMode_t::SIN,
              FLAGDNN_POINTWISE_SIN,
              "acdnnBackendExecute(POINTWISE_SIN)", true};
    }
    if (operation == "tan") {
      return {"tan", fe::PointwiseMode_t::TAN,
              FLAGDNN_POINTWISE_TAN,
              "acdnnBackendExecute(POINTWISE_TAN)", true};
    }
    if (operation == "softplus") {
      return {"softplus", fe::PointwiseMode_t::SOFTPLUS_FWD,
              FLAGDNN_POINTWISE_SOFTPLUS_FWD,
              "acdnnBackendExecute(POINTWISE_SOFTPLUS_FWD,beta=1)", true};
    }
    if (operation == "swish") {
      return {"swish", fe::PointwiseMode_t::SWISH_FWD,
              FLAGDNN_POINTWISE_SWISH_FWD,
              "acdnnBackendExecute(POINTWISE_SWISH_FWD,beta=1.25)", true};
    }
    if (operation == "gelu_approx_tanh") {
      return {"gelu_approx_tanh",
              fe::PointwiseMode_t::GELU_APPROX_TANH_FWD,
              FLAGDNN_POINTWISE_GELU_APPROX_TANH_FWD,
              "acdnnBackendExecute(POINTWISE_GELU_APPROX_TANH_FWD)",
              true};
    }
    if (operation == "div") {
      return {"div", fe::PointwiseMode_t::DIV, FLAGDNN_POINTWISE_DIV,
              "acdnnBackendExecute(POINTWISE_DIV)", false};
    }
    if (operation == "pow") {
      return {"pow", fe::PointwiseMode_t::POW, FLAGDNN_POINTWISE_POW,
              "acdnnBackendExecute(POINTWISE_POW)", false};
    }
    if (operation == "mod") {
      return {"mod", fe::PointwiseMode_t::MOD, FLAGDNN_POINTWISE_MOD,
              "acdnnBackendExecute(POINTWISE_MOD)", false};
    }
    if (operation == "sigmoid_backward") {
      return {"sigmoid_backward", fe::PointwiseMode_t::SIGMOID_BWD,
              FLAGDNN_POINTWISE_SIGMOID_BWD,
              "acdnnBackendExecute(POINTWISE_SIGMOID_BWD)", false};
    }
    if (operation == "reciprocal") {
      return {"reciprocal", fe::PointwiseMode_t::RECIPROCAL,
              FLAGDNN_POINTWISE_RECIPROCAL,
              "acdnnBackendExecute(POINTWISE_DIV,numerator=1)", true};
    }
    if (operation == "add_square") {
      return {"add_square", fe::PointwiseMode_t::ADD,
              FLAGDNN_POINTWISE_NOT_SET,
              "acdnnBackendExecute(POINTWISE_ADD_SQUARE)", false};
    }
    if (operation == "cmp_eq") {
      return {"cmp_eq", fe::PointwiseMode_t::CMP_EQ,
              FLAGDNN_POINTWISE_CMP_EQ,
              "acdnnBackendExecute(POINTWISE_CMP_EQ)", false, true};
    }
    if (operation == "cmp_neq") {
      return {"cmp_neq", fe::PointwiseMode_t::CMP_NEQ,
              FLAGDNN_POINTWISE_CMP_NEQ,
              "acdnnBackendExecute(POINTWISE_CMP_NEQ)", false, true};
    }
    if (operation == "cmp_gt") {
      return {"cmp_gt", fe::PointwiseMode_t::CMP_GT,
              FLAGDNN_POINTWISE_CMP_GT,
              "acdnnBackendExecute(POINTWISE_CMP_GT)", false, true};
    }
    if (operation == "cmp_ge") {
      return {"cmp_ge", fe::PointwiseMode_t::CMP_GE,
              FLAGDNN_POINTWISE_CMP_GE,
              "acdnnBackendExecute(POINTWISE_CMP_GE)", false, true};
    }
    if (operation == "cmp_lt") {
      return {"cmp_lt", fe::PointwiseMode_t::CMP_LT,
              FLAGDNN_POINTWISE_CMP_LT,
              "acdnnBackendExecute(POINTWISE_CMP_LT)", false, true};
    }
    if (operation == "cmp_le") {
      return {"cmp_le", fe::PointwiseMode_t::CMP_LE,
              FLAGDNN_POINTWISE_CMP_LE,
              "acdnnBackendExecute(POINTWISE_CMP_LE)", false, true};
    }
  }
  throw std::invalid_argument(
      "usage: test_graph PLUGIN COMPILER_EXECUTABLE COMPILER_ENTRY "
      "[--operation add|sub|mul|min|max|scale|relu|leaky_relu|sigmoid|tanh|elu|identity|gelu|sqrt|neg|abs|ceil|floor|exp|log|cos|rsqrt|sin|tan|softplus|swish|gelu_approx_tanh|div|pow|mod|sigmoid_backward|reciprocal|add_square|cmp_eq|cmp_neq|cmp_gt|cmp_ge|cmp_lt|cmp_le]");
}

void require(bool condition, std::string_view message) {
  if (!condition) {
    throw std::runtime_error(std::string(message));
  }
}

void check_frontend(const fe::error_t &status, const char *operation) {
  if (status.is_bad()) {
    throw std::runtime_error(std::string(operation) + " failed: " +
                             status.get_message());
  }
}

class AllocationProbe final {
 public:
  struct Activity {
    std::size_t allocations = 0;
    std::size_t deallocations = 0;
  };

  AllocationProbe() {
    if (flagdnn_thead_allocation_probe::active) {
      throw std::logic_error("host allocation probes must not nest");
    }
    flagdnn_thead_allocation_probe::allocations = 0;
    flagdnn_thead_allocation_probe::deallocations = 0;
    flagdnn_thead_allocation_probe::active = true;
    active_ = true;
  }

  ~AllocationProbe() { stop_tracking(); }

  AllocationProbe(const AllocationProbe &) = delete;
  AllocationProbe &operator=(const AllocationProbe &) = delete;

  [[nodiscard]] Activity finish() noexcept {
    const Activity result{flagdnn_thead_allocation_probe::allocations,
                          flagdnn_thead_allocation_probe::deallocations};
    stop_tracking();
    return result;
  }

 private:
  void stop_tracking() noexcept {
    if (active_) {
      flagdnn_thead_allocation_probe::active = false;
      active_ = false;
    }
  }

  bool active_ = false;
};

class TemporaryDirectory final {
 public:
  TemporaryDirectory() {
    std::string pattern =
        (std::filesystem::temp_directory_path() /
         "flagdnn-thead-graph-XXXXXX")
            .string();
    std::vector<char> writable(pattern.begin(), pattern.end());
    writable.push_back('\0');
    char *const created = ::mkdtemp(writable.data());
    if (created == nullptr) {
      throw std::runtime_error("mkdtemp failed for THead Graph contract");
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

struct CacheEntry {
  std::string relative_path;
  std::filesystem::file_type type = std::filesystem::file_type::none;
  std::uintmax_t size = 0;
  std::filesystem::file_time_type write_time{};
  std::string link_target;
  std::vector<char> contents;

  [[nodiscard]] bool operator==(const CacheEntry &) const = default;
};

std::vector<char> read_file(const std::filesystem::path &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("cannot read Graph cache entry " +
                             path.string());
  }
  return {std::istreambuf_iterator<char>(input),
          std::istreambuf_iterator<char>()};
}

std::vector<CacheEntry> snapshot(const std::filesystem::path &root) {
  std::vector<CacheEntry> result;
  if (!std::filesystem::exists(root)) {
    return result;
  }
  for (const auto &entry :
       std::filesystem::recursive_directory_iterator(root)) {
    CacheEntry value;
    value.relative_path =
        entry.path().lexically_relative(root).generic_string();
    value.type = entry.symlink_status().type();
    value.write_time = entry.last_write_time();
    if (entry.is_regular_file()) {
      value.size = entry.file_size();
      value.contents = read_file(entry.path());
    } else if (entry.is_symlink()) {
      value.link_target =
          std::filesystem::read_symlink(entry.path()).generic_string();
    }
    result.push_back(std::move(value));
  }
  std::sort(result.begin(), result.end(),
            [](const CacheEntry &left, const CacheEntry &right) {
              return left.relative_path < right.relative_path;
            });
  return result;
}

class DriverEvent final {
 public:
  DriverEvent() {
    tv::check_driver(cuEventCreate(&event_, CU_EVENT_DISABLE_TIMING),
                     "cuEventCreate(Graph dependency)");
    require(event_ != nullptr,
            "PPU driver returned a null Graph dependency event");
  }

  ~DriverEvent() {
    if (event_ != nullptr) {
      (void)cuEventDestroy(event_);
    }
  }

  DriverEvent(const DriverEvent &) = delete;
  DriverEvent &operator=(const DriverEvent &) = delete;

  [[nodiscard]] CUevent get() const noexcept { return event_; }

 private:
  CUevent event_ = nullptr;
};

struct GraphNodeSummary {
  std::size_t total = 0;
  std::size_t kernels = 0;
  std::size_t memsets = 0;
  std::size_t event_records = 0;
  std::size_t event_waits = 0;
  std::size_t empty = 0;
};

GraphNodeSummary inspect_graph(CUgraph graph) {
  GraphNodeSummary result;
  tv::check_driver(cuGraphGetNodes(graph, nullptr, &result.total),
                   "cuGraphGetNodes(THead runtime Graph count)");
  require(result.total != 0, "THead runtime capture produced an empty graph");
  std::vector<CUgraphNode> nodes(result.total);
  std::size_t count = result.total;
  tv::check_driver(cuGraphGetNodes(graph, nodes.data(), &count),
                   "cuGraphGetNodes(THead runtime Graph)");
  require(count == result.total,
          "THead runtime Graph node count changed while inspecting it");
  for (CUgraphNode node : nodes) {
    CUgraphNodeType type = CU_GRAPH_NODE_TYPE_EMPTY;
    tv::check_driver(cuGraphNodeGetType(node, &type),
                     "cuGraphNodeGetType(THead runtime Graph)");
    switch (type) {
      case CU_GRAPH_NODE_TYPE_KERNEL:
        ++result.kernels;
        break;
      case CU_GRAPH_NODE_TYPE_MEMSET:
        ++result.memsets;
        break;
      case CU_GRAPH_NODE_TYPE_EVENT_RECORD:
        ++result.event_records;
        break;
      case CU_GRAPH_NODE_TYPE_WAIT_EVENT:
        ++result.event_waits;
        break;
      case CU_GRAPH_NODE_TYPE_EMPTY:
        ++result.empty;
        break;
      default:
        throw std::runtime_error(
            "THead runtime capture contains a forbidden node type");
    }
  }
  return result;
}

template <typename Function>
CUgraph capture_graph(CUstream stream, Function &&capture_body) {
  tv::check_driver(
      cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_RELAXED),
      "cuStreamBeginCapture(THead runtime Graph)");
  bool active = true;
  CUgraph graph = nullptr;
  try {
    std::forward<Function>(capture_body)();
    const CUresult end_result = cuStreamEndCapture(stream, &graph);
    active = false;
    tv::check_driver(end_result, "cuStreamEndCapture(THead runtime Graph)");
    require(graph != nullptr,
            "PPU driver returned a null THead runtime Graph");
    return graph;
  } catch (...) {
    if (active) {
      CUgraph abandoned = nullptr;
      if (cuStreamEndCapture(stream, &abandoned) == CUDA_SUCCESS &&
          abandoned != nullptr) {
        (void)cuGraphDestroy(abandoned);
      }
    } else if (graph != nullptr) {
      (void)cuGraphDestroy(graph);
    }
    throw;
  }
}

class ExecutableGraph final {
 public:
  explicit ExecutableGraph(CUgraph source) {
    require(source != nullptr,
            "cannot instantiate a null THead runtime Graph");
    try {
      summary_ = inspect_graph(source);
      tv::check_driver(cuGraphInstantiate(&executable_, source, 0),
                       "cuGraphInstantiate(THead runtime Graph)");
      const CUresult destroy_result = cuGraphDestroy(source);
      if (destroy_result == CUDA_SUCCESS) {
        source = nullptr;
        source_destroyed_ = true;
      }
      tv::check_driver(destroy_result,
                       "cuGraphDestroy(THead runtime Graph source)");
    } catch (...) {
      if (executable_ != nullptr) {
        (void)cuGraphExecDestroy(executable_);
        executable_ = nullptr;
      }
      if (source != nullptr) {
        (void)cuGraphDestroy(source);
      }
      throw;
    }
  }

  ~ExecutableGraph() {
    if (executable_ != nullptr) {
      (void)cuGraphExecDestroy(executable_);
    }
  }

  ExecutableGraph(const ExecutableGraph &) = delete;
  ExecutableGraph &operator=(const ExecutableGraph &) = delete;

  void launch(CUstream stream) const {
    require(executable_ != nullptr,
            "cannot launch a destroyed THead executable Graph");
    require(source_destroyed_,
            "THead source Graph was not destroyed before replay");
    tv::check_driver(cuGraphLaunch(executable_, stream),
                     "cuGraphLaunch(THead runtime Graph)");
  }

  void destroy_checked() {
    require(executable_ != nullptr,
            "THead executable Graph was already destroyed");
    const CUresult result = cuGraphExecDestroy(executable_);
    if (result == CUDA_SUCCESS) {
      executable_ = nullptr;
    }
    tv::check_driver(result, "cuGraphExecDestroy(THead runtime Graph)");
  }

  [[nodiscard]] const GraphNodeSummary &summary() const noexcept {
    return summary_;
  }

  [[nodiscard]] bool source_destroyed() const noexcept {
    return source_destroyed_;
  }

 private:
  CUgraphExec executable_ = nullptr;
  GraphNodeSummary summary_;
  bool source_destroyed_ = false;
};

tv::CapabilityRecord graph_reference_capability(
    const PointwiseOperation &operation) {
  tv::CapabilityConstraints constraints;
  constraints.dtypes = {"fp32"};
  constraints.compute_type = operation.comparison ? "bool" : "fp32";
  constraints.rank = {1, 8};
  constraints.shape = {"same_shape", "positive_extents"};
  constraints.layouts = {"contiguous"};
  constraints.stride_policy = "dense_contiguous";
  constraints.broadcast = "none";
  if (operation.unary) {
    constraints.attributes = {
        {"reference_nan_policy", {"not_propagate_finite_inputs_only"}},
        {"autotune", {"false"}},
    };
    if (operation.flagdnn_mode == FLAGDNN_POINTWISE_RELU_FWD) {
      constraints.attributes.insert({"relu_lower_clip", {"0"}});
      constraints.attributes.insert({"relu_upper_clip", {"0"}});
      constraints.attributes.insert({"relu_upper_clip_set", {"false"}});
      constraints.attributes.insert(
          {"relu_lower_clip_slope",
           {std::string_view(operation.name) == "leaky_relu" ? "0.2"
                                                               : "0"}});
    } else if (operation.flagdnn_mode == FLAGDNN_POINTWISE_ELU_FWD) {
      constraints.attributes.insert({"elu_alpha", {"1"}});
    } else if (operation.flagdnn_mode ==
               FLAGDNN_POINTWISE_SOFTPLUS_FWD) {
      constraints.attributes.insert({"softplus_beta", {"1"}});
    } else if (operation.flagdnn_mode == FLAGDNN_POINTWISE_SWISH_FWD) {
      constraints.attributes.insert({"swish_beta", {"1.25"}});
    }
  } else {
    constraints.attributes = {{"alpha", {"1"}},
                              {"autotune", {"false"}}};
  }
  const bool leaky_relu = std::string_view(operation.name) == "leaky_relu";
  return {
      .status = tv::CapabilityStatus::kProbeRequired,
      .path = (operation.flagdnn_mode == FLAGDNN_POINTWISE_ABS ||
               operation.flagdnn_mode == FLAGDNN_POINTWISE_CEIL ||
               operation.flagdnn_mode == FLAGDNN_POINTWISE_FLOOR ||
               operation.flagdnn_mode == FLAGDNN_POINTWISE_EXP ||
               operation.flagdnn_mode == FLAGDNN_POINTWISE_LOG ||
               operation.flagdnn_mode == FLAGDNN_POINTWISE_COS ||
               operation.flagdnn_mode == FLAGDNN_POINTWISE_RSQRT ||
               operation.flagdnn_mode == FLAGDNN_POINTWISE_SIN ||
               operation.flagdnn_mode == FLAGDNN_POINTWISE_TAN ||
               operation.flagdnn_mode == FLAGDNN_POINTWISE_SOFTPLUS_FWD ||
               operation.flagdnn_mode == FLAGDNN_POINTWISE_SWISH_FWD ||
               operation.flagdnn_mode ==
                   FLAGDNN_POINTWISE_GELU_APPROX_TANH_FWD ||
               operation.flagdnn_mode == FLAGDNN_POINTWISE_DIV ||
               operation.flagdnn_mode == FLAGDNN_POINTWISE_POW ||
               operation.flagdnn_mode == FLAGDNN_POINTWISE_MOD ||
               operation.flagdnn_mode == FLAGDNN_POINTWISE_SIGMOID_BWD ||
               operation.flagdnn_mode == FLAGDNN_POINTWISE_RECIPROCAL ||
               leaky_relu || operation.comparison)
                  ? tv::ReferencePath::kBackendDescriptor
                  : tv::ReferencePath::kStablePrimitive,
      .reference_plan = leaky_relu
          ? std::vector<std::string>{
                "acdnnBackendExecute(POINTWISE_IDENTITY_FWD,leaky-input-fp32)",
                "acdnnTransformTensor(alpha=-1,beta=0)",
                "acdnnActivationForward(RELU,NOT_PROPAGATE_NAN,positive)",
                "acdnnActivationForward(RELU,NOT_PROPAGATE_NAN,negative)",
                "acdnnOpTensor(ADD,alpha_right=-0.2)",
                "acdnnBackendExecute(POINTWISE_IDENTITY_FWD,leaky-output-data-type)"}
          : std::string_view(operation.name) == "add_square"
              ? std::vector<std::string>{"acdnnOpTensor(MUL)",
                                         "acdnnOpTensor(ADD)"}
              : std::vector<std::string>{operation.acdnn_primitive},
      .constraints = std::move(constraints),
      .reason_code = "real_device_qualification_pending",
      .detail = std::string("runtime Graph capture qualification against "
                            "exact acDNN ") +
                operation.name,
  };
}

flagdnn::testing::TestTensor test_tensor(
    std::int64_t uid,
    flagdnnDataType_t data_type = FLAGDNN_DATA_FLOAT32) {
  return {
      .uid = uid,
      .data_type = data_type,
      .dimensions = {2, 4, 8},
      .strides = {32, 8, 1},
      .binding_byte_offset = 0,
  };
}

std::unique_ptr<fe::graph::Graph> build_pointwise_graph(
    flagdnn::Handle &handle, const PointwiseOperation &operation) {
  auto graph = std::make_unique<fe::graph::Graph>();
  graph->set_name(std::string("thead_runtime_graph_capture_") +
                  operation.name)
      .set_io_data_type(fe::DataType_t::FLOAT)
      .set_intermediate_data_type(fe::DataType_t::FLOAT)
      .set_compute_data_type(fe::DataType_t::FLOAT)
      .set_autotune(false);
  const auto left = graph->tensor(fe::graph::Tensor_attributes()
                                      .set_name("left")
                                      .set_uid(1)
                                      .set_data_type(fe::DataType_t::FLOAT)
                                      .set_dim({2, 4, 8})
                                      .set_stride({32, 8, 1}));
  fe::graph::Pointwise_attributes attributes;
  attributes.set_name(operation.name)
      .set_mode(operation.frontend_mode)
      .set_compute_data_type(operation.comparison
                                 ? fe::DataType_t::BOOLEAN
                                 : fe::DataType_t::FLOAT)
      .set_alpha(1.0);
  if (operation.flagdnn_mode == FLAGDNN_POINTWISE_ELU_FWD) {
    attributes.set_elu_alpha(1.0F);
  } else if (std::string_view(operation.name) == "leaky_relu") {
    attributes.set_relu_lower_clip_slope(0.2F);
  } else if (operation.flagdnn_mode ==
             FLAGDNN_POINTWISE_SOFTPLUS_FWD) {
    attributes.set_softplus_beta(1.0F);
  } else if (operation.flagdnn_mode == FLAGDNN_POINTWISE_SWISH_FWD) {
    attributes.set_swish_beta(1.25F);
  }
  fe::graph::Graph::Tensor output;
  if (std::string_view(operation.name) == "add_square") {
    const auto right = graph->tensor(fe::graph::Tensor_attributes()
                                         .set_name("right")
                                         .set_uid(2)
                                         .set_data_type(fe::DataType_t::FLOAT)
                                         .set_dim({2, 4, 8})
                                         .set_stride({32, 8, 1}));
    const auto square = graph->pointwise(
        right, right,
        fe::graph::Pointwise_attributes()
            .set_name("square")
            .set_mode(fe::PointwiseMode_t::MUL)
            .set_compute_data_type(fe::DataType_t::FLOAT));
    square->set_name("square")
        .set_uid(4)
        .set_data_type(fe::DataType_t::FLOAT)
        .set_dim({2, 4, 8})
        .set_stride({32, 8, 1})
        .set_is_virtual(true);
    output = graph->pointwise(
        left, square,
        fe::graph::Pointwise_attributes()
            .set_name("add_square")
            .set_mode(fe::PointwiseMode_t::ADD)
            .set_compute_data_type(fe::DataType_t::FLOAT));
  } else if (operation.unary) {
    output = graph->pointwise(left, attributes);
  } else {
    const auto right = graph->tensor(fe::graph::Tensor_attributes()
                                         .set_name("right")
                                         .set_uid(2)
                                         .set_data_type(fe::DataType_t::FLOAT)
                                         .set_dim({2, 4, 8})
                                         .set_stride({32, 8, 1}));
    output = graph->pointwise(left, right, attributes);
  }
  output->set_name("output")
      .set_uid(3)
      .set_data_type(operation.comparison
                         ? fe::DataType_t::BOOLEAN
                         : fe::DataType_t::FLOAT)
      .set_dim({2, 4, 8})
      .set_stride({32, 8, 1})
      .set_output(true);
  check_frontend(graph->build(handle, {fe::HeurMode_t::A}),
                 "FlagDNN THead Graph build");
  require(graph->get_workspace_size() == 0,
          "qualified THead Graph has an unexpected workspace size");
  return graph;
}

std::array<float, kElementCount> make_input(
    std::size_t input_index, flagdnnPointwiseMode_t mode) {
  std::array<float, kElementCount> result{};
  for (std::size_t index = 0; index < result.size(); ++index) {
    const int centered =
        static_cast<int>((index * 17 + input_index * 11) % 41) - 20;
    const bool positive_only = mode == FLAGDNN_POINTWISE_SQRT ||
                               mode == FLAGDNN_POINTWISE_LOG ||
                               mode == FLAGDNN_POINTWISE_RSQRT ||
                               mode == FLAGDNN_POINTWISE_DIV ||
                               mode == FLAGDNN_POINTWISE_POW ||
                               mode == FLAGDNN_POINTWISE_MOD ||
                               mode == FLAGDNN_POINTWISE_RECIPROCAL;
    result[index] = positive_only
                        ? static_cast<float>((index * 17 +
                                              input_index * 11) % 41 + 1) /
                              static_cast<float>(13 + input_index)
                    : mode == FLAGDNN_POINTWISE_TAN
                        ? static_cast<float>(centered) / 32.0F
                        : static_cast<float>(centered) /
                              static_cast<float>(13 + input_index);
  }
  return result;
}

std::array<float, kElementCount> read_output(const tv::DeviceBuffer &buffer,
                                             CUstream stream,
                                             bool comparison) {
  std::array<float, kElementCount> result{};
  if (comparison) {
    std::array<std::uint8_t, kElementCount> bytes{};
    tv::copy_from_device_async(std::span<std::uint8_t>(bytes), buffer, 0,
                               stream);
    tv::check_driver(cuStreamSynchronize(stream),
                     "cuStreamSynchronize(read THead Graph bool output)");
    std::transform(bytes.begin(), bytes.end(), result.begin(),
                   [](std::uint8_t value) {
                     return value == 0U ? 0.0F : 1.0F;
                   });
    return result;
  }
  tv::copy_from_device_async(std::span<float>(result), buffer, 0, stream);
  tv::check_driver(cuStreamSynchronize(stream),
                   "cuStreamSynchronize(read THead Graph output)");
  return result;
}

double compare_outputs(std::span<const float> actual,
                       std::span<const float> expected,
                       std::string_view stage,
                       flagdnnPointwiseMode_t mode) {
  require(actual.size() == expected.size(),
          "THead Graph and acDNN output sizes differ");
  double maximum_absolute = 0.0;
  for (std::size_t index = 0; index < actual.size(); ++index) {
    const double difference =
        std::abs(static_cast<double>(actual[index]) - expected[index]);
    const double relative =
        difference /
        std::max({std::abs(static_cast<double>(actual[index])),
                  std::abs(static_cast<double>(expected[index])), 1.0e-30});
    const bool exact = mode == FLAGDNN_POINTWISE_ADD ||
                       mode == FLAGDNN_POINTWISE_SUB ||
                       mode == FLAGDNN_POINTWISE_MUL ||
                       mode == FLAGDNN_POINTWISE_MIN ||
                       mode == FLAGDNN_POINTWISE_MAX ||
                       mode == FLAGDNN_POINTWISE_IDENTITY ||
                       mode == FLAGDNN_POINTWISE_NEG ||
                       mode == FLAGDNN_POINTWISE_ABS ||
                       mode == FLAGDNN_POINTWISE_CEIL ||
                       mode == FLAGDNN_POINTWISE_FLOOR ||
                       mode == FLAGDNN_POINTWISE_CMP_EQ ||
                       mode == FLAGDNN_POINTWISE_CMP_NEQ ||
                       mode == FLAGDNN_POINTWISE_CMP_GT ||
                       mode == FLAGDNN_POINTWISE_CMP_GE ||
                       mode == FLAGDNN_POINTWISE_CMP_LT ||
                       mode == FLAGDNN_POINTWISE_CMP_LE;
    if (!std::isfinite(difference) ||
        (difference > (exact ? 0.0 : 2.0e-5) &&
         relative > (exact ? 0.0 : 1.0e-5))) {
      throw std::runtime_error(std::string(stage) +
                               " differs from acDNN at element " +
                               std::to_string(index));
    }
    maximum_absolute = std::max(maximum_absolute, difference);
  }
  return maximum_absolute;
}

void require_not_capturing(CUstream stream, std::string_view description) {
  CUstreamCaptureStatus status = CU_STREAM_CAPTURE_STATUS_ACTIVE;
  tv::check_driver(cuStreamIsCapturing(stream, &status),
                   "cuStreamIsCapturing(after THead capture)");
  require(status == CU_STREAM_CAPTURE_STATUS_NONE, description);
}

}  // namespace

int main(int argc, char **argv) {
  try {
    const PointwiseOperation operation = requested_operation(argc, argv);
    const std::filesystem::path plugin = std::filesystem::canonical(argv[1]);
    TemporaryDirectory temporary;
    const std::filesystem::path flagdnn_cache = temporary.path() / "flagdnn";
    const std::filesystem::path triton_cache = temporary.path() / "triton";
    require(setenv("FLAGDNN_BACKEND_PATH", plugin.parent_path().c_str(), 1) ==
                    0 &&
                setenv("FLAGDNN_EXECUTION_ENGINE", "libtriton_jit", 1) == 0 &&
                setenv("TRITON_CACHE_DIR", triton_cache.c_str(), 1) == 0 &&
                setenv("TRITON_JIT_BACKEND", "CUDA", 1) == 0 &&
                setenv("PYTHONDONTWRITEBYTECODE", "1", 1) == 0,
            "cannot configure the THead Graph test environment");

    const tv::CapabilityCatalog catalog = tv::CapabilityCatalog::load(
        FLAGDNN_THEAD_ACDNN_CAPABILITY_CATALOG);
    catalog.validate_versions(FLAGDNN_THEAD_PPU_SDK_VERSION, ACDNN_VERSION,
                              static_cast<std::int64_t>(acdnnGetVersion()));

    tv::check_driver(cuInit(0), "cuInit");
    CUdevice device = 0;
    tv::check_driver(cuDeviceGet(&device, 0), "cuDeviceGet");
    tv::PrimaryContext primary(device);
    tv::ScopedCurrentContext current(primary.get());
    tv::DeviceStream capture_stream;
    tv::DeviceStream dependency_stream;
    require(capture_stream.get() != nullptr &&
                dependency_stream.get() != nullptr,
            "THead Graph requires two non-default streams");
    require(capture_stream.get() != dependency_stream.get(),
            "THead Graph dependency streams must be distinct");
    require(capture_stream.context() == primary.get() &&
                dependency_stream.context() == primary.get(),
            "THead Graph streams do not belong to the retained context");

    std::size_t jit_launches = 0;
    triton_jit::set_launch_enter_hook(
        [&](const triton_jit::LaunchMetadata &metadata) {
          const std::string_view expected_kernel =
              std::string_view(operation.name) == "add_square"
                  ? "add_square_contiguous_kernel"
                  : operation.unary ? "unary_pointwise_contiguous_kernel"
                                    : "binary_contiguous_kernel";
          require(metadata.kernel_name == expected_kernel,
                  "THead Graph prepared an unexpected Triton kernel");
          require(metadata.stream != nullptr,
                  "THead Graph preparation used the default stream");
          ++jit_launches;
        });

    flagdnn::Handle handle("thead", 0);
    require(handle.backend_name() == "thead",
            "THead Graph loaded the wrong backend");
    handle.set_compiler(argv[2], argv[3], flagdnn_cache.string());
    auto graph = build_pointwise_graph(handle, operation);
    require(jit_launches == 1,
            "THead Graph build did not prepare exactly one JIT launch");

    const std::array<float, kElementCount> host_left =
        make_input(0, operation.flagdnn_mode);
    const std::array<float, kElementCount> host_right =
        make_input(1, operation.flagdnn_mode);
    tv::DeviceBuffer device_left(sizeof(host_left));
    tv::DeviceBuffer device_right(sizeof(host_right));
    tv::DeviceBuffer device_output(sizeof(host_left));
    tv::DeviceBuffer reference_output(sizeof(host_left));
    tv::DeviceBuffer workspace(graph->get_workspace_size());
    tv::DeviceBuffer dependency_bytes(128);
    tv::copy_to_device_async(
        device_left,
        std::span<const float>(host_left.data(), host_left.size()), 0,
        capture_stream.get());
    if (!operation.unary) {
      tv::copy_to_device_async(
          device_right,
          std::span<const float>(host_right.data(), host_right.size()), 0,
          capture_stream.get());
    }

    std::vector<flagdnnBinding_t> bindings = {{1, device_left.data()}};
    std::vector<flagdnnBinding_t> reference_bindings = {
        {1, device_left.data()}};
    if (!operation.unary) {
      bindings.push_back({2, device_right.data()});
      reference_bindings.push_back({2, device_right.data()});
    }
    bindings.push_back({3, device_output.data()});
    reference_bindings.push_back({3, reference_output.data()});

    const flagdnn::testing::TestTensor left = test_tensor(1);
    const flagdnn::testing::TestTensor right = test_tensor(2);
    const flagdnn::testing::TestTensor output = test_tensor(
        3, operation.comparison ? FLAGDNN_DATA_BOOLEAN
                                : FLAGDNN_DATA_FLOAT32);
    std::vector<flagdnn::testing::TestTensor> reference_inputs = {left};
    if (!operation.unary) {
      reference_inputs.push_back(right);
    }
    flagdnnPointwiseAttributes_t reference_attributes =
        FLAGDNN_POINTWISE_ATTRIBUTES_INITIALIZER;
    if (operation.flagdnn_mode == FLAGDNN_POINTWISE_SOFTPLUS_FWD) {
      reference_attributes.flags =
          FLAGDNN_POINTWISE_ATTRIBUTE_SOFTPLUS_BETA;
      reference_attributes.softplus_beta = 1.0;
    } else if (std::string_view(operation.name) == "leaky_relu") {
      reference_attributes.flags =
          FLAGDNN_POINTWISE_ATTRIBUTE_RELU_LOWER_CLIP_SLOPE;
      reference_attributes.relu_lower_clip_slope = 0.2;
    } else if (operation.flagdnn_mode == FLAGDNN_POINTWISE_SWISH_FWD) {
      reference_attributes.flags = FLAGDNN_POINTWISE_ATTRIBUTE_SWISH_BETA;
      reference_attributes.swish_beta = 1.25;
    }
    std::unique_ptr<flagdnn::testing::TestExecutable> reference;
    if (std::string_view(operation.name) == "add_square") {
      reference = tv::make_acdnn_add_square_reference(
          reference_inputs.at(0), reference_inputs.at(1), output,
          graph_reference_capability(operation));
    } else {
      reference = tv::make_acdnn_pointwise_reference(
          {.mode = operation.flagdnn_mode,
           .inputs = std::move(reference_inputs),
           .output = output,
           .alpha = 1.0,
           .attributes = reference_attributes},
          graph_reference_capability(operation));
    }
    tv::DeviceBuffer reference_workspace(reference->workspace_size());
    reference->prepare(reference_bindings, capture_stream.opaque());
    reference->execute(reference_bindings, reference_workspace.data(),
                       reference_workspace.size(),
                       capture_stream.opaque());

    const auto execute_without_host_activity = [&] {
      AllocationProbe probe;
      const fe::error_t status = graph->execute(
          handle, std::span<const flagdnnBinding_t>(bindings),
          workspace.data(), workspace.size(), capture_stream.opaque());
      const AllocationProbe::Activity activity = probe.finish();
      check_frontend(status, "FlagDNN execute in THead runtime Graph");
      require(activity.allocations == 0,
              "THead steady execute allocated host memory");
      require(activity.deallocations == 0,
              "THead steady execute deallocated host memory");
    };

    execute_without_host_activity();
    tv::check_driver(cuStreamSynchronize(capture_stream.get()),
                     "cuStreamSynchronize(THead Graph preparation)");
    const std::array<float, kElementCount> expected =
        read_output(reference_output, capture_stream.get(),
                    operation.comparison);
    (void)compare_outputs(read_output(device_output, capture_stream.get(),
                                      operation.comparison),
                          expected, "direct execution before capture",
                          operation.flagdnn_mode);

    tv::check_driver(cuMemsetD8Async(device_output.address(), 0,
                                    device_output.size(),
                                    capture_stream.get()),
                     "cuMemsetD8Async(THead basic Graph output)");
    tv::check_driver(cuMemsetD8Async(dependency_bytes.address(), 0,
                                    dependency_bytes.size(),
                                    capture_stream.get()),
                     "cuMemsetD8Async(THead Graph dependency bytes)");
    tv::check_driver(cuStreamSynchronize(capture_stream.get()),
                     "cuStreamSynchronize(before THead Graph capture)");

    const std::vector<CacheEntry> cache_before = snapshot(temporary.path());
    const std::size_t launches_before_capture = jit_launches;

    ExecutableGraph basic(capture_graph(
        capture_stream.get(), execute_without_host_activity));
    require(basic.source_destroyed(),
            "basic THead source Graph survived instantiation");
    require(basic.summary().total == 1 && basic.summary().kernels == 1,
            "basic THead pointwise capture must contain exactly one kernel");

    DriverEvent fork;
    DriverEvent join;
    ExecutableGraph cross_stream(capture_graph(capture_stream.get(), [&] {
      execute_without_host_activity();
      tv::check_driver(cuEventRecord(fork.get(), capture_stream.get()),
                       "cuEventRecord(THead Graph fork)");
      tv::check_driver(
          cuStreamWaitEvent(dependency_stream.get(), fork.get(), 0),
          "cuStreamWaitEvent(THead Graph fork)");
      CUstreamCaptureStatus dependency_status =
          CU_STREAM_CAPTURE_STATUS_NONE;
      tv::check_driver(
          cuStreamIsCapturing(dependency_stream.get(), &dependency_status),
          "cuStreamIsCapturing(THead dependency stream)");
      require(dependency_status == CU_STREAM_CAPTURE_STATUS_ACTIVE,
              "event dependency did not join the secondary stream to capture");
      tv::check_driver(
          cuMemsetD8Async(dependency_bytes.address(), kDependencyMarker,
                          dependency_bytes.size(), dependency_stream.get()),
          "cuMemsetD8Async(THead captured dependency)");
      tv::check_driver(cuEventRecord(join.get(), dependency_stream.get()),
                       "cuEventRecord(THead Graph join)");
      tv::check_driver(cuStreamWaitEvent(capture_stream.get(), join.get(), 0),
                       "cuStreamWaitEvent(THead Graph join)");
    }));
    require(cross_stream.source_destroyed(),
            "cross-stream THead source Graph survived instantiation");
    require(cross_stream.summary().kernels == 1 &&
                cross_stream.summary().memsets == 1,
            "cross-stream THead Graph lost its kernel or dependency work");
    require_not_capturing(capture_stream.get(),
                          "origin stream remained in capture mode");
    require_not_capturing(dependency_stream.get(),
                          "dependency stream remained in capture mode");
    require(jit_launches == launches_before_capture,
            "THead Graph capture re-entered libtriton_jit");

    graph.reset();
    for (std::size_t replay = 0; replay < kReplayCount; ++replay) {
      basic.launch(capture_stream.get());
    }
    tv::DeviceEvent repeated_replay_complete;
    repeated_replay_complete.record(capture_stream.get());
    repeated_replay_complete.synchronize();
    double maximum_absolute = compare_outputs(
        read_output(device_output, capture_stream.get(), operation.comparison),
        expected,
        "repeated basic Graph replay", operation.flagdnn_mode);

    tv::check_driver(cuMemsetD8Async(device_output.address(), 0,
                                    device_output.size(),
                                    capture_stream.get()),
                     "cuMemsetD8Async(THead cross-stream output)");
    tv::check_driver(cuMemsetD8Async(dependency_bytes.address(), 0,
                                    dependency_bytes.size(),
                                    capture_stream.get()),
                     "cuMemsetD8Async(THead cross-stream dependency)");
    tv::check_driver(cuStreamSynchronize(capture_stream.get()),
                     "cuStreamSynchronize(before cross-stream replay)");

    cross_stream.launch(capture_stream.get());
    tv::DeviceEvent asynchronous_launch_complete;
    asynchronous_launch_complete.record(capture_stream.get());
    cross_stream.destroy_checked();
    asynchronous_launch_complete.synchronize();
    maximum_absolute =
        std::max(maximum_absolute,
                 compare_outputs(read_output(device_output,
                                             capture_stream.get(),
                                             operation.comparison),
                                 expected, "cross-stream Graph replay",
                                 operation.flagdnn_mode));
    std::array<unsigned char, 128> dependency_result{};
    tv::copy_from_device_async(std::span<unsigned char>(dependency_result),
                               dependency_bytes, 0, capture_stream.get());
    tv::check_driver(cuStreamSynchronize(capture_stream.get()),
                     "cuStreamSynchronize(read Graph dependency)");
    require(std::all_of(dependency_result.begin(), dependency_result.end(),
                        [](unsigned char value) {
                          return value == kDependencyMarker;
                        }),
            "cross-stream event dependency did not replay secondary work");

    basic.destroy_checked();
    require(jit_launches == launches_before_capture,
            "THead Graph replay re-entered libtriton_jit");
    require(snapshot(temporary.path()) == cache_before,
            "THead Graph capture/replay changed compiler or JIT cache files");
    triton_jit::clear_launch_hooks();

    std::cout << "FLAGDNN_THEAD_GRAPH: PASS"
              << " operation=" << operation.name
              << " captures=2"
              << " replays=" << (kReplayCount + 1)
              << " basic_nodes=" << basic.summary().total
              << " cross_nodes=" << cross_stream.summary().total
              << " max_abs=" << maximum_absolute
              << " sdk=" << FLAGDNN_THEAD_PPU_SDK_VERSION
              << " acdnn=" << acdnnGetVersion() << '\n';
    return 0;
  } catch (const std::exception &error) {
    triton_jit::clear_launch_hooks();
    std::cerr << "FLAGDNN_THEAD_GRAPH: FAIL reason=" << error.what() << '\n';
    return 1;
  }
}
