// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "backends/thead/validation/ppu_driver.hpp"

#include <flagdnn/flagdnn.hpp>
#include <flagdnn_frontend.h>

#include <triton_jit/backend_config.h>
#include <triton_jit/backends/cuda_backend.h>
#include <triton_jit/triton_kernel.h>

#include <cuda.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#ifndef BACKEND_CUDA
#error "The THead JIT contract requires CUDA-backend libtriton_jit"
#endif

static_assert(triton_jit::CudaBackend::WARP_SIZE == 32);
static_assert(
    std::is_same_v<triton_jit::DefaultBackend, triton_jit::CudaBackend>);

namespace {

namespace fe = ::flagdnn_frontend;
namespace tv = ::flagdnn::validation::thead;

void require(bool condition, std::string message) {
  if (!condition) {
    throw std::runtime_error(std::move(message));
  }
}

void check_frontend(fe::error_t status, const char* operation) {
  if (status.is_bad()) {
    throw std::runtime_error(std::string(operation) + " failed: " +
                             status.get_message());
  }
}

class TemporaryDirectory final {
 public:
  TemporaryDirectory() {
    std::string pattern =
        (std::filesystem::temp_directory_path() /
         "flagdnn-thead-jit-XXXXXX")
            .string();
    std::vector<char> writable(pattern.begin(), pattern.end());
    writable.push_back('\0');
    char* created = ::mkdtemp(writable.data());
    if (created == nullptr) {
      throw std::runtime_error("mkdtemp failed for THead JIT contract");
    }
    path_ = created;
  }

  ~TemporaryDirectory() {
    std::error_code ignored;
    std::filesystem::remove_all(path_, ignored);
  }

  TemporaryDirectory(const TemporaryDirectory&) = delete;
  TemporaryDirectory& operator=(const TemporaryDirectory&) = delete;

  [[nodiscard]] const std::filesystem::path& path() const noexcept {
    return path_;
  }

 private:
  std::filesystem::path path_;
};

class DeviceBuffer final {
 public:
  explicit DeviceBuffer(std::size_t size) : size_(size) {
    if (size_ != 0) {
      tv::check_driver(cuMemAlloc(&pointer_, size_), "cuMemAlloc");
    }
  }

  ~DeviceBuffer() {
    if (pointer_ != 0) {
      (void)cuMemFree(pointer_);
    }
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  [[nodiscard]] CUdeviceptr get() const noexcept { return pointer_; }

  [[nodiscard]] void* opaque() const noexcept {
    return reinterpret_cast<void*>(static_cast<std::uintptr_t>(pointer_));
  }

 private:
  std::size_t size_ = 0;
  CUdeviceptr pointer_ = 0;
};

struct CacheEntry {
  std::string relative_path;
  std::uintmax_t size = 0;
  std::filesystem::file_time_type write_time{};
  std::string link_target;

  [[nodiscard]] bool operator==(const CacheEntry&) const = default;
};

std::vector<CacheEntry> snapshot(const std::filesystem::path& root) {
  std::vector<CacheEntry> result;
  if (!std::filesystem::exists(root)) {
    return result;
  }
  for (const auto& entry :
       std::filesystem::recursive_directory_iterator(root)) {
    if (!entry.is_regular_file() && !entry.is_symlink()) {
      continue;
    }
    CacheEntry value;
    value.relative_path =
        entry.path().lexically_relative(root).generic_string();
    value.size = entry.is_regular_file() ? entry.file_size() : 0;
    value.write_time = entry.last_write_time();
    if (entry.is_symlink()) {
      value.link_target = std::filesystem::read_symlink(entry.path()).string();
    }
    result.push_back(std::move(value));
  }
  std::sort(result.begin(), result.end(), [](const auto& left,
                                              const auto& right) {
    return left.relative_path < right.relative_path;
  });
  return result;
}

void validate_triton_cubins(const std::filesystem::path& root) {
  std::size_t metadata_count = 0;
  std::size_t cubin_count = 0;
  for (const auto& entry :
       std::filesystem::recursive_directory_iterator(root)) {
    const std::filesystem::path path = entry.path();
    if (path.extension() == ".json" && entry.is_regular_file()) {
      ++metadata_count;
    } else if (path.extension() == ".cubin") {
      require(entry.is_regular_file() && !entry.is_symlink() &&
                  entry.file_size() != 0,
              "Triton CUDA backend emitted an invalid .cubin artifact");
      ++cubin_count;
    }
  }
  require(metadata_count != 0, "PPU JIT cache contains no metadata");
  require(cubin_count != 0, "PPU JIT cache contains no Triton .cubin");
}

fe::graph::Graph::Tensor make_tensor(fe::graph::Graph& graph,
                                     const char* name,
                                     std::int64_t uid) {
  return graph.tensor(fe::graph::Tensor_attributes()
                          .set_name(name)
                          .set_uid(uid)
                          .set_data_type(fe::DataType_t::FLOAT)
                          .set_dim({1024})
                          .set_stride({1}));
}

enum class ExpectedOperation {
  kAdd,
  kSub,
  kMul,
  kMin,
  kMax,
  kRelu,
  kLeakyRelu,
  kSigmoid,
  kTanh,
  kElu,
  kIdentity,
  kGelu,
  kSqrt,
  kNeg,
  kAbs,
  kCeil,
  kFloor,
  kExp,
  kLog,
  kCos,
  kRsqrt,
  kSin,
  kTan,
  kSoftplus,
  kSwish,
  kGeluApproxTanh,
  kDiv,
  kPow,
  kMod,
  kSigmoidBackward,
  kReciprocal,
  kAddSquare,
  kCmpEq,
  kCmpNeq,
  kCmpGt,
  kCmpGe,
  kCmpLt,
  kCmpLe,
};

struct PointwiseOperation {
  const char* name;
  fe::PointwiseMode_t mode;
  ExpectedOperation expected;
  bool unary;
  bool comparison = false;
};

PointwiseOperation requested_operation(int argc, char** argv) {
  if (argc == 4) {
    return {"add", fe::PointwiseMode_t::ADD, ExpectedOperation::kAdd, false};
  }
  if (argc == 6 && std::string_view(argv[4]) == "--operation") {
    const std::string_view operation(argv[5]);
    if (operation == "add") {
      return {"add", fe::PointwiseMode_t::ADD, ExpectedOperation::kAdd,
              false};
    }
    if (operation == "sub") {
      return {"sub", fe::PointwiseMode_t::SUB, ExpectedOperation::kSub,
              false};
    }
    if (operation == "mul") {
      return {"mul", fe::PointwiseMode_t::MUL, ExpectedOperation::kMul,
              false};
    }
    if (operation == "min") {
      return {"min", fe::PointwiseMode_t::MIN, ExpectedOperation::kMin,
              false};
    }
    if (operation == "max") {
      return {"max", fe::PointwiseMode_t::MAX, ExpectedOperation::kMax,
              false};
    }
    if (operation == "scale") {
      return {"scale", fe::PointwiseMode_t::MUL, ExpectedOperation::kMul,
              false};
    }
    if (operation == "relu") {
      return {"relu", fe::PointwiseMode_t::RELU_FWD,
              ExpectedOperation::kRelu, true};
    }
    if (operation == "leaky_relu") {
      return {"leaky_relu", fe::PointwiseMode_t::RELU_FWD,
              ExpectedOperation::kLeakyRelu, true};
    }
    if (operation == "sigmoid") {
      return {"sigmoid", fe::PointwiseMode_t::SIGMOID_FWD,
              ExpectedOperation::kSigmoid, true};
    }
    if (operation == "tanh") {
      return {"tanh", fe::PointwiseMode_t::TANH_FWD,
              ExpectedOperation::kTanh, true};
    }
    if (operation == "elu") {
      return {"elu", fe::PointwiseMode_t::ELU_FWD,
              ExpectedOperation::kElu, true};
    }
    if (operation == "identity") {
      return {"identity", fe::PointwiseMode_t::IDENTITY,
              ExpectedOperation::kIdentity, true};
    }
    if (operation == "gelu") {
      return {"gelu", fe::PointwiseMode_t::GELU_FWD,
              ExpectedOperation::kGelu, true};
    }
    if (operation == "sqrt") {
      return {"sqrt", fe::PointwiseMode_t::SQRT,
              ExpectedOperation::kSqrt, true};
    }
    if (operation == "neg") {
      return {"neg", fe::PointwiseMode_t::NEG,
              ExpectedOperation::kNeg, true};
    }
    if (operation == "abs") {
      return {"abs", fe::PointwiseMode_t::ABS,
              ExpectedOperation::kAbs, true};
    }
    if (operation == "ceil") {
      return {"ceil", fe::PointwiseMode_t::CEIL,
              ExpectedOperation::kCeil, true};
    }
    if (operation == "floor") {
      return {"floor", fe::PointwiseMode_t::FLOOR,
              ExpectedOperation::kFloor, true};
    }
    if (operation == "exp") {
      return {"exp", fe::PointwiseMode_t::EXP,
              ExpectedOperation::kExp, true};
    }
    if (operation == "log") {
      return {"log", fe::PointwiseMode_t::LOG,
              ExpectedOperation::kLog, true};
    }
    if (operation == "cos") {
      return {"cos", fe::PointwiseMode_t::COS,
              ExpectedOperation::kCos, true};
    }
    if (operation == "rsqrt") {
      return {"rsqrt", fe::PointwiseMode_t::RSQRT,
              ExpectedOperation::kRsqrt, true};
    }
    if (operation == "sin") {
      return {"sin", fe::PointwiseMode_t::SIN,
              ExpectedOperation::kSin, true};
    }
    if (operation == "tan") {
      return {"tan", fe::PointwiseMode_t::TAN,
              ExpectedOperation::kTan, true};
    }
    if (operation == "softplus") {
      return {"softplus", fe::PointwiseMode_t::SOFTPLUS_FWD,
              ExpectedOperation::kSoftplus, true};
    }
    if (operation == "swish") {
      return {"swish", fe::PointwiseMode_t::SWISH_FWD,
              ExpectedOperation::kSwish, true};
    }
    if (operation == "gelu_approx_tanh") {
      return {"gelu_approx_tanh",
              fe::PointwiseMode_t::GELU_APPROX_TANH_FWD,
              ExpectedOperation::kGeluApproxTanh, true};
    }
    if (operation == "div") {
      return {"div", fe::PointwiseMode_t::DIV,
              ExpectedOperation::kDiv, false};
    }
    if (operation == "pow") {
      return {"pow", fe::PointwiseMode_t::POW,
              ExpectedOperation::kPow, false};
    }
    if (operation == "mod") {
      return {"mod", fe::PointwiseMode_t::MOD,
              ExpectedOperation::kMod, false};
    }
    if (operation == "sigmoid_backward") {
      return {"sigmoid_backward", fe::PointwiseMode_t::SIGMOID_BWD,
              ExpectedOperation::kSigmoidBackward, false};
    }
    if (operation == "reciprocal") {
      return {"reciprocal", fe::PointwiseMode_t::RECIPROCAL,
              ExpectedOperation::kReciprocal, true};
    }
    if (operation == "add_square") {
      return {"add_square", fe::PointwiseMode_t::ADD,
              ExpectedOperation::kAddSquare, false};
    }
    if (operation == "cmp_eq") {
      return {"cmp_eq", fe::PointwiseMode_t::CMP_EQ,
              ExpectedOperation::kCmpEq, false, true};
    }
    if (operation == "cmp_neq") {
      return {"cmp_neq", fe::PointwiseMode_t::CMP_NEQ,
              ExpectedOperation::kCmpNeq, false, true};
    }
    if (operation == "cmp_gt") {
      return {"cmp_gt", fe::PointwiseMode_t::CMP_GT,
              ExpectedOperation::kCmpGt, false, true};
    }
    if (operation == "cmp_ge") {
      return {"cmp_ge", fe::PointwiseMode_t::CMP_GE,
              ExpectedOperation::kCmpGe, false, true};
    }
    if (operation == "cmp_lt") {
      return {"cmp_lt", fe::PointwiseMode_t::CMP_LT,
              ExpectedOperation::kCmpLt, false, true};
    }
    if (operation == "cmp_le") {
      return {"cmp_le", fe::PointwiseMode_t::CMP_LE,
              ExpectedOperation::kCmpLe, false, true};
    }
  }
  throw std::invalid_argument(
      "usage: test_jit PLUGIN COMPILER_EXECUTABLE COMPILER_ENTRY "
      "[--operation add|sub|mul|min|max|scale|relu|leaky_relu|sigmoid|tanh|elu|identity|gelu|sqrt|neg|abs|ceil|floor|exp|log|cos|rsqrt|sin|tan|softplus|swish|gelu_approx_tanh|div|pow|mod|sigmoid_backward|reciprocal|add_square|cmp_eq|cmp_neq|cmp_gt|cmp_ge|cmp_lt|cmp_le]");
}

float expected_value(ExpectedOperation operation, float left, float right) {
  switch (operation) {
    case ExpectedOperation::kAdd:
      return left + right;
    case ExpectedOperation::kSub:
      return left - right;
    case ExpectedOperation::kMul:
      return left * right;
    case ExpectedOperation::kMin:
      return std::min(left, right);
    case ExpectedOperation::kMax:
      return std::max(left, right);
    case ExpectedOperation::kRelu:
      return std::max(left, 0.0F);
    case ExpectedOperation::kLeakyRelu:
      return left < 0.0F ? 0.2F * left : left;
    case ExpectedOperation::kSigmoid:
      return 1.0F / (1.0F + std::exp(-left));
    case ExpectedOperation::kTanh:
      return std::tanh(left);
    case ExpectedOperation::kElu:
      return left > 0.0F ? left : std::exp(left) - 1.0F;
    case ExpectedOperation::kIdentity:
      return left;
    case ExpectedOperation::kGelu:
      return 0.5F * left *
             (1.0F + std::erf(left / std::sqrt(2.0F)));
    case ExpectedOperation::kSqrt:
      return std::sqrt(left);
    case ExpectedOperation::kNeg:
      return -left;
    case ExpectedOperation::kAbs:
      return std::abs(left);
    case ExpectedOperation::kCeil:
      return std::ceil(left);
    case ExpectedOperation::kFloor:
      return std::floor(left);
    case ExpectedOperation::kExp:
      return std::exp(left);
    case ExpectedOperation::kLog:
      return std::log(left);
    case ExpectedOperation::kCos:
      return std::cos(left);
    case ExpectedOperation::kRsqrt:
      return 1.0F / std::sqrt(left);
    case ExpectedOperation::kSin:
      return std::sin(left);
    case ExpectedOperation::kTan:
      return std::tan(left);
    case ExpectedOperation::kSoftplus:
      return std::max(left, 0.0F) +
             std::log1p(std::exp(-std::abs(left)));
    case ExpectedOperation::kSwish:
      return left / (1.0F + std::exp(-1.25F * left));
    case ExpectedOperation::kGeluApproxTanh: {
      const float inner =
          0.7978845608028654F *
          (left + 0.044715F * left * left * left);
      return 0.5F * left * (1.0F + std::tanh(inner));
    }
    case ExpectedOperation::kDiv:
      return left / right;
    case ExpectedOperation::kPow:
      return std::pow(left, right);
    case ExpectedOperation::kMod:
      return std::fmod(left, right);
    case ExpectedOperation::kSigmoidBackward: {
      const float sigmoid = 1.0F / (1.0F + std::exp(-right));
      return left * sigmoid * (1.0F - sigmoid);
    }
    case ExpectedOperation::kReciprocal:
      return 1.0F / left;
    case ExpectedOperation::kAddSquare:
      return left + right * right;
    case ExpectedOperation::kCmpEq:
      return left == right ? 1.0F : 0.0F;
    case ExpectedOperation::kCmpNeq:
      return left != right ? 1.0F : 0.0F;
    case ExpectedOperation::kCmpGt:
      return left > right ? 1.0F : 0.0F;
    case ExpectedOperation::kCmpGe:
      return left >= right ? 1.0F : 0.0F;
    case ExpectedOperation::kCmpLt:
      return left < right ? 1.0F : 0.0F;
    case ExpectedOperation::kCmpLe:
      return left <= right ? 1.0F : 0.0F;
  }
  throw std::logic_error("unknown THead JIT expected operation");
}

}  // namespace

int main(int argc, char** argv) {
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
            "cannot configure the THead JIT test environment");

    tv::check_driver(cuInit(0), "cuInit");
    CUdevice device = 0;
    tv::check_driver(cuDeviceGet(&device, 0), "cuDeviceGet");
    tv::PrimaryContext primary(device);
    tv::ScopedCurrentContext current(primary.get());
    tv::Stream stream;

    std::atomic<std::size_t> preparation_launches{0};
    triton_jit::set_launch_enter_hook(
        [&](const triton_jit::LaunchMetadata& metadata) {
          const std::string_view expected_kernel =
              operation.expected == ExpectedOperation::kAddSquare
                  ? "add_square_contiguous_kernel"
                  : operation.unary ? "unary_pointwise_contiguous_kernel"
                                    : "binary_contiguous_kernel";
          require(metadata.kernel_name == expected_kernel,
                  "THead prepared an unexpected Triton kernel");
          require(metadata.stream != nullptr,
                  "THead JIT preparation used a null/default stream");
          preparation_launches.fetch_add(1, std::memory_order_relaxed);
        });

    flagdnn::Handle handle("thead", 0);
    require(handle.backend_name() == "thead",
            "public Handle reports a backend other than thead");
    require(handle.target_fingerprint().starts_with("ppu_"),
            "THead public target does not start with ppu_");
    handle.set_compiler(argv[2], argv[3], flagdnn_cache.string());

    fe::graph::Graph graph;
    graph.set_name(std::string("thead_") + operation.name +
                   "_jit_lifecycle")
        .set_io_data_type(fe::DataType_t::FLOAT)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT)
        .set_autotune(false);
    const auto left = make_tensor(graph, "left", 1);
    fe::graph::Pointwise_attributes attributes;
    attributes.set_name(operation.name)
        .set_mode(operation.mode)
        .set_compute_data_type(operation.comparison
                                   ? fe::DataType_t::BOOLEAN
                                   : fe::DataType_t::FLOAT)
        .set_alpha(1.0);
    if (operation.expected == ExpectedOperation::kElu) {
      attributes.set_elu_alpha(1.0F);
    } else if (operation.expected == ExpectedOperation::kLeakyRelu) {
      attributes.set_relu_lower_clip_slope(0.2F);
    } else if (operation.expected == ExpectedOperation::kSoftplus) {
      attributes.set_softplus_beta(1.0F);
    } else if (operation.expected == ExpectedOperation::kSwish) {
      attributes.set_swish_beta(1.25F);
    }
    fe::graph::Graph::Tensor output;
    if (operation.expected == ExpectedOperation::kAddSquare) {
      const auto right = make_tensor(graph, "right", 2);
      const auto square = graph.pointwise(
          right, right,
          fe::graph::Pointwise_attributes()
              .set_name("square")
              .set_mode(fe::PointwiseMode_t::MUL)
              .set_compute_data_type(fe::DataType_t::FLOAT));
      square->set_name("square")
          .set_uid(4)
          .set_data_type(fe::DataType_t::FLOAT)
          .set_dim({1024})
          .set_stride({1})
          .set_is_virtual(true);
      output = graph.pointwise(
          left, square,
          fe::graph::Pointwise_attributes()
              .set_name("add_square")
              .set_mode(fe::PointwiseMode_t::ADD)
              .set_compute_data_type(fe::DataType_t::FLOAT));
    } else if (operation.unary) {
      output = graph.pointwise(left, attributes);
    } else {
      const auto right = make_tensor(graph, "right", 2);
      output = graph.pointwise(left, right, attributes);
    }
    output->set_name("output")
        .set_uid(3)
        .set_data_type(operation.comparison
                           ? fe::DataType_t::BOOLEAN
                           : fe::DataType_t::FLOAT)
        .set_dim({1024})
        .set_stride({1})
        .set_output(true);

    check_frontend(graph.build(handle, {fe::HeurMode_t::A}),
                   "FlagDNN THead pointwise graph build");
    const std::size_t build_launches =
        preparation_launches.load(std::memory_order_relaxed);
    require(build_launches == 1,
            "THead executable build did not prepare exactly one JIT kernel");
    validate_triton_cubins(triton_cache);
    const std::vector<CacheEntry> cache_after_build = snapshot(temporary.path());

    std::array<float, 1024> host_left{};
    std::array<float, 1024> host_right{};
    std::array<float, 1024> host_output{};
    std::array<std::uint8_t, 1024> host_boolean_output{};
    host_output.fill(-12345.0F);
    host_boolean_output.fill(0x7fU);
    for (std::size_t index = 0; index < host_left.size(); ++index) {
      const bool positive_input =
          operation.expected == ExpectedOperation::kSqrt ||
          operation.expected == ExpectedOperation::kLog ||
          operation.expected == ExpectedOperation::kRsqrt ||
          operation.expected == ExpectedOperation::kReciprocal;
      const bool positive_binary =
          operation.expected == ExpectedOperation::kDiv ||
          operation.expected == ExpectedOperation::kPow ||
          operation.expected == ExpectedOperation::kMod;
      host_left[index] =
          positive_input || positive_binary
              ? static_cast<float>((index % 31) + 1) / 7.0F
              : operation.expected == ExpectedOperation::kExp ||
                        operation.expected == ExpectedOperation::kTan
                    ? static_cast<float>(
                          static_cast<int>(index % 31) - 15) /
                          8.0F
              : operation.unary
                    ? static_cast<float>(
                          static_cast<int>(index % 31) - 15)
                    : static_cast<float>(index % 31);
      host_right[index] =
          operation.expected == ExpectedOperation::kPow
              ? static_cast<float>(index % 5)
              : positive_binary
                    ? static_cast<float>((index % 13) + 1) / 5.0F
                    : static_cast<float>(index % 13);
    }
    DeviceBuffer device_left(sizeof(host_left));
    DeviceBuffer device_right(sizeof(host_right));
    DeviceBuffer device_output(operation.comparison
                                   ? sizeof(host_boolean_output)
                                   : sizeof(host_output));
    DeviceBuffer workspace(graph.get_workspace_size());
    tv::check_driver(cuMemcpyHtoDAsync(device_left.get(),
                                      host_left.data(),
                                      sizeof(host_left),
                                      stream.get()),
                     "cuMemcpyHtoDAsync(left)");
    if (!operation.unary) {
      tv::check_driver(cuMemcpyHtoDAsync(device_right.get(),
                                        host_right.data(),
                                        sizeof(host_right),
                                        stream.get()),
                       "cuMemcpyHtoDAsync(right)");
    }
    tv::check_driver(cuMemcpyHtoDAsync(device_output.get(),
                                      operation.comparison
                                          ? static_cast<const void*>(
                                                host_boolean_output.data())
                                          : static_cast<const void*>(
                                                host_output.data()),
                                      operation.comparison
                                          ? sizeof(host_boolean_output)
                                          : sizeof(host_output),
                                      stream.get()),
                     "cuMemcpyHtoDAsync(output sentinel)");

    std::vector<flagdnnBinding_t> bindings = {{1, device_left.opaque()}};
    if (!operation.unary) {
      bindings.push_back({2, device_right.opaque()});
    }
    bindings.push_back({3, device_output.opaque()});
    for (int repetition = 0; repetition < 2; ++repetition) {
      check_frontend(
          graph.execute(handle,
                        std::span<const flagdnnBinding_t>(bindings),
                        workspace.opaque(),
                        graph.get_workspace_size(),
                        stream.opaque()),
          "FlagDNN THead pointwise execute");
    }
    tv::check_driver(cuMemcpyDtoHAsync(operation.comparison
                                          ? static_cast<void*>(
                                                host_boolean_output.data())
                                          : static_cast<void*>(
                                                host_output.data()),
                                      device_output.get(),
                                      operation.comparison
                                          ? sizeof(host_boolean_output)
                                          : sizeof(host_output),
                                      stream.get()),
                     "cuMemcpyDtoHAsync(output)");
    tv::check_driver(cuStreamSynchronize(stream.get()), "cuStreamSynchronize");
    triton_jit::clear_launch_hooks();

    require(preparation_launches.load(std::memory_order_relaxed) ==
                build_launches,
            "steady-state execute re-entered libtriton_jit");
    require(snapshot(temporary.path()) == cache_after_build,
            "steady-state execute changed compiler/JIT cache files");
    for (std::size_t index = 0; index < host_output.size(); ++index) {
      const float expected = expected_value(
          operation.expected, host_left[index], host_right[index]);
      if (operation.comparison) {
        require(host_boolean_output[index] ==
                    static_cast<std::uint8_t>(expected),
                "prepared THead comparison kernel produced a wrong value");
        continue;
      }
      const bool approximate_activation =
          operation.expected == ExpectedOperation::kLeakyRelu ||
          operation.expected == ExpectedOperation::kSigmoid ||
          operation.expected == ExpectedOperation::kTanh ||
          operation.expected == ExpectedOperation::kElu ||
          operation.expected == ExpectedOperation::kGelu ||
          operation.expected == ExpectedOperation::kSqrt ||
          operation.expected == ExpectedOperation::kExp ||
          operation.expected == ExpectedOperation::kLog ||
          operation.expected == ExpectedOperation::kCos ||
          operation.expected == ExpectedOperation::kRsqrt ||
          operation.expected == ExpectedOperation::kSin ||
          operation.expected == ExpectedOperation::kTan ||
          operation.expected == ExpectedOperation::kSoftplus ||
          operation.expected == ExpectedOperation::kSwish ||
          operation.expected == ExpectedOperation::kGeluApproxTanh;
      const bool approximate = approximate_activation ||
                               operation.expected == ExpectedOperation::kDiv ||
                               operation.expected == ExpectedOperation::kPow ||
                               operation.expected == ExpectedOperation::kMod ||
                               operation.expected ==
                                   ExpectedOperation::kSigmoidBackward;
      const float absolute = std::abs(host_output[index] - expected);
      const float relative =
          absolute /
          std::max({std::abs(host_output[index]), std::abs(expected),
                    1.0e-30F});
      require(std::isfinite(host_output[index]) &&
                  (absolute <= (approximate ? 2.0e-5F : 0.0F) ||
                   relative <= (approximate ? 1.0e-5F : 0.0F)),
              "prepared THead pointwise kernel produced a wrong value");
    }

    std::cout << "PASS THead Graph -> CUDA libtriton_jit -> PPU-aware Triton "
              << "prepared " << operation.name << " lifecycle\n";
    return 0;
  } catch (const std::exception& error) {
    triton_jit::clear_launch_hooks();
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
