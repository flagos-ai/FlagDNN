// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "acdnn_reference.hpp"
#include "acdnn_composite_reference.hpp"
#include "backend_pointwise_reference.hpp"
#include "backends/autotune_policy.hpp"
#include "backends/thead/engines/jit_candidate_compatibility.hpp"
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
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#ifndef BACKEND_CUDA
#error "The THead autotune contract requires CUDA-backend libtriton_jit"
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

namespace {

namespace autotune = flagdnn::backend::autotune;
namespace fe = ::flagdnn_frontend;
namespace tv = ::flagdnn::validation::thead;

constexpr std::size_t kElementCount = 1024;

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
      return {"log", fe::PointwiseMode_t::LOG, FLAGDNN_POINTWISE_LOG,
              "acdnnBackendExecute(POINTWISE_LOG)", true};
    }
    if (operation == "cos") {
      return {"cos", fe::PointwiseMode_t::COS, FLAGDNN_POINTWISE_COS,
              "acdnnBackendExecute(POINTWISE_COS)", true};
    }
    if (operation == "rsqrt") {
      return {"rsqrt", fe::PointwiseMode_t::RSQRT,
              FLAGDNN_POINTWISE_RSQRT,
              "acdnnBackendExecute(POINTWISE_RSQRT)", true};
    }
    if (operation == "sin") {
      return {"sin", fe::PointwiseMode_t::SIN, FLAGDNN_POINTWISE_SIN,
              "acdnnBackendExecute(POINTWISE_SIN)", true};
    }
    if (operation == "tan") {
      return {"tan", fe::PointwiseMode_t::TAN, FLAGDNN_POINTWISE_TAN,
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
      "usage: test_autotune PLUGIN COMPILER_EXECUTABLE COMPILER_ENTRY "
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

class TemporaryDirectory final {
 public:
  explicit TemporaryDirectory(std::string_view prefix) {
    std::string pattern =
        (std::filesystem::temp_directory_path() /
         (std::string(prefix) + "-XXXXXX"))
            .string();
    std::vector<char> writable(pattern.begin(), pattern.end());
    writable.push_back('\0');
    const char *created = ::mkdtemp(writable.data());
    if (created == nullptr) {
      throw std::runtime_error("mkdtemp failed for THead autotune contract");
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
  return {std::istreambuf_iterator<char>(input),
          std::istreambuf_iterator<char>()};
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

void require_no_temporary_cache_files(const std::filesystem::path &root) {
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
        filename == ".flagdnn-autotune-v1-stage-0.json") {
      result = entry.path();
      ++count;
    }
  }
  require(count == 1,
          "THead autotune must publish exactly one winner cache");
  return result;
}

autotune::SelectionRequest policy_request(
    const std::filesystem::path &cache) {
  autotune::SelectionRequest request;
  request.candidate_identity = std::string(64, '1');
  request.device_identity = "ppu-device-contract";
  request.measurement_identity = "thead-cuda-event-contract-v1";
  request.cache_path = cache;
  request.candidate_ids = {"block128_w4_s1", "block256_w4_s1"};
  request.warmup_milliseconds = 1;
  request.benchmark_milliseconds = 1;
  return request;
}

void test_shared_policy_contract() {
  TemporaryDirectory temporary("flagdnn-thead-autotune-policy");
  const std::filesystem::path cache = temporary.path() / "winner.json";
  autotune::SelectionRequest request = policy_request(cache);
  std::size_t warmup_calls = 0;
  std::size_t measure_calls = 0;
  const auto warmup = [&](std::size_t, unsigned int iterations) {
    require(iterations != 0, "policy requested an empty warmup batch");
    ++warmup_calls;
  };
  const auto measure = [&](std::size_t index, unsigned int iterations) {
    require(iterations != 0, "policy requested an empty timing batch");
    ++measure_calls;
    return index == 0 ? 0.20F : 0.10F;
  };

  const autotune::SelectionResult miss =
      autotune::select_best_candidate(request, warmup, measure);
  require(!miss.cache_hit && miss.candidate_index == 1,
          "autotune miss did not select exactly the fastest candidate");
  require(warmup_calls != 0 && measure_calls != 0,
          "autotune miss did not measure the candidate space");
  require(std::filesystem::is_regular_file(cache),
          "autotune miss did not atomically publish the winner");
  require(read_file(cache).find("\"variant_id\":\"block256_w4_s1\"") !=
              std::string::npos,
          "winner cache did not name the selected candidate");
  require_no_temporary_cache_files(temporary.path());

  warmup_calls = 0;
  measure_calls = 0;
  const autotune::SelectionResult hit =
      autotune::select_best_candidate(request, warmup, measure);
  require(hit.cache_hit && hit.candidate_index == 1,
          "second build did not reuse the cached winner");
  require(warmup_calls == 0 && measure_calls == 0,
          "winner-cache hit unexpectedly rebenchmarked candidates");

  const auto require_miss = [&](autotune::SelectionRequest changed,
                                std::string_view description) {
    require(!autotune::find_cached_candidate(changed).has_value(),
            description);
  };
  for (const std::string_view mutation :
       {"kernel", "tuning", "Triton"}) {
    autotune::SelectionRequest changed = request;
    changed.candidate_identity = std::string(64, mutation.front());
    require_miss(changed,
                 std::string(mutation) +
                     " identity mutation did not invalidate the winner");
  }
  autotune::SelectionRequest changed = request;
  changed.measurement_identity = "thead-cuda-event-contract-v2-jit-mutation";
  require_miss(changed,
               "libtriton_jit identity mutation did not invalidate winner");
  changed = request;
  changed.device_identity = "another-ppu-device";
  require_miss(changed, "device identity mutation did not invalidate winner");
  changed = request;
  changed.candidate_ids = {"block64_w2_s1", "block128_w4_s1"};
  require_miss(changed,
               "candidate-space mutation did not invalidate the winner");

  write_file(cache, "{}\n");
  measure_calls = 0;
  const autotune::SelectionResult retuned =
      autotune::select_best_candidate(request, warmup, measure);
  require(!retuned.cache_hit && retuned.candidate_index == 1 &&
              measure_calls != 0,
          "corrupt winner cache did not trigger full retuning");
  require_no_temporary_cache_files(temporary.path());

  for (const std::string_view failure :
       {"schema failure", "ABI failure", "JIT failure", "runtime failure"}) {
    autotune::discard_cached_candidate(request);
    try {
      (void)autotune::select_best_candidate(
          request, warmup,
          [failure](std::size_t, unsigned int) -> float {
            throw std::runtime_error(std::string(failure));
          });
    } catch (const std::runtime_error &error) {
      require(error.what() == failure,
              "autotune rewrote a fatal candidate diagnostic");
      continue;
    }
    throw std::runtime_error(
        "autotune filtered a fatal candidate error instead of failing build");
  }

  const auto require_invalid_timing = [&](float timing) {
    autotune::discard_cached_candidate(request);
    try {
      (void)autotune::select_best_candidate(
          request, warmup,
          [timing](std::size_t, unsigned int) { return timing; });
    } catch (const std::runtime_error &) {
      return;
    }
    throw std::runtime_error("autotune accepted an invalid event timing");
  };
  require_invalid_timing(0.0F);
  require_invalid_timing(std::numeric_limits<float>::infinity());
}

void test_resource_compatibility_contract() {
  flagdnn::thead::KernelVariant variant;
  variant.grid = {4, 1, 1};
  variant.block = {128, 1, 1};
  variant.shared_memory = 0;
  const flagdnn::thead::CandidateDeviceLimits limits{
      .maximum_threads_per_block = 1024,
      .maximum_block = {1024, 1024, 64},
      .maximum_grid = {65535, 65535, 65535},
      .maximum_shared_memory = 65536,
  };
  flagdnn::thead::validate_candidate_resources(variant, limits);

  const auto require_filtered = [&](flagdnn::thead::KernelVariant changed,
                                    std::string_view description) {
    try {
      flagdnn::thead::validate_candidate_resources(changed, limits);
    } catch (const flagdnn::thead::CandidateCompatibilityError &) {
      return;
    }
    throw std::runtime_error(std::string(description));
  };
  flagdnn::thead::KernelVariant changed = variant;
  changed.grid[0] = limits.maximum_grid[0] + 1;
  require_filtered(changed, "oversized grid candidate was not filtered");
  changed = variant;
  changed.block[2] = limits.maximum_block[2] + 1;
  require_filtered(changed, "oversized block dimension was not filtered");
  changed = variant;
  changed.block = {1024, 2, 1};
  require_filtered(changed, "oversized thread count was not filtered");
  changed = variant;
  changed.shared_memory = limits.maximum_shared_memory + 1;
  require_filtered(changed, "oversized shared memory was not filtered");
}

tv::CapabilityRecord reference_capability(
    const PointwiseOperation &operation) {
  tv::CapabilityConstraints constraints;
  constraints.dtypes = {"fp32"};
  constraints.compute_type = operation.comparison ? "bool" : "fp32";
  constraints.rank = {1};
  constraints.shape = {"same_shape", "positive_extents"};
  constraints.layouts = {"contiguous"};
  constraints.stride_policy = "dense_contiguous";
  constraints.broadcast = "none";
  if (operation.unary) {
    constraints.attributes = {
        {"reference_nan_policy", {"not_propagate_finite_inputs_only"}},
        {"autotune", {"true"}},
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
    constraints.attributes = {{"alpha", {"1"}}, {"autotune", {"true"}}};
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
      .detail = std::string("autotuned FlagDNN ") + operation.name +
                " qualification against exact acDNN pointwise operation",
  };
}

flagdnn::testing::TestTensor test_tensor(
    std::int64_t uid,
    flagdnnDataType_t data_type = FLAGDNN_DATA_FLOAT32) {
  return {
      .uid = uid,
      .data_type = data_type,
      .dimensions = {static_cast<std::int64_t>(kElementCount)},
      .strides = {1},
      .binding_byte_offset = 0,
  };
}

std::unique_ptr<fe::graph::Graph> build_pointwise_graph(
    flagdnn::Handle &handle, const PointwiseOperation &operation) {
  auto graph = std::make_unique<fe::graph::Graph>();
  graph->set_name(std::string("thead_") + operation.name +
                  "_autotune_contract")
      .set_io_data_type(fe::DataType_t::FLOAT)
      .set_intermediate_data_type(fe::DataType_t::FLOAT)
      .set_compute_data_type(fe::DataType_t::FLOAT)
      .set_autotune(true);
  const auto left = graph->tensor(fe::graph::Tensor_attributes()
                                      .set_name("left")
                                      .set_uid(1)
                                      .set_data_type(fe::DataType_t::FLOAT)
                                      .set_dim({kElementCount})
                                      .set_stride({1}));
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
                                         .set_dim({kElementCount})
                                         .set_stride({1}));
    const auto square = graph->pointwise(
        right, right,
        fe::graph::Pointwise_attributes()
            .set_name("square")
            .set_mode(fe::PointwiseMode_t::MUL)
            .set_compute_data_type(fe::DataType_t::FLOAT));
    square->set_name("square")
        .set_uid(4)
        .set_data_type(fe::DataType_t::FLOAT)
        .set_dim({kElementCount})
        .set_stride({1})
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
                                         .set_dim({kElementCount})
                                         .set_stride({1}));
    output = graph->pointwise(left, right, attributes);
  }
  output->set_name("output")
      .set_uid(3)
      .set_data_type(operation.comparison
                         ? fe::DataType_t::BOOLEAN
                         : fe::DataType_t::FLOAT)
      .set_dim({kElementCount})
      .set_stride({1})
      .set_output(true);
  check_frontend(graph->build(handle, {fe::HeurMode_t::A}),
                 "FlagDNN THead autotune pointwise build");
  require(graph->get_workspace_size() == 0,
          "autotuned operation has an unexpected workspace size");
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
                     "cuStreamSynchronize(THead autotune bool output)");
    std::transform(bytes.begin(), bytes.end(), result.begin(),
                   [](std::uint8_t value) {
                     return value == 0U ? 0.0F : 1.0F;
                   });
    return result;
  }
  tv::copy_from_device_async(std::span<float>(result), buffer, 0, stream);
  tv::check_driver(cuStreamSynchronize(stream),
                   "cuStreamSynchronize(THead autotune output)");
  return result;
}

void compare_with_acdnn(std::span<const float> actual,
                        std::span<const float> reference,
                        flagdnnPointwiseMode_t mode) {
  require(actual.size() == reference.size(),
          "FlagDNN and acDNN autotune output sizes differ");
  for (std::size_t index = 0; index < actual.size(); ++index) {
    const double difference =
        std::abs(static_cast<double>(actual[index]) - reference[index]);
    const double relative =
        difference /
        std::max({std::abs(static_cast<double>(actual[index])),
                  std::abs(static_cast<double>(reference[index])), 1.0e-30});
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
      throw std::runtime_error(
          "autotuned FlagDNN pointwise operation differs from acDNN at "
          "element " +
          std::to_string(index));
    }
  }
}

void test_device_autotune_contract(int argc, char **argv) {
  const PointwiseOperation operation = requested_operation(argc, argv);
  const std::filesystem::path plugin = std::filesystem::canonical(argv[1]);
  TemporaryDirectory temporary("flagdnn-thead-autotune-device");
  const std::filesystem::path flagdnn_cache = temporary.path() / "flagdnn";
  const std::filesystem::path triton_cache = temporary.path() / "triton";
  require(setenv("FLAGDNN_BACKEND_PATH", plugin.parent_path().c_str(), 1) ==
                  0 &&
              setenv("FLAGDNN_EXECUTION_ENGINE", "libtriton_jit", 1) == 0 &&
              setenv("TRITON_CACHE_DIR", triton_cache.c_str(), 1) == 0 &&
              setenv("TRITON_JIT_BACKEND", "CUDA", 1) == 0 &&
              setenv("PYTHONDONTWRITEBYTECODE", "1", 1) == 0,
          "cannot configure the THead autotune test environment");

  const tv::CapabilityCatalog catalog = tv::CapabilityCatalog::load(
      FLAGDNN_THEAD_ACDNN_CAPABILITY_CATALOG);
  catalog.validate_versions(FLAGDNN_THEAD_PPU_SDK_VERSION, ACDNN_VERSION,
                            static_cast<std::int64_t>(acdnnGetVersion()));

  tv::check_driver(cuInit(0), "cuInit");
  CUdevice device = 0;
  tv::check_driver(cuDeviceGet(&device, 0), "cuDeviceGet");
  tv::PrimaryContext primary(device);
  tv::ScopedCurrentContext current(primary.get());
  tv::DeviceStream stream;

  std::atomic<std::size_t> jit_launches{0};
  triton_jit::set_launch_enter_hook(
      [&](const triton_jit::LaunchMetadata &metadata) {
        const std::string_view expected_kernel =
            std::string_view(operation.name) == "add_square"
                ? "add_square_contiguous_kernel"
            : std::string_view(operation.name) == "sigmoid_backward"
                ? "activation_backward_contiguous_kernel"
            : operation.unary ? "unary_pointwise_contiguous_kernel"
                              : "binary_contiguous_kernel";
        require(metadata.kernel_name == expected_kernel,
                "THead autotune launched an unexpected Triton kernel");
        require(metadata.stream != nullptr,
                "THead autotune used the default stream");
        jit_launches.fetch_add(1, std::memory_order_relaxed);
      });

  flagdnn::Handle handle("thead", 0);
  handle.set_compiler(argv[2], argv[3], flagdnn_cache.string());
  auto miss_graph = build_pointwise_graph(handle, operation);
  const std::size_t miss_launches =
      jit_launches.load(std::memory_order_relaxed);
  require(miss_launches > 3,
          "autotune cache miss did not prepare and time both candidates");
  const std::filesystem::path selection =
      find_selection_cache(flagdnn_cache);
  const std::string winner = read_file(selection);
  require_no_temporary_cache_files(flagdnn_cache);

  auto hit_graph = build_pointwise_graph(handle, operation);
  const std::size_t hit_launches =
      jit_launches.load(std::memory_order_relaxed) - miss_launches;
  require(hit_launches == 2,
          "winner-cache hit did not prepare only the selected candidate: "
          "expected=2 actual=" +
              std::to_string(hit_launches));
  require(read_file(selection) == winner,
          "winner-cache hit unexpectedly rewrote the selection");

  const std::array<float, kElementCount> host_left =
      make_input(0, operation.flagdnn_mode);
  const std::array<float, kElementCount> host_right =
      make_input(1, operation.flagdnn_mode);
  tv::DeviceBuffer left(sizeof(host_left));
  tv::DeviceBuffer right(sizeof(host_right));
  tv::DeviceBuffer output(sizeof(host_left));
  tv::DeviceBuffer reference_output(sizeof(host_left));
  tv::copy_to_device_async(left, std::span<const float>(host_left), 0,
                           stream.get());
  if (!operation.unary) {
    tv::copy_to_device_async(right, std::span<const float>(host_right), 0,
                             stream.get());
  }
  std::vector<flagdnnBinding_t> bindings = {{1, left.data()}};
  std::vector<flagdnnBinding_t> reference_bindings = {{1, left.data()}};
  if (!operation.unary) {
    bindings.push_back({2, right.data()});
    reference_bindings.push_back({2, right.data()});
  }
  bindings.push_back({3, output.data()});
  reference_bindings.push_back({3, reference_output.data()});

  std::vector<flagdnn::testing::TestTensor> reference_inputs = {
      test_tensor(1)};
  if (!operation.unary) {
    reference_inputs.push_back(test_tensor(2));
  }
  flagdnnPointwiseAttributes_t reference_attributes =
      FLAGDNN_POINTWISE_ATTRIBUTES_INITIALIZER;
  if (operation.flagdnn_mode == FLAGDNN_POINTWISE_SOFTPLUS_FWD) {
    reference_attributes.flags = FLAGDNN_POINTWISE_ATTRIBUTE_SOFTPLUS_BETA;
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
        reference_inputs.at(0), reference_inputs.at(1),
        test_tensor(3), reference_capability(operation));
  } else {
    reference = tv::make_acdnn_pointwise_reference(
        {.mode = operation.flagdnn_mode,
         .inputs = std::move(reference_inputs),
         .output = test_tensor(
             3, operation.comparison ? FLAGDNN_DATA_BOOLEAN
                                     : FLAGDNN_DATA_FLOAT32),
         .alpha = 1.0,
         .attributes = reference_attributes},
        reference_capability(operation));
  }
  tv::DeviceBuffer reference_workspace(reference->workspace_size());
  reference->prepare(reference_bindings, stream.opaque());
  reference->execute(reference_bindings, reference_workspace.data(),
                     reference_workspace.size(), stream.opaque());

  const std::size_t before_execute =
      jit_launches.load(std::memory_order_relaxed);
  tv::DeviceBuffer graph_workspace(hit_graph->get_workspace_size());
  for (int repetition = 0; repetition < 3; ++repetition) {
    check_frontend(hit_graph->execute(
                       handle, std::span<const flagdnnBinding_t>(bindings),
                       graph_workspace.data(), graph_workspace.size(),
                       stream.opaque()),
                   "FlagDNN THead autotuned pointwise execute");
  }
  require(jit_launches.load(std::memory_order_relaxed) == before_execute,
          "steady execute re-entered JIT or autotune");
  require(read_file(selection) == winner,
          "steady execute read or rewrote the winner cache");
  compare_with_acdnn(read_output(output, stream.get(), operation.comparison),
                     read_output(reference_output, stream.get(),
                                 operation.comparison),
                     operation.flagdnn_mode);

  write_file(selection, "{}\n");
  const std::size_t before_retune =
      jit_launches.load(std::memory_order_relaxed);
  auto retuned_graph = build_pointwise_graph(handle, operation);
  const std::size_t retune_launches =
      jit_launches.load(std::memory_order_relaxed) - before_retune;
  require(retune_launches > hit_launches,
          "corrupt winner cache did not trigger candidate rebenchmarking");
  require(read_file(selection) != "{}\n",
          "corrupt winner cache was not atomically replaced");
  require_no_temporary_cache_files(flagdnn_cache);

  triton_jit::clear_launch_hooks();
  std::cout << "FLAGDNN_THEAD_AUTOTUNE: PASS miss_jit_launches="
            << miss_launches << " hit_jit_launches=" << hit_launches
            << " retune_jit_launches=" << retune_launches
            << " operation=" << operation.name
            << " acdnn=" << acdnnGetVersion() << '\n';
}

}  // namespace

int main(int argc, char **argv) {
  try {
    test_shared_policy_contract();
    test_resource_compatibility_contract();
    if (argc == 2 && std::string_view(argv[1]) == "--policy-contract") {
      std::cout << "FLAGDNN_THEAD_AUTOTUNE_POLICY: PASS\n";
      return 0;
    }
    test_device_autotune_contract(argc, argv);
    return 0;
  } catch (const std::exception &error) {
    triton_jit::clear_launch_hooks();
    std::cerr << "FLAGDNN_THEAD_AUTOTUNE: FAIL reason=" << error.what()
              << '\n';
    return 1;
  }
}
