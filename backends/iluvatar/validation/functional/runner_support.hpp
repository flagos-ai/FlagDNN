// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef FLAGDNN_BACKENDS_ILUVATAR_VALIDATION_FUNCTIONAL_RUNNER_SUPPORT_HPP_
#define FLAGDNN_BACKENDS_ILUVATAR_VALIDATION_FUNCTIONAL_RUNNER_SUPPORT_HPP_

#include "common/common.hpp"
#include "corex_cudnn_reference.hpp"

#include <cuda.h>
#include <flagdnn/flagdnn.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace flagdnn::iluvatar::validation::functional {

inline constexpr float kPaddingSentinel = -64.0F;

void check_cuda(CUresult status, const char *operation);

class DriverContext final {
public:
  DriverContext();
  ~DriverContext() noexcept;
  DriverContext(const DriverContext &) = delete;
  DriverContext &operator=(const DriverContext &) = delete;

private:
  CUdevice device_ = 0;
  CUcontext context_ = nullptr;
};

class Stream final {
public:
  Stream();
  ~Stream() noexcept;
  Stream(const Stream &) = delete;
  Stream &operator=(const Stream &) = delete;
  [[nodiscard]] CUstream get() const noexcept { return stream_; }
  [[nodiscard]] flagdnnStream_t opaque() const noexcept {
    return reinterpret_cast<flagdnnStream_t>(stream_);
  }
  void synchronize() const;

private:
  CUstream stream_ = nullptr;
};

class DeviceBuffer final {
public:
  explicit DeviceBuffer(std::size_t bytes);
  ~DeviceBuffer() noexcept;
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;
  [[nodiscard]] void *at(std::size_t byte_offset = 0) const;
  void copy_from_host(const void *source, std::size_t bytes,
                      std::size_t byte_offset, CUstream stream) const;
  void copy_to_host(void *destination, std::size_t bytes,
                    std::size_t byte_offset, CUstream stream) const;

private:
  CUdeviceptr pointer_ = 0;
  std::size_t bytes_ = 0;
};

[[nodiscard]] std::size_t data_type_size(flagdnnDataType_t data_type);
[[nodiscard]] std::size_t
element_count(const flagdnn::testing::TestTensor &tensor);
[[nodiscard]] std::size_t
storage_element_count(const flagdnn::testing::TestTensor &tensor);
[[nodiscard]] std::vector<float>
scatter(std::span<const float> logical,
        const flagdnn::testing::TestTensor &tensor);
[[nodiscard]] std::vector<float>
gather(std::span<const float> physical,
       const flagdnn::testing::TestTensor &tensor);
[[nodiscard]] std::vector<std::uint8_t> encode(std::span<const float> values,
                                               flagdnnDataType_t data_type);
[[nodiscard]] std::vector<float> decode(std::span<const std::uint8_t> bytes,
                                        flagdnnDataType_t data_type,
                                        std::size_t element_count);

void emit_reference_skip(std::string_view operation, std::string_view case_name,
                         const flagdnn::testing::TestTensor &tensor,
                         std::string_view reason);

enum class InputDomain {
  kReal,
  kPositive,
  kScaled,
  kTan,
  kDivisor,
  kModulo,
  kPower,
  kModuloSigned,
  kComparison,
  kLogical,
  kUnitScale,
  kVariance,
  kStats,
};

struct PlannedInput {
  flagdnn::testing::TestTensor tensor;
  InputDomain domain = InputDomain::kReal;
  std::vector<float> exact_values;
};

struct PlannedOutput {
  flagdnn::testing::TestTensor tensor;
  double absolute_tolerance = 0.0;
  double relative_tolerance = 0.0;
  std::string label = "output";
};

struct CasePlan {
  std::string operation;
  std::string case_name;
  std::vector<PlannedInput> inputs;
  std::vector<PlannedOutput> outputs;
};

using BuildExecutable =
    std::function<std::unique_ptr<flagdnn::testing::TestExecutable>()>;

using HostReference = std::function<std::vector<std::vector<float>>(
    const std::vector<std::vector<float>> &)>;

class FunctionalSuite final {
public:
  FunctionalSuite(int argc, char **argv, std::string operation,
                  std::string marker);
  ~FunctionalSuite() noexcept;
  FunctionalSuite(const FunctionalSuite &) = delete;
  FunctionalSuite &operator=(const FunctionalSuite &) = delete;

  [[nodiscard]] flagdnn::Handle &handle() noexcept { return handle_; }
  void run(const CasePlan &plan, const BuildExecutable &build_production,
           const BuildExecutable &build_reference,
           const HostReference &host_reference = {},
           bool probe_reference = false,
           const HostReference &cpu_fallback = {});
  void skip_benchmark_case(const CasePlan &plan, std::string_view reason);
  void run_raw(const CasePlan &plan, const BuildExecutable &build_production,
               const std::vector<std::vector<std::uint8_t>> &inputs,
               const std::vector<std::vector<std::uint8_t>> &expected,
               const BuildExecutable &build_reference = {});
  [[nodiscard]] int finish();

private:
  DriverContext driver_;
  Stream stream_;
  bool owns_cache_ = true;
  std::filesystem::path cache_path_;
  flagdnn::Handle handle_;
  CorexCudnnCapabilityCatalog catalog_;
  std::string operation_;
  std::string marker_;
  std::size_t cases_ = 0;
  std::size_t production_executed_ = 0;
  std::size_t reference_executed_ = 0;
  std::size_t reference_skipped_ = 0;
  bool benchmark_ = false;
  bool finished_ = false;
  bool qualify_candidates_ = false;
};

[[nodiscard]] InputDomain pointwise_input_domain(int domain);

} // namespace flagdnn::iluvatar::validation::functional

#endif // FLAGDNN_BACKENDS_ILUVATAR_VALIDATION_FUNCTIONAL_RUNNER_SUPPORT_HPP_
