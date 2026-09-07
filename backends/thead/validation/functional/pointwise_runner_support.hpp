// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef FLAGDNN_BACKENDS_THEAD_VALIDATION_FUNCTIONAL_POINTWISE_RUNNER_SUPPORT_HPP_
#define FLAGDNN_BACKENDS_THEAD_VALIDATION_FUNCTIONAL_POINTWISE_RUNNER_SUPPORT_HPP_

#include "capability.hpp"
#include "common/common.hpp"
#include "common/pointwise.hpp"
#include "tensor_io.hpp"

#include <flagdnn/flagdnn.hpp>

#include <cstddef>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace flagdnn::validation::thead::functional {

inline constexpr float kPaddingSentinel = -64.0F;

class TemporaryCache final {
 public:
  TemporaryCache();
  ~TemporaryCache() noexcept;

  TemporaryCache(const TemporaryCache &) = delete;
  TemporaryCache &operator=(const TemporaryCache &) = delete;

  [[nodiscard]] const std::filesystem::path &path() const noexcept {
    return path_;
  }

 private:
  std::filesystem::path path_;
  bool owned_ = false;
};

struct BoundTensor {
  flagdnn::testing::TestTensor specification;
  std::unique_ptr<DeviceBuffer> buffer;
};

[[nodiscard]] std::size_t
element_count(const flagdnn::testing::TestTensor &tensor);
[[nodiscard]] std::size_t
storage_element_count(const flagdnn::testing::TestTensor &tensor);
[[nodiscard]] bool
is_contiguous(const flagdnn::testing::TestTensor &tensor);
[[nodiscard]] std::string
data_type_name(flagdnnDataType_t data_type);
[[nodiscard]] std::string
shape_name(const flagdnn::testing::TestTensor &tensor);
[[nodiscard]] std::string
layout_name(const flagdnn::testing::TestTensor &tensor);

[[nodiscard]] BoundTensor
make_input_buffer(const flagdnn::testing::TestTensor &tensor,
                  std::size_t input_index, CUstream stream,
                  flagdnn::testing::PointwiseInputDomain domain =
                      flagdnn::testing::PointwiseInputDomain::kReal);
[[nodiscard]] BoundTensor
make_output_buffer(const flagdnn::testing::TestTensor &tensor,
                   CUstream stream);
[[nodiscard]] std::vector<float>
read_output(const BoundTensor &tensor, CUstream stream);
[[nodiscard]] std::vector<float>
gather(std::span<const float> physical,
       const flagdnn::testing::TestTensor &tensor);
void require_padding_unchanged(
    std::string_view provider, std::span<const float> physical,
    const flagdnn::testing::TestTensor &tensor);

[[nodiscard]] std::vector<flagdnnBinding_t>
bindings(std::span<const BoundTensor> inputs,
         std::span<const BoundTensor> outputs);
void execute(flagdnn::testing::TestExecutable &executable,
             std::span<const flagdnnBinding_t> bindings,
             DeviceBuffer &workspace, DeviceStream &stream);

}  // namespace flagdnn::validation::thead::functional

#endif  // FLAGDNN_BACKENDS_THEAD_VALIDATION_FUNCTIONAL_POINTWISE_RUNNER_SUPPORT_HPP_
