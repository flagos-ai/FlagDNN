/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_TESTS_COMMON_COMMON_HPP_
#define FLAGDNN_TESTS_COMMON_COMMON_HPP_

#include <flagdnn/flagdnn.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace flagdnn {
class Handle;
}

namespace flagdnn::testing {

struct TestTensor {
  std::int64_t uid = 0;
  flagdnnDataType_t data_type = FLAGDNN_DATA_FLOAT32;
  std::vector<std::int64_t> dimensions;
  std::vector<std::int64_t> strides;
  std::size_t binding_byte_offset = 0;
};

class TestExecutable {
 public:
  virtual ~TestExecutable() = default;

  [[nodiscard]] virtual std::size_t workspace_size() const noexcept = 0;
  virtual void execute(std::span<const flagdnnBinding_t> bindings,
                       void* workspace,
                       std::size_t workspace_size,
                       flagdnnStream_t stream) = 0;
};

}  // namespace flagdnn::testing

#endif  // FLAGDNN_TESTS_COMMON_COMMON_HPP_
