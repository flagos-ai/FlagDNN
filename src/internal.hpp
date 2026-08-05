/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_INTERNAL_HPP_
#define FLAGDNN_INTERNAL_HPP_

#include "error.hpp"
#include "graph/graph.hpp"
#include "runtime/context.hpp"

#include <memory>
#include <string>
#include <utility>

struct flagdnnContext {
  explicit flagdnnContext(flagdnnBackend_t backend, std::int32_t device)
      : implementation(backend, device) {}
  explicit flagdnnContext(std::string backend_name, std::int32_t device)
      : implementation(std::move(backend_name), device) {}
  flagdnn::native::RuntimeContext implementation;
};

struct flagdnnTensorDescriptor {
  flagdnn::native::TensorSpec specification;
};

struct flagdnnOperationDescriptor {
  explicit flagdnnOperationDescriptor(flagdnnOperation_t operation)
      : specification(operation) {}
  flagdnn::native::OperationSpec specification;
};

struct flagdnnGraph {
  flagdnn::native::GraphSpec specification;
};

struct flagdnnExecutable {
  explicit flagdnnExecutable(
      std::unique_ptr<flagdnn::native::Executable> value)
      : implementation(std::move(value)) {}
  std::unique_ptr<flagdnn::native::Executable> implementation;
};

#endif  // FLAGDNN_INTERNAL_HPP_
