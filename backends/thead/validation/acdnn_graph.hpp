// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef FLAGDNN_THEAD_VALIDATION_ACDNN_GRAPH_HPP_
#define FLAGDNN_THEAD_VALIDATION_ACDNN_GRAPH_HPP_
#include <algorithm>
#include <map>
#include <memory>
#include <span>
#include <vector>

#include "acdnn_reference.hpp"
#include "common/common.hpp"

namespace flagdnn::validation::thead {
class AcdnnDescriptor {
 public:
  explicit AcdnnDescriptor(acdnnBackendDescriptorType_t type) {
    check_acdnn(acdnnBackendCreateDescriptor(type, &value_),
                "acdnnBackendCreateDescriptor(extended)");
  }
  ~AcdnnDescriptor() {
    if (value_) (void)acdnnBackendDestroyDescriptor(value_);
  }
  AcdnnDescriptor(const AcdnnDescriptor&) = delete;
  AcdnnDescriptor& operator=(const AcdnnDescriptor&) = delete;
  acdnnBackendDescriptor_t get() const { return value_; }
  template <class T>
  void set(acdnnBackendAttributeName_t name, acdnnBackendAttributeType_t type,
           const T& value) {
    set_array(name, type, 1, &value);
  }
  void set_array(acdnnBackendAttributeName_t name,
                 acdnnBackendAttributeType_t type, std::int64_t count,
                 const void* values) {
    check_acdnn(
        acdnnBackendSetAttribute(value_, name, type, count, values),
        "acdnnBackendSetAttribute(extended " + std::to_string(name) + ")");
  }
  void finalize() {
    check_acdnn(acdnnBackendFinalize(value_), "acdnnBackendFinalize(extended)");
  }

 private:
  acdnnBackendDescriptor_t value_ = nullptr;
};
inline acdnnDataType_t acdnn_float_type(flagdnnDataType_t type) {
  switch (type) {
    case FLAGDNN_DATA_FLOAT32:
      return ACDNN_DATA_FLOAT;
    case FLAGDNN_DATA_FLOAT16:
      return ACDNN_DATA_HALF;
    case FLAGDNN_DATA_BFLOAT16:
      return ACDNN_DATA_BF16;
    case FLAGDNN_DATA_INT32:
      return ACDNN_DATA_INT32;
    default:
      throw std::invalid_argument("acDNN extended tensor dtype is unsupported");
  }
}
class AcdnnGraphReference : public flagdnn::testing::TestExecutable {
 public:
  std::size_t workspace_size() const noexcept override { return workspace_; }
  void execute(std::span<const flagdnnBinding_t> bindings, void* workspace,
               std::size_t size, flagdnnStream_t stream) override {
    if (size < workspace_ || (workspace_ && !workspace))
      throw std::invalid_argument("acDNN graph workspace mismatch");
    handle_.set_stream(reinterpret_cast<hggcStream_t>(stream));
    std::vector<std::int64_t> uids;
    std::vector<void*> pointers;
    for (const auto& b : bindings) {
      uids.push_back(b.uid);
      pointers.push_back(b.device_pointer);
    }
    for (auto& [uid, data] : constants_) {
      uids.push_back(uid);
      pointers.push_back(data.data());
    }
    AcdnnDescriptor pack(ACDNN_BACKEND_VARIANT_PACK_DESCRIPTOR);
    pack.set_array(ACDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, ACDNN_TYPE_INT64,
                   static_cast<std::int64_t>(uids.size()), uids.data());
    pack.set_array(ACDNN_ATTR_VARIANT_PACK_DATA_POINTERS, ACDNN_TYPE_VOID_PTR,
                   static_cast<std::int64_t>(pointers.size()), pointers.data());
    pack.set(ACDNN_ATTR_VARIANT_PACK_WORKSPACE, ACDNN_TYPE_VOID_PTR, workspace);
    pack.set(ACDNN_ATTR_VARIANT_PACK_WORKSPACE_SIZE, ACDNN_TYPE_INT64,
             static_cast<std::int64_t>(size));
    pack.finalize();
    check_acdnn(acdnnBackendExecute(handle_.get(), plan_->get(), pack.get()),
                "acdnnBackendExecute(extended)");
  }

 protected:
  AcdnnDescriptor& descriptor(acdnnBackendDescriptorType_t type) {
    descriptors_.push_back(std::make_unique<AcdnnDescriptor>(type));
    return *descriptors_.back();
  }
  acdnnBackendDescriptor_t tensor(const flagdnn::testing::TestTensor& t,
                                  acdnnDataType_t type,
                                  std::int64_t alignment = 16) {
    auto& d = descriptor(ACDNN_BACKEND_TENSOR_DESCRIPTOR);
    d.set(ACDNN_ATTR_TENSOR_UNIQUE_ID, ACDNN_TYPE_INT64, t.uid);
    d.set(ACDNN_ATTR_TENSOR_DATA_TYPE, ACDNN_TYPE_DATA_TYPE, type);
    d.set(ACDNN_ATTR_TENSOR_BYTE_ALIGNMENT, ACDNN_TYPE_INT64, alignment);
    d.set_array(ACDNN_ATTR_TENSOR_DIMENSIONS, ACDNN_TYPE_INT64,
                static_cast<std::int64_t>(t.dimensions.size()),
                t.dimensions.data());
    d.set_array(ACDNN_ATTR_TENSOR_STRIDES, ACDNN_TYPE_INT64,
                static_cast<std::int64_t>(t.strides.size()), t.strides.data());
    d.finalize();
    return d.get();
  }
  template <class T>
  acdnnBackendDescriptor_t constant(std::int64_t uid, T value,
                                    acdnnDataType_t type) {
    constants_.emplace(uid, DeviceBuffer(sizeof(T)));
    check_driver(cuMemcpyHtoD(constants_.at(uid).address(), &value, sizeof(T)),
                 "cuMemcpyHtoD(acDNN scalar)");
    check_driver(cuStreamSynchronize(nullptr), "acDNN scalar ready");
    return tensor({uid, FLAGDNN_DATA_FLOAT32, {1, 1, 1, 1}, {1, 1, 1, 1}},
                  type);
  }
  void build(acdnnBackendDescriptor_t operation) {
    auto& graph = descriptor(ACDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR);
    graph.set(ACDNN_ATTR_OPERATIONGRAPH_OPS, ACDNN_TYPE_BACKEND_DESCRIPTOR,
              operation);
    graph.set(ACDNN_ATTR_OPERATIONGRAPH_HANDLE, ACDNN_TYPE_HANDLE,
              handle_.get());
    graph.finalize();
    auto& heur = descriptor(ACDNN_BACKEND_ENGINEHEUR_DESCRIPTOR);
    heur.set(ACDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
             ACDNN_TYPE_BACKEND_DESCRIPTOR, graph.get());
    heur.set(ACDNN_ATTR_ENGINEHEUR_MODE, ACDNN_TYPE_HEUR_MODE,
             ACDNN_HEUR_MODE_A);
    heur.finalize();
    auto& config = descriptor(ACDNN_BACKEND_ENGINECFG_DESCRIPTOR);
    auto config_handle = config.get();
    std::int64_t count = 0;
    // SDK 1400 reports the total available count even when only one result
    // is requested. Follow the existing MatMul reference contract.
    check_acdnn(acdnnBackendGetAttribute(
                    heur.get(), ACDNN_ATTR_ENGINEHEUR_RESULTS,
                    ACDNN_TYPE_BACKEND_DESCRIPTOR, 0, &count, nullptr),
                "acdnnBackendGetAttribute(extended heuristic count)");
    if (count <= 0)
      throw std::runtime_error(
          "acDNN extended graph has no executable engine configuration");
    count = 0;
    check_acdnn(acdnnBackendGetAttribute(
                    heur.get(), ACDNN_ATTR_ENGINEHEUR_RESULTS,
                    ACDNN_TYPE_BACKEND_DESCRIPTOR, 1, &count, &config_handle),
                "acdnnBackendGetAttribute(extended heuristics)");
    if (count <= 0)
      throw std::runtime_error(
          "acDNN extended graph has no executable engine configuration");
    plan_ = &descriptor(ACDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR);
    plan_->set(ACDNN_ATTR_EXECUTION_PLAN_HANDLE, ACDNN_TYPE_HANDLE,
               handle_.get());
    plan_->set(ACDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
               ACDNN_TYPE_BACKEND_DESCRIPTOR, config_handle);
    plan_->finalize();
    std::int64_t bytes = 0;
    check_acdnn(acdnnBackendGetAttribute(
                    plan_->get(), ACDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE,
                    ACDNN_TYPE_INT64, 1, &count, &bytes),
                "acdnnBackendGetAttribute(extended workspace)");
    if (bytes < 0 || count != 1)
      throw std::runtime_error("invalid acDNN extended workspace");
    workspace_ = static_cast<std::size_t>(bytes);
  }

 private:
  AcdnnHandle handle_;
  std::vector<std::unique_ptr<AcdnnDescriptor>> descriptors_;
  std::map<std::int64_t, DeviceBuffer> constants_;
  AcdnnDescriptor* plan_ = nullptr;
  std::size_t workspace_ = 0;
};
}  // namespace flagdnn::validation::thead
#endif
