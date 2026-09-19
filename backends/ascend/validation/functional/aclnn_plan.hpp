/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include "common/common.hpp"
#include "validation/acl_runtime.hpp"
#include "validation/tensor_io.hpp"
#include <aclnn/acl_meta.h>
#include <aclnn/aclnn_base.h>
#include <aclnnop/aclnn_copy.h>
#include <algorithm>
#include <functional>
#include <memory>
#include <source_location>
#include <unordered_map>
#include <vector>

namespace flagdnn::testing::ascend {
namespace acl = flagdnn::validation::ascend;
namespace io = acl::tensor_io;
struct Unsupported : std::runtime_error {
  using std::runtime_error::runtime_error;
};
inline aclDataType dtype(flagdnnDataType_t type) {
  switch (type) {
  case FLAGDNN_DATA_FLOAT32:
    return ACL_FLOAT;
  case FLAGDNN_DATA_FLOAT16:
    return ACL_FLOAT16;
  case FLAGDNN_DATA_BFLOAT16:
    return ACL_BF16;
  case FLAGDNN_DATA_INT32:
    return ACL_INT32;
  case FLAGDNN_DATA_BOOLEAN:
    return ACL_BOOL;
  default:
    throw Unsupported("Ascend 910B ACLNN does not support FP8 storage");
  }
}
inline void status(aclnnStatus code, const char *operation) {
  if (code == 0)
    return;
  const auto message = acl::acl_error_message(code, operation);
  if (code == ACL_ERROR_UNSUPPORTED_DATA_TYPE ||
      code == ACL_ERROR_OP_UNSUPPORTED_DYNAMIC ||
      code == ACL_ERROR_API_NOT_SUPPORT ||
      code == ACL_ERROR_FEATURE_UNSUPPORTED)
    throw Unsupported(message);
  throw std::runtime_error(message);
}
inline std::vector<std::int64_t>
dense_strides(const std::vector<std::int64_t> &shape) {
  std::vector<std::int64_t> strides(shape.size(), 1);
  for (std::size_t i = shape.size(); i > 1; --i)
    strides[i - 2] = strides[i - 1] * shape[i - 1];
  return strides;
}
class Plan final : public TestExecutable {
public:
  using Launch = aclnnStatus (*)(void *, std::uint64_t, aclOpExecutor *,
                                 aclrtStream);
  using Builder = std::function<void(Plan &)>;
  Plan(std::vector<TestTensor> tensors, Builder builder)
      : specs(std::move(tensors)), builder_(std::move(builder)) {}
  ~Plan() override {
    if (stream_)
      (void)aclrtSynchronizeStream(stream_);
    for (auto &op : ops_)
      if (op.executor)
        (void)aclDestroyAclOpExecutor(op.executor);
    for (auto *a : lists_)
      (void)aclDestroyTensorList(a);
    for (auto *a : tensors_)
      (void)aclDestroyTensor(a);
    for (auto *a : scalars_)
      (void)aclDestroyScalar(a);
    for (auto *a : arrays_)
      (void)aclDestroyIntArray(a);
    for (auto *a : bools_)
      (void)aclDestroyBoolArray(a);
  }
  void prepare(std::span<const flagdnnBinding_t> bindings,
               flagdnnStream_t stream) override {
    if (stream_)
      throw std::logic_error("ACLNN plan already prepared");
    stream_ = reinterpret_cast<aclrtStream>(stream);
    for (const auto &spec : specs) {
      auto it = std::find_if(bindings.begin(), bindings.end(),
                             [&](const auto &b) { return b.uid == spec.uid; });
      if (it == bindings.end())
        throw std::invalid_argument("ACLNN plan binding missing");
      pointers_.push_back(it->device_pointer);
      ports.push_back(tensor(spec, it->device_pointer));
    }
    builder_(*this);
  }
  std::size_t workspace_size() const noexcept override { return workspace_; }
  void execute(std::span<const flagdnnBinding_t> bindings, void *workspace,
               std::size_t bytes, flagdnnStream_t stream) override {
    if (!stream_ || reinterpret_cast<aclrtStream>(stream) != stream_ ||
        bytes < workspace_)
      throw std::invalid_argument("ACLNN plan launch state mismatch");
    for (std::size_t i = 0; i < specs.size(); ++i) {
      auto it =
          std::find_if(bindings.begin(), bindings.end(),
                       [&](const auto &b) { return b.uid == specs[i].uid; });
      if (it == bindings.end() || it->device_pointer != pointers_[i])
        throw std::invalid_argument("ACLNN plan binding changed");
    }
    for (auto &op : ops_) {
      if (op.launch)
        status(op.launch(op.bytes ? workspace : nullptr, op.bytes, op.executor,
                         stream_),
               "ACLNN execute");
      else
        acl::check_acl(aclrtMemcpyAsync(op.destination, op.bytes, op.source,
                                        op.bytes, ACL_MEMCPY_DEVICE_TO_DEVICE,
                                        stream_),
                       "ACLNN reference dense copy");
    }
  }
  template <class Query>
  void add(Launch launch, Query query,
           std::source_location location = std::source_location::current()) {
    Op op{launch, nullptr, 0, nullptr, nullptr};
    const auto code = query(&op.bytes, &op.executor);
    if (code) {
      if (op.executor)
        (void)aclDestroyAclOpExecutor(op.executor);
      status(code, "ACLNN GetWorkspaceSize");
    }
    ops_.push_back(op);
    status(aclSetAclOpExecutorRepeatable(op.executor),
           (std::string("aclSetAclOpExecutorRepeatable at ") +
            location.file_name() + ":" + std::to_string(location.line()))
               .c_str());
    workspace_ = std::max<std::size_t>(workspace_, op.bytes);
  }
  aclTensor *tensor(const TestTensor &s, void *pointer,
                    aclFormat format = ACL_FORMAT_ND) {
    const std::int64_t storage = io::storage_element_count(s);
    auto *t = aclCreateTensor(s.dimensions.data(), s.dimensions.size(),
                              dtype(s.data_type), s.strides.data(), 0, format,
                              &storage, 1, pointer);
    if (!t)
      throw std::runtime_error("aclCreateTensor failed");
    tensors_.push_back(t);
    memory_.emplace(t, Memory{s, pointer});
    return t;
  }
  void copy(aclTensor *source, aclTensor *destination) {
    const auto &src = memory_.at(source);
    const auto &dst = memory_.at(destination);
    if (src.spec.data_type != dst.spec.data_type ||
        src.spec.dimensions != dst.spec.dimensions)
      throw std::invalid_argument("native copy shape or dtype mismatch");
    auto contiguous = [](const TestTensor &t) {
      const auto strides = dense_strides(t.dimensions);
      for (std::size_t i = 0; i < strides.size(); ++i)
        if (t.dimensions[i] != 1 && t.strides[i] != strides[i])
          return false;
      return true;
    };
    if (contiguous(src.spec) && contiguous(dst.spec)) {
      ops_.push_back({nullptr, nullptr, io::encoded_byte_count(src.spec),
                      src.pointer, dst.pointer});
    } else
      p_copy(source, destination);
  }
  void p_copy(aclTensor *source, aclTensor *destination) {
    add(aclnnInplaceCopy, [&](auto *w, auto **e) {
      return aclnnInplaceCopyGetWorkspaceSize(destination, source, w, e);
    });
  }
  void initialize(aclTensor *t, std::span<const std::uint8_t> data) {
    const auto memory = memory_.at(t);
    if (data.size() != io::encoded_byte_count(memory.spec))
      throw std::invalid_argument("ACLNN initialization size mismatch");
    acl::check_acl(aclrtMemcpyAsync(memory.pointer, data.size(), data.data(),
                                    data.size(), ACL_MEMCPY_HOST_TO_DEVICE,
                                    stream_),
                   "ACLNN initialize");
    acl::check_acl(aclrtSynchronizeStream(stream_),
                   "ACLNN initialize synchronize");
  }
  void run_initialization() {
    acl::DeviceBuffer workspace(workspace_);
    for (auto &op : ops_) {
      if (op.launch)
        status(op.launch(op.bytes ? workspace.opaque() : nullptr, op.bytes,
                         op.executor, stream_),
               "ACLNN reference initialization");
      else
        acl::check_acl(aclrtMemcpyAsync(op.destination, op.bytes, op.source,
                                        op.bytes, ACL_MEMCPY_DEVICE_TO_DEVICE,
                                        stream_),
                       "ACLNN reference initialization copy");
    }
    acl::check_acl(aclrtSynchronizeStream(stream_),
                   "ACLNN reference initialization synchronize");
    for (auto &op : ops_)
      if (op.executor)
        (void)aclDestroyAclOpExecutor(op.executor);
    ops_.clear();
    workspace_ = 0;
  }
  aclTensor *permuted(aclTensor *source,
                      const std::vector<std::int64_t> &axes) {
    auto memory = memory_.at(source);
    auto spec = memory.spec;
    for (std::size_t i = 0; i < axes.size(); ++i) {
      spec.dimensions[i] = memory.spec.dimensions[axes[i]];
      spec.strides[i] = memory.spec.strides[axes[i]];
    }
    return tensor(spec, memory.pointer);
  }
  aclTensor *formatted(aclTensor *source, aclFormat format) {
    auto memory = memory_.at(source);
    return tensor(memory.spec, memory.pointer, format);
  }
  aclTensor *alias(aclTensor *source, std::vector<std::int64_t> shape,
                   std::vector<std::int64_t> strides, std::size_t offset = 0) {
    const auto memory = memory_.at(source);
    auto spec = memory.spec;
    spec.dimensions = std::move(shape);
    spec.strides = std::move(strides);
    if (offset + io::storage_element_count(spec) >
        io::storage_element_count(memory.spec))
      throw std::invalid_argument("ACLNN alias exceeds storage");
    return tensor(spec, static_cast<std::uint8_t *>(memory.pointer) +
                            offset * io::data_type_size(spec.data_type));
  }
  aclBoolArray *output_mask(bool x, bool w, bool b) {
    const bool flags[3] = {x, w, b};
    auto *a = aclCreateBoolArray(flags, 3);
    if (!a)
      throw std::runtime_error("aclCreateBoolArray failed");
    bools_.push_back(a);
    return a;
  }
  aclTensor *view(std::size_t i, std::vector<std::int64_t> shape) {
    auto spec = specs.at(i);
    spec.dimensions = std::move(shape);
    spec.strides = dense_strides(spec.dimensions);
    if (io::storage_element_count(spec) >
        io::storage_element_count(specs.at(i)))
      throw std::invalid_argument("ACLNN view exceeds storage");
    return tensor(spec, pointers_.at(i));
  }
  aclTensor *temporary(TestTensor spec) {
    spec.strides = dense_strides(spec.dimensions);
    scratch_.push_back(
        std::make_unique<acl::DeviceBuffer>(io::encoded_byte_count(spec)));
    return tensor(spec, scratch_.back()->opaque());
  }
  aclScalar *integer_scalar(std::int64_t value) {
    auto *s = aclCreateScalar(&value, ACL_INT64);
    if (!s)
      throw std::runtime_error("aclCreateScalar failed");
    scalars_.push_back(s);
    return s;
  }
  aclScalar *scalar(double value) {
    auto *s = aclCreateScalar(&value, ACL_DOUBLE);
    if (!s)
      throw std::runtime_error("aclCreateScalar failed");
    scalars_.push_back(s);
    return s;
  }
  aclIntArray *array(const std::vector<std::int64_t> &values) {
    auto *a = aclCreateIntArray(values.data(), values.size());
    if (!a)
      throw std::runtime_error("aclCreateIntArray failed");
    arrays_.push_back(a);
    return a;
  }
  aclBoolArray *all_outputs() {
    const bool flags[3] = {true, true, true};
    auto *a = aclCreateBoolArray(flags, 3);
    if (!a)
      throw std::runtime_error("aclCreateBoolArray failed");
    bools_.push_back(a);
    return a;
  }
  aclTensorList *list(const std::vector<aclTensor *> &values) {
    // aclDestroyTensorList also destroys its member descriptors. Give each
    // list private descriptor copies; graph ports and intermediates stay owned
    // by this plan and may occur in more than one list.
    std::vector<aclTensor *> members;
    for (auto *value : values) {
      const auto memory = memory_.at(value);
      members.push_back(tensor(memory.spec, memory.pointer));
    }
    auto *result = aclCreateTensorList(members.data(), members.size());
    if (!result)
      throw std::runtime_error("aclCreateTensorList failed");
    for (auto *member : members) {
      std::erase(tensors_, member);
      memory_.erase(member);
    }
    lists_.push_back(result);
    return result;
  }
  std::vector<TestTensor> specs;
  std::vector<aclTensor *> ports;

private:
  struct Op {
    Launch launch;
    aclOpExecutor *executor;
    std::uint64_t bytes;
    void *source;
    void *destination;
  };
  struct Memory {
    TestTensor spec;
    void *pointer;
  };
  std::unordered_map<aclTensor *, Memory> memory_;
  Builder builder_;
  std::vector<Op> ops_;
  std::size_t workspace_ = 0;
  aclrtStream stream_ = nullptr;
  std::vector<void *> pointers_;
  std::vector<aclTensor *> tensors_;
  std::vector<aclScalar *> scalars_;
  std::vector<aclIntArray *> arrays_;
  std::vector<aclBoolArray *> bools_;
  std::vector<aclTensorList *> lists_;
  std::vector<std::unique_ptr<acl::DeviceBuffer>> scratch_;
};
} // namespace flagdnn::testing::ascend
