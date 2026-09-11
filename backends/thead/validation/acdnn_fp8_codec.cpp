// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#include "acdnn_fp8_codec.hpp"
#include "acdnn_reference.hpp"
#include "backend_pointwise_reference.hpp"
#include "numeric_types.hpp"

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <stdexcept>
#include <utility>

namespace flagdnn::validation::thead {
namespace {
using flagdnn::testing::TestExecutable;
using flagdnn::testing::TestTensor;

class Extrema final : public TestExecutable {
 public:
  Extrema(acdnnOpTensorOp_t mode, std::int64_t count,
          std::array<std::int64_t,3> uids) : uids_(uids) {
    if (count <= 0 || count > std::numeric_limits<int>::max()) {
      throw std::invalid_argument("acDNN FP8 codec extent exceeds int32");
    }
    const auto n=static_cast<int>(count);
    tensor_.set(ACDNN_DATA_FLOAT,std::array<int,4>{1,1,1,n},
                std::array<int,4>{n,n,n,1});
    operation_.set(mode,ACDNN_DATA_FLOAT,ACDNN_NOT_PROPAGATE_NAN);
  }
  std::size_t workspace_size() const noexcept override { return 0; }
  void execute(std::span<const flagdnnBinding_t> bindings, void *, std::size_t,
               flagdnnStream_t stream) override {
    std::map<std::int64_t,void*> pointers;
    for (const auto &binding:bindings) pointers.emplace(binding.uid,binding.device_pointer);
    handle_.set_stream(reinterpret_cast<hggcStream_t>(stream));
    const float one=1,zero=0;
    check_acdnn(acdnnOpTensor(handle_.get(),operation_.get(),&one,tensor_.get(),
        pointers.at(uids_[0]),&one,tensor_.get(),pointers.at(uids_[1]),&zero,
        tensor_.get(),pointers.at(uids_[2])),"acdnnOpTensor(FP8 codec bound)");
  }
 private:
  std::array<std::int64_t,3> uids_;
  AcdnnHandle handle_;
  AcdnnTensorDescriptor tensor_;
  AcdnnOpTensorDescriptor operation_;
};

class Codec final : public TestExecutable {
 public:
  Codec(const TestTensor &input,const TestTensor &output)
      : input_(input),output_(output) {
    const auto fp8=[](flagdnnDataType_t type) {
      return type==FLAGDNN_DATA_FP8_E4M3 || type==FLAGDNN_DATA_FP8_E5M2;
    };
    if (input.uid<=0 || output.uid<=0 || input.uid==output.uid ||
        input.dimensions.empty() || input.dimensions!=output.dimensions ||
        !((fp8(input.data_type) && output.data_type==FLAGDNN_DATA_FLOAT32) ||
          (input.data_type==FLAGDNN_DATA_FLOAT32 && fp8(output.data_type)))) {
      throw std::invalid_argument("acDNN FP8 codec requires one FP8 and one FP32 tensor");
    }
    std::int64_t dense=1;
    if (input.dimensions.size()!=input.strides.size() ||
        output.dimensions.size()!=output.strides.size()) {
      throw std::invalid_argument("acDNN FP8 codec stride rank mismatch");
    }
    for (std::size_t axis=input.dimensions.size();axis!=0;--axis) {
      const auto extent=input.dimensions[axis-1];
      if (extent<=0 || input.strides[axis-1]!=dense || output.strides[axis-1]!=dense ||
          dense>std::numeric_limits<int>::max()/extent) {
        throw std::invalid_argument("acDNN FP8 codec requires bounded dense tensors");
      }
      dense*=extent;
    }
    count_=dense;next_uid_=std::max(input.uid,output.uid);
    const auto dtype=fp8(input.data_type)?input.data_type:output.data_type;
    const int mantissa=dtype==FLAGDNN_DATA_FP8_E4M3?3:2;
    const int bias=dtype==FLAGDNN_DATA_FP8_E4M3?7:15;
    const float units=static_cast<float>(1<<mantissa);
    const auto zero=constant(0),one=constant(1),two=constant(2);
    TestTensor result;
    if (fp8(input.data_type)) {
      // IDENTITY yields signed storage bytes, not FP8 numeric values.
      auto raw=unary(ACDNN_POINTWISE_IDENTITY_FWD,input,true);
      auto sign=compare(ACDNN_POINTWISE_CMP_LT,raw,zero);
      auto magnitude=add(raw,mul(sign,constant(128)));
      auto exponent=unary(ACDNN_POINTWISE_FLOOR,div(magnitude,constant(units)));
      auto fraction=div(sub(magnitude,mul(exponent,constant(units))),constant(units));
      auto normal=compare(ACDNN_POINTWISE_CMP_GT,exponent,zero);
      auto effective_exponent=extrema(ACDNN_OP_TENSOR_MAX,exponent,one);
      auto power=binary(ACDNN_POINTWISE_POW,two,sub(effective_exponent,constant(static_cast<float>(bias))));
      result=mul(mul(add(normal,fraction),power),sub(one,mul(two,sign)));
    } else {
      // The reciprocal distinguishes -0 as well as negative finite values.
      auto sign=compare(ACDNN_POINTWISE_CMP_LT,div(one,input),zero);
      auto magnitude=unary(ACDNN_POINTWISE_ABS,input);
      magnitude=extrema(ACDNN_OP_TENSOR_MIN,magnitude,
                        constant(dtype==FLAGDNN_DATA_FP8_E4M3?448.0F:57344.0F));
      auto logarithm=mul(unary(ACDNN_POINTWISE_LOG,magnitude),constant(1.4426950408889634F));
      auto exponent=extrema(ACDNN_OP_TENSOR_MAX,unary(ACDNN_POINTWISE_FLOOR,logarithm),
                            constant(static_cast<float>(1-bias)));
      auto step=binary(ACDNN_POINTWISE_POW,two,sub(exponent,constant(static_cast<float>(mantissa))));
      auto scaled=div(magnitude,step);
      auto lower=unary(ACDNN_POINTWISE_FLOOR,scaled);
      auto fraction=sub(scaled,lower);
      auto greater=compare(ACDNN_POINTWISE_CMP_GT,fraction,constant(0.5F));
      auto tie=compare(ACDNN_POINTWISE_CMP_EQ,fraction,constant(0.5F));
      auto odd=sub(lower,mul(two,unary(ACDNN_POINTWISE_FLOOR,div(lower,two))));
      auto rounded=add(add(lower,greater),mul(tie,odd));
      auto code=add(mul(add(exponent,constant(static_cast<float>(bias-1))),constant(units)),rounded);
      result=add(code,mul(sign,constant(128)));
    }
    copy(result,output,fp8(output.data_type));
  }
  std::size_t workspace_size() const noexcept override { return workspace_; }
  void execute(std::span<const flagdnnBinding_t> bindings,void *workspace,
               std::size_t size,flagdnnStream_t stream) override {
    if (bindings.size()!=2 || size<workspace_ || (workspace_ && !workspace)) {
      throw std::invalid_argument("acDNN FP8 codec binding/workspace mismatch");
    }
    std::map<std::int64_t,void*> pointers;
    for (const auto &binding:bindings) {
      if ((binding.uid!=input_.uid && binding.uid!=output_.uid) || !binding.device_pointer ||
          !pointers.emplace(binding.uid,binding.device_pointer).second) {
        throw std::invalid_argument("acDNN FP8 codec invalid pointer");
      }
    }
    for (auto &[uid,buffer]:buffers_) pointers.emplace(uid,buffer.data());
    for (const auto &[uid,source]:aliases_) pointers.emplace(uid,pointers.at(source));
    for (auto &step:steps_) {
      std::vector<flagdnnBinding_t> arguments;
      for (auto uid:step.uids) arguments.push_back({uid,pointers.at(uid)});
      const auto bytes=step.executable->workspace_size();
      step.executable->prepare(arguments,stream);
      step.executable->execute(arguments,bytes?workspace:nullptr,bytes,stream);
    }
  }
 private:
  std::int64_t uid() {
    if (next_uid_==std::numeric_limits<std::int64_t>::max()) throw std::overflow_error("FP8 codec UID overflows");
    return ++next_uid_;
  }
  TestTensor tensor(flagdnnDataType_t type=FLAGDNN_DATA_FLOAT32) {
    TestTensor result{uid(),type,{1,1,count_},{count_,count_,1}};
    buffers_.emplace(result.uid,DeviceBuffer(static_cast<std::size_t>(count_)*element_size(type)));
    return result;
  }
  TestTensor alias(const TestTensor &source) {
    TestTensor result{uid(),source.data_type,{1,1,count_},{count_,count_,1},source.binding_byte_offset};
    aliases_.emplace_back(result.uid,source.uid);return result;
  }
  TestTensor constant(float value) {
    auto result=tensor();check_driver(cuMemsetD32(buffers_.at(result.uid).address(),
        std::bit_cast<unsigned int>(value),static_cast<std::size_t>(count_)),"FP8 codec constant");return result;
  }
  void append(std::unique_ptr<TestExecutable> executable,std::vector<std::int64_t> uids) {
    workspace_=std::max(workspace_,executable->workspace_size());
    steps_.push_back({std::move(executable),std::move(uids)});
  }
  void pointwise(acdnnPointwiseMode_t mode,std::vector<TestTensor> inputs,
                 const TestTensor &output,bool raw=false) {
    std::vector<std::int64_t> uids;
    for (auto &input:inputs) {input=alias(input);uids.push_back(input.uid);}
    auto result=alias(output);uids.push_back(result.uid);
    BackendPointwiseReferenceSpecification specification;
    specification.mode=mode;specification.inputs=std::move(inputs);specification.output=result;
    specification.primitive="acdnnBackendExecute(FP8 codec node)";specification.fp8_storage_bytes=raw;
    append(make_acdnn_backend_pointwise_reference(std::move(specification)),std::move(uids));
  }
  void copy(const TestTensor &input,const TestTensor &output,bool raw=false) {
    pointwise(ACDNN_POINTWISE_IDENTITY_FWD,{input},output,raw);
  }
  TestTensor unary(acdnnPointwiseMode_t mode,const TestTensor &input,bool raw=false) {
    auto result=tensor();pointwise(mode,{input},result,raw);return result;
  }
  TestTensor binary(acdnnPointwiseMode_t mode,const TestTensor &left,const TestTensor &right) {
    auto result=tensor();pointwise(mode,{left,right},result);return result;
  }
  TestTensor compare(acdnnPointwiseMode_t mode,const TestTensor &left,const TestTensor &right) {
    auto boolean=tensor(FLAGDNN_DATA_BOOLEAN);pointwise(mode,{left,right},boolean);
    return unary(ACDNN_POINTWISE_IDENTITY_FWD,boolean);
  }
  TestTensor extrema(acdnnOpTensorOp_t mode,const TestTensor &left,const TestTensor &right) {
    auto a=alias(left),b=alias(right),result=tensor();
    append(std::make_unique<Extrema>(mode,count_,std::array<std::int64_t,3>{a.uid,b.uid,result.uid}),
           {a.uid,b.uid,result.uid});return result;
  }
  TestTensor add(const TestTensor &a,const TestTensor &b) {return binary(ACDNN_POINTWISE_ADD,a,b);}
  TestTensor sub(const TestTensor &a,const TestTensor &b) {return binary(ACDNN_POINTWISE_SUB,a,b);}
  TestTensor mul(const TestTensor &a,const TestTensor &b) {return binary(ACDNN_POINTWISE_MUL,a,b);}
  TestTensor div(const TestTensor &a,const TestTensor &b) {return binary(ACDNN_POINTWISE_DIV,a,b);}
  struct Step {std::unique_ptr<TestExecutable> executable;std::vector<std::int64_t> uids;};
  TestTensor input_,output_;
  std::int64_t count_=0,next_uid_=0;
  std::size_t workspace_=0;
  std::map<std::int64_t,DeviceBuffer> buffers_;
  std::vector<std::pair<std::int64_t,std::int64_t>> aliases_;
  std::vector<Step> steps_;
};
}
std::unique_ptr<flagdnn::testing::TestExecutable> make_acdnn_fp8_codec(
    const TestTensor &input,const TestTensor &output) {
  return std::make_unique<Codec>(input,output);
}
}
