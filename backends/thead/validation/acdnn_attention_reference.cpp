// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#include "acdnn_attention_reference.hpp"
#include "acdnn_fp8_codec.hpp"
#include "acdnn_matmul_reference.hpp"
#include "acdnn_reference.hpp"
#include "backend_pointwise_reference.hpp"
#include "numeric_types.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <utility>

namespace flagdnn::validation::thead {
namespace {
using flagdnn::testing::TestExecutable;
using flagdnn::testing::TestTensor;
using flagdnn::testing::SdpaTestCase;
using flagdnn::testing::SdpaBackwardTestCase;

std::size_t count(const TestTensor &tensor) {
  std::size_t result = 1;
  for (auto extent : tensor.dimensions) {
    if (extent <= 0 || static_cast<std::uint64_t>(extent) >
        std::numeric_limits<std::size_t>::max() / result) {
      throw std::overflow_error("acDNN attention tensor size overflows");
    }
    result *= static_cast<std::size_t>(extent);
  }
  return result;
}
std::vector<std::int64_t> strides(const std::vector<std::int64_t> &dimensions) {
  std::vector<std::int64_t> result(dimensions.size());
  std::int64_t stride = 1;
  for (std::size_t axis = dimensions.size(); axis != 0; --axis) {
    if (dimensions[axis - 1] <= 0 || stride >
        std::numeric_limits<std::int64_t>::max() / dimensions[axis - 1]) {
      throw std::overflow_error("acDNN attention stride overflows");
    }
    result[axis - 1] = stride;
    stride *= dimensions[axis - 1];
  }
  return result;
}

class Softmax final : public TestExecutable {
 public:
  Softmax(std::int64_t rows, std::int64_t columns, std::vector<std::int64_t> uids,
          bool backward = false, bool logarithmic = false)
      : uids_(std::move(uids)), backward_(backward), logarithmic_(logarithmic) {
    if (rows <= 0 || columns <= 0 || rows > std::numeric_limits<int>::max() ||
        columns > std::numeric_limits<int>::max()) {
      throw std::invalid_argument("acDNN attention softmax dimensions exceed int32");
    }
    auto m = static_cast<int>(rows), n = static_cast<int>(columns);
    tensor_.set(ACDNN_DATA_FLOAT, std::array<int, 4>{m,n,1,1},
                 std::array<int, 4>{n,1,1,1});
  }
  std::size_t workspace_size() const noexcept override { return 0; }
  void execute(std::span<const flagdnnBinding_t> bindings, void *, std::size_t,
               flagdnnStream_t stream) override {
    std::map<std::int64_t, void *> pointers;
    for (const auto &binding : bindings) pointers.emplace(binding.uid, binding.device_pointer);
    const float one = 1, zero = 0;
    handle_.set_stream(reinterpret_cast<hggcStream_t>(stream));
    if (backward_) {
      check_acdnn(acdnnSoftmaxBackward(handle_.get(), ACDNN_SOFTMAX_ACCURATE,
          ACDNN_SOFTMAX_MODE_CHANNEL, &one, tensor_.get(), pointers.at(uids_[0]),
          tensor_.get(), pointers.at(uids_[1]), &zero, tensor_.get(), pointers.at(uids_[2])),
          "acdnnSoftmaxBackward(attention DAG)");
    } else {
      check_acdnn(acdnnSoftmaxForward(handle_.get(),
          logarithmic_ ? ACDNN_SOFTMAX_LOG : ACDNN_SOFTMAX_ACCURATE,
          ACDNN_SOFTMAX_MODE_CHANNEL, &one, tensor_.get(), pointers.at(uids_[0]),
          &zero, tensor_.get(), pointers.at(uids_[1])),
          "acdnnSoftmaxForward(attention DAG)");
    }
  }
 private:
  std::vector<std::int64_t> uids_;
  bool backward_, logarithmic_;
  AcdnnHandle handle_;
  AcdnnTensorDescriptor tensor_;
};

class FirstColumn final : public TestExecutable {
 public:
  FirstColumn(std::int64_t rows, std::int64_t columns,
               std::int64_t input, std::int64_t output)
      : rows_(rows), columns_(columns), input_uid_(input), output_uid_(output) {
    tensor_.set(ACDNN_DATA_FLOAT, std::array<int,3>{1,1,1}, std::array<int,3>{1,1,1});
  }
  std::size_t workspace_size() const noexcept override { return 0; }
  void execute(std::span<const flagdnnBinding_t> bindings, void *, std::size_t,
               flagdnnStream_t stream) override {
    std::map<std::int64_t, void *> pointers;
    for (const auto &binding : bindings) pointers.emplace(binding.uid, binding.device_pointer);
    auto *input = static_cast<float *>(pointers.at(input_uid_));
    auto *output = static_cast<float *>(pointers.at(output_uid_));
    handle_.set_stream(reinterpret_cast<hggcStream_t>(stream));
    const float one = 1, zero = 0;
    for (std::int64_t row = 0; row < rows_; ++row) {
      check_acdnn(acdnnTransformTensor(handle_.get(), &one, tensor_.get(), input + row*columns_,
          &zero, tensor_.get(), output + row), "acdnnTransformTensor(attention statistics column)");
    }
  }
 private:
  std::int64_t rows_, columns_, input_uid_, output_uid_;
  AcdnnHandle handle_;
  AcdnnTensorDescriptor tensor_;
};

void floating_descriptor(AcdnnTensorDescriptor &descriptor,const TestTensor &tensor) {
  std::vector<int> dims,layout;
  if (tensor.dimensions.size()>4 || tensor.data_type!=FLAGDNN_DATA_FLOAT32) {
    throw std::invalid_argument("acDNN attention floating primitive requires rank at most four");
  }
  const auto elements=count(tensor);
  if (elements>static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::overflow_error("acDNN attention primitive extent exceeds int32");
  }
  while (dims.size()+tensor.dimensions.size()<4) {
    dims.push_back(1);layout.push_back(static_cast<int>(elements));
  }
  for (std::size_t axis=0;axis<tensor.dimensions.size();++axis) {
    if (tensor.dimensions[axis]<=0 || tensor.strides[axis]<=0 ||
        tensor.dimensions[axis]>std::numeric_limits<int>::max() ||
        tensor.strides[axis]>std::numeric_limits<int>::max()) {
      throw std::invalid_argument("acDNN attention primitive metadata exceeds int32");
    }
    dims.push_back(static_cast<int>(tensor.dimensions[axis]));
    layout.push_back(static_cast<int>(tensor.strides[axis]));
  }
  descriptor.set(ACDNN_DATA_FLOAT,dims,layout);
}

class FloatingOp final : public TestExecutable {
 public:
  FloatingOp(acdnnOpTensorOp_t operation,const TestTensor &a,
             const TestTensor &b,const TestTensor &output) : uids_{a.uid,b.uid,output.uid} {
    floating_descriptor(tensors_[0],a);floating_descriptor(tensors_[1],b);
    floating_descriptor(tensors_[2],output);
    operation_.set(operation,ACDNN_DATA_FLOAT,ACDNN_NOT_PROPAGATE_NAN);
  }
  std::size_t workspace_size() const noexcept override {return 0;}
  void execute(std::span<const flagdnnBinding_t> bindings,void *,std::size_t,
               flagdnnStream_t stream) override {
    std::map<std::int64_t,void*> pointers;
    for (const auto &binding:bindings) pointers.emplace(binding.uid,binding.device_pointer);
    handle_.set_stream(reinterpret_cast<hggcStream_t>(stream));
    const float one=1,zero=0;
    check_acdnn(acdnnOpTensor(handle_.get(),operation_.get(),&one,tensors_[0].get(),
        pointers.at(uids_[0]),&one,tensors_[1].get(),pointers.at(uids_[1]),&zero,
        tensors_[2].get(),pointers.at(uids_[2])),"acdnnOpTensor(attention broadcast/extrema)");
  }
 private:
  std::array<std::int64_t,3> uids_;
  AcdnnHandle handle_;
  AcdnnOpTensorDescriptor operation_;
  std::array<AcdnnTensorDescriptor,3> tensors_;
};

class FloatingReduction final : public TestExecutable {
 public:
  FloatingReduction(acdnnReduceTensorOp_t operation,const TestTensor &input,
                    const TestTensor &output) : uids_{input.uid,output.uid} {
    floating_descriptor(input_,input);floating_descriptor(output_,output);
    check_acdnn(acdnnCreateReduceTensorDescriptor(&reduction_),"acdnnCreateReduceTensorDescriptor(attention)");
    try {
      check_acdnn(acdnnSetReduceTensorDescriptor(reduction_,operation,ACDNN_DATA_FLOAT,
          ACDNN_NOT_PROPAGATE_NAN,ACDNN_REDUCE_TENSOR_NO_INDICES,ACDNN_32BIT_INDICES),
          "acdnnSetReduceTensorDescriptor(attention)");
      check_acdnn(acdnnGetReductionWorkspaceSize(handle_.get(),reduction_,input_.get(),output_.get(),&workspace_),
          "acdnnGetReductionWorkspaceSize(attention)");
    } catch (...) {acdnnDestroyReduceTensorDescriptor(reduction_);reduction_=nullptr;throw;}
  }
  ~FloatingReduction() override {if (reduction_)acdnnDestroyReduceTensorDescriptor(reduction_);}
  std::size_t workspace_size() const noexcept override {return std::max<std::size_t>(workspace_,256);}
  void execute(std::span<const flagdnnBinding_t> bindings,void *workspace,std::size_t size,
               flagdnnStream_t stream) override {
    if (!workspace || size<workspace_size())throw std::invalid_argument("acDNN attention reduction requires scratch");
    std::map<std::int64_t,void*> pointers;
    for (const auto &binding:bindings) pointers.emplace(binding.uid,binding.device_pointer);
    handle_.set_stream(reinterpret_cast<hggcStream_t>(stream));
    const float one=1,zero=0;
    check_acdnn(acdnnReduceTensor(handle_.get(),reduction_,workspace,0,workspace,workspace_,
        &one,input_.get(),pointers.at(uids_[0]),&zero,output_.get(),pointers.at(uids_[1])),
        "acdnnReduceTensor(attention rows/amax)");
  }
 private:
  std::array<std::int64_t,2> uids_;
  AcdnnHandle handle_;
  AcdnnTensorDescriptor input_,output_;
  acdnnReduceTensorDescriptor_t reduction_=nullptr;
  std::size_t workspace_=0;
};

class AttentionDag final : public TestExecutable {
 public:
  AttentionDag(const SdpaTestCase &test_case, const CapabilityRecord &capability)
      : capability_(capability) {
    flagdnn::testing::validate_sdpa_case(test_case);
    check_plan(false);
    add_external(test_case.q); add_external(test_case.k); add_external(test_case.v);
    if (test_case.bias) add_external(*test_case.bias);
    add_external(test_case.output); if (test_case.stats) add_external(*test_case.stats);
    build(test_case, nullptr);
  }
  AttentionDag(const SdpaBackwardTestCase &test_case, const CapabilityRecord &capability)
      : capability_(capability) {
    flagdnn::testing::validate_sdpa_backward_case(test_case);
    check_plan(true);
    for (const auto &tensor : {test_case.q,test_case.k,test_case.v,test_case.output,
                               test_case.doutput,test_case.stats,test_case.dq,test_case.dk,test_case.dv}) add_external(tensor);
    if (test_case.bias) add_external(*test_case.bias);
    if (test_case.dbias) add_external(*test_case.dbias);
    SdpaTestCase forward;
    forward.q=test_case.q;forward.k=test_case.k;forward.v=test_case.v;
    forward.bias=test_case.bias;forward.output=test_case.output;forward.options=test_case.options;
    build(forward, &test_case);
  }
  AttentionDag(const flagdnn::testing::SdpaFp8TestCase &test_case,const CapabilityRecord &capability)
      : capability_(capability),allow_fp8_(true),fp8_type_(test_case.q.data_type) {
    flagdnn::testing::validate_sdpa_fp8_case(test_case);check_fp8_plan(false);
    for(const auto &value:{test_case.q,test_case.k,test_case.v,test_case.output,test_case.amax_s,test_case.amax_o})add_external(value);
    for(const auto &value:{test_case.descale_q,test_case.descale_k,test_case.descale_v,test_case.descale_s,test_case.scale_s,test_case.scale_o})add_external(value.tensor);
    if(test_case.stats)add_external(*test_case.stats);
    if(test_case.bias)add_external(*test_case.bias);
    build_fp8(test_case);
  }
  AttentionDag(const flagdnn::testing::SdpaFp8BackwardTestCase &test_case,const CapabilityRecord &capability)
      : capability_(capability),allow_fp8_(true),fp8_type_(test_case.q.data_type) {
    flagdnn::testing::validate_sdpa_fp8_backward_case(test_case);check_fp8_plan(true);
    for(const auto &value:{test_case.q,test_case.k,test_case.v,test_case.output,test_case.doutput,test_case.stats,
        test_case.dq,test_case.dk,test_case.dv,test_case.amax_dq,test_case.amax_dk,test_case.amax_dv,test_case.amax_dp})add_external(value);
    for(const auto &value:{test_case.descale_q,test_case.descale_k,test_case.descale_v,test_case.descale_o,test_case.descale_doutput,
        test_case.descale_s,test_case.descale_dp,test_case.scale_s,test_case.scale_dp,test_case.scale_dq,test_case.scale_dk,test_case.scale_dv})add_external(value.tensor);
    build_fp8(test_case);
  }
  std::size_t workspace_size() const noexcept override { return workspace_; }
  void execute(std::span<const flagdnnBinding_t> bindings, void *workspace,
               std::size_t size, flagdnnStream_t stream) override {
    if (size < workspace_ || (workspace_ && !workspace) || bindings.size() != external_.size()) {
      throw std::invalid_argument("acDNN attention workspace or binding count mismatch");
    }
    std::map<std::int64_t, void *> pointers;
    for (const auto &binding : bindings) {
      if (!external_.contains(binding.uid) || !binding.device_pointer ||
          !pointers.emplace(binding.uid,binding.device_pointer).second) {
        throw std::invalid_argument("acDNN attention invalid external binding");
      }
    }
    for (auto &[uid, buffer] : buffers_) pointers.emplace(uid,buffer.data());
    // Aliases are allocated in topological order and carry metadata-only offsets.
    for (const auto &alias : aliases_) {
      pointers.emplace(alias.uid,static_cast<std::byte *>(pointers.at(alias.source)) + alias.offset);
    }
    for (auto &step : steps_) {
      std::vector<flagdnnBinding_t> arguments;
      for (auto uid : step.uids) arguments.push_back({uid,pointers.at(uid)});
      step.executable->prepare(arguments,stream);
      const auto bytes=step.executable->workspace_size();
      step.executable->execute(arguments,bytes ? workspace : nullptr,bytes,stream);
    }
  }
 private:
  void check_plan(bool backward) {
    const auto selection=select_reference(capability_);
    if (!std::holds_alternative<ReferencePlan>(selection) ||
        capability_.path != ReferencePath::kBackendDescriptor ||
        capability_.reference_plan != acdnn_attention_plan(backward)) {
      throw std::invalid_argument("acDNN attention reference plan mismatch");
    }
  }
  void add_external(const TestTensor &tensor) {
    if (tensor.uid <= 0 || tensor.strides != strides(tensor.dimensions) ||
        (tensor.data_type != FLAGDNN_DATA_FLOAT32 && tensor.data_type != FLAGDNN_DATA_FLOAT16 &&
         tensor.data_type != FLAGDNN_DATA_BFLOAT16 &&
         !(allow_fp8_ && (tensor.data_type==FLAGDNN_DATA_FP8_E4M3 || tensor.data_type==FLAGDNN_DATA_FP8_E5M2))) || !external_.insert(tensor.uid).second) {
      throw std::invalid_argument("acDNN attention requires distinct dense floating tensors");
    }
    next_uid_=std::max(next_uid_,tensor.uid);
  }
  std::int64_t uid() {
    if (next_uid_==std::numeric_limits<std::int64_t>::max()) throw std::overflow_error("acDNN attention UID overflow");
    return ++next_uid_;
  }
  TestTensor tensor(std::vector<std::int64_t> dimensions, flagdnnDataType_t dtype=FLAGDNN_DATA_FLOAT32) {
    TestTensor result{uid(),dtype,std::move(dimensions),{}};result.strides=strides(result.dimensions);
    const auto elements=count(result),width=element_size(dtype);
    if (elements>std::numeric_limits<std::size_t>::max()/width) throw std::overflow_error("acDNN attention allocation overflows");
    buffers_.emplace(result.uid,DeviceBuffer(elements*width));
    return result;
  }
  TestTensor view(const TestTensor &source, std::vector<std::int64_t> dimensions,
                  std::vector<std::int64_t> layout, std::size_t offset=0) {
    TestTensor result=source;result.uid=uid();result.dimensions=std::move(dimensions);result.strides=std::move(layout);
    if (offset>std::numeric_limits<std::size_t>::max()-result.binding_byte_offset) throw std::overflow_error("acDNN attention view overflows");
    result.binding_byte_offset+=offset;
    aliases_.push_back({result.uid,source.uid,offset});return result;
  }
  TestTensor flat(const TestTensor &source) {
    const auto elements=static_cast<std::int64_t>(count(source));
    return view(source,{1,1,elements},{elements,elements,1});
  }
  TestTensor head(const TestTensor &source, std::int64_t batch,std::int64_t head_index) {
    const auto rows=source.dimensions[2],columns=source.dimensions[3];
    auto offset=static_cast<std::size_t>(batch*source.strides[0]+head_index*source.strides[1])*element_size(source.data_type);
    return view(source,{1,rows,columns},{rows*columns,columns,1},offset);
  }
  void append(std::unique_ptr<TestExecutable> executable,std::vector<std::int64_t> uids) {
    workspace_=std::max(workspace_,executable->workspace_size());
    steps_.push_back({std::move(executable),std::move(uids)});
  }
  void pointwise(acdnnPointwiseMode_t mode,std::vector<TestTensor> inputs,const TestTensor &output) {
    std::vector<std::int64_t> uids;
    for (auto &input : inputs) {input=flat(input);uids.push_back(input.uid);}
    auto result=flat(output);uids.push_back(result.uid);
    append(make_acdnn_backend_pointwise_reference({.mode=mode,.inputs=std::move(inputs),.output=result,
        .primitive="acdnnBackendExecute(attention pointwise node)"}),std::move(uids));
  }
  TestTensor convert(const TestTensor &input) {
    auto result=tensor(input.dimensions);
    if (input.data_type==FLAGDNN_DATA_FP8_E4M3 || input.data_type==FLAGDNN_DATA_FP8_E5M2) {
      append(make_acdnn_fp8_codec(input,result),{input.uid,result.uid});
    } else {
      pointwise(ACDNN_POINTWISE_IDENTITY_FWD,{input},result);
    }
    return result;
  }
  TestTensor constant(const std::vector<std::int64_t> &shape,float value) {
    auto result=tensor(shape);check_driver(cuMemsetD32(buffers_.at(result.uid).address(),std::bit_cast<unsigned int>(value),count(result)),"acDNN attention constant");return result;
  }
  TestTensor binary(acdnnPointwiseMode_t mode,const TestTensor &a,const TestTensor &b) {
    auto result=tensor(a.dimensions);pointwise(mode,{a,b},result);return result;
  }
  TestTensor scale(const TestTensor &input,float value) {
    return binary(ACDNN_POINTWISE_MUL,input,constant(input.dimensions,value));
  }
  TestTensor transpose(const TestTensor &input) {
    return view(input,{1,input.dimensions[2],input.dimensions[1]},
                      {input.strides[0],input.strides[2],input.strides[1]});
  }
  TestTensor matmul(const TestTensor &a,const TestTensor &b) {
    auto result=tensor({1,a.dimensions[1],b.dimensions[2]});
    auto capability=capability_;capability.reference_plan={"acdnnBackendExecute(MATMUL)"};
    append(make_acdnn_matmul_reference({.name="acDNN attention matrix product",.a=a,.b=b,.output=result},capability),{a.uid,b.uid,result.uid});
    return result;
  }
  TestTensor softmax(const TestTensor &input,bool logarithmic=false) {
    auto result=tensor(input.dimensions);
    append(std::make_unique<Softmax>(input.dimensions[1],input.dimensions[2],
        std::vector<std::int64_t>{input.uid,result.uid},false,logarithmic),{input.uid,result.uid});return result;
  }
  TestTensor softmax_backward(const TestTensor &probabilities,const TestTensor &gradient) {
    auto result=tensor(probabilities.dimensions);
    append(std::make_unique<Softmax>(probabilities.dimensions[1],probabilities.dimensions[2],
        std::vector<std::int64_t>{probabilities.uid,gradient.uid,result.uid},true),{probabilities.uid,gradient.uid,result.uid});return result;
  }
  void accumulate(std::map<std::pair<std::int64_t,std::int64_t>,TestTensor> &sums,
                  std::pair<std::int64_t,std::int64_t> key,const TestTensor &value) {
    auto found=sums.find(key);
    if (found==sums.end()) sums.emplace(key,value);
    else found->second=binary(ACDNN_POINTWISE_ADD,found->second,value);
  }
  void check_fp8_plan(bool backward) {
    if (!std::holds_alternative<ReferencePlan>(select_reference(capability_)) ||
        capability_.path!=ReferencePath::kBackendDescriptor ||
        capability_.reference_plan!=acdnn_fp8_attention_plan(backward)) {
      throw std::invalid_argument("acDNN FP8 attention reference plan mismatch");
    }
  }
  TestTensor floating_op(acdnnOpTensorOp_t mode,const TestTensor &left,const TestTensor &right) {
    auto a=view(left,left.dimensions,left.strides),b=view(right,right.dimensions,right.strides);
    auto output=tensor(left.dimensions);
    append(std::make_unique<FloatingOp>(mode,a,b,output),{a.uid,b.uid,output.uid});return output;
  }
  TestTensor expand(const TestTensor &input,const std::vector<std::int64_t> &shape) {
    if (input.dimensions==shape)return input;
    return floating_op(ACDNN_OP_TENSOR_ADD,constant(shape,0),input);
  }
  TestTensor scalar(const flagdnn::testing::Fp8Scalar &value) {return flat(value.tensor);}
  TestTensor multiply(const TestTensor &input,const TestTensor &factor) {
    return binary(ACDNN_POINTWISE_MUL,input,expand(factor,input.dimensions));
  }
  TestTensor unary(acdnnPointwiseMode_t mode,const TestTensor &input) {
    auto output=tensor(input.dimensions);pointwise(mode,{input},output);return output;
  }
  TestTensor reduce(acdnnReduceTensorOp_t mode,const TestTensor &input,bool rows=false) {
    auto output=tensor(rows?std::vector<std::int64_t>{1,input.dimensions[1],1}:std::vector<std::int64_t>{1,1,1});
    append(std::make_unique<FloatingReduction>(mode,input,output),{input.uid,output.uid});return output;
  }
  void encode_fp8(const TestTensor &input,const TestTensor &output) {
    append(make_acdnn_fp8_codec(input,output),{input.uid,output.uid});
  }
  TestTensor quantized_fp8(const TestTensor &input) {
    auto bytes=tensor(input.dimensions,fp8_type_);encode_fp8(input,bytes);return convert(bytes);
  }
  void maximum(std::optional<TestTensor> &current,const TestTensor &input) {
    auto value=reduce(ACDNN_REDUCE_TENSOR_MAX,unary(ACDNN_POINTWISE_ABS,input));
    current=current?floating_op(ACDNN_OP_TENSOR_MAX,*current,value):value;
  }
  void write_maximum(const std::optional<TestTensor> &value,const TestTensor &output) {
    if (!value)throw std::logic_error("acDNN FP8 attention amax has no producer");
    pointwise(ACDNN_POINTWISE_IDENTITY_FWD,{*value},output);
  }
  TestTensor masked_logits(const TestTensor &input,const flagdnn::testing::AttentionOptions &options) {
    if (!options.diagonal_band_right_bound)return input;
    auto mask=tensor(input.dimensions);std::vector<float> values(count(mask));
    const auto rows=input.dimensions[1],columns=input.dimensions[2];
    for(std::int64_t row=0;row<rows;++row) for(std::int64_t column=0;column<columns;++column) {
      values[static_cast<std::size_t>(row*columns+column)]=column>row?-std::numeric_limits<float>::infinity():0;
    }
    check_driver(cuMemcpyHtoD(buffers_.at(mask.uid).address(),values.data(),values.size()*sizeof(float)),"acDNN FP8 attention geometric mask");
    return binary(ACDNN_POINTWISE_ADD,input,mask);
  }
  template<class Case> void require_fp8_geometry(const Case &test_case) {
    const auto &q=test_case.q,&k=test_case.k,&v=test_case.v;
    if (q.dimensions[0]!=1 || q.dimensions[3]!=128 || v.dimensions[3]!=128 ||
        q.dimensions[1]>4 || q.dimensions[2]>64 || k.dimensions[2]>64 ||
        k.dimensions[1]!=v.dimensions[1] ||
        test_case.options.diagonal_band_left_bound ||
        (test_case.options.diagonal_band_right_bound && *test_case.options.diagonal_band_right_bound!=0) ||
        test_case.options.diagonal_alignment!=flagdnn::testing::AttentionDiagonalAlignment::kTopLeft) {
      throw std::invalid_argument("acDNN FP8 attention geometry is not qualified");
    }
  }
  void build_fp8(const flagdnn::testing::SdpaFp8TestCase &test_case) {
    require_fp8_geometry(test_case);
    if (test_case.bias)throw std::invalid_argument("acDNN FP8 attention bias is not qualified");
    const auto &q=test_case.q,&k=test_case.k,&v=test_case.v;
    const auto qk_scale=scale(binary(ACDNN_POINTWISE_MUL,scalar(test_case.descale_q),scalar(test_case.descale_k)),
        test_case.options.attention_scale.value_or(1.0F/std::sqrt(static_cast<float>(q.dimensions[3]))));
    const auto sv_scale=binary(ACDNN_POINTWISE_MUL,scalar(test_case.descale_s),scalar(test_case.descale_v));
    std::optional<TestTensor> amax_s,amax_o;
    const auto groups=q.dimensions[1]/k.dimensions[1];
    for(std::int64_t b=0;b<q.dimensions[0];++b) for(std::int64_t h=0;h<q.dimensions[1];++h) {
      auto qh=convert(head(q,b,h)),kh=convert(head(k,b,h/groups)),vh=convert(head(v,b,h/groups));
      auto logits=masked_logits(multiply(matmul(qh,transpose(kh)),qk_scale),test_case.options);
      auto row_max=reduce(ACDNN_REDUCE_TENSOR_MAX,logits,true);
      auto probabilities=unary(ACDNN_POINTWISE_EXP,binary(ACDNN_POINTWISE_SUB,logits,expand(row_max,logits.dimensions)));
      auto denominator=reduce(ACDNN_REDUCE_TENSOR_ADD,probabilities,true);
      auto normalized=binary(ACDNN_POINTWISE_DIV,probabilities,expand(denominator,probabilities.dimensions));
      maximum(amax_s,normalized);
      auto p_quantized=quantized_fp8(multiply(probabilities,scalar(test_case.scale_s)));
      auto output=multiply(matmul(p_quantized,vh),sv_scale);
      output=binary(ACDNN_POINTWISE_DIV,output,expand(denominator,output.dimensions));
      maximum(amax_o,output);
      encode_fp8(multiply(output,scalar(test_case.scale_o)),head(test_case.output,b,h));
      if (test_case.stats) {
        auto stats=binary(ACDNN_POINTWISE_ADD,row_max,unary(ACDNN_POINTWISE_LOG,denominator));
        pointwise(ACDNN_POINTWISE_IDENTITY_FWD,{stats},head(*test_case.stats,b,h));
      }
    }
    write_maximum(amax_s,test_case.amax_s);write_maximum(amax_o,test_case.amax_o);
  }
  void build_fp8(const flagdnn::testing::SdpaFp8BackwardTestCase &test_case) {
    require_fp8_geometry(test_case);
    const auto &q=test_case.q,&k=test_case.k,&v=test_case.v;
    const auto attention_scale=test_case.options.attention_scale.value_or(1.0F/std::sqrt(static_cast<float>(q.dimensions[3])));
    const auto qk_scale=scale(binary(ACDNN_POINTWISE_MUL,scalar(test_case.descale_q),scalar(test_case.descale_k)),attention_scale);
    const auto ov_scale=binary(ACDNN_POINTWISE_MUL,scalar(test_case.descale_o),scalar(test_case.descale_doutput));
    const auto do_v_scale=binary(ACDNN_POINTWISE_MUL,scalar(test_case.descale_doutput),scalar(test_case.descale_v));
    const auto dq_scale=binary(ACDNN_POINTWISE_MUL,scalar(test_case.descale_dp),scalar(test_case.descale_k));
    const auto dk_scale=binary(ACDNN_POINTWISE_MUL,scalar(test_case.descale_dp),scalar(test_case.descale_q));
    const auto dv_scale=binary(ACDNN_POINTWISE_MUL,scalar(test_case.descale_s),scalar(test_case.descale_doutput));
    std::optional<TestTensor> amax_dq,amax_dk,amax_dv,amax_dp;
    std::map<std::pair<std::int64_t,std::int64_t>,TestTensor> dk,dv;
    const auto groups=q.dimensions[1]/k.dimensions[1];
    for(std::int64_t b=0;b<q.dimensions[0];++b) for(std::int64_t h=0;h<q.dimensions[1];++h) {
      auto qh=convert(head(q,b,h)),kh=convert(head(k,b,h/groups)),vh=convert(head(v,b,h/groups));
      auto output=convert(head(test_case.output,b,h)),doutput=convert(head(test_case.doutput,b,h));
      auto logits=masked_logits(multiply(matmul(qh,transpose(kh)),qk_scale),test_case.options);
      auto probabilities=unary(ACDNN_POINTWISE_EXP,binary(ACDNN_POINTWISE_SUB,logits,expand(head(test_case.stats,b,h),logits.dimensions)));
      auto dp=multiply(matmul(doutput,transpose(vh)),do_v_scale);
      auto delta=multiply(reduce(ACDNN_REDUCE_TENSOR_ADD,binary(ACDNN_POINTWISE_MUL,output,doutput),true),ov_scale);
      auto ds=scale(binary(ACDNN_POINTWISE_MUL,probabilities,binary(ACDNN_POINTWISE_SUB,dp,expand(delta,dp.dimensions))),attention_scale);
      maximum(amax_dp,ds);
      auto ds_quantized=quantized_fp8(multiply(ds,scalar(test_case.scale_dp)));
      auto p_quantized=quantized_fp8(multiply(probabilities,scalar(test_case.scale_s)));
      auto dq=multiply(matmul(ds_quantized,kh),dq_scale);maximum(amax_dq,dq);
      encode_fp8(multiply(dq,scalar(test_case.scale_dq)),head(test_case.dq,b,h));
      accumulate(dk,{b,h/groups},multiply(matmul(transpose(ds_quantized),qh),dk_scale));
      accumulate(dv,{b,h/groups},multiply(matmul(transpose(p_quantized),doutput),dv_scale));
    }
    for(const auto &[key,value]:dk) {
      maximum(amax_dk,value);encode_fp8(multiply(value,scalar(test_case.scale_dk)),head(test_case.dk,key.first,key.second));
    }
    for(const auto &[key,value]:dv) {
      maximum(amax_dv,value);encode_fp8(multiply(value,scalar(test_case.scale_dv)),head(test_case.dv,key.first,key.second));
    }
    write_maximum(amax_dq,test_case.amax_dq);write_maximum(amax_dk,test_case.amax_dk);
    write_maximum(amax_dv,test_case.amax_dv);write_maximum(amax_dp,test_case.amax_dp);
  }
  void build(const SdpaTestCase &test_case,const SdpaBackwardTestCase *backward) {
    const auto &q=test_case.q,&k=test_case.k,&v=test_case.v;
    if (q.dimensions.size()!=4 || k.dimensions.size()!=4 || v.dimensions.size()!=4 ||
        q.dimensions[0]!=k.dimensions[0] || q.dimensions[0]!=v.dimensions[0] ||
        q.dimensions[1]%k.dimensions[1] || k.dimensions[1]!=v.dimensions[1] ||
        q.dimensions[3]!=k.dimensions[3] || k.dimensions[2]!=v.dimensions[2] ||
        test_case.options.diagonal_band_left_bound ||
        (test_case.options.diagonal_band_right_bound && *test_case.options.diagonal_band_right_bound!=0) ||
        test_case.options.diagonal_alignment!=flagdnn::testing::AttentionDiagonalAlignment::kTopLeft) {
      throw std::invalid_argument("acDNN attention DAG supports dense or top-left causal GQA");
    }
    const float attention_scale=test_case.options.attention_scale.value_or(1.0F/std::sqrt(static_cast<float>(q.dimensions[3])));
    if (!std::isfinite(attention_scale)) throw std::invalid_argument("acDNN attention scale is not finite");
    const auto groups=q.dimensions[1]/k.dimensions[1];
    std::map<std::pair<std::int64_t,std::int64_t>,TestTensor> dk,dv,dbias;
    for (std::int64_t b=0;b<q.dimensions[0];++b) for(std::int64_t h=0;h<q.dimensions[1];++h) {
      auto q_head=convert(head(q,b,h)),k_head=convert(head(k,b,h/groups)),v_head=convert(head(v,b,h/groups));
      auto logits=scale(matmul(q_head,transpose(k_head)),attention_scale);
      std::pair<std::int64_t,std::int64_t> bias_key;
      if (test_case.bias) {
        const auto &bias=*test_case.bias;
        if (bias.dimensions.size()!=4 || (bias.dimensions[0]!=1 && bias.dimensions[0]!=q.dimensions[0]) ||
            (bias.dimensions[1]!=1 && bias.dimensions[1]!=q.dimensions[1]) ||
            bias.dimensions[2]!=q.dimensions[2] || bias.dimensions[3]!=k.dimensions[2]) {
          throw std::invalid_argument("acDNN attention bias broadcast contract mismatch");
        }
        bias_key={bias.dimensions[0]==1?0:b,bias.dimensions[1]==1?0:h};
        logits=binary(ACDNN_POINTWISE_ADD,logits,convert(head(bias,bias_key.first,bias_key.second)));
      }
      if (test_case.options.diagonal_band_right_bound) {
        auto mask=tensor(logits.dimensions);std::vector<float> values(count(mask));
        for(std::int64_t row=0;row<q.dimensions[2];++row) for(std::int64_t column=0;column<k.dimensions[2];++column) {
          values[static_cast<std::size_t>(row*k.dimensions[2]+column)]=column>row ? -std::numeric_limits<float>::infinity() : 0;
        }
        check_driver(cuMemcpyHtoD(buffers_.at(mask.uid).address(),values.data(),values.size()*sizeof(float)),"acDNN attention geometric mask");
        logits=binary(ACDNN_POINTWISE_ADD,logits,mask);
      }
      auto probabilities=softmax(logits);
      if (!backward) {
        auto result=matmul(probabilities,v_head);
        pointwise(ACDNN_POINTWISE_IDENTITY_FWD,{result},head(test_case.output,b,h));
        if(test_case.stats) {
          auto logarithms=softmax(logits,true);
          auto repeated_stats=binary(ACDNN_POINTWISE_SUB,logits,logarithms);
          auto stats=head(*test_case.stats,b,h);
          append(std::make_unique<FirstColumn>(q.dimensions[2],k.dimensions[2],repeated_stats.uid,stats.uid),{repeated_stats.uid,stats.uid});
        }
      } else {
        auto doutput=convert(head(backward->doutput,b,h));
        auto dp=matmul(doutput,transpose(v_head));
        auto ds=softmax_backward(probabilities,dp);
        auto dq_head=scale(matmul(ds,k_head),attention_scale);
        pointwise(ACDNN_POINTWISE_IDENTITY_FWD,{dq_head},head(backward->dq,b,h));
        accumulate(dk,{b,h/groups},scale(matmul(transpose(ds),q_head),attention_scale));
        accumulate(dv,{b,h/groups},matmul(transpose(probabilities),doutput));
        if(backward->dbias) accumulate(dbias,bias_key,ds);
      }
    }
    if(backward) {
      for(const auto &[key,value] : dk) pointwise(ACDNN_POINTWISE_IDENTITY_FWD,{value},head(backward->dk,key.first,key.second));
      for(const auto &[key,value] : dv) pointwise(ACDNN_POINTWISE_IDENTITY_FWD,{value},head(backward->dv,key.first,key.second));
      for(const auto &[key,value] : dbias) pointwise(ACDNN_POINTWISE_IDENTITY_FWD,{value},head(*backward->dbias,key.first,key.second));
    }
  }
  struct Alias {std::int64_t uid,source;std::size_t offset;};
  struct Step {std::unique_ptr<TestExecutable> executable;std::vector<std::int64_t> uids;};
  CapabilityRecord capability_;
  bool allow_fp8_=false;
  flagdnnDataType_t fp8_type_=FLAGDNN_DATA_FP8_E4M3;
  std::set<std::int64_t> external_;
  std::map<std::int64_t,DeviceBuffer> buffers_;
  std::vector<Alias> aliases_;
  std::vector<Step> steps_;
  std::int64_t next_uid_=0;
  std::size_t workspace_=0;
};
}  // namespace
std::vector<std::string> acdnn_fp8_attention_plan(bool backward) {
  std::vector<std::string> plan={
      "acdnnBackendExecute(IDENTITY_INT8_FLOAT,FP8-storage)",
      "acdnnBackendExecute(POINTWISE,FP8-decode-encode-RNE)",
      "acdnnOpTensor(MIN,MAX,FP8-saturation)",
      "acdnnBackendExecute(MATMUL,FP8-attention-heads)",
      "acdnnOpTensor(ADD,FP8-scale-row-broadcast)",
      "acdnnReduceTensor(MAX,ADD,FP8-attention-rows)",
      "acdnnBackendExecute(POINTWISE,FP8-attention-scale-exp-normalize)",
      "acdnnReduceTensor(MAX,FP8-attention-amax)"};
  if(backward)plan.push_back("acdnnBackendExecute(MATMUL,POINTWISE_ADD,FP8-attention-gradients)");
  return plan;
}
std::vector<std::string> acdnn_attention_plan(bool backward) {
  std::vector<std::string> result={
      "acdnnBackendExecute(POINTWISE_IDENTITY_FWD,attention-conversion)",
      "acdnnBackendExecute(MATMUL,attention-heads)",
      "acdnnBackendExecute(POINTWISE_MUL,ADD,attention-scale-bias-mask)",
      "acdnnSoftmaxForward(ACCURATE,CHANNEL)"};
  if(backward) {
    result.push_back("acdnnSoftmaxBackward(ACCURATE,CHANNEL)");
    result.push_back("acdnnBackendExecute(MATMUL,POINTWISE_ADD,attention-gradients)");
  } else {
    result.push_back("acdnnSoftmaxForward(LOG,CHANNEL)");
    result.push_back("acdnnBackendExecute(POINTWISE_SUB,attention-statistics)");
    result.push_back("acdnnTransformTensor(attention-statistics-column)");
  }
  return result;
}
std::unique_ptr<flagdnn::testing::AttentionExecutable> make_acdnn_attention_reference(
    const SdpaTestCase &test_case,const CapabilityRecord &capability) {
  return std::make_unique<AttentionDag>(test_case,capability);
}
std::unique_ptr<flagdnn::testing::AttentionExecutable> make_acdnn_attention_reference(
    const SdpaBackwardTestCase &test_case,const CapabilityRecord &capability) {
  return std::make_unique<AttentionDag>(test_case,capability);
}
std::unique_ptr<flagdnn::testing::AttentionExecutable> make_acdnn_attention_reference(
    const flagdnn::testing::SdpaFp8TestCase &test_case,const CapabilityRecord &capability) {
  return std::make_unique<AttentionDag>(test_case,capability);
}
std::unique_ptr<flagdnn::testing::AttentionExecutable> make_acdnn_attention_reference(
    const flagdnn::testing::SdpaFp8BackwardTestCase &test_case,const CapabilityRecord &capability) {
  return std::make_unique<AttentionDag>(test_case,capability);
}
}  // namespace flagdnn::validation::thead
