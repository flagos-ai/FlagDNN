/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
/*
 * Standalone NVIDIA native-reference measurements. Reuses the registered
 * native case catalogs, input formulas, tensor I/O and cuDNN builders. No
 * FlagDNN executable is built. --catalog never initializes CUDA; --case NAME
 * permits process isolation. See benchmark_nvidia_extended.py for the build
 * command and isolated full-catalog execution.
 *
 * Successful measurements assert finite outputs and intact padding only,
 * except explicitly named CPU/exact checks. They do not assert FlagDNN
 * correctness. CUDA Graph batch/warmup/sample counts match the source runners.
 */
#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <fstream>
#include <set>
#include <iomanip>
#include <iostream>
#include <map>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include "common/attention.hpp"
#include "common/layout.hpp"
#include "common/pointwise.hpp"
#include "validation/functional/cudnn_extended.hpp"
#include "validation/benchmark/timing.hpp"
#include "validation/tensor_io.hpp"

namespace native_baseline {
using namespace flagdnn::testing;
using namespace flagdnn::validation::nvidia;
namespace io = flagdnn::testing::cuda;
namespace timing = flagdnn::validation::nvidia::timing;

std::string quote(std::string_view s) {
  std::ostringstream out; out << '"';
  for (unsigned char c : s) {
    if (c == '"' || c == '\\') out << '\\' << c;
    else if (c == '\n') out << "\\n";
    else if (c == '\r') out << "\\r";
    else if (c == '\t') out << "\\t";
    else if (c < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c) << std::dec;
    else out << c;
  }
  return out.str() + '"';
}
template<class T> std::string array(const std::vector<T>& v) {
  std::ostringstream o; o << std::setprecision(12) << '[';
  for (std::size_t i=0;i<v.size();++i) { if(i) o<<','; o<<v[i]; }
  return o.str()+']';
}
std::string dtype(flagdnnDataType_t t) {
  switch(t) {
    case FLAGDNN_DATA_FLOAT32:return "fp32";
    case FLAGDNN_DATA_FLOAT16:return "fp16";
    case FLAGDNN_DATA_BFLOAT16:return "bfloat16";
    case FLAGDNN_DATA_INT32:return "int32";
    case FLAGDNN_DATA_BOOLEAN:return "boolean";
    case FLAGDNN_DATA_FP8_E4M3:return "fp8_e4m3";
    case FLAGDNN_DATA_FP8_E5M2:return "fp8_e5m2";
    case FLAGDNN_DATA_FP8_E8M0:return "fp8_e8m0";
    default:return "dtype_"+std::to_string(int(t));
  }
}
struct Data;
struct Case {
  std::string op,name,attributes="{}",catalog="native",blocked;
  std::vector<TestTensor> inputs,outputs;
  std::function<std::vector<std::vector<float>>()> values;
  std::function<std::unique_ptr<TestExecutable>()> build;
  std::function<void(Data&,Stream&)> prepare;
  std::function<void(const Data&,Stream&)> exact;
  std::string validation="nan_initialized_outputs_written_finite_and_padding_sanity";
  io::BooleanEncoding boolean_encoding=io::BooleanEncoding::kByte;
  int warmup=10,batch=50,samples=20;
  bool allow_negative_infinity = false;
};
struct Data {
  std::map<std::int64_t,std::unique_ptr<DeviceBuffer>> buffers;
  std::map<std::int64_t,std::vector<std::uint8_t>> initial;
  std::vector<flagdnnBinding_t> bindings;
  io::BooleanEncoding boolean_encoding;
  void add(const TestTensor& t, std::vector<std::uint8_t> bytes,Stream& s) {
    auto b=std::make_unique<DeviceBuffer>(t.binding_byte_offset+bytes.size());
    b->copy_from_host_at(bytes.data(),bytes.size(),t.binding_byte_offset,s.get());
    bindings.push_back({t.uid,b->opaque_at(t.binding_byte_offset)});
    initial.emplace(t.uid,std::move(bytes)); buffers.emplace(t.uid,std::move(b));
  }
  std::vector<std::uint8_t> read(const TestTensor& t,Stream& s) const {
    std::vector<std::uint8_t> bytes(initial.at(t.uid).size());
    buffers.at(t.uid)->copy_to_host_at(bytes.data(),bytes.size(),t.binding_byte_offset,s.get());
    s.synchronize();return bytes;
  }
  std::vector<float> logical(const TestTensor& t,Stream& s) const {
    return io::gather(io::decode(read(t,s),t.data_type,io::storage_element_count(t),boolean_encoding),t);
  }
};
std::string tensors(const Case& c) {
  std::ostringstream o; o<<'[';bool first=true;
  for(auto role:{"input","output"}) for(const auto& t:std::string(role)=="input"?c.inputs:c.outputs) {
    if(!first)o<<',';first=false;
    o<<"{\"uid\":"<<t.uid<<",\"role\":"<<quote(role)<<",\"dtype\":"<<quote(dtype(t.data_type))
     <<",\"dimensions\":"<<array(t.dimensions)<<",\"strides\":"<<array(t.strides)
     <<",\"binding_byte_offset\":"<<t.binding_byte_offset<<",\"logical_elements\":"<<io::element_count(t)
     <<",\"storage_elements\":"<<io::storage_element_count(t)
     <<",\"boolean_encoding\":"<<quote(c.boolean_encoding==io::BooleanEncoding::kBitPacked?"bitpacked":"byte")<<'}';
  } return o.str()+']';
}
std::string metadata(const Case& c) {
  std::ostringstream o;
  o<<"\"case_name\":"<<quote(c.name)<<",\"op\":"<<quote(c.op)<<",\"provider\":\"cudnn\",\"catalog\":"<<quote(c.catalog)
   <<",\"tensors\":"<<tensors(c)<<",\"attributes\":"<<c.attributes
   <<",\"validation_scope\":"<<quote(c.exact?"cpu_exact":"sanity")<<",\"validation_method\":"<<quote(c.validation)
   <<",\"timing_method\":\"cuda_event_cuda_graph_batch\",\"warmup_iterations\":"<<c.warmup
   <<",\"graph_batch_iterations\":"<<c.batch<<",\"sample_count\":"<<c.samples
   <<",\"graph_warmup_replays\":1,\"percentile_method\":\"nearest_rank\"";
  return o.str();
}
bool measure(const Case& c) {
  bool fatal_context_error = false;
  std::string stage="build",validation="NOT_RUN";double build_us=0;std::size_t workspace=0;
  std::vector<double> samples,batches;
  try {
    if(!c.blocked.empty())throw std::runtime_error(c.blocked);
    Stream stream;
    const auto start=timing::HostClock::now();auto executable=c.build();build_us=timing::host_microseconds_since(start);
    workspace=executable->workspace_size();DeviceBuffer work(workspace);
    stage="inputs";Data data;data.boolean_encoding=c.boolean_encoding;const auto values=c.values();
    if(values.size()!=c.inputs.size())throw std::runtime_error("input count differs from source case");
    for(std::size_t i=0;i<c.inputs.size();++i) {
      const auto& t=c.inputs[i];auto bytes=io::encode(io::scatter(values[i],t),t.data_type,c.boolean_encoding);
      data.add(t,std::move(bytes),stream);
    }
    for(const auto& t:c.outputs) {
      auto physical = std::vector<float>(io::storage_element_count(t), io::padding_sentinel());
      if (t.data_type != FLAGDNN_DATA_INT32 && t.data_type != FLAGDNN_DATA_BOOLEAN)
        for (std::size_t i = 0; i < io::element_count(t); ++i)
          physical[io::logical_offset(i, t)] = std::numeric_limits<float>::quiet_NaN();
      auto bytes=io::encode(physical,t.data_type,c.boolean_encoding);
      data.add(t,std::move(bytes),stream);
    }
    stage="prepare";if(c.prepare)c.prepare(data,stream);stream.synchronize();
    auto launch=[&]{executable->execute(data.bindings,work.opaque(),workspace,stream.opaque());};
    stage="execute";launch();stream.synchronize();stage="validation";
    if(c.exact)c.exact(data,stream);
    for(const auto& t:c.outputs) {
      const auto output_bytes = data.read(t, stream);
      if (output_bytes == data.initial.at(t.uid))
        throw std::runtime_error("output storage was not written");
      const auto physical=io::decode(output_bytes,t.data_type,io::storage_element_count(t),c.boolean_encoding);
      io::require_padding_unchanged("cuDNN",physical,t);
      for (float value : io::gather(physical, t))
        if (!std::isfinite(value) &&
            !(c.allow_negative_infinity && std::isinf(value) && value < 0))
          throw std::runtime_error("nonfinite output");
    }
    validation=c.exact?"PASS":"SANITY_PASS";
    stage="warmup";for(int i=0;i<c.warmup;++i)launch();stream.synchronize();
    stage="capture";timing::CapturedExecutionBatch graph(stream.get(),c.batch,launch);graph.launch(stream.get());stream.synchronize();
    stage="timing";EventTimer timer;
    for(int i=0;i<c.samples;++i) {
      const auto us=timer.measure_microseconds(stream.get(),1,[&]{graph.launch(stream.get());});
      if(!std::isfinite(us)||us<=0)throw std::runtime_error("invalid timing sample");
      batches.push_back(us);samples.push_back(us/c.batch);
    }
    std::cout<<'{'<<metadata(c)<<",\"status\":\"ok\",\"correctness\":"<<quote(validation)
      <<",\"workspace_bytes\":"<<workspace<<",\"build_us\":"<<build_us<<",\"samples_us\":"<<array(samples)
      <<",\"sample_batch_us\":"<<array(batches)<<",\"median_us\":"<<timing::percentile(samples,.5)
      <<",\"p90_us\":"<<timing::percentile(samples,.9)<<"}\n";
  }catch(const std::exception& e) {
    fatal_context_error = c.blocked.empty();  // Restart after any failure; cuDNN may hide CUDA error names.
    std::cout<<'{'<<metadata(c)<<",\"status\":"<<quote(c.blocked.empty()?"failed":"blocked")
      <<",\"stage\":"<<quote(stage)<<",\"correctness\":"<<quote(validation)<<",\"reason\":"<<quote(e.what())<<"}\n";
  }
  std::cout.flush();
  return fatal_context_error;
}
template<class T,class Builder,class Values> Case basic(std::string op,const T& t,Builder builder,Values values) {
  Case c;c.op=op;c.name=t.name;c.inputs=t.inputs;c.outputs=t.outputs;
  c.build=[t,builder]{return builder(t);};c.values=[t,values]{return values(t);};return c;
}
std::vector<std::vector<float>> indexed_inputs(const std::vector<TestTensor>& tensors,bool by_uid=false,bool saved_last=false,float divisor=16) {
  std::vector<std::vector<float>> result;
  for(const auto& t:tensors) {
    std::vector<float> v(io::element_count(t));
    for(std::size_t i=0;i<v.size();++i) {
      v[i]=float(int((i*17+(by_uid?t.uid:result.size())*7)%61)-30)/divisor;
      if(saved_last&&result.size()+1==tensors.size())v[i]=.5f+std::abs(v[i]);
    } result.push_back(std::move(v));
  }return result;
}
std::string attention_options(const AttentionOptions& a) {
  std::ostringstream o;o<<"{\"attention_scale\":";
  if(a.attention_scale)o<<*a.attention_scale;else o<<"null";
  o<<",\"diagonal_band_left_bound\":";if(a.diagonal_band_left_bound)o<<*a.diagonal_band_left_bound;else o<<"null";
  o<<",\"diagonal_band_right_bound\":";if(a.diagonal_band_right_bound)o<<*a.diagonal_band_right_bound;else o<<"null";
  o<<",\"diagonal_alignment\":"<<quote(a.diagonal_alignment==AttentionDiagonalAlignment::kTopLeft?"top_left":"bottom_right")<<'}';return o.str();
}
std::vector<float> attention_values(const TestTensor& t,std::size_t index,float scale) {
  std::vector<float> v(io::element_count(t));
  for(std::size_t i=0;i<v.size();++i)v[i]=scale*float(int((i*37+index*19)%101)-50)/float(53+index);
  return v;
}
void execute_primal(TestExecutable& e,Data& data,Stream& s) {
  DeviceBuffer work(e.workspace_size());s.synchronize();
  e.execute(data.bindings,work.opaque(),e.workspace_size(),s.opaque());s.synchronize();
}
std::vector<Case> catalog(bool extra_variants = false) {
  std::vector<Case> all;
  for(const std::string op:{"concatenate","gen_index"})for(const auto& t:make_index_cases(op)) {
    Case c;c.op=op;c.name=t.name;c.inputs=t.inputs;c.outputs={t.output};c.attributes="{\"axis\":"+std::to_string(t.axis)+"}";
    c.build=[t]{return build_cudnn_index(t);};c.values=[t]{return indexed_inputs(t.inputs);};
    if(op=="concatenate"&&t.output.data_type!=FLAGDNN_DATA_FLOAT32&&t.output.data_type!=FLAGDNN_DATA_FLOAT16&&t.output.data_type!=FLAGDNN_DATA_BFLOAT16)
      c.blocked="Excluded by NVIDIA native index runner: concatenate supports only fp32/fp16/bfloat16 reference cases";
    all.push_back(std::move(c));
  }
  for(const auto& t:make_causal_convolution_cases()) {
    auto c=basic("causal_conv1d",t,build_cudnn_causal_convolution,causal_convolution_inputs);
    c.attributes="{\"precision\":"+std::to_string(t.precision)+",\"dilation\":"+std::to_string(t.dilation)+",\"silu\":"+(t.silu?"true":"false")+"}";all.push_back(std::move(c));
  }
  for(const auto& t:make_fp8_matmul_cases()) {
    auto c=basic(t.plain_matmul ? "matmul" : "matmul_fp8",t,build_cudnn_fp8_matmul,fp8_matmul_inputs);
    c.attributes="{\"scale_mode\":"+std::to_string(t.scale_mode)+",\"scale_variant\":"+std::to_string(t.scale_variant)+",\"plain_matmul\":"+(t.plain_matmul?"true":"false")+"}";all.push_back(std::move(c));
  }
  for(const auto& t:make_moe_matmul_cases(false)) {
    auto c=basic("moe_grouped_matmul",t,build_cudnn_moe_matmul,moe_matmul_inputs);
    c.attributes="{\"mode\":"+std::to_string(t.mode)+",\"top_k\":"+std::to_string(t.top_k)+",\"backward\":false,\"offsets\":"+array(t.offsets)+",\"token_index\":"+array(t.token_index)+",\"token_ks\":"+array(t.token_ks)+"}";all.push_back(std::move(c));
  }
  for(const auto& t:make_resample_cases()) {
    auto c=basic("resample",t,build_cudnn_resample,resample_inputs);
    if (t.mode == 5 && t.padding == 2) {
      c.validation = "cpu_maxpool_exact_including_empty_window_negative_infinity_and_padding";
      c.allow_negative_infinity = true;
      c.exact = [t](const Data& data, Stream& stream) {
        const auto& input = t.inputs[0];
        const auto& output = t.outputs[0];
        const auto source = io::gather(
            io::decode(data.initial.at(input.uid), input.data_type,
                       io::storage_element_count(input), data.boolean_encoding), input);
        const auto actual = data.logical(output, stream);
        const auto height = input.dimensions[2], width = input.dimensions[3];
        const auto out_height = output.dimensions[2], out_width = output.dimensions[3];
        for (std::size_t index = 0; index < actual.size(); ++index) {
          const auto column = index % out_width;
          const auto row = (index / out_width) % out_height;
          const auto plane = index / (out_width * out_height);
          float expected = -std::numeric_limits<float>::infinity();
          for (std::int64_t dy = 0; dy < t.window[0]; ++dy)
            for (std::int64_t dx = 0; dx < t.window[1]; ++dx) {
              const auto y = static_cast<std::int64_t>(row) * t.stride[0] - t.pre[0] + dy;
              const auto x = static_cast<std::int64_t>(column) * t.stride[1] - t.pre[1] + dx;
              if (y >= 0 && y < height && x >= 0 && x < width)
                expected = std::max(expected, source[plane * height * width + y * width + x]);
            }
          if (actual[index] != expected)
            throw std::runtime_error("CPU maxpool mismatch at element " + std::to_string(index));
        }
      };
    }
    c.attributes="{\"mode\":"+std::to_string(t.mode)+",\"padding\":"+std::to_string(t.padding)+",\"window\":"+array(t.window)+",\"stride\":"+array(t.stride)+",\"pre\":"+array(t.pre)+",\"post\":"+array(t.post)+",\"align_corners\":"+(t.align_corners?"true":"false")+"}";all.push_back(std::move(c));
  }
  for(const std::string op:{"batchnorm_backward","layernorm_backward","rmsnorm_backward","instancenorm","instancenorm_backward","adalayernorm","adalayernorm_backward"})
    for(const auto& t:make_extended_normalization_cases(op)) {
      auto c=basic(op,t,build_cudnn_extended_normalization,[](const auto& v){return indexed_inputs(v.inputs,true,v.operation.ends_with("_backward"),19);});
      c.attributes="{\"axes\":"+array(t.axes)+",\"epsilon\":"+std::to_string(t.epsilon)+"}";all.push_back(std::move(c));
    }
  for(const auto& t:make_bn_finalize_cases()) {
    auto c=basic("bn_finalize",t,build_cudnn_statistics,[](const auto& v){
      std::vector<std::vector<float>> result;for(const auto& tensor:v.inputs) {
        std::vector<float> x(io::element_count(tensor));for(std::size_t i=0;i<x.size();++i)x[i]=statistics_input_value(v,result.size(),i);result.push_back(std::move(x));
      }return result;});
    c.attributes="{\"epsilon\":"+std::to_string(t.epsilon)+",\"accum_count\":"+std::to_string(t.accum_count)+",\"momentum\":"+std::to_string(t.momentum)+"}";all.push_back(std::move(c));
  }
  for(const auto& pair:std::vector<std::pair<std::string,LayoutOperation>>{{"reshape",LayoutOperation::kReshape},{"transpose",LayoutOperation::kTranspose},{"slice",LayoutOperation::kSlice}})
    for(const auto& t:make_layout_cases(pair.second)) {
      Case c;c.op=pair.first;c.name=t.name;c.catalog="native_dtype_copy";c.inputs={t.input};c.outputs={t.output};
      c.build=[t]{return build_layout_reference(t);};c.values=[t]{std::vector<float> v(io::element_count(t.input));for(std::size_t i=0;i<v.size();++i)v[i]=float(int((i*17)%41)-20)/13.f;return std::vector<std::vector<float>>{v};};
      std::ostringstream attrs;attrs<<"{\"permutation\":"<<array(t.permutation)<<",\"slice_strides\":"<<array(t.slice_strides)<<",\"slices\":[";
      for(std::size_t i=0;i<t.slices.size();++i){if(i)attrs<<',';attrs<<'['<<t.slices[i].first<<','<<t.slices[i].second<<']';}attrs<<"]}";c.attributes=attrs.str();
      if(t.input.data_type!=FLAGDNN_DATA_FLOAT32&&t.input.data_type!=FLAGDNN_DATA_FLOAT16&&t.input.data_type!=FLAGDNN_DATA_BFLOAT16)c.blocked="Excluded by NVIDIA native layout runner: only fp32/fp16/bfloat16 reference cases are registered";
      all.push_back(std::move(c));
    }
  {
    PointwiseCaseDefinition def;
    def.operation_name = "logical_not";
    def.mode = FLAGDNN_POINTWISE_LOGICAL_NOT;
    def.input_domain = PointwiseInputDomain::kLogical;
    for (const auto& t : make_unary_pointwise_cases(def)) {
      Case c;
      c.op = "logical_not";
      c.name = t.name;
      c.catalog = "native_dtype_boolean";
      c.inputs = t.inputs;
      c.outputs = {t.output};
      c.build = [t] { return build_pointwise_reference(t); };
      c.boolean_encoding = io::BooleanEncoding::kBitPacked;
      c.values = [t] {
        std::vector<float> values(io::element_count(t.inputs[0]));
        for (std::size_t i = 0; i < values.size(); ++i)
          values[i] = float((i * 17) % 3 != 0);
        return std::vector<std::vector<float>>{values};
      };
      c.validation = "cpu_boolean_exact_and_padding";
      c.exact = [t](const Data& data, Stream& stream) {
        const auto actual = data.logical(t.output, stream);
        for (std::size_t i = 0; i < actual.size(); ++i)
          if (actual[i] != float((i * 17) % 3 == 0))
            throw std::runtime_error("CPU logical_not result differs");
      };
      all.push_back(std::move(c));
    }
  }
  // Attention primal execution is performed before measurement exactly as in
  // the native runner. Forward outputs/statistics are inputs to backward.
  for(const auto& t : extra_variants ? make_sdpa_cases() : make_sdpa_benchmark_cases())
    if(extra_variants ? t.q.data_type == FLAGDNN_DATA_FLOAT32 : t.q.data_type != FLAGDNN_DATA_FLOAT32) {
    Case c;c.op="sdpa";c.name=t.name;c.inputs={t.q,t.k,t.v};if(t.bias)c.inputs.push_back(*t.bias);c.outputs={t.output};if(t.stats)c.outputs.push_back(*t.stats);
    c.build=[t]{return build_sdpa_reference(t);};c.values=[t]{std::vector<std::vector<float>> v{attention_values(t.q,0,.5f),attention_values(t.k,1,.5f),attention_values(t.v,2,.5f)};if(t.bias)v.push_back(attention_values(*t.bias,3,.25f));return v;};c.attributes=attention_options(t.options);c.catalog = extra_variants ? "shared_functional_attention_capability_probe" : "native_attention";
    if (extra_variants) {
      c.attributes.pop_back();
      c.attributes += ",\"autotune\":" + std::string(t.autotune ? "true" : "false") + "}";
    }
    c.warmup=5;c.batch=20;c.samples=10;all.push_back(std::move(c));
  }
  for(const auto& t : extra_variants ? make_sdpa_backward_cases() : make_sdpa_backward_benchmark_cases())
    if(extra_variants ? (t.q.data_type == FLAGDNN_DATA_FLOAT32 || t.dbias.has_value()) : t.q.data_type != FLAGDNN_DATA_FLOAT32) {
    Case c;c.op="sdpa_backward";c.name=t.name;c.inputs={t.q,t.k,t.v,t.doutput,t.output,t.stats};if(t.bias)c.inputs.push_back(*t.bias);c.outputs={t.dq,t.dk,t.dv};if(t.dbias)c.outputs.push_back(*t.dbias);
    c.build=[t]{return build_sdpa_backward_reference(t);};c.values=[t]{std::vector<std::vector<float>> v{attention_values(t.q,10,.5f),attention_values(t.k,11,.5f),attention_values(t.v,12,.5f),attention_values(t.doutput,13,.25f),std::vector<float>(io::element_count(t.output)),std::vector<float>(io::element_count(t.stats))};if(t.bias)v.push_back(attention_values(*t.bias,14,.25f));return v;};
    c.prepare=[t](Data& d,Stream& s){SdpaTestCase p;p.name=t.name+"::primal";p.q=t.q;p.k=t.k;p.v=t.v;p.bias=t.bias;p.output=t.output;p.stats=t.stats;p.options=t.options;auto e=build_sdpa_reference(p);execute_primal(*e,d,s);};
    c.attributes=attention_options(t.options);c.attributes.pop_back();c.attributes+=",\"deterministic\":"+std::string(t.deterministic?"true":"false")+",\"backward_inputs_from_cudnn_forward\":true}";c.catalog = extra_variants ? "shared_functional_attention_capability_probe" : "native_attention";
    if (extra_variants) {
      c.attributes.pop_back();
      c.attributes += ",\"autotune\":" + std::string(t.autotune ? "true" : "false") + "}";
    }
    c.warmup=5;c.batch=20;c.samples=10;all.push_back(std::move(c));
  }
  for(const auto& t:make_sdpa_fp8_benchmark_cases()) {
    Case c;c.op="sdpa_fp8";c.name=t.name;c.inputs={t.q,t.k,t.v};
    const std::vector<Fp8Scalar> scalars{t.descale_q,t.descale_k,t.descale_v,t.descale_s,t.scale_s,t.scale_o};for(const auto& s:scalars)c.inputs.push_back(s.tensor);if(t.bias)c.inputs.push_back(*t.bias);
    c.outputs={t.output,t.amax_s,t.amax_o};if(t.stats)c.outputs.push_back(*t.stats);c.build=[t]{return build_sdpa_fp8_reference(t);};
    c.values=[t,scalars]{std::vector<std::vector<float>> v{attention_values(t.q,20,1),attention_values(t.k,21,1),attention_values(t.v,22,1)};for(const auto& s:scalars)v.push_back({s.value});if(t.bias)v.push_back(attention_values(*t.bias,23,.125f));return v;};
    c.attributes=attention_options(t.options);c.attributes.pop_back();std::vector<float> sv;for(auto& s:scalars)sv.push_back(s.value);c.attributes+=",\"fp8_scalar_values_input_order\":"+array(sv)+"}";
    c.catalog="native_attention";c.warmup=5;c.batch=20;c.samples=10;all.push_back(std::move(c));
  }
  for(const auto& t:make_sdpa_fp8_backward_benchmark_cases()) {
    Case c;c.op="sdpa_fp8_backward";c.name=t.name;c.inputs={t.q,t.k,t.v,t.doutput,t.output,t.stats};
    const std::vector<Fp8Scalar> scalars{t.descale_q,t.descale_k,t.descale_v,t.descale_o,t.descale_doutput,t.descale_s,t.descale_dp,t.scale_s,t.scale_dq,t.scale_dk,t.scale_dv,t.scale_dp};for(const auto& s:scalars)c.inputs.push_back(s.tensor);
    c.outputs={t.dq,t.dk,t.dv,t.amax_dq,t.amax_dk,t.amax_dv,t.amax_dp};c.build=[t]{return build_sdpa_fp8_backward_reference(t);};
    c.values=[t,scalars]{std::vector<std::vector<float>> v{attention_values(t.q,30,1),attention_values(t.k,31,1),attention_values(t.v,32,1),attention_values(t.doutput,33,.5f),std::vector<float>(io::element_count(t.output)),std::vector<float>(io::element_count(t.stats))};for(auto& s:scalars)v.push_back({s.value});return v;};
    c.prepare=[t](Data& d,Stream& s){SdpaFp8TestCase p;p.name=t.name+"::primal";p.q=t.q;p.k=t.k;p.v=t.v;p.descale_q=t.descale_q;p.descale_k=t.descale_k;p.descale_v=t.descale_v;p.descale_s=t.descale_s;p.scale_s=t.scale_s;p.output=t.output;p.stats=t.stats;p.options=t.options;
      p.scale_o={{t.amax_dp.uid+1,FLAGDNN_DATA_FLOAT32,{1,1,1,1},{1,1,1,1}},1.f/t.descale_o.value};p.amax_s={t.amax_dp.uid+2,FLAGDNN_DATA_FLOAT32,{1,1,1,1},{1,1,1,1}};p.amax_o={t.amax_dp.uid+3,FLAGDNN_DATA_FLOAT32,{1,1,1,1},{1,1,1,1}};
      d.add(p.scale_o.tensor,io::encode(std::vector<float>{p.scale_o.value},FLAGDNN_DATA_FLOAT32,io::BooleanEncoding::kByte),s);for(auto& a:{p.amax_s,p.amax_o})d.add(a,io::encode(std::vector<float>{0},FLAGDNN_DATA_FLOAT32,io::BooleanEncoding::kByte),s);
      auto e=build_sdpa_fp8_reference(p);execute_primal(*e,d,s);
    };
    c.attributes=attention_options(t.options);c.attributes.pop_back();std::vector<float> sv;for(auto& s:scalars)sv.push_back(s.value);c.attributes+=",\"fp8_scalar_values_input_order\":"+array(sv)+",\"backward_inputs_from_cudnn_forward\":true}";
    c.catalog="native_attention";c.warmup=5;c.batch=20;c.samples=10;all.push_back(std::move(c));
  }
  for(const std::string op:{"genstats","rng","rope","rope_backward","moe_grouped_matmul_bwd"}) {
    Case c;c.op=op;c.name=op+"::native_reference_unavailable";c.blocked="NVIDIA native reference unavailable: operation listed in backends/nvidia/validation/cudnn_unsupported_operators.txt; no registered executable cuDNN comparator";all.push_back(std::move(c));
  }
  if (extra_variants) {
    std::erase_if(all, [](const Case& c) {
      return c.catalog != "shared_functional_attention_capability_probe";
    });
  }
  return all;
}
} // namespace native_baseline
int main(int argc, char** argv) {
  using namespace native_baseline;
  std::cout << std::setprecision(12);
  bool list = false;
  bool extra_variants = false;
  std::string selected, case_file;
  std::set<std::string> selected_cases;
  for (int i = 1; i < argc; ++i) {
    std::string argument = argv[i];
    if (argument == "--catalog") list = true;
    else if (argument == "--extra-variants") extra_variants = true;
    else if (argument == "--case" && i + 1 < argc) selected = argv[++i];
    else if (argument == "--case-file" && i + 1 < argc) case_file = argv[++i];
    else {
      std::cerr << "usage: " << argv[0]
                << " [--catalog] [--extra-variants] [--case NAME] [--case-file PATH]\n";
      return 2;
    }
  }
  if (!case_file.empty()) {
    std::ifstream input(case_file);
    if (!input) return 2;
    for (std::string name; std::getline(input, name);)
      if (!name.empty()) selected_cases.insert(name);
  }
  std::unique_ptr<DriverContext> context;
  if (!list) context = std::make_unique<DriverContext>();
  std::size_t count = 0;
  for (const auto& c : catalog(extra_variants)) {
    if (!selected.empty() && selected != c.name) continue;
    if (!case_file.empty() && !selected_cases.contains(c.name)) continue;
    ++count;
    if (list) {
      std::cout << '{' << metadata(c) << ",\"status\":"
                << quote(c.blocked.empty() ? "catalog" : "blocked")
                << ",\"reason\":" << quote(c.blocked) << "}\n";
    } else if (measure(c)) {
      // Conservatively restart after every failed case; cuDNN may hide a fatal CUDA context error.
      return 3;
    }
  }
  return count ? 0 : 2;
}
