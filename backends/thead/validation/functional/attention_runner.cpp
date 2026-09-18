// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#include "common/attention_runner.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <flagdnn/flagdnn.hpp>
#include <iostream>
#include <stdexcept>
#include <type_traits>

#include "acdnn_attention_reference.hpp"
#include "acdnn_fp8_codec.hpp"
#include "acdnn_reference.hpp"
#include "functional/paired.hpp"
#include "pointwise_runner_support.hpp"

namespace flagdnn::testing {
namespace {
namespace tv=flagdnn::validation::thead;
namespace functional=tv::functional;

bool is_fp8(flagdnnDataType_t type) {
  return type==FLAGDNN_DATA_FP8_E4M3 || type==FLAGDNN_DATA_FP8_E5M2;
}
functional::BoundTensor fp8_buffer(const TestTensor &specification,CUstream stream,
                                   std::optional<std::size_t> salt=std::nullopt) {
  if (!is_fp8(specification.data_type) || !functional::is_contiguous(specification) ||
      specification.binding_byte_offset!=0) {
    throw std::invalid_argument("FP8 test requires dense byte storage without offsets");
  }
  const auto count=functional::element_count(specification);
  std::vector<std::uint8_t> bytes(count+64,0x55);
  if (salt) {
    const auto base=specification.data_type==FLAGDNN_DATA_FP8_E4M3?0x28:0x34;
    const auto span=specification.data_type==FLAGDNN_DATA_FP8_E4M3?24:12;
    for(std::size_t i=0;i<count;++i) {
      const auto magnitude=i%19==0?0:i%17==0?1:base+(i*17+*salt*11)%span;
      bytes[i]=static_cast<std::uint8_t>(magnitude+(((i/3+*salt)%2)?128:0));
    }
  }
  auto buffer=std::make_unique<tv::DeviceBuffer>(bytes.size());
  tv::copy_to_device_async<std::uint8_t>(*buffer,bytes,0,stream);
  return {specification,std::move(buffer)};
}
functional::BoundTensor attention_output(const TestTensor &tensor,CUstream stream) {
  return is_fp8(tensor.data_type)?fp8_buffer(tensor,stream):functional::make_output_buffer(tensor,stream);
}
functional::BoundTensor scale_input(const Fp8Scalar &scalar,CUstream stream) {
  auto tensor=functional::make_output_buffer(scalar.tensor,stream);
  const std::array<float,1> value{scalar.value};
  tv::copy_to_device_async<float>(*tensor.buffer,value,0,stream);return tensor;
}
std::vector<float> attention_values(const functional::BoundTensor &tensor,CUstream stream) {
  if (!is_fp8(tensor.specification.data_type))return functional::read_output(tensor,stream);
  const auto count=functional::element_count(tensor.specification);
  std::array<std::uint8_t,64> guard{};
  tv::copy_from_device_async<std::uint8_t>(guard,*tensor.buffer,count,stream);
  TestTensor decoded=tensor.specification;decoded.uid=decoded.uid==1?2:1;
  decoded.data_type=FLAGDNN_DATA_FLOAT32;
  auto reference=tv::make_acdnn_fp8_codec(tensor.specification,decoded);
  tv::DeviceBuffer output(count*sizeof(float)),workspace(reference->workspace_size());
  const std::array<flagdnnBinding_t,2> bindings{{
      {tensor.specification.uid,tensor.buffer->data()},{decoded.uid,output.data()}}};
  reference->execute(bindings,workspace.data(),workspace.size(),reinterpret_cast<flagdnnStream_t>(stream));
  std::vector<float> values(count);
  tv::copy_from_device_async<float>(values,output,0,stream);
  tv::check_driver(cuStreamSynchronize(stream),"FP8 acDNN decoding synchronize");
  if (!std::all_of(guard.begin(),guard.end(),[](auto value){return value==0x55;})) {
    throw std::runtime_error("FP8 attention modified its output guard");
  }
  return values;
}

void compare(const functional::BoundTensor &actual,const functional::BoundTensor &reference,
             double absolute,double relative,tv::DeviceStream &stream,std::string_view name) {
  const auto left=attention_values(actual,stream.get());
  const auto right=attention_values(reference,stream.get());
  functional::require_padding_unchanged("FlagDNN",left,actual.specification);
  functional::require_padding_unchanged("acDNN",right,reference.specification);
  if(left.size()!=right.size())throw std::runtime_error("attention output sizes differ");
  for(std::size_t i=0;i<left.size();++i) {
    const double error=std::abs(double(left[i])-double(right[i]));
    if(!std::isfinite(left[i]) || !std::isfinite(right[i]) ||
        (error>absolute && error>relative*std::max(std::abs(left[i]),std::abs(right[i])))) {
      std::string storage_detail;
      if (is_fp8(actual.specification.data_type)) {
        std::array<std::uint8_t, 1> actual_byte{}, reference_byte{};
        tv::copy_from_device_async<std::uint8_t>(
            actual_byte, *actual.buffer, i, stream.get());
        tv::copy_from_device_async<std::uint8_t>(
            reference_byte, *reference.buffer, i, stream.get());
        tv::check_driver(cuStreamSynchronize(stream.get()),
                         "FP8 mismatch storage synchronize");
        storage_detail = " raw_flagdnn=" + std::to_string(actual_byte[0]) +
                         " raw_acdnn=" + std::to_string(reference_byte[0]);
      }
      throw std::runtime_error(std::string(name)+" differs at element "+std::to_string(i)+
          ": FlagDNN="+std::to_string(left[i])+" acDNN="+std::to_string(right[i]) + storage_detail);
    }
  }
}

void execute_production(TestExecutable &executable,std::span<const flagdnnBinding_t> bindings,
                        tv::DeviceBuffer &workspace,tv::DeviceStream &stream) {
  functional::execute(executable,bindings,workspace,stream);
  tv::check_driver(cuStreamSynchronize(stream.get()),"attention execute synchronize");
  CUgraph graph=nullptr;CUgraphExec replay=nullptr;bool capturing=false;
  try {
    tv::check_driver(cuStreamBeginCapture(stream.get(),CU_STREAM_CAPTURE_MODE_RELAXED),"attention capture begin");capturing=true;
    executable.execute(bindings,workspace.data(),workspace.size(),stream.opaque());
    tv::check_driver(cuStreamEndCapture(stream.get(),&graph),"attention capture end");capturing=false;
    tv::check_driver(cuGraphInstantiate(&replay,graph,0),"attention capture instantiate");
    for(int iteration=0;iteration<2;++iteration)tv::check_driver(cuGraphLaunch(replay,stream.get()),"attention graph replay");
    tv::check_driver(cuStreamSynchronize(stream.get()),"attention replay synchronize");
    tv::check_driver(cuGraphExecDestroy(replay),"attention graph exec destroy");replay=nullptr;
    tv::check_driver(cuGraphDestroy(graph),"attention graph destroy");graph=nullptr;
  }catch(...) {
    if(capturing)cuStreamEndCapture(stream.get(),&graph);
    if(replay)cuGraphExecDestroy(replay);
    if(graph)cuGraphDestroy(graph);
    throw;
  }
}

void forward_case(const SdpaTestCase &test_case, flagdnn::Handle &handle,
                  tv::DeviceStream &stream, const tv::CapabilityRecord &record,
                  bool benchmark = false) {
  auto production=build_flagdnn_sdpa(handle,test_case);
  auto reference=tv::make_acdnn_attention_reference(test_case,record);
  std::vector<functional::BoundTensor> inputs;
  for(const auto &tensor:{test_case.q,test_case.k,test_case.v}) inputs.push_back(functional::make_input_buffer(tensor,inputs.size(),stream.get()));
  if(test_case.bias)inputs.push_back(functional::make_input_buffer(*test_case.bias,inputs.size(),stream.get()));
  std::vector<functional::BoundTensor> actual,expected;
  actual.push_back(functional::make_output_buffer(test_case.output,stream.get()));
  expected.push_back(functional::make_output_buffer(test_case.output,stream.get()));
  if(test_case.stats) {
    actual.push_back(functional::make_output_buffer(*test_case.stats,stream.get()));
    expected.push_back(functional::make_output_buffer(*test_case.stats,stream.get()));
  }
  tv::DeviceBuffer production_workspace(production->workspace_size()),reference_workspace(reference->workspace_size());
  execute_production(*production,functional::bindings(inputs,actual),production_workspace,stream);
  functional::execute(*reference,functional::bindings(inputs,expected),reference_workspace,stream);
  compare(actual[0],expected[0],test_case.output_absolute_tolerance,test_case.output_relative_tolerance,stream,test_case.name+" output");
  if(test_case.stats)compare(actual[1],expected[1],test_case.stats_absolute_tolerance,test_case.stats_relative_tolerance,stream,test_case.name+" stats");
  if (benchmark)
    functional::measure_pair(test_case.name, *production, *reference,
                             functional::bindings(inputs, actual),
                             functional::bindings(inputs, expected),
                             production_workspace, reference_workspace, stream);
}

void backward_case(const SdpaBackwardTestCase &test_case,
                   flagdnn::Handle &handle, tv::DeviceStream &stream,
                   const tv::CapabilityRecord &record, bool benchmark = false) {
  SdpaTestCase forward;
  forward.name=test_case.name+"_forward_inputs";forward.q=test_case.q;forward.k=test_case.k;forward.v=test_case.v;
  forward.bias=test_case.bias;forward.output=test_case.output;forward.stats=test_case.stats;forward.options=test_case.options;
  auto forward_record=record;forward_record.reference_plan=tv::acdnn_attention_plan(false);
  auto forward_production=build_flagdnn_sdpa(handle,forward);
  auto forward_reference=tv::make_acdnn_attention_reference(forward,forward_record);
  std::vector<functional::BoundTensor> inputs;
  for(const auto &tensor:{test_case.q,test_case.k,test_case.v}) inputs.push_back(functional::make_input_buffer(tensor,inputs.size(),stream.get()));
  if(test_case.bias)inputs.push_back(functional::make_input_buffer(*test_case.bias,inputs.size(),stream.get()));
  std::vector<functional::BoundTensor> forward_actual,forward_expected;
  for(const auto &tensor:{test_case.output,test_case.stats}) {
    forward_actual.push_back(functional::make_output_buffer(tensor,stream.get()));
    forward_expected.push_back(functional::make_output_buffer(tensor,stream.get()));
  }
  tv::DeviceBuffer forward_workspace(forward_production->workspace_size()),reference_forward_workspace(forward_reference->workspace_size());
  execute_production(*forward_production,functional::bindings(inputs,forward_actual),forward_workspace,stream);
  functional::execute(*forward_reference,functional::bindings(inputs,forward_expected),reference_forward_workspace,stream);
  const double forward_tolerance=test_case.q.data_type==FLAGDNN_DATA_BFLOAT16?0.1:0.05;
  compare(forward_actual[0],forward_expected[0],forward_tolerance,0.05,stream,test_case.name+" forward output");
  compare(forward_actual[1],forward_expected[1],0.02,0.02,stream,test_case.name+" forward stats");
  inputs.push_back(functional::make_input_buffer(test_case.doutput,inputs.size(),stream.get()));
  auto production=build_flagdnn_sdpa_backward(handle,test_case);
  auto reference=tv::make_acdnn_attention_reference(test_case,record);
  std::vector<functional::BoundTensor> actual,expected;
  std::vector<TestTensor> outputs{test_case.dq,test_case.dk,test_case.dv};
  if(test_case.dbias)outputs.push_back(*test_case.dbias);
  for(const auto &tensor:outputs) {
    actual.push_back(functional::make_output_buffer(tensor,stream.get()));
    expected.push_back(functional::make_output_buffer(tensor,stream.get()));
  }
  auto production_bindings=functional::bindings(inputs,actual);
  auto reference_bindings=functional::bindings(inputs,expected);
  // Compare the backward implementations with identical primal inputs.
  // Forward correctness was checked above; acDNN supplies O and stats here.
  for (const auto &tensor : forward_expected) {
    for (auto *bindings : {&production_bindings, &reference_bindings}) {
      bindings->push_back({tensor.specification.uid, tensor.buffer->data()});
    }
  }
  tv::DeviceBuffer workspace(production->workspace_size()),reference_workspace(reference->workspace_size());
  execute_production(*production,production_bindings,workspace,stream);
  functional::execute(*reference,reference_bindings,reference_workspace,stream);
  for(std::size_t index=0;index<actual.size();++index)compare(actual[index],expected[index],test_case.absolute_tolerance,test_case.relative_tolerance,stream,test_case.name+" gradient "+std::to_string(index));
  if (benchmark)
    functional::measure_pair(test_case.name, *production, *reference,
                             production_bindings, reference_bindings, workspace,
                             reference_workspace, stream);
}

struct Fp8ForwardOutputs {
  std::vector<functional::BoundTensor> actual,expected;
};
Fp8ForwardOutputs execute_fp8_forward(const SdpaFp8TestCase &test_case,
                                      flagdnn::Handle &handle,
                                      tv::DeviceStream &stream,
                                      const tv::CapabilityRecord &record,
                                      bool benchmark = false) {
  auto production=build_flagdnn_sdpa_fp8(handle,test_case);
  auto reference=tv::make_acdnn_attention_reference(test_case,record);
  std::vector<functional::BoundTensor> inputs;
  for(const auto &tensor:{test_case.q,test_case.k,test_case.v}) inputs.push_back(fp8_buffer(tensor,stream.get(),inputs.size()));
  for(const auto &scalar:{test_case.descale_q,test_case.descale_k,test_case.descale_v,
                         test_case.descale_s,test_case.scale_s,test_case.scale_o}) inputs.push_back(scale_input(scalar,stream.get()));
  Fp8ForwardOutputs outputs;
  std::vector<TestTensor> specifications{test_case.output};
  if(test_case.stats)specifications.push_back(*test_case.stats);
  specifications.push_back(test_case.amax_s);specifications.push_back(test_case.amax_o);
  for(const auto &tensor:specifications) {
    outputs.actual.push_back(attention_output(tensor,stream.get()));
    outputs.expected.push_back(attention_output(tensor,stream.get()));
  }
  tv::DeviceBuffer workspace(production->workspace_size()),reference_workspace(reference->workspace_size());
  execute_production(*production,functional::bindings(inputs,outputs.actual),workspace,stream);
  functional::execute(*reference,functional::bindings(inputs,outputs.expected),reference_workspace,stream);
  for(std::size_t index=0;index<specifications.size();++index) {
    const bool stats=test_case.stats && index==1;
    const auto absolute=index==0?test_case.output_absolute_tolerance:stats?test_case.stats_absolute_tolerance:test_case.amax_absolute_tolerance;
    const auto relative=index==0?test_case.output_relative_tolerance:stats?test_case.stats_relative_tolerance:test_case.amax_relative_tolerance;
    compare(outputs.actual[index],outputs.expected[index],absolute,relative,stream,test_case.name+" output "+std::to_string(index));
  }
  if (benchmark)
    functional::measure_pair(test_case.name, *production, *reference,
                             functional::bindings(inputs, outputs.actual),
                             functional::bindings(inputs, outputs.expected),
                             workspace, reference_workspace, stream);
  return outputs;
}
void forward_case(const SdpaFp8TestCase &test_case, flagdnn::Handle &handle,
                  tv::DeviceStream &stream, const tv::CapabilityRecord &record,
                  bool benchmark = false) {
  (void)execute_fp8_forward(test_case, handle, stream, record, benchmark);
}
void backward_case(const SdpaFp8BackwardTestCase &test_case,
                   flagdnn::Handle &handle, tv::DeviceStream &stream,
                   const tv::CapabilityRecord &record, bool benchmark = false) {
  SdpaFp8TestCase forward;
  forward.name=test_case.name+"_forward_inputs";forward.q=test_case.q;forward.k=test_case.k;forward.v=test_case.v;
  forward.descale_q=test_case.descale_q;forward.descale_k=test_case.descale_k;forward.descale_v=test_case.descale_v;
  forward.descale_s=test_case.descale_s;forward.scale_s=test_case.scale_s;
  // Construct the forward input scale corresponding to the public backward
  // test's supplied output descale. All subsequent arithmetic uses GPU inputs.
  const auto extra_uid=test_case.amax_dp.uid+1000;
  forward.scale_o={{extra_uid,FLAGDNN_DATA_FLOAT32,{1,1,1,1},{1,1,1,1}},1.0F/test_case.descale_o.value};
  forward.output=test_case.output;forward.stats=test_case.stats;forward.options=test_case.options;
  forward.amax_s={extra_uid+1,FLAGDNN_DATA_FLOAT32,{1,1,1,1},{1,1,1,1}};
  forward.amax_o={extra_uid+2,FLAGDNN_DATA_FLOAT32,{1,1,1,1},{1,1,1,1}};
  forward.output_absolute_tolerance=0.5;forward.output_relative_tolerance=0.35;
  forward.stats_absolute_tolerance=0.08;forward.stats_relative_tolerance=0.08;
  forward.amax_absolute_tolerance=0.15;forward.amax_relative_tolerance=0.25;
  auto forward_record=record;forward_record.reference_plan=tv::acdnn_fp8_attention_plan(false);
  auto forward_outputs=execute_fp8_forward(forward,handle,stream,forward_record);
  std::vector<functional::BoundTensor> inputs;
  for(const auto &tensor:{test_case.q,test_case.k,test_case.v,test_case.doutput}) inputs.push_back(fp8_buffer(tensor,stream.get(),inputs.size()));
  for(const auto &scalar:{test_case.descale_q,test_case.descale_k,test_case.descale_v,test_case.descale_o,test_case.descale_doutput,
                         test_case.descale_s,test_case.descale_dp,test_case.scale_s,test_case.scale_dq,test_case.scale_dk,test_case.scale_dv,test_case.scale_dp}) inputs.push_back(scale_input(scalar,stream.get()));
  std::vector<functional::BoundTensor> actual,expected;
  for(const auto &tensor:{test_case.dq,test_case.dk,test_case.dv,test_case.amax_dq,test_case.amax_dk,test_case.amax_dv,test_case.amax_dp}) {
    actual.push_back(attention_output(tensor,stream.get()));expected.push_back(attention_output(tensor,stream.get()));
  }
  auto production_bindings=functional::bindings(inputs,actual),reference_bindings=functional::bindings(inputs,expected);
  // O and stats are shared acDNN-generated inputs, including their FP8
  // rounding. The separate forward comparison above still checks production.
  for (std::size_t index = 0; index < 2; ++index) {
    const auto &tensor = forward_outputs.expected[index];
    for (auto *bindings : {&production_bindings, &reference_bindings}) {
      bindings->push_back({tensor.specification.uid, tensor.buffer->data()});
    }
  }
  auto production=build_flagdnn_sdpa_fp8_backward(handle,test_case);
  auto reference=tv::make_acdnn_attention_reference(test_case,record);
  tv::DeviceBuffer workspace(production->workspace_size()),reference_workspace(reference->workspace_size());
  execute_production(*production,production_bindings,workspace,stream);
  functional::execute(*reference,reference_bindings,reference_workspace,stream);
  for(std::size_t index=0;index<actual.size();++index) {
    compare(actual[index],expected[index],index<3?test_case.gradient_absolute_tolerance:test_case.amax_absolute_tolerance,
            index<3?test_case.gradient_relative_tolerance:test_case.amax_relative_tolerance,stream,test_case.name+" output "+std::to_string(index));
  }
  if (benchmark)
    functional::measure_pair(test_case.name, *production, *reference,
                             production_bindings, reference_bindings, workspace,
                             reference_workspace, stream);
}

template <class Case>
int run(int argc, char **argv, std::span<const Case> cases,
        std::string_view operation, std::string_view suite,
        const char *filter_name, bool benchmark = false) {
  try {
    if (argc == 2 && std::string_view(argv[1]) == "--dump-cases") {
      for (const auto &t : cases)
        std::cout << operation << '\t' << t.name << '\n';
      return 0;
    }
    if(argc!=3 || cases.empty())throw std::invalid_argument("THEAD attention requires compiler arguments and cases");
    const auto catalog=tv::CapabilityCatalog::load(FLAGDNN_THEAD_ACDNN_CAPABILITY_CATALOG);
    const auto selected_cases = catalog.select_cases(operation, cases);
    cases = selected_cases;
    catalog.validate_versions(FLAGDNN_THEAD_PPU_SDK_VERSION,ACDNN_VERSION,static_cast<std::int64_t>(acdnnGetVersion()));
    tv::check_driver(cuInit(0),"cuInit");CUdevice device;
    tv::check_driver(cuDeviceGet(&device,0),"cuDeviceGet");tv::PrimaryContext primary(device);tv::ScopedCurrentContext current(primary.get());tv::DeviceStream stream;
    functional::TemporaryCache cache;flagdnn::Handle handle("thead",0);handle.set_compiler(argv[1],argv[2],cache.path().string());
    const char *filter=std::getenv(filter_name);std::size_t selected=0,executed=0,skipped=0;
    for(const auto &test_case:cases) {
      if(filter && *filter && test_case.name.find(filter)==std::string::npos)continue;
      ++selected;
      if constexpr(std::is_same_v<Case,SdpaTestCase>)validate_sdpa_case(test_case);
      else if constexpr(std::is_same_v<Case,SdpaBackwardTestCase>)validate_sdpa_backward_case(test_case);
      else if constexpr(std::is_same_v<Case,SdpaFp8TestCase>)validate_sdpa_fp8_case(test_case);
      else validate_sdpa_fp8_backward_case(test_case);
      const auto &record=catalog.lookup(operation,test_case.name);
      if(record.status==tv::CapabilityStatus::kUnsupported) {
        std::cout<<"[SKIP][acdnn] op="<<operation<<" case="<<test_case.name<<" reason="<<record.reason_code
                 <<" sdk="<<FLAGDNN_THEAD_PPU_SDK_VERSION<<" acdnn_header="<<ACDNN_VERSION<<" acdnn_runtime="<<acdnnGetVersion()
                 <<" target="<<handle.target_fingerprint()<<" dtype="<<functional::data_type_name(test_case.q.data_type)
                 <<" layout=contiguous shape="<<functional::shape_name(test_case.q)<<'\n';++skipped;continue;
      }
      if(record.status==tv::CapabilityStatus::kProbeRequired && !std::getenv("FLAGDNN_THEAD_QUALIFY_PROBES"))throw std::runtime_error("unqualified acDNN attention reference");
      if constexpr (std::is_same_v<Case, SdpaTestCase> ||
                    std::is_same_v<Case, SdpaFp8TestCase>)
        forward_case(test_case, handle, stream, record, benchmark);
      else
        backward_case(test_case, handle, stream, record, benchmark);
      ++executed;std::cout<<test_case.name<<": FlagDNN Graph/capture/replay vs acDNN PASS\n";
    }
    if(!selected || selected!=executed+skipped)throw std::runtime_error("THEAD attention case accounting mismatch");
    std::cout << suite << ": " << (executed ? "PASS" : "SKIP")
              << " cases=" << selected
              << (benchmark ? " comparable_executed=" : " executed=")
              << executed << (benchmark ? " reference_skipped=" : " skipped=")
              << skipped << '\n';
    return executed ? 0 : 77;
  }catch(const std::exception &error){std::cerr<<suite<<": FAIL reason="<<error.what()<<'\n';return 1;}
}
}  // namespace
std::unique_ptr<AttentionExecutable> build_sdpa_reference(const SdpaTestCase &test_case) {
  const auto catalog=tv::CapabilityCatalog::load(FLAGDNN_THEAD_ACDNN_CAPABILITY_CATALOG);
  return tv::make_acdnn_attention_reference(test_case,catalog.lookup("sdpa",test_case.name));
}
std::unique_ptr<AttentionExecutable> build_sdpa_backward_reference(const SdpaBackwardTestCase &test_case) {
  const auto catalog=tv::CapabilityCatalog::load(FLAGDNN_THEAD_ACDNN_CAPABILITY_CATALOG);
  return tv::make_acdnn_attention_reference(test_case,catalog.lookup("sdpa_backward",test_case.name));
}
int run_sdpa_functional_test(int argc,char **argv,std::span<const SdpaTestCase> cases) {
  return run(argc,argv,cases,"sdpa","FLAGDNN_SDPA_FUNCTIONAL","FLAGDNN_SDPA_CASE");
}
int run_sdpa_backward_functional_test(int argc,char **argv,std::span<const SdpaBackwardTestCase> cases) {
  return run(argc,argv,cases,"sdpa_backward","FLAGDNN_SDPA_BACKWARD_FUNCTIONAL","FLAGDNN_SDPA_BACKWARD_CASE");
}
std::unique_ptr<AttentionExecutable> build_sdpa_fp8_reference(const SdpaFp8TestCase &test_case) {
  const auto catalog=tv::CapabilityCatalog::load(FLAGDNN_THEAD_ACDNN_CAPABILITY_CATALOG);
  return tv::make_acdnn_attention_reference(test_case,catalog.lookup("sdpa_fp8",test_case.name));
}
int run_sdpa_fp8_functional_test(int argc,char **argv,std::span<const SdpaFp8TestCase> cases) {
  return run(argc,argv,cases,"sdpa_fp8","FLAGDNN_SDPA_FP8_FUNCTIONAL","FLAGDNN_SDPA_FP8_CASE");
}
std::unique_ptr<AttentionExecutable> build_sdpa_fp8_backward_reference(const SdpaFp8BackwardTestCase &test_case) {
  const auto catalog=tv::CapabilityCatalog::load(FLAGDNN_THEAD_ACDNN_CAPABILITY_CATALOG);
  return tv::make_acdnn_attention_reference(test_case,catalog.lookup("sdpa_fp8_backward",test_case.name));
}
int run_sdpa_fp8_backward_functional_test(int argc,char **argv,std::span<const SdpaFp8BackwardTestCase> cases) {
  return run(argc,argv,cases,"sdpa_fp8_backward","FLAGDNN_SDPA_FP8_BACKWARD_FUNCTIONAL","FLAGDNN_SDPA_FP8_BACKWARD_CASE");
}
int run_attention_benchmark_test(int argc, char **argv,
                                 AttentionBenchmarkOperation operation) {
  switch (operation) {
    case AttentionBenchmarkOperation::kForward:
      return run<SdpaTestCase>(argc, argv, make_sdpa_benchmark_cases(), "sdpa",
                               "FLAGDNN_SDPA_BENCHMARK", "FLAGDNN_SDPA_CASE",
                               true);
    case AttentionBenchmarkOperation::kBackward:
      return run<SdpaBackwardTestCase>(
          argc, argv, make_sdpa_backward_benchmark_cases(), "sdpa_backward",
          "FLAGDNN_SDPA_BACKWARD_BENCHMARK", "FLAGDNN_SDPA_BACKWARD_CASE",
          true);
    case AttentionBenchmarkOperation::kFp8Forward:
      return run<SdpaFp8TestCase>(argc, argv, make_sdpa_fp8_benchmark_cases(),
                                  "sdpa_fp8", "FLAGDNN_SDPA_FP8_BENCHMARK",
                                  "FLAGDNN_SDPA_FP8_CASE", true);
    case AttentionBenchmarkOperation::kFp8Backward:
      return run<SdpaFp8BackwardTestCase>(
          argc, argv, make_sdpa_fp8_backward_benchmark_cases(),
          "sdpa_fp8_backward", "FLAGDNN_SDPA_FP8_BACKWARD_BENCHMARK",
          "FLAGDNN_SDPA_FP8_BACKWARD_CASE", true);
  }
  throw std::invalid_argument("invalid attention benchmark operation");
}
}  // namespace flagdnn::testing
