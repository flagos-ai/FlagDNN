/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "cpu_cudnn_benchmark.hpp"
#include "common/matmul.hpp"
#include "common/convolution.hpp"
#include "common/reduction.hpp"
#include "reference/cpu/matrix.hpp"
#include <tuple>

namespace flagdnn::reference::tests {
void matrix(Suite& suite) {
  const auto matmuls=t::make_matmul_cases();
  for(auto type:{FLAGDNN_DATA_FLOAT32,FLAGDNN_DATA_FLOAT16,FLAGDNN_DATA_BFLOAT16}) {
    std::set<std::size_t> selected;
    for(auto c:matmuls) {
      const auto rank=c.output.dimensions.size();
      if(c.a.data_type!=type || rank<2 ||rank>3 ||selected.contains(rank) ||
         io::element_count(c.output)*c.a.dimensions.back()>500000 ||c.input_precision==2)continue;
      selected.insert(rank);
      if(type==FLAGDNN_DATA_FLOAT32)c.input_precision=1;
      suite.run(c.name,[&] {
        cpu::MatmulParameters p{c.a.dimensions,c.b.dimensions,c.output.dimensions,type,c.input_precision};
        const auto r = suite.check({c.a,c.b},{c.output},t::build_matmul_reference(c),
            [&](const Arrays& x){return Arrays{cpu::evaluate_matmul(p,x[0],x[1])};},
            c.absolute_tolerance,c.relative_tolerance);
        check_formula(formula_case(b::Operation::kMatmul, {c.a,c.b}, {c.output}),
                      r, c.absolute_tolerance, c.relative_tolerance);
      });
    }
    if(selected.size()!=2)throw std::runtime_error("missing rank 2/3 matmul selection");
  }
  for(auto direction:{t::ConvolutionDirection::kFprop,t::ConvolutionDirection::kDgrad,t::ConvolutionDirection::kWgrad}) {
    const auto cases=t::make_convolution_cases(direction);
    for(auto type:{FLAGDNN_DATA_FLOAT32,FLAGDNN_DATA_FLOAT16,FLAGDNN_DATA_BFLOAT16}) {
      std::set<bool> selected;
      for(auto c:cases) {
        const bool grouped=c.groups>1;
        if(c.x.data_type!=type ||c.x.dimensions.size()!=4 ||selected.contains(grouped) ||
           io::element_count(c.y)*(io::element_count(c.w)/c.w.dimensions[0])>1000000 ||
           c.input_precision==2)continue;
        selected.insert(grouped);
        if(type==FLAGDNN_DATA_FLOAT32)c.input_precision=1;
        suite.run(c.name,[&] {
          const auto d=direction==t::ConvolutionDirection::kFprop?cpu::ConvolutionDirection::kForward:
                       direction==t::ConvolutionDirection::kDgrad?cpu::ConvolutionDirection::kInputGradient:
                       cpu::ConvolutionDirection::kWeightGradient;
          cpu::ConvolutionParameters p{c.x.dimensions,c.w.dimensions,c.y.dimensions,
              c.stride,c.pre_padding,c.dilation,d,c.groups,c.input_precision,
              c.mode==t::ConvolutionMode::kConvolution};
          std::vector<t::TestTensor> inputs=direction==t::ConvolutionDirection::kFprop?
              std::vector{c.x,c.w}:direction==t::ConvolutionDirection::kDgrad?
              std::vector{c.y,c.w}:std::vector{c.y,c.x};
          const auto r = suite.check(inputs,{t::convolution_output_tensor(c)},t::build_convolution_reference(c),
              [&](const Arrays& x){
                Arrays semantic(3);
                if(direction==t::ConvolutionDirection::kFprop){semantic[0]=x[0];semantic[1]=x[1];}
                else if(direction==t::ConvolutionDirection::kDgrad){semantic[2]=x[0];semantic[1]=x[1];}
                else{semantic[2]=x[0];semantic[0]=x[1];}
                return Arrays{cpu::evaluate_convolution(p,semantic)};
              },c.absolute_tolerance,c.relative_tolerance);
          if (direction != t::ConvolutionDirection::kFprop) {
            auto spec = formula_case(direction == t::ConvolutionDirection::kDgrad ?
                b::Operation::kConvolutionDgrad : b::Operation::kConvolutionWgrad,
                inputs, {t::convolution_output_tensor(c)});
            spec.convolution.spatial_rank = static_cast<std::int32_t>(c.stride.size());
            spec.convolution.pre_padding = c.pre_padding;
            spec.convolution.post_padding = c.post_padding;
            spec.convolution.stride = c.stride;
            spec.convolution.dilation = c.dilation;
            spec.convolution.groups = c.groups;
            spec.convolution.mode = c.mode == t::ConvolutionMode::kConvolution ?
                b::ConvolutionMode::kConvolution : b::ConvolutionMode::kCrossCorrelation;
            check_formula(spec, r, c.absolute_tolerance, c.relative_tolerance);
          }
        });
      }
      if(selected.size()!=2)throw std::runtime_error("missing grouped/ungrouped conv selection");
    }
  }

  std::set<std::tuple<int, int, int, bool>> covered;
  for (const auto& c : t::make_reduction_cases()) {
    const auto key = std::tuple{int(c.mode), int(c.input.data_type),
                               c.axis, c.keep_dimensions};
    if (io::element_count(c.input) > 100000 || !covered.insert(key).second) continue;
    suite.run(c.name, [&] {
      const auto in = t::reduction_reference_input_tensor(c);
      const auto r = suite.check({in}, {c.output}, t::build_reduction_reference(c),
          [&](const Arrays& x) {
            cpu::ReductionParameters p{c.input.dimensions, c.axis, c.mode, false};
            const auto high = cpu::evaluate_reduction(p, x[0]);
            return Arrays{high};
          }, c.absolute_tolerance, c.relative_tolerance,
          {t::reduction_host_input(c)});
      const auto low = cpu::evaluate_reduction(
          {c.input.dimensions, c.axis, c.mode, true}, r.inputs[0]);
      compare(r.outputs[0], quantize(low, c.output.data_type),
              c.absolute_tolerance, c.relative_tolerance);
      auto spec = formula_case(b::Operation::kReduction, {in}, {c.output});
      spec.reduction_mode = c.mode;
      spec.reduction_axis = c.axis;
      spec.keep_dimensions = c.keep_dimensions;
      check_formula(spec, r, c.absolute_tolerance, c.relative_tolerance);
    });
  }
}

void matrix_tf32(Suite& suite) {
  std::set<std::size_t> selected;
  for (auto c : t::make_matmul_cases()) {
    const auto rank = c.output.dimensions.size();
    if (c.a.data_type != FLAGDNN_DATA_FLOAT32 || rank < 2 || rank > 3 ||
        c.a.dimensions.back() % 4 || c.output.dimensions.back() % 4 ||
        io::element_count(c.output) * c.a.dimensions.back() > 500000 ||
        !selected.insert(rank).second) continue;
    c.input_precision = 2;
    suite.run(c.name + "_cpu_tf32", [&] {
      cpu::MatmulParameters p{c.a.dimensions, c.b.dimensions,
          c.output.dimensions, c.a.data_type, 2};
      const auto r = suite.check({c.a, c.b}, {c.output}, t::build_matmul_reference(c),
          [&](const Arrays& x) { return Arrays{cpu::evaluate_matmul(p, x[0], x[1])}; },
          c.absolute_tolerance, c.relative_tolerance);
      p.input_precision = 0;  // Aligned rank-2/3 defaults to TF32 in this CPU API.
      compare(r.outputs[0], cpu::evaluate_matmul(p, r.inputs[0], r.inputs[1]),
              c.absolute_tolerance, c.relative_tolerance);
    });
  }
  if (selected.size() != 2) throw std::runtime_error("missing TF32 matmul ranks");
  for (const auto direction : {t::ConvolutionDirection::kFprop,
                                t::ConvolutionDirection::kDgrad,
                                t::ConvolutionDirection::kWgrad}) {
    bool selected_conv = false;
    for (auto c : t::make_convolution_cases(direction)) {
      if (c.x.data_type != FLAGDNN_DATA_FLOAT32 || c.x.dimensions.size() != 4 ||
          io::element_count(c.y) * (io::element_count(c.w) / c.w.dimensions[0]) > 1000000)
        continue;
      c.input_precision = 2;
      selected_conv = true;
      suite.run(c.name + "_cpu_tf32", [&] {
        const auto d = direction == t::ConvolutionDirection::kFprop ?
            cpu::ConvolutionDirection::kForward :
            direction == t::ConvolutionDirection::kDgrad ?
            cpu::ConvolutionDirection::kInputGradient : cpu::ConvolutionDirection::kWeightGradient;
        cpu::ConvolutionParameters p{c.x.dimensions, c.w.dimensions, c.y.dimensions,
            c.stride, c.pre_padding, c.dilation, d, c.groups, 2,
            c.mode == t::ConvolutionMode::kConvolution};
        const auto inputs = direction == t::ConvolutionDirection::kFprop ?
            std::vector{c.x, c.w} : direction == t::ConvolutionDirection::kDgrad ?
            std::vector{c.y, c.w} : std::vector{c.y, c.x};
        suite.check(inputs, {t::convolution_output_tensor(c)}, t::build_convolution_reference(c),
            [&](const Arrays& x) {
              Arrays semantic(3);
              if (direction == t::ConvolutionDirection::kFprop) {
                semantic[0] = x[0]; semantic[1] = x[1];
              } else if (direction == t::ConvolutionDirection::kDgrad) {
                semantic[2] = x[0]; semantic[1] = x[1];
              } else {
                semantic[2] = x[0]; semantic[0] = x[1];
              }
              return Arrays{cpu::evaluate_convolution(p, semantic)};
            }, c.absolute_tolerance, c.relative_tolerance);
      });
      break;
    }
    if (!selected_conv) throw std::runtime_error("missing TF32 convolution direction");
  }
}
}  // namespace flagdnn::reference::tests
