/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "cpu_cudnn.hpp"
#include "validation/functional/cudnn_extended.hpp"
#include "reference/cpu/causal_convolution.hpp"
#include "reference/cpu/fp8_matmul.hpp"
#include "reference/cpu/moe_matmul.hpp"
#include "reference/cpu/resample.hpp"
#include "reference/cpu/statistics.hpp"
#include <tuple>

namespace flagdnn::reference::tests {
void extended(Suite& suite, const std::string& family) {
  if (family == "statistics") {
    const auto cases = t::make_bn_finalize_cases();
    std::set<std::tuple<std::string, int, std::size_t>> covered;
    for (const auto& c : cases) {
      const auto key = std::tuple{c.operation, int(c.inputs[0].data_type), c.outputs.size()};
      if (io::element_count(c.inputs[0]) > 100000 || !covered.insert(key).second) continue;
      suite.run(c.name, [&] {
        auto values = inputs_for(c.inputs);
        for (std::size_t i = 0; i < values.size(); ++i)
          for (std::size_t j = 0; j < values[i].size(); ++j)
            values[i][j] = t::statistics_input_value(c, i, j);
        const auto r = suite.check(c.inputs, c.outputs, t::build_cudnn_statistics(c),
            [&](const Arrays& x) {
              return cpu::evaluate_bn_finalize(c.outputs.size(), c.epsilon, c.accum_count, c.momentum, x);
            }, 2e-4, 2e-4, values);
        if (c.operation == "bn_finalize") {
          const Arrays without_running(r.inputs.begin(), r.inputs.begin() + 4);
          const auto four_outputs = cpu::evaluate_bn_finalize(4, c.epsilon, c.accum_count, c.momentum,
              without_running);
          for (std::size_t i = 0; i < four_outputs.size(); ++i)
            compare(r.outputs[i], quantize(four_outputs[i], c.outputs[i].data_type), 2e-4, 2e-4);
        }
      });
    }
  } else if (family == "resample") {
    std::set<std::tuple<int, int, int, std::size_t, bool>> covered;
    for (const auto& c : t::make_resample_cases()) {
      const auto key = std::tuple{c.mode, c.padding, int(c.inputs[0].data_type),
                                 c.outputs.size(), c.align_corners};
      if (io::element_count(c.outputs[0]) > 100000 || !covered.insert(key).second) continue;
      suite.run(c.name, [&] {
        cpu::ResampleParameters p{c.inputs[0].dimensions, c.outputs[0].dimensions,
            c.window, c.stride, c.pre, c.post, c.outputs.size(), c.mode,
            c.padding, c.align_corners};
        suite.check(c.inputs, c.outputs, t::build_cudnn_resample(c),
            [&](const Arrays& x) { return cpu::evaluate_resample(p, x); },
            c.inputs[0].data_type == FLAGDNN_DATA_BFLOAT16 ? 8e-3 : 1e-3,
            c.inputs[0].data_type == FLAGDNN_DATA_BFLOAT16 ? 8e-3 : 1e-3,
            t::resample_inputs(c));
      });
    }
  } else if (family == "causal_convolution") {
    std::set<std::tuple<int, int, bool, std::size_t, std::int64_t>> covered;
    for (const auto& c : t::make_causal_convolution_cases()) {
      const auto key = std::tuple{int(c.inputs[0].data_type), c.precision, c.silu,
                                 c.inputs.size(), c.dilation};
      if (io::element_count(c.outputs[0]) > 100000 || !covered.insert(key).second) continue;
      suite.run(c.name, [&] {
        cpu::CausalConvolutionParameters p{c.inputs[0].dimensions,
            c.inputs[1].dimensions.back(), c.dilation, c.precision, c.silu};
        const double tol = c.inputs[0].data_type == FLAGDNN_DATA_BFLOAT16 ? 8e-3 : 1e-3;
        suite.check(c.inputs, c.outputs, t::build_cudnn_causal_convolution(c),
            [&](const Arrays& x) { return cpu::evaluate_causal_convolution(p, x); },
            tol, tol, t::causal_convolution_inputs(c));
      });
    }
  } else if (family == "fp8_matmul") {
    std::set<std::tuple<int, int, int, bool>> covered;
    for (const auto& c : t::make_fp8_matmul_cases()) {
      const auto key = std::tuple{int(c.inputs[0].data_type), c.scale_mode,
                                 c.scale_variant, c.plain_matmul};
      if (io::element_count(c.outputs[0]) > 100000 || !covered.insert(key).second) continue;
      suite.run(c.name, [&] {
        cpu::Fp8MatmulParameters p{shapes(c.inputs), c.outputs[0].dimensions, c.scale_mode};
        suite.check(c.inputs, c.outputs, t::build_cudnn_fp8_matmul(c),
            [&](const Arrays& x) { return cpu::evaluate_fp8_matmul(p, x); },
            2e-3, 2e-3, t::fp8_matmul_inputs(c));
      });
    }
  } else if (family == "moe_matmul") {
    {
      std::set<std::tuple<int, int, int>> covered;
      for (const auto& c : t::make_moe_matmul_cases()) {
        const auto key = std::tuple{int(c.inputs[0].data_type), c.mode, c.top_k};
        if (io::element_count(c.outputs[0]) > 100000 || !covered.insert(key).second) continue;
        suite.run(c.name, [&] {
          cpu::MoeMatmulParameters p{shapes(c.inputs), c.outputs[0].dimensions,
              c.offsets, c.token_index, c.token_ks, c.mode, c.top_k};
          const double tol = c.inputs[0].data_type == FLAGDNN_DATA_BFLOAT16 ? 1e-2 : 2e-3;
          suite.check(c.inputs, c.outputs, t::build_cudnn_moe_matmul(c),
              [&](const Arrays& x) { return cpu::evaluate_moe_matmul(p, x); },
              tol, tol, t::moe_matmul_inputs(c));
        });
      }
    }
  } else {
    throw std::invalid_argument("unknown family: " + family);
  }
}
}  // namespace flagdnn::reference::tests
