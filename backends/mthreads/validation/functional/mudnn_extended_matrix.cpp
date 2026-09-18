/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include <array>

#include "backends/mthreads/validation/functional/mudnn_extended.hpp"

namespace flagdnn::testing::mthreads {
std::unique_ptr<TestExecutable> reference(const CausalConvolutionTestCase& c) {
  auto p = std::make_unique<NativeProgram>(c.inputs, c.outputs);
  const auto& shape = c.inputs[0].dimensions;
  const auto n = shape[0], channels = shape[1], length = shape[2],
             width = c.inputs[1].dimensions[1];
  const auto pad = (width - 1) * c.dilation, result_length = length + pad;
  const auto x = p->floating(0), w = p->floating(1);
  const auto image = p->view(x, {n, channels, 1, length},
                             {channels * length, length, length, 1});
  const auto filter =
      p->view(w, {channels, 1, 1, width}, {width, width, width, 1});
  const auto result = p->temporary({n, channels, 1, result_length});
  auto op = std::make_shared<md::Convolution>();
  mv::check_mudnn(op->SetGroups(static_cast<int>(channels)),
                  "muDNN causal convolution groups");
  mv::check_mudnn(op->SetNdInfo({0, static_cast<int>(pad)}, {1, 1},
                                {1, static_cast<int>(c.dilation)}),
                  "muDNN causal convolution geometry");
  mv::check_mudnn(op->SetComputeMode(md::Convolution::ComputeMode::SCALAR),
                  "muDNN causal convolution precision");
  p->add([r = p.get(), op, image, filter, result] {
    mv::check_mudnn(
        op->Run(r->handle(), r->tensor(result), r->tensor(image),
                r->tensor(filter), md::Convolution::Algorithm::IMPLICIT_GEMM,
                r->maintainer()),
        "muDNN causal Convolution::Run");
  });
  auto y =
      p->view(result, shape, {channels * result_length, result_length, 1});
  if (c.inputs.size() == 3) {
    const auto bias =
        p->view(p->floating(2), {1, channels, 1}, {channels, 1, 1});
    y = p->binary(Binary::ADD, y, bias);
  }
  if (c.silu) y = p->unary(Unary::SILU, y);
  p->unary_to(p->output(0), Unary::IDENTITY, y);
  return p;
}
std::unique_ptr<TestExecutable> reference(const Fp8MatmulTestCase& c) {
  if (c.scale_mode == 2)
    throw mv::ReferenceUnsupported(
        "muDNN 3.1.5 has no E8M0 scale type for MXFP8 GEMM");
  auto p = std::make_unique<NativeProgram>(c.inputs, c.outputs);
  // RunLt requires row-major operands. Copying uses native muDNN and retains
  // FP8 storage; it does not replace FP8 contraction with floating GEMM.
  std::array<std::size_t, 2> operands;
  for (std::size_t i = 0; i < 2; ++i) {
    operands[i] = p->temporary(c.inputs[i].dimensions, c.inputs[i].data_type);
    p->unary_to(operands[i], Unary::IDENTITY, i);
  }
  auto op = std::make_shared<md::BatchMatMul>();
  auto param = std::make_shared<md::MatMulLtParam>();
  mv::check_mudnn(op->SetComputeMode(md::BatchMatMul::ComputeMode::TENSOR),
                  "muDNN FP8 compute mode");
  mv::check_mudnn(op->SetTranspose(false, false), "muDNN FP8 transpose");
  mv::check_mudnn(op->SetAlpha(1.0), "muDNN FP8 alpha");
  mv::check_mudnn(op->SetBeta(0.0), "muDNN FP8 beta");
  const auto result =
      p->temporary(c.outputs[0].dimensions, c.outputs[0].data_type);
  p->add([r = p.get(), op, param, operands, result,
          scaled = c.scale_mode == 1] {
    md::Tensor empty;
    if (scaled)
      mv::check_mudnn(
          param->SetScale(r->tensor(2), r->tensor(3), empty, empty),
          "muDNN FP8 scales");
    mv::check_mudnn(op->RunLt(r->handle(), r->tensor(result),
                              r->tensor(operands[0]), r->tensor(operands[1]),
                              empty, empty, *param, r->maintainer()),
                    "muDNN FP8 BatchMatMul::RunLt");
  });
  p->unary_to(p->output(0), Unary::IDENTITY, result);
  return p;
}
std::unique_ptr<TestExecutable> reference(const MoeMatmulTestCase& c) {
  auto p = std::make_unique<NativeProgram>(c.inputs, c.outputs);
  const auto experts = static_cast<std::int64_t>(c.offsets.size());
  const auto routed = static_cast<std::int64_t>(c.token_index.size());
  const auto k =
      c.backward ? c.inputs[1].dimensions[2] : c.inputs[0].dimensions[2];
  const auto n =
      c.backward ? c.inputs[0].dimensions[2] : c.inputs[1].dimensions[2];
  const auto type = c.inputs[0].data_type;
  const auto item_size =
      type == FLAGDNN_DATA_FLOAT16 || type == FLAGDNN_DATA_BFLOAT16 ? 2U : 1U;
  const auto out_size =
      c.outputs[0].data_type == FLAGDNN_DATA_FLOAT32 ? 4U : 2U;
  std::size_t tokens = c.backward ? 1 : 0;
  if (c.mode == 1) {
    const auto input = p->view(0, {c.inputs[0].dimensions[1], k}, {k, 1});
    const auto index = p->view(3, {routed}, {1});
    tokens = p->temporary({routed, k}, type);
    auto gather = std::make_shared<md::GatherX>();
    mv::check_mudnn(gather->SetMode(md::GatherX::Mode::GATHER),
                    "muDNN MoE Gather mode");
    mv::check_mudnn(gather->SetAxis(0), "muDNN MoE Gather axis");
    p->add([r = p.get(), gather, input, index, tokens] {
      mv::check_mudnn(gather->Run(r->handle(), r->tensor(tokens),
                                  r->tensor(index), r->tensor(input)),
                      "muDNN MoE GatherX::Run");
    });
  }
  const auto result = c.mode == 2 ? p->temporary(c.outputs[0].dimensions,
                                                 c.outputs[0].data_type)
                                  : p->output(0);
  auto zero = std::make_shared<md::Fill>();
  mv::check_mudnn(zero->SetValue(0.0), "muDNN MoE zero value");
  p->add([r = p.get(), zero, result] {
    mv::check_mudnn(zero->Run(r->handle(), r->tensor(result)),
                    "muDNN MoE zero empty experts");
  });
  std::vector<std::array<std::size_t, 3>> groups;
  for (std::int64_t e = 0; e < experts; ++e) {
    const std::int64_t begin = c.offsets[e],
                       end = e + 1 < experts ? c.offsets[e + 1] : routed;
    const auto count = end - begin;
    if (!count) continue;
    if (c.backward) {
      groups.push_back(
          {p->view(tokens, {count, k}, {k, 1}, begin * k * item_size),
           p->view(0, {count, n}, {n, 1}, begin * n * item_size),
           p->view(result, {k, n}, {n, 1}, e * k * n * out_size)});
    } else {
      groups.push_back(
          {p->view(tokens, {count, k}, {k, 1}, begin * k * item_size),
           p->view(1, {k, n}, {n, 1}, e * k * n * item_size),
           p->view(result, {count, n}, {n, 1}, begin * n * out_size)});
    }
  }
  auto op = std::make_shared<md::GroupedMatMul>();
  mv::check_mudnn(op->SetComputeMode(md::GroupedMatMul::ComputeMode::TENSOR),
                  "muDNN MoE compute mode");
  mv::check_mudnn(op->SetTranspose(c.backward, false), "muDNN MoE transpose");
  mv::check_mudnn(op->SetAlpha(1.0), "muDNN MoE alpha");
  mv::check_mudnn(op->SetBeta(0.0), "muDNN MoE beta");
  const auto params = std::shared_ptr<md::MatMulLtParam[]>(
      new md::MatMulLtParam[groups.size()]);
  // Reuse descriptor arrays for the lifetime of the native executable.
  auto descriptors =
      std::make_shared<std::array<std::vector<md::Tensor>, 4>>();
  for (auto& tensors : *descriptors) tensors.resize(groups.size());
  p->add([r = p.get(), op, params, groups, descriptors, fp8 = item_size == 1] {
    auto& [a, b, d, empty] = *descriptors;
    for (std::size_t i = 0; i < groups.size(); ++i) {
      a[i] = r->tensor(groups[i][0]);
      b[i] = r->tensor(groups[i][1]);
      d[i] = r->tensor(groups[i][2]);
    }
    const auto status =
        fp8 ? op->RunLt(r->handle(), d.data(), a.data(), b.data(),
                        empty.data(), empty.data(), params.get(),
                        static_cast<int>(groups.size()), r->maintainer())
            : op->Run(r->handle(), d.data(), a.data(), b.data(),
                      static_cast<int>(groups.size()), r->maintainer());
    mv::check_mudnn(status, "muDNN MoE GroupedMatMul::Run/RunLt");
  });
  if (c.mode == 2) {
    const auto index = p->temporary({1, routed, 1}, FLAGDNN_DATA_INT32);
    const auto multiply = std::make_shared<md::Unary>();
    mv::check_mudnn(multiply->SetMode(Unary::MUL),
                    "muDNN MoE scatter slot mode");
    mv::check_mudnn(multiply->SetAlpha(static_cast<std::int64_t>(c.top_k)),
                    "muDNN MoE scatter slot factor");
    p->add([r = p.get(), multiply, index] {
      mv::check_mudnn(
          multiply->Run(r->handle(), r->tensor(index), r->tensor(3)),
          "muDNN MoE scatter slot");
    });
    p->binary_to(index, Binary::ADD, index, 4);
    const auto index_view = p->view(index, {routed, 1}, {1, 1});
    const auto update = p->view(result, {routed, n}, {n, 1});
    const auto output = p->view(p->output(0), {routed, n}, {n, 1});
    auto scatter = std::make_shared<md::ScatterND>();
    mv::check_mudnn(scatter->SetMode(md::ScatterND::Mode::UPDATE_ONLY),
                    "muDNN MoE scatter mode");
    p->add([r = p.get(), scatter, index_view, update, output] {
      mv::check_mudnn(
          scatter->Run(r->handle(), r->tensor(output), r->tensor(index_view),
                       r->tensor(update), r->maintainer()),
          "muDNN MoE ScatterND::Run");
    });
  }
  return p;
}
}  // namespace flagdnn::testing::mthreads
