/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "backends/mthreads/validation/mudnn_attention.hpp"

#include <cmath>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "backends/mthreads/validation/functional/mudnn_extended.hpp"
#include "common/attention.hpp"

namespace flagdnn::testing {
namespace {
namespace mv = validation::mthreads;
namespace md = musa::dnn;

template <typename Operation>
class MudnnAttentionExecutable final : public AttentionExecutable {
 public:
  template <typename Descriptor>
  explicit MudnnAttentionExecutable(Descriptor descriptor)
      : operation_(std::move(descriptor)) {}
  std::size_t workspace_size() const noexcept override {
    return operation_.workspace_size();
  }
  void execute(std::span<const flagdnnBinding_t> bindings, void* workspace,
               std::size_t workspace_size, flagdnnStream_t stream) override {
    operation_.execute(bindings, workspace, workspace_size, stream);
  }

 private:
  Operation operation_;
};

double attention_scale(const AttentionOptions& options, const TestTensor& q) {
  return options.attention_scale.value_or(
      1.0F / std::sqrt(static_cast<float>(q.dimensions[3])));
}

bool causal_option(const AttentionOptions& options) {
  if (options.diagonal_alignment != AttentionDiagonalAlignment::kTopLeft ||
      options.diagonal_band_left_bound.has_value() ||
      (options.diagonal_band_right_bound.has_value() &&
       *options.diagonal_band_right_bound != 0))
    throw mv::ReferenceUnsupported(
        "muDNN SDPA exposes unmasked or top-left causal attention only");
  return options.diagonal_band_right_bound.has_value();
}

std::optional<mv::TensorDescriptor> describe_optional(
    const std::optional<TestTensor>& tensor) {
  return tensor ? std::optional(mv::describe_tensor(*tensor)) : std::nullopt;
}

md::Tensor empty_tensor(md::Tensor::Type type) {
  md::Tensor result;
  const std::int64_t dims[] = {0}, strides[] = {1};
  mv::check_mudnn(result.SetType(type), "muDNN SDPA empty type");
  mv::check_mudnn(result.SetAddr(nullptr), "muDNN SDPA empty address");
  mv::check_mudnn(result.SetNdInfo(1, dims, strides),
                  "muDNN SDPA empty shape");
  return result;
}

// RunMath supports FP32 but requires matching Q/K/V head counts. Its public
// probability output permits a native backward; LOGSUMEXP supplies Graph's
// saved statistics without a CPU oracle.
std::unique_ptr<AttentionExecutable> math_reference(
    std::vector<TestTensor> inputs, std::vector<TestTensor> outputs,
    const AttentionOptions& options, bool backward, bool has_bias,
    bool has_stats, bool deterministic = false) {
  const auto& q = inputs[0];
  const auto& k = inputs[1];
  const auto& v = inputs[2];
  if (q.dimensions[1] != k.dimensions[1] || q.dimensions[1] != v.dimensions[1])
    throw mv::ReferenceUnsupported(
        "muDNN 3.1.5 RunMath requires equal Q/K/V heads; RunFlash rejects "
        "FP32");
  if (q.dimensions[3] != v.dimensions[3])
    throw mv::ReferenceUnsupported(
        "muDNN 3.1.5 RunMath requires equal Q/K and V head dimensions; "
        "RunFlash rejects FP32");
  const bool causal = causal_option(options);
  if (causal && has_stats)
    throw mv::ReferenceUnsupported(
        "muDNN 3.1.5 RunMath has no causal FP32 logsumexp output");
  auto p = std::make_unique<mthreads::NativeProgram>(inputs, outputs);
  auto op = std::make_shared<md::ScaledDotProductAttention>();
  const double scale = attention_scale(options, q);
  mv::check_mudnn(op->SetEmbedDim(q.dimensions[1] * v.dimensions[3]),
                  "muDNN SDPA math embed dimension");
  mv::check_mudnn(op->SetHeadsNum(q.dimensions[1]), "muDNN SDPA math heads");
  mv::check_mudnn(op->SetCausal(causal), "muDNN SDPA math causal");
  mv::check_mudnn(op->SetMaskMode(false), "muDNN SDPA math mask mode");
  mv::check_mudnn(op->SetKeyFormat(false), "muDNN SDPA math key format");
  mv::check_mudnn(op->SetScale(scale), "muDNN SDPA math scale");
  mv::check_mudnn(op->SetDropoutP(0.0), "muDNN SDPA math dropout");
  mv::check_mudnn(op->SetIsDeterministic(deterministic),
                  "muDNN SDPA math deterministic");
  const mthreads::Shape scores_shape{q.dimensions[0], q.dimensions[1],
                                     q.dimensions[2], k.dimensions[2]};
  const auto probabilities = p->temporary(scores_shape);
  const auto output =
      backward ? p->temporary(inputs[3].dimensions) : p->output(0);
  const auto bias_index = inputs.size() - 1;
  p->add([r = p.get(), op, probabilities, output, has_bias, bias_index] {
    auto mask = empty_tensor(md::Tensor::Type::FLOAT);
    auto dropout = empty_tensor(md::Tensor::Type::BOOL);
    mv::check_mudnn(
        op->RunMath(r->handle(), r->tensor(output), r->tensor(probabilities),
                    r->tensor(0), r->tensor(1), r->tensor(2),
                    has_bias ? r->tensor(bias_index) : mask, dropout,
                    r->maintainer()),
        "muDNN SDPA::RunMath");
  });
  if (backward) {
    const auto probability_gradient = p->temporary(scores_shape);
    auto zero = std::make_shared<md::Fill>();
    mv::check_mudnn(zero->SetValue(0.0),
                    "muDNN SDPA probability gradient zero");
    p->add([r = p.get(), zero, probability_gradient] {
      mv::check_mudnn(zero->Run(r->handle(), r->tensor(probability_gradient)),
                      "muDNN SDPA probability gradient Fill::Run");
    });
    p->add([r = p.get(), op, probabilities, probability_gradient] {
      auto dropout = empty_tensor(md::Tensor::Type::BOOL);
      mv::check_mudnn(
          op->RunMathBwd(r->handle(), r->tensor(r->output(0)),
                         r->tensor(r->output(1)), r->tensor(r->output(2)),
                         r->tensor(probability_gradient), r->tensor(4),
                         r->tensor(0), r->tensor(1), r->tensor(2),
                         r->tensor(probabilities), dropout, r->maintainer()),
          "muDNN SDPA::RunMathBwd");
    });
  } else if (has_stats) {
    auto matmul = std::make_shared<md::BatchMatMul>();
    mv::check_mudnn(matmul->SetTranspose(false, true),
                    "muDNN SDPA scores transpose");
    mv::check_mudnn(
        matmul->SetComputeMode(md::BatchMatMul::ComputeMode::SCALAR),
        "muDNN SDPA scores precision");
    mv::check_mudnn(matmul->SetAlpha(scale), "muDNN SDPA scores scale");
    mv::check_mudnn(matmul->SetBeta(0.0), "muDNN SDPA scores beta");
    auto scores = p->temporary(scores_shape);
    const auto collapse_heads = [&](std::size_t tensor) {
      const auto shape = p->descriptor(tensor).dimensions;
      return p->view(tensor, {shape[0] * shape[1], shape[2], shape[3]},
                     {shape[2] * shape[3], shape[3], 1});
    };
    const auto q_matrix = collapse_heads(0), k_matrix = collapse_heads(1),
               score_matrix = collapse_heads(scores);
    p->add([r = p.get(), matmul, q_matrix, k_matrix, score_matrix] {
      mv::check_mudnn(matmul->Run(r->handle(), r->tensor(score_matrix),
                                  r->tensor(q_matrix), r->tensor(k_matrix),
                                  r->maintainer()),
                      "muDNN SDPA scores BatchMatMul::Run");
    });
    if (has_bias)
      scores = p->binary(mthreads::Binary::ADD, scores, bias_index);
    auto logsumexp = std::make_shared<md::Softmax>();
    mv::check_mudnn(logsumexp->SetDim(-1), "muDNN SDPA logsumexp axis");
    mv::check_mudnn(logsumexp->SetMode(md::Softmax::Mode::LOGSUMEXP),
                    "muDNN SDPA logsumexp mode");
    p->add([r = p.get(), logsumexp, scores] {
      mv::check_mudnn(logsumexp->Run(r->handle(), r->tensor(r->output(1)),
                                     r->tensor(scores), r->maintainer()),
                      "muDNN SDPA logsumexp");
    });
  }
  return p;
}
}  // namespace

std::unique_ptr<AttentionExecutable> build_sdpa_reference(
    const SdpaTestCase& c) {
  validate_sdpa_case(c);
  if (c.q.data_type == FLAGDNN_DATA_FLOAT32) {
    std::vector<TestTensor> inputs{c.q, c.k, c.v}, outputs{c.output};
    if (c.bias) inputs.push_back(*c.bias);
    if (c.stats) outputs.push_back(*c.stats);
    return math_reference(inputs, outputs, c.options, false,
                          c.bias.has_value(), c.stats.has_value());
  }
  return std::make_unique<MudnnAttentionExecutable<mv::MudnnSdpaOperation>>(
      mv::MudnnSdpaDescriptor{
          mv::describe_tensor(c.q), mv::describe_tensor(c.k),
          mv::describe_tensor(c.v), describe_optional(c.bias),
          mv::describe_tensor(c.output), describe_optional(c.stats),
          attention_scale(c.options, c.q), causal_option(c.options)});
}

std::unique_ptr<AttentionExecutable> build_sdpa_backward_reference(
    const SdpaBackwardTestCase& c) {
  validate_sdpa_backward_case(c);
  if (c.dbias)
    throw mv::ReferenceUnsupported(
        "muDNN 3.1.5 SDPA backward has no dBias output");
  if (c.q.data_type == FLAGDNN_DATA_FLOAT32) {
    std::vector<TestTensor> inputs{c.q,      c.k,       c.v,
                                   c.output, c.doutput, c.stats};
    if (c.bias) inputs.push_back(*c.bias);
    return math_reference(inputs, {c.dq, c.dk, c.dv}, c.options, true,
                          c.bias.has_value(), false, c.deterministic);
  }
  return std::make_unique<
      MudnnAttentionExecutable<mv::MudnnSdpaBackwardOperation>>(
      mv::MudnnSdpaBackwardDescriptor{
          mv::describe_tensor(c.q), mv::describe_tensor(c.k),
          mv::describe_tensor(c.v), describe_optional(c.bias),
          mv::describe_tensor(c.output), mv::describe_tensor(c.doutput),
          mv::describe_tensor(c.stats), mv::describe_tensor(c.dq),
          mv::describe_tensor(c.dk), mv::describe_tensor(c.dv),
          describe_optional(c.dbias), attention_scale(c.options, c.q),
          causal_option(c.options), c.deterministic});
}

std::unique_ptr<AttentionExecutable> build_sdpa_fp8_reference(
    const SdpaFp8TestCase& c) {
  validate_sdpa_fp8_case(c);
  throw mv::ReferenceUnsupported(
      "muDNN 3.1.5 RunFlash and RunMath reject FP8 Q/K/V and expose no FP8 "
      "scales/amax");
}
std::unique_ptr<AttentionExecutable> build_sdpa_fp8_backward_reference(
    const SdpaFp8BackwardTestCase& c) {
  validate_sdpa_fp8_backward_case(c);
  throw mv::ReferenceUnsupported(
      "muDNN 3.1.5 SDPA backward has no FP8 Q/K/V, scales or amax outputs");
}
}  // namespace flagdnn::testing
