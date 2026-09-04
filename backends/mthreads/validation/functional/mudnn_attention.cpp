/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/attention.hpp"

#include "backends/mthreads/validation/functional/tensor_io_adapter.hpp"
#include "backends/mthreads/validation/mudnn_attention.hpp"
#include "backends/mthreads/validation/musa_driver.hpp"

#include <musa_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace flagdnn::testing {
namespace {

namespace mv = validation::mthreads;
namespace io = validation::mthreads::tensor_io;

using BindingMap = std::unordered_map<std::int64_t, void*>;

template <typename Operation>
class MudnnAttentionExecutable final : public AttentionExecutable {
 public:
  template <typename Descriptor>
  explicit MudnnAttentionExecutable(Descriptor descriptor)
      : operation_(std::move(descriptor)) {}

  [[nodiscard]] std::size_t workspace_size() const noexcept override {
    return operation_.workspace_size();
  }

  void execute(std::span<const flagdnnBinding_t> bindings,
               void* workspace,
               std::size_t workspace_size,
               flagdnnStream_t stream) override {
    operation_.execute(bindings, workspace, workspace_size, stream);
  }

 private:
  Operation operation_;
};

double attention_scale(const AttentionOptions& options,
                       const TestTensor& q) {
  return options.attention_scale.value_or(
      1.0F / std::sqrt(static_cast<float>(q.dimensions[3])));
}

bool causal_option(const AttentionOptions& options,
                   std::string_view provider) {
  if (options.diagonal_alignment != AttentionDiagonalAlignment::kTopLeft ||
      options.diagonal_band_left_bound.has_value() ||
      (options.diagonal_band_right_bound.has_value() &&
       *options.diagonal_band_right_bound != 0)) {
    throw std::invalid_argument(
        std::string(provider) +
        " supports only unmasked or top-left causal Attention");
  }
  return options.diagonal_band_right_bound.has_value();
}

std::optional<mv::TensorDescriptor> describe_optional(
    const std::optional<TestTensor>& tensor) {
  return tensor.has_value()
             ? std::optional<mv::TensorDescriptor>(
                   mv::describe_tensor(*tensor))
             : std::nullopt;
}

BindingMap parse_bindings(std::span<const flagdnnBinding_t> bindings,
                          std::span<const TestTensor* const> tensors,
                          std::string_view operation) {
  std::size_t expected = 0;
  for (const TestTensor* tensor : tensors) {
    expected += static_cast<std::size_t>(tensor != nullptr);
  }
  if (bindings.size() != expected) {
    throw std::invalid_argument(
        std::string(operation) + " binding count is invalid");
  }
  BindingMap result;
  result.reserve(expected);
  for (const flagdnnBinding_t& binding : bindings) {
    if (binding.uid <= 0 || binding.device_pointer == nullptr ||
        !result.emplace(binding.uid, binding.device_pointer).second) {
      throw std::invalid_argument(
          std::string(operation) + " binding is invalid");
    }
  }
  for (const TestTensor* tensor : tensors) {
    if (tensor != nullptr && result.find(tensor->uid) == result.end()) {
      throw std::invalid_argument(
          std::string(operation) + " binding UID is missing");
    }
  }
  return result;
}

void* pointer(const TestTensor& tensor, const BindingMap& bindings) {
  const auto found = bindings.find(tensor.uid);
  if (found == bindings.end()) {
    throw std::invalid_argument("FP8 Attention binding UID is missing");
  }
  return found->second;
}

std::size_t tensor_bytes(const TestTensor& tensor) {
  const std::size_t elements = io::storage_element_count(tensor);
  const std::size_t width = io::data_type_size(tensor.data_type);
  if (elements > std::numeric_limits<std::size_t>::max() / width) {
    throw std::overflow_error("FP8 Attention tensor bytes overflow");
  }
  return elements * width;
}

std::vector<float> read_logical(const TestTensor& tensor,
                                const BindingMap& bindings,
                                musaStream_t stream) {
  std::vector<std::uint8_t> encoded(tensor_bytes(tensor));
  mv::check_musa(musaMemcpyAsync(encoded.data(),
                                 pointer(tensor, bindings),
                                 encoded.size(),
                                 musaMemcpyDeviceToHost,
                                 stream),
                 "musaMemcpyAsync(FP8 oracle device-to-host)");
  mv::check_musa(musaStreamSynchronize(stream),
                 "musaStreamSynchronize(FP8 oracle input)");
  return io::gather(io::decode(encoded, tensor.data_type), tensor);
}

float read_scalar(const Fp8Scalar& scalar,
                  const BindingMap& bindings,
                  musaStream_t stream) {
  const std::vector<float> values =
      read_logical(scalar.tensor, bindings, stream);
  if (values.size() != 1 || !std::isfinite(values[0]) || values[0] <= 0.0F) {
    throw std::invalid_argument("FP8 Attention scale binding is invalid");
  }
  return values[0];
}

void write_logical(const TestTensor& tensor,
                   std::span<const float> logical,
                   const BindingMap& bindings,
                   musaStream_t stream) {
  const std::vector<float> physical = io::scatter(logical, tensor);
  const std::vector<std::uint8_t> encoded =
      io::encode(physical, tensor.data_type);
  mv::check_musa(musaMemcpyAsync(pointer(tensor, bindings),
                                 encoded.data(),
                                 encoded.size(),
                                 musaMemcpyHostToDevice,
                                 stream),
                 "musaMemcpyAsync(FP8 oracle host-to-device)");
  // The source storage is owned by this stack frame, so complete the copy
  // before it is released.  This oracle is validation-only and synchronous.
  mv::check_musa(musaStreamSynchronize(stream),
                 "musaStreamSynchronize(FP8 oracle output)");
}

std::size_t offset(const TestTensor& tensor,
                   std::int64_t b,
                   std::int64_t h,
                   std::int64_t s,
                   std::int64_t d) {
  return static_cast<std::size_t>(
      (((b * tensor.dimensions[1] + h) * tensor.dimensions[2] + s) *
       tensor.dimensions[3]) + d);
}

struct Fp8ForwardResult {
  std::vector<float> output;
  std::vector<float> stats;
  float amax_s = 0.0F;
  float amax_o = 0.0F;
};

Fp8ForwardResult fp8_forward(
    const SdpaFp8TestCase& test_case,
    std::span<const float> q,
    std::span<const float> k,
    std::span<const float> v,
    float descale_q,
    float descale_k,
    float descale_v,
    float descale_s,
    float scale_s,
    float scale_o) {
  const std::int64_t batch = test_case.q.dimensions[0];
  const std::int64_t query_heads = test_case.q.dimensions[1];
  const std::int64_t key_heads = test_case.k.dimensions[1];
  const std::int64_t value_heads = test_case.v.dimensions[1];
  const std::int64_t sequence_q = test_case.q.dimensions[2];
  const std::int64_t sequence_kv = test_case.k.dimensions[2];
  const std::int64_t head_dimension = test_case.q.dimensions[3];
  const std::int64_t value_dimension = test_case.v.dimensions[3];
  const double scale = attention_scale(test_case.options, test_case.q);
  const bool causal = causal_option(
      test_case.options, "MThreads FP8 mathematical oracle");
  Fp8ForwardResult result;
  result.output.resize(io::element_count(test_case.output));
  result.stats.resize(static_cast<std::size_t>(
      batch * query_heads * sequence_q));
  std::vector<double> scores(static_cast<std::size_t>(sequence_kv));
  std::vector<double> numerators(static_cast<std::size_t>(sequence_kv));

  for (std::int64_t b = 0; b < batch; ++b) {
    for (std::int64_t h = 0; h < query_heads; ++h) {
      const std::int64_t kh = h / (query_heads / key_heads);
      const std::int64_t vh = h / (query_heads / value_heads);
      for (std::int64_t m = 0; m < sequence_q; ++m) {
        double maximum = -std::numeric_limits<double>::infinity();
        for (std::int64_t n = 0; n < sequence_kv; ++n) {
          if (causal && n > m) {
            scores[static_cast<std::size_t>(n)] =
                -std::numeric_limits<double>::infinity();
            continue;
          }
          double score = 0.0;
          for (std::int64_t d = 0; d < head_dimension; ++d) {
            score += static_cast<double>(
                         q[offset(test_case.q, b, h, m, d)]) *
                     static_cast<double>(
                         k[offset(test_case.k, b, kh, n, d)]);
          }
          score *= static_cast<double>(descale_q) * descale_k * scale;
          scores[static_cast<std::size_t>(n)] = score;
          maximum = std::max(maximum, score);
        }
        double denominator = 0.0;
        for (std::int64_t n = 0; n < sequence_kv; ++n) {
          const double numerator =
              std::isfinite(scores[static_cast<std::size_t>(n)])
                  ? std::exp(scores[static_cast<std::size_t>(n)] - maximum)
                  : 0.0;
          numerators[static_cast<std::size_t>(n)] = numerator;
          denominator += numerator;
        }
        if (!(denominator > 0.0) || !std::isfinite(denominator)) {
          throw std::runtime_error("FP8 Attention oracle has an empty row");
        }
        result.stats[static_cast<std::size_t>(
            (b * query_heads + h) * sequence_q + m)] =
            static_cast<float>(maximum + std::log(denominator));
        result.amax_s = std::max(
            result.amax_s, static_cast<float>(1.0 / denominator));
        for (std::int64_t d = 0; d < value_dimension; ++d) {
          double accumulator = 0.0;
          for (std::int64_t n = 0; n < sequence_kv; ++n) {
            const float probability_raw = io::quantize_scalar(
                static_cast<float>(
                    numerators[static_cast<std::size_t>(n)] * scale_s),
                test_case.q.data_type);
            accumulator += static_cast<double>(probability_raw) *
                           static_cast<double>(
                               v[offset(test_case.v, b, vh, n, d)]);
          }
          const float output_value = static_cast<float>(
              accumulator * descale_s * descale_v / denominator);
          result.amax_o = std::max(result.amax_o, std::abs(output_value));
          result.output[offset(test_case.output, b, h, m, d)] =
              io::quantize_scalar(output_value * scale_o,
                                  test_case.output.data_type);
        }
      }
    }
  }
  return result;
}

class Fp8ForwardOracle final : public AttentionExecutable {
 public:
  explicit Fp8ForwardOracle(SdpaFp8TestCase test_case)
      : test_case_(std::move(test_case)) {
    validate_sdpa_fp8_case(test_case_);
    if (test_case_.bias.has_value()) {
      throw std::invalid_argument(
          "MThreads FP8 mathematical oracle does not support bias");
    }
    static_cast<void>(causal_option(
        test_case_.options, "MThreads FP8 mathematical oracle"));
  }

  [[nodiscard]] std::size_t workspace_size() const noexcept override {
    return 0;
  }

  void execute(std::span<const flagdnnBinding_t> raw_bindings,
               void*,
               std::size_t workspace_size,
               flagdnnStream_t opaque_stream) override {
    if (workspace_size != 0 || opaque_stream == nullptr) {
      throw std::invalid_argument(
          "FP8 Attention oracle execution contract is invalid");
    }
    const TestTensor* stats =
        test_case_.stats.has_value() ? &*test_case_.stats : nullptr;
    const std::vector<const TestTensor*> tensors{
        &test_case_.q,
        &test_case_.k,
        &test_case_.v,
        &test_case_.descale_q.tensor,
        &test_case_.descale_k.tensor,
        &test_case_.descale_v.tensor,
        &test_case_.descale_s.tensor,
        &test_case_.scale_s.tensor,
        &test_case_.scale_o.tensor,
        &test_case_.output,
        stats,
        &test_case_.amax_s,
        &test_case_.amax_o};
    const BindingMap bindings =
        parse_bindings(raw_bindings, tensors, "FP8 Attention oracle");
    const musaStream_t stream =
        reinterpret_cast<musaStream_t>(opaque_stream);
    const Fp8ForwardResult result = fp8_forward(
        test_case_,
        read_logical(test_case_.q, bindings, stream),
        read_logical(test_case_.k, bindings, stream),
        read_logical(test_case_.v, bindings, stream),
        read_scalar(test_case_.descale_q, bindings, stream),
        read_scalar(test_case_.descale_k, bindings, stream),
        read_scalar(test_case_.descale_v, bindings, stream),
        read_scalar(test_case_.descale_s, bindings, stream),
        read_scalar(test_case_.scale_s, bindings, stream),
        read_scalar(test_case_.scale_o, bindings, stream));
    write_logical(test_case_.output, result.output, bindings, stream);
    if (test_case_.stats.has_value()) {
      write_logical(*test_case_.stats, result.stats, bindings, stream);
    }
    const float amax_s[] = {result.amax_s};
    const float amax_o[] = {result.amax_o};
    write_logical(test_case_.amax_s, amax_s, bindings, stream);
    write_logical(test_case_.amax_o, amax_o, bindings, stream);
  }

 private:
  SdpaFp8TestCase test_case_;
};

struct Fp8BackwardResult {
  std::vector<float> dq;
  std::vector<float> dk;
  std::vector<float> dv;
  float amax_dq = 0.0F;
  float amax_dk = 0.0F;
  float amax_dv = 0.0F;
  float amax_dp = 0.0F;
};

class Fp8BackwardOracle final : public AttentionExecutable {
 public:
  explicit Fp8BackwardOracle(SdpaFp8BackwardTestCase test_case)
      : test_case_(std::move(test_case)) {
    validate_sdpa_fp8_backward_case(test_case_);
    static_cast<void>(causal_option(
        test_case_.options, "MThreads FP8 backward mathematical oracle"));
  }

  [[nodiscard]] std::size_t workspace_size() const noexcept override {
    return 0;
  }

  void execute(std::span<const flagdnnBinding_t> raw_bindings,
               void*,
               std::size_t workspace_size,
               flagdnnStream_t opaque_stream) override {
    if (workspace_size != 0 || opaque_stream == nullptr) {
      throw std::invalid_argument(
          "FP8 Attention backward oracle execution contract is invalid");
    }
    const std::vector<const TestTensor*> tensors{
        &test_case_.q,
        &test_case_.k,
        &test_case_.v,
        &test_case_.output,
        &test_case_.doutput,
        &test_case_.stats,
        &test_case_.descale_q.tensor,
        &test_case_.descale_k.tensor,
        &test_case_.descale_v.tensor,
        &test_case_.descale_o.tensor,
        &test_case_.descale_doutput.tensor,
        &test_case_.descale_s.tensor,
        &test_case_.descale_dp.tensor,
        &test_case_.scale_s.tensor,
        &test_case_.scale_dq.tensor,
        &test_case_.scale_dk.tensor,
        &test_case_.scale_dv.tensor,
        &test_case_.scale_dp.tensor,
        &test_case_.dq,
        &test_case_.dk,
        &test_case_.dv,
        &test_case_.amax_dq,
        &test_case_.amax_dk,
        &test_case_.amax_dv,
        &test_case_.amax_dp};
    const BindingMap bindings = parse_bindings(
        raw_bindings, tensors, "FP8 Attention backward oracle");
    const musaStream_t stream =
        reinterpret_cast<musaStream_t>(opaque_stream);
    Fp8BackwardResult result = calculate(bindings, stream);
    write_logical(test_case_.dq, result.dq, bindings, stream);
    write_logical(test_case_.dk, result.dk, bindings, stream);
    write_logical(test_case_.dv, result.dv, bindings, stream);
    const float amax_dq[] = {result.amax_dq};
    const float amax_dk[] = {result.amax_dk};
    const float amax_dv[] = {result.amax_dv};
    const float amax_dp[] = {result.amax_dp};
    write_logical(test_case_.amax_dq, amax_dq, bindings, stream);
    write_logical(test_case_.amax_dk, amax_dk, bindings, stream);
    write_logical(test_case_.amax_dv, amax_dv, bindings, stream);
    write_logical(test_case_.amax_dp, amax_dp, bindings, stream);
  }

 private:
  Fp8BackwardResult calculate(const BindingMap& bindings,
                              musaStream_t stream) const {
    const std::vector<float> q =
        read_logical(test_case_.q, bindings, stream);
    const std::vector<float> k =
        read_logical(test_case_.k, bindings, stream);
    const std::vector<float> v =
        read_logical(test_case_.v, bindings, stream);
    const std::vector<float> output =
        read_logical(test_case_.output, bindings, stream);
    const std::vector<float> doutput =
        read_logical(test_case_.doutput, bindings, stream);
    const std::vector<float> stats =
        read_logical(test_case_.stats, bindings, stream);
    const float descale_q =
        read_scalar(test_case_.descale_q, bindings, stream);
    const float descale_k =
        read_scalar(test_case_.descale_k, bindings, stream);
    const float descale_v =
        read_scalar(test_case_.descale_v, bindings, stream);
    const float descale_o =
        read_scalar(test_case_.descale_o, bindings, stream);
    const float descale_do =
        read_scalar(test_case_.descale_doutput, bindings, stream);
    const float descale_s =
        read_scalar(test_case_.descale_s, bindings, stream);
    const float descale_dp =
        read_scalar(test_case_.descale_dp, bindings, stream);
    const float scale_s =
        read_scalar(test_case_.scale_s, bindings, stream);
    const float scale_dq =
        read_scalar(test_case_.scale_dq, bindings, stream);
    const float scale_dk =
        read_scalar(test_case_.scale_dk, bindings, stream);
    const float scale_dv =
        read_scalar(test_case_.scale_dv, bindings, stream);
    const float scale_dp =
        read_scalar(test_case_.scale_dp, bindings, stream);
    const double attn_scale = attention_scale(test_case_.options,
                                               test_case_.q);
    const bool causal = causal_option(
        test_case_.options, "MThreads FP8 backward mathematical oracle");
    const std::int64_t batch = test_case_.q.dimensions[0];
    const std::int64_t query_heads = test_case_.q.dimensions[1];
    const std::int64_t kv_heads = test_case_.k.dimensions[1];
    const std::int64_t sequence_q = test_case_.q.dimensions[2];
    const std::int64_t sequence_kv = test_case_.k.dimensions[2];
    const std::int64_t dimension = test_case_.q.dimensions[3];
    const std::int64_t group = query_heads / kv_heads;

    Fp8BackwardResult result;
    result.dq.assign(io::element_count(test_case_.dq), 0.0F);
    result.dk.assign(io::element_count(test_case_.dk), 0.0F);
    result.dv.assign(io::element_count(test_case_.dv), 0.0F);
    std::vector<double> dk_accumulator(result.dk.size(), 0.0);
    std::vector<double> dv_accumulator(result.dv.size(), 0.0);
    std::vector<float> ds_quant(static_cast<std::size_t>(sequence_kv));
    std::vector<float> p_quant(static_cast<std::size_t>(sequence_kv));

    for (std::int64_t b = 0; b < batch; ++b) {
      for (std::int64_t h = 0; h < query_heads; ++h) {
        const std::int64_t kh = h / group;
        for (std::int64_t m = 0; m < sequence_q; ++m) {
          double row_delta = 0.0;
          for (std::int64_t d = 0; d < dimension; ++d) {
            row_delta += static_cast<double>(output[offset(
                             test_case_.output, b, h, m, d)]) *
                         static_cast<double>(doutput[offset(
                             test_case_.doutput, b, h, m, d)]);
          }
          row_delta *= static_cast<double>(descale_o) * descale_do;
          const double lse = stats[static_cast<std::size_t>(
              (b * query_heads + h) * sequence_q + m)];
          for (std::int64_t n = 0; n < sequence_kv; ++n) {
            if (causal && n > m) {
              ds_quant[static_cast<std::size_t>(n)] = 0.0F;
              p_quant[static_cast<std::size_t>(n)] = 0.0F;
              continue;
            }
            double score = 0.0;
            double dp = 0.0;
            for (std::int64_t d = 0; d < dimension; ++d) {
              score += static_cast<double>(q[offset(
                           test_case_.q, b, h, m, d)]) *
                       static_cast<double>(k[offset(
                           test_case_.k, b, kh, n, d)]);
              dp += static_cast<double>(doutput[offset(
                        test_case_.doutput, b, h, m, d)]) *
                    static_cast<double>(v[offset(
                        test_case_.v, b, kh, n, d)]);
            }
            score *= static_cast<double>(descale_q) * descale_k *
                     attn_scale;
            dp *= static_cast<double>(descale_do) * descale_v;
            const double probability = std::exp(score - lse);
            const float ds = static_cast<float>(
                probability * (dp - row_delta) * attn_scale);
            result.amax_dp = std::max(result.amax_dp, std::abs(ds));
            ds_quant[static_cast<std::size_t>(n)] = io::quantize_scalar(
                ds * scale_dp, test_case_.q.data_type);
            p_quant[static_cast<std::size_t>(n)] = io::quantize_scalar(
                static_cast<float>(probability) * scale_s,
                test_case_.q.data_type);
          }

          for (std::int64_t d = 0; d < dimension; ++d) {
            double dq = 0.0;
            for (std::int64_t n = 0; n < sequence_kv; ++n) {
              const float ds = ds_quant[static_cast<std::size_t>(n)];
              dq += static_cast<double>(ds) *
                    static_cast<double>(k[offset(
                        test_case_.k, b, kh, n, d)]);
              dk_accumulator[offset(test_case_.dk, b, kh, n, d)] +=
                  static_cast<double>(ds) *
                  static_cast<double>(q[offset(
                      test_case_.q, b, h, m, d)]);
              dv_accumulator[offset(test_case_.dv, b, kh, n, d)] +=
                  static_cast<double>(p_quant[static_cast<std::size_t>(n)]) *
                  static_cast<double>(doutput[offset(
                      test_case_.doutput, b, h, m, d)]);
            }
            const float unscaled =
                static_cast<float>(dq * descale_dp * descale_k);
            result.amax_dq = std::max(result.amax_dq, std::abs(unscaled));
            result.dq[offset(test_case_.dq, b, h, m, d)] =
                io::quantize_scalar(unscaled * scale_dq,
                                    test_case_.dq.data_type);
          }
        }
      }
    }
    for (std::size_t index = 0; index < result.dk.size(); ++index) {
      const float dk = static_cast<float>(
          dk_accumulator[index] * descale_dp * descale_q);
      const float dv = static_cast<float>(
          dv_accumulator[index] * descale_s * descale_do);
      result.amax_dk = std::max(result.amax_dk, std::abs(dk));
      result.amax_dv = std::max(result.amax_dv, std::abs(dv));
      result.dk[index] = io::quantize_scalar(
          dk * scale_dk, test_case_.dk.data_type);
      result.dv[index] = io::quantize_scalar(
          dv * scale_dv, test_case_.dv.data_type);
    }
    return result;
  }

  SdpaFp8BackwardTestCase test_case_;
};

}  // namespace

std::unique_ptr<AttentionExecutable> build_sdpa_reference(
    const SdpaTestCase& test_case) {
  validate_sdpa_case(test_case);
  return std::make_unique<
      MudnnAttentionExecutable<mv::MudnnSdpaOperation>>(
      mv::MudnnSdpaDescriptor{
          mv::describe_tensor(test_case.q),
          mv::describe_tensor(test_case.k),
          mv::describe_tensor(test_case.v),
          describe_optional(test_case.bias),
          mv::describe_tensor(test_case.output),
          describe_optional(test_case.stats),
          attention_scale(test_case.options, test_case.q),
          causal_option(test_case.options, "muDNN FlashAttention")});
}

std::unique_ptr<AttentionExecutable> build_sdpa_backward_reference(
    const SdpaBackwardTestCase& test_case) {
  validate_sdpa_backward_case(test_case);
  return std::make_unique<
      MudnnAttentionExecutable<mv::MudnnSdpaBackwardOperation>>(
      mv::MudnnSdpaBackwardDescriptor{
          mv::describe_tensor(test_case.q),
          mv::describe_tensor(test_case.k),
          mv::describe_tensor(test_case.v),
          describe_optional(test_case.bias),
          mv::describe_tensor(test_case.output),
          mv::describe_tensor(test_case.doutput),
          mv::describe_tensor(test_case.stats),
          mv::describe_tensor(test_case.dq),
          mv::describe_tensor(test_case.dk),
          mv::describe_tensor(test_case.dv),
          describe_optional(test_case.dbias),
          attention_scale(test_case.options, test_case.q),
          causal_option(test_case.options, "muDNN FlashAttention backward"),
          test_case.deterministic});
}

std::unique_ptr<AttentionExecutable> build_sdpa_fp8_reference(
    const SdpaFp8TestCase& test_case) {
  return std::make_unique<Fp8ForwardOracle>(test_case);
}

std::unique_ptr<AttentionExecutable> build_sdpa_fp8_backward_reference(
    const SdpaFp8BackwardTestCase& test_case) {
  return std::make_unique<Fp8BackwardOracle>(test_case);
}

}  // namespace flagdnn::testing
