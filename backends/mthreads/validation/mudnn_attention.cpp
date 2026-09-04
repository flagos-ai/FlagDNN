/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "backends/mthreads/validation/mudnn_attention.hpp"

#include "backends/mthreads/validation/musa_driver.hpp"
#include "backends/mthreads/validation/tensor_io.hpp"

#include <mudnn.h>
#include <musa_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace flagdnn::validation::mthreads {
namespace {

constexpr std::size_t kWorkspaceAlignment = 256;
constexpr std::size_t kMudnnScratchBytes = 64U * 1024U * 1024U;

using BindingMap = std::unordered_map<std::int64_t, void*>;

bool is_io_type(flagdnnDataType_t data_type) noexcept {
  return data_type == FLAGDNN_DATA_FLOAT16 ||
         data_type == FLAGDNN_DATA_BFLOAT16;
}

musa::dnn::Tensor::Type mudnn_data_type(flagdnnDataType_t data_type) {
  switch (data_type) {
    case FLAGDNN_DATA_FLOAT32:
      return musa::dnn::Tensor::Type::FLOAT;
    case FLAGDNN_DATA_FLOAT16:
      return musa::dnn::Tensor::Type::HALF;
    case FLAGDNN_DATA_BFLOAT16:
      return musa::dnn::Tensor::Type::BFLOAT16;
    case FLAGDNN_DATA_BOOLEAN:
      return musa::dnn::Tensor::Type::BOOL;
    case FLAGDNN_DATA_FP8_E4M3:
    case FLAGDNN_DATA_FP8_E5M2:
      break;
  }
  throw std::invalid_argument("muDNN SDPA tensor type is unsupported");
}

std::size_t checked_add(std::size_t left,
                        std::size_t right,
                        std::string_view description) {
  if (left > std::numeric_limits<std::size_t>::max() - right) {
    throw std::overflow_error(std::string(description) + " overflows");
  }
  return left + right;
}

std::size_t align_up(std::size_t value) {
  const std::size_t remainder = value % kWorkspaceAlignment;
  return remainder == 0
             ? value
             : checked_add(value,
                           kWorkspaceAlignment - remainder,
                           "muDNN SDPA workspace alignment");
}

std::size_t tensor_bytes(const TensorDescriptor& tensor) {
  const std::size_t elements = tensor_io::storage_element_count(tensor);
  const std::size_t width = tensor_io::data_type_size(tensor.data_type);
  if (elements > std::numeric_limits<std::size_t>::max() / width) {
    throw std::overflow_error("muDNN SDPA tensor byte size overflows");
  }
  return elements * width;
}

bool is_row_major_contiguous(const TensorDescriptor& tensor) {
  std::int64_t running = 1;
  for (std::size_t remaining = tensor.dimensions.size(); remaining != 0;
       --remaining) {
    const std::size_t axis = remaining - 1;
    if (tensor.dimensions[axis] != 1 && tensor.strides[axis] != running) {
      return false;
    }
    if (tensor.dimensions[axis] >
        std::numeric_limits<std::int64_t>::max() / running) {
      throw std::overflow_error("muDNN SDPA dense extent overflows int64");
    }
    running *= tensor.dimensions[axis];
  }
  return true;
}

void validate_tensor(const TensorDescriptor& tensor,
                     std::string_view name,
                     flagdnnDataType_t expected_type) {
  if (tensor.uid <= 0 || tensor.data_type != expected_type ||
      tensor.dimensions.size() != 4 ||
      tensor.dimensions.size() != tensor.strides.size() ||
      !is_row_major_contiguous(tensor)) {
    throw std::invalid_argument(
        std::string(name) + " muDNN SDPA descriptor is invalid");
  }
  for (std::size_t axis = 0; axis < tensor.dimensions.size(); ++axis) {
    if (tensor.dimensions[axis] <= 0 || tensor.strides[axis] <= 0) {
      throw std::invalid_argument(
          std::string(name) + " muDNN SDPA extent is invalid");
    }
  }
  static_cast<void>(mudnn_data_type(tensor.data_type));
  static_cast<void>(tensor_io::element_count(tensor));
  static_cast<void>(tensor_io::storage_element_count(tensor));
}

struct AttentionGeometry {
  std::int64_t batch = 0;
  std::int64_t query_heads = 0;
  std::int64_t key_value_heads = 0;
  std::int64_t sequence_q = 0;
  std::int64_t sequence_kv = 0;
  std::int64_t head_dimension = 0;
  std::int64_t value_dimension = 0;
};

AttentionGeometry validate_qkv(const TensorDescriptor& q,
                               const TensorDescriptor& k,
                               const TensorDescriptor& v) {
  if (!is_io_type(q.data_type)) {
    throw std::invalid_argument(
        "muDNN FlashAttention supports FP16/BF16 reference tensors only");
  }
  validate_tensor(q, "Q", q.data_type);
  validate_tensor(k, "K", q.data_type);
  validate_tensor(v, "V", q.data_type);
  if (q.dimensions[0] != k.dimensions[0] ||
      q.dimensions[0] != v.dimensions[0] ||
      q.dimensions[3] != k.dimensions[3] ||
      k.dimensions[1] != v.dimensions[1] ||
      k.dimensions[2] != v.dimensions[2] ||
      q.dimensions[1] % k.dimensions[1] != 0) {
    throw std::invalid_argument("muDNN SDPA Q/K/V shapes are inconsistent");
  }
  if (q.dimensions[1] > std::numeric_limits<int>::max() ||
      v.dimensions[3] > std::numeric_limits<int>::max() /
                                q.dimensions[1]) {
    throw std::overflow_error("muDNN SDPA embed dimension overflows int");
  }
  return {q.dimensions[0],
          q.dimensions[1],
          k.dimensions[1],
          q.dimensions[2],
          k.dimensions[2],
          q.dimensions[3],
          v.dimensions[3]};
}

void validate_output(const TensorDescriptor& output,
                     const AttentionGeometry& geometry,
                     flagdnnDataType_t data_type,
                     std::string_view name) {
  validate_tensor(output, name, data_type);
  if (output.dimensions !=
      std::vector<std::int64_t>{geometry.batch,
                                geometry.query_heads,
                                geometry.sequence_q,
                                geometry.value_dimension}) {
    throw std::invalid_argument(
        std::string(name) + " muDNN SDPA shape is invalid");
  }
}

void validate_stats(const TensorDescriptor& stats,
                    const AttentionGeometry& geometry) {
  validate_tensor(stats, "stats", FLAGDNN_DATA_FLOAT32);
  if (stats.dimensions !=
      std::vector<std::int64_t>{geometry.batch,
                                geometry.query_heads,
                                geometry.sequence_q,
                                1}) {
    throw std::invalid_argument("muDNN SDPA stats shape is invalid");
  }
}

void validate_bias(const TensorDescriptor& bias,
                   const AttentionGeometry& geometry,
                   flagdnnDataType_t data_type,
                   std::string_view name) {
  validate_tensor(bias, name, data_type);
  if ((bias.dimensions[0] != 1 &&
       bias.dimensions[0] != geometry.batch) ||
      (bias.dimensions[1] != 1 &&
       bias.dimensions[1] != geometry.query_heads) ||
      bias.dimensions[2] != geometry.sequence_q ||
      bias.dimensions[3] != geometry.sequence_kv) {
    throw std::invalid_argument(
        std::string(name) + " muDNN SDPA shape is invalid");
  }
}

void require_unique_uids(
    std::span<const TensorDescriptor* const> tensors) {
  std::unordered_set<std::int64_t> uids;
  for (const TensorDescriptor* tensor : tensors) {
    if (tensor != nullptr && !uids.insert(tensor->uid).second) {
      throw std::invalid_argument("muDNN SDPA tensor UIDs must be distinct");
    }
  }
}

BindingMap parse_bindings(std::span<const flagdnnBinding_t> bindings,
                          std::size_t expected) {
  if (bindings.size() != expected) {
    throw std::invalid_argument("muDNN SDPA binding count is invalid");
  }
  BindingMap result;
  result.reserve(bindings.size());
  for (const flagdnnBinding_t& binding : bindings) {
    if (binding.uid <= 0 || binding.device_pointer == nullptr ||
        !result.emplace(binding.uid, binding.device_pointer).second) {
      throw std::invalid_argument("muDNN SDPA binding is invalid");
    }
  }
  return result;
}

void* binding_pointer(const TensorDescriptor& tensor,
                      const BindingMap& bindings) {
  const auto found = bindings.find(tensor.uid);
  if (found == bindings.end()) {
    throw std::invalid_argument("muDNN SDPA binding UID is missing");
  }
  return found->second;
}

void set_stream(musa::dnn::Handle& handle, flagdnnStream_t stream) {
  if (stream == nullptr) {
    throw std::invalid_argument("muDNN SDPA caller stream is null");
  }
  const musaStream_t musa_stream = reinterpret_cast<musaStream_t>(stream);
  check_mudnn(handle.SetStream(musa_stream), "muDNN Handle::SetStream(SDPA)");
  if (handle.GetStream() != musa_stream) {
    throw std::runtime_error("muDNN SDPA did not retain the caller stream");
  }
}

void configure_tensor(musa::dnn::Tensor& tensor,
                      const TensorDescriptor& descriptor,
                      void* pointer) {
  check_mudnn(tensor.SetAddr(pointer), "muDNN Tensor::SetAddr(SDPA)");
  check_mudnn(tensor.SetType(mudnn_data_type(descriptor.data_type)),
              "muDNN Tensor::SetType(SDPA)");
  check_mudnn(tensor.SetNdInfo(
                  static_cast<std::int64_t>(descriptor.dimensions.size()),
                  descriptor.dimensions.data(),
                  descriptor.strides.data()),
              "muDNN Tensor::SetNdInfo(SDPA)");
}

void configure_empty(musa::dnn::Tensor& tensor,
                     musa::dnn::Tensor::Type type) {
  const std::int64_t dimensions[] = {0};
  const std::int64_t strides[] = {1};
  check_mudnn(tensor.SetAddr(nullptr), "muDNN Tensor::SetAddr(empty SDPA)");
  check_mudnn(tensor.SetType(type), "muDNN Tensor::SetType(empty SDPA)");
  check_mudnn(tensor.SetNdInfo(1, dimensions, strides),
              "muDNN Tensor::SetNdInfo(empty SDPA)");
}

void configure_attention(musa::dnn::ScaledDotProductAttention& attention,
                         const AttentionGeometry& geometry,
                         double scale,
                         bool causal) {
  if (!std::isfinite(scale) || scale <= 0.0) {
    throw std::invalid_argument("muDNN SDPA scale must be positive and finite");
  }
  check_mudnn(attention.SetEmbedDim(static_cast<int>(
                  geometry.query_heads * geometry.value_dimension)),
              "muDNN SDPA::SetEmbedDim");
  check_mudnn(attention.SetHeadsNum(
                  static_cast<int>(geometry.query_heads)),
              "muDNN SDPA::SetHeadsNum");
  check_mudnn(attention.SetCausal(causal), "muDNN SDPA::SetCausal");
  check_mudnn(attention.SetMaskMode(false), "muDNN SDPA::SetMaskMode");
  check_mudnn(attention.SetKeyFormat(false), "muDNN SDPA::SetKeyFormat");
  check_mudnn(attention.SetScale(scale), "muDNN SDPA::SetScale");
}

class WorkspaceBump final {
 public:
  WorkspaceBump(void* workspace, std::size_t size)
      : pointer_(static_cast<std::byte*>(workspace)), size_(size) {
    if (size_ != 0 && pointer_ == nullptr) {
      throw std::invalid_argument("muDNN SDPA workspace is null");
    }
  }

  musa::dnn::MemoryHandler allocate(std::size_t requested) {
    if (requested == 0) {
      return musa::dnn::MemoryHandler(nullptr, [](void*) {});
    }
    const std::size_t start = align_up(offset_);
    if (start > size_ || requested > size_ - start) {
      throw std::runtime_error(
          "muDNN SDPA workspace request exceeds its contract: requested=" +
          std::to_string(requested) + ", available=" +
          std::to_string(start <= size_ ? size_ - start : 0));
    }
    offset_ = start + requested;
    return musa::dnn::MemoryHandler(pointer_ + start, [](void*) {});
  }

 private:
  std::byte* pointer_ = nullptr;
  std::size_t size_ = 0;
  std::size_t offset_ = 0;
};

std::vector<float> copy_logical_to_host(const TensorDescriptor& tensor,
                                        void* pointer,
                                        musaStream_t stream) {
  std::vector<std::uint8_t> encoded(tensor_bytes(tensor));
  check_musa(musaMemcpyAsync(encoded.data(),
                             pointer,
                             encoded.size(),
                             musaMemcpyDeviceToHost,
                             stream),
             "musaMemcpyAsync(SDPA device-to-host)");
  check_musa(musaStreamSynchronize(stream),
             "musaStreamSynchronize(SDPA host reference)");
  return tensor_io::gather(
      tensor_io::decode(encoded, tensor.data_type), tensor);
}

std::size_t logical_offset(const TensorDescriptor& tensor,
                           std::int64_t b,
                           std::int64_t h,
                           std::int64_t s,
                           std::int64_t d) {
  return static_cast<std::size_t>(
      (((b * tensor.dimensions[1] + h) * tensor.dimensions[2] + s) *
       tensor.dimensions[3]) + d);
}

void calculate_dbias(const MudnnSdpaBackwardDescriptor& descriptor,
                     const AttentionGeometry& geometry,
                     const BindingMap& bindings,
                     musaStream_t stream) {
  if (!descriptor.dbias.has_value()) {
    return;
  }
  const std::vector<float> q = copy_logical_to_host(
      descriptor.q, binding_pointer(descriptor.q, bindings), stream);
  const std::vector<float> k = copy_logical_to_host(
      descriptor.k, binding_pointer(descriptor.k, bindings), stream);
  const std::vector<float> v = copy_logical_to_host(
      descriptor.v, binding_pointer(descriptor.v, bindings), stream);
  const std::vector<float> output = copy_logical_to_host(
      descriptor.output, binding_pointer(descriptor.output, bindings), stream);
  const std::vector<float> doutput = copy_logical_to_host(
      descriptor.doutput,
      binding_pointer(descriptor.doutput, bindings),
      stream);
  const std::vector<float> stats = copy_logical_to_host(
      descriptor.stats, binding_pointer(descriptor.stats, bindings), stream);
  const std::vector<float> bias = copy_logical_to_host(
      *descriptor.bias, binding_pointer(*descriptor.bias, bindings), stream);

  const TensorDescriptor& dbias_descriptor = *descriptor.dbias;
  std::vector<float> dbias(tensor_io::element_count(dbias_descriptor), 0.0F);
  const std::int64_t head_group =
      geometry.query_heads / geometry.key_value_heads;
  for (std::int64_t b = 0; b < geometry.batch; ++b) {
    for (std::int64_t h = 0; h < geometry.query_heads; ++h) {
      const std::int64_t kv_head = h / head_group;
      for (std::int64_t m = 0; m < geometry.sequence_q; ++m) {
        double delta = 0.0;
        for (std::int64_t d = 0; d < geometry.value_dimension; ++d) {
          delta += static_cast<double>(output[logical_offset(
                       descriptor.output, b, h, m, d)]) *
                   static_cast<double>(doutput[logical_offset(
                       descriptor.doutput, b, h, m, d)]);
        }
        const double logsumexp = stats[static_cast<std::size_t>(
            (b * geometry.query_heads + h) * geometry.sequence_q + m)];
        for (std::int64_t n = 0; n < geometry.sequence_kv; ++n) {
          if (descriptor.causal && n > m) {
            continue;
          }
          double score = 0.0;
          for (std::int64_t d = 0; d < geometry.head_dimension; ++d) {
            score += static_cast<double>(q[logical_offset(
                         descriptor.q, b, h, m, d)]) *
                     static_cast<double>(k[logical_offset(
                         descriptor.k, b, kv_head, n, d)]);
          }
          score *= descriptor.attention_scale;
          const std::int64_t bias_batch =
              descriptor.bias->dimensions[0] == 1 ? 0 : b;
          const std::int64_t bias_head =
              descriptor.bias->dimensions[1] == 1 ? 0 : h;
          score += bias[logical_offset(
              *descriptor.bias, bias_batch, bias_head, m, n)];
          const double probability = std::exp(score - logsumexp);
          double dp = 0.0;
          for (std::int64_t d = 0; d < geometry.value_dimension; ++d) {
            dp += static_cast<double>(doutput[logical_offset(
                      descriptor.doutput, b, h, m, d)]) *
                  static_cast<double>(v[logical_offset(
                      descriptor.v, b, kv_head, n, d)]);
          }
          const std::int64_t dbias_batch =
              dbias_descriptor.dimensions[0] == 1 ? 0 : b;
          const std::int64_t dbias_head =
              dbias_descriptor.dimensions[1] == 1 ? 0 : h;
          dbias[logical_offset(
              dbias_descriptor, dbias_batch, dbias_head, m, n)] +=
              static_cast<float>(probability * (dp - delta));
        }
      }
    }
  }

  const std::vector<float> physical =
      tensor_io::scatter(dbias, dbias_descriptor);
  const std::vector<std::uint8_t> encoded =
      tensor_io::encode(physical, dbias_descriptor.data_type);
  check_musa(musaMemcpyAsync(binding_pointer(dbias_descriptor, bindings),
                             encoded.data(),
                             encoded.size(),
                             musaMemcpyHostToDevice,
                             stream),
             "musaMemcpyAsync(SDPA dBias host-to-device)");
  check_musa(musaStreamSynchronize(stream),
             "musaStreamSynchronize(SDPA dBias)");
}

}  // namespace

struct MudnnSdpaOperation::Impl {
  explicit Impl(MudnnSdpaDescriptor value)
      : descriptor(std::move(value)),
        geometry(validate_qkv(descriptor.q, descriptor.k, descriptor.v)),
        handle(0) {
    validate_output(descriptor.output,
                    geometry,
                    descriptor.q.data_type,
                    "output");
    if (descriptor.bias.has_value()) {
      validate_bias(*descriptor.bias,
                    geometry,
                    descriptor.q.data_type,
                    "bias");
    }
    if (descriptor.stats.has_value()) {
      validate_stats(*descriptor.stats, geometry);
    } else {
      temporary_stats = TensorDescriptor{
          1,
          FLAGDNN_DATA_FLOAT32,
          {geometry.batch, geometry.query_heads, geometry.sequence_q, 1},
          {geometry.query_heads * geometry.sequence_q,
           geometry.sequence_q,
           1,
           1},
          0};
    }
    const TensorDescriptor* bias =
        descriptor.bias.has_value() ? &*descriptor.bias : nullptr;
    const TensorDescriptor* stats =
        descriptor.stats.has_value() ? &*descriptor.stats : nullptr;
    const std::vector<const TensorDescriptor*> tensors{
        &descriptor.q,
        &descriptor.k,
        &descriptor.v,
        bias,
        &descriptor.output,
        stats};
    require_unique_uids(tensors);
    const std::size_t stats_bytes = descriptor.stats.has_value()
                                        ? 0
                                        : align_up(tensor_bytes(temporary_stats));
    workspace_bytes = checked_add(
        stats_bytes, kMudnnScratchBytes, "muDNN SDPA workspace size");
    configure_attention(attention,
                        geometry,
                        descriptor.attention_scale,
                        descriptor.causal);
  }

  MudnnSdpaDescriptor descriptor;
  AttentionGeometry geometry;
  TensorDescriptor temporary_stats;
  std::size_t workspace_bytes = 0;
  musa::dnn::Handle handle;
  musa::dnn::ScaledDotProductAttention attention;
};

MudnnSdpaOperation::MudnnSdpaOperation(MudnnSdpaDescriptor descriptor)
    : implementation_(std::make_unique<Impl>(std::move(descriptor))) {}

MudnnSdpaOperation::~MudnnSdpaOperation() = default;

std::size_t MudnnSdpaOperation::workspace_size() const noexcept {
  return implementation_->workspace_bytes;
}

void MudnnSdpaOperation::execute(
    std::span<const flagdnnBinding_t> raw_bindings,
    void* workspace,
    std::size_t workspace_size,
    flagdnnStream_t stream) {
  Impl& state = *implementation_;
  if (workspace_size < state.workspace_bytes || workspace == nullptr) {
    throw std::invalid_argument("muDNN SDPA workspace is too small");
  }
  const std::size_t expected =
      4 + static_cast<std::size_t>(state.descriptor.bias.has_value()) +
      static_cast<std::size_t>(state.descriptor.stats.has_value());
  const BindingMap bindings = parse_bindings(raw_bindings, expected);
  set_stream(state.handle, stream);

  musa::dnn::Tensor q;
  musa::dnn::Tensor k;
  musa::dnn::Tensor v;
  musa::dnn::Tensor bias;
  musa::dnn::Tensor output;
  musa::dnn::Tensor stats;
  musa::dnn::Tensor dropout;
  configure_tensor(q,
                   state.descriptor.q,
                   binding_pointer(state.descriptor.q, bindings));
  configure_tensor(k,
                   state.descriptor.k,
                   binding_pointer(state.descriptor.k, bindings));
  configure_tensor(v,
                   state.descriptor.v,
                   binding_pointer(state.descriptor.v, bindings));
  configure_tensor(output,
                   state.descriptor.output,
                   binding_pointer(state.descriptor.output, bindings));
  if (state.descriptor.bias.has_value()) {
    configure_tensor(bias,
                     *state.descriptor.bias,
                     binding_pointer(*state.descriptor.bias, bindings));
  } else {
    configure_empty(bias, mudnn_data_type(state.descriptor.q.data_type));
  }

  std::size_t scratch_offset = 0;
  if (state.descriptor.stats.has_value()) {
    configure_tensor(stats,
                     *state.descriptor.stats,
                     binding_pointer(*state.descriptor.stats, bindings));
  } else {
    configure_tensor(stats, state.temporary_stats, workspace);
    scratch_offset = align_up(tensor_bytes(state.temporary_stats));
  }
  configure_empty(dropout, musa::dnn::Tensor::Type::BOOL);
  auto* scratch = static_cast<std::byte*>(workspace) + scratch_offset;
  WorkspaceBump arena(scratch, workspace_size - scratch_offset);
  const musa::dnn::MemoryMaintainer maintainer =
      [&arena](std::size_t requested) { return arena.allocate(requested); };
  check_mudnn(state.attention.RunFlash(state.handle,
                                       output,
                                       stats,
                                       q,
                                       k,
                                       v,
                                       bias,
                                       dropout,
                                       maintainer),
              "muDNN SDPA::RunFlash");
}

struct MudnnSdpaBackwardOperation::Impl {
  explicit Impl(MudnnSdpaBackwardDescriptor value)
      : descriptor(std::move(value)),
        geometry(validate_qkv(descriptor.q, descriptor.k, descriptor.v)),
        handle(0) {
    validate_output(descriptor.output,
                    geometry,
                    descriptor.q.data_type,
                    "output");
    validate_output(descriptor.doutput,
                    geometry,
                    descriptor.q.data_type,
                    "doutput");
    validate_tensor(descriptor.dq, "dQ", descriptor.q.data_type);
    validate_tensor(descriptor.dk, "dK", descriptor.q.data_type);
    validate_tensor(descriptor.dv, "dV", descriptor.q.data_type);
    validate_stats(descriptor.stats, geometry);
    if (descriptor.dq.dimensions != descriptor.q.dimensions ||
        descriptor.dk.dimensions != descriptor.k.dimensions ||
        descriptor.dv.dimensions != descriptor.v.dimensions) {
      throw std::invalid_argument("muDNN SDPA gradient shapes are invalid");
    }
    if (descriptor.bias.has_value()) {
      validate_bias(*descriptor.bias,
                    geometry,
                    descriptor.q.data_type,
                    "bias");
    }
    if (descriptor.dbias.has_value()) {
      if (!descriptor.bias.has_value()) {
        throw std::invalid_argument("muDNN SDPA dBias requires bias");
      }
      validate_bias(*descriptor.dbias,
                    geometry,
                    descriptor.q.data_type,
                    "dBias");
      if (descriptor.dbias->dimensions != descriptor.bias->dimensions) {
        throw std::invalid_argument("muDNN SDPA bias/dBias shapes differ");
      }
    }
    const TensorDescriptor* bias =
        descriptor.bias.has_value() ? &*descriptor.bias : nullptr;
    const TensorDescriptor* dbias =
        descriptor.dbias.has_value() ? &*descriptor.dbias : nullptr;
    const std::vector<const TensorDescriptor*> tensors{
        &descriptor.q,
        &descriptor.k,
        &descriptor.v,
        bias,
        &descriptor.output,
        &descriptor.doutput,
        &descriptor.stats,
        &descriptor.dq,
        &descriptor.dk,
        &descriptor.dv,
        dbias};
    require_unique_uids(tensors);
    workspace_bytes = kMudnnScratchBytes;
    configure_attention(attention,
                        geometry,
                        descriptor.attention_scale,
                        descriptor.causal);
    check_mudnn(attention.SetTraining(true),
                "muDNN SDPA backward::SetTraining");
    check_mudnn(attention.SetIsDeterministic(descriptor.deterministic),
                "muDNN SDPA backward::SetIsDeterministic");
  }

  MudnnSdpaBackwardDescriptor descriptor;
  AttentionGeometry geometry;
  std::size_t workspace_bytes = 0;
  musa::dnn::Handle handle;
  musa::dnn::ScaledDotProductAttention attention;
};

MudnnSdpaBackwardOperation::MudnnSdpaBackwardOperation(
    MudnnSdpaBackwardDescriptor descriptor)
    : implementation_(std::make_unique<Impl>(std::move(descriptor))) {}

MudnnSdpaBackwardOperation::~MudnnSdpaBackwardOperation() = default;

std::size_t MudnnSdpaBackwardOperation::workspace_size() const noexcept {
  return implementation_->workspace_bytes;
}

void MudnnSdpaBackwardOperation::execute(
    std::span<const flagdnnBinding_t> raw_bindings,
    void* workspace,
    std::size_t workspace_size,
    flagdnnStream_t stream) {
  Impl& state = *implementation_;
  if (workspace_size < state.workspace_bytes || workspace == nullptr) {
    throw std::invalid_argument(
        "muDNN SDPA backward workspace is too small");
  }
  const std::size_t expected =
      9 + static_cast<std::size_t>(state.descriptor.bias.has_value()) +
      static_cast<std::size_t>(state.descriptor.dbias.has_value());
  const BindingMap bindings = parse_bindings(raw_bindings, expected);
  set_stream(state.handle, stream);

  musa::dnn::Tensor q;
  musa::dnn::Tensor k;
  musa::dnn::Tensor v;
  musa::dnn::Tensor bias;
  musa::dnn::Tensor output;
  musa::dnn::Tensor doutput;
  musa::dnn::Tensor stats;
  musa::dnn::Tensor dq;
  musa::dnn::Tensor dk;
  musa::dnn::Tensor dv;
  musa::dnn::Tensor dropout;
  configure_tensor(q,
                   state.descriptor.q,
                   binding_pointer(state.descriptor.q, bindings));
  configure_tensor(k,
                   state.descriptor.k,
                   binding_pointer(state.descriptor.k, bindings));
  configure_tensor(v,
                   state.descriptor.v,
                   binding_pointer(state.descriptor.v, bindings));
  configure_tensor(output,
                   state.descriptor.output,
                   binding_pointer(state.descriptor.output, bindings));
  configure_tensor(doutput,
                   state.descriptor.doutput,
                   binding_pointer(state.descriptor.doutput, bindings));
  configure_tensor(stats,
                   state.descriptor.stats,
                   binding_pointer(state.descriptor.stats, bindings));
  configure_tensor(dq,
                   state.descriptor.dq,
                   binding_pointer(state.descriptor.dq, bindings));
  configure_tensor(dk,
                   state.descriptor.dk,
                   binding_pointer(state.descriptor.dk, bindings));
  configure_tensor(dv,
                   state.descriptor.dv,
                   binding_pointer(state.descriptor.dv, bindings));
  if (state.descriptor.bias.has_value()) {
    configure_tensor(bias,
                     *state.descriptor.bias,
                     binding_pointer(*state.descriptor.bias, bindings));
  } else {
    configure_empty(bias, mudnn_data_type(state.descriptor.q.data_type));
  }
  configure_empty(dropout, musa::dnn::Tensor::Type::BOOL);

  WorkspaceBump arena(workspace, workspace_size);
  const musa::dnn::MemoryMaintainer maintainer =
      [&arena](std::size_t requested) { return arena.allocate(requested); };
  check_mudnn(state.attention.RunFlashBwd(state.handle,
                                          dq,
                                          dk,
                                          dv,
                                          doutput,
                                          q,
                                          k,
                                          v,
                                          bias,
                                          output,
                                          stats,
                                          dropout,
                                          maintainer),
              "muDNN SDPA::RunFlashBwd");
  calculate_dbias(state.descriptor,
                  state.geometry,
                  bindings,
                  reinterpret_cast<musaStream_t>(stream));
}

}  // namespace flagdnn::validation::mthreads
