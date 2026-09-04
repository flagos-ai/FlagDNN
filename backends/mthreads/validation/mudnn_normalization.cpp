/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "backends/mthreads/validation/mudnn_normalization.hpp"

#include "backends/mthreads/validation/musa_driver.hpp"
#include "backends/mthreads/validation/mudnn_workspace.hpp"
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
constexpr std::size_t kUnreportedWorkspaceReserve = 64 * 1024;

musa::dnn::Tensor::Type mudnn_data_type(flagdnnDataType_t data_type) {
  switch (data_type) {
    case FLAGDNN_DATA_FLOAT32:
      return musa::dnn::Tensor::Type::FLOAT;
    case FLAGDNN_DATA_FLOAT16:
      return musa::dnn::Tensor::Type::HALF;
    case FLAGDNN_DATA_BFLOAT16:
      return musa::dnn::Tensor::Type::BFLOAT16;
    case FLAGDNN_DATA_BOOLEAN:
    case FLAGDNN_DATA_FP8_E4M3:
    case FLAGDNN_DATA_FP8_E5M2:
      break;
  }
  throw std::invalid_argument(
      "muDNN normalization tensor type is unsupported");
}

bool is_floating(flagdnnDataType_t data_type) noexcept {
  return data_type == FLAGDNN_DATA_FLOAT32 ||
         data_type == FLAGDNN_DATA_FLOAT16 ||
         data_type == FLAGDNN_DATA_BFLOAT16;
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
      throw std::overflow_error(
          "muDNN normalization dense extent overflows int64");
    }
    running *= tensor.dimensions[axis];
  }
  return true;
}

void validate_tensor(const TensorDescriptor& tensor,
                     std::string_view name) {
  if (tensor.uid <= 0 || tensor.dimensions.empty() ||
      tensor.dimensions.size() > 8 ||
      tensor.dimensions.size() != tensor.strides.size() ||
      !is_floating(tensor.data_type)) {
    throw std::invalid_argument(
        std::string(name) + " muDNN normalization descriptor is invalid");
  }
  for (std::size_t axis = 0; axis < tensor.dimensions.size(); ++axis) {
    if (tensor.dimensions[axis] <= 0 || tensor.strides[axis] <= 0) {
      throw std::invalid_argument(
          std::string(name) +
          " muDNN normalization dimensions/strides must be positive");
    }
  }
  static_cast<void>(mudnn_data_type(tensor.data_type));
  static_cast<void>(tensor_io::element_count(tensor));
  static_cast<void>(tensor_io::storage_element_count(tensor));
}

template <typename... Tensors>
void require_distinct_uids(const Tensors&... tensors) {
  const std::unordered_set<std::int64_t> uids = {tensors.uid...};
  if (uids.size() != sizeof...(tensors)) {
    throw std::invalid_argument(
        "muDNN normalization tensor UIDs must be distinct");
  }
}

std::size_t tensor_bytes(const TensorDescriptor& tensor) {
  const std::size_t elements = tensor_io::storage_element_count(tensor);
  const std::size_t width = tensor_io::data_type_size(tensor.data_type);
  if (elements > std::numeric_limits<std::size_t>::max() / width) {
    throw std::overflow_error(
        "muDNN normalization tensor byte size overflows");
  }
  return elements * width;
}

std::size_t align_up(std::size_t value) {
  const std::size_t remainder = value % kWorkspaceAlignment;
  if (remainder == 0) {
    return value;
  }
  const std::size_t increment = kWorkspaceAlignment - remainder;
  if (value > std::numeric_limits<std::size_t>::max() - increment) {
    throw std::overflow_error(
        "muDNN normalization workspace alignment overflows");
  }
  return value + increment;
}

std::size_t checked_add(std::size_t left, std::size_t right) {
  if (left > std::numeric_limits<std::size_t>::max() - right) {
    throw std::overflow_error(
        "muDNN normalization workspace size overflows");
  }
  return left + right;
}

void configure_tensor_descriptor(musa::dnn::Tensor& tensor,
                                 const TensorDescriptor& descriptor) {
  check_mudnn(
      tensor.SetType(mudnn_data_type(descriptor.data_type)),
      "muDNN Tensor::SetType(normalization)");
  check_mudnn(
      tensor.SetNdInfo(
          static_cast<std::int64_t>(descriptor.dimensions.size()),
          descriptor.dimensions.data(),
          descriptor.strides.data()),
      "muDNN Tensor::SetNdInfo(normalization)");
}

void configure_tensor(musa::dnn::Tensor& tensor,
                      const TensorDescriptor& descriptor,
                      void* pointer) {
  check_mudnn(
      tensor.SetAddr(pointer), "muDNN Tensor::SetAddr(normalization)");
  configure_tensor_descriptor(tensor, descriptor);
}

using BindingMap = std::unordered_map<std::int64_t, void*>;

BindingMap parse_bindings(std::span<const flagdnnBinding_t> bindings,
                          std::size_t expected) {
  if (bindings.size() != expected) {
    throw std::invalid_argument(
        "muDNN normalization binding count is invalid");
  }
  BindingMap result;
  result.reserve(bindings.size());
  for (const flagdnnBinding_t& binding : bindings) {
    if (binding.uid <= 0 || binding.device_pointer == nullptr ||
        !result.emplace(binding.uid, binding.device_pointer).second) {
      throw std::invalid_argument(
          "muDNN normalization binding is invalid");
    }
  }
  return result;
}

void* binding_pointer(const TensorDescriptor& tensor,
                      const BindingMap& bindings) {
  const auto found = bindings.find(tensor.uid);
  if (found == bindings.end()) {
    throw std::invalid_argument(
        "muDNN normalization binding UID is missing");
  }
  return found->second;
}

void set_stream(musa::dnn::Handle& handle, flagdnnStream_t stream) {
  if (stream == nullptr) {
    throw std::invalid_argument(
        "muDNN normalization caller stream is null");
  }
  const musaStream_t musa_stream = reinterpret_cast<musaStream_t>(stream);
  check_mudnn(
      handle.SetStream(musa_stream),
      "muDNN Handle::SetStream(normalization)");
  if (handle.GetStream() != musa_stream) {
    throw std::runtime_error(
        "muDNN normalization did not retain the caller stream");
  }
}

struct NormalizationGeometry {
  std::vector<int> axes;
  std::vector<std::int64_t> statistics;
};

NormalizationGeometry validate_layer_geometry(
    const TensorDescriptor& x,
    const TensorDescriptor& scale,
    const TensorDescriptor& bias,
    const TensorDescriptor& y,
    const TensorDescriptor& inv_variance,
    const TensorDescriptor* mean,
    double epsilon,
    std::string_view operation) {
  validate_tensor(x, "X");
  validate_tensor(scale, "scale");
  validate_tensor(bias, "bias");
  validate_tensor(y, "Y");
  validate_tensor(inv_variance, "inverse variance");
  if (mean != nullptr) {
    validate_tensor(*mean, "mean");
  }
  if (x.data_type != y.data_type || x.data_type != scale.data_type ||
      x.data_type != bias.data_type || x.dimensions != y.dimensions ||
      x.strides != y.strides || scale.dimensions != bias.dimensions ||
      scale.dimensions.size() > x.dimensions.size() ||
      inv_variance.data_type != FLAGDNN_DATA_FLOAT32 ||
      (mean != nullptr && mean->data_type != FLAGDNN_DATA_FLOAT32) ||
      !is_row_major_contiguous(x) || !is_row_major_contiguous(y) ||
      !is_row_major_contiguous(scale) ||
      !is_row_major_contiguous(bias) ||
      !is_row_major_contiguous(inv_variance) ||
      (mean != nullptr && !is_row_major_contiguous(*mean)) ||
      !std::isfinite(epsilon) || epsilon <= 0.0) {
    throw std::invalid_argument(
        std::string("muDNN ") + std::string(operation) +
        " tensor metadata is invalid");
  }

  const std::size_t leading =
      x.dimensions.size() - scale.dimensions.size();
  std::size_t normalized_start = x.dimensions.size();
  for (std::size_t axis = 0; axis < x.dimensions.size(); ++axis) {
    const std::int64_t parameter_dimension =
        axis < leading ? 1 : scale.dimensions[axis - leading];
    if (parameter_dimension != 1) {
      if (parameter_dimension != x.dimensions[axis]) {
        throw std::invalid_argument(
            std::string("muDNN ") + std::string(operation) +
            " scale does not match the normalized suffix");
      }
      if (normalized_start == x.dimensions.size()) {
        normalized_start = axis;
      }
    } else if (normalized_start != x.dimensions.size() &&
               x.dimensions[axis] != 1) {
      throw std::invalid_argument(
          std::string("muDNN ") + std::string(operation) +
          " scale is not a suffix");
    }
  }
  if (normalized_start == x.dimensions.size()) {
    normalized_start = x.dimensions.size() - 1;
  }
  std::size_t normalized_elements = 1;
  for (std::size_t axis = normalized_start; axis < x.dimensions.size();
       ++axis) {
    normalized_elements *= static_cast<std::size_t>(x.dimensions[axis]);
  }
  std::vector<std::int64_t> statistics = x.dimensions;
  std::fill(statistics.begin() + static_cast<std::ptrdiff_t>(normalized_start),
            statistics.end(),
            1);
  if (tensor_io::element_count(scale) != normalized_elements ||
      inv_variance.dimensions != statistics ||
      (mean != nullptr && mean->dimensions != statistics)) {
    throw std::invalid_argument(
        std::string("muDNN ") + std::string(operation) +
        " row geometry is invalid");
  }
  NormalizationGeometry result;
  result.statistics = std::move(statistics);
  for (std::size_t axis = normalized_start; axis < x.dimensions.size();
       ++axis) {
    result.axes.push_back(static_cast<int>(axis));
  }
  return result;
}

void validate_batch_parameter(const TensorDescriptor& tensor,
                              std::int64_t channels,
                              flagdnnDataType_t data_type,
                              std::string_view name) {
  validate_tensor(tensor, name);
  if (tensor.data_type != data_type ||
      tensor_io::element_count(tensor) !=
          static_cast<std::size_t>(channels) ||
      !is_row_major_contiguous(tensor)) {
    throw std::invalid_argument(
        std::string("muDNN BatchNorm ") + std::string(name) +
        " metadata is invalid");
  }
}

void validate_batch_data(const TensorDescriptor& x,
                         const TensorDescriptor& y) {
  validate_tensor(x, "BatchNorm X");
  validate_tensor(y, "BatchNorm Y");
  if (x.dimensions.size() < 2 || x.data_type != y.data_type ||
      x.dimensions != y.dimensions) {
    throw std::invalid_argument(
        "muDNN BatchNorm X/Y metadata is invalid");
  }
}

bool requires_batchnorm_primitive_fallback(
    const TensorDescriptor& x) {
  std::size_t spatial = 1;
  for (std::size_t axis = 2; axis < x.dimensions.size(); ++axis) {
    const std::size_t dimension =
        static_cast<std::size_t>(x.dimensions[axis]);
    if (spatial > std::numeric_limits<std::size_t>::max() / dimension) {
      throw std::overflow_error(
          "muDNN BatchNorm spatial extent overflows");
    }
    spatial *= dimension;
  }
  // muDNN 3.1.5's fused BatchNorm path selects an invalid implementation
  // when every spatial dimension is one. A composition of its tensor
  // primitives has the same semantics and remains capturable.
  return spatial == 1;
}

struct BatchnormAffinePipeline {
  BatchnormAffinePipeline() {
    check_mudnn(cast_input.SetMode(musa::dnn::Unary::Mode::CAST),
                "muDNN Unary::SetMode(BatchNorm input cast)");
    check_mudnn(subtract_mean.SetMode(musa::dnn::Binary::Mode::SUB),
                "muDNN Binary::SetMode(BatchNorm subtract mean)");
    check_mudnn(multiply_inv_variance.SetMode(
                    musa::dnn::Binary::Mode::MUL),
                "muDNN Binary::SetMode(BatchNorm inverse variance)");
    check_mudnn(multiply_scale.SetMode(musa::dnn::Binary::Mode::MUL),
                "muDNN Binary::SetMode(BatchNorm scale)");
    check_mudnn(add_bias.SetMode(musa::dnn::Binary::Mode::ADD),
                "muDNN Binary::SetMode(BatchNorm bias)");
    check_mudnn(cast_output.SetMode(musa::dnn::Unary::Mode::CAST),
                "muDNN Unary::SetMode(BatchNorm output cast)");
  }

  void run(musa::dnn::Handle& handle,
           musa::dnn::Tensor& y,
           const musa::dnn::Tensor& x,
           const musa::dnn::Tensor& mean,
           const musa::dnn::Tensor& inv_variance,
           const musa::dnn::Tensor& scale,
           const musa::dnn::Tensor& bias,
           musa::dnn::Tensor& temporary0,
           musa::dnn::Tensor& temporary1) const {
    check_mudnn(cast_input.Run(handle, temporary0, x),
                "muDNN Unary::Run(BatchNorm input cast)");
    check_mudnn(subtract_mean.Run(handle, temporary1, temporary0, mean),
                "muDNN Binary::Run(BatchNorm subtract mean)");
    check_mudnn(multiply_inv_variance.Run(
                    handle, temporary0, temporary1, inv_variance),
                "muDNN Binary::Run(BatchNorm inverse variance)");
    check_mudnn(multiply_scale.Run(
                    handle, temporary1, temporary0, scale),
                "muDNN Binary::Run(BatchNorm scale)");
    check_mudnn(add_bias.Run(handle, temporary0, temporary1, bias),
                "muDNN Binary::Run(BatchNorm bias)");
    check_mudnn(cast_output.Run(handle, y, temporary0),
                "muDNN Unary::Run(BatchNorm output cast)");
  }

  musa::dnn::Unary cast_input;
  musa::dnn::Binary subtract_mean;
  musa::dnn::Binary multiply_inv_variance;
  musa::dnn::Binary multiply_scale;
  musa::dnn::Binary add_bias;
  musa::dnn::Unary cast_output;
};

}  // namespace

struct MudnnLayernormOperation::Impl {
  explicit Impl(MudnnLayernormDescriptor value)
      : descriptor(std::move(value)), handle(0) {
    require_distinct_uids(descriptor.x,
                          descriptor.scale,
                          descriptor.bias,
                          descriptor.y,
                          descriptor.mean,
                          descriptor.inv_variance);
    const NormalizationGeometry geometry = validate_layer_geometry(
        descriptor.x,
        descriptor.scale,
        descriptor.bias,
        descriptor.y,
        descriptor.inv_variance,
        &descriptor.mean,
        descriptor.epsilon,
        "LayerNorm");
    axes = geometry.axes;
    check_mudnn(layernorm.SetVarMode(musa::dnn::LayerNorm::VarMode::DIRECT),
                "muDNN LayerNorm::SetVarMode");
    check_mudnn(layernorm.SetEpsilon(descriptor.epsilon),
                "muDNN LayerNorm::SetEpsilon");
    check_mudnn(layernorm.SetAxis(axes.size(), axes.data()),
                "muDNN LayerNorm::SetAxis");
  }

  MudnnLayernormDescriptor descriptor;
  std::vector<int> axes;
  musa::dnn::Handle handle;
  musa::dnn::LayerNorm layernorm;
};

MudnnLayernormOperation::MudnnLayernormOperation(
    MudnnLayernormDescriptor descriptor)
    : implementation_(std::make_unique<Impl>(std::move(descriptor))) {}

MudnnLayernormOperation::~MudnnLayernormOperation() = default;

std::size_t MudnnLayernormOperation::workspace_size() const noexcept {
  return 0;
}

void MudnnLayernormOperation::execute(
    std::span<const flagdnnBinding_t> raw_bindings,
    void* workspace,
    std::size_t workspace_size,
    flagdnnStream_t stream) {
  if (workspace != nullptr || workspace_size != 0) {
    throw std::invalid_argument(
        "muDNN LayerNorm does not accept external workspace");
  }
  Impl& state = *implementation_;
  const BindingMap bindings = parse_bindings(raw_bindings, 6);
  set_stream(state.handle, stream);
  musa::dnn::Tensor x;
  musa::dnn::Tensor scale;
  musa::dnn::Tensor bias;
  musa::dnn::Tensor y;
  musa::dnn::Tensor mean;
  musa::dnn::Tensor inv_variance;
  configure_tensor(
      x, state.descriptor.x, binding_pointer(state.descriptor.x, bindings));
  configure_tensor(scale,
                   state.descriptor.scale,
                   binding_pointer(state.descriptor.scale, bindings));
  configure_tensor(bias,
                   state.descriptor.bias,
                   binding_pointer(state.descriptor.bias, bindings));
  configure_tensor(
      y, state.descriptor.y, binding_pointer(state.descriptor.y, bindings));
  configure_tensor(mean,
                   state.descriptor.mean,
                   binding_pointer(state.descriptor.mean, bindings));
  configure_tensor(
      inv_variance,
      state.descriptor.inv_variance,
      binding_pointer(state.descriptor.inv_variance, bindings));
  check_mudnn(
      state.layernorm.Run(
          state.handle, y, mean, inv_variance, x, scale, bias),
      "muDNN LayerNorm::Run");
}

struct MudnnRmsnormOperation::Impl {
  explicit Impl(MudnnRmsnormDescriptor value)
      : descriptor(std::move(value)), handle(0), temporary(descriptor.y) {
    require_distinct_uids(descriptor.x,
                          descriptor.scale,
                          descriptor.bias,
                          descriptor.y,
                          descriptor.inv_variance);
    const NormalizationGeometry geometry = validate_layer_geometry(
        descriptor.x,
        descriptor.scale,
        descriptor.bias,
        descriptor.y,
        descriptor.inv_variance,
        nullptr,
        descriptor.epsilon,
        "RMSNorm");
    axes = geometry.axes;
    temporary.uid = 0;
    temporary.binding_byte_offset = 0;
    workspace_bytes = tensor_bytes(temporary);
    check_mudnn(rmsnorm.SetVarMode(musa::dnn::RMSNorm::VarMode::DIRECT),
                "muDNN RMSNorm::SetVarMode");
    check_mudnn(rmsnorm.SetEpsilon(descriptor.epsilon),
                "muDNN RMSNorm::SetEpsilon");
    check_mudnn(rmsnorm.SetAxis(axes.size(), axes.data()),
                "muDNN RMSNorm::SetAxis");
    check_mudnn(add_bias.SetMode(musa::dnn::Binary::Mode::ADD),
                "muDNN Binary::SetMode(RMSNorm bias)");
  }

  MudnnRmsnormDescriptor descriptor;
  std::vector<int> axes;
  musa::dnn::Handle handle;
  musa::dnn::RMSNorm rmsnorm;
  musa::dnn::Binary add_bias;
  TensorDescriptor temporary;
  std::size_t workspace_bytes = 0;
};

MudnnRmsnormOperation::MudnnRmsnormOperation(
    MudnnRmsnormDescriptor descriptor)
    : implementation_(std::make_unique<Impl>(std::move(descriptor))) {}

MudnnRmsnormOperation::~MudnnRmsnormOperation() = default;

std::size_t MudnnRmsnormOperation::workspace_size() const noexcept {
  return implementation_->workspace_bytes;
}

void MudnnRmsnormOperation::execute(
    std::span<const flagdnnBinding_t> raw_bindings,
    void* workspace,
    std::size_t workspace_size,
    flagdnnStream_t stream) {
  Impl& state = *implementation_;
  if (workspace == nullptr || workspace_size < state.workspace_bytes) {
    throw std::invalid_argument(
        "muDNN RMSNorm workspace is too small");
  }
  const BindingMap bindings = parse_bindings(raw_bindings, 5);
  set_stream(state.handle, stream);
  musa::dnn::Tensor x;
  musa::dnn::Tensor scale;
  musa::dnn::Tensor bias;
  musa::dnn::Tensor y;
  musa::dnn::Tensor inv_variance;
  musa::dnn::Tensor temporary;
  configure_tensor(
      x, state.descriptor.x, binding_pointer(state.descriptor.x, bindings));
  configure_tensor(scale,
                   state.descriptor.scale,
                   binding_pointer(state.descriptor.scale, bindings));
  configure_tensor(bias,
                   state.descriptor.bias,
                   binding_pointer(state.descriptor.bias, bindings));
  configure_tensor(
      y, state.descriptor.y, binding_pointer(state.descriptor.y, bindings));
  configure_tensor(
      inv_variance,
      state.descriptor.inv_variance,
      binding_pointer(state.descriptor.inv_variance, bindings));
  configure_tensor(temporary, state.temporary, workspace);
  check_mudnn(
      state.rmsnorm.Run(state.handle, temporary, inv_variance, x, scale),
      "muDNN RMSNorm::Run");
  check_mudnn(
      state.add_bias.Run(state.handle, y, temporary, bias),
      "muDNN Binary::Run(RMSNorm bias)");
}

struct MudnnBatchnormOperation::Impl {
  explicit Impl(MudnnBatchnormDescriptor value)
      : descriptor(std::move(value)), handle(0) {
    require_distinct_uids(descriptor.x,
                          descriptor.scale,
                          descriptor.bias,
                          descriptor.previous_running_mean,
                          descriptor.previous_running_variance,
                          descriptor.y,
                          descriptor.mean,
                          descriptor.inv_variance,
                          descriptor.next_running_mean,
                          descriptor.next_running_variance);
    validate_batch_data(descriptor.x, descriptor.y);
    const std::int64_t channels = descriptor.x.dimensions[1];
    validate_batch_parameter(
        descriptor.scale, channels, FLAGDNN_DATA_FLOAT32, "scale");
    validate_batch_parameter(
        descriptor.bias, channels, FLAGDNN_DATA_FLOAT32, "bias");
    for (const auto& [tensor, name] :
         std::vector<std::pair<const TensorDescriptor*, std::string_view>>{
             {&descriptor.previous_running_mean, "previous running mean"},
             {&descriptor.previous_running_variance,
              "previous running variance"},
             {&descriptor.mean, "mean"},
             {&descriptor.inv_variance, "inverse variance"},
             {&descriptor.next_running_mean, "next running mean"},
             {&descriptor.next_running_variance,
              "next running variance"}}) {
      validate_batch_parameter(
          *tensor, channels, FLAGDNN_DATA_FLOAT32, name);
    }
    if (!std::isfinite(descriptor.epsilon) || descriptor.epsilon <= 0.0 ||
        !std::isfinite(descriptor.momentum) || descriptor.momentum < 0.0 ||
        descriptor.momentum > 1.0 ||
        descriptor.previous_running_mean.dimensions !=
            descriptor.next_running_mean.dimensions ||
        descriptor.previous_running_mean.strides !=
            descriptor.next_running_mean.strides ||
        descriptor.previous_running_variance.dimensions !=
            descriptor.next_running_variance.dimensions ||
        descriptor.previous_running_variance.strides !=
            descriptor.next_running_variance.strides) {
      throw std::invalid_argument(
          "muDNN BatchNorm scalar/running-stat metadata is invalid");
    }
    primitive_fallback =
        requires_batchnorm_primitive_fallback(descriptor.x);
    if (primitive_fallback) {
      raw_variance = descriptor.inv_variance;
      variance_plus_epsilon = raw_variance;
      scaled_previous = raw_variance;
      scaled_fresh = raw_variance;
      temporary0 = descriptor.x;
      temporary0.data_type = FLAGDNN_DATA_FLOAT32;
      temporary1 = temporary0;
      for (TensorDescriptor* tensor :
           {&raw_variance,
            &variance_plus_epsilon,
            &scaled_previous,
            &scaled_fresh,
            &temporary0,
            &temporary1}) {
        tensor->uid = 0;
        tensor->binding_byte_offset = 0;
      }

      check_mudnn(statistics.SetMode(musa::dnn::Reduce::Mode::VARIANCE),
                  "muDNN Reduce::SetMode(BatchNorm statistics)");
      const int batch_axis = 0;
      check_mudnn(statistics.SetDim(1, &batch_axis),
                  "muDNN Reduce::SetDim(BatchNorm statistics)");
      check_mudnn(statistics.SetCorrection(0),
                  "muDNN Reduce::SetCorrection(BatchNorm statistics)");
      musa::dnn::Tensor reduction_input;
      musa::dnn::Tensor reduction_output;
      configure_tensor_descriptor(reduction_input, descriptor.x);
      configure_tensor_descriptor(reduction_output, raw_variance);
      check_mudnn(
          statistics.GetWorkspaceSize(
              handle, scratch_bytes, reduction_output, reduction_input),
          "muDNN Reduce::GetWorkspaceSize(BatchNorm statistics)");
      std::size_t scratch_reserve = checked_add(
          tensor_bytes(descriptor.x), tensor_bytes(raw_variance));
      scratch_reserve = checked_add(
          scratch_reserve, tensor_bytes(descriptor.mean));
      scratch_reserve =
          std::max(scratch_reserve, kUnreportedWorkspaceReserve);
      scratch_bytes = checked_add(
          align_up(scratch_bytes), align_up(scratch_reserve));

      std::size_t next = 0;
      const auto append_workspace =
          [&](const TensorDescriptor& tensor, std::size_t& offset) {
        offset = next;
        next = align_up(checked_add(next, tensor_bytes(tensor)));
      };
      append_workspace(raw_variance, raw_variance_offset);
      append_workspace(variance_plus_epsilon,
                       variance_plus_epsilon_offset);
      append_workspace(scaled_previous, scaled_previous_offset);
      append_workspace(scaled_fresh, scaled_fresh_offset);
      append_workspace(temporary0, temporary0_offset);
      append_workspace(temporary1, temporary1_offset);
      scratch_offset = next;
      workspace_bytes = checked_add(scratch_offset, scratch_bytes);

      check_mudnn(add_epsilon.SetMode(musa::dnn::Unary::Mode::ADD),
                  "muDNN Unary::SetMode(BatchNorm epsilon)");
      check_mudnn(add_epsilon.SetAlpha(descriptor.epsilon),
                  "muDNN Unary::SetAlpha(BatchNorm epsilon)");
      check_mudnn(inv_sqrt.SetMode(musa::dnn::Unary::Mode::RSQRT),
                  "muDNN Unary::SetMode(BatchNorm inverse square root)");
      check_mudnn(scale_previous.SetMode(musa::dnn::Unary::Mode::MUL),
                  "muDNN Unary::SetMode(BatchNorm previous statistic)");
      check_mudnn(scale_previous.SetAlpha(1.0 - descriptor.momentum),
                  "muDNN Unary::SetAlpha(BatchNorm previous statistic)");
      check_mudnn(scale_fresh_mean.SetMode(musa::dnn::Unary::Mode::MUL),
                  "muDNN Unary::SetMode(BatchNorm fresh mean)");
      check_mudnn(scale_fresh_mean.SetAlpha(descriptor.momentum),
                  "muDNN Unary::SetAlpha(BatchNorm fresh mean)");
      const double count =
          static_cast<double>(descriptor.x.dimensions.front());
      const double correction = count > 1.0 ? count / (count - 1.0) : 1.0;
      check_mudnn(
          scale_fresh_variance.SetMode(musa::dnn::Unary::Mode::MUL),
          "muDNN Unary::SetMode(BatchNorm fresh variance)");
      check_mudnn(
          scale_fresh_variance.SetAlpha(
              descriptor.momentum * correction),
          "muDNN Unary::SetAlpha(BatchNorm fresh variance)");
      check_mudnn(add_running.SetMode(musa::dnn::Binary::Mode::ADD),
                  "muDNN Binary::SetMode(BatchNorm running statistic)");
      primitive_pipeline =
          std::make_unique<BatchnormAffinePipeline>();
      return;
    }
    check_mudnn(batchnorm.SetMode(musa::dnn::BatchNorm::Mode::PER_CHANNEL),
                "muDNN BatchNorm::SetMode");
    check_mudnn(batchnorm.SetEpsilon(descriptor.epsilon),
                "muDNN BatchNorm::SetEpsilon");
    check_mudnn(batchnorm.SetTraining(true),
                "muDNN BatchNorm::SetTraining");
    musa::dnn::Tensor x;
    musa::dnn::Tensor scale;
    musa::dnn::Tensor bias;
    musa::dnn::Tensor y;
    musa::dnn::Tensor mean;
    musa::dnn::Tensor inv_variance;
    musa::dnn::Tensor next_mean;
    musa::dnn::Tensor next_variance;
    configure_tensor_descriptor(x, descriptor.x);
    configure_tensor_descriptor(scale, descriptor.scale);
    configure_tensor_descriptor(bias, descriptor.bias);
    configure_tensor_descriptor(y, descriptor.y);
    configure_tensor_descriptor(mean, descriptor.mean);
    configure_tensor_descriptor(inv_variance, descriptor.inv_variance);
    configure_tensor_descriptor(next_mean, descriptor.next_running_mean);
    configure_tensor_descriptor(
        next_variance, descriptor.next_running_variance);
    check_mudnn(
        batchnorm.GetWorkspaceSizeComposite(handle,
                                            workspace_bytes,
                                            y,
                                            x,
                                            next_mean,
                                            next_variance,
                                            mean,
                                            inv_variance,
                                            scale,
                                            bias),
        "muDNN BatchNorm::GetWorkspaceSizeComposite");
    std::size_t geometry_reserve = checked_add(
        tensor_bytes(descriptor.x), tensor_bytes(descriptor.y));
    geometry_reserve = checked_add(
        geometry_reserve, tensor_bytes(descriptor.scale));
    geometry_reserve = checked_add(
        geometry_reserve, tensor_bytes(descriptor.bias));
    geometry_reserve = checked_add(
        geometry_reserve, tensor_bytes(descriptor.mean));
    geometry_reserve = checked_add(
        geometry_reserve, tensor_bytes(descriptor.inv_variance));
    geometry_reserve =
        std::max(geometry_reserve, kUnreportedWorkspaceReserve);
    workspace_bytes = checked_add(
        align_up(workspace_bytes), align_up(geometry_reserve));
  }

  MudnnBatchnormDescriptor descriptor;
  musa::dnn::Handle handle;
  musa::dnn::BatchNorm batchnorm;
  musa::dnn::Reduce statistics;
  musa::dnn::Unary add_epsilon;
  musa::dnn::Unary inv_sqrt;
  musa::dnn::Unary scale_previous;
  musa::dnn::Unary scale_fresh_mean;
  musa::dnn::Unary scale_fresh_variance;
  musa::dnn::Binary add_running;
  std::unique_ptr<BatchnormAffinePipeline> primitive_pipeline;
  TensorDescriptor raw_variance;
  TensorDescriptor variance_plus_epsilon;
  TensorDescriptor scaled_previous;
  TensorDescriptor scaled_fresh;
  TensorDescriptor temporary0;
  TensorDescriptor temporary1;
  bool primitive_fallback = false;
  std::size_t raw_variance_offset = 0;
  std::size_t variance_plus_epsilon_offset = 0;
  std::size_t scaled_previous_offset = 0;
  std::size_t scaled_fresh_offset = 0;
  std::size_t temporary0_offset = 0;
  std::size_t temporary1_offset = 0;
  std::size_t scratch_offset = 0;
  std::size_t scratch_bytes = 0;
  std::size_t workspace_bytes = 0;
};

MudnnBatchnormOperation::MudnnBatchnormOperation(
    MudnnBatchnormDescriptor descriptor)
    : implementation_(std::make_unique<Impl>(std::move(descriptor))) {}

MudnnBatchnormOperation::~MudnnBatchnormOperation() = default;

std::size_t MudnnBatchnormOperation::workspace_size() const noexcept {
  return implementation_->workspace_bytes;
}

void MudnnBatchnormOperation::execute(
    std::span<const flagdnnBinding_t> raw_bindings,
    void* workspace,
    std::size_t workspace_size,
    flagdnnStream_t stream) {
  Impl& state = *implementation_;
  if (workspace_size < state.workspace_bytes ||
      (state.workspace_bytes != 0 && workspace == nullptr)) {
    throw std::invalid_argument(
        "muDNN BatchNorm workspace is too small");
  }
  const BindingMap bindings = parse_bindings(raw_bindings, 10);
  set_stream(state.handle, stream);
  const musaStream_t musa_stream = reinterpret_cast<musaStream_t>(stream);
  const auto copy_running_stat = [&](const TensorDescriptor& source,
                                     const TensorDescriptor& destination) {
    check_musa(
        musaMemcpyAsync(binding_pointer(destination, bindings),
                        binding_pointer(source, bindings),
                        tensor_bytes(source),
                        musaMemcpyDeviceToDevice,
                        musa_stream),
        "musaMemcpyAsync(BatchNorm running statistic)");
  };
  if (!state.primitive_fallback) {
    copy_running_stat(state.descriptor.previous_running_mean,
                      state.descriptor.next_running_mean);
    copy_running_stat(state.descriptor.previous_running_variance,
                      state.descriptor.next_running_variance);
  }

  musa::dnn::Tensor x;
  musa::dnn::Tensor scale;
  musa::dnn::Tensor bias;
  musa::dnn::Tensor y;
  musa::dnn::Tensor mean;
  musa::dnn::Tensor inv_variance;
  musa::dnn::Tensor previous_mean;
  musa::dnn::Tensor previous_variance;
  musa::dnn::Tensor next_mean;
  musa::dnn::Tensor next_variance;
  configure_tensor(
      x, state.descriptor.x, binding_pointer(state.descriptor.x, bindings));
  configure_tensor(scale,
                   state.descriptor.scale,
                   binding_pointer(state.descriptor.scale, bindings));
  configure_tensor(bias,
                   state.descriptor.bias,
                   binding_pointer(state.descriptor.bias, bindings));
  configure_tensor(
      y, state.descriptor.y, binding_pointer(state.descriptor.y, bindings));
  configure_tensor(mean,
                   state.descriptor.mean,
                   binding_pointer(state.descriptor.mean, bindings));
  configure_tensor(
      inv_variance,
      state.descriptor.inv_variance,
      binding_pointer(state.descriptor.inv_variance, bindings));
  configure_tensor(
      previous_mean,
      state.descriptor.previous_running_mean,
      binding_pointer(state.descriptor.previous_running_mean, bindings));
  configure_tensor(
      previous_variance,
      state.descriptor.previous_running_variance,
      binding_pointer(state.descriptor.previous_running_variance, bindings));
  configure_tensor(
      next_mean,
      state.descriptor.next_running_mean,
      binding_pointer(state.descriptor.next_running_mean, bindings));
  configure_tensor(
      next_variance,
      state.descriptor.next_running_variance,
      binding_pointer(state.descriptor.next_running_variance, bindings));
  if (state.primitive_fallback) {
    if (state.primitive_pipeline == nullptr) {
      throw std::logic_error(
          "muDNN BatchNorm primitive pipeline is unavailable");
    }
    auto* bytes = static_cast<std::byte*>(workspace);
    musa::dnn::Tensor raw_variance;
    musa::dnn::Tensor variance_plus_epsilon;
    musa::dnn::Tensor scaled_previous;
    musa::dnn::Tensor scaled_fresh;
    musa::dnn::Tensor temporary0;
    musa::dnn::Tensor temporary1;
    configure_tensor(raw_variance,
                     state.raw_variance,
                     bytes + state.raw_variance_offset);
    configure_tensor(variance_plus_epsilon,
                     state.variance_plus_epsilon,
                     bytes + state.variance_plus_epsilon_offset);
    configure_tensor(scaled_previous,
                     state.scaled_previous,
                     bytes + state.scaled_previous_offset);
    configure_tensor(scaled_fresh,
                     state.scaled_fresh,
                     bytes + state.scaled_fresh_offset);
    configure_tensor(temporary0,
                     state.temporary0,
                     bytes + state.temporary0_offset);
    configure_tensor(temporary1,
                     state.temporary1,
                     bytes + state.temporary1_offset);
    void* const scratch =
        state.scratch_bytes == 0
            ? nullptr
            : bytes + state.scratch_offset;
    const musa::dnn::MemoryMaintainer statistics_maintainer =
        make_mudnn_workspace_maintainer(
            scratch,
            state.scratch_bytes,
            "muDNN BatchNorm statistics");
    check_mudnn(
        state.statistics.RunMeanAndVar(state.handle,
                                       raw_variance,
                                       mean,
                                       x,
                                       statistics_maintainer),
        "muDNN Reduce::RunMeanAndVar(BatchNorm)");
    check_mudnn(
        state.add_epsilon.Run(
            state.handle, variance_plus_epsilon, raw_variance),
        "muDNN Unary::Run(BatchNorm epsilon)");
    check_mudnn(
        state.inv_sqrt.Run(
            state.handle, inv_variance, variance_plus_epsilon),
        "muDNN Unary::Run(BatchNorm inverse square root)");
    check_mudnn(
        state.scale_previous.Run(
            state.handle, scaled_previous, previous_mean),
        "muDNN Unary::Run(BatchNorm previous mean)");
    check_mudnn(
        state.scale_fresh_mean.Run(state.handle, scaled_fresh, mean),
        "muDNN Unary::Run(BatchNorm fresh mean)");
    check_mudnn(
        state.add_running.Run(
            state.handle, next_mean, scaled_previous, scaled_fresh),
        "muDNN Binary::Run(BatchNorm next running mean)");
    check_mudnn(
        state.scale_previous.Run(
            state.handle, scaled_previous, previous_variance),
        "muDNN Unary::Run(BatchNorm previous variance)");
    check_mudnn(
        state.scale_fresh_variance.Run(
            state.handle, scaled_fresh, raw_variance),
        "muDNN Unary::Run(BatchNorm fresh variance)");
    check_mudnn(
        state.add_running.Run(
            state.handle, next_variance, scaled_previous, scaled_fresh),
        "muDNN Binary::Run(BatchNorm next running variance)");
    state.primitive_pipeline->run(state.handle,
                                  y,
                                  x,
                                  mean,
                                  inv_variance,
                                  scale,
                                  bias,
                                  temporary0,
                                  temporary1);
    return;
  }
  const musa::dnn::MemoryMaintainer maintainer =
      make_mudnn_workspace_maintainer(
          workspace, state.workspace_bytes, "muDNN BatchNorm");
  check_mudnn(
      state.batchnorm.RunComposite(state.handle,
                                   y,
                                   x,
                                   next_mean,
                                   next_variance,
                                   mean,
                                   inv_variance,
                                   scale,
                                   bias,
                                   state.descriptor.momentum,
                                   maintainer),
      "muDNN BatchNorm::RunComposite");
}

struct MudnnBatchnormInferenceOperation::Impl {
  explicit Impl(MudnnBatchnormInferenceDescriptor value)
      : descriptor(std::move(value)),
        handle(0),
        squared(descriptor.inv_variance),
        variance(descriptor.inv_variance) {
    require_distinct_uids(descriptor.x,
                          descriptor.mean,
                          descriptor.inv_variance,
                          descriptor.scale,
                          descriptor.bias,
                          descriptor.y);
    validate_batch_data(descriptor.x, descriptor.y);
    const std::int64_t channels = descriptor.x.dimensions[1];
    validate_batch_parameter(
        descriptor.mean, channels, FLAGDNN_DATA_FLOAT32, "mean");
    validate_batch_parameter(descriptor.inv_variance,
                             channels,
                             FLAGDNN_DATA_FLOAT32,
                             "inverse variance");
    validate_batch_parameter(
        descriptor.scale, channels, FLAGDNN_DATA_FLOAT32, "scale");
    validate_batch_parameter(
        descriptor.bias, channels, FLAGDNN_DATA_FLOAT32, "bias");
    primitive_fallback =
        requires_batchnorm_primitive_fallback(descriptor.x);
    squared.uid = 0;
    squared.binding_byte_offset = 0;
    variance.uid = 0;
    variance.binding_byte_offset = 0;
    if (primitive_fallback) {
      temporary0 = descriptor.x;
      temporary0.uid = 0;
      temporary0.binding_byte_offset = 0;
      temporary0.data_type = FLAGDNN_DATA_FLOAT32;
      temporary1 = temporary0;
      temporary0_offset = 0;
      temporary1_offset = align_up(tensor_bytes(temporary0));
      workspace_bytes = checked_add(
          temporary1_offset, tensor_bytes(temporary1));
      primitive_pipeline =
          std::make_unique<BatchnormAffinePipeline>();
    } else {
      squared_offset = 0;
      variance_offset = align_up(tensor_bytes(squared));
      workspace_bytes =
          checked_add(variance_offset, tensor_bytes(variance));
    }
    check_mudnn(square.SetMode(musa::dnn::Unary::Mode::SQUARE),
                "muDNN Unary::SetMode(BatchNorm inverse square)");
    check_mudnn(reciprocal.SetMode(musa::dnn::Unary::Mode::RECIPROCAL),
                "muDNN Unary::SetMode(BatchNorm reciprocal)");
    check_mudnn(batchnorm.SetMode(musa::dnn::BatchNorm::Mode::PER_CHANNEL),
                "muDNN BatchNorm::SetMode(inference)");
    // FlagDNN inference receives inverse variance directly. RunPure receives
    // variance and applies epsilon, so its epsilon must be zero after the
    // device-side inverse-variance-to-variance conversion below.
    check_mudnn(batchnorm.SetEpsilon(0.0),
                "muDNN BatchNorm::SetEpsilon(inference)");
    check_mudnn(batchnorm.SetTraining(false),
                "muDNN BatchNorm::SetTraining(inference)");
  }

  MudnnBatchnormInferenceDescriptor descriptor;
  musa::dnn::Handle handle;
  musa::dnn::Unary square;
  musa::dnn::Unary reciprocal;
  musa::dnn::BatchNorm batchnorm;
  TensorDescriptor squared;
  TensorDescriptor variance;
  TensorDescriptor temporary0;
  TensorDescriptor temporary1;
  std::unique_ptr<BatchnormAffinePipeline> primitive_pipeline;
  bool primitive_fallback = false;
  std::size_t squared_offset = 0;
  std::size_t variance_offset = 0;
  std::size_t temporary0_offset = 0;
  std::size_t temporary1_offset = 0;
  std::size_t workspace_bytes = 0;
};

MudnnBatchnormInferenceOperation::MudnnBatchnormInferenceOperation(
    MudnnBatchnormInferenceDescriptor descriptor)
    : implementation_(std::make_unique<Impl>(std::move(descriptor))) {}

MudnnBatchnormInferenceOperation::~MudnnBatchnormInferenceOperation() =
    default;

std::size_t MudnnBatchnormInferenceOperation::workspace_size() const noexcept {
  return implementation_->workspace_bytes;
}

void MudnnBatchnormInferenceOperation::execute(
    std::span<const flagdnnBinding_t> raw_bindings,
    void* workspace,
    std::size_t workspace_size,
    flagdnnStream_t stream) {
  Impl& state = *implementation_;
  if (workspace == nullptr || workspace_size < state.workspace_bytes) {
    throw std::invalid_argument(
        "muDNN BatchNorm inference workspace is too small");
  }
  const BindingMap bindings = parse_bindings(raw_bindings, 6);
  set_stream(state.handle, stream);
  auto* bytes = static_cast<std::byte*>(workspace);
  musa::dnn::Tensor x;
  musa::dnn::Tensor mean;
  musa::dnn::Tensor inv_variance;
  musa::dnn::Tensor scale;
  musa::dnn::Tensor bias;
  musa::dnn::Tensor y;
  musa::dnn::Tensor squared;
  musa::dnn::Tensor variance;
  configure_tensor(
      x, state.descriptor.x, binding_pointer(state.descriptor.x, bindings));
  configure_tensor(mean,
                   state.descriptor.mean,
                   binding_pointer(state.descriptor.mean, bindings));
  configure_tensor(
      inv_variance,
      state.descriptor.inv_variance,
      binding_pointer(state.descriptor.inv_variance, bindings));
  configure_tensor(scale,
                   state.descriptor.scale,
                   binding_pointer(state.descriptor.scale, bindings));
  configure_tensor(bias,
                   state.descriptor.bias,
                   binding_pointer(state.descriptor.bias, bindings));
  configure_tensor(
      y, state.descriptor.y, binding_pointer(state.descriptor.y, bindings));
  if (state.primitive_fallback) {
    if (state.primitive_pipeline == nullptr) {
      throw std::logic_error(
          "muDNN BatchNorm inference primitive pipeline is unavailable");
    }
    musa::dnn::Tensor temporary0;
    musa::dnn::Tensor temporary1;
    configure_tensor(temporary0,
                     state.temporary0,
                     bytes + state.temporary0_offset);
    configure_tensor(temporary1,
                     state.temporary1,
                     bytes + state.temporary1_offset);
    state.primitive_pipeline->run(state.handle,
                                  y,
                                  x,
                                  mean,
                                  inv_variance,
                                  scale,
                                  bias,
                                  temporary0,
                                  temporary1);
    return;
  }
  configure_tensor(
      squared, state.squared, bytes + state.squared_offset);
  configure_tensor(
      variance, state.variance, bytes + state.variance_offset);
  check_mudnn(
      state.square.Run(state.handle, squared, inv_variance),
      "muDNN Unary::Run(BatchNorm inverse square)");
  check_mudnn(
      state.reciprocal.Run(state.handle, variance, squared),
      "muDNN Unary::Run(BatchNorm reciprocal)");
  check_mudnn(
      state.batchnorm.RunPure(
          state.handle, y, x, mean, variance, scale, bias),
      "muDNN BatchNorm::RunPure");
}

}  // namespace flagdnn::validation::mthreads
