/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
// CPU-only: no HIP handle, stream, device allocation or executable plan.
#include "common/reduction.hpp"
#include "reference/cpu/matrix.hpp"
#include "tensor_io.hpp"
#include "tensor_reference.hpp"

#include <array>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
namespace hv = flagdnn::validation::hygon;
namespace io = hv::tensor_io;
namespace cpu = flagdnn::reference::cpu;
using Tensors = std::array<hv::ReferenceTensor, 2>;

void require(bool condition, const std::string &message) {
  if (!condition) throw std::runtime_error(message);
}

void require_supported(const hv::HipdnnTensorOperation &operation,
                       const Tensors &tensors, const std::string &name) {
  const auto result = hv::hipdnn_tensor_capability(operation, tensors);
  require(result.supported &&
              result.classification == hv::HipdnnCapabilityClass::kSupported &&
              result.reason.empty(),
          name + " unexpectedly rejected: " + result.reason);
}

void require_invalid(const hv::HipdnnTensorOperation &operation,
                     const Tensors &tensors, const std::string &name) {
  const auto result = hv::hipdnn_tensor_capability(operation, tensors);
  require(!result.supported &&
              result.classification ==
                  hv::HipdnnCapabilityClass::kInvalidAdapterContract &&
              !result.reason.empty(),
          name + " must reject invalid metadata rather than skip");
}

void require_vendor(const hv::HipdnnTensorOperation &operation,
                    const Tensors &tensors, const std::string &reason) {
  const auto result = hv::hipdnn_tensor_capability(operation, tensors);
  require(!result.supported &&
              result.classification ==
                  hv::HipdnnCapabilityClass::kVendorUnsupported &&
              result.reason.find(reason) != std::string::npos,
          "expected explicit limitation: " + reason + "; actual=" + result.reason);
}

void contract() {
  int fp32 = 0, fp16 = 0, bf16 = 0, int32 = 0, middle = 0, views = 0;
  for (const auto &value : flagdnn::testing::make_reduction_cases()) {
    if (value.name.find("_to_fp32_") == std::string::npos ||
        value.output.data_type != FLAGDNN_DATA_FLOAT32)
      continue;
    const auto type = value.input.data_type;
    // Cover the full original fp32_output benchmark catalog, including
    // integer inputs whose legal conversion requires the CPU/equivalent path.
    if (type == FLAGDNN_DATA_BFLOAT16 || type == FLAGDNN_DATA_INT32) {
      if ((value.input.dimensions.size() != 2 &&
           value.axis != static_cast<std::int32_t>(
                             value.input.dimensions.size() - 1)) ||
          value.mode != FLAGDNN_REDUCTION_ADD)
        continue;
      if (type == FLAGDNN_DATA_BFLOAT16) ++bf16;
      else ++int32;
    } else if (type != FLAGDNN_DATA_FLOAT16 && type != FLAGDNN_DATA_FLOAT32)
      continue;
    const auto operation = hv::make_hipdnn_reduction_operation(
        value.mode, value.axis, value.keep_dimensions);
    const Tensors tensors = {hv::as_reference_tensor(value.input),
                             hv::as_reference_tensor(value.output)};
    require(tensors[0].binding_byte_offset == 16 &&
                tensors[1].binding_byte_offset == 32,
            "shared mixed-output catalog offsets changed");
    auto wrong_axis = operation;
    wrong_axis.reduction_axis =
        static_cast<std::int32_t>(value.input.dimensions.size());
    require_invalid(wrong_axis, tensors, value.name + " axis");
    Tensors changed = tensors;
    changed[0].binding_byte_offset = 17;
    require_invalid(operation, changed, value.name + " input alignment");
    changed = tensors;
    changed[1].binding_byte_offset = 34;
    require_invalid(operation, changed, value.name + " FP32 output alignment");
    changed = tensors;
    changed[1].dimensions.front() += 1;
    require_invalid(operation, changed, value.name + " output shape");
    changed = tensors;
    changed[0].uid = changed[1].uid;
    require_invalid(operation, changed, value.name + " duplicate UID");
    changed = tensors;
    changed[0].data_type = FLAGDNN_DATA_FLOAT32;
    changed[1].data_type = FLAGDNN_DATA_FLOAT16;
    require_invalid(operation, changed, value.name + " narrowing dtype pair");
    if (type == FLAGDNN_DATA_BFLOAT16 || type == FLAGDNN_DATA_INT32) {
      require_vendor(operation, tensors,
                     type == FLAGDNN_DATA_BFLOAT16
                         ? "no validated executable primitive"
                         : "reduction is validated only for FP32/FP16");
      bool rejected = false;
      try { (void)hv::hipdnn_reduction_descriptors(operation, tensors); }
      catch (const std::invalid_argument &) { rejected = true; }
      require(rejected, "unsupported dtype descriptor query must reject");
      continue;
    }
    const auto rank = value.input.dimensions.size();
    if (type == FLAGDNN_DATA_FLOAT16 && rank > 2 && value.axis != 0 &&
        value.axis != static_cast<std::int32_t>(rank - 1)) {
      require_vendor(operation, tensors, "middle-axis");
      bool rejected = false;
      try { (void)hv::hipdnn_reduction_descriptors(operation, tensors); }
      catch (const std::invalid_argument &) { rejected = true; }
      require(rejected, "middle-axis descriptor query must reject");
      ++middle;
      continue;
    }
    if (type == FLAGDNN_DATA_FLOAT16) ++fp16;
    else ++fp32;
    require_supported(operation, tensors, value.name);
    auto negative_axis = operation;
    negative_axis.reduction_axis -= static_cast<std::int32_t>(rank);
    require_supported(negative_axis, tensors, value.name + " negative axis");
    const auto mapped = hv::hipdnn_reduction_descriptors(operation, tensors);
    for (std::size_t i = 0; i < 2; ++i)
      require(mapped[i].uid == tensors[i].uid &&
                  mapped[i].data_type == tensors[i].data_type &&
                  mapped[i].binding_byte_offset == tensors[i].binding_byte_offset &&
                  io::element_count(mapped[i]) == io::element_count(tensors[i]) &&
                  io::storage_element_count(mapped[i]) ==
                      io::storage_element_count(tensors[i]),
              value.name + " descriptor changed logical/physical storage");
    auto mapped_axis = value.axis;
    if (type == FLAGDNN_DATA_FLOAT16 && rank > 2) {
      require(mapped[0].dimensions.size() == 2 &&
                  mapped[1].dimensions.size() == 2,
              "higher-rank mixed output must use its validated rank-two view");
      mapped_axis = value.axis == 0 ? 0 : 1;
      ++views;
    } else
      require(mapped[0].dimensions == value.input.dimensions &&
                  mapped[0].strides == value.input.strides,
              "same-dtype and rank-two descriptors changed unexpectedly");
    const auto raw = io::encode(
        flagdnn::testing::reduction_host_input(value), value.input.data_type);
    const auto input = io::decode(raw, value.input.data_type,
                                  io::element_count(value.input));
    require(cpu::evaluate_reduction(
                {value.input.dimensions, value.axis, value.mode}, input) ==
                cpu::evaluate_reduction(
                    {mapped[0].dimensions, mapped_axis, value.mode}, input),
            value.name + " descriptor view changed reduction semantics");
    if (type == FLAGDNN_DATA_FLOAT16) {
      changed = tensors;
      changed[0].binding_byte_offset = 18;
      require_supported(operation, changed,
                        value.name + " independent FP16 input alignment");
      const auto shifted = hv::hipdnn_reduction_descriptors(operation, changed);
      require(shifted[0].binding_byte_offset == 18 &&
                  shifted[1].binding_byte_offset == 32,
              "each descriptor must retain its own byte offset");
      changed = tensors;
      changed[0].strides[0] += 1;
      require_vendor(operation, changed, "dense");
      changed = tensors;
      changed[1].strides[0] += 1;
      require_vendor(operation, changed, "dense");
    }
  }
  require(fp16 == 30 && fp32 == 36 && bf16 == 16 && int32 == 9 &&
              middle == 6 && views == 9,
          "expected 97 cases: FP16 30 native/6 middle, FP32 36, BF16 16, INT32 9");
  const auto operation =
      hv::make_hipdnn_reduction_operation(FLAGDNN_REDUCTION_ADD, 1, false);
  Tensors same_bf16 = {
      hv::ReferenceTensor{1, FLAGDNN_DATA_BFLOAT16, {2, 4}, {4, 1}, 0},
      hv::ReferenceTensor{2, FLAGDNN_DATA_BFLOAT16, {2}, {1}, 0}};
  require_vendor(operation, same_bf16, "no validated executable primitive");
  Tensors same_fp16 = same_bf16;
  same_fp16[0].data_type = FLAGDNN_DATA_FLOAT16;
  same_fp16[1].data_type = FLAGDNN_DATA_FLOAT16;
  require_supported(operation, same_fp16, "same-dtype FP16");
  const auto slice = hv::make_hipdnn_slice_operation({{0, 2}, {0, 4}}, {1, 1});
  const Tensors mixed_slice = {
      hv::ReferenceTensor{1, FLAGDNN_DATA_FLOAT16, {2, 4}, {4, 1}, 16},
      hv::ReferenceTensor{2, FLAGDNN_DATA_FLOAT32, {2, 4}, {4, 1}, 32}};
  require_invalid(slice, mixed_slice, "mixed slice");
  const auto large_axis =
      hv::make_hipdnn_reduction_operation(FLAGDNN_REDUCTION_ADD, 2, false);
  const Tensors large = {
      hv::ReferenceTensor{1, FLAGDNN_DATA_FLOAT16, {1048576, 1048576, 2},
                          {2097152, 2, 1}, 16},
      hv::ReferenceTensor{2, FLAGDNN_DATA_FLOAT32, {1048576, 1048576},
                          {1048576, 1}, 32}};
  require_vendor(large_axis, large, "do not fit");
  std::cout << "Hygon reduction metadata contract PASS: fp16=30 (views=9) "
               "fp32=36 bf16_rejected=16 int32_rejected=9 middle_rejected=6; "
               "alignment/shape/axis/stride/storage/overflow guards checked\n";
}
} // namespace

int main() {
  try { contract(); }
  catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
