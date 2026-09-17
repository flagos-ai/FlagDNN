// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "benchmark/corex_cudnn_capability_policy.hpp"

#include <algorithm>
#include <iterator>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace flagdnn::iluvatar::validation::benchmark {
namespace {

using flagdnn::benchmarking::BenchmarkCase;
using flagdnn::benchmarking::ConvolutionMode;
using flagdnn::benchmarking::ProviderCapability;
using flagdnn::benchmarking::TensorSpec;

struct Evidence {
  std::string_view case_name;
  const CorexCudnnCatalogRecord *record = nullptr;
};

std::string data_type_token(flagdnnDataType_t data_type) {
  switch (data_type) {
  case FLAGDNN_DATA_INT32:
    return "int32";

  case FLAGDNN_DATA_FP8_E8M0:
    return "fp8_e8m0";
  case FLAGDNN_DATA_FLOAT32:
    return "fp32";
  case FLAGDNN_DATA_FLOAT16:
    return "fp16";
  case FLAGDNN_DATA_BFLOAT16:
    return "bfloat16";
  case FLAGDNN_DATA_BOOLEAN:
    return "bool";
  case FLAGDNN_DATA_FP8_E4M3:
    return "fp8_e4m3";
  case FLAGDNN_DATA_FP8_E5M2:
    return "fp8_e5m2";
  }
  throw std::invalid_argument("unknown benchmark data type");
}

bool contains_token(std::string_view value, std::string_view token) {
  const std::string surrounded = "_" + std::string(token) + "_";
  const std::string suffix = "_" + std::string(token);
  return value.find(surrounded) != std::string_view::npos ||
         value.ends_with(suffix);
}

bool is_contiguous(const TensorSpec &tensor) {
  return tensor.strides ==
         flagdnn::benchmarking::contiguous_strides(tensor.dimensions);
}

template <typename Predicate>
void retain(std::vector<Evidence> &evidence, Predicate &&predicate,
            std::string_view operation, std::string_view qualifier) {
  std::vector<Evidence> selected;
  std::copy_if(evidence.begin(), evidence.end(), std::back_inserter(selected),
               std::forward<Predicate>(predicate));
  if (selected.empty()) {
    throw std::runtime_error(
        "CoreX cuDNN catalog has no benchmark evidence for " +
        std::string(operation) + " qualifier=" + std::string(qualifier));
  }
  evidence = std::move(selected);
}

std::optional<ProviderCapability>
uniform_capability(const std::vector<Evidence> &evidence) {
  bool supported = false;
  bool unsupported = false;
  std::string reason;
  for (const Evidence &item : evidence) {
    if (item.record == nullptr) {
      throw std::logic_error("CoreX cuDNN benchmark evidence is null");
    }
    switch (item.record->classification) {
    case CorexCudnnCatalogClassification::kCandidate:
      throw std::runtime_error(
          "unqualified CoreX cuDNN candidate reached benchmark evidence: " +
          std::string(item.case_name));
    case CorexCudnnCatalogClassification::kQualifiedSupported:
      supported = true;
      break;
    case CorexCudnnCatalogClassification::kSemanticUnsupported:
    case CorexCudnnCatalogClassification::kVendorUnsupported:
      unsupported = true;
      if (reason.empty()) {
        reason = item.record->reason_code;
      } else if (reason != item.record->reason_code) {
        return std::nullopt;
      }
      break;
    }
  }
  if (supported && !unsupported) {
    return ProviderCapability{};
  }
  if (!supported && unsupported && !reason.empty()) {
    return ProviderCapability::unsupported(std::move(reason));
  }
  return std::nullopt;
}

std::string reduction_token(flagdnnReductionMode_t mode) {
  switch (mode) {
  case FLAGDNN_REDUCTION_ADD:
    return "reduction_sum";
  case FLAGDNN_REDUCTION_AVG:
    return "reduction_avg";
  case FLAGDNN_REDUCTION_MUL:
    return "reduction_mul";
  }
  throw std::invalid_argument("unknown benchmark reduction mode");
}

ProviderCapability finish(const std::vector<Evidence> &evidence,
                          std::string_view operation) {
  if (const auto capability = uniform_capability(evidence);
      capability.has_value()) {
    return *capability;
  }
  throw std::runtime_error(
      "CoreX cuDNN benchmark qualification remains ambiguous for " +
      std::string(operation));
}

} // namespace

ProviderCapability
benchmark_capability_from_catalog(const CorexCudnnCapabilityCatalog &catalog,
                                  const BenchmarkCase &specification,
                                  std::string_view operation) {
  const auto operation_iterator =
      catalog.records().find(std::string(operation));
  if (operation_iterator == catalog.records().end()) {
    throw std::runtime_error("CoreX cuDNN catalog has no benchmark operator " +
                             std::string(operation));
  }
  if (specification.name.ends_with("_shared")) {
    const auto name =
        specification.name.substr(0, specification.name.size() - 7);
    const auto exact = operation_iterator->second.find(name);
    if (exact == operation_iterator->second.end())
      throw std::runtime_error(
          "shared benchmark has no exact functional evidence: " + name);
    return finish({{exact->first, &exact->second}}, operation);
  }
  std::vector<Evidence> evidence;
  evidence.reserve(operation_iterator->second.size());
  for (const auto &[case_name, record] : operation_iterator->second) {
    // Default benchmark groups use same-dtype IEEE semantics. The extra
    // precision/output groups are resolved against exact shared cases above.
    if (((operation == "relu_backward" || operation == "elu_backward" ||
          operation == "leaky_relu_backward") &&
         case_name.find("_attributes_") != std::string::npos) ||
        case_name.ends_with("_tf32") ||
        (operation == "reduction" &&
         case_name.find("_to_fp32_") != std::string::npos))
      continue;
    evidence.push_back({case_name, &record});
  }
  if (const auto capability = uniform_capability(evidence);
      capability.has_value()) {
    return *capability;
  }
  if (specification.tensors.empty()) {
    throw std::invalid_argument("benchmark capability case has no tensors");
  }

  const std::string dtype =
      data_type_token(specification.tensors.front().data_type);
  retain(
      evidence,
      [&](const Evidence &item) {
        return contains_token(item.case_name, dtype) ||
               (dtype == "bfloat16" && contains_token(item.case_name, "bf16"));
      },
      operation, dtype);
  if (const auto capability = uniform_capability(evidence);
      capability.has_value()) {
    return *capability;
  }

  using flagdnn::benchmarking::Operation;
  switch (specification.operation) {
  case Operation::kPointwise: {
    const bool explicitly_strided =
        specification.name.find("_strided_") != std::string_view::npos;
    retain(
        evidence,
        [explicitly_strided](const Evidence &item) {
          const bool named_strided =
              item.case_name.find("_strided_") != std::string_view::npos;
          return explicitly_strided == named_strided;
        },
        operation, explicitly_strided ? "explicit_strided" : "standard_layout");
    break;
  }
  case Operation::kBatchnormInference: {
    const bool contiguous = is_contiguous(specification.tensors.front());
    retain(
        evidence,
        [contiguous](const Evidence &item) {
          const bool channels_last =
              item.case_name.find("_channels_last") != std::string_view::npos;
          return contiguous ? !channels_last : channels_last;
        },
        operation, contiguous ? "contiguous" : "channels_last");
    break;
  }
  case Operation::kConvolutionFprop:
  case Operation::kConvolutionDgrad:
  case Operation::kConvolutionWgrad: {
    const int rank = specification.convolution.spatial_rank;
    if (rank < 1 || rank > 3) {
      throw std::invalid_argument(
          "benchmark convolution spatial rank is invalid");
    }
    const std::string prefix = "conv" + std::to_string(rank) + "d_";
    retain(
        evidence,
        [&](const Evidence &item) {
          return item.case_name.starts_with(prefix);
        },
        operation, prefix);
    if (rank == 1 && specification.operation == Operation::kConvolutionFprop &&
        !uniform_capability(evidence).has_value()) {
      const bool asymmetric = specification.convolution.pre_padding !=
                              specification.convolution.post_padding;
      retain(
          evidence,
          [asymmetric](const Evidence &item) {
            return asymmetric == (item.case_name.find("_asymmetric_") !=
                                  std::string_view::npos);
          },
          operation, asymmetric ? "asymmetric_padding" : "symmetric_padding");
    }
    if (specification.operation == Operation::kConvolutionDgrad ||
        specification.operation == Operation::kConvolutionWgrad) {
      const bool convolution =
          specification.convolution.mode == ConvolutionMode::kConvolution;
      retain(
          evidence,
          [convolution](const Evidence &item) {
            const bool named_convolution =
                item.case_name.find("_convolution_mode") !=
                std::string_view::npos;
            return convolution == named_convolution;
          },
          operation, convolution ? "convolution" : "cross_correlation");
    }
    if (specification.operation == Operation::kConvolutionWgrad &&
        !uniform_capability(evidence).has_value()) {
      const auto &input = specification.tensors.front();
      const bool channels_last = input.strides[1] == 1;
      retain(
          evidence,
          [channels_last](const Evidence &item) {
            const bool named_channels_last =
                item.case_name.find("_nwc_") != std::string_view::npos ||
                item.case_name.find("_nhwc_") != std::string_view::npos ||
                item.case_name.find("_ndhwc_") != std::string_view::npos;
            return channels_last == named_channels_last;
          },
          operation, channels_last ? "channels_last" : "contiguous");
    }
    break;
  }
  case Operation::kReduction: {
    const std::string mode = reduction_token(specification.reduction_mode);
    retain(
        evidence,
        [&](const Evidence &item) {
          return item.case_name.starts_with(mode + "_");
        },
        operation, mode);
    const TensorSpec &input = specification.tensors.front();
    const bool unaligned = input.binding_byte_offset != 0;
    const bool strided = !is_contiguous(input);
    retain(
        evidence,
        [unaligned, strided](const Evidence &item) {
          const bool named_unaligned =
              item.case_name.find("_unaligned_input_entrance_") !=
              std::string_view::npos;
          const bool named_channels_last =
              item.case_name.find("_channels_last_") != std::string_view::npos;
          if (unaligned) {
            return named_unaligned;
          }
          if (strided) {
            return named_channels_last;
          }
          return !named_unaligned && !named_channels_last;
        },
        operation,
        unaligned ? "unaligned" : (strided ? "strided" : "contiguous"));
    break;
  }
  default:
    break;
  }
  return finish(evidence, operation);
}

} // namespace flagdnn::iluvatar::validation::benchmark
