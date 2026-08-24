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
  std::vector<Evidence> evidence;
  evidence.reserve(operation_iterator->second.size());
  for (const auto &[case_name, record] : operation_iterator->second) {
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
        return contains_token(item.case_name, dtype);
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
