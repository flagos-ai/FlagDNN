/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "platform_provider_skeleton.hpp"
#include "common/cases.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using flagdnn::benchmarking::BenchmarkCase;
using flagdnn::benchmarking::Operation;
using flagdnn::benchmarking::TensorSpec;

void append(std::vector<BenchmarkCase>& destination,
            std::vector<BenchmarkCase> source) {
  destination.insert(destination.end(),
                     std::make_move_iterator(source.begin()),
                     std::make_move_iterator(source.end()));
}

void validate_tensor(const TensorSpec& tensor) {
  if (tensor.uid <= 0 ||
      tensor.dimensions.size() != tensor.strides.size()) {
    throw std::runtime_error("catalog tensor metadata is invalid");
  }
  for (std::size_t axis = 0; axis < tensor.dimensions.size(); ++axis) {
    if (tensor.dimensions[axis] <= 0 || tensor.strides[axis] <= 0) {
      throw std::runtime_error(
          "catalog tensor dimensions and strides must be positive");
    }
  }
}

void validate_case(const BenchmarkCase& specification) {
  if (specification.name.empty() || specification.tensors.size() < 2) {
    throw std::runtime_error("catalog case identity is invalid");
  }
  if (specification.output_count == 0 ||
      specification.output_count >= specification.tensors.size()) {
    throw std::runtime_error("catalog case output count is invalid");
  }
  std::set<std::int64_t> uids;
  for (const TensorSpec& tensor : specification.tensors) {
    validate_tensor(tensor);
    if (!uids.insert(tensor.uid).second) {
      throw std::runtime_error("catalog tensor UIDs must be unique");
    }
  }
  if (!specification.input_domains.empty() &&
      specification.input_domains.size() + specification.output_count !=
          specification.tensors.size()) {
    throw std::runtime_error(
        "catalog per-tensor input domains must cover every input");
  }
  for (const TensorSpec& tensor : specification.graph.intermediates) {
    validate_tensor(tensor);
    if (!uids.insert(tensor.uid).second) {
      throw std::runtime_error("catalog tensor UIDs must be unique");
    }
  }
  if (specification.operation != Operation::kGraph) {
    if (specification.operation == Operation::kConvolutionFprop ||
        specification.operation == Operation::kConvolutionDgrad ||
        specification.operation == Operation::kConvolutionWgrad) {
      const std::size_t spatial_rank = static_cast<std::size_t>(
          specification.convolution.spatial_rank);
      if (spatial_rank == 0 || spatial_rank > 3 ||
          specification.convolution.pre_padding.size() != spatial_rank ||
          specification.convolution.post_padding.size() != spatial_rank ||
          specification.convolution.stride.size() != spatial_rank ||
          specification.convolution.dilation.size() != spatial_rank ||
          specification.convolution.groups <= 0) {
        throw std::runtime_error(
            "catalog convolution metadata is invalid");
      }
    }
    if (!specification.graph.nodes.empty() ||
        !specification.graph.intermediates.empty()) {
      throw std::runtime_error(
          "non-graph case contains graph-only metadata");
    }
    return;
  }
  if (specification.output_count != 1) {
    throw std::runtime_error(
        "generic graph catalog cases currently require one output");
  }
  if (specification.graph.nodes.empty()) {
    throw std::runtime_error("graph case has no nodes");
  }
  std::set<std::int64_t> available;
  for (std::size_t index = 0;
       index < input_tensor_count(specification);
       ++index) {
    available.insert(specification.tensors[index].uid);
  }
  std::set<std::int64_t> produced;
  for (const auto& node : specification.graph.nodes) {
    if (node.name.empty() || node.input_uids.empty() ||
        uids.find(node.output_uid) == uids.end()) {
      throw std::runtime_error("graph node metadata is invalid");
    }
    if (node.operation == Operation::kPointwise) {
      if ((node.input_uids.size() != 1 && node.input_uids.size() != 2 &&
           node.input_uids.size() != 3) ||
          node.pointwise_mode == FLAGDNN_POINTWISE_NOT_SET) {
        throw std::runtime_error(
            "graph pointwise node metadata is invalid");
      }
    } else if (node.operation == Operation::kConvolutionFprop) {
      const std::size_t spatial_rank = static_cast<std::size_t>(
          node.convolution.spatial_rank);
      if (node.input_uids.size() != 2 || spatial_rank == 0 ||
          node.convolution.pre_padding.size() != spatial_rank ||
          node.convolution.post_padding.size() != spatial_rank ||
          node.convolution.stride.size() != spatial_rank ||
          node.convolution.dilation.size() != spatial_rank ||
          node.convolution.groups <= 0) {
        throw std::runtime_error(
            "graph convolution node metadata is invalid");
      }
    } else {
      throw std::runtime_error(
          "graph node operation is not supported by the catalog contract");
    }
    for (const std::int64_t input_uid : node.input_uids) {
      if (available.find(input_uid) == available.end()) {
        throw std::runtime_error(
            "graph nodes must be in dependency order");
      }
    }
    if (!produced.insert(node.output_uid).second ||
        !available.insert(node.output_uid).second) {
      throw std::runtime_error(
          "graph node output must have exactly one producer");
    }
  }
  if (available.find(specification.tensors.back().uid) ==
      available.end()) {
    throw std::runtime_error(
        "graph case does not produce its external output");
  }
}

std::size_t operation_index(Operation operation) {
  switch (operation) {
    case Operation::kRelu:
      return 0;
    case Operation::kPointwise:
      return 1;
    case Operation::kAdd:
      return 2;
    case Operation::kReduction:
      return 3;
    case Operation::kConvolutionFprop:
      return 4;
    case Operation::kConvolutionDgrad:
      return 5;
    case Operation::kConvolutionWgrad:
      return 6;
    case Operation::kMatmul:
      return 7;
    case Operation::kReshape:
      return 8;
    case Operation::kTranspose:
      return 9;
    case Operation::kSlice:
      return 10;
    case Operation::kLayernorm:
      return 11;
    case Operation::kRmsnorm:
      return 12;
    case Operation::kBatchnorm:
      return 13;
    case Operation::kBatchnormInference:
      return 14;
    case Operation::kGraph:
      return 15;
  }
  throw std::runtime_error("unknown catalog operation");
}

}  // namespace

int main() {
  try {
    std::vector<BenchmarkCase> cases = flagdnn::benchmarking::all_cases();
    append(cases,
           flagdnn::benchmarking::unary_pointwise_cases(
               FLAGDNN_POINTWISE_ABS, "abs"));
    append(cases,
           flagdnn::benchmarking::binary_pointwise_cases(
               FLAGDNN_POINTWISE_SUB, "sub"));
    append(cases, flagdnn::benchmarking::binary_select_cases());
    append(cases, flagdnn::benchmarking::add_square_cases());
    append(cases, flagdnn::benchmarking::conv_bias_relu_benchmark_cases());
    if (cases.empty()) {
      throw std::runtime_error("platform-neutral catalog is empty");
    }

    flagdnn::benchmarking::PlatformProviderSkeleton provider(
        "second_platform_skeleton",
        "native vendor reference adapter is not connected");
    std::set<std::string> names;
    std::array<std::size_t, 16> operation_counts{};
    std::size_t unsupported_count = 0;
    for (const BenchmarkCase& specification : cases) {
      validate_case(specification);
      if (!names.insert(specification.name).second) {
        throw std::runtime_error("catalog case names must be unique");
      }
      ++operation_counts[operation_index(specification.operation)];
      const auto capability = provider.capability(specification);
      if (capability.supported || capability.reason.empty()) {
        throw std::runtime_error(
            "provider skeleton must report an explicit capability gap");
      }
      ++unsupported_count;
    }
    if (std::any_of(operation_counts.begin(),
                    operation_counts.end(),
                    [](std::size_t count) { return count == 0; })) {
      throw std::runtime_error(
          "platform contract does not cover every native operation family");
    }

    bool build_rejected = false;
    try {
      (void)provider.build(cases.front());
    } catch (const std::logic_error&) {
      build_rejected = true;
    }
    if (!build_rejected) {
      throw std::runtime_error(
          "provider skeleton fabricated an executable");
    }

    std::cout << "PASS platform-neutral-catalog cases=" << cases.size()
              << " explicit_unsupported=" << unsupported_count
              << " families=" << operation_counts.size() << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "REFERENCE_CATALOG_CONTRACT_FAILED: " << error.what()
              << '\n';
    return 1;
  }
}
