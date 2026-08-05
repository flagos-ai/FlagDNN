/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "ops.hpp"

#include "validation/benchmark/cudnn_common.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace flagdnn::benchmarking::cudnn_detail {
namespace {

std::int64_t element_count(const TensorSpec& specification) {
  std::int64_t result = 1;
  for (const std::int64_t dimension : specification.dimensions) {
    result *= dimension;
  }
  return result;
}

TensorSpec flatten_tensor(const TensorSpec& specification) {
  TensorSpec result = specification;
  const std::int64_t elements = element_count(specification);
  result.dimensions = {1, elements, 1, 1};
  result.strides = {elements, 1, 1, 1};
  return result;
}

bool contains_convolution(const BenchmarkCase& specification) {
  for (const GraphNodeSpec& node : specification.graph.nodes) {
    if (node.operation == Operation::kConvolutionFprop) {
      return true;
    }
  }
  return false;
}

TensorSpec canonical_tensor(const TensorSpec& specification,
                            bool preserve_layout) {
  return preserve_layout ? specification : flatten_tensor(specification);
}

const TensorSpec& tensor_spec(const BenchmarkCase& specification,
                              std::int64_t uid) {
  for (const TensorSpec& tensor : specification.tensors) {
    if (tensor.uid == uid) {
      return tensor;
    }
  }
  for (const TensorSpec& tensor : specification.graph.intermediates) {
    if (tensor.uid == uid) {
      return tensor;
    }
  }
  throw std::invalid_argument(
      "cuDNN graph node references unknown tensor UID");
}

fe::PointwiseMode_t pointwise_mode(flagdnnPointwiseMode_t mode) {
  switch (mode) {
    case FLAGDNN_POINTWISE_RELU_FWD:
      return fe::PointwiseMode_t::RELU_FWD;
    case FLAGDNN_POINTWISE_ADD:
      return fe::PointwiseMode_t::ADD;
    case FLAGDNN_POINTWISE_MUL:
      return fe::PointwiseMode_t::MUL;
    case FLAGDNN_POINTWISE_BINARY_SELECT:
      return fe::PointwiseMode_t::BINARY_SELECT;
    default:
      throw std::invalid_argument(
          "cuDNN graph reference does not support this pointwise mode");
  }
}

class GraphExecutable final : public ExecutableBase {
 public:
  explicit GraphExecutable(const BenchmarkCase& specification) {
    if (specification.operation != Operation::kGraph ||
        specification.tensors.size() < 2 ||
        specification.graph.nodes.empty()) {
      throw std::invalid_argument("cuDNN graph case is invalid");
    }

    const bool preserve_layout = contains_convolution(specification);
    graph_ = std::make_shared<fe::graph::Graph>();
    graph_->set_name(specification.name)
        .set_io_data_type(data_type(specification.tensors[0].data_type))
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    using Tensor = std::shared_ptr<fe::graph::Tensor_attributes>;
    std::unordered_map<std::int64_t, Tensor> values;
    for (std::size_t index = 0;
         index + 1 < specification.tensors.size();
         ++index) {
      const TensorSpec canonical = canonical_tensor(
          specification.tensors[index], preserve_layout);
      if (!values.emplace(
              canonical.uid,
              make_tensor(graph_,
                          canonical,
                          "graph_input_" +
                              std::to_string(canonical.uid),
                          false))
               .second) {
        throw std::invalid_argument(
            "cuDNN graph external tensor UID is duplicate");
      }
    }

    const auto value = [&](std::int64_t uid) -> Tensor {
      const auto found = values.find(uid);
      if (found == values.end()) {
        throw std::invalid_argument(
            "cuDNN graph nodes are not in dependency order");
      }
      return found->second;
    };

    for (const GraphNodeSpec& node : specification.graph.nodes) {
      Tensor output;
      if (node.operation == Operation::kPointwise) {
        if (node.input_uids.size() != 1 &&
            node.input_uids.size() != 2 &&
            node.input_uids.size() != 3) {
          throw std::invalid_argument(
              "cuDNN graph pointwise node requires one, two, or three inputs");
        }
        fe::graph::Pointwise_attributes attributes;
        attributes.set_name(node.name)
            .set_mode(pointwise_mode(node.pointwise_mode))
            .set_compute_data_type(fe::DataType_t::FLOAT);
        if (node.input_uids.size() == 1) {
          output = graph_->pointwise(
              value(node.input_uids[0]), attributes);
        } else if (node.input_uids.size() == 2) {
          output = graph_->pointwise(
              value(node.input_uids[0]),
              value(node.input_uids[1]),
              attributes);
        } else {
          output = graph_->pointwise(
              value(node.input_uids[0]),
              value(node.input_uids[1]),
              value(node.input_uids[2]),
              attributes);
        }
      } else if (node.operation == Operation::kConvolutionFprop) {
        if (node.input_uids.size() != 2 ||
            node.convolution.groups != 1) {
          throw std::invalid_argument(
              "cuDNN graph convolution requires input, filter, "
              "and groups=1");
        }
        output = graph_->conv_fprop(
            value(node.input_uids[0]),
            value(node.input_uids[1]),
            fe::graph::Conv_fprop_attributes()
                .set_name(node.name)
                .set_pre_padding(node.convolution.pre_padding)
                .set_post_padding(node.convolution.post_padding)
                .set_stride(node.convolution.stride)
                .set_dilation(node.convolution.dilation));
      } else {
        throw std::invalid_argument(
            "cuDNN graph node operation is not implemented");
      }

      const TensorSpec canonical = canonical_tensor(
          tensor_spec(specification, node.output_uid), preserve_layout);
      output->set_name(node.name + "_output")
          .set_uid(canonical.uid)
          .set_data_type(data_type(canonical.data_type))
          .set_dim(canonical.dimensions)
          .set_stride(canonical.strides);
      if (node.output_uid == specification.tensors.back().uid) {
        output->set_output(true);
      } else {
        output->set_is_virtual(true);
      }
      if (!values.emplace(node.output_uid, output).second) {
        throw std::invalid_argument(
            "cuDNN graph node output UID has multiple producers");
      }
    }

    if (values.find(specification.tensors.back().uid) == values.end()) {
      throw std::invalid_argument(
          "cuDNN graph case does not produce its external output");
    }
    check_frontend(
        graph_->build(handle(), {fe::HeurMode_t::A}),
        "cuDNN graph build");
    std::int64_t workspace = 0;
    check_frontend(
        graph_->get_workspace_size(workspace),
        "cuDNN graph workspace query");
    set_workspace_size(workspace);
  }

  void execute(std::span<const flagdnnBinding_t> bindings,
               void* workspace,
               std::size_t workspace_size,
               flagdnnStream_t stream) override {
    begin_execute(workspace, workspace_size, stream);
    BindingMap pointers = make_binding_map(bindings);
    check_frontend(graph_->execute(handle(), pointers, workspace),
                   "cuDNN graph execute");
  }

 private:
  std::shared_ptr<fe::graph::Graph> graph_;
};

}  // namespace

std::unique_ptr<BenchmarkExecutable> build_graph(
    const BenchmarkCase& specification) {
  return std::make_unique<GraphExecutable>(specification);
}

}  // namespace flagdnn::benchmarking::cudnn_detail
