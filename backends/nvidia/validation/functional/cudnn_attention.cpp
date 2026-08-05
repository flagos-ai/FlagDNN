/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/attention.hpp"
#include "validation/functional/cudnn_graph.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace flagdnn::testing {
namespace {

namespace cfe = cuda::cfe;

template <typename Attributes>
void apply_options(Attributes& attributes,
                   const AttentionOptions& options) {
  if (options.attention_scale.has_value()) {
    attributes.set_attn_scale(*options.attention_scale);
  }
  attributes.set_diagonal_alignment(
      options.diagonal_alignment == AttentionDiagonalAlignment::kTopLeft
          ? cfe::DiagonalAlignment_t::TOP_LEFT
          : cfe::DiagonalAlignment_t::BOTTOM_RIGHT);
  if (options.diagonal_band_left_bound.has_value()) {
    attributes.set_diagonal_band_left_bound(
        *options.diagonal_band_left_bound);
  }
  if (options.diagonal_band_right_bound.has_value()) {
    attributes.set_diagonal_band_right_bound(
        *options.diagonal_band_right_bound);
  }
}

void set_output(std::shared_ptr<cfe::graph::Tensor_attributes>& tensor,
                const TestTensor& specification,
                std::string_view name) {
  tensor->set_name(std::string(name))
      .set_uid(specification.uid)
      .set_data_type(
          cuda::cudnn_frontend_data_type(specification.data_type))
      .set_dim(specification.dimensions)
      .set_stride(specification.strides)
      .set_output(true);
}

class CudnnSdpaExecutable final : public cuda::CudnnGraphExecutable {
 public:
  explicit CudnnSdpaExecutable(const SdpaTestCase& test_case)
      : graph_(std::make_shared<cfe::graph::Graph>()) {
    validate_sdpa_case(test_case);
    graph_->set_name(test_case.name + "::cudnn")
        .set_io_data_type(
            cuda::cudnn_frontend_data_type(test_case.q.data_type))
        .set_intermediate_data_type(cfe::DataType_t::FLOAT)
        .set_compute_data_type(cfe::DataType_t::FLOAT);
    const auto q = cuda::make_cudnn_tensor(graph_, test_case.q, "q");
    const auto k = cuda::make_cudnn_tensor(graph_, test_case.k, "k");
    const auto v = cuda::make_cudnn_tensor(graph_, test_case.v, "v");
    cfe::graph::SDPA_attributes attributes;
    attributes.set_name("sdpa")
        .set_generate_stats(test_case.stats.has_value());
    apply_options(attributes, test_case.options);
    if (test_case.bias.has_value()) {
      attributes.set_bias(
          cuda::make_cudnn_tensor(graph_, *test_case.bias, "bias"));
    }
    auto result = graph_->sdpa(q, k, v, attributes);
    set_output(result[0], test_case.output, "output");
    if (test_case.stats.has_value()) {
      if (result[1] == nullptr) {
        throw std::logic_error("cuDNN SDPA did not return requested stats");
      }
      set_output(result[1], *test_case.stats, "stats");
    } else if (result[1] != nullptr) {
      throw std::logic_error("cuDNN SDPA returned unrequested stats");
    }

    cuda::check_cudnn_frontend(
        graph_->build(handle(), {cfe::HeurMode_t::A}),
        "cuDNN SDPA graph build");
    std::int64_t workspace_size = 0;
    cuda::check_cudnn_frontend(graph_->get_workspace_size(workspace_size),
                               "cuDNN SDPA workspace query");
    set_workspace_size(workspace_size);
  }

  void execute(std::span<const flagdnnBinding_t> bindings,
               void* workspace,
               std::size_t workspace_size,
               flagdnnStream_t stream) override {
    begin_execute(workspace, workspace_size, stream);
    cuda::CudnnBindingMap pointers =
        cuda::make_cudnn_binding_map(bindings);
    cuda::check_cudnn_frontend(
        graph_->execute(handle(), pointers, workspace),
        "cuDNN SDPA graph execute");
  }

 private:
  std::shared_ptr<cfe::graph::Graph> graph_;
};

class CudnnSdpaBackwardExecutable final
    : public cuda::CudnnGraphExecutable {
 public:
  explicit CudnnSdpaBackwardExecutable(
      const SdpaBackwardTestCase& test_case)
      : graph_(std::make_shared<cfe::graph::Graph>()) {
    validate_sdpa_backward_case(test_case);
    graph_->set_name(test_case.name + "::cudnn")
        .set_io_data_type(
            cuda::cudnn_frontend_data_type(test_case.q.data_type))
        .set_intermediate_data_type(cfe::DataType_t::FLOAT)
        .set_compute_data_type(cfe::DataType_t::FLOAT);
    const auto q = cuda::make_cudnn_tensor(graph_, test_case.q, "q");
    const auto k = cuda::make_cudnn_tensor(graph_, test_case.k, "k");
    const auto v = cuda::make_cudnn_tensor(graph_, test_case.v, "v");
    const auto output =
        cuda::make_cudnn_tensor(graph_, test_case.output, "output");
    const auto doutput =
        cuda::make_cudnn_tensor(graph_, test_case.doutput, "doutput");
    const auto stats =
        cuda::make_cudnn_tensor(graph_, test_case.stats, "stats");
    cfe::graph::SDPA_backward_attributes attributes;
    attributes.set_name("sdpa_backward")
        .set_deterministic_algorithm(test_case.deterministic);
    apply_options(attributes, test_case.options);
    if (test_case.bias.has_value()) {
      attributes.set_bias(
          cuda::make_cudnn_tensor(graph_, *test_case.bias, "bias"));
    }
    if (test_case.dbias.has_value()) {
      auto dbias =
          cuda::make_cudnn_tensor(graph_, *test_case.dbias, "dbias");
      dbias->set_output(true);
      attributes.set_dbias(std::move(dbias));
    }
    auto gradients = graph_->sdpa_backward(
        q, k, v, output, doutput, stats, attributes);
    set_output(gradients[0], test_case.dq, "dq");
    set_output(gradients[1], test_case.dk, "dk");
    set_output(gradients[2], test_case.dv, "dv");

    cuda::check_cudnn_frontend(
        graph_->build(handle(), {cfe::HeurMode_t::A}),
        "cuDNN SDPA backward graph build");
    std::int64_t workspace_size = 0;
    cuda::check_cudnn_frontend(graph_->get_workspace_size(workspace_size),
                               "cuDNN SDPA backward workspace query");
    set_workspace_size(workspace_size);
  }

  void execute(std::span<const flagdnnBinding_t> bindings,
               void* workspace,
               std::size_t workspace_size,
               flagdnnStream_t stream) override {
    begin_execute(workspace, workspace_size, stream);
    cuda::CudnnBindingMap pointers =
        cuda::make_cudnn_binding_map(bindings);
    cuda::check_cudnn_frontend(
        graph_->execute(handle(), pointers, workspace),
        "cuDNN SDPA backward graph execute");
  }

 private:
  std::shared_ptr<cfe::graph::Graph> graph_;
};

class CudnnSdpaFp8Executable final : public cuda::CudnnGraphExecutable {
 public:
  explicit CudnnSdpaFp8Executable(const SdpaFp8TestCase& test_case)
      : graph_(std::make_shared<cfe::graph::Graph>()) {
    validate_sdpa_fp8_case(test_case);
    graph_->set_name(test_case.name + "::cudnn")
        .set_io_data_type(
            cuda::cudnn_frontend_data_type(test_case.q.data_type))
        .set_intermediate_data_type(cfe::DataType_t::FLOAT)
        .set_compute_data_type(cfe::DataType_t::FLOAT);
    const auto q = cuda::make_cudnn_tensor(graph_, test_case.q, "q");
    const auto k = cuda::make_cudnn_tensor(graph_, test_case.k, "k");
    const auto v = cuda::make_cudnn_tensor(graph_, test_case.v, "v");
    const auto scalar = [&](const Fp8Scalar& value,
                            std::string_view name) {
      return cuda::make_cudnn_tensor(graph_, value.tensor, name);
    };
    cfe::graph::SDPA_fp8_attributes attributes;
    attributes.set_name("sdpa_fp8")
        .set_generate_stats(test_case.stats.has_value());
    apply_options(attributes, test_case.options);
    if (test_case.bias.has_value()) {
      attributes.set_bias(
          cuda::make_cudnn_tensor(graph_, *test_case.bias, "bias"));
    }
    auto result = graph_->sdpa_fp8(
        q,
        k,
        v,
        scalar(test_case.descale_q, "descale_q"),
        scalar(test_case.descale_k, "descale_k"),
        scalar(test_case.descale_v, "descale_v"),
        scalar(test_case.descale_s, "descale_s"),
        scalar(test_case.scale_s, "scale_s"),
        scalar(test_case.scale_o, "scale_o"),
        attributes);
    set_output(result[0], test_case.output, "output");
    if (test_case.stats.has_value()) {
      if (result[1] == nullptr) {
        throw std::logic_error(
            "cuDNN FP8 SDPA did not return requested stats");
      }
      set_output(result[1], *test_case.stats, "stats");
    } else if (result[1] != nullptr) {
      throw std::logic_error("cuDNN FP8 SDPA returned unrequested stats");
    }
    set_output(result[2], test_case.amax_s, "amax_s");
    set_output(result[3], test_case.amax_o, "amax_o");

    cuda::check_cudnn_frontend(
        graph_->build(handle(), {cfe::HeurMode_t::A}),
        "cuDNN FP8 SDPA graph build");
    std::int64_t workspace_size = 0;
    cuda::check_cudnn_frontend(graph_->get_workspace_size(workspace_size),
                               "cuDNN FP8 SDPA workspace query");
    set_workspace_size(workspace_size);
  }

  void execute(std::span<const flagdnnBinding_t> bindings,
               void* workspace,
               std::size_t workspace_size,
               flagdnnStream_t stream) override {
    begin_execute(workspace, workspace_size, stream);
    cuda::CudnnBindingMap pointers =
        cuda::make_cudnn_binding_map(bindings);
    cuda::check_cudnn_frontend(
        graph_->execute(handle(), pointers, workspace),
        "cuDNN FP8 SDPA graph execute");
  }

 private:
  std::shared_ptr<cfe::graph::Graph> graph_;
};

class CudnnSdpaFp8BackwardExecutable final
    : public cuda::CudnnGraphExecutable {
 public:
  explicit CudnnSdpaFp8BackwardExecutable(
      const SdpaFp8BackwardTestCase& test_case)
      : graph_(std::make_shared<cfe::graph::Graph>()) {
    validate_sdpa_fp8_backward_case(test_case);
    graph_->set_name(test_case.name + "::cudnn")
        .set_io_data_type(
            cuda::cudnn_frontend_data_type(test_case.q.data_type))
        .set_intermediate_data_type(cfe::DataType_t::FLOAT)
        .set_compute_data_type(cfe::DataType_t::FLOAT);
    const auto q = cuda::make_cudnn_tensor(graph_, test_case.q, "q");
    const auto k = cuda::make_cudnn_tensor(graph_, test_case.k, "k");
    const auto v = cuda::make_cudnn_tensor(graph_, test_case.v, "v");
    const auto output =
        cuda::make_cudnn_tensor(graph_, test_case.output, "output");
    const auto doutput =
        cuda::make_cudnn_tensor(graph_, test_case.doutput, "doutput");
    const auto stats =
        cuda::make_cudnn_tensor(graph_, test_case.stats, "stats");
    const auto scalar = [&](const Fp8Scalar& value,
                            std::string_view name) {
      return cuda::make_cudnn_tensor(graph_, value.tensor, name);
    };
    cfe::graph::SDPA_fp8_backward_attributes attributes;
    attributes.set_name("sdpa_fp8_backward");
    apply_options(attributes, test_case.options);
    auto result = graph_->sdpa_fp8_backward(
        q,
        k,
        v,
        output,
        doutput,
        stats,
        scalar(test_case.descale_q, "descale_q"),
        scalar(test_case.descale_k, "descale_k"),
        scalar(test_case.descale_v, "descale_v"),
        scalar(test_case.descale_o, "descale_o"),
        scalar(test_case.descale_doutput, "descale_doutput"),
        scalar(test_case.descale_s, "descale_s"),
        scalar(test_case.descale_dp, "descale_dp"),
        scalar(test_case.scale_s, "scale_s"),
        scalar(test_case.scale_dq, "scale_dq"),
        scalar(test_case.scale_dk, "scale_dk"),
        scalar(test_case.scale_dv, "scale_dv"),
        scalar(test_case.scale_dp, "scale_dp"),
        attributes);
    set_output(result[0], test_case.dq, "dq");
    set_output(result[1], test_case.dk, "dk");
    set_output(result[2], test_case.dv, "dv");
    set_output(result[3], test_case.amax_dq, "amax_dq");
    set_output(result[4], test_case.amax_dk, "amax_dk");
    set_output(result[5], test_case.amax_dv, "amax_dv");
    set_output(result[6], test_case.amax_dp, "amax_dp");

    cuda::check_cudnn_frontend(
        graph_->build(handle(), {cfe::HeurMode_t::A}),
        "cuDNN FP8 SDPA backward graph build");
    std::int64_t workspace_size = 0;
    cuda::check_cudnn_frontend(graph_->get_workspace_size(workspace_size),
                               "cuDNN FP8 SDPA backward workspace query");
    set_workspace_size(workspace_size);
  }

  void execute(std::span<const flagdnnBinding_t> bindings,
               void* workspace,
               std::size_t workspace_size,
               flagdnnStream_t stream) override {
    begin_execute(workspace, workspace_size, stream);
    cuda::CudnnBindingMap pointers =
        cuda::make_cudnn_binding_map(bindings);
    cuda::check_cudnn_frontend(
        graph_->execute(handle(), pointers, workspace),
        "cuDNN FP8 SDPA backward graph execute");
  }

 private:
  std::shared_ptr<cfe::graph::Graph> graph_;
};

}  // namespace

std::unique_ptr<AttentionExecutable> build_sdpa_reference(
    const SdpaTestCase& test_case) {
  return std::make_unique<CudnnSdpaExecutable>(test_case);
}

std::unique_ptr<AttentionExecutable> build_sdpa_backward_reference(
    const SdpaBackwardTestCase& test_case) {
  return std::make_unique<CudnnSdpaBackwardExecutable>(test_case);
}

std::unique_ptr<AttentionExecutable> build_sdpa_fp8_reference(
    const SdpaFp8TestCase& test_case) {
  return std::make_unique<CudnnSdpaFp8Executable>(test_case);
}

std::unique_ptr<AttentionExecutable> build_sdpa_fp8_backward_reference(
    const SdpaFp8BackwardTestCase& test_case) {
  return std::make_unique<CudnnSdpaFp8BackwardExecutable>(test_case);
}

}  // namespace flagdnn::testing
