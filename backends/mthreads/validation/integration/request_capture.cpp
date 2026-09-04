/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include <flagdnn/flagdnn.hpp>
#include <flagdnn_frontend.h>

#include <array>
#include <cstddef>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

namespace fe = flagdnn_frontend;

fe::graph::Graph::Tensor make_tensor(
    fe::graph::Graph& graph,
    std::string name,
    std::int64_t uid) {
  return graph.tensor(
      fe::graph::Tensor_attributes()
          .set_name(std::move(name))
          .set_uid(uid)
          .set_data_type(fe::DataType_t::FLOAT)
          .set_dim({2, 3, 4})
          .set_stride({12, 4, 1})
          .set_alignment(16));
}

fe::graph::Graph::Tensor make_tensor(
    fe::graph::Graph& graph,
    std::string name,
    std::int64_t uid,
    std::vector<std::int64_t> dimensions,
    std::vector<std::int64_t> strides) {
  return graph.tensor(
      fe::graph::Tensor_attributes()
          .set_name(std::move(name))
          .set_uid(uid)
          .set_data_type(fe::DataType_t::FLOAT)
          .set_dim(std::move(dimensions))
          .set_stride(std::move(strides))
          .set_alignment(16));
}

fe::graph::Graph::Tensor make_typed_tensor(
    fe::graph::Graph& graph,
    std::string name,
    std::int64_t uid,
    fe::DataType_t data_type,
    std::vector<std::int64_t> dimensions,
    std::vector<std::int64_t> strides) {
  return graph.tensor(
      fe::graph::Tensor_attributes()
          .set_name(std::move(name))
          .set_uid(uid)
          .set_data_type(data_type)
          .set_dim(std::move(dimensions))
          .set_stride(std::move(strides))
          .set_alignment(16));
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 4) {
      throw std::invalid_argument(
          "usage: request_capture <python> <capture-compiler> <cache>");
    }
    const char* capture =
        std::getenv("FLAGDNN_MTHREADS_CAPTURE_REQUEST");
    if (capture == nullptr || *capture == '\0') {
      throw std::runtime_error(
          "FLAGDNN_MTHREADS_CAPTURE_REQUEST is required");
    }

    flagdnn::Handle handle("mthreads", 0);
    handle.set_compiler(argv[1], argv[2], argv[3]);

    fe::graph::Graph graph;
    graph.set_name("mthreads public Graph capture")
        .set_io_data_type(fe::DataType_t::FLOAT)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT)
        .set_autotune(false);
    const char* configured_graph =
        std::getenv("FLAGDNN_MTHREADS_CAPTURE_GRAPH");
    const std::string_view graph_kind =
        configured_graph == nullptr || configured_graph[0] == '\0'
            ? std::string_view("add")
            : std::string_view(configured_graph);
    if (graph_kind == "add") {
      const auto left = make_tensor(graph, "left", 100);
      const auto right = make_tensor(graph, "right", 101);
      auto output = graph.pointwise(
          left,
          right,
          fe::graph::Pointwise_attributes()
              .set_name("add")
              .set_mode(fe::PointwiseMode_t::ADD)
              .set_compute_data_type(fe::DataType_t::FLOAT)
              .set_alpha(-0.75));
      output->set_name("output")
          .set_uid(102)
          .set_data_type(fe::DataType_t::FLOAT)
          .set_dim({2, 3, 4})
          .set_stride({12, 4, 1})
          .set_alignment(16)
          .set_output(true);
    } else if (graph_kind == "add_square") {
      const auto left = make_tensor(graph, "left", 100);
      const auto right = make_tensor(graph, "right", 101);
      auto square = graph.pointwise(
          right,
          right,
          fe::graph::Pointwise_attributes()
              .set_name("square")
              .set_mode(fe::PointwiseMode_t::MUL)
              .set_compute_data_type(fe::DataType_t::FLOAT));
      square->set_name("square")
          .set_uid(103)
          .set_data_type(fe::DataType_t::FLOAT)
          .set_dim({2, 3, 4})
          .set_stride({12, 4, 1})
          .set_alignment(16)
          .set_is_virtual(true);
      auto output = graph.pointwise(
          left,
          square,
          fe::graph::Pointwise_attributes()
              .set_name("add_square")
              .set_mode(fe::PointwiseMode_t::ADD)
              .set_compute_data_type(fe::DataType_t::FLOAT));
      output->set_name("output")
          .set_uid(102)
          .set_data_type(fe::DataType_t::FLOAT)
          .set_dim({2, 3, 4})
          .set_stride({12, 4, 1})
          .set_alignment(16)
          .set_output(true);
    } else if (graph_kind == "conv_bias_relu") {
      const auto input = make_tensor(
          graph, "input", 200, {1, 4, 5, 6}, {120, 1, 24, 4});
      const auto filter = make_tensor(
          graph, "filter", 201, {6, 4, 3, 3}, {36, 1, 12, 4});
      const auto bias = make_tensor(
          graph, "bias", 202, {1, 6, 1, 1}, {6, 1, 6, 6});
      auto convolution = graph.conv_fprop(
          input,
          filter,
          fe::graph::Conv_fprop_attributes()
              .set_name("convolution")
              .set_compute_data_type(fe::DataType_t::FLOAT)
              .set_pre_padding({1, 1})
              .set_post_padding({1, 1})
              .set_stride({1, 1})
              .set_dilation({1, 1})
              .set_convolution_mode(
                  fe::ConvolutionMode_t::CROSS_CORRELATION)
              .set_groups(1));
      convolution->set_name("convolution")
          .set_uid(204)
          .set_data_type(fe::DataType_t::FLOAT)
          .set_dim({1, 6, 5, 6})
          .set_stride({180, 1, 36, 6})
          .set_alignment(16)
          .set_is_virtual(true);
      auto biased = graph.pointwise(
          convolution,
          bias,
          fe::graph::Pointwise_attributes()
              .set_name("bias_add")
              .set_mode(fe::PointwiseMode_t::ADD)
              .set_compute_data_type(fe::DataType_t::FLOAT));
      biased->set_name("biased")
          .set_uid(205)
          .set_data_type(fe::DataType_t::FLOAT)
          .set_dim({1, 6, 5, 6})
          .set_stride({180, 1, 36, 6})
          .set_alignment(16)
          .set_is_virtual(true);
      auto output = graph.pointwise(
          biased,
          fe::graph::Pointwise_attributes()
              .set_name("relu")
              .set_mode(fe::PointwiseMode_t::RELU_FWD)
              .set_compute_data_type(fe::DataType_t::FLOAT));
      output->set_name("output")
          .set_uid(203)
          .set_data_type(fe::DataType_t::FLOAT)
          .set_dim({1, 6, 5, 6})
          .set_stride({180, 1, 36, 6})
          .set_alignment(16)
          .set_output(true);
    } else if (graph_kind == "layernorm") {
      const auto x = make_tensor(
          graph, "x", 300, {2, 3, 4}, {12, 4, 1});
      const auto scale = make_tensor(
          graph, "scale", 301, {1, 1, 4}, {4, 4, 1});
      const auto bias = make_tensor(
          graph, "bias", 302, {1, 1, 4}, {4, 4, 1});
      auto outputs = graph.layernorm(
          x,
          scale,
          bias,
          fe::graph::Layernorm_attributes()
              .set_name("layernorm")
              .set_compute_data_type(fe::DataType_t::FLOAT)
              .set_forward_phase(fe::NormFwdPhase_t::TRAINING)
              .set_epsilon(1.0e-3F));
      outputs[0]->set_name("y")
          .set_uid(303)
          .set_data_type(fe::DataType_t::FLOAT)
          .set_dim({2, 3, 4})
          .set_stride({12, 4, 1})
          .set_alignment(16)
          .set_output(true);
      outputs[1]->set_name("mean")
          .set_uid(304)
          .set_data_type(fe::DataType_t::FLOAT)
          .set_dim({2, 3, 1})
          .set_stride({3, 1, 1})
          .set_alignment(16)
          .set_output(true);
      outputs[2]->set_name("inv_variance")
          .set_uid(305)
          .set_data_type(fe::DataType_t::FLOAT)
          .set_dim({2, 3, 1})
          .set_stride({3, 1, 1})
          .set_alignment(16)
          .set_output(true);
    } else if (graph_kind == "rmsnorm") {
      const auto x = make_tensor(
          graph, "x", 310, {2, 3, 4}, {12, 4, 1});
      const auto scale = make_tensor(
          graph, "scale", 311, {1, 1, 4}, {4, 4, 1});
      auto bias = make_tensor(
          graph, "bias", 312, {1, 1, 4}, {4, 4, 1});
      auto epsilon = graph.tensor(
          1.0e-3F, fe::graph::ScalarType::COMPILE_TIME_CONST);
      auto outputs = graph.rmsnorm(
          x,
          scale,
          fe::graph::Rmsnorm_attributes()
              .set_name("rmsnorm")
              .set_compute_data_type(fe::DataType_t::FLOAT)
              .set_forward_phase(fe::NormFwdPhase_t::TRAINING)
              .set_bias(bias)
              .set_epsilon(epsilon));
      outputs[0]->set_name("y")
          .set_uid(313)
          .set_data_type(fe::DataType_t::FLOAT)
          .set_dim({2, 3, 4})
          .set_stride({12, 4, 1})
          .set_alignment(16)
          .set_output(true);
      outputs[1]->set_name("inv_variance")
          .set_uid(314)
          .set_data_type(fe::DataType_t::FLOAT)
          .set_dim({2, 3, 1})
          .set_stride({3, 1, 1})
          .set_alignment(16)
          .set_output(true);
    } else if (graph_kind == "batchnorm") {
      const auto x = make_tensor(
          graph, "x", 320, {2, 4, 3, 5}, {60, 1, 20, 4});
      const auto scale = make_tensor(
          graph, "scale", 321, {1, 4, 1, 1}, {4, 1, 1, 1});
      const auto bias = make_tensor(
          graph, "bias", 322, {1, 4, 1, 1}, {4, 1, 1, 1});
      auto previous_mean = make_tensor(
          graph,
          "previous_running_mean",
          323,
          {1, 4, 1, 1},
          {4, 1, 1, 1});
      auto previous_variance = make_tensor(
          graph,
          "previous_running_variance",
          324,
          {1, 4, 1, 1},
          {4, 1, 1, 1});
      auto epsilon = graph.tensor(
          1.0e-3F, fe::graph::ScalarType::COMPILE_TIME_CONST);
      auto momentum = graph.tensor(
          0.1F, fe::graph::ScalarType::COMPILE_TIME_CONST);
      auto outputs = graph.batchnorm(
          x,
          scale,
          bias,
          fe::graph::Batchnorm_attributes()
              .set_name("batchnorm")
              .set_compute_data_type(fe::DataType_t::FLOAT)
              .set_previous_running_stats(
                  previous_mean, previous_variance, momentum)
              .set_epsilon(epsilon));
      outputs[0]->set_name("y")
          .set_uid(325)
          .set_data_type(fe::DataType_t::FLOAT)
          .set_dim({2, 4, 3, 5})
          .set_stride({60, 1, 20, 4})
          .set_alignment(16)
          .set_output(true);
      for (std::size_t index = 1; index < outputs.size(); ++index) {
        outputs[index]
            ->set_name(
                index == 1
                    ? "mean"
                    : (index == 2
                           ? "inv_variance"
                           : (index == 3
                                  ? "next_running_mean"
                                  : "next_running_variance")))
            .set_uid(325 + static_cast<std::int64_t>(index))
            .set_data_type(fe::DataType_t::FLOAT)
            .set_dim({1, 4, 1, 1})
            .set_stride({4, 1, 1, 1})
            .set_alignment(16)
            .set_output(true);
      }
    } else if (graph_kind == "batchnorm_inference") {
      const auto x = make_tensor(
          graph, "x", 340, {2, 4, 3, 5}, {60, 1, 20, 4});
      const auto mean = make_tensor(
          graph, "mean", 341, {1, 4, 1, 1}, {4, 1, 1, 1});
      const auto inv_variance = make_tensor(
          graph,
          "inv_variance",
          342,
          {1, 4, 1, 1},
          {4, 1, 1, 1});
      const auto scale = make_tensor(
          graph, "scale", 343, {1, 4, 1, 1}, {4, 1, 1, 1});
      const auto bias = make_tensor(
          graph, "bias", 344, {1, 4, 1, 1}, {4, 1, 1, 1});
      auto output = graph.batchnorm_inference(
          x,
          mean,
          inv_variance,
          scale,
          bias,
          fe::graph::Batchnorm_inference_attributes()
              .set_name("batchnorm_inference")
              .set_compute_data_type(fe::DataType_t::FLOAT));
      output->set_name("y")
          .set_uid(345)
          .set_data_type(fe::DataType_t::FLOAT)
          .set_dim({2, 4, 3, 5})
          .set_stride({60, 1, 20, 4})
          .set_alignment(16)
          .set_output(true);
    } else if (graph_kind == "sdpa") {
      const auto q = make_typed_tensor(
          graph, "q", 400, fe::DataType_t::HALF,
          {1, 2, 16, 32}, {1024, 512, 32, 1});
      const auto k = make_typed_tensor(
          graph, "k", 401, fe::DataType_t::HALF,
          {1, 2, 24, 32}, {1536, 768, 32, 1});
      const auto v = make_typed_tensor(
          graph, "v", 402, fe::DataType_t::HALF,
          {1, 2, 24, 48}, {2304, 1152, 48, 1});
      const auto bias = make_typed_tensor(
          graph, "bias", 403, fe::DataType_t::HALF,
          {1, 2, 16, 24}, {768, 384, 24, 1});
      auto outputs = graph.sdpa(
          q,
          k,
          v,
          fe::graph::SDPA_attributes()
              .set_name("sdpa")
              .set_compute_data_type(fe::DataType_t::FLOAT)
              .set_attn_scale(0.176776695F)
              .set_generate_stats(true)
              .set_diagonal_band_right_bound(0)
              .set_bias(bias));
      outputs[0]->set_name("output")
          .set_uid(404)
          .set_data_type(fe::DataType_t::HALF)
          .set_dim({1, 2, 16, 48})
          .set_stride({1536, 768, 48, 1})
          .set_alignment(16)
          .set_output(true);
      outputs[1]->set_name("stats")
          .set_uid(405)
          .set_data_type(fe::DataType_t::FLOAT)
          .set_dim({1, 2, 16, 1})
          .set_stride({32, 16, 1, 1})
          .set_alignment(16)
          .set_output(true);
    } else if (graph_kind == "sdpa_backward") {
      const auto q = make_typed_tensor(
          graph, "q", 410, fe::DataType_t::HALF,
          {1, 2, 16, 32}, {1024, 512, 32, 1});
      const auto k = make_typed_tensor(
          graph, "k", 411, fe::DataType_t::HALF,
          {1, 2, 24, 32}, {1536, 768, 32, 1});
      const auto v = make_typed_tensor(
          graph, "v", 412, fe::DataType_t::HALF,
          {1, 2, 24, 48}, {2304, 1152, 48, 1});
      const auto output = make_typed_tensor(
          graph, "output", 413, fe::DataType_t::HALF,
          {1, 2, 16, 48}, {1536, 768, 48, 1});
      const auto doutput = make_typed_tensor(
          graph, "doutput", 414, fe::DataType_t::HALF,
          {1, 2, 16, 48}, {1536, 768, 48, 1});
      const auto stats = make_typed_tensor(
          graph, "stats", 415, fe::DataType_t::FLOAT,
          {1, 2, 16, 1}, {32, 16, 1, 1});
      auto gradients = graph.sdpa_backward(
          q,
          k,
          v,
          output,
          doutput,
          stats,
          fe::graph::SDPA_backward_attributes()
              .set_name("sdpa_backward")
              .set_compute_data_type(fe::DataType_t::FLOAT)
              .set_attn_scale(0.176776695F)
              .set_deterministic_algorithm(true));
      const std::array<std::int64_t, 3> uids{416, 417, 418};
      const std::array<std::vector<std::int64_t>, 3> dimensions{
          std::vector<std::int64_t>{1, 2, 16, 32},
          std::vector<std::int64_t>{1, 2, 24, 32},
          std::vector<std::int64_t>{1, 2, 24, 48}};
      const std::array<std::vector<std::int64_t>, 3> strides{
          std::vector<std::int64_t>{1024, 512, 32, 1},
          std::vector<std::int64_t>{1536, 768, 32, 1},
          std::vector<std::int64_t>{2304, 1152, 48, 1}};
      for (std::size_t index = 0; index < gradients.size(); ++index) {
        gradients[index]
            ->set_uid(uids[index])
            .set_data_type(fe::DataType_t::HALF)
            .set_dim(dimensions[index])
            .set_stride(strides[index])
            .set_alignment(16)
            .set_output(true);
      }
    } else if (graph_kind == "sdpa_fp8") {
      const auto fp8 = fe::DataType_t::FP8_E4M3;
      const auto q = make_typed_tensor(
          graph, "q", 420, fp8,
          {1, 2, 16, 128}, {4096, 2048, 128, 1});
      const auto k = make_typed_tensor(
          graph, "k", 421, fp8,
          {1, 2, 24, 128}, {6144, 3072, 128, 1});
      const auto v = make_typed_tensor(
          graph, "v", 422, fp8,
          {1, 2, 24, 128}, {6144, 3072, 128, 1});
      const auto scalar = [&](std::string name, std::int64_t uid) {
        return make_typed_tensor(
            graph, std::move(name), uid, fe::DataType_t::FLOAT,
            {1, 1, 1, 1}, {1, 1, 1, 1});
      };
      auto outputs = graph.sdpa_fp8(
          q,
          k,
          v,
          scalar("descale_q", 423),
          scalar("descale_k", 424),
          scalar("descale_v", 425),
          scalar("descale_s", 426),
          scalar("scale_s", 427),
          scalar("scale_o", 428),
          fe::graph::SDPA_fp8_attributes()
              .set_name("sdpa_fp8")
              .set_compute_data_type(fe::DataType_t::FLOAT)
              .set_attn_scale(0.0883883476F)
              .set_generate_stats(true));
      outputs[0]->set_name("output")
          .set_uid(429)
          .set_data_type(fp8)
          .set_dim({1, 2, 16, 128})
          .set_stride({4096, 2048, 128, 1})
          .set_alignment(16)
          .set_output(true);
      outputs[1]->set_name("stats")
          .set_uid(430)
          .set_data_type(fe::DataType_t::FLOAT)
          .set_dim({1, 2, 16, 1})
          .set_stride({32, 16, 1, 1})
          .set_alignment(16)
          .set_output(true);
      for (std::size_t index = 2; index < outputs.size(); ++index) {
        outputs[index]
            ->set_uid(429 + static_cast<std::int64_t>(index))
            .set_data_type(fe::DataType_t::FLOAT)
            .set_dim({1, 1, 1, 1})
            .set_stride({1, 1, 1, 1})
            .set_alignment(16)
            .set_output(true);
      }
    } else if (graph_kind == "sdpa_fp8_backward") {
      const auto fp8 = fe::DataType_t::FP8_E4M3;
      const auto tensor4 = [&](std::string name, std::int64_t uid) {
        return make_typed_tensor(
            graph, std::move(name), uid, fp8,
            {1, 2, 16, 128}, {4096, 2048, 128, 1});
      };
      const auto scalar = [&](std::string name, std::int64_t uid) {
        return make_typed_tensor(
            graph, std::move(name), uid, fe::DataType_t::FLOAT,
            {1, 1, 1, 1}, {1, 1, 1, 1});
      };
      auto outputs = graph.sdpa_fp8_backward(
          tensor4("q", 440),
          tensor4("k", 441),
          tensor4("v", 442),
          tensor4("output", 443),
          tensor4("doutput", 444),
          make_typed_tensor(
              graph, "stats", 445, fe::DataType_t::FLOAT,
              {1, 2, 16, 1}, {32, 16, 1, 1}),
          scalar("descale_q", 446),
          scalar("descale_k", 447),
          scalar("descale_v", 448),
          scalar("descale_o", 449),
          scalar("descale_doutput", 450),
          scalar("descale_s", 451),
          scalar("descale_dp", 452),
          scalar("scale_s", 453),
          scalar("scale_dq", 454),
          scalar("scale_dk", 455),
          scalar("scale_dv", 456),
          scalar("scale_dp", 457),
          fe::graph::SDPA_fp8_backward_attributes()
              .set_name("sdpa_fp8_backward")
              .set_compute_data_type(fe::DataType_t::FLOAT)
              .set_attn_scale(0.0883883476F));
      for (std::size_t index = 0; index < outputs.size(); ++index) {
        outputs[index]
            ->set_uid(458 + static_cast<std::int64_t>(index))
            .set_data_type(index < 3 ? fp8 : fe::DataType_t::FLOAT)
            .set_dim(index < 3 ? std::vector<std::int64_t>{1, 2, 16, 128}
                               : std::vector<std::int64_t>{1, 1, 1, 1})
            .set_stride(index < 3
                            ? std::vector<std::int64_t>{4096, 2048, 128, 1}
                            : std::vector<std::int64_t>{1, 1, 1, 1})
            .set_alignment(16)
            .set_output(true);
      }
    } else {
      throw std::invalid_argument(
          "FLAGDNN_MTHREADS_CAPTURE_GRAPH is unsupported");
    }

    const fe::error_t status = graph.build(handle, {fe::HeurMode_t::A});
    if (
        !status.is_bad()
        || status.get_status() != FLAGDNN_STATUS_COMPILATION_FAILED
        || !std::filesystem::is_regular_file(capture)
    ) {
      throw std::runtime_error(
          "public Graph request capture did not stop at the compiler "
          "boundary: " +
          status.get_message());
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
