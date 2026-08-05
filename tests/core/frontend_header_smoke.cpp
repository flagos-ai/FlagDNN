#include <flagdnn_frontend.h>

#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace fe = flagdnn_frontend;

static_assert(!std::is_copy_constructible_v<fe::graph::Graph>);
static_assert(std::is_move_constructible_v<fe::graph::Graph>);

int main() {
  fe::graph::Graph relu_graph;
  relu_graph.set_name("relu")
      .set_io_data_type(fe::DataType_t::FLOAT)
      .set_intermediate_data_type(fe::DataType_t::FLOAT)
      .set_compute_data_type(fe::DataType_t::FLOAT);
  const auto input = relu_graph.tensor(
      fe::graph::Tensor_attributes()
          .set_name("x")
          .set_uid(1)
          .set_dim({2, 3})
          .set_stride({3, 1}));
  const auto output = relu_graph.pointwise(
      input,
      fe::graph::Pointwise_attributes()
          .set_name("relu")
          .set_mode(fe::PointwiseMode_t::RELU_FWD));
  output->set_name("y").set_uid(2).set_output(true);

  fe::graph::Graph abs_graph;
  abs_graph.set_name("abs").set_io_data_type(fe::DataType_t::FLOAT);
  const auto abs_input = abs_graph.tensor(
      fe::graph::Tensor_attributes()
          .set_name("abs_x")
          .set_uid(11)
          .set_dim({2, 3})
          .set_stride({3, 1}));
  const auto abs_output = abs_graph.pointwise(
      abs_input,
      fe::graph::Pointwise_attributes()
          .set_name("abs")
          .set_mode(fe::PointwiseMode_t::ABS)
          .set_compute_data_type(fe::DataType_t::FLOAT));
  abs_output->set_name("abs_y").set_uid(12).set_output(true);
  const auto sigmoid_output = abs_graph.pointwise(
      abs_input,
      fe::graph::Pointwise_attributes()
          .set_name("sigmoid")
          .set_mode(fe::PointwiseMode_t::SIGMOID_FWD)
          .set_compute_data_type(fe::DataType_t::FLOAT));
  const auto tanh_output = abs_graph.pointwise(
      abs_input,
      fe::graph::Pointwise_attributes()
          .set_name("tanh")
          .set_mode(fe::PointwiseMode_t::TANH_FWD)
          .set_compute_data_type(fe::DataType_t::FLOAT));
  const auto leaky_relu_output = abs_graph.pointwise(
      abs_input,
      fe::graph::Pointwise_attributes()
          .set_name("leaky_relu")
          .set_mode(fe::PointwiseMode_t::RELU_FWD)
          .set_relu_lower_clip_slope(0.2F));
  const auto elu_output = abs_graph.pointwise(
      abs_input,
      fe::graph::Pointwise_attributes()
          .set_name("elu")
          .set_mode(fe::PointwiseMode_t::ELU_FWD)
          .set_elu_alpha(1.0F));
  const auto gelu_output = abs_graph.pointwise(
      abs_input,
      fe::graph::Pointwise_attributes()
          .set_name("gelu")
          .set_mode(fe::PointwiseMode_t::GELU_FWD));
  const auto gelu_approx_output = abs_graph.pointwise(
      abs_input,
      fe::graph::Pointwise_attributes()
          .set_name("gelu_approx_tanh")
          .set_mode(fe::PointwiseMode_t::GELU_APPROX_TANH_FWD));
  const auto softplus_output = abs_graph.pointwise(
      abs_input,
      fe::graph::Pointwise_attributes()
          .set_name("softplus")
          .set_mode(fe::PointwiseMode_t::SOFTPLUS_FWD)
          .set_softplus_beta(1.0F));
  fe::graph::Pointwise_attributes swish_attributes;
  swish_attributes.set_name("swish")
      .set_mode(fe::PointwiseMode_t::SWISH_FWD)
      .set_swish_beta(1.25F);
  const auto swish_output =
      abs_graph.pointwise(abs_input, swish_attributes);

  fe::graph::Graph add_graph;
  add_graph.set_io_data_type(fe::DataType_t::HALF);
  const auto left = add_graph.tensor(
      fe::graph::Tensor_attributes()
          .set_uid(3)
          .set_dim({2, 3, 4})
          .set_stride({12, 4, 1}));
  const auto right = add_graph.tensor(
      fe::graph::Tensor_attributes()
          .set_uid(4)
          .set_dim({1, 4})
          .set_stride({4, 1}));
  const auto sum = add_graph.pointwise(
      left,
      right,
      fe::graph::Pointwise_attributes()
          .set_mode(fe::PointwiseMode_t::ADD)
          .set_alpha(-0.75));
  sum->set_uid(5).set_output(true);

  fe::graph::Graph binary_graph;
  binary_graph.set_io_data_type(fe::DataType_t::HALF);
  const auto binary_left = binary_graph.tensor(
      fe::graph::Tensor_attributes()
          .set_uid(13)
          .set_dim({2, 3, 4})
          .set_stride({12, 4, 1}));
  const auto binary_right = binary_graph.tensor(
      fe::graph::Tensor_attributes()
          .set_uid(14)
          .set_dim({1, 3, 4})
          .set_stride({12, 4, 1}));
  const auto difference = binary_graph.pointwise(
      binary_left,
      binary_right,
      fe::graph::Pointwise_attributes()
          .set_mode(fe::PointwiseMode_t::SUB)
          .set_alpha(-2.0));
  difference->set_uid(15).set_output(true);

  const auto sigmoid_backward_input = binary_graph.tensor(
      fe::graph::Tensor_attributes()
          .set_uid(19)
          .set_dim({2, 3, 4})
          .set_stride({12, 4, 1}));
  const auto sigmoid_gradient = binary_graph.pointwise(
      binary_left,
      sigmoid_backward_input,
      fe::graph::Pointwise_attributes()
          .set_mode(fe::PointwiseMode_t::SIGMOID_BWD));
  sigmoid_gradient->set_uid(20).set_output(true);
  bool invalid_sigmoid_backward_rejected = false;
  try {
    (void)binary_graph.pointwise(
        binary_left,
        binary_right,
        fe::graph::Pointwise_attributes()
            .set_mode(fe::PointwiseMode_t::SIGMOID_BWD));
  } catch (const std::invalid_argument&) {
    invalid_sigmoid_backward_rejected = true;
  }

  const auto select_mask = binary_graph.tensor(
      fe::graph::Tensor_attributes()
          .set_uid(21)
          .set_data_type(fe::DataType_t::BOOLEAN)
          .set_dim({2, 1, 4})
          .set_stride({4, 4, 1}));
  const auto selected = binary_graph.pointwise(
      binary_left,
      sigmoid_backward_input,
      select_mask,
      fe::graph::Pointwise_attributes()
          .set_mode(fe::PointwiseMode_t::BINARY_SELECT));
  selected->set_uid(22).set_output(true);
  bool invalid_binary_select_mask_rejected = false;
  try {
    (void)binary_graph.pointwise(
        binary_left,
        sigmoid_backward_input,
        binary_left,
        fe::graph::Pointwise_attributes()
            .set_mode(fe::PointwiseMode_t::BINARY_SELECT));
  } catch (const std::invalid_argument&) {
    invalid_binary_select_mask_rejected = true;
  }

  const auto comparison = binary_graph.pointwise(
      binary_left,
      binary_right,
      fe::graph::Pointwise_attributes()
          .set_mode(fe::PointwiseMode_t::CMP_LT)
          .set_compute_data_type(fe::DataType_t::BOOLEAN));
  comparison->set_uid(16).set_output(true);

  fe::graph::Graph logical_graph;
  logical_graph.set_io_data_type(fe::DataType_t::BOOLEAN);
  const auto logical_input = logical_graph.tensor(
      fe::graph::Tensor_attributes()
          .set_uid(17)
          .set_dim({2, 3, 4})
          .set_stride({12, 4, 1}));
  const auto logical_output = logical_graph.pointwise(
      logical_input,
      fe::graph::Pointwise_attributes()
          .set_mode(fe::PointwiseMode_t::LOGICAL_NOT)
          .set_compute_data_type(fe::DataType_t::BOOLEAN));
  logical_output->set_uid(18).set_output(true);

  fe::graph::Graph reduction_graph;
  reduction_graph.set_io_data_type(fe::DataType_t::BFLOAT16);
  const auto reduction_input = reduction_graph.tensor(
      fe::graph::Tensor_attributes()
          .set_uid(6)
          .set_dim({2, 3, 4})
          .set_stride({12, 4, 1}));
  const auto reduced = reduction_graph.reduction(
      reduction_input,
      fe::graph::Reduction_attributes()
          .set_mode(fe::ReductionMode_t::AVG)
          .set_axis(1)
          .set_keep_dimensions(true));
  reduced->set_uid(7).set_output(true);

  fe::graph::Graph convolution_graph;
  convolution_graph.set_io_data_type(fe::DataType_t::HALF);
  const auto convolution_input = convolution_graph.tensor(
      fe::graph::Tensor_attributes()
          .set_uid(8)
          .set_dim({1, 2, 8})
          .set_stride({16, 8, 1}));
  const auto filter = convolution_graph.tensor(
      fe::graph::Tensor_attributes()
          .set_uid(9)
          .set_dim({4, 2, 3})
          .set_stride({6, 3, 1}));
  const auto convolution_output = convolution_graph.conv_fprop(
      convolution_input,
      filter,
      fe::graph::Conv_fprop_attributes()
          .set_name("conv1d")
          .set_padding({1})
          .set_stride({1})
          .set_dilation({1})
          .set_groups(1));
  convolution_output->set_uid(10).set_output(true);

  fe::graph::Graph dgrad_graph;
  dgrad_graph.set_io_data_type(fe::DataType_t::HALF)
      .set_compute_data_type(fe::DataType_t::FLOAT);
  const auto dgrad_loss = dgrad_graph.tensor(
      fe::graph::Tensor_attributes()
          .set_uid(40)
          .set_dim({1, 4, 8})
          .set_stride({32, 8, 1}));
  const auto dgrad_filter = dgrad_graph.tensor(
      fe::graph::Tensor_attributes()
          .set_uid(41)
          .set_dim({4, 2, 3})
          .set_stride({6, 3, 1}));
  fe::graph::Conv_dgrad_attributes dgrad_attributes;
  dgrad_attributes.set_name("conv1d_dgrad")
      .set_compute_data_type(fe::DataType_t::FLOAT)
      .set_padding({1})
      .set_stride({1})
      .set_dilation({1})
      .set_convolution_mode(fe::ConvolutionMode_t::CONVOLUTION)
      .set_groups(1);
  const auto dgrad_output = dgrad_graph.conv_dgrad(
      dgrad_loss, dgrad_filter, dgrad_attributes);
  const bool dgrad_requires_output_metadata =
      dgrad_output->get_dim().empty() &&
      dgrad_output->get_stride().empty();
  dgrad_output->set_uid(42)
      .set_dim({1, 2, 8})
      .set_stride({16, 8, 1})
      .set_output(true);
  const auto dgrad_validation = dgrad_graph.validate();

  fe::graph::Graph wgrad_graph;
  wgrad_graph.set_io_data_type(fe::DataType_t::HALF)
      .set_compute_data_type(fe::DataType_t::FLOAT);
  const auto wgrad_loss = wgrad_graph.tensor(
      fe::graph::Tensor_attributes()
          .set_uid(43)
          .set_dim({1, 4, 8})
          .set_stride({32, 8, 1}));
  const auto wgrad_image = wgrad_graph.tensor(
      fe::graph::Tensor_attributes()
          .set_uid(44)
          .set_dim({1, 2, 8})
          .set_stride({16, 8, 1}));
  fe::graph::Conv_wgrad_attributes wgrad_attributes;
  wgrad_attributes.set_name("conv1d_wgrad")
      .set_compute_data_type(fe::DataType_t::FLOAT)
      .set_padding({1})
      .set_stride({1})
      .set_dilation({1})
      .set_groups(1);
  const auto wgrad_output = wgrad_graph.conv_wgrad(
      wgrad_loss, wgrad_image, wgrad_attributes);
  const bool wgrad_requires_output_metadata =
      wgrad_output->get_dim().empty() &&
      wgrad_output->get_stride().empty();
  wgrad_output->set_uid(45)
      .set_dim({4, 2, 3})
      .set_stride({6, 3, 1})
      .set_output(true);
  const auto wgrad_validation = wgrad_graph.validate();

  fe::graph::Graph batchnorm_graph;
  batchnorm_graph.set_name("batchnorm_inference")
      .set_io_data_type(fe::DataType_t::HALF)
      .set_compute_data_type(fe::DataType_t::FLOAT);
  const auto batchnorm_x = batchnorm_graph.tensor(
      fe::graph::Tensor_attributes()
          .set_name("x")
          .set_uid(46)
          .set_dim({2, 8, 4, 4})
          .set_stride({128, 16, 4, 1}));
  const auto batchnorm_mean = batchnorm_graph.tensor(
      fe::graph::Tensor_attributes()
          .set_name("mean")
          .set_uid(47)
          .set_data_type(fe::DataType_t::FLOAT)
          .set_dim({1, 8, 1, 1})
          .set_stride({8, 1, 1, 1}));
  const auto batchnorm_inv_variance = batchnorm_graph.tensor(
      fe::graph::Tensor_attributes()
          .set_name("inv_variance")
          .set_uid(48)
          .set_data_type(fe::DataType_t::FLOAT)
          .set_dim({1, 8, 1, 1})
          .set_stride({8, 1, 1, 1}));
  const auto batchnorm_scale = batchnorm_graph.tensor(
      fe::graph::Tensor_attributes()
          .set_name("scale")
          .set_uid(49)
          .set_data_type(fe::DataType_t::FLOAT)
          .set_dim({1, 8, 1, 1})
          .set_stride({8, 1, 1, 1}));
  const auto batchnorm_bias = batchnorm_graph.tensor(
      fe::graph::Tensor_attributes()
          .set_name("bias")
          .set_uid(50)
          .set_data_type(fe::DataType_t::FLOAT)
          .set_dim({1, 8, 1, 1})
          .set_stride({8, 1, 1, 1}));
  fe::graph::Batchnorm_inference_attributes batchnorm_attributes;
  batchnorm_attributes.set_name("batchnorm_inference")
      .set_compute_data_type(fe::DataType_t::FLOAT);
  const auto batchnorm_output = batchnorm_graph.batchnorm_inference(
      batchnorm_x,
      batchnorm_mean,
      batchnorm_inv_variance,
      batchnorm_scale,
      batchnorm_bias,
      batchnorm_attributes);
  batchnorm_output->set_uid(51).set_output(true);
  const auto batchnorm_validation = batchnorm_graph.validate();

  fe::graph::Graph batchnorm_training_graph;
  batchnorm_training_graph.set_name("batchnorm")
      .set_io_data_type(fe::DataType_t::HALF)
      .set_compute_data_type(fe::DataType_t::FLOAT);
  auto training_x = batchnorm_training_graph.tensor(
      fe::graph::Tensor_attributes()
          .set_uid(110)
          .set_dim({2, 8, 4, 4})
          .set_stride({128, 16, 4, 1}));
  auto training_scale = batchnorm_training_graph.tensor(
      fe::graph::Tensor_attributes()
          .set_uid(111)
          .set_dim({1, 8, 1, 1})
          .set_stride({8, 1, 1, 1}));
  auto training_bias = batchnorm_training_graph.tensor(
      fe::graph::Tensor_attributes()
          .set_uid(112)
          .set_dim({1, 8, 1, 1})
          .set_stride({8, 1, 1, 1}));
  auto previous_mean = batchnorm_training_graph.tensor(
      fe::graph::Tensor_attributes()
          .set_uid(113)
          .set_data_type(fe::DataType_t::FLOAT)
          .set_dim({1, 8, 1, 1})
          .set_stride({8, 1, 1, 1}));
  auto previous_variance = batchnorm_training_graph.tensor(
      fe::graph::Tensor_attributes()
          .set_uid(114)
          .set_data_type(fe::DataType_t::FLOAT)
          .set_dim({1, 8, 1, 1})
          .set_stride({8, 1, 1, 1}));
  auto epsilon = batchnorm_training_graph.tensor(
      1.0e-3F, fe::graph::ScalarType::COMPILE_TIME_CONST);
  auto momentum = batchnorm_training_graph.tensor(
      0.1F, fe::graph::ScalarType::COMPILE_TIME_CONST);
  fe::graph::Batchnorm_attributes training_attributes;
  training_attributes.set_name("batchnorm")
      .set_compute_data_type(fe::DataType_t::FLOAT)
      .set_previous_running_stats(
          previous_mean, previous_variance, momentum)
      .set_epsilon(epsilon);
  const auto training_outputs = batchnorm_training_graph.batchnorm(
      training_x, training_scale, training_bias, training_attributes);
  for (std::size_t index = 0; index < training_outputs.size(); ++index) {
    training_outputs[index]->set_uid(115 + static_cast<std::int64_t>(index))
        .set_output(true);
  }
  const auto batchnorm_training_validation =
      batchnorm_training_graph.validate();


  fe::graph::Graph layernorm_graph;
  layernorm_graph.set_name("layernorm")
      .set_io_data_type(fe::DataType_t::HALF)
      .set_compute_data_type(fe::DataType_t::FLOAT);
  auto layernorm_x = layernorm_graph.tensor(
      fe::graph::Tensor_attributes()
          .set_uid(120)
          .set_dim({2, 5, 17})
          .set_stride({85, 17, 1}));
  auto layernorm_scale = layernorm_graph.tensor(
      fe::graph::Tensor_attributes()
          .set_uid(121)
          .set_dim({1, 1, 17})
          .set_stride({17, 17, 1}));
  auto layernorm_bias = layernorm_graph.tensor(
      fe::graph::Tensor_attributes()
          .set_uid(122)
          .set_dim({1, 1, 17})
          .set_stride({17, 17, 1}));
  fe::graph::Layernorm_attributes layernorm_attributes;
  layernorm_attributes.set_name("layernorm")
      .set_compute_data_type(fe::DataType_t::FLOAT)
      .set_forward_phase(fe::NormFwdPhase_t::TRAINING)
      .set_epsilon(1.0e-3F);
  const auto layernorm_outputs = layernorm_graph.layernorm(
      layernorm_x, layernorm_scale, layernorm_bias, layernorm_attributes);
  for (std::size_t index = 0; index < layernorm_outputs.size(); ++index) {
    layernorm_outputs[index]
        ->set_uid(123 + static_cast<std::int64_t>(index))
        .set_output(true);
  }
  const auto layernorm_validation = layernorm_graph.validate();

  fe::graph::Graph rmsnorm_graph;
  rmsnorm_graph.set_name("rmsnorm")
      .set_io_data_type(fe::DataType_t::HALF)
      .set_compute_data_type(fe::DataType_t::FLOAT);
  auto rmsnorm_x = rmsnorm_graph.tensor(
      fe::graph::Tensor_attributes()
          .set_uid(130)
          .set_dim({2, 5, 17})
          .set_stride({85, 17, 1}));
  auto rmsnorm_scale = rmsnorm_graph.tensor(
      fe::graph::Tensor_attributes()
          .set_uid(131)
          .set_dim({1, 1, 17})
          .set_stride({17, 17, 1}));
  auto rmsnorm_bias = rmsnorm_graph.tensor(
      fe::graph::Tensor_attributes()
          .set_uid(132)
          .set_dim({1, 1, 17})
          .set_stride({17, 17, 1}));
  auto rmsnorm_epsilon = rmsnorm_graph.tensor(
      1.0e-3F, fe::graph::ScalarType::COMPILE_TIME_CONST);
  fe::graph::Rmsnorm_attributes rmsnorm_attributes;
  rmsnorm_attributes.set_name("rmsnorm")
      .set_compute_data_type(fe::DataType_t::FLOAT)
      .set_forward_phase(fe::NormFwdPhase_t::TRAINING)
      .set_bias(rmsnorm_bias)
      .set_epsilon(rmsnorm_epsilon);
  const auto rmsnorm_outputs =
      rmsnorm_graph.rmsnorm(rmsnorm_x, rmsnorm_scale, rmsnorm_attributes);
  for (std::size_t index = 0; index < rmsnorm_outputs.size(); ++index) {
    rmsnorm_outputs[index]
        ->set_uid(133 + static_cast<std::int64_t>(index))
        .set_output(true);
  }
  const auto rmsnorm_validation = rmsnorm_graph.validate();


  fe::graph::Graph matmul_graph;
  matmul_graph.set_io_data_type(fe::DataType_t::HALF)
      .set_compute_data_type(fe::DataType_t::FLOAT);
  const auto matmul_a = matmul_graph.tensor(
      fe::graph::Tensor_attributes()
          .set_uid(19)
          .set_dim({2, 1, 17, 30})
          .set_stride({510, 510, 30, 1}));
  const auto matmul_b = matmul_graph.tensor(
      fe::graph::Tensor_attributes()
          .set_uid(20)
          .set_dim({3, 30, 23})
          .set_stride({690, 23, 1}));
  const auto matmul_output = matmul_graph.matmul(
      matmul_a,
      matmul_b,
      fe::graph::Matmul_attributes()
          .set_name("matmul")
          .set_compute_data_type(fe::DataType_t::FLOAT));
  matmul_output->set_uid(21).set_output(true);

  fe::graph::Graph layout_graph;
  layout_graph.set_name("layout")
      .set_io_data_type(fe::DataType_t::HALF)
      .set_compute_data_type(fe::DataType_t::FLOAT);
  const auto layout_input = layout_graph.tensor(
      fe::graph::Tensor_attributes()
          .set_name("layout_x")
          .set_uid(30)
          .set_dim({2, 3, 4})
          .set_stride({12, 4, 1}));

  fe::graph::Reshape_attributes reshape_attributes;
  reshape_attributes.set_name("reshape")
      .set_compute_data_type(fe::DataType_t::FLOAT)
      .set_dim({6, 4})
      .set_stride({4, 1})
      .set_reshape_mode(fe::ReshapeMode_t::LOGICAL);
  const auto reshaped =
      layout_graph.reshape(layout_input, reshape_attributes);
  reshaped->set_uid(31).set_output(true);

  fe::graph::Transpose_attributes transpose_attributes;
  transpose_attributes.set_name("transpose")
      .set_compute_data_type(fe::DataType_t::FLOAT)
      .set_permutation({2, 0, 1});
  const auto transposed =
      layout_graph.transpose(layout_input, transpose_attributes);
  transposed->set_uid(32).set_output(true);

  fe::graph::Slice_attributes slice_attributes;
  slice_attributes.set_name("slice")
      .set_compute_data_type(fe::DataType_t::FLOAT)
      .set_slices({{0, 2}, {1, 3}, {0, 4}})
      .set_strides({1, 1, 2});
  const auto sliced = layout_graph.slice(layout_input, slice_attributes);
  sliced->set_uid(33).set_output(true);

  bool invalid_reshape_rejected = false;
  try {
    (void)layout_graph.reshape(
        layout_input,
        fe::graph::Reshape_attributes()
            .set_dim({5, 5})
            .set_reshape_mode(fe::ReshapeMode_t::LOGICAL));
  } catch (const std::invalid_argument&) {
    invalid_reshape_rejected = true;
  }
  bool invalid_transpose_rejected = false;
  try {
    (void)layout_graph.transpose(
        layout_input,
        fe::graph::Transpose_attributes().set_permutation({0, 0, 2}));
  } catch (const std::invalid_argument&) {
    invalid_transpose_rejected = true;
  }
  bool invalid_slice_rejected = false;
  try {
    (void)layout_graph.slice(
        layout_input,
        fe::graph::Slice_attributes()
            .set_slices({{0, 2}, {0, 3}, {0, 4}})
            .set_strides({1, 0, 1}));
  } catch (const std::invalid_argument&) {
    invalid_slice_rejected = true;
  }
  const fe::error_t layout_status = layout_graph.validate();
  if (layout_status.is_bad()) {
    return 3;
  }

  fe::graph::Graph mixed_storage_graph;
  mixed_storage_graph.set_io_data_type(fe::DataType_t::HALF)
      .set_intermediate_data_type(fe::DataType_t::FLOAT)
      .set_compute_data_type(fe::DataType_t::FLOAT);
  const auto mixed_input = mixed_storage_graph.tensor(
      fe::graph::Tensor_attributes()
          .set_uid(101)
          .set_dim({4})
          .set_stride({1}));
  const auto mixed_intermediate = mixed_storage_graph.pointwise(
      mixed_input,
      fe::graph::Pointwise_attributes()
          .set_mode(fe::PointwiseMode_t::RELU_FWD));
  const auto mixed_output = mixed_storage_graph.pointwise(
      mixed_intermediate,
      fe::graph::Pointwise_attributes()
          .set_mode(fe::PointwiseMode_t::RELU_FWD));
  mixed_output->set_uid(102).set_output(true);
  const fe::error_t mixed_status = mixed_storage_graph.validate();
  if (mixed_status.get_status() != FLAGDNN_STATUS_INVALID_VALUE) {
    return 2;
  }

  return output->get_dim() == input->get_dim() &&
                 abs_output->get_dim() == abs_input->get_dim() &&
                 sigmoid_output->get_dim() == abs_input->get_dim() &&
                 tanh_output->get_dim() == abs_input->get_dim() &&
                 leaky_relu_output->get_dim() == abs_input->get_dim() &&
                 elu_output->get_dim() == abs_input->get_dim() &&
                 gelu_output->get_dim() == abs_input->get_dim() &&
                 gelu_approx_output->get_dim() == abs_input->get_dim() &&
                 softplus_output->get_dim() == abs_input->get_dim() &&
                 swish_output->get_dim() == abs_input->get_dim() &&
                 swish_attributes.get_swish_beta().value_or(0.0F) ==
                     1.25F &&
                 sum->get_dim() == left->get_dim() &&
                 difference->get_dim() == binary_left->get_dim() &&
                 sigmoid_gradient->get_dim() == binary_left->get_dim() &&
                 invalid_sigmoid_backward_rejected &&
                 selected->get_dim() == binary_left->get_dim() &&
                 invalid_binary_select_mask_rejected &&
                 comparison->get_data_type() == fe::DataType_t::BOOLEAN &&
                 logical_output->get_data_type() == fe::DataType_t::BOOLEAN &&
                 reduced->get_dim() ==
                     std::vector<std::int64_t>({2, 1, 4}) &&
                 batchnorm_attributes.get_name() ==
                     "batchnorm_inference" &&
                 batchnorm_attributes.get_compute_data_type() ==
                     fe::DataType_t::FLOAT &&
                 batchnorm_output->get_dim() == batchnorm_x->get_dim() &&
                 batchnorm_output->get_stride() ==
                     batchnorm_x->get_stride() &&
                 batchnorm_validation.is_good() &&
                 batchnorm_training_validation.is_good() &&
                 training_outputs.size() == 5 &&
                 training_outputs[0]->get_dim() == training_x->get_dim() &&
                 training_outputs[1]->get_data_type() ==
                     fe::DataType_t::FLOAT &&
                 epsilon->is_scalar() &&
                 epsilon->get_scalar_type() ==
                     fe::graph::ScalarType::COMPILE_TIME_CONST &&
                 layernorm_validation.is_good() &&
                 layernorm_outputs.size() == 3 &&
                 layernorm_outputs[0]->get_dim() ==
                     layernorm_x->get_dim() &&
                 layernorm_outputs[1]->get_data_type() ==
                     fe::DataType_t::FLOAT &&
                 layernorm_outputs[1]->get_dim() ==
                     std::vector<std::int64_t>({2, 5, 1}) &&
                 layernorm_outputs[2]->get_data_type() ==
                     fe::DataType_t::FLOAT &&
                 layernorm_attributes.get_forward_phase() ==
                     fe::NormFwdPhase_t::TRAINING &&
                 rmsnorm_validation.is_good() &&
                 rmsnorm_outputs.size() == 2 &&
                 rmsnorm_outputs[0]->get_dim() == rmsnorm_x->get_dim() &&
                 rmsnorm_outputs[1]->get_data_type() ==
                     fe::DataType_t::FLOAT &&
                 rmsnorm_outputs[1]->get_dim() ==
                     std::vector<std::int64_t>({2, 5, 1}) &&
                 rmsnorm_attributes.get_forward_phase() ==
                     fe::NormFwdPhase_t::TRAINING &&
                 matmul_output->get_dim() ==
                     std::vector<std::int64_t>({2, 3, 17, 23}) &&
                 reshape_attributes.get_name() == "reshape" &&
                 reshape_attributes.get_compute_data_type() ==
                     fe::DataType_t::FLOAT &&
                 reshape_attributes.get_dim() ==
                     std::vector<std::int64_t>({6, 4}) &&
                 reshape_attributes.get_stride() ==
                     std::vector<std::int64_t>({4, 1}) &&
                 reshape_attributes.get_reshape_mode() ==
                     fe::ReshapeMode_t::LOGICAL &&
                 reshaped->get_dim() ==
                     std::vector<std::int64_t>({6, 4}) &&
                 reshaped->get_stride() ==
                     std::vector<std::int64_t>({4, 1}) &&
                 transpose_attributes.get_permutation() ==
                     std::vector<std::int64_t>({2, 0, 1}) &&
                 transposed->get_dim() ==
                     std::vector<std::int64_t>({4, 2, 3}) &&
                 transposed->get_stride() ==
                     std::vector<std::int64_t>({1, 12, 4}) &&
                 slice_attributes.get_slices() ==
                     std::vector<std::pair<std::int64_t, std::int64_t>>(
                         {{0, 2}, {1, 3}, {0, 4}}) &&
                 slice_attributes.get_strides() ==
                     std::vector<std::int64_t>({1, 1, 2}) &&
                 sliced->get_dim() ==
                     std::vector<std::int64_t>({2, 2, 2}) &&
                 sliced->get_stride() ==
                     std::vector<std::int64_t>({12, 4, 2}) &&
                 invalid_reshape_rejected &&
                 invalid_transpose_rejected &&
                 invalid_slice_rejected &&
                 convolution_output->get_dim() ==
                     std::vector<std::int64_t>({1, 4, 8}) &&
                 dgrad_requires_output_metadata &&
                 dgrad_validation.is_good() &&
                 dgrad_attributes.get_compute_data_type() ==
                     fe::DataType_t::FLOAT &&
                 dgrad_attributes.get_convolution_mode() ==
                     fe::ConvolutionMode_t::CONVOLUTION &&
                 dgrad_attributes.get_groups() == 1 &&
                 dgrad_output->get_dim() ==
                     std::vector<std::int64_t>({1, 2, 8}) &&
                 wgrad_requires_output_metadata &&
                 wgrad_validation.is_good() &&
                 wgrad_attributes.get_pre_padding() ==
                     std::vector<std::int64_t>({1}) &&
                 wgrad_attributes.get_post_padding() ==
                     std::vector<std::int64_t>({1}) &&
                 wgrad_output->get_dim() ==
                     std::vector<std::int64_t>({4, 2, 3})
             ? 0
             : 1;
}
