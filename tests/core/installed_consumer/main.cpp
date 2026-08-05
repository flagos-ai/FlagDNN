#include <flagdnn/flagdnn.hpp>
#include <flagdnn_frontend.h>

#include <array>
#include <cstdint>
#include <type_traits>

static_assert(std::is_move_constructible_v<flagdnn::Graph>);
static_assert(!std::is_copy_constructible_v<flagdnn::Graph>);
static_assert(std::is_move_constructible_v<flagdnn::Executable>);
static_assert(!std::is_copy_constructible_v<flagdnn::Executable>);
static_assert(
    std::is_move_constructible_v<flagdnn_frontend::graph::Graph>);
static_assert(
    !std::is_copy_constructible_v<flagdnn_frontend::graph::Graph>);
static_assert(FLAGDNN_POINTWISE_SIGMOID_FWD == 33);
static_assert(FLAGDNN_POINTWISE_TANH_FWD == 34);
static_assert(FLAGDNN_POINTWISE_ELU_FWD == 35);
static_assert(FLAGDNN_POINTWISE_GELU_FWD == 36);
static_assert(FLAGDNN_POINTWISE_SOFTPLUS_FWD == 37);
static_assert(FLAGDNN_POINTWISE_SWISH_FWD == 38);
static_assert(FLAGDNN_POINTWISE_GELU_APPROX_TANH_FWD == 39);
static_assert(FLAGDNN_POINTWISE_SIGMOID_BWD == 40);
static_assert(FLAGDNN_POINTWISE_BINARY_SELECT == 41);

int main() {
  constexpr std::array<std::int64_t, 3> left_dimensions = {2, 3, 4};
  constexpr std::array<std::int64_t, 3> left_strides = {1, 8, 2};
  constexpr std::array<std::int64_t, 2> right_dimensions = {1, 4};
  constexpr std::array<std::int64_t, 2> right_strides = {4, 1};
  flagdnn::TensorDescriptor left(
      1, FLAGDNN_DATA_FLOAT32, left_dimensions, left_strides);
  flagdnn::TensorDescriptor right(
      2, FLAGDNN_DATA_FLOAT32, right_dimensions, right_strides);
  flagdnn::TensorDescriptor select_b(
      4, FLAGDNN_DATA_FLOAT32, left_dimensions, left_strides);
  flagdnn::TensorDescriptor predicate(
      5, FLAGDNN_DATA_BOOLEAN, left_dimensions, left_strides);
  flagdnn::TensorDescriptor output(
      3, FLAGDNN_DATA_FLOAT32, left_dimensions, left_strides);
  flagdnn::Graph graph;
  graph.pointwise(left, right, FLAGDNN_POINTWISE_ADD, output, -0.75);
  graph.finalize();

  flagdnn::TensorDescriptor intermediate(
      22, FLAGDNN_DATA_FLOAT32, left_dimensions, left_strides);
  intermediate.set_virtual();
  flagdnn::TensorDescriptor terminal(
      23, FLAGDNN_DATA_FLOAT32, left_dimensions, left_strides);
  flagdnn::Graph multi_operation_graph;
  multi_operation_graph.relu(left, intermediate);
  multi_operation_graph.pointwise(intermediate, right, FLAGDNN_POINTWISE_ADD, terminal);
  multi_operation_graph.finalize();

  flagdnn::Graph binary_graph;
  binary_graph.pointwise(
      left, right, FLAGDNN_POINTWISE_MUL, output);
  binary_graph.finalize();

  flagdnn::Graph binary_select_graph;
  binary_select_graph.pointwise(
      left,
      select_b,
      predicate,
      FLAGDNN_POINTWISE_BINARY_SELECT,
      output);
  binary_select_graph.finalize();

  flagdnn::Graph sigmoid_graph;
  sigmoid_graph.pointwise(
      left, FLAGDNN_POINTWISE_SIGMOID_FWD, output);
  sigmoid_graph.finalize();

  flagdnn::Graph tanh_graph;
  tanh_graph.pointwise(left, FLAGDNN_POINTWISE_TANH_FWD, output);
  tanh_graph.finalize();

  flagdnnPointwiseAttributes_t swish_attributes =
      FLAGDNN_POINTWISE_ATTRIBUTES_INITIALIZER;
  swish_attributes.flags = FLAGDNN_POINTWISE_ATTRIBUTE_SWISH_BETA;
  swish_attributes.swish_beta = 1.25;
  flagdnn::Graph swish_graph;
  swish_graph.pointwise(
      left, FLAGDNN_POINTWISE_SWISH_FWD, output, swish_attributes);
  swish_graph.finalize();

  flagdnn::TensorDescriptor boolean_output(
      16, FLAGDNN_DATA_BOOLEAN, left_dimensions, left_strides);
  flagdnn::Graph comparison_graph;
  comparison_graph.pointwise(
      left, right, FLAGDNN_POINTWISE_CMP_LT, boolean_output);
  comparison_graph.finalize();

  flagdnn::TensorDescriptor boolean_input(
      17, FLAGDNN_DATA_BOOLEAN, left_dimensions, left_strides);
  flagdnn::TensorDescriptor logical_output(
      18, FLAGDNN_DATA_BOOLEAN, left_dimensions, left_strides);
  flagdnn::Graph logical_graph;
  logical_graph.pointwise(
      boolean_input, FLAGDNN_POINTWISE_LOGICAL_NOT, logical_output);
  logical_graph.finalize();

  constexpr std::array<std::int64_t, 2> reduction_input_dimensions = {2, 3};
  constexpr std::array<std::int64_t, 2> reduction_input_strides = {3, 1};
  constexpr std::array<std::int64_t, 2> reduction_output_dimensions = {2, 1};
  constexpr std::array<std::int64_t, 2> reduction_output_strides = {1, 1};
  flagdnn::TensorDescriptor reduction_input(4,
                                             FLAGDNN_DATA_FLOAT32,
                                             reduction_input_dimensions,
                                             reduction_input_strides);
  flagdnn::TensorDescriptor reduction_output(5,
                                              FLAGDNN_DATA_FLOAT32,
                                              reduction_output_dimensions,
                                              reduction_output_strides);
  flagdnn::Graph reduction_graph;
  reduction_graph.reduction_avg(
      reduction_input, 1, true, reduction_output);
  reduction_graph.finalize();

  constexpr std::array<std::int64_t, 4> conv_input_dimensions = {
      1, 2, 5, 5};
  constexpr std::array<std::int64_t, 4> conv_input_strides = {
      50, 25, 5, 1};
  constexpr std::array<std::int64_t, 4> conv_filter_dimensions = {
      2, 2, 3, 3};
  constexpr std::array<std::int64_t, 4> conv_filter_strides = {
      18, 9, 3, 1};
  constexpr std::array<std::int64_t, 4> conv_output_dimensions = {
      1, 2, 6, 2};
  constexpr std::array<std::int64_t, 4> conv_output_strides = {
      24, 12, 2, 1};
  constexpr std::array<std::int64_t, 2> pre_padding = {1, 0};
  constexpr std::array<std::int64_t, 2> post_padding = {2, 1};
  constexpr std::array<std::int64_t, 2> conv_stride = {1, 2};
  constexpr std::array<std::int64_t, 2> conv_dilation = {1, 1};
  flagdnn::TensorDescriptor conv_input(6,
                                       FLAGDNN_DATA_FLOAT16,
                                       conv_input_dimensions,
                                       conv_input_strides);
  flagdnn::TensorDescriptor conv_filter(7,
                                        FLAGDNN_DATA_FLOAT16,
                                        conv_filter_dimensions,
                                        conv_filter_strides);
  flagdnn::TensorDescriptor conv_output(8,
                                        FLAGDNN_DATA_FLOAT16,
                                        conv_output_dimensions,
                                        conv_output_strides);
  flagdnn::Graph convolution_graph;
  convolution_graph.conv2d_fprop(conv_input,
                                 conv_filter,
                                 pre_padding,
                                 post_padding,
                                 conv_stride,
                                 conv_dilation,
                                 1,
                                 conv_output);
  convolution_graph.finalize();

  constexpr std::array<std::int64_t, 3> conv1d_input_dimensions = {
      1, 2, 8};
  constexpr std::array<std::int64_t, 3> conv1d_input_strides = {
      16, 8, 1};
  constexpr std::array<std::int64_t, 3> conv1d_filter_dimensions = {
      4, 2, 3};
  constexpr std::array<std::int64_t, 3> conv1d_filter_strides = {
      6, 3, 1};
  constexpr std::array<std::int64_t, 3> conv1d_output_dimensions = {
      1, 4, 8};
  constexpr std::array<std::int64_t, 3> conv1d_output_strides = {
      32, 8, 1};
  constexpr std::array<std::int64_t, 1> conv1d_padding = {1};
  constexpr std::array<std::int64_t, 1> conv1d_stride = {1};
  constexpr std::array<std::int64_t, 1> conv1d_dilation = {1};
  flagdnn::TensorDescriptor conv1d_input(9,
                                         FLAGDNN_DATA_FLOAT32,
                                         conv1d_input_dimensions,
                                         conv1d_input_strides);
  flagdnn::TensorDescriptor conv1d_filter(10,
                                          FLAGDNN_DATA_FLOAT32,
                                          conv1d_filter_dimensions,
                                          conv1d_filter_strides);
  flagdnn::TensorDescriptor conv1d_output(11,
                                          FLAGDNN_DATA_FLOAT32,
                                          conv1d_output_dimensions,
                                          conv1d_output_strides);
  flagdnn::Graph convolution_nd_graph;
  convolution_nd_graph.conv1d_fprop(conv1d_input,
                                    conv1d_filter,
                                    conv1d_padding,
                                    conv1d_stride,
                                    conv1d_dilation,
                                    1,
                                    conv1d_output);
  convolution_nd_graph.finalize();

  namespace fe = flagdnn_frontend;
  fe::graph::Graph frontend_dgrad_graph;
  frontend_dgrad_graph.set_io_data_type(fe::DataType_t::FLOAT)
      .set_compute_data_type(fe::DataType_t::FLOAT);
  const auto frontend_dy = frontend_dgrad_graph.tensor(
      fe::graph::Tensor_attributes()
          .set_uid(50)
          .set_dim({1, 4, 8})
          .set_stride({32, 8, 1}));
  const auto frontend_w = frontend_dgrad_graph.tensor(
      fe::graph::Tensor_attributes()
          .set_uid(51)
          .set_dim({4, 2, 3})
          .set_stride({6, 3, 1}));
  const auto frontend_dx = frontend_dgrad_graph.conv_dgrad(
      frontend_dy,
      frontend_w,
      fe::graph::Conv_dgrad_attributes()
          .set_name("installed_consumer_dgrad")
          .set_compute_data_type(fe::DataType_t::FLOAT)
          .set_padding({1})
          .set_stride({1})
          .set_dilation({1})
          .set_groups(1));
  const bool frontend_dx_requires_metadata =
      frontend_dx->get_dim().empty();
  frontend_dx->set_uid(52)
      .set_dim({1, 2, 8})
      .set_stride({16, 8, 1})
      .set_output(true);

  fe::graph::Graph frontend_wgrad_graph;
  frontend_wgrad_graph.set_io_data_type(fe::DataType_t::FLOAT)
      .set_compute_data_type(fe::DataType_t::FLOAT);
  const auto frontend_wgrad_dy = frontend_wgrad_graph.tensor(
      fe::graph::Tensor_attributes()
          .set_uid(53)
          .set_dim({1, 4, 8})
          .set_stride({32, 8, 1}));
  const auto frontend_x = frontend_wgrad_graph.tensor(
      fe::graph::Tensor_attributes()
          .set_uid(54)
          .set_dim({1, 2, 8})
          .set_stride({16, 8, 1}));
  const auto frontend_dw = frontend_wgrad_graph.conv_wgrad(
      frontend_wgrad_dy,
      frontend_x,
      fe::graph::Conv_wgrad_attributes()
          .set_name("installed_consumer_wgrad")
          .set_compute_data_type(fe::DataType_t::FLOAT)
          .set_padding({1})
          .set_stride({1})
          .set_dilation({1})
          .set_convolution_mode(fe::ConvolutionMode_t::CONVOLUTION)
          .set_groups(1));
  const bool frontend_dw_requires_metadata =
      frontend_dw->get_dim().empty();
  frontend_dw->set_uid(55)
      .set_dim({4, 2, 3})
      .set_stride({6, 3, 1})
      .set_output(true);

  fe::graph::Graph frontend_graph;
  frontend_graph.set_name("installed_consumer_abs")
      .set_io_data_type(fe::DataType_t::FLOAT)
      .set_compute_data_type(fe::DataType_t::FLOAT);
  const auto frontend_input = frontend_graph.tensor(
      fe::graph::Tensor_attributes()
          .set_name("x")
          .set_uid(12)
          .set_dim({2, 3})
          .set_stride({3, 1}));
  const auto frontend_output = frontend_graph.pointwise(
      frontend_input,
      fe::graph::Pointwise_attributes()
          .set_name("abs")
          .set_mode(fe::PointwiseMode_t::ABS));
  frontend_output->set_name("y").set_uid(13).set_output(true);
  const auto frontend_sigmoid = frontend_graph.pointwise(
      frontend_input,
      fe::graph::Pointwise_attributes()
          .set_name("sigmoid")
          .set_mode(fe::PointwiseMode_t::SIGMOID_FWD));
  const auto frontend_tanh = frontend_graph.pointwise(
      frontend_input,
      fe::graph::Pointwise_attributes()
          .set_name("tanh")
          .set_mode(fe::PointwiseMode_t::TANH_FWD));
  fe::graph::Pointwise_attributes frontend_swish_attributes;
  frontend_swish_attributes.set_name("swish")
      .set_mode(fe::PointwiseMode_t::SWISH_FWD)
      .set_swish_beta(1.25F);
  const auto frontend_swish = frontend_graph.pointwise(
      frontend_input, frontend_swish_attributes);
  const auto frontend_elu = frontend_graph.pointwise(
      frontend_input,
      fe::graph::Pointwise_attributes()
          .set_name("elu")
          .set_mode(fe::PointwiseMode_t::ELU_FWD)
          .set_elu_alpha(1.0F));
  const auto frontend_right = frontend_graph.tensor(
      fe::graph::Tensor_attributes()
          .set_name("right")
          .set_uid(14)
          .set_dim({1, 3})
          .set_stride({3, 1}));
  const auto frontend_difference = frontend_graph.pointwise(
      frontend_input,
      frontend_right,
      fe::graph::Pointwise_attributes()
          .set_name("sub")
          .set_mode(fe::PointwiseMode_t::SUB));
  frontend_difference->set_name("difference").set_uid(15).set_output(true);

  fe::graph::Graph frontend_layout_graph;
  frontend_layout_graph.set_io_data_type(fe::DataType_t::HALF)
      .set_compute_data_type(fe::DataType_t::FLOAT);
  const auto frontend_layout_input = frontend_layout_graph.tensor(
      fe::graph::Tensor_attributes()
          .set_uid(30)
          .set_dim({2, 3, 4})
          .set_stride({12, 4, 1}));
  const auto frontend_reshape = frontend_layout_graph.reshape(
      frontend_layout_input,
      fe::graph::Reshape_attributes()
          .set_name("reshape")
          .set_dim({6, 4})
          .set_stride({4, 1})
          .set_reshape_mode(fe::ReshapeMode_t::LOGICAL));
  frontend_reshape->set_uid(31).set_output(true);
  const auto frontend_transpose = frontend_layout_graph.transpose(
      frontend_layout_input,
      fe::graph::Transpose_attributes()
          .set_name("transpose")
          .set_permutation({2, 0, 1}));
  frontend_transpose->set_uid(32).set_output(true);
  const auto frontend_slice = frontend_layout_graph.slice(
      frontend_layout_input,
      fe::graph::Slice_attributes()
          .set_name("slice")
          .set_slices({{0, 2}, {1, 3}, {0, 4}})
          .set_strides({1, 1, 2}));
  frontend_slice->set_uid(33).set_output(true);
  const fe::error_t frontend_layout_status =
      frontend_layout_graph.validate();

  fe::graph::Graph frontend_batchnorm_graph;
  frontend_batchnorm_graph
      .set_name("installed_consumer_batchnorm_inference")
      .set_io_data_type(fe::DataType_t::HALF)
      .set_compute_data_type(fe::DataType_t::FLOAT);
  const auto frontend_batchnorm_x = frontend_batchnorm_graph.tensor(
      fe::graph::Tensor_attributes()
          .set_name("x")
          .set_uid(56)
          .set_dim({2, 8, 4, 4})
          .set_stride({128, 16, 4, 1}));
  const auto make_batchnorm_parameter =
      [&frontend_batchnorm_graph](std::int64_t uid, const char* name) {
        return frontend_batchnorm_graph.tensor(
            fe::graph::Tensor_attributes()
                .set_name(name)
                .set_uid(uid)
                .set_data_type(fe::DataType_t::FLOAT)
                .set_dim({1, 8, 1, 1})
                .set_stride({8, 1, 1, 1}));
      };
  const auto frontend_batchnorm_mean =
      make_batchnorm_parameter(57, "mean");
  const auto frontend_batchnorm_inv_variance =
      make_batchnorm_parameter(58, "inv_variance");
  const auto frontend_batchnorm_scale =
      make_batchnorm_parameter(59, "scale");
  const auto frontend_batchnorm_bias =
      make_batchnorm_parameter(60, "bias");
  const auto frontend_batchnorm_output =
      frontend_batchnorm_graph.batchnorm_inference(
          frontend_batchnorm_x,
          frontend_batchnorm_mean,
          frontend_batchnorm_inv_variance,
          frontend_batchnorm_scale,
          frontend_batchnorm_bias,
          fe::graph::Batchnorm_inference_attributes()
              .set_name("batchnorm_inference")
              .set_compute_data_type(fe::DataType_t::FLOAT));
  frontend_batchnorm_output->set_uid(61).set_output(true);
  const fe::error_t frontend_batchnorm_status =
      frontend_batchnorm_graph.validate();

  fe::graph::Graph frontend_comparison_graph;
  frontend_comparison_graph.set_io_data_type(fe::DataType_t::HALF);
  const auto frontend_comparison_left = frontend_comparison_graph.tensor(
      fe::graph::Tensor_attributes()
          .set_uid(19)
          .set_dim({2, 3})
          .set_stride({3, 1}));
  const auto frontend_comparison_right = frontend_comparison_graph.tensor(
      fe::graph::Tensor_attributes()
          .set_uid(20)
          .set_dim({1, 3})
          .set_stride({3, 1}));
  const auto frontend_comparison = frontend_comparison_graph.pointwise(
      frontend_comparison_left,
      frontend_comparison_right,
      fe::graph::Pointwise_attributes().set_mode(
          fe::PointwiseMode_t::CMP_GE));
  frontend_comparison->set_uid(21).set_output(true);

  return flagdnnGetVersion() == 100U && graph.operation_count() == 1 &&
                 intermediate.is_virtual() &&
                 multi_operation_graph.operation_count() == 2 &&
                 binary_graph.operation_count() == 1 &&
                 sigmoid_graph.operation_count() == 1 &&
                 tanh_graph.operation_count() == 1 &&
                 swish_graph.operation_count() == 1 &&
                 comparison_graph.operation_count() == 1 &&
                 logical_graph.operation_count() == 1 &&
                 reduction_graph.operation_count() == 1 &&
                 convolution_graph.operation_count() == 1 &&
                 convolution_nd_graph.operation_count() == 1 &&
                 frontend_dx_requires_metadata &&
                 frontend_dx->get_dim() ==
                     std::vector<std::int64_t>({1, 2, 8}) &&
                 frontend_dw_requires_metadata &&
                 frontend_dw->get_dim() ==
                     std::vector<std::int64_t>({4, 2, 3}) &&
                 frontend_output->get_dim() == frontend_input->get_dim() &&
                 frontend_sigmoid->get_dim() == frontend_input->get_dim() &&
                 frontend_tanh->get_dim() == frontend_input->get_dim() &&
                 frontend_swish->get_dim() == frontend_input->get_dim() &&
                 frontend_elu->get_dim() == frontend_input->get_dim() &&
                 frontend_swish_attributes.get_swish_beta().value_or(0.0F) ==
                     1.25F &&
                 frontend_difference->get_dim() == frontend_input->get_dim() &&
                 frontend_layout_status.is_good() &&
                 frontend_reshape->get_dim() ==
                     std::vector<std::int64_t>({6, 4}) &&
                 frontend_transpose->get_dim() ==
                     std::vector<std::int64_t>({4, 2, 3}) &&
                 frontend_transpose->get_stride() ==
                     std::vector<std::int64_t>({1, 12, 4}) &&
                 frontend_slice->get_dim() ==
                     std::vector<std::int64_t>({2, 2, 2}) &&
                 frontend_slice->get_stride() ==
                     std::vector<std::int64_t>({12, 4, 2}) &&
                 frontend_batchnorm_status.is_good() &&
                 frontend_batchnorm_output->get_dim() ==
                     frontend_batchnorm_x->get_dim() &&
                 frontend_batchnorm_output->get_stride() ==
                     frontend_batchnorm_x->get_stride() &&
                 frontend_comparison->get_data_type() ==
                        fe::DataType_t::BOOLEAN
             ? 0
             : 1;
}
