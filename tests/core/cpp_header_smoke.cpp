#include <flagdnn/flagdnn.hpp>

#include <array>
#include <type_traits>

static_assert(!std::is_copy_constructible_v<flagdnn::Handle>);
static_assert(std::is_move_constructible_v<flagdnn::Handle>);
static_assert(!std::is_copy_constructible_v<flagdnn::TensorDescriptor>);
static_assert(std::is_move_constructible_v<flagdnn::TensorDescriptor>);
static_assert(!std::is_copy_constructible_v<flagdnn::OperationDescriptor>);
static_assert(std::is_move_constructible_v<flagdnn::OperationDescriptor>);
static_assert(!std::is_copy_constructible_v<flagdnn::Graph>);
static_assert(std::is_move_constructible_v<flagdnn::Graph>);
static_assert(!std::is_copy_constructible_v<flagdnn::Executable>);
static_assert(std::is_move_constructible_v<flagdnn::Executable>);
static_assert(FLAGDNN_OPERATION_REDUCTION ==
              FLAGDNN_OPERATION_REDUCTION_SUM);
static_assert(FLAGDNN_OPERATION_CONVOLUTION_FPROP ==
              FLAGDNN_OPERATION_CONV2D_FPROP);
static_assert(FLAGDNN_REDUCTION_ADD == FLAGDNN_REDUCTION_SUM);
static_assert(FLAGDNN_OPERATION_POINTWISE !=
              FLAGDNN_OPERATION_CONVOLUTION_FPROP);
static_assert(FLAGDNN_POINTWISE_ABS != FLAGDNN_POINTWISE_NOT_SET);
static_assert(FLAGDNN_POINTWISE_SUB == 17);
static_assert(FLAGDNN_POINTWISE_POW == 23);
static_assert(FLAGDNN_DATA_BOOLEAN == 3);
static_assert(FLAGDNN_POINTWISE_LOGICAL_NOT == 24);
static_assert(FLAGDNN_POINTWISE_CMP_EQ == 25);
static_assert(FLAGDNN_POINTWISE_LOGICAL_OR == 32);
static_assert(FLAGDNN_POINTWISE_SIGMOID_FWD == 33);
static_assert(FLAGDNN_POINTWISE_TANH_FWD == 34);
static_assert(FLAGDNN_POINTWISE_ELU_FWD == 35);
static_assert(FLAGDNN_POINTWISE_GELU_FWD == 36);
static_assert(FLAGDNN_POINTWISE_SOFTPLUS_FWD == 37);
static_assert(FLAGDNN_POINTWISE_SWISH_FWD == 38);
static_assert(FLAGDNN_POINTWISE_GELU_APPROX_TANH_FWD == 39);
static_assert(FLAGDNN_POINTWISE_SIGMOID_BWD == 40);
static_assert(FLAGDNN_POINTWISE_BINARY_SELECT == 41);
static_assert(FLAGDNN_POINTWISE_ATTRIBUTE_FLAGS_ALL == 63U);
static_assert(FLAGDNN_BUILD_OPTION_AUTOTUNE == 4U);
static_assert(FLAGDNN_BUILD_OPTION_FLAGS_ALL == 7U);

int main() {
  const flagdnnPointwiseAttributes_t attributes =
      FLAGDNN_POINTWISE_ATTRIBUTES_INITIALIZER;
  const std::array<std::int64_t, 1> dimensions = {8};
  const std::array<std::int64_t, 1> strides = {1};
  flagdnn::TensorDescriptor tensor(
      91, FLAGDNN_DATA_FLOAT32, dimensions, strides);
  tensor.set_virtual();
  const std::array<std::int64_t, 2> a_dimensions = {2, 3};
  const std::array<std::int64_t, 2> a_strides = {3, 1};
  const std::array<std::int64_t, 2> b_dimensions = {3, 4};
  const std::array<std::int64_t, 2> b_strides = {4, 1};
  const std::array<std::int64_t, 2> c_dimensions = {2, 4};
  const std::array<std::int64_t, 2> c_strides = {4, 1};
  flagdnn::TensorDescriptor a(92, FLAGDNN_DATA_FLOAT32,
                              a_dimensions, a_strides);
  flagdnn::TensorDescriptor b(93, FLAGDNN_DATA_FLOAT32,
                              b_dimensions, b_strides);
  flagdnn::TensorDescriptor c(94, FLAGDNN_DATA_FLOAT32,
                              c_dimensions, c_strides);
  flagdnn::Graph matmul_graph;
  matmul_graph.matmul(a, b, c);
  matmul_graph.validate();
  return flagdnnGetVersion() == 100U &&
                 attributes.struct_size == sizeof(attributes) &&
                 attributes.version ==
                     FLAGDNN_POINTWISE_ATTRIBUTES_VERSION &&
                 attributes.flags == 0U && attributes.swish_beta == 1.0 &&
                 tensor.is_virtual()
             ? 0
             : 1;
}
