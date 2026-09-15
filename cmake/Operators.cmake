# Public C++ Graph operators covered by native performance tests. Adding a
# name here requires benchmark/test_<name>.cpp and a platform benchmark
# provider. Functional coverage is tracked separately because a correct
# operator may intentionally precede performance qualification.
set(FLAGDNN_ACTIVATION_BACKWARD_EXTENSIONS
  relu_backward
  tanh_backward
  elu_backward
  gelu_backward
  softplus_backward
  swish_backward
  gelu_approx_tanh_backward
  leaky_relu_backward)

set(FLAGDNN_BENCHMARK_OPERATORS
  abs
  add
  add_square
  batchnorm
  batchnorm_inference
  binary_select
  ceil
  cmp_eq
  cmp_ge
  cmp_gt
  cmp_le
  cmp_lt
  cmp_neq
  conv_bias_relu
  conv_dgrad
  conv_fprop
  conv_wgrad
  cos
  div
  elu
  erf
  exp
  floor
  gelu
  gelu_approx_tanh
  identity
  layernorm
  leaky_relu
  log
  logical_and
  logical_not
  logical_or
  matmul
  max
  min
  mod
  mul
  neg
  pow
  reciprocal
  reduction
  relu
  reshape
  rmsnorm
  rsqrt
  scale
  sigmoid
  sigmoid_backward
  ${FLAGDNN_ACTIVATION_BACKWARD_EXTENSIONS}
  sin
  slice
  softplus
  sqrt
  sub
  swish
  tan
  tanh
  transpose)

# Public C++ Graph operators covered by native functional tests. Adding a name
# here is a correctness contract: tests/test_<name>.cpp must exercise the
# public Graph API and every enabled platform must provide a real reference
# adapter or an explicit capability gate.
set(FLAGDNN_EXTENDED_OPERATORS
  moe_grouped_matmul
  moe_grouped_matmul_bwd
  matmul_fp8
  causal_conv1d
  resample
  rng
  rope
  rope_backward
  bn_finalize
  instancenorm
  adalayernorm
  instancenorm_backward
  adalayernorm_backward
  layernorm_backward
  rmsnorm_backward
  batchnorm_backward
  concatenate
  gen_index
  genstats)

set(FLAGDNN_FUNCTIONAL_OPERATORS
  ${FLAGDNN_EXTENDED_OPERATORS}
  ${FLAGDNN_BENCHMARK_OPERATORS}
  sdpa
  sdpa_backward
  sdpa_fp8
  sdpa_fp8_backward)
