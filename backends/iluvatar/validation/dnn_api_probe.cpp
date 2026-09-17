// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <dlfcn.h>
#include <iostream>
#include <source_location>
#include <vector>
namespace {
void check(cudnnStatus_t status,
           std::source_location location = std::source_location::current()) {
  if (status != CUDNN_STATUS_SUCCESS)
    throw std::runtime_error(std::string(cudnnGetErrorString(status)) + " at " +
                             std::to_string(location.line()));
}
void check(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
}
void probe_int32(cudnnHandle_t handle) {
  cudnnTensorDescriptor_t input, output;
  check(cudnnCreateTensorDescriptor(&input));
  check(cudnnCreateTensorDescriptor(&output));
  check(cudnnSetTensor4dDescriptor(input, CUDNN_TENSOR_NCHW, CUDNN_DATA_INT32,
                                   1, 1, 1, 4));
  check(cudnnSetTensor4dDescriptor(output, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                   1, 1, 1, 1));
  void *x, *b, *y;
  check(cudaMalloc(&x, 16));
  check(cudaMalloc(&b, 16));
  check(cudaMalloc(&y, 16));
  const std::int32_t values[4] = {16777217, 3, -5, 2}, other[4] = {1, 2, 4, 7};
  check(cudaMemcpy(x, values, 16, cudaMemcpyHostToDevice));
  check(cudaMemcpy(b, other, 16, cudaMemcpyHostToDevice));
  const float one = 1, zero = 0;
  for (auto mode : {CUDNN_OP_TENSOR_ADD, CUDNN_OP_TENSOR_MUL,
                    CUDNN_OP_TENSOR_MIN, CUDNN_OP_TENSOR_MAX}) {
    cudnnOpTensorDescriptor_t op;
    check(cudnnCreateOpTensorDescriptor(&op));
    const auto set = cudnnSetOpTensorDescriptor(op, mode, CUDNN_DATA_FLOAT,
                                                CUDNN_PROPAGATE_NAN);
    check(cudaMemset(y, 0xa5, 16));
    const auto status = set == CUDNN_STATUS_SUCCESS
                            ? cudnnOpTensor(handle, op, &one, input, x, &one,
                                            input, b, &zero, input, y)
                            : set;
    check(cudaDeviceSynchronize());
    std::int32_t result[4];
    check(cudaMemcpy(result, y, 16, cudaMemcpyDeviceToHost));
    const auto integer_compute = cudnnSetOpTensorDescriptor(
        op, mode, CUDNN_DATA_INT32, CUDNN_PROPAGATE_NAN);
    // Some CoreX unsupported-type diagnostics go to stdout without a newline.
    std::cout << "\n{\"kind\":\"int32_pointwise\",\"mode\":" << mode
              << ",\"set\":" << set << ",\"execute\":" << status
              << ",\"first\":" << result[0]
              << ",\"integer_compute_descriptor_status\":" << integer_compute
              << "}" << std::endl;
    check(cudnnDestroyOpTensorDescriptor(op));
  }
  cudnnReduceTensorDescriptor_t reduction;
  check(cudnnCreateReduceTensorDescriptor(&reduction));
  check(cudnnSetReduceTensorDescriptor(
      reduction, CUDNN_REDUCE_TENSOR_ADD, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN,
      CUDNN_REDUCE_TENSOR_NO_INDICES, CUDNN_32BIT_INDICES));
  std::size_t bytes = 0;
  check(
      cudnnGetReductionWorkspaceSize(handle, reduction, input, output, &bytes));
  void *workspace;
  check(cudaMalloc(&workspace, bytes ? bytes : 1));
  check(cudaMemset(y, 0xa5, 16));
  const auto status =
      cudnnReduceTensor(handle, reduction, nullptr, 0, workspace, bytes, &one,
                        input, x, &zero, output, y);
  check(cudaDeviceSynchronize());
  float result;
  check(cudaMemcpy(&result, y, sizeof(result), cudaMemcpyDeviceToHost));
  std::uint32_t bits;
  std::memcpy(&bits, &result, sizeof(bits));
  std::cout << "{\"kind\":\"int32_reduction\",\"execute\":" << status
            << ",\"output_bits\":" << bits
            << ",\"finite\":" << (std::isfinite(result) ? "true" : "false")
            << "}" << std::endl;
  for (void *p : {workspace, x, b, y})
    check(cudaFree(p));
  check(cudnnDestroyReduceTensorDescriptor(reduction));
  check(cudnnDestroyTensorDescriptor(input));
  check(cudnnDestroyTensorDescriptor(output));
}

void probe_rmsnorm(cudnnHandle_t handle) {
  for (auto type : {CUDNN_DATA_FLOAT, CUDNN_DATA_HALF, CUDNN_DATA_BFLOAT16}) {
    cudnnTensorDescriptor_t x, gamma, stats;
    check(cudnnCreateTensorDescriptor(&x));
    check(cudnnCreateTensorDescriptor(&gamma));
    check(cudnnCreateTensorDescriptor(&stats));
    check(cudnnSetTensor4dDescriptor(x, CUDNN_TENSOR_NCHW, type, 1, 1, 10, 17));
    check(cudnnSetTensor4dDescriptor(gamma, CUDNN_TENSOR_NCHW, type, 1, 1, 1,
                                     17));
    check(cudnnSetTensor4dDescriptor(stats, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                     1, 1, 10, 1));
    void *xp, *gp, *yp, *sp;
    check(cudaMalloc(&xp, 680));
    check(cudaMalloc(&gp, 68));
    check(cudaMalloc(&yp, 680));
    check(cudaMalloc(&sp, 40));
    std::vector<float> f(170, .5f), fg(17, 1), inverse(10);
    std::vector<std::uint16_t> low(170,
                                   type == CUDNN_DATA_HALF ? 0x3800 : 0x3f00),
        lowg(17, type == CUDNN_DATA_HALF ? 0x3c00 : 0x3f80);
    const bool fp32 = type == CUDNN_DATA_FLOAT;
    check(cudaMemcpy(xp, fp32 ? static_cast<void *>(f.data()) : low.data(),
                     fp32 ? 680 : 340, cudaMemcpyHostToDevice));
    check(cudaMemcpy(gp, fp32 ? static_cast<void *>(fg.data()) : lowg.data(),
                     fp32 ? 68 : 34, cudaMemcpyHostToDevice));
    check(cudaMemset(yp, 0xa5, 680));
    check(cudaMemset(sp, 0, 40));
    auto status = cudnnRmsNormalizationForward(handle, x, xp, gamma, gp, .001f,
                                               x, yp, stats, sp);
    check(cudaDeviceSynchronize());
    check(cudaMemcpy(inverse.data(), sp, 40, cudaMemcpyDeviceToHost));
    std::vector<std::uint16_t> out(170);
    check(cudaMemcpy(out.data(), yp, 340, cudaMemcpyDeviceToHost));
    std::cout << "{\"kind\":\"rmsnorm\",\"dtype\":" << type
              << ",\"execute\":" << status
              << ",\"inverse_first\":" << inverse.front()
              << ",\"inverse_last\":" << inverse.back()
              << ",\"output_first_bits\":" << out.front()
              << ",\"output_last_bits\":" << out.back() << "}" << std::endl;
    for (void *p : {xp, gp, yp, sp})
      check(cudaFree(p));
    check(cudnnDestroyTensorDescriptor(x));
    check(cudnnDestroyTensorDescriptor(gamma));
    check(cudnnDestroyTensorDescriptor(stats));
  }
}
void probe_bn_finalize(cudnnHandle_t handle) {
  for (auto type : {CUDNN_DATA_FLOAT}) {
    constexpr auto operation = CUDNN_FUSED_BN_FINALIZE_STATISTICS_TRAINING;
    cudnnFusedOpsPlan_t plan;
    cudnnFusedOpsConstParamPack_t constants;
    cudnnFusedOpsVariantParamPack_t variants;
    check(cudnnCreateFusedOpsPlan(&plan, operation));
    check(cudnnCreateFusedOpsConstParamPack(&constants, operation));
    check(cudnnCreateFusedOpsVariantParamPack(&variants, operation));
    cudnnTensorDescriptor_t stats, parameters;
    check(cudnnCreateTensorDescriptor(&stats));
    check(cudnnCreateTensorDescriptor(&parameters));
    check(cudnnSetTensor4dDescriptor(stats, CUDNN_TENSOR_NCHW, type, 1, 16, 1,
                                     1));
    check(cudnnSetTensor4dDescriptor(parameters, CUDNN_TENSOR_NCHW,
                                     CUDNN_DATA_FLOAT, 1, 16, 1, 1));
    for (auto label :
         {CUDNN_PARAM_YSTATS_DESC, CUDNN_PARAM_BN_SCALEBIAS_MEANVAR_DESC,
          CUDNN_PARAM_BN_EQSCALEBIAS_DESC})
      check(cudnnSetFusedOpsConstParamPackAttribute(
          constants, label,
          label == CUDNN_PARAM_YSTATS_DESC ? stats : parameters));
    auto mode = CUDNN_BATCHNORM_SPATIAL;
    check(cudnnSetFusedOpsConstParamPackAttribute(constants,
                                                  CUDNN_PARAM_BN_MODE, &mode));
    auto aligned = CUDNN_PTR_16B_ALIGNED;
    for (auto label :
         {CUDNN_PARAM_BN_EQSCALE_PLACEHOLDER, CUDNN_PARAM_BN_EQBIAS_PLACEHOLDER,
          CUDNN_PARAM_YSUM_PLACEHOLDER, CUDNN_PARAM_YSQSUM_PLACEHOLDER,
          CUDNN_PARAM_BN_SCALE_PLACEHOLDER, CUDNN_PARAM_BN_BIAS_PLACEHOLDER,
          CUDNN_PARAM_BN_SAVED_MEAN_PLACEHOLDER,
          CUDNN_PARAM_BN_SAVED_INVSTD_PLACEHOLDER,
          CUDNN_PARAM_BN_RUNNING_MEAN_PLACEHOLDER,
          CUDNN_PARAM_BN_RUNNING_VAR_PLACEHOLDER})
      check(
          cudnnSetFusedOpsConstParamPackAttribute(constants, label, &aligned));
    std::size_t workspace_bytes = 0;
    check(cudnnMakeFusedOpsPlan(handle, plan, constants, &workspace_bytes));
    void *workspace = nullptr;
    check(cudaMalloc(&workspace, workspace_bytes ? workspace_bytes : 1));
    std::vector<void *> pointers;
    std::vector<int> get_status;
    int i = 0;
    for (auto label :
         {CUDNN_PTR_YSUM, CUDNN_PTR_YSQSUM, CUDNN_PTR_BN_SCALE,
          CUDNN_PTR_BN_BIAS, CUDNN_PTR_BN_EQSCALE, CUDNN_PTR_BN_EQBIAS,
          CUDNN_PTR_BN_SAVED_MEAN, CUDNN_PTR_BN_SAVED_INVSTD,
          CUDNN_PTR_BN_RUNNING_MEAN, CUDNN_PTR_BN_RUNNING_VAR}) {
      void *p;
      check(cudaMalloc(&p, 128));
      pointers.push_back(p);
      const float value = i == 0             ? 16
                          : i == 1           ? 32
                          : i == 2           ? 1
                          : i == 3 || i >= 8 ? 0
                                             : -64;
      std::vector<float> f(16, value);
      std::vector<double> d(16, value);
      const bool wide = type == CUDNN_DATA_DOUBLE && i < 2;
      check(cudaMemcpy(p, wide ? static_cast<void *>(d.data()) : f.data(),
                       wide ? 128 : 64, cudaMemcpyHostToDevice));
      check(cudnnSetFusedOpsVariantParamPackAttribute(variants, label, p));
      void *readback = nullptr;
      get_status.push_back(cudnnGetFusedOpsVariantParamPackAttribute(
          variants, label, &readback));
      ++i;
    }
    std::int64_t count = 16;
    double epsilon = 1e-5, momentum = .1;
    check(cudnnSetFusedOpsVariantParamPackAttribute(
        variants, CUDNN_PTR_WORKSPACE, workspace));
    check(cudnnSetFusedOpsVariantParamPackAttribute(
        variants, CUDNN_SCALAR_SIZE_T_WORKSPACE_SIZE_IN_BYTES,
        &workspace_bytes));
    check(cudnnSetFusedOpsVariantParamPackAttribute(
        variants, CUDNN_SCALAR_INT64_T_BN_ACCUMULATION_COUNT, &count));
    check(cudnnSetFusedOpsVariantParamPackAttribute(
        variants, CUDNN_SCALAR_DOUBLE_BN_EXP_AVG_FACTOR, &momentum));
    check(cudnnSetFusedOpsVariantParamPackAttribute(
        variants, CUDNN_SCALAR_DOUBLE_BN_EPSILON, &epsilon));
    auto status = cudnnFusedOpsExecute(handle, plan, variants);
    check(cudaDeviceSynchronize());
    bool unchanged = true;
    for (int index = 4; index < 10; ++index) {
      std::vector<float> output(16);
      check(cudaMemcpy(output.data(), pointers[index], 64,
                       cudaMemcpyDeviceToHost));
      for (float value : output)
        unchanged &= value == (index < 8 ? -64 : 0);
    }
    std::cout << "{\"kind\":\"bn_finalize\",\"stats_dtype\":" << type
              << ",\"execute\":" << status
              << ",\"outputs_unchanged\":" << (unchanged ? "true" : "false")
              << ",\"get_pointer_status\":[";
    for (std::size_t index = 0; index < get_status.size(); ++index)
      std::cout << (index ? "," : "") << get_status[index];
    std::cout << "]}" << std::endl;
    for (void *p : pointers)
      check(cudaFree(p));
    if (workspace)
      check(cudaFree(workspace));
    check(cudnnDestroyFusedOpsVariantParamPack(variants));
    check(cudnnDestroyFusedOpsConstParamPack(constants));
    check(cudnnDestroyFusedOpsPlan(plan));
    check(cudnnDestroyTensorDescriptor(stats));
    check(cudnnDestroyTensorDescriptor(parameters));
  }
}
} // namespace

int main() {
  if (cudaSetDevice(0) != cudaSuccess)
    return 1;
  std::cout << "{\"kind\":\"version\",\"header\":" << CUDNN_VERSION
            << ",\"runtime\":" << cudnnGetVersion() << "}" << std::endl;
  cudnnHandle_t h = nullptr;
  if (cudnnCreate(&h) != CUDNN_STATUS_SUCCESS)
    return 2;
  for (const char *s :
       {"cudnnBackendCreateDescriptor", "cudnnFlashAttnForward",
        "cudnnFlashAttnBackward", "cudnnBatchNormalizationBackward",
        "cudnnPoolingForward", "cudnnSpatialTfSamplerForward",
        "cudnnTransformTensor", "cudnnReduceTensor", "cudnnActivationBackward",
        "cudnnRmsNormalizationForward", "cudnnFusedOpsExecute",
        "cudnnCausalConv1dForward", "cudnnRmsNormalizationBackward"})
    std::cout << "{\"kind\":\"symbol\",\"name\":\"" << s << "\",\"available\":"
              << (dlsym(RTLD_DEFAULT, s) ? "true" : "false") << "}"
              << std::endl;
  for (auto type : {CUDNN_DATA_FLOAT, CUDNN_DATA_HALF, CUDNN_DATA_BFLOAT16}) {
    cudnnTensorDescriptor_t d = nullptr;
    cudnnCreateTensorDescriptor(&d);
    if (cudnnSetTensor4dDescriptor(d, CUDNN_TENSOR_NCHW, type, 1, 1, 1, 16) !=
        CUDNN_STATUS_SUCCESS)
      return 3;
    void *x = nullptr, *y = nullptr, *dy = nullptr, *dx = nullptr;
    for (void **p : {&x, &y, &dy, &dx})
      if (cudaMalloc(p, 64) != cudaSuccess)
        return 4;
    std::vector<float> f(16, 0.5f);
    std::vector<std::uint16_t> low(16,
                                   type == CUDNN_DATA_HALF ? 0x3800 : 0x3f00);
    const void *data = type == CUDNN_DATA_FLOAT
                           ? static_cast<void *>(f.data())
                           : static_cast<void *>(low.data());
    auto bytes = type == CUDNN_DATA_FLOAT ? 64 : 32;
    cudaMemcpy(x, data, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dy, data, bytes, cudaMemcpyHostToDevice);
    for (int mode = 0; mode <= 8; ++mode) {
      cudnnActivationDescriptor_t a = nullptr;
      cudnnCreateActivationDescriptor(&a);
      auto set = cudnnSetActivationDescriptor(
          a, static_cast<cudnnActivationMode_t>(mode), CUDNN_PROPAGATE_NAN,
          mode == CUDNN_ACTIVATION_CLIPPED_RELU ||
                  mode == CUDNN_ACTIVATION_ELU || mode == CUDNN_ACTIVATION_SWISH
              ? 1
              : 0);
      const float one = 1, zero = 0;
      auto fw = set == CUDNN_STATUS_SUCCESS
                    ? cudnnActivationForward(h, a, &one, d, x, &zero, d, y)
                    : set;
      cudaDeviceSynchronize();
      if (fw != CUDNN_STATUS_SUCCESS)
        cudaMemcpy(y, data, bytes, cudaMemcpyHostToDevice);
      auto bw = set == CUDNN_STATUS_SUCCESS
                    ? cudnnActivationBackward(h, a, &one, d, y, d, dy, d, x,
                                              &zero, d, dx)
                    : set;
      cudaDeviceSynchronize();
      std::cout << "{\"kind\":\"activation\",\"dtype\":" << type
                << ",\"mode\":" << mode << ",\"set\":" << set
                << ",\"forward\":" << fw << ",\"backward\":" << bw << "}"
                << std::endl;
      cudnnDestroyActivationDescriptor(a);
    }
    for (void *p : {x, y, dy, dx})
      cudaFree(p);
    cudnnDestroyTensorDescriptor(d);
  }
  for (auto compute : {CUDNN_DATA_FLOAT, CUDNN_DATA_INT8}) {
    cudnnTensorDescriptor_t d = nullptr;
    cudnnCreateTensorDescriptor(&d);
    cudnnSetTensor4dDescriptor(d, CUDNN_TENSOR_NCHW, CUDNN_DATA_INT8, 1, 1, 1,
                               16);
    cudnnOpTensorDescriptor_t op = nullptr;
    cudnnCreateOpTensorDescriptor(&op);
    auto set = cudnnSetOpTensorDescriptor(op, CUDNN_OP_TENSOR_NOT, compute,
                                          CUDNN_PROPAGATE_NAN);
    void *x = nullptr, *y = nullptr;
    cudaMalloc(&x, 16);
    cudaMalloc(&y, 16);
    std::vector<unsigned char> input(16), out(16);
    for (int i = 0; i < 16; ++i)
      input[i] = i % 2;
    cudaMemcpy(x, input.data(), 16, cudaMemcpyHostToDevice);
    const float one = 1, zero = 0;
    auto status =
        set == CUDNN_STATUS_SUCCESS
            ? cudnnOpTensor(h, op, &one, d, x, &one, d, x, &zero, d, y)
            : set;
    cudaDeviceSynchronize();
    cudaMemcpy(out.data(), y, 16, cudaMemcpyDeviceToHost);
    std::cout << "{\"kind\":\"logical_not\",\"compute\":" << compute
              << ",\"set\":" << set << ",\"execute\":" << status
              << ",\"output0\":" << int(out[0])
              << ",\"output1\":" << int(out[1]) << "}" << std::endl;
    cudaFree(x);
    cudaFree(y);
    cudnnDestroyOpTensorDescriptor(op);
    cudnnDestroyTensorDescriptor(d);
  }
  for (auto source : {CUDNN_DATA_FLOAT, CUDNN_DATA_HALF, CUDNN_DATA_BFLOAT16,
                      CUDNN_DATA_INT32, CUDNN_DATA_INT8}) {
    for (auto dest : {source, CUDNN_DATA_FLOAT}) {
      cudnnTensorDescriptor_t xdesc = nullptr, ydesc = nullptr;
      cudnnCreateTensorDescriptor(&xdesc);
      cudnnCreateTensorDescriptor(&ydesc);
      cudnnSetTensor4dDescriptor(xdesc, CUDNN_TENSOR_NCHW, source, 1, 1, 1, 16);
      cudnnSetTensor4dDescriptor(ydesc, CUDNN_TENSOR_NCHW, dest, 1, 1, 1, 16);
      void *x = nullptr, *y = nullptr;
      cudaMalloc(&x, 64);
      cudaMalloc(&y, 64);
      std::vector<unsigned char> input(64), output(64);
      for (int i = 0; i < 16; ++i) {
        if (source == CUDNN_DATA_FLOAT) {
          const float v = 0.5f;
          std::memcpy(input.data() + 4 * i, &v, 4);
        } else if (source == CUDNN_DATA_INT32) {
          const std::int32_t v = 16777217;
          std::memcpy(input.data() + 4 * i, &v, 4);
        } else if (source == CUDNN_DATA_INT8)
          input[i] = 1;
        else {
          const std::uint16_t v = source == CUDNN_DATA_HALF ? 0x3800 : 0x3f00;
          std::memcpy(input.data() + 2 * i, &v, 2);
        }
      }
      cudaMemcpy(x, input.data(), 64, cudaMemcpyHostToDevice);
      cudaMemset(y, 0xa5, 64);
      const float one = 1, zero = 0;
      const auto status =
          cudnnTransformTensor(h, &one, xdesc, x, &zero, ydesc, y);
      cudaDeviceSynchronize();
      cudaMemcpy(output.data(), y, 64, cudaMemcpyDeviceToHost);
      std::uint32_t actual = 0, expected = 0;
      const int width =
          dest == CUDNN_DATA_INT8                                    ? 1
          : (dest == CUDNN_DATA_HALF || dest == CUDNN_DATA_BFLOAT16) ? 2
                                                                     : 4;
      std::memcpy(&actual, output.data(), width);
      if (dest == source)
        std::memcpy(&expected, input.data(), width);
      else {
        const float v = source == CUDNN_DATA_INT32  ? float(16777217)
                        : source == CUDNN_DATA_INT8 ? 1.0f
                                                    : 0.5f;
        std::memcpy(&expected, &v, 4);
      }
      std::cout << "{\"kind\":\"transform\",\"source\":" << source
                << ",\"destination\":" << dest << ",\"execute\":" << status
                << ",\"actual\":" << actual << ",\"expected\":" << expected
                << ",\"correct\":" << (actual == expected ? "true" : "false")
                << "}" << std::endl;
      cudaFree(x);
      cudaFree(y);
      cudnnDestroyTensorDescriptor(xdesc);
      cudnnDestroyTensorDescriptor(ydesc);
    }
  }
  probe_int32(h);
  probe_rmsnorm(h);
  probe_bn_finalize(h);
  cudnnDestroy(h);
}
