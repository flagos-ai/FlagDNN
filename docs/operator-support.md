<!-- 有功能测试通过且有性能用例实际完成计时的算子列为“已支持”；部分 dtype 或 shape 跳过不影响该分类。类型列为功能支持范围。 -->

# Nvidia

## 已支持算子

| 算子 | 输入与输出类型 |
| --- | --- |
| identity、reshape、transpose、slice、concatenate | 保持输入类型与原始位模式；支持 fp32、fp16、bf16、int32、bool、FP8 E4M3/E5M2/E8M0 |
| gen_index | 输出 int32 或 fp32 |
| add、sub、mul、div、pow、max、min、mod、add_square、scale | fp32、fp16、bf16、int32 |
| cmp_eq、cmp_neq、cmp_gt、cmp_ge、cmp_lt、cmp_le | fp32、fp16、bf16、int32 → bool |
| logical_and、logical_or、logical_not | bool |
| binary_select | bool 条件与 fp32、fp16、bf16、int32 数值 |
| abs、neg、sqrt、rsqrt、reciprocal、ceil、floor、exp、log、erf、sin、cos、tan | fp32、fp16、bf16 |
| relu、leaky_relu、elu、gelu、gelu_approx_tanh、sigmoid、swish、tanh、softplus 及对应 backward | fp32、fp16、bf16 |
| conv_fprop、conv_dgrad、conv_wgrad、causal_conv1d | fp32、tf32、fp16、bf16 |
| matmul | fp32、tf32、fp16、bf16、FP8 E4M3/E5M2 |
| matmul_fp8 | FP8 E4M3/E5M2 或 MXFP8；输出 fp32、fp16、bf16 |
| moe_grouped_matmul | fp16、bf16、FP8 E4M3/E5M2；输出 fp32、fp16、bf16 |
| reduction | fp32、fp16、bf16 → 同类型或 fp32；int32 → fp32 |
| resample | fp32、fp16、bf16 |
| batchnorm、batchnorm_backward、batchnorm_inference、layernorm、layernorm_backward、rmsnorm、rmsnorm_backward、instancenorm、instancenorm_backward、adalayernorm、adalayernorm_backward、bn_finalize | 数据 fp32、fp16、bf16；统计量 fp32，仿射参数梯度默认 fp32 |
| conv_bias_relu | fp32、fp16、bf16 |
| sdpa、sdpa_backward | fp32、fp16、bf16 |
| sdpa_fp8、sdpa_fp8_backward | FP8 E4M3/E5M2 |

## 不完全支持算子

| 算子 | 数据类型 | 未注册 NVIDIA 测试的原因（H100、cuDNN 9.24） |
| --- | --- | --- |
| genstats | fp32、fp16、bf16 → fp32 | 已尝试的独立图均未取得可执行计划。 |
| rng | fp32、fp16、bf16 | 已尝试的 uniform、normal、Bernoulli 独立图均未取得可执行计划。 |
| rope | fp32、fp16、bf16 | 当前 cuDNN RoPE 接口用于 SDPA 融合，缺少独立对照。 |
| rope_backward | fp32、fp16、bf16 | 当前 cuDNN RoPE 接口用于 SDPA 融合，缺少独立反向对照。 |
| moe_grouped_matmul_bwd | fp16、bf16、FP8 E4M3/E5M2 | FP16/BF16 对照要求 cuBLASLt ≥13.5，当前加载 12.8.4；FP8 不在 cuDNN backward 支持表内。 |


# Hygon

## 已支持算子

| 算子 | 输入与输出类型 |
| --- | --- |
| identity | fp32、fp16、bf16、int32、bool、FP8 E4M3/E5M2/E8M0；保持原始位模式 |
| add、sub、mul、max、min、add_square、scale | fp32、fp16、bf16、int32 |
| abs | fp32、fp16、bf16 |
| relu、elu、sigmoid、swish、tanh、softplus 及对应 backward | fp32、fp16、bf16 |
| conv_fprop、conv_dgrad、conv_wgrad | fp32、fp16、bf16 |
| conv_bias_relu | fp32、fp16、bf16 |
| reduction | sum、avg、mul；fp32、fp16、bf16 → 同类型或 fp32；int32 → fp32 |
| batchnorm | 数据 fp32、fp16、bf16；统计量 fp32 |

当前 gfx936 的 HCU Triton 不支持 TF32，相关用例跳过。

## 不完全支持算子

| 算子 | 输入与输出类型 | 不完全支持的原因 |
| --- | --- | --- |
| reshape、transpose | 同 identity | 当前没有等价的 hipDNN 性能对照。 |
| slice | 同 identity | hipDNN 对照受 dtype、rank 和输出布局限制，当前性能用例跳过。 |
| concatenate | 同 identity | 当前没有等价的 hipDNN 性能对照。 |
| gen_index | 输出 int32 或 fp32 | 当前没有等价的 hipDNN 性能对照。 |
| div、pow、mod | fp32、fp16、bf16、int32 | 缺少等价的 hipDNN pointwise 性能对照。 |
| cmp_eq、cmp_neq、cmp_gt、cmp_ge、cmp_lt、cmp_le | fp32、fp16、bf16、int32 → bool | 缺少等价的 hipDNN 比较运算性能对照。 |
| logical_and、logical_or、logical_not | bool | 当前 hipDNN 对照不支持这些 bool 运算。 |
| binary_select | bool 条件与 fp32、fp16、bf16、int32 数值 | 缺少等价的 hipDNN 选择运算性能对照。 |
| neg | fp32、fp16、bf16 | hipDNN 的 IDENTITY alpha=-1 路径不适用于重复 HIP Graph 计时；bf16 缺少对照。 |
| sqrt | fp32、fp16、bf16 | 当前 DTK 的 hipDNN OpTensor SQRT 不可用。 |
| rsqrt、reciprocal、ceil、floor、exp、log、erf、sin、cos、tan | fp32、fp16、bf16 | 缺少等价的 hipDNN pointwise 性能对照。 |
| leaky_relu、leaky_relu_backward | fp32、fp16、bf16 | 当前 DTK 的 hipDNN LEAKYRELU 不能准确设置 slope，缺少等价性能对照。 |
| gelu、gelu_approx_tanh 及对应 backward | fp32、fp16、bf16 | 缺少等价的 hipDNN GELU 性能对照。 |
| causal_conv1d | fp32、fp16、bf16 | gfx936 的 HCU Triton 不支持 TF32；当前没有等价的 hipDNN 性能对照。 |
| matmul | fp32、fp16、bf16、FP8 E4M3/E5M2 | gfx936 的 HCU Triton 不支持 TF32；当前没有等价的 hipDNN MatMul 性能对照。 |
| matmul_fp8 | FP8 E4M3/E5M2 或 MXFP8；输出 fp32、fp16、bf16 | 当前没有等价的 hipDNN FP8/MXFP8 MatMul 性能对照。 |
| moe_grouped_matmul | fp16、bf16、FP8 E4M3/E5M2；输出 fp32、fp16、bf16 | 当前没有等价的 hipDNN 分组 MatMul 性能对照。 |
| moe_grouped_matmul_bwd | fp16、bf16、FP8 E4M3/E5M2；输出 fp32、fp16、bf16 | 当前未注册性能测试，缺少等价性能对照。 |
| resample | fp32、fp16、bf16；maxpool 可输出 int32 索引 | 当前未提供等价的 hipDNN pooling/resize 性能对照。 |
| batchnorm_inference | 数据 fp32、fp16、bf16；统计量 fp32 | FlagDNN 输入 inverse variance，hipDNN 输入 variance 与 epsilon，无法按相同输入直接对比。 |
| layernorm、rmsnorm | 数据 fp32、fp16、bf16；统计量 fp32 | 当前 hipDNN 未提供对应的独立归一化原语，缺少性能对照。 |
| batchnorm_backward、layernorm_backward、rmsnorm_backward、instancenorm、instancenorm_backward、adalayernorm、adalayernorm_backward、bn_finalize | 数据 fp32、fp16、bf16；统计量 fp32，仿射参数梯度默认 fp32 | 当前未提供等价的 hipDNN 性能对照。 |
| genstats | fp32、fp16、bf16 → fp32 | 当前未注册性能测试，缺少等价性能对照。 |
| rng | fp32、fp16、bf16；uniform、normal、Bernoulli | 当前未注册性能测试，缺少等价性能对照。 |
| rope、rope_backward | fp32、fp16、bf16 | 当前未注册性能测试，缺少等价性能对照。 |
| sdpa、sdpa_backward | fp32、fp16、bf16 | 当前 hipDNN 未提供等价的 SDPA 性能对照。 |
| sdpa_fp8、sdpa_fp8_backward | FP8 E4M3/E5M2 | 当前 hipDNN 未提供等价的 FP8 SDPA 性能对照。 |


# Iluvatar

## 已支持算子

| 算子 | 输入与输出类型 |
| --- | --- |
| identity、reshape、transpose、slice、concatenate | fp32、fp16、bf16、int32、bool、FP8 E4M3/E5M2/E8M0；保持原始位模式 |
| add、sub、mul、max、min、add_square、scale | fp32、fp16、bf16、int32 |
| logical_and、logical_or、logical_not | bool |
| abs、neg、sqrt | fp32、fp16、bf16 |
| relu、leaky_relu、elu、sigmoid、swish、tanh 及对应 backward | fp32、fp16、bf16 |
| softplus_backward | fp32、fp16、bf16 |
| conv_fprop、conv_dgrad、conv_wgrad | fp32、tf32、fp16、bf16 |
| conv_bias_relu | fp32、fp16、bf16 |
| reduction | sum、avg、mul；fp32、fp16、bf16 → 同类型或 fp32；int32 → fp32 |
| resample | fp32、fp16、bf16；maxpool 可输出 int32 索引 |
| batchnorm、batchnorm_backward、batchnorm_inference、rmsnorm | 数据 fp32、fp16、bf16；统计量 fp32，仿射参数梯度默认 fp32 |

## 不完全支持算子

| 算子 | 输入与输出类型 | 不完全支持的原因 |
| --- | --- | --- |
| gen_index | 输出 int32 或 fp32 | 当前 CoreX DNN 没有等价的独立索引生成接口，缺少性能对照。 |
| div、pow、mod | fp32、fp16、bf16、int32 | 当前 CoreX DNN 缺少等价的 pointwise 性能对照。 |
| cmp_eq、cmp_neq、cmp_gt、cmp_ge、cmp_lt、cmp_le | fp32、fp16、bf16、int32 → bool | 当前 CoreX DNN 缺少等价的比较运算性能对照。 |
| binary_select | bool 条件与 fp32、fp16、bf16、int32 数值 | 当前 CoreX DNN 缺少等价的选择运算性能对照。 |
| rsqrt、reciprocal、ceil、floor、exp、log、erf、sin、cos、tan | fp32、fp16、bf16 | 当前 CoreX DNN 缺少对应的独立算子；功能仅检查生产执行并记录参考缺失，性能用例跳过。 |
| softplus | fp32、fp16、bf16 | 当前 CoreX DNN 缺少等价的前向接口；功能仅检查生产执行并记录参考缺失，性能用例跳过。 |
| gelu、gelu_approx_tanh 及对应 backward | fp32、fp16、bf16 | CoreX DNN 的直接 activation 模式返回 BAD_PARAM，缺少等价性能对照；功能保留 CPU 校验。 |
| causal_conv1d | fp32、tf32、fp16、bf16 | 当前 CoreX DNN 没有等价的独立因果卷积接口，缺少性能对照。 |
| matmul | fp32、tf32、fp16、bf16 | 当前 CoreX DNN 没有独立 GEMM/MatMul 接口，缺少性能对照；FP8 E4M3/E5M2 子用例按 DNN 能力跳过。 |
| matmul_fp8 | FP8 E4M3/E5M2 或 MXFP8；输出 fp32、fp16、bf16 | 当前 CoreX DNN 缺少 FP8/MXFP8 dtype 与 MatMul 接口，功能和性能均按 DNN 能力跳过，尚未验证生产支持。 |
| moe_grouped_matmul | fp16、bf16、FP8 E4M3/E5M2；输出 fp32、fp16、bf16 | 当前 CoreX DNN 没有分组 GEMM 或 graph 接口，功能和性能均按 DNN 能力跳过，尚未验证生产支持。 |
| moe_grouped_matmul_bwd | fp16、bf16、FP8 E4M3/E5M2；输出 fp32、fp16、bf16 | 当前 CoreX DNN 没有分组 GEMM 反向或 graph 接口，功能和性能均按 DNN 能力跳过，尚未验证生产支持。 |
| layernorm | 数据 fp32、fp16、bf16；统计量 fp32 | 当前 CoreX DNN 缺少等价的独立归一化接口；功能仅检查生产执行并记录参考缺失，性能用例跳过。 |
| layernorm_backward、rmsnorm_backward、instancenorm、instancenorm_backward、adalayernorm、adalayernorm_backward | 数据 fp32、fp16、bf16；统计量 fp32，仿射参数梯度默认 fp32 | 当前 CoreX DNN 缺少对应的独立归一化接口或 graph API，缺少性能对照；功能保留 CPU 校验。 |
| bn_finalize | 数据和统计量 fp32 | CoreX DNN 的 FP32 FusedOps 计划返回成功但未写入统计及仿射输出，性能用例跳过；功能保留 CPU 校验。 |
| genstats | fp32、fp16、bf16 → fp32 | 当前 CoreX DNN 没有等价的独立统计接口，缺少性能对照。 |
| rng | fp32、fp16、bf16；uniform、normal、Bernoulli | CoreX DNN 的 Dropout 接口不满足独立 RNG 的 seed、offset 和分布契约，缺少等价性能对照。 |
| rope、rope_backward | fp32、fp16、bf16 | 当前 CoreX DNN 没有等价的独立 RoPE 接口，缺少性能对照。 |
| sdpa、sdpa_backward | fp32、fp16、bf16 | 当前 CoreX DNN 未提供 Flash Attention 或 graph 接口；功能仅检查生产执行并记录参考缺失，性能用例跳过。 |
| sdpa_fp8、sdpa_fp8_backward | FP8 E4M3/E5M2 | 当前 CoreX DNN 缺少 FP8 dtype 与 Attention 接口；功能仅检查生产执行并记录参考缺失，性能用例跳过。 |

# Mthreads

## 已支持算子

| 算子 | 输入与输出类型 |
| --- | --- |
| identity、reshape、transpose、slice、concatenate | 保持输入类型与原始位模式；支持 fp32、fp16、bf16、int32、bool、FP8 E4M3/E5M2/E8M0 |
| gen_index | 输出 int32 或 fp32 |
| add、sub、mul、div、max、min、mod、add_square、scale | fp32、fp16、bf16、int32 |
| pow | fp32、fp16、bf16 |
| cmp_eq、cmp_neq、cmp_gt、cmp_ge、cmp_lt、cmp_le | fp32、fp16、bf16、int32 → bool |
| logical_and、logical_or、logical_not | bool |
| binary_select | bool 条件与 fp32、fp16、bf16、int32 数值 |
| abs、neg、sqrt、rsqrt、reciprocal、ceil、floor、exp、log、erf、sin、cos、tan | fp32、fp16、bf16 |
| relu、leaky_relu、elu、gelu、gelu_approx_tanh、sigmoid、swish、tanh、softplus | fp32、fp16、bf16 |
| relu_backward、leaky_relu_backward、gelu_backward、gelu_approx_tanh_backward、sigmoid_backward、swish_backward、tanh_backward | fp32、fp16、bf16 |
| conv_fprop、conv_dgrad、conv_wgrad、causal_conv1d | fp32、tf32、fp16、bf16 |
| conv_bias_relu | fp32、fp16、bf16 |
| matmul | fp32、tf32、fp16、bf16、FP8 E4M3/E5M2 |
| matmul_fp8 | FP8 E4M3/E5M2；输出 fp32、fp16、bf16 |
| moe_grouped_matmul、moe_grouped_matmul_bwd | fp16、bf16、FP8 E4M3/E5M2；输出 fp32、fp16、bf16 |
| reduction | fp32、fp16、bf16 → 同类型或 fp32；int32 → fp32 |
| resample | fp32、fp16、bf16；maxpool 可输出 int32 索引 |
| batchnorm、batchnorm_backward、batchnorm_inference、layernorm、layernorm_backward、rmsnorm、rmsnorm_backward、instancenorm、instancenorm_backward、adalayernorm、adalayernorm_backward | 数据 fp32、fp16、bf16；统计量 fp32，仿射参数梯度默认 fp32 |
| bn_finalize | fp32 统计量与参数 → fp32 |
| genstats | fp32、fp16、bf16 → fp32 |
| rope、rope_backward | fp32、fp16、bf16 |
| sdpa、sdpa_backward | fp32、fp16、bf16 |

## 不完全支持算子

| 算子 | 输入与输出类型 | 不完全支持的原因（MTT S5000、muDNN 3.1.5） |
| --- | --- | --- |
| pow | int32 | 当前 muDNN Binary POW 没有 int32 kernel，相关类型用例跳过。 |
| relu_backward、leaky_relu_backward | fp32、fp16、bf16 | 当前 muDNN LEAKY_RELU_BW 无法表达带上下界裁剪的梯度，相关属性用例跳过。 |
| elu_backward、softplus_backward | fp32、fp16、bf16 | 当前 muDNN 没有对应的原生 backward 模式，缺少等价性能对照，相关用例全部跳过。 |
| swish_backward | fp32、fp16、bf16 | 当前 muDNN SILU_BW 的 beta 固定为 1，非默认 beta 的属性用例跳过。 |
| matmul_fp8 | MXFP8（E8M0 scales）；输出 fp32、fp16、bf16 | 当前 muDNN Tensor::Type 没有 E8M0，MXFP8 用例跳过。 |
| rng | fp32、fp16、bf16；uniform、normal、Bernoulli | 当前 muDNN 没有与 Graph RNG 对应的独立计数器随机数接口，缺少等价性能对照，相关用例全部跳过。 |
| sdpa、sdpa_backward | fp32 | 当前 muDNN RunFlash 不支持 fp32，RunMath 要求 Q/K/V 头数与头维度一致；fp32 的 GQA 或不同 QK、V 头维度用例跳过。 |
| sdpa_backward | fp32、fp16、bf16 | 当前 muDNN RunFlashBwd / RunMathBwd 没有 dBias 输出，需要该梯度输出的用例跳过。 |
| sdpa_fp8、sdpa_fp8_backward | FP8 E4M3/E5M2 | 当前 muDNN RunFlash / RunMath 不支持 FP8 Q/K/V，且没有 FP8 scale/amax 接口，相关用例全部跳过。 |
