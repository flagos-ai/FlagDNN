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
