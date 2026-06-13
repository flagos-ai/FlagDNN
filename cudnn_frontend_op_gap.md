# FlagDNN 对标 cuDNN Frontend 算子差距文档

本文件基于本机 `/home/wangbingjie/cudnn-frontend` 的 Python API 生成，重点覆盖：

- `cudnn.pygraph` 暴露的 graph 算子；
- `cudnn.ops` 和部分 SM100 高阶 Python wrapper；
- 当前 `src/flag_dnn/ops` 已有算子；
- 每类算子建议在 FlagDNN 的实现位置、graph 接入位置、测试位置。

当前 cuDNN backend version：`91002`。

## 1. 实现原则

FlagDNN 对标 cuDNN Frontend 时，不建议只补 eager 单算子。每个新增算子至少要考虑四层：

| 层 | 位置 | 职责 |
|---|---|---|
| eager op | `src/flag_dnn/ops/<op>.py` | 真实 Torch/Triton 算子实现 |
| Python 导出 | `src/flag_dnn/ops/__init__.py`、`src/flag_dnn/__init__.py` | 让用户能 `flag_dnn.<op>` 调用 |
| graph op | `src/flag_dnn/graph/registry.py`、`wrappers.py` | capture、shape/dtype 推导、executor lowering |
| graph optimization | `src/flag_dnn/graph/passes/`、`backend.py`、`kernel_selector.py`、`kernels/` | fusion pattern、候选 kernel、autotune、workspace |

测试建议：

| 测试类型 | 位置 |
|---|---|
| 正确性 | `tests/test_<op>.py` |
| 性能 | `benchmark/test_<op>_perf.py` |
| cuDNN 功能对标 | `tests_graph/test_<op>.py` |
| cuDNN graph 性能对标 | `benchmark_graph/test_<op>_perf.py` |
| graph 子图对标 | `tests_graph/test_graph_<pattern>.py` |

## 2. 总体差距结论

当前 FlagDNN 已经覆盖不少 PyTorch 风格 eager op，例如 `conv1d`、`conv2d`、`mm`、`add/sub/mul/div`、`relu`、`gelu`、`silu`、`softmax`、`layer_norm`、`rms_norm`、`batch_norm`、pooling、reduction 等。

但 cuDNN Frontend 的覆盖重点和 FlagDNN 当前重点不同：

1. cuDNN Frontend 有大量 graph primitive：`reshape`、`transpose`、`slice`、`concatenate`、`identity`、`reduction`、`binary_select`、`gen_index`。
2. cuDNN Frontend 点算子更完整：`exp/log/erf/sin/cos/tan/ceil/floor/reciprocal/sigmoid/min/max/mod/logical_*`。
3. cuDNN Frontend 有 backward graph op：activation backward、norm backward、conv dgrad/wgrad、SDPA backward。
4. cuDNN Frontend 强项是 fused pattern：SDPA、FP8 SDPA、MXFP8 SDPA、Norm、Block-scale quant/dequant、MoE grouped GEMM。
5. FlagDNN graph registry 已覆盖基础 utility、二元/比较/logical、`binary_select`、unary pointwise、部分 activation、`conv_fprop`、`matmul`、`reduction` 和 norm forward；剩余差距主要集中在 backward、attention、低精度和 MoE wrapper。

## 3. 优先级建议

| 优先级 | 目标 | 原因 |
|---|---|---|
| P0 | graph alias hygiene + edge-case coverage | utility、binary/logical、`binary_select` 和 unary pointwise 主体已完成，后续主要是接口别名、边界 dtype/layout 与测试覆盖查漏 |
| P1 | conv dgrad/wgrad、norm backward、matmul/bmm 边界和低精度增强 | cuDNN 基础训练能力，训练/推理都常用 |
| P2 | SDPA forward/backward | cuDNN frontend 的核心对标点，收益高 |
| P3 | block-scale quant/dequant、FP8/MXFP8 SDPA | 面向低精度和 Hopper/Blackwell 路线 |
| P4 | SM100 MoE grouped GEMM wrappers | 专用场景强，但实现复杂、硬件相关强 |

## 4. cuDNN pygraph 算子覆盖表

状态说明：

- `已有`：FlagDNN eager 基本已有同类算子。
- `部分`：语义接近但参数、输出、layout、训练态或 graph 接入不完整。
- `缺失`：没有明确对应实现。
- `Graph缺`：eager 有，但 `src/flag_dnn/graph/registry.py` / `wrappers.py` 未覆盖或只覆盖子集。

### 4.1 Tensor / graph utility

| cuDNN op | 参数 | FlagDNN 状态 | 建议实现位置 |
|---|---|---|---|
| `identity` | `input, compute_data_type=NOT_SET, name=''` | Graph utility；默认 view/no-op，`out` 走 Triton copy | `ops/identity.py`；graph registry |
| `reshape` | `input, name='', reshape_mode=VIEW_ONLY` | Graph utility；view-only，不做 torch materialize fallback | `ops/reshape.py`；graph shape-only op |
| `transpose` | `input, permutation, compute_data_type=NOT_SET, name=''` | Graph utility；view-only，materialize 暂不启用 | `ops/transpose.py` |
| `slice` | `input, slices=[], compute_data_type=NOT_SET, name=''` | Graph utility；view-only，materialize 暂不启用 | `ops/slice.py` |
| `concatenate` | `inputs, axis, in_place_index=None, name=''` | 已有 Triton + Graph；用于真实数据搬运 | `ops/concatenate.py` |
| `gen_index` | `input, axis, compute_data_type=NOT_SET, name=''` | 已有 Triton + Graph；用于真实数据生成 | `ops/gen_index.py` |

这组算子主要服务 graph IR 表达，不作为稳定的 standalone cuDNN 性能对标项。`reshape/transpose/slice` 优先保持 view/metadata 语义；`identity(out=...)`、`gen_index` 和 `concatenate` 才使用 Triton kernel 做真实数据搬运或生成。

### 4.2 Binary / comparison / logical

| cuDNN op | 参数 | FlagDNN 状态 | 建议实现位置 |
|---|---|---|---|
| `add` | `a, b, compute_data_type=NOT_SET, name=''` | 已有 `add(input, other, alpha=1, out=None)`；Graph 有 | 已有 |
| `sub` | `a, b, compute_data_type=NOT_SET, name=''` | 已有；Graph 有 | 已有 |
| `mul` | `a, b, compute_data_type=NOT_SET, name=''` | 已有；Graph 有 | 已有 |
| `div` | `a, b, compute_data_type=NOT_SET, name=''` | 已有；Graph 有 | 已有 |
| `pow` | `input0, input1, compute_data_type=NOT_SET, name=''` | 已有 eager + Graph | 已有 |
| `max` | `input0, input1, compute_data_type=NOT_SET, name=''` | 已有 elementwise max；Graph 有 | `ops/max.py` |
| `min` | `input0, input1, compute_data_type=NOT_SET, name=''` | 已有 cudnn-style `min` graph alias；复用 `minimum` Triton 实现，`tests_graph/benchmark_graph` 已覆盖 | `ops/min.py` + `ops/minimum.py`；graph registry |
| `mod` | `input0, input1, compute_data_type=NOT_SET, name=''` | 已有 Triton + Graph；`tests_graph/benchmark_graph` 已覆盖 | `ops/mod.py` |
| `cmp_eq` | `input, comparison, compute_data_type=NOT_SET, name=''` | 已有 cudnn-style alias，复用 `eq`；Graph 有，`tests_graph/benchmark_graph` 已覆盖 | 已有 |
| `cmp_neq` | `input, comparison, compute_data_type=NOT_SET, name=''` | 已有 cudnn-style alias，复用 `ne`；Graph 有，`tests_graph/benchmark_graph` 已覆盖 | 已有 |
| `cmp_gt` | `input, comparison, compute_data_type=NOT_SET, name=''` | 已有 cudnn-style alias，复用 `gt`；Graph 有，`tests_graph/benchmark_graph` 已覆盖 | 已有 |
| `cmp_ge` | `input, comparison, compute_data_type=NOT_SET, name=''` | 已有 cudnn-style alias，复用 `ge`；Graph 有，`tests_graph/benchmark_graph` 已覆盖 | 已有 |
| `cmp_lt` | `input, comparison, compute_data_type=NOT_SET, name=''` | 已有 cudnn-style alias，复用 `lt`；Graph 有，`tests_graph/benchmark_graph` 已覆盖 | 已有 |
| `cmp_le` | `input, comparison, compute_data_type=NOT_SET, name=''` | 已有 cudnn-style alias，复用 `le`；Graph 有，`tests_graph/benchmark_graph` 已覆盖 | 已有 |
| `logical_and` | `a, b, compute_data_type=NOT_SET, name=''` | 已有 bool tensor cudnn-style graph 封装；复用 `bitwise_and` Triton 实现，cuDNN 当前环境无 valid engine 时测试 skip | `ops/logical_and.py`；graph registry |
| `logical_or` | `a, b, compute_data_type=NOT_SET, name=''` | 已有 bool tensor cudnn-style graph 封装；复用 `bitwise_or` Triton 实现，cuDNN 当前环境无 valid engine 时测试 skip | `ops/logical_or.py`；graph registry |
| `logical_not` | `input, compute_data_type=NOT_SET, name=''` | 已有 bool tensor cudnn-style graph 封装；复用 `bitwise_not` Triton 实现，cuDNN 当前环境无 valid engine 时测试 skip | `ops/logical_not.py`；graph registry |
| `binary_select` | `input0, input1, mask, compute_data_type=NOT_SET, name=''` | 已有 Triton + Graph；`tests_graph/benchmark_graph` 已覆盖 | `ops/binary_select.py`；`where` 仅作为兼容 alias |
| `add_square` | `a, b, compute_data_type=NOT_SET, name=''` | 已有 cudnn-style graph 封装；复用 `square` + `add`，`tests_graph/benchmark_graph` 已覆盖 | `ops/add_square.py`，语义 `a + b*b` |

### 4.3 Unary pointwise

| cuDNN op | 参数 | FlagDNN 状态 | 建议实现位置 |
|---|---|---|---|
| `abs` | `input, compute_data_type=NOT_SET, name=''` | 已有；Graph 有 | 已有 |
| `neg` | `input, compute_data_type=NOT_SET, name=''` | 已有；Graph 有 | 已有 |
| `sqrt` | `input, compute_data_type=NOT_SET, name=''` | 已有；Graph 有 | 已有 |
| `rsqrt` | `input, compute_data_type=NOT_SET, name=''` | 已有 Triton + Graph；`tests_graph/benchmark_graph` 已覆盖 | `ops/rsqrt.py` |
| `reciprocal` | `input, compute_data_type=NOT_SET, name=''` | 已有 Triton + Graph；`tests_graph/benchmark_graph` 已覆盖 | `ops/reciprocal.py` |
| `ceil` | `input, compute_data_type=NOT_SET, name=''` | 已有 Triton + Graph；`tests_graph/benchmark_graph` 已覆盖 | `ops/ceil.py` |
| `floor` | `input, compute_data_type=NOT_SET, name=''` | 已有 Triton + Graph；`tests_graph/benchmark_graph` 已覆盖 | `ops/floor.py` |
| `exp` | `input, compute_data_type=NOT_SET, name=''` | 已有 Triton + Graph；`tests_graph/benchmark_graph` 已覆盖 | `ops/exp.py` |
| `log` | `input, compute_data_type=NOT_SET, name=''` | 已有 Triton + Graph；`tests_graph/benchmark_graph` 已覆盖 | `ops/log.py` |
| `erf` | `input, compute_data_type=NOT_SET, name=''` | 已有 Triton + Graph；`tests_graph/benchmark_graph` 已覆盖 | `ops/erf.py` |
| `sin` | `input, compute_data_type=NOT_SET, name=''` | 已有 Triton + Graph；`tests_graph/benchmark_graph` 已覆盖 | `ops/sin.py` |
| `cos` | `input, compute_data_type=NOT_SET, name=''` | 已有 Triton + Graph；`tests_graph/benchmark_graph` 已覆盖 | `ops/cos.py` |
| `tan` | `input, compute_data_type=NOT_SET, name=''` | 已有 Triton + Graph；`tests_graph/benchmark_graph` 已覆盖 | `ops/tan.py` |
| `scale` | `input, scale, compute_data_type=NOT_SET, name=''` | 已有 eager + Graph；映射到 `mul` | 已有 |

这组 cuDNN unary pointwise 已通过扩展 `src/flag_dnn/ops/unary.py` 统一模板实现；4D channels-last/NHWC case 走 dense flat fast path，避免退化到 strided kernel。

### 4.4 Activation forward/backward

| cuDNN op | 参数 | FlagDNN 状态 | 建议实现位置 |
|---|---|---|---|
| `relu` | `input, negative_slope=None, lower_clip=None, upper_clip=None, compute_data_type=NOT_SET, name=''` | 已有 eager + Graph；支持 slope/lower_clip/upper_clip | 已有 |
| `relu_backward` | `loss, input, negative_slope=None, lower_clip=None, upper_clip=None, compute_data_type=NOT_SET, name=''` | 缺失 | `ops/relu_backward.py` |
| `leaky_relu` | `input, negative_slope, compute_data_type=NOT_SET, name=''` | 已有 eager + Graph | 已有 |
| `leaky_relu_backward` | `loss, input, negative_slope, compute_data_type=NOT_SET, name=''` | 缺失 | `ops/leaky_relu_backward.py` |
| `elu` | `input, compute_data_type=NOT_SET, name=''` | 已有 eager + Graph | 已有 |
| `elu_backward` | `loss, input, compute_data_type=NOT_SET, name=''` | 缺失 | `ops/elu_backward.py` |
| `gelu` | `input, compute_data_type=NOT_SET, name=''` | 部分；Graph 有，但当前 `gelu` schema 主要支持 `approximate`，cuDNN-style `compute_data_type/name` 覆盖集中在 `gelu_approx_tanh` alias | 已有；可补齐 `gelu` attr 对齐 |
| `gelu_backward` | `loss, input, compute_data_type=NOT_SET, name=''` | 缺失 | `ops/gelu_backward.py` |
| `gelu_approx_tanh` | `input, compute_data_type=NOT_SET, name=''` | 已有 eager + Graph alias | 已有 |
| `gelu_approx_tanh_backward` | `loss, input, compute_data_type=NOT_SET, name=''` | 缺失 | `ops/gelu_backward.py` 支持 approximate |
| `sigmoid` | `input, compute_data_type=NOT_SET, name=''` | 已有 eager + Graph | `ops/sigmoid.py` |
| `sigmoid_backward` | `loss, input, compute_data_type=NOT_SET, name=''` | 已有 eager + Graph | `ops/sigmoid_backward.py` |
| `swish` | `input, swish_beta=None, compute_data_type=NOT_SET, name=''` | 已有 eager + Graph；支持 `swish_beta` | 已有 |
| `swish_backward` | `loss, input, compute_data_type=NOT_SET, swish_beta=None, name=''` | 缺失 | `ops/swish_backward.py` |
| `tanh` | `input, compute_data_type=NOT_SET, name=''` | 已有；Graph 有 | 已有 |
| `tanh_backward` | `loss, input, compute_data_type=NOT_SET, name=''` | 缺失 | `ops/tanh_backward.py` |
| `softplus` | `input, compute_data_type=NOT_SET, name=''` | 已有 eager + Graph | 已有 |
| `softplus_backward` | `loss, input, compute_data_type=NOT_SET, name=''` | 缺失 | `ops/softplus_backward.py` |

### 4.5 Convolution

| cuDNN op | 参数 | FlagDNN 状态 | 建议实现位置 |
|---|---|---|---|
| `conv_fprop` | `image, weight, padding/stride/dilation` 或 `pre_padding, post_padding, stride, dilation, convolution_mode=CROSS_CORRELATION, compute_data_type=NOT_SET, name=''` | 已有 eager + Graph；rank1/2/3、pre/post padding、`CROSS_CORRELATION`/`CONVOLUTION` 和 benchmark 输出已覆盖 | 已有 |
| `conv_dgrad` | `loss, filter, padding/stride/dilation` 或 `pre_padding, post_padding, stride, dilation, convolution_mode, compute_data_type, name` | 缺失 | `ops/conv_dgrad.py` 或 `ops/conv_transpose*.py` |
| `conv_wgrad` | `image, loss, padding/stride/dilation` 或 `pre_padding, post_padding, stride, dilation, convolution_mode, compute_data_type, name` | 缺失 | `ops/conv_wgrad.py` |
| `causal_conv1d` high-level | `x, weight, bias=None, activation='identity'` | 已有 eager + Graph；支持 depthwise causal padding、bias、`identity`/`silu` | 已有 |

`conv_fprop` 已把 graph attrs 对齐到 cuDNN 常用 forward 参数；`CONVOLUTION` mode 通过空间维 flip weight 后复用现有 cross-correlation kernel。

### 4.6 Matmul / GEMM

| cuDNN op | 参数 | FlagDNN 状态 | 建议实现位置 |
|---|---|---|---|
| `matmul` | `A, B, compute_data_type=NOT_SET, padding=0.0, name=''` | 已有；2D 走 `mm`，3D 同 batch 走 Triton 快路径，其它 rank/batch broadcast 也走 Triton batched kernel；`padding` 按 cuDNN backend padding-value 兼容参数接收，标准 K 匹配时不改变数学结果 | 已有；后续可补动态 K override/低精度专用 GEMM |
| `moe_grouped_matmul` | `token, weight, first_token_offset, token_index=None, token_ks=None, mode=NONE, compute_data_type=FLOAT, top_k=0, name=''` | 缺失 | `ops/moe_grouped_matmul.py`，graph kernel |
| `moe_grouped_matmul_bwd` | `doutput, token, first_token_offset, compute_data_type=FLOAT, name=''` | 缺失 | `ops/moe_grouped_matmul_bwd.py` |

普通 `matmul` 已覆盖 2D、同 batch 3D 优化路径和其它 rank/broadcast；cuDNN `padding` 是 backend matmul descriptor 的 padding value，FlagDNN 侧作为兼容参数接收，shape/K 仍按标准 matmul 校验。

### 4.7 Reduction / stats

| cuDNN op | 参数 | FlagDNN 状态 | 建议实现位置 |
|---|---|---|---|
| `reduction` | `input, mode, compute_data_type=NOT_SET, name=''` | 已有 eager + Graph；支持 `ADD/AVG/MUL/MIN/MAX/AMAX/NORM1/NORM2/MUL_NO_ZEROS`，其中 `MIN/MAX/AMAX/MUL_NO_ZEROS` 已改为 Triton 原生实现；FlagDNN graph 需显式 `dim/keepdim`；`ADD/AVG/MUL` 已有 cuDNN graph 对标，扩展模式用 torch-reference 覆盖 | 已有；后续可补 cuDNN 输出 shape 推导式 API |
| `genstats` | `input, compute_data_type=NOT_SET, name=''` -> `[mean, inv_variance]` | 缺失 | `ops/genstats.py` |

cuDNN `reduction_mode` 已映射到 FlagDNN graph 的显式 `dim/keepdim` reduction；当前 `ADD/AVG/MUL` 走 cuDNN Frontend 对标，`MIN/MAX/AMAX/NORM1/NORM2/MUL_NO_ZEROS` 作为 FlagDNN 扩展 coverage 保留 torch-reference 测试。

### 4.8 Norm

| cuDNN op | 参数 | FlagDNN 状态 | 建议实现位置 |
|---|---|---|---|
| `batchnorm` | `input, scale, bias, in_running_mean, in_running_var, epsilon, momentum, peer_stats=[], compute_data_type=NOT_SET, name=''` -> list | 部分；已有 cudnn-style eager + Graph 多输出 forward，返回 `Y/mean/inv_var/next_running_mean/next_running_var`；单设备 `peer_stats=[tensor]` 已可透传并按本地 BN 语义执行，多设备同步 peer stats 仍未支持 | `ops/batchnorm.py`；后续补多 GPU peer_stats/backward |
| `batchnorm_inference` | `input, mean, inv_variance, scale, bias, compute_data_type=NOT_SET, name=''` | 已有 eager + Graph；功能/性能已改用 legacy cuDNN standalone `cudnnBatchNormalizationForwardInference` 对标；cuDNN Frontend graph standalone 在本环境无 plan | 已有 |
| `batchnorm_backward` | `grad, input, scale, mean, inv_variance, peer_stats=[], compute_data_type=NOT_SET, name=''` | 缺失 | `ops/batch_norm_backward.py` |
| `layernorm` | `norm_forward_phase, input, scale, bias, epsilon, compute_data_type=NOT_SET, name=''` -> list | 已有 cudnn-style eager + Graph 多输出 forward，返回 `Y/mean/inv_var`；保留原 `layer_norm` API | 已有；backward 另补 |
| `layernorm_backward` | `grad, input, scale, mean, inv_variance, compute_data_type=NOT_SET, name=''` | 缺失 | `ops/layer_norm_backward.py` |
| `rmsnorm` | `norm_forward_phase, input, scale, bias=None, epsilon, compute_data_type=NOT_SET, name=''` -> list | 已有 cudnn-style eager + Graph 多输出 forward，返回 `Y/inv_var` 并支持 bias；保留原 `rms_norm` API | 已有；backward 另补 |
| `rmsnorm_backward` | `grad, input, scale, inv_variance, has_dbias, compute_data_type=NOT_SET, name=''` | 缺失 | `ops/rms_norm_backward.py` |
| `instancenorm` | `norm_forward_phase, input, scale, bias, epsilon, compute_data_type=NOT_SET, name=''` | 缺失 | `ops/instance_norm.py` |
| `instancenorm_backward` | `grad, input, scale, mean=None, inv_variance=None, compute_data_type=NOT_SET, name=''` | 缺失 | `ops/instance_norm_backward.py` |
| `adalayernorm` | `norm_forward_phase, input, scale, bias=None, epsilon, compute_data_type=NOT_SET, name=''` | 缺失 | `ops/adalayer_norm.py` |
| `adalayernorm_backward` | `grad, input, scale, mean, inv_variance, compute_data_type=NOT_SET, name=''` | 缺失 | `ops/adalayer_norm_backward.py` |

Norm 的 cuDNN API 返回 list，通常包含 output 和保存给 backward 的 stats。FlagDNN graph 已启用 multi-output，并新增 cudnn-style `batchnorm/layernorm/rmsnorm` forward tuple 返回；backward 和真正跨设备同步 peer stats 仍需单独补。

### 4.9 Quant / FP8

| cuDNN op | 参数 | FlagDNN 状态 | 建议实现位置 |
|---|---|---|---|
| `block_scale_quantize` | `input, block_size, axis=None, transpose=False, compute_data_type=NOT_SET, name=''` -> `[quantized, scale]` | 缺失 | `ops/block_scale_quantize.py` |
| `block_scale_dequantize` | `input, descale, block_size, is_negative_scale=False, compute_data_type=NOT_SET, name=''` | 缺失 | `ops/block_scale_dequantize.py` |

这类 op 是 FP8/MXFP8 SDPA 和 block-scaled GEMM 的基础，建议 P3 实现。

### 4.10 Attention

| cuDNN op | 参数摘要 | FlagDNN 状态 | 建议实现位置 |
|---|---|---|---|
| `sdpa` / `scaled_dot_product_flash_attention` | `q, k, v, is_inference=None, attn_scale=None, bias=None, block_mask=None, use_alibi_mask=False, use_padding_mask=False, seq_len_q=None, seq_len_kv=None, use_causal_mask=False, use_causal_mask_bottom_right=False, sliding_window_length=None, diagonal_alignment=TOP_LEFT, diagonal_band_left_bound=None, diagonal_band_right_bound=None, dropout=None, rng_dump=None, paged_attention_k_table=None, paged_attention_v_table=None, paged_attention_max_seq_len_kv=None, compute_data_type=NOT_SET, name='', score_mod=None, generate_stats=None, implementation=AUTO, score_max=None, score_sum_exp=None, sink_token=None, unfuse_fma=False` | 缺失 | `ops/sdpa.py`，graph fused op |
| `sdpa_backward` | `q, k, v, o, dO, stats, attn_scale=None, bias=None, dBias=None, use_alibi_mask=False, use_padding_mask=False, seq_len_q=None, seq_len_kv=None, max_total_seq_len_q=None, max_total_seq_len_kv=None, use_causal_mask=False, use_causal_mask_bottom_right=False, sliding_window_length=None, diagonal_alignment=TOP_LEFT, diagonal_band_left_bound=None, diagonal_band_right_bound=None, dropout=None, rng_dump=None, use_deterministic_algorithm=False, compute_data_type=NOT_SET, name='', score_mod=None, score_mod_bprop=None, sink_token=None, dSink_token=None` | 缺失 | `ops/sdpa_backward.py` |
| `sdpa_fp8` | `q, k, v, descale_q, descale_k, descale_v, descale_s, scale_s, scale_o, ...` | 缺失 | `ops/sdpa_fp8.py` |
| `sdpa_fp8_backward` | `q, k, v, o, dO, stats, descale_q, descale_k, descale_v, descale_o, descale_dO, descale_s, descale_dP, scale_s, scale_dQ, scale_dK, scale_dV, scale_dP, ...` | 缺失 | `ops/sdpa_fp8_backward.py` |
| `sdpa_mxfp8` | `q, k, v, descale_q, descale_k, descale_v, attn_scale=None, use_causal_mask=False, ... generate_stats, sink_token=None, unfuse_fma=False` | 缺失 | `ops/sdpa_mxfp8.py` |
| `sdpa_mxfp8_backward` | `q, q_T, k, k_T, v, o_f16, dO_f16, dO, dO_T, stats, descale_* ...` | 缺失 | `ops/sdpa_mxfp8_backward.py` |

建议路线：

1. P2 先做 `sdpa(q,k,v, attn_scale, bias, causal, dropout=0)` forward。
2. 再做 variable length：`seq_len_q/seq_len_kv`。
3. 再做 backward。
4. 最后做 FP8/MXFP8 和 paged attention。

### 4.11 Native sparse attention wrappers

| cuDNN wrapper | 参数摘要 | FlagDNN 状态 | 建议实现位置 |
|---|---|---|---|
| `selection_attention_wrapper` | `q_tensor, k_tensor, v_tensor, block_indices_tensor, block_counts_tensor, cum_seqlen_q_tensor=None, cum_seqlen_k_tensor=None, block_size=64, scale_softmax=None, o_dtype=None, acc_dtype=float32, max_s_q=None, max_s_k=None, stream=None` | 缺失 | `ops/selection_attention.py` |
| `compression_attention_wrapper` | `q_tensor, k_tensor, v_tensor, cum_seqlen_q_tensor=None, cum_seqlen_k_tensor=None, enable_lse=False, o_dtype=None, qk_acc_dtype=float32, pv_acc_dtype=float32, mma_tiler_mn=(128,128), is_persistent=False, scale_q=1.0, scale_k=1.0, scale_v=1.0, inv_scale_o=1.0, scale_softmax=None, stream=None` | 缺失 | `ops/compression_attention.py` |
| `sliding_window_attention_wrapper` | `q_tensor, k_tensor, v_tensor, seq_len_q_tensor=None, seq_len_kv_tensor=None, q/k/v/o/stats_ragged_offset_tensor=None, left_bound=0, right_bound=0, is_infer=False, attn_scale=None, o_dtype=None, intermediate_data_type=float32, compute_data_type=float32, cudnn_handle=None, stream=None` | 缺失 | `ops/sliding_window_attention.py` |
| `topk_reduction_wrapper` | `q_tensor, k_tensor, lse_tensor, cum_seqlen_q_tensor=None, cum_seqlen_k_tensor=None, max_s_q=None, max_s_k=None, acc_dtype=float32, k_value=16, selection_block_size=64, compress_stride=32, is_causal=True, mma_tiler_mn=(128,128), scale_softmax=None, current_stream=None` | 缺失 | `ops/topk_reduction.py` |

这些是后期目标，依赖 SDPA 和 block sparse 基础设施。

### 4.12 SM100 GEMM / MoE high-level wrappers

这些 wrapper 硬件相关强，通常依赖 Blackwell/SM100、CUTLASS/CuTe、block-scaled FP8/BF16 路线。建议先在 graph 层预留 op schema，不急于完整实现。

| wrapper | 参数摘要 | FlagDNN 状态 | 建议实现位置 |
|---|---|---|---|
| `gemm_amax_wrapper_sm100` | `a_tensor, b_tensor, sfa_tensor, sfb_tensor, c_major='n', c_dtype=float32, acc_dtype=float32, mma_tiler_mn=(128,128), cluster_shape_mn=(1,1), sf_vec_size=32, stream=None` | 缺失 | `ops/gemm_amax.py` |
| `gemm_srelu_wrapper_sm100` | `a_tensor, b_tensor, sfa_tensor, sfb_tensor, prob_tensor, alpha=1.0, c_major='n', c_dtype=bf16, d_dtype=bf16, acc_dtype=float32, mma_tiler_mn=(256,256), cluster_shape_mn=None, norm_const_tensor=None, sf_vec_size=16, vector_f32=False, stream=None` | 缺失 | `ops/gemm_srelu.py` |
| `gemm_dsrelu_wrapper_sm100` | `a_tensor, b_tensor, c_tensor, sfa_tensor, sfb_tensor, prob_tensor, alpha=1.0, d_major='n', d_dtype=bf16, acc_dtype=float32, mma_tiler_mn=(256,256), cluster_shape_mn=None, norm_const_tensor=None, sf_vec_size=16, vector_f32=False, stream=None` | 缺失 | `ops/gemm_dsrelu.py` |
| `gemm_swiglu_wrapper_sm100` | `a_tensor, b_tensor, alpha=1.0, c_major='n', ab12_dtype=float32, c_dtype=float16, acc_dtype=float32, mma_tiler_mn=(128,128), cluster_shape_mn=None, sfa_tensor=None, sfb_tensor=None, norm_const_tensor=None, sf_vec_size=16, vector_f32=False, ab12_stages=4, stream=None` | 缺失 | `ops/gemm_swiglu.py` |
| `grouped_gemm_*_wrapper_sm100` | grouped/MoE variants with `padded_offsets`, `alpha_tensor`, optional pointer tensors, scale tensors, `mma_tiler_mn`, `cluster_shape_mn`, `sf_vec_size`, scheduler flags | 缺失 | `ops/grouped_gemm.py` plus specialized graph kernels |
| `discrete_grouped_gemm_*_wrapper_sm100` | pointer-based discrete grouped GEMM variants | 缺失 | `ops/discrete_grouped_gemm.py` |
| `rmsnorm_rht_amax_wrapper_sm100` | `x_tensor, w_tensor, eps=1e-5, num_threads=None, rows_per_cta=None, current_stream=None` | 已有；Triton 实现 `RMSNorm + 16 点 normalized RHT + per-CTA amax`，支持官方 wrapper 的 2D/尾部 1 维 squeeze、bf16 row-major 输入和 `rows_per_cta/num_threads` 校验；不是 SM100 CUTE 专用优化 kernel | 已有；后续可补 SM100/CUTE 专用性能路径 |

## 5. 当前剩余新增/补齐文件清单

以下只列仍缺失或需要新增的文件。`identity/reshape/transpose/slice/concatenate/gen_index`、binary/compare/logical、`binary_select`、unary pointwise、`matmul/conv_fprop/reduction`、norm forward 和 `rmsnorm_rht_amax.py` 等当前已存在，不再作为新增项列出。

### P0：低成本补齐

```text
暂无新增 op 文件；当前主要剩余工作是 graph alias hygiene 和边界 coverage 查漏。
```

同步修改：

```text
src/flag_dnn/ops/__init__.py
src/flag_dnn/__init__.py
src/flag_dnn/graph/registry.py
src/flag_dnn/graph/wrappers.py
tests/test_<op>.py
tests_graph/test_<op>.py
benchmark_graph/test_<op>_perf.py
```

### P1：基础训练/推理核心

```text
src/flag_dnn/ops/conv_dgrad.py
src/flag_dnn/ops/conv_wgrad.py
src/flag_dnn/ops/genstats.py
src/flag_dnn/ops/batch_norm_backward.py
src/flag_dnn/ops/layer_norm_backward.py
src/flag_dnn/ops/rms_norm_backward.py
src/flag_dnn/ops/instance_norm.py
src/flag_dnn/ops/instance_norm_backward.py
src/flag_dnn/ops/adalayer_norm.py
src/flag_dnn/ops/adalayer_norm_backward.py
```

`matmul.py`、`conv_fprop.py`、`conv3d.py`、`reduction.py`、`causal_conv1d.py` 已存在；后续是在现有文件上补动态 K override、低精度专用 kernel 或更多 cuDNN shape/layout 边界。

### P2：Transformer 核心

```text
src/flag_dnn/ops/sdpa.py
src/flag_dnn/ops/sdpa_backward.py
src/flag_dnn/graph/kernels/sdpa.py
src/flag_dnn/graph/passes/attention.py
tests_graph/test_graph_sdpa.py
```

### P3：低精度

```text
src/flag_dnn/ops/block_scale_quantize.py
src/flag_dnn/ops/block_scale_dequantize.py
src/flag_dnn/ops/sdpa_fp8.py
src/flag_dnn/ops/sdpa_fp8_backward.py
src/flag_dnn/ops/sdpa_mxfp8.py
src/flag_dnn/ops/sdpa_mxfp8_backward.py
```

### P4：SM100/MoE 专用

```text
src/flag_dnn/ops/gemm_amax.py
src/flag_dnn/ops/gemm_srelu.py
src/flag_dnn/ops/gemm_dsrelu.py
src/flag_dnn/ops/gemm_swiglu.py
src/flag_dnn/ops/moe_grouped_matmul.py
src/flag_dnn/ops/moe_grouped_matmul_bwd.py
src/flag_dnn/ops/grouped_gemm.py
src/flag_dnn/ops/discrete_grouped_gemm.py
```

`rmsnorm_rht_amax.py` 已有 Triton 版本；若要追 SM100/CUTE 官方 wrapper 性能，建议在现文件补专用 kernel 路径。

## 6. Graph 接入模板

新增一个 eager op 后，graph 侧需要三处同步：

1. `src/flag_dnn/graph/registry.py`
   - `OpSchema(name, normalize_fn, shape_fn, run_fn, fusible=...)`
   - shape/dtype 推导必须准确；
   - multi-output op 已由 `Graph.add_op` 和 `OpSchema.num_outputs` 支持；新增时 `shape_fn` 要返回对应数量的 `TensorSpec`，`run_fn` 要返回等长 tuple/list。

2. `src/flag_dnn/graph/wrappers.py`
   - 把 op 名加入 `GRAPH_AWARE_OPS`。
   - 如果 eager API 返回 dict，要在 `_DICT_OUTPUT_OPS` 登记输出 key。

3. `src/flag_dnn/graph/backend.py` / `kernel_selector.py`
   - 对 fused op 或高性能 op 注册 `KernelCandidate`；
   - 加入 dtype/layout/shape constraints；
   - 后续 autotune 会在候选 plan 中选择。

建议 op schema 例子：

```python
register_op(
    OpSchema(
        name="sigmoid",
        normalize_fn=_normalize_unary(
            "sigmoid", ("compute_data_type", "name")
        ),
        shape_fn=_shape_like_first,
        run_fn=_run_sigmoid(flag_ops),
        fusible=True,
    )
)
```

## 7. cuDNN 对标测试模板

cuDNN 对标测试优先只比较 FlagDNN 和 cuDNN，避免把 Torch 混入同一个断言目标。若当前 cuDNN 环境没有 valid engine、Python API 未暴露该 op，或 FlagDNN 扩展了 cuDNN 当前不可执行的模式，可以保留单独的 Torch reference 测试，并在测试名/注释里标清这是 fallback coverage。

建议结构：

```text
tests_graph/
  test_<op>.py              # FlagDNN graph op vs cudnn pygraph / fallback reference
  test_graph_<pattern>.py   # FlagDNN graph pattern/fusion vs cudnn pygraph
```

测试断言：

1. cuDNN graph 构造成功，否则 `pytest.skip`。
2. cuDNN 可执行时，FlagDNN 输出和 cuDNN 输出 close。
3. cuDNN 不可执行但需要保留 coverage 时，用独立 Torch-reference 测试覆盖 FlagDNN 语义。
4. graph pattern 测试额外断言 `compiled.graph.nodes` 的 op type，确认 fusion 真的发生。

## 8. 推荐落地顺序

当前已完成 utility 主体、基础 pointwise/binary/compare/logical、`matmul`、`conv_fprop`、`reduction`、norm forward 和部分 SM100 wrapper 验证。后续建议按以下顺序继续：

1. 清理 graph alias hygiene 和边界 coverage，继续同步 `tests_graph` 与 `benchmark_graph`。
2. 补 activation/norm/conv backward，尤其是 norm backward 和 `conv_dgrad/wgrad`。
3. SDPA forward：建立 transformer 对标核心。
4. SDPA backward + variable length/paged/sliding-window：进入训练和复杂 attention graph。
5. block-scale quant/dequant + FP8/MXFP8 SDPA：低精度路线。
6. SM100 grouped GEMM/MoE：专用高性能路线。

## 9. SDPA 开发文件清单与职责

当前 SDPA 的实现闭环不是单个 `ops/sdpa.py` 文件，而是 eager Triton kernel、graph schema、graph 编译期 fast path、调参配置、测试和 benchmark 一起维护。开发或扩展一个 SDPA 相关能力时，按下面清单同步。

### 9.1 Forward：`sdpa`

| 文件 | 需要实现/维护的内容 |
|---|---|
| `src/flag_dnn/ops/sdpa.py` | eager forward 主实现。包含 cuDNN-style API 参数、输入/ dtype / device 校验、`attn_scale` 默认值、`generate_stats/is_inference` 解析、causal/sliding-window/diagonal band 解析、dropout 限制、bias 检查、输出和 stats 分配，以及 Triton kernel 调度。当前 forward kernel 包括 generic `_sdpa_fwd_kernel`、D128 dense exact fast path、GQA causal fast path、decode split/combine fast path。新增 mask、layout、dtype 或 fast path 时首先改这里。 |
| `src/flag_dnn/ops/__init__.py` | 从 `flag_dnn.ops.sdpa` 导入 `sdpa`，并加入 `__all__`，保证 `flag_dnn.ops.sdpa` 可直接使用。 |
| `src/flag_dnn/__init__.py` | 顶层导入并导出 `sdpa`。如果需要参与 torch.library 覆盖，还要同步注册表；graph wrapper 安装依赖这里的全局符号。 |
| `src/flag_dnn/graph/registry.py` | graph op 接入。`_normalize_sdpa` 负责把 cuDNN/Python 参数标准化成 graph attrs，并把 `q/k/v/bias` 转成 graph value；`_sdpa_shape` 负责输出和可选 stats 的 shape/dtype 推导；`_run_sdpa` 负责 executor fallback 调用 eager op；`OpSchema(name="sdpa", num_outputs=2, fusible=True)` 负责注册。新增参数时这里必须和 eager API 同步。 |
| `src/flag_dnn/graph/wrappers.py` | 把 `"sdpa"` 放进 `GRAPH_AWARE_OPS`，保证 `@flag_dnn.graph` capture 时调用的是 graph op，而不是直接执行 eager op。 |
| `src/flag_dnn/graph/prepared_ops.py` | graph 编译期 runner。`prepare_run_fn` 路由到 `_prepare_sdpa`；`_prepare_sdpa` 在 shape/stride/attrs 静态可知时绑定 scalar 参数、选择 fast path、缓存已编译 kernel 和 grid，运行时只分配输出并 launch kernel。当前 forward prepared path 覆盖 decode、GQA causal、dense exact 和 generic；输入 stride 或 bias stride 不匹配时 fallback 到 registry 的默认 run_fn。新增 forward fast path 时这里要和 `ops/sdpa.py` 的 eager dispatch 保持一致。 |
| `src/flag_dnn/runtime/backend/_nvidia/tune_configs.yaml` | NVIDIA backend autotune 配置。forward 相关 key 当前包括 `sdpa`、`sdpa_dense_exact`、`sdpa_gqa_causal`、`sdpa_decode`。新增 `@runtime.heuristics`/`@runtime.autotune` kernel 或调整 meta 参数时，要同步新增或更新这里的配置。 |

### 9.2 Backward：`sdpa_backward`

| 文件 | 需要实现/维护的内容 |
|---|---|
| `src/flag_dnn/ops/sdpa_backward.py` | eager backward 主实现。包含 unsupported 参数拒绝、输入/输出/stats/dBias 校验、band/causal 解析、dQ/dK/dV/dBias 输出分配，以及 backward Triton kernel。当前包含普通分阶段 dQ/dK/dV/dBias kernel、fused atomic dense/causal/GQA fast path、triangular causal fast path、exact causal/delta fast path、zero/delta 预处理 kernel。新增 backward 语义或训练 fast path 时首先改这里。 |
| `src/flag_dnn/ops/__init__.py`、`src/flag_dnn/__init__.py` | 导入并导出 `sdpa_backward`，必要时同步 torch.library 注册。 |
| `src/flag_dnn/graph/registry.py` | `_normalize_sdpa_backward` 对齐 eager/cudnn 参数，处理 `bias/dBias` 可选输入并拒绝暂不支持的 seq_len/dropout/score callback/sink token 等；`_sdpa_backward_shape` 推导 `dQ/dK/dV`；`_run_sdpa_backward` 调 eager backward；`OpSchema(name="sdpa_backward", num_outputs=3)` 注册。 |
| `src/flag_dnn/graph/wrappers.py` | 把 `"sdpa_backward"` 放进 `GRAPH_AWARE_OPS`。 |
| `src/flag_dnn/graph/prepared_ops.py` | `_prepare_sdpa_backward` 做 graph 编译期 fast path：静态绑定 shape/stride/attrs，选择 fused atomic、causal、GQA、triangular、exact/delta 等路径，缓存 kernel/grid，并在运行时 fallback 到默认 graph run_fn。新增 backward kernel 时要同步 eager dispatch 和 prepared dispatch。 |
| `src/flag_dnn/runtime/backend/_nvidia/tune_configs.yaml` | backward 相关 key 当前包括 `sdpa_backward_fused_atomic`、`sdpa_backward_fused_atomic_causal`、`sdpa_backward_fused_atomic_causal_d128`、`sdpa_backward_fused_atomic_gqa_causal_d128`、`sdpa_backward_zero_delta`、`sdpa_backward_dq`、`sdpa_backward_dkdv`、`sdpa_backward_dk`、`sdpa_backward_dv`。 |

### 9.3 测试、benchmark 和用例集合

| 文件 | 需要实现/维护的内容 |
|---|---|
| `tests_graph/test_graph_sdpa.py` | forward cuDNN graph 对标测试。覆盖默认 scale、`generate_stats`、inference 单输出、GQA decode、GQA causal D128、causal/top-left、bottom-right causal、sliding window、bias、fp32 bias torch fallback、显式 `attn_scale`、causal shorthand 等。新增 forward 语义时补这里。 |
| `tests_graph/test_graph_sdpa_backward.py` | backward cuDNN graph 对标测试。覆盖 dense backward、causal backward、bias + dBias 等，并断言 graph capture 后节点类型是 `sdpa_backward`。新增 backward 语义时补这里。 |
| `tests_graph/consts.py` | graph 正确性用例集合。forward 主要维护 `SDPA_CASES`、`SDPA_MASKED_CASES`；新增 shape、GQA/MQA、D128、seq_q/seq_kv 边界时同步这里。 |
| `benchmark_graph/test_sdpa_perf.py` | forward graph 性能对标。构造 cuDNN graph runner 和 FlagDNN compiled graph runner，断言节点是 `sdpa`，并按 benchmark consts 跑 dtype/shape。新增 forward fast path 后应加性能覆盖。 |
| `benchmark_graph/test_sdpa_backward_perf.py` | backward graph 性能对标。用 `flag_dnn.sdpa(..., generate_stats=True)` 生成 backward 输入，比较 cuDNN graph 和 FlagDNN compiled graph，断言节点是 `sdpa_backward`。新增 backward fast path 后应加性能覆盖。 |
| `benchmark_graph/consts.py` | benchmark shape 集合和编译选项。新增性能场景时同步 `SDPA_SHAPES`、`SDPA_BACKWARD_SHAPES` 等。 |

### 9.4 开发顺序建议

1. 先在 `ops/sdpa.py` 或 `ops/sdpa_backward.py` 做 eager 语义和 Triton kernel，确保输入校验、输出 shape、stats/dBias 行为和 cuDNN 语义对齐。
2. 再同步 `ops/__init__.py`、顶层 `__init__.py`、`graph/wrappers.py`，保证 eager 和 graph capture 都能看到新 op。
3. 在 `graph/registry.py` 补 normalize、shape、run_fn 和 `OpSchema`，特别注意 multi-output 的 `num_outputs`、可选输入顺序、attrs 名称和 fallback 行为。
4. 对热点场景在 `graph/prepared_ops.py` 加 prepared fast path，把 shape/stride/attrs、kernel constexpr、grid 和缓存逻辑静态化；不满足约束时必须 fallback 到默认 run_fn。
5. 给新 kernel 在 `runtime/backend/_nvidia/tune_configs.yaml` 加 tune key，并让 eager dispatch 和 prepared dispatch 使用同一组配置名。
6. 最后补 `tests_graph` 正确性、`benchmark_graph` 性能和 consts shape 集合；每个新语义至少要覆盖 eager graph capture、cuDNN 对标或明确的 torch fallback，以及 compiled graph 节点断言。
