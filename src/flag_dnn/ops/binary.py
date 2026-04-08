import logging
from typing import Union, Optional

import torch
import triton
import triton.language as tl

from flag_dnn import runtime
from flag_dnn.runtime import torch_device_fn
from flag_dnn.utils import libentry, libtuner
from flag_dnn.utils import triton_lang_extension as tle


logger = logging.getLogger(__name__)


# 维度坍缩：合并连续维度，丢弃 size=1 的维度，将任意 N 维化简为最小维度
def collapse_dims(shape, strides_a, strides_b):
    if not shape:
        return [1], [0], [0]
        
    c_shape, c_str_a, c_str_b = [], [], []

    # 从内向外 (从右向左) 遍历维度
    for i in reversed(range(len(shape))):
        s = shape[i]
        
        # 直接丢弃所有大小为 1 的维度，因为它对内存偏移的贡献是 0
        if s == 1:
            continue

        if not c_shape:
            # 初始化最内层维度
            c_shape.append(s)
            c_str_a.append(strides_a[i])
            c_str_b.append(strides_b[i])
        else:
            prev_shape = c_shape[-1]
            # 判断当前维度与前一个维度在内存上是否连续
            # 连续的条件：当前维度的 stride == 前一个维度的 stride * 前一个维度的 size
            is_contig_a = (strides_a[i] == c_str_a[-1] * prev_shape)
            is_contig_b = (strides_b[i] == c_str_b[-1] * prev_shape)

            if is_contig_a and is_contig_b:
                # 坍缩，将当前维度乘入上一个维度，stride 保持为最内层 stride
                c_shape[-1] *= s
            else:
                # 无法连续，作为一个新的独立维度加入
                c_shape.append(s)
                c_str_a.append(strides_a[i])
                c_str_b.append(strides_b[i])

    if not c_shape: # 如果全都是 1
        return [1], [0], [0]

    # 因为是从右向左遍历，最后需要翻转回来
    return c_shape[::-1], c_str_a[::-1], c_str_b[::-1]


def pad_to_max_dims(shape, strides_a, strides_b, max_dims=6):
    # 将坍缩后的 shape/strides 填充到固定的 max_dims，以便传入 Triton
    shape = list(shape)
    strides_a = list(strides_a)
    strides_b = list(strides_b)
    
    if len(shape) > max_dims:
        raise RuntimeError(f"坍缩后依然超过 {max_dims} 维，Not Support.")
        
    # 在最外层(左侧)填充 size=1, stride=0
    while len(shape) < max_dims:
        shape.insert(0, 1)
        strides_a.insert(0, 0)
        strides_b.insert(0, 0)
        
    return shape, strides_a, strides_b


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("binary"),
    key=["n_elements"],
    strategy=["align32"],
    warmup=5,
    rep=10,
)
@triton.jit
def binary_tensor_kernel(
    x_ptr, y_ptr, out_ptr,
    n_elements,
    alpha_val,
    ROUND_MODE: tl.constexpr,
    OP_TYPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    if OP_TYPE == 'add':
        res = x + alpha_val * y
    elif OP_TYPE == 'sub':
        res = x - alpha_val * y
    elif OP_TYPE == 'mul':
        res = x * y
    elif OP_TYPE == 'div':
        res = x / y
        if ROUND_MODE == 1:
            res = tl.where(res >= 0, tl.math.floor(res), tl.math.ceil(res))
        elif ROUND_MODE == 2:
            res = tl.math.floor(res)
    elif OP_TYPE == 'eq':
        res = x == y
    elif OP_TYPE == 'ne':
        res = x != y

    # 结果强制转换回输出张量的目标类型，防止隐式提升导致的错误
    tl.store(out_ptr + offsets, res.to(out_ptr.dtype.element_ty), mask=mask)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("binary"),
    key=["n_elements"],
    strategy=["align32"],
    warmup=5,
    rep=10,
)
@triton.jit
def binary_scalar_kernel(
    x_ptr, out_ptr,
    n_elements,
    other_val, alpha_val,
    ROUND_MODE: tl.constexpr,
    OP_TYPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    
    if OP_TYPE == 'add':
        res = x + alpha_val * other_val
    elif OP_TYPE == 'sub':
        res = x - alpha_val * other_val
    elif OP_TYPE == 'mul':
        res = x * other_val
    elif OP_TYPE == 'div':
        res = x / other_val
        if ROUND_MODE == 1:
            res = tl.where(res >= 0, tl.math.floor(res), tl.math.ceil(res))
        elif ROUND_MODE == 2:
            res = tl.math.floor(res)
    elif OP_TYPE == 'eq':
        res = x == other_val
    elif OP_TYPE == 'ne':
        res = x != other_val

    tl.store(out_ptr + offsets, res.to(out_ptr.dtype.element_ty), mask=mask)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("binary"),
    key=["n_elements"],
    strategy=["align32"],
    warmup=5,
    rep=10,
)
@triton.jit
def binary_broadcast_tensor_kernel(
    x_ptr, y_ptr, out_ptr, n_elements,
    # 填充后的 6D 形状
    s1, s2, s3, s4, s5,
    # X 的 6D Strides
    sx0, sx1, sx2, sx3, sx4, sx5,
    # Y 的 6D Strides
    sy0, sy1, sy2, sy3, sy4, sy5,
    alpha_val,
    ROUND_MODE: tl.constexpr,
    OP_TYPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # 坐标还原（从内向外剥洋葱）
    # 由于做了坍缩，许多 sX 实际上是 1，Triton 编译器遇到 x % 1 或 x // 1 会直接优化掉?
    idx5 = offsets % s5
    rem4 = offsets // s5
    
    idx4 = rem4 % s4
    rem3 = rem4 // s4
    
    idx3 = rem3 % s3
    rem2 = rem3 // s3
    
    idx2 = rem2 % s2
    rem1 = rem2 // s2
    
    idx1 = rem1 % s1
    idx0 = rem1 // s1

    # 计算物理偏移并加载数据
    x_off = (idx0 * sx0 + idx1 * sx1 + idx2 * sx2 + 
             idx3 * sx3 + idx4 * sx4 + idx5 * sx5)
    y_off = (idx0 * sy0 + idx1 * sy1 + idx2 * sy2 + 
             idx3 * sy3 + idx4 * sy4 + idx5 * sy5)

    x = tl.load(x_ptr + x_off, mask=mask)
    y = tl.load(y_ptr + y_off, mask=mask)

    if OP_TYPE == 'add':
        res = x + alpha_val * y
    elif OP_TYPE == 'sub':
        res = x - alpha_val * y
    elif OP_TYPE == 'mul':
        res = x * y
    elif OP_TYPE == 'div':
        res = x / y
        # 处理舍入模式 (0: None, 1: trunc, 2: floor)
        if ROUND_MODE == 1:
            # trunc (向零取整): 正数向下取整，负数向上取整
            res = tl.where(res >= 0, tl.math.floor(res), tl.math.ceil(res))
        elif ROUND_MODE == 2:
            # floor (向下取整)
            res = tl.math.floor(res)
    elif OP_TYPE == 'eq':
        res = x == y
    elif OP_TYPE == 'ne':
        res = x != y
    
    tl.store(out_ptr + offsets, res.to(out_ptr.dtype.element_ty), mask=mask)


'''
计算类型采用原类型，因此float16，bfloat16可能存在精度问题，尤其div op，可在triton kernel中提升精度计算
'''
def binary(
    input: torch.Tensor,
    other: Union[torch.Tensor, int, float, bool],
    *,
    alpha: float = 1.0,
    rounding_mode: Optional[str] = None,
    out: Optional[torch.Tensor] = None,
    op_type: str = ""
) -> torch.Tensor:
    logger.debug(f"FLAG_DNN {op_type.upper()})")

    if op_type not in ["eq", "ne"] and type(other) is bool:
        raise RuntimeError(f"当 other 是 bool 类型时，仅支持 eq/ne 操作，但 op_type={op_type}")

    if not input.is_contiguous():
        assert False, "input must be contiguous."
        input = input.contiguous()

    is_other_tensor = isinstance(other, torch.Tensor)
    out_shape = torch.broadcast_shapes(input.shape, other.shape) if is_other_tensor else input.shape

    mode_idx = 0
    if op_type == "div":
        mode_map = {None: 0, 'trunc': 1, 'floor': 2}
        if rounding_mode not in mode_map:
            raise RuntimeError(f"div expected rounding_mode to be one of None, 'trunc', 'floor' but found {rounding_mode}")
        mode_idx = mode_map[rounding_mode]

    # Type promotion
    if op_type == "add":
        dummy_in = input.new_empty((0,))
        dummy_oth = other.new_empty((0,)) if is_other_tensor else other
        out_dtype = (dummy_in + alpha * dummy_oth).dtype
    elif op_type == "sub":
        dummy_in = input.new_empty((0,))
        dummy_oth = other.new_empty((0,)) if is_other_tensor else other
        out_dtype = (dummy_in - alpha * dummy_oth).dtype
    elif op_type == "mul":
        dummy_in = input.new_empty((0,))
        dummy_oth = other.new_empty((0,)) if is_other_tensor else other
        out_dtype = (dummy_in * dummy_oth).dtype
    elif op_type == "div":
        dummy_in = input.new_empty((0,))
        dummy_oth = other.new_empty((0,)) if is_other_tensor else other
        out_dtype = torch.div(dummy_in, dummy_oth, rounding_mode=rounding_mode).dtype
    elif op_type == "eq" or op_type == "ne":
        out_dtype = torch.bool
    else:
        raise RuntimeError(f"Unsupported OP_TYPE={op_type} in binary")

    if out is None:
        out = torch.empty(out_shape, dtype=out_dtype, device=input.device)

    n_elements = out.numel()
    if n_elements == 0:
        return out

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )

    if is_other_tensor:
        # 形状一致且连续，走一维 Kernel
        if input.shape == other.shape and input.is_contiguous() and other.is_contiguous():
            binary_tensor_kernel[grid](
                input, other, out,
                n_elements, float(alpha),
                ROUND_MODE=mode_idx,
                OP_TYPE=op_type
            )
        # broadcast
        else:
            # 仅逻辑扩展，不触发显存复制
            in_exp = input.expand(out_shape)
            oth_exp = other.expand(out_shape)
            
            # 维度坍缩
            c_shape, c_sx, c_sy = collapse_dims(out_shape, in_exp.stride(), oth_exp.stride())
            
            # 填充到 6 维
            f_shape, f_sx, f_sy = pad_to_max_dims(c_shape, c_sx, c_sy, max_dims=6)
            
            binary_broadcast_tensor_kernel[grid](
                input, other, out, n_elements,
                *f_shape[1:], # 传入 s1 到 s5
                *f_sx,        # 传入 sx0 到 sx5
                *f_sy,        # 传入 sy0 到 sy5
                float(alpha),
                ROUND_MODE=mode_idx,
                OP_TYPE=op_type
            )
    else:
        if op_type == "eq":
            other_val = torch.tensor(other, dtype=input.dtype).item()
        else:
            other_val = float(other)
        binary_scalar_kernel[grid](
            input, out, n_elements, other_val, float(alpha),
            ROUND_MODE=mode_idx,
            OP_TYPE=op_type
        )

    return out