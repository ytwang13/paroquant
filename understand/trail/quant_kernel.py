"""
FP8 and NVFP4 quantization kernels (torch + triton).

Formats
-------
- ``fp8_e4m3`` / ``fp8_e5m2``: fake quant — explicit E(x)M(y) grid in fp16/fp32.
- ``e4m3`` / ``e5m2``: direct grid — scale then ``.to(float8_*)`` cast
  (COAT real quant: https://github.com/NVlabs/COAT/blob/main/coat/activation/real_quantization/_quantize_perblock.py).
- ``nvfp4`` / ``nvfp4_plus``: fake quant only (no native dtype).

Per-channel: one scale per slice along ``axis``.
Per-block: COAT-style column blocks (``row_block=1``, ``column_block=block_size``).
"""

from __future__ import annotations

from typing import Literal, Tuple

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

Backend = Literal["torch", "triton"]


@triton.jit
def _round_half_to_even(x):
    """Match ``torch.round`` (banker's rounding) for non-negative ``x``."""
    int_part = tl.floor(x)
    frac = x - int_part
    even = (tl.cast(int_part, tl.int32) & 1) == 0
    return tl.where(
        frac > 0.5,
        int_part + 1.0,
        tl.where(frac < 0.5, int_part, tl.where(even, int_part, int_part + 1.0)),
    )

# (e_bits, m_bits) for fake-quant E(x)M(y) grid
FP8_E4M3 = (4, 3)
FP8_E5M2 = (5, 2)
NVFP4_E2M1 = (2, 1)

# Native FP8 dtypes + representable max (COAT real_quant/common.py)
FP8_CAST_DTYPE = {
    "e4m3": torch.float8_e4m3fn,
    "e5m2": torch.float8_e5m2,
}
FP8_CAST_MAX = {
    "e4m3": 448.0,
    "e5m2": 57344.0,
}

SCALE_MIN_THRES = 1e-10

# kind: "exmy" | "cast" | "nv"
_FORMAT_SPECS = {
    "fp8_e4m3": ("exmy", FP8_E4M3, "none"),
    "fp8_e5m2": ("exmy", FP8_E5M2, "none"),
    "e4m3": ("cast", None, "none"),
    "e5m2": ("cast", None, "none"),
    "nvfp4": ("nv", NVFP4_E2M1, "e4m3"),
    "nvfp4_plus": ("nv", NVFP4_E2M1, "e4m3_plus"),
}


def _qp_symmetric(e_bit: int, m_bit: int) -> float:
    # quantization positive?
    qp = (2 - 2 ** (-m_bit)) * (2 ** (2 ** (e_bit - 1)))
    if e_bit == 4 and m_bit == 3:
        qp = 448.0
    elif e_bit == 5 and m_bit == 2:
        qp = 57344.0
    return qp


def _parse_format(fmt: str) -> Tuple[str, str, Tuple[int, int] | None, str]:
    if fmt not in _FORMAT_SPECS:
        raise ValueError(f"Unknown format {fmt!r}; choose from {list(_FORMAT_SPECS)}")
    kind, em_bits, scale_mode = _FORMAT_SPECS[fmt]
    return fmt, kind, em_bits, scale_mode


# ---------------------------------------------------------------------------
# Direct FP8 grid cast (scale + native dtype), COAT real-quant style
# ---------------------------------------------------------------------------


def fp8_grid_cast_torch(x: torch.Tensor, fmt: str) -> torch.Tensor:
    """Round to FP8 grid via dtype cast; output dtype matches ``x``."""
    if fmt not in FP8_CAST_DTYPE:
        raise ValueError(f"fp8_grid_cast requires {list(FP8_CAST_DTYPE)}; got {fmt!r}")
    fp8_dtype = FP8_CAST_DTYPE[fmt]
    return x.to(fp8_dtype).to(x.dtype)


@triton.jit
def _fp8_cast_quantize_perblock_kernel(
    output_ptr,
    output_scale_ptr,
    input_ptr,
    M,
    N,
    SM,
    SN,
    QB: tl.constexpr,
    fp8_max,
    input_stride_0,
    input_stride_1,
    output_stride_0,
    output_stride_1,
    s_output_stride_0,
    s_output_stride_1,
    SCALE_MIN_THRES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    num_block_n = tl.cdiv(N, BLOCK_N)
    pid_dim0 = pid // num_block_n
    pid_dim1 = pid % num_block_n

    input_block_ptr = tl.make_block_ptr(
        base=input_ptr,
        shape=(M, N),
        strides=(input_stride_0, input_stride_1),
        offsets=(pid_dim0 * BLOCK_M, pid_dim1 * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    inp = tl.load(input_block_ptr).to(tl.float32)
    abs_out = tl.abs(inp)
    max_val = tl.max(abs_out) + SCALE_MIN_THRES
    scale = max_val / fp8_max
    out = (inp / scale).to(output_ptr.dtype.element_ty)

    output_block_ptr = tl.make_block_ptr(
        base=output_ptr,
        shape=(M, N),
        strides=(output_stride_0, output_stride_1),
        offsets=(pid_dim0 * BLOCK_M, pid_dim1 * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    scale_ptr = tl.make_block_ptr(
        base=output_scale_ptr,
        shape=(SM, SN),
        strides=(s_output_stride_0, s_output_stride_1),
        offsets=(pid_dim0, pid_dim1),
        block_shape=(1, 1),
        order=(1, 0),
    )
    tl.store(output_block_ptr, out)
    tl.store(scale_ptr, scale.to(output_scale_ptr.dtype.element_ty))


def _fp8_cast_quantize_perblock_triton(
    x: torch.Tensor,
    fmt: str,
    block_size: int,
    scale_dtype: torch.dtype | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per ``QB×QB`` block: scale = max|x|/fp8_max, store ``float8`` + scales."""
    if x.dim() == 3:
        x = x.reshape(-1, x.shape[-1])
    if x.dim() != 2:
        raise ValueError(f"Expected 2D input, got shape {x.shape}")
    m, n = x.shape
    if m % block_size != 0 or n % block_size != 0:
        raise ValueError(
            f"Shape {x.shape} must be divisible by block_size={block_size}"
        )
    sm, sn = m // block_size, n // block_size
    if scale_dtype is None:
        scale_dtype = x.dtype
    fp8_dtype = FP8_CAST_DTYPE[fmt]
    y = torch.empty((m, n), dtype=fp8_dtype, device=x.device)
    s_y = torch.empty((sm, sn), dtype=scale_dtype, device=x.device)
    grid = lambda meta: (
        triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),
    )
    _fp8_cast_quantize_perblock_kernel[grid](
        y,
        s_y,
        x,
        m,
        n,
        sm,
        sn,
        block_size,
        FP8_CAST_MAX[fmt],
        x.stride(0),
        x.stride(1),
        y.stride(0),
        y.stride(1),
        s_y.stride(0),
        s_y.stride(1),
        SCALE_MIN_THRES=SCALE_MIN_THRES,
        BLOCK_M=block_size,
        BLOCK_N=block_size,
    )
    return y, s_y


def _dequant_fp8_perblock(
    y_fp8: torch.Tensor,
    scales: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Broadcast block scales and dequantize FP8 tile to float."""
    m, n = y_fp8.shape
    sm, sn = scales.shape
    y = y_fp8.float().reshape(sm, block_size, sn, block_size)
    s = scales[:, None, :, None].to(y.dtype)
    return (y * s).reshape(m, n)


def _fp8_cast_quantize_perblock_torch(
    x: torch.Tensor,
    fmt: str,
    block_size: int,
    epsilon: float = SCALE_MIN_THRES,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """COAT-style ``QB×QB`` tile quantize via torch cast (reference for triton)."""
    if x.dim() == 3:
        x = x.reshape(-1, x.shape[-1])
    m, n = x.shape
    sm, sn = m // block_size, n // block_size
    fp8_dtype = FP8_CAST_DTYPE[fmt]
    fp8_max = FP8_CAST_MAX[fmt]
    y_fp8 = torch.empty((m, n), dtype=fp8_dtype, device=x.device)
    scales = torch.empty((sm, sn), dtype=x.dtype, device=x.device)
    for i in range(sm):
        for j in range(sn):
            block = x[
                i * block_size : (i + 1) * block_size,
                j * block_size : (j + 1) * block_size,
            ]
            scale = block.abs().amax() + epsilon
            scale = scale / fp8_max
            scales[i, j] = scale
            y_fp8[
                i * block_size : (i + 1) * block_size,
                j * block_size : (j + 1) * block_size,
            ] = (block / scale).to(fp8_dtype)
    rq = _dequant_fp8_perblock(y_fp8, scales, block_size)
    return rq, scales


# ---------------------------------------------------------------------------
# Element-wise float E(x)M(y) quantize (torch / triton) — fake quant
# ---------------------------------------------------------------------------


def float_exmy_quantize_torch(
    x: torch.Tensor,
    e_bit: int,
    m_bit: int,
    stochastic: bool = False,
    ceil: bool = False,
) -> torch.Tensor:
    sign, x_abs = x.sign(), x.abs()
    elow = -(2 ** (e_bit - 1)) + 2
    ehigh = 2 ** (e_bit - 1)
    mhigh = 2**m_bit
    expo = torch.floor(torch.log2(x_abs.clamp(min=0.0)))
    expo = torch.clamp(expo, min=elow, max=ehigh)
    mant = x_abs / torch.exp2(expo)
    mant_int = torch.floor(mant)
    mant_frac = (mant - mant_int) * mhigh
    if stochastic:
        mant_frac = mant_frac + torch.empty_like(mant_frac).uniform_(-0.5, 0.5)
    if ceil:
        mant_frac = torch.ceil(mant_frac)
    else:
        mant_frac = torch.round(mant_frac)
    mant_q = mant_int + mant_frac / mhigh
    return (sign * torch.exp2(expo) * mant_q).to(x.dtype)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 2048}, num_stages=1),
    ],
    key=["n_elements"],
)
@triton.jit
def _float_exmy_quantize_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    e_bit,
    m_bit,
    BLOCK_SIZE: tl.constexpr,
):
    ebit = e_bit.value if isinstance(e_bit, tl.constexpr) else e_bit
    mbit = m_bit.value if isinstance(m_bit, tl.constexpr) else m_bit

    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)

    sign = 1.0 - 2.0 * libdevice.signbit(x)
    x_abs = tl.abs(x)
    elow = -tl.exp2((ebit - 1).to(tl.float32)) + 2.0
    ehigh = tl.exp2((ebit - 1).to(tl.float32))
    mhigh = tl.exp2(mbit.to(tl.float32))
    expo = tl.floor(tl.log2(x_abs))
    expo = tl.clamp(expo, min=elow, max=ehigh)
    mant = x_abs / tl.exp2(expo)
    mant_int = tl.floor(mant)
    mant_frac = _round_half_to_even((mant - mant_int) * mhigh)
    mant_q = mant_int + mant_frac / mhigh
    y = sign * tl.exp2(expo) * mant_q
    tl.store(output_ptr + offsets, y.to(x_ptr.dtype.element_ty), mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 2048}, num_stages=1),
    ],
    key=["n_elements"],
)
@triton.jit
def _float_exmy_stochastic_quantize_kernel(
    x_ptr,
    noise_ptr,
    output_ptr,
    n_elements,
    e_bit,
    m_bit,
    BLOCK_SIZE: tl.constexpr,
):
    ebit = e_bit.value if isinstance(e_bit, tl.constexpr) else e_bit
    mbit = m_bit.value if isinstance(m_bit, tl.constexpr) else m_bit

    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    noise = tl.load(noise_ptr + offsets, mask=mask)

    sign = 1.0 - 2.0 * libdevice.signbit(x)
    x_abs = tl.abs(x)
    elow = -tl.exp2((ebit - 1).to(tl.float32)) + 2.0
    ehigh = tl.exp2((ebit - 1).to(tl.float32))
    mhigh = tl.exp2(mbit.to(tl.float32))
    expo = tl.floor(tl.log2(x_abs))
    expo = tl.clamp(expo, min=elow, max=ehigh)
    mant = x_abs / tl.exp2(expo)
    mant_int = tl.floor(mant)
    mant_frac = _round_half_to_even((mant - mant_int) * mhigh + noise)
    mant_q = mant_int + mant_frac / mhigh
    y = sign * tl.exp2(expo) * mant_q
    tl.store(output_ptr + offsets, y.to(x_ptr.dtype.element_ty), mask=mask)


def float_exmy_quantize_triton(
    x: torch.Tensor,
    e_bit: int,
    m_bit: int,
    stochastic: bool = False,
) -> torch.Tensor:
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise NotImplementedError(f"Triton quantize does not support dtype {x.dtype}")
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    y = torch.empty_like(x)
    if stochastic:
        noise = x.new_empty(x.shape).uniform_(-0.5, 0.5)
        _float_exmy_stochastic_quantize_kernel[grid](
            x, noise, y, n_elements, e_bit, m_bit
        )
    else:
        _float_exmy_quantize_kernel[grid](x, y, n_elements, e_bit, m_bit)
    return y


def float_exmy_quantize(
    x: torch.Tensor,
    e_bit: int,
    m_bit: int,
    stochastic: bool = False,
    ceil: bool = False,
    backend: Backend = "torch",
) -> torch.Tensor:
    if backend == "torch":
        return float_exmy_quantize_torch(x, e_bit, m_bit, stochastic, ceil)
    if backend == "triton":
        if ceil:
            raise ValueError("Triton backend does not support ceil=True; use torch.")
        return float_exmy_quantize_triton(x, e_bit, m_bit, stochastic)
    raise ValueError(f"Unknown backend {backend!r}")


# ---------------------------------------------------------------------------
# Block layout helpers (COAT-style column blocks along the last dim)
# ---------------------------------------------------------------------------


def block_cut(
    input: torch.Tensor,
    row_block: int,
    column_block: int,
    pad_block: bool = False,
) -> torch.Tensor:
    original_shape = input.shape
    if input.dim() > 2:
        flat = input.reshape(-1, input.shape[-1])
    elif input.dim() == 2:
        flat = input
    else:
        raise ValueError(f"Expected rank >= 2, got {input.shape}")

    m, n = flat.shape
    if row_block == -1:
        row_block = m
    if column_block == -1:
        column_block = n

    if pad_block:
        row_pad = (row_block - m % row_block) % row_block
        col_pad = (column_block - n % column_block) % column_block
        flat = torch.nn.functional.pad(flat, (0, col_pad, 0, row_pad))
        m, n = flat.shape

    row_num, col_num = m // row_block, n // column_block
    return (
        flat.reshape(row_num, row_block, col_num, column_block)
        .permute(0, 2, 1, 3)
        .reshape(row_num * col_num, row_block, column_block)
    )


def block_reshape(
    blocks: torch.Tensor,
    origin: torch.Tensor,
    row_block: int,
    column_block: int,
    pad_block: bool = False,
) -> torch.Tensor:
    if origin.dim() > 2:
        flat_origin = origin.reshape(-1, origin.shape[-1])
    elif origin.dim() == 2:
        flat_origin = origin
    else:
        raise ValueError(f"Expected rank >= 2, got {origin.shape}")

    m, n = flat_origin.shape
    if row_block == -1:
        row_block = m
    if column_block == -1:
        column_block = n

    if pad_block:
        row_pad = (row_block - m % row_block) % row_block
        col_pad = (column_block - n % column_block) % column_block
        m_pad, n_pad = m + row_pad, n + col_pad
        row_num, col_num = m_pad // row_block, n_pad // column_block
    else:
        row_num, col_num = m // row_block, n // column_block

    out = (
        blocks.reshape(row_num, col_num, row_block, column_block)
        .permute(0, 2, 1, 3)
        .reshape(row_num * row_block, col_num * column_block)
    )
    out = out[:m, :n]
    if origin.dim() > 2:
        return out.reshape(origin.shape)
    return out


def _quantize_nv_scale(
    scale: torch.Tensor,
    scale_mode: str,
) -> torch.Tensor:
    if scale_mode == "none":
        return scale
    if scale_mode == "e4m3":
        return float_exmy_quantize_torch(scale, 4, 3, ceil=True)
    if scale_mode == "e4m3_plus":
        double_scale = scale.abs().amax().float() / 448.0
        q = float_exmy_quantize_torch(scale / double_scale, 4, 3, ceil=True)
        return q * double_scale
    raise ValueError(scale_mode)


def _symmetric_scale_from_absmax(
    absmax: torch.Tensor,
    fmt: str,
    kind: str,
    e_bit: int | None,
    m_bit: int | None,
    scale_mode: str,
    epsilon: float,
) -> torch.Tensor:
    if kind == "cast":
        scale = (absmax + epsilon) / FP8_CAST_MAX[fmt]
    else:
        assert e_bit is not None and m_bit is not None
        qp = _qp_symmetric(e_bit, m_bit)
        scale = (2.0 * absmax + epsilon) / (2.0 * qp)
    return _quantize_nv_scale(scale, scale_mode)


def _apply_value_quant(
    normalized: torch.Tensor,
    fmt: str,
    kind: str,
    e_bit: int | None,
    m_bit: int | None,
    stochastic: bool,
    backend: Backend,
) -> torch.Tensor:
    if kind == "cast":
        if backend == "triton":
            raise ValueError(
                "Triton backend for cast uses fp8_cast_quantize_perblock only; "
                "use quantize_per_block(..., backend='triton') or backend='torch'."
            )
        return fp8_grid_cast_torch(normalized, fmt)
    assert e_bit is not None and m_bit is not None
    if backend == "torch":
        return float_exmy_quantize_torch(normalized, e_bit, m_bit, stochastic)
    return float_exmy_quantize_triton(normalized, e_bit, m_bit, stochastic)


# ---------------------------------------------------------------------------
# Per-block and per-channel quantize
# ---------------------------------------------------------------------------


def quantize_per_block(
    x: torch.Tensor,
    fmt: str = "fp8_e4m3",
    block_size: int = 32,
    row_block: int = 1,
    stochastic: bool = False,
    epsilon: float = 1e-12,
    backend: Backend = "torch",
    pad_block: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Group-wise symmetric quantize with ``column_block=block_size`` (COAT layout).

    For ``e4m3``/``e5m2`` with ``backend='triton'``, uses the COAT per-block
    FP8 cast kernel (``QB×QB`` tiles; ``row_block`` must be 1 and both dims
  divisible by ``block_size``).

    Returns ``(dequantized_high_precision, scale_factors)``.
    """
    fmt, kind, em_bits, scale_mode = _parse_format(fmt)
    e_bit, m_bit = (em_bits if em_bits is not None else (None, None))

    if kind == "cast":
        if row_block != 1 or pad_block:
            raise ValueError("FP8 cast per-block uses QB×QB tiles; set row_block=1, pad_block=False")
        flat = x.reshape(-1, x.shape[-1]) if x.dim() > 2 else x
        if backend == "triton":
            y_fp8, s_tile = _fp8_cast_quantize_perblock_triton(flat, fmt, block_size)
        else:
            rq_flat, s_tile = _fp8_cast_quantize_perblock_torch(
                flat, fmt, block_size, epsilon
            )
            if x.dim() > 2:
                return rq_flat.reshape(x.shape), s_tile
            return rq_flat, s_tile
        rq_flat = _dequant_fp8_perblock(y_fp8, s_tile, block_size)
        if x.dim() > 2:
            return rq_flat.reshape(x.shape), s_tile
        return rq_flat, s_tile

    blocks = block_cut(x, row_block, block_size, pad_block=pad_block)
    absmax = blocks.abs().amax(dim=2, keepdim=True) + epsilon
    scales = _symmetric_scale_from_absmax(
        absmax, fmt, kind, e_bit, m_bit, scale_mode, 0.0
    )
    normalized = blocks / scales
    q_blocks = _apply_value_quant(
        normalized, fmt, kind, e_bit, m_bit, stochastic, backend
    )
    rq = block_reshape(q_blocks * scales, x, row_block, block_size, pad_block=pad_block)
    return rq, scales


def quantize_per_channel(
    x: torch.Tensor,
    fmt: str = "fp8_e4m3",
    axis: int = 0,
    stochastic: bool = False,
    epsilon: float = 1e-12,
    backend: Backend = "torch",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """One scale per slice along ``axis`` (absmax over all other dimensions)."""
    fmt, kind, em_bits, scale_mode = _parse_format(fmt)
    e_bit, m_bit = (em_bits if em_bits is not None else (None, None))
    if kind == "cast" and backend == "triton":
        raise ValueError("Cast formats use torch fp8 cast; set backend='torch'.")

    reduce_dims = tuple(i for i in range(x.dim()) if i != axis)
    absmax = x.abs().amax(dim=reduce_dims, keepdim=True) + epsilon
    scales = _symmetric_scale_from_absmax(
        absmax, fmt, kind, e_bit, m_bit, scale_mode, 0.0
    )
    normalized = x / scales
    q = _apply_value_quant(
        normalized, fmt, kind, e_bit, m_bit, stochastic, backend
    )
    return q * scales, scales


def quantize_per_tensor(
    x: torch.Tensor,
    fmt: str = "fp8_e4m3",
    stochastic: bool = False,
    epsilon: float = 1e-12,
    backend: Backend = "torch",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Single scale for the whole tensor (COAT per-tensor mode)."""
    fmt, kind, em_bits, scale_mode = _parse_format(fmt)
    e_bit, m_bit = (em_bits if em_bits is not None else (None, None))
    if kind == "cast" and backend == "triton":
        raise ValueError("Cast formats use torch fp8 cast; set backend='torch'.")

    absmax = x.abs().amax().reshape(()) + epsilon
    scales = _symmetric_scale_from_absmax(
        absmax, fmt, kind, e_bit, m_bit, scale_mode, 0.0
    )
    normalized = x / scales
    q = _apply_value_quant(
        normalized, fmt, kind, e_bit, m_bit, stochastic, backend
    )
    return q * scales, scales
