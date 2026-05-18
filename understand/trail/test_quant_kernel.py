"""
Correctness + speedup for quant kernels (single entry point).

Run:
  python understand/trail/test_quant_kernel.py
  python understand/trail/test_quant_kernel.py --m 4096 --n 4096 --block-size 128

Case ID pattern:
  {fmt}_{exp|cast}_{ptensor|pchan|blk{N}}_triton_vs_torch
  {fmt}_cast_{ptensor|pchan}_torch_only   (no Triton kernel; bench torch only)

SUMMARY columns:
  case_id | correct | mse_fp16 | rel_mse | max_err | torch_ms | triton_ms | speedup

``correct``: kernel check (triton vs torch, or cast smoke).
``mse_fp16`` / ``rel_mse`` / ``max_err``: reconstructed quant output vs original fp16 input.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch

TRAIL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TRAIL_DIR))

from quant_kernel import (  # noqa: E402
    quantize_per_block,
    quantize_per_channel,
    quantize_per_tensor,
)

CompareFn = Callable[[], None]
BenchFn = Callable[[], None]

# exp API formats
EXP_FP8 = [("fp8_e4m3", "fp8e4m3"), ("fp8_e5m2", "fp8e5m2")]
EXP_NV = [("nvfp4", "nvfp4"), ("nvfp4_plus", "nvfp4plus")]
CAST_FMT = [("e4m3", "e4m3"), ("e5m2", "e5m2")]

CAST_RQ_MEAN = {"e4m3": 1e-3, "e5m2": 2e-3}
CAST_RQ_MAX = {"e4m3": 128.0, "e5m2": 200.0}

# correctness tensor shapes (fixed)
_CORRECT_PTENSOR_SHAPE = (512, 512)
_CORRECT_PCHAN_SHAPE = (256, 1024)
_CORRECT_BLK_SHAPE = (4096, 4096)


@dataclass
class CaseResult:
    case_id: str
    correct: str
    correct_detail: str = ""
    mse_fp16: Optional[float] = None
    rel_mse_fp16: Optional[float] = None
    max_err_fp16: Optional[float] = None
    torch_ms: Optional[float] = None
    triton_ms: Optional[float] = None

    @property
    def speedup(self) -> Optional[float]:
        if self.torch_ms is None or self.triton_ms is None or self.triton_ms <= 0:
            return None
        return self.torch_ms / self.triton_ms


@dataclass
class KernelCase:
    case_id: str
    correct_fn: Optional[CompareFn] = None
    quality_fn: Optional[Callable[[], Dict[str, float]]] = None
    bench_torch: Optional[BenchFn] = None
    bench_triton: Optional[BenchFn] = None


def _eq(a: torch.Tensor, b: torch.Tensor) -> None:
    if not torch.allclose(a, b, rtol=0.0, atol=0.0, equal_nan=False):
        d = (a - b).abs()
        raise AssertionError(f"max={d.max().item():.4g} mean={d.mean().item():.4g}")


def _approx(a: torch.Tensor, b: torch.Tensor, *, mean: float, max_: float) -> None:
    d = (a - b).abs()
    if d.mean() >= mean or d.max() >= max_:
        raise AssertionError(f"max={d.max().item():.4g} mean={d.mean().item():.4g}")


def _cuda(*shape: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.manual_seed(0)
    # Build in fp32 so outliers stay finite, then cast (matches typical fp16 activation path).
    x = torch.randn(*shape, device="cuda", dtype=torch.float32)
    if len(shape) >= 2:
        x[:, min(100, shape[-1] - 1)] *= 500
        x[min(100, shape[0] - 1), :] *= 500
    if dtype == torch.float16:
        x = x.clamp(-65504.0, 65504.0)
    return x.to(dtype)


def _time_fn(fn: BenchFn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def _run_correct(fn: CompareFn) -> tuple[str, str]:
    try:
        fn()
        return "PASS", ""
    except Exception as e:
        return "FAIL", str(e)


def _quantize_rq(
    x: torch.Tensor,
    fmt: str,
    gran: str,
    *,
    block_size: int,
    backend: str = "torch",
) -> torch.Tensor:
    if gran == "ptensor":
        rq, _ = quantize_per_tensor(x, fmt=fmt, backend=backend)
    elif gran == "pchan":
        rq, _ = quantize_per_channel(x, fmt=fmt, axis=0, backend=backend)
    elif gran == "blk":
        rq, _ = quantize_per_block(x, fmt=fmt, block_size=block_size, backend=backend)
    else:
        raise ValueError(gran)
    return rq


def _metrics_vs_fp16_baseline(
    x_fp16: torch.Tensor,
    fmt: str,
    gran: str,
    block_size: int,
) -> Dict[str, float]:
    """Compare quant round-trip (torch, computed in fp32) against the fp16 baseline."""
    ref = x_fp16.float()
    rq = _quantize_rq(ref, fmt, gran, block_size=block_size, backend="torch")
    if not torch.isfinite(rq).all():
        raise ValueError("non-finite reconstructed tensor")
    out = rq.float()
    err = ref - out
    mse = err.pow(2).mean().item()
    var = ref.pow(2).mean().item()
    rel_mse = mse / max(var, 1e-12)
    return {
        "mse": mse,
        "rel_mse": rel_mse,
        "max_err": err.abs().max().item(),
    }


def _make_quality_fn(
    fmt: str,
    gran: str,
    m: int,
    n: int,
    block_size: int,
    baseline_dtype: torch.dtype,
) -> Callable[[], Dict[str, float]]:
    def _fn() -> Dict[str, float]:
        torch.manual_seed(0)
        x = _cuda(m, n, dtype=baseline_dtype)
        return _metrics_vs_fp16_baseline(x, fmt, gran, block_size)
    return _fn


def _exp_triton_vs_torch(
    api_fmt: str,
    gran: str,
    *,
    block_size: int = 32,
    axis: int = 0,
    shape: Optional[Tuple[int, ...]] = None,
) -> CompareFn:
    def _fn() -> None:
        if gran == "ptensor":
            x = _cuda(*(_CORRECT_PTENSOR_SHAPE if shape is None else shape))
            _eq(
                quantize_per_tensor(x, fmt=api_fmt, backend="torch")[0],
                quantize_per_tensor(x, fmt=api_fmt, backend="triton")[0],
            )
        elif gran == "pchan":
            x = _cuda(*(_CORRECT_PCHAN_SHAPE if shape is None else shape))
            _eq(
                quantize_per_channel(x, fmt=api_fmt, axis=axis, backend="torch")[0],
                quantize_per_channel(x, fmt=api_fmt, axis=axis, backend="triton")[0],
            )
        elif gran == "blk":
            x = _cuda(*(_CORRECT_BLK_SHAPE if shape is None else shape))
            _eq(
                quantize_per_block(x, fmt=api_fmt, block_size=block_size, backend="torch")[0],
                quantize_per_block(x, fmt=api_fmt, block_size=block_size, backend="triton")[0],
            )
        else:
            raise ValueError(gran)
    return _fn


def _cast_blk_correct(
    cast_fmt: str, m: int, n: int, block_size: int, dtype: torch.dtype
) -> CompareFn:
    def _fn() -> None:
        torch.manual_seed(0)
        x = _cuda(m, n, dtype=dtype)
        _, s_t = quantize_per_block(x, fmt=cast_fmt, block_size=block_size, backend="torch")
        _, s_r = quantize_per_block(x, fmt=cast_fmt, block_size=block_size, backend="triton")
        _eq(s_t, s_r)
        rq_t, _ = quantize_per_block(x, fmt=cast_fmt, block_size=block_size, backend="torch")
        rq_r, _ = quantize_per_block(x, fmt=cast_fmt, block_size=block_size, backend="triton")
        _approx(rq_t, rq_r, mean=CAST_RQ_MEAN[cast_fmt], max_=CAST_RQ_MAX[cast_fmt])
    return _fn


def _cast_torch_smoke(cast_fmt: str, gran: str, block_size: int) -> CompareFn:
    """Cast ptensor/pchan: torch-only (no Triton); correctness on fp32 (fp16 scales can overflow)."""

    def _fn() -> None:
        if gran == "ptensor":
            x = _cuda(*_CORRECT_PTENSOR_SHAPE, dtype=torch.float32)
            rq, s = quantize_per_tensor(x, fmt=cast_fmt, backend="torch")
        elif gran == "pchan":
            x = _cuda(*_CORRECT_PCHAN_SHAPE, dtype=torch.float32)
            rq, s = quantize_per_channel(x, fmt=cast_fmt, axis=0, backend="torch")
        else:
            x = _cuda(*_CORRECT_BLK_SHAPE, dtype=torch.float32)
            rq, s = quantize_per_block(x, fmt=cast_fmt, block_size=block_size, backend="torch")
        if not torch.isfinite(rq).all() or not torch.isfinite(s).all():
            raise AssertionError("non-finite cast output")
    return _fn


def _append_exp_case(
    cases: List[KernelCase],
    api_fmt: str,
    cid: str,
    gran: str,
    x_bench: torch.Tensor,
    *,
    block_size: int,
    bench_block_size: Optional[int] = None,
) -> None:
    bs = bench_block_size if bench_block_size is not None else block_size
    if gran == "ptensor":
        correct = _exp_triton_vs_torch(api_fmt, "ptensor")
        b_t = lambda f=api_fmt: quantize_per_tensor(x_bench, fmt=f, backend="torch")
        b_r = lambda f=api_fmt: quantize_per_tensor(x_bench, fmt=f, backend="triton")
        case_id = f"{cid}_exp_ptensor_triton_vs_torch"
    elif gran == "pchan":
        correct = _exp_triton_vs_torch(api_fmt, "pchan")
        b_t = lambda f=api_fmt: quantize_per_channel(x_bench, fmt=f, axis=0, backend="torch")
        b_r = lambda f=api_fmt: quantize_per_channel(x_bench, fmt=f, axis=0, backend="triton")
        case_id = f"{cid}_exp_pchan_triton_vs_torch"
    elif gran == "blk":
        correct = _exp_triton_vs_torch(api_fmt, "blk", block_size=bs)
        b_t = lambda f=api_fmt, b=bs: quantize_per_block(
            x_bench, fmt=f, block_size=b, backend="torch"
        )
        b_r = lambda f=api_fmt, b=bs: quantize_per_block(
            x_bench, fmt=f, block_size=b, backend="triton"
        )
        case_id = f"{cid}_exp_blk{bs}_triton_vs_torch"
    else:
        raise ValueError(gran)
    cases.append(
        KernelCase(
            case_id,
            correct,
            _make_quality_fn(api_fmt, gran, x_bench.shape[0], x_bench.shape[1], bs, x_bench.dtype),
            b_t,
            b_r,
        )
    )


def _append_cast_case(
    cases: List[KernelCase],
    cast_fmt: str,
    cid: str,
    gran: str,
    x_bench: torch.Tensor,
    m: int,
    n: int,
    block_size: int,
    dtype: torch.dtype,
    *,
    blk_ok: bool,
) -> None:
    q_fn = _make_quality_fn(cast_fmt, gran, m, n, block_size, dtype)
    if gran == "ptensor":
        cases.append(
            KernelCase(
                f"{cid}_cast_ptensor_torch_only",
                _cast_torch_smoke(cast_fmt, "ptensor", block_size),
                q_fn,
                lambda f=cast_fmt: quantize_per_tensor(x_bench, fmt=f, backend="torch"),
                None,
            )
        )
    elif gran == "pchan":
        cases.append(
            KernelCase(
                f"{cid}_cast_pchan_torch_only",
                _cast_torch_smoke(cast_fmt, "pchan", block_size),
                q_fn,
                lambda f=cast_fmt: quantize_per_channel(
                    x_bench, fmt=f, axis=0, backend="torch"
                ),
                None,
            )
        )
    elif gran == "blk":
        if not blk_ok:
            return
        cases.append(
            KernelCase(
                f"{cid}_cast_blk{block_size}_triton_vs_torch",
                _cast_blk_correct(cast_fmt, m, n, block_size, dtype),
                q_fn,
                lambda f=cast_fmt, b=block_size: quantize_per_block(
                    x_bench, fmt=f, block_size=b, backend="torch"
                ),
                lambda f=cast_fmt, b=block_size: quantize_per_block(
                    x_bench, fmt=f, block_size=b, backend="triton"
                ),
            )
        )


def build_cases(
    m: int,
    n: int,
    block_size: int,
    dtype: torch.dtype,
) -> List[KernelCase]:
    x = _cuda(m, n, dtype=dtype)
    cases: List[KernelCase] = []
    blk_ok = m % block_size == 0 and n % block_size == 0
    grans = ("ptensor", "pchan", "blk")

    for api_fmt, cid in EXP_FP8 + EXP_NV:
        for gran in grans:
            _append_exp_case(cases, api_fmt, cid, gran, x, block_size=block_size)

    for cast_fmt, cid in CAST_FMT:
        for gran in grans:
            _append_cast_case(
                cases, cast_fmt, cid, gran, x, m, n, block_size, dtype, blk_ok=blk_ok
            )

    return cases


def run_all_cases(
    cases: List[KernelCase],
    *,
    run_correctness: bool,
    run_quality: bool,
    run_bench: bool,
    warmup: int,
    iters: int,
) -> List[CaseResult]:
    results: List[CaseResult] = []
    for case in cases:
        correct, detail = "n/a", ""
        if run_correctness and case.correct_fn is not None:
            correct, detail = _run_correct(case.correct_fn)

        mse_fp16 = rel_mse = max_err = None
        if run_quality and case.quality_fn is not None:
            try:
                metrics = case.quality_fn()
                mse_fp16 = metrics["mse"]
                rel_mse = metrics["rel_mse"]
                max_err = metrics["max_err"]
            except Exception as e:
                correct = "FAIL"
                detail = (detail + "; " if detail else "") + f"vs_fp16: {e}"

        torch_ms = triton_ms = None
        if run_bench and case.bench_torch is not None:
            torch_ms = _time_fn(case.bench_torch, warmup, iters)
            if case.bench_triton is not None:
                triton_ms = _time_fn(case.bench_triton, warmup, iters)

        results.append(
            CaseResult(
                case_id=case.case_id,
                correct=correct,
                correct_detail=detail,
                mse_fp16=mse_fp16,
                rel_mse_fp16=rel_mse,
                max_err_fp16=max_err,
                torch_ms=torch_ms,
                triton_ms=triton_ms,
            )
        )
    return results


def print_summary_table(
    results: List[CaseResult],
    m: int,
    n: int,
    iters: int,
    *,
    baseline_dtype: torch.dtype,
) -> None:
    elems = m * n
    elem_bytes = 2 if baseline_dtype == torch.float16 else 4
    gib = elems * elem_bytes / (1024**3)
    print(f"\n{'=' * 128}")
    print(
        f"SUMMARY  shape=({m},{n})  baseline={baseline_dtype}  elems={elems:,}  "
        f"~{gib:.3f} GiB/iter  bench_iters={iters}"
    )
    print(f"{'=' * 128}")
    hdr = (
        f"{'case_id':<40} {'ok':>4} {'mse_fp16':>10} {'rel_mse':>9} {'max_err':>9} "
        f"{'torch_ms':>9} {'tri_ms':>9} {'spdup':>7}  note"
    )
    print(hdr)
    print("-" * len(hdr))

    n_fail = 0
    for r in results:
        sp = f"{r.speedup:.1f}x" if r.speedup is not None else "—"
        t_ms = f"{r.torch_ms:.3f}" if r.torch_ms is not None else "—"
        tr_ms = f"{r.triton_ms:.3f}" if r.triton_ms is not None else "—"
        mse_s = f"{r.mse_fp16:.4g}" if r.mse_fp16 is not None else "—"
        rel_s = f"{r.rel_mse_fp16:.4g}" if r.rel_mse_fp16 is not None else "—"
        max_s = f"{r.max_err_fp16:.4g}" if r.max_err_fp16 is not None else "—"
        note = ""
        if r.correct == "FAIL":
            note = r.correct_detail[:20]
            n_fail += 1
        elif r.triton_ms is None and r.torch_ms is not None:
            note = "torch-only"
        print(
            f"{r.case_id:<40} {r.correct:>4} {mse_s:>10} {rel_s:>9} {max_s:>9} "
            f"{t_ms:>9} {tr_ms:>9} {sp:>7}  {note}"
        )

    print("-" * len(hdr))
    print(
        "ok: kernel + vs_fp16   mse_fp16/rel_mse/max_err = ||rq-x|| vs fp16 baseline (torch quant)   "
        "spdup = torch_ms/triton_ms"
    )
    if n_fail:
        print(f"\n{n_fail} failure(s)")
        return n_fail
    print("\nAll checks passed.")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="Quant kernel correctness + speedup")
    p.add_argument("--m", type=int, default=4096)
    p.add_argument("--n", type=int, default=4096)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), default="fp16")
    p.add_argument("--correctness-only", action="store_true")
    p.add_argument("--bench-only", action="store_true")
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="optional path to write summary text (e.g. understand/trail/results)",
    )
    args = p.parse_args()

    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    baseline_dtype = dtype_map[args.dtype]
    run_correctness = not args.bench_only
    run_quality = not args.bench_only
    run_bench = not args.correctness_only

    if run_correctness or run_quality:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if run_bench:
        torch.backends.cudnn.benchmark = True

    cases = build_cases(args.m, args.n, args.block_size, baseline_dtype)
    results = run_all_cases(
        cases,
        run_correctness=run_correctness,
        run_quality=run_quality,
        run_bench=run_bench,
        warmup=args.warmup,
        iters=args.iters,
    )

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        n_fail = print_summary_table(
            results, args.m, args.n, args.iters, baseline_dtype=baseline_dtype
        )
    out = buf.getvalue()
    print(out, end="")
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(out)
        print(f"Wrote {out_path}")
    if n_fail:
        raise SystemExit(n_fail)


if __name__ == "__main__":
    main()
