#!/usr/bin/env python3
"""Exponent histograms and fixed-rate code size for BF16 weights.

Reads safetensors (not committed). GPT-2 FP32 weights are rounded to BF16
with round-to-nearest-even. Qwen tensors that are already BF16 are used as
stored. Prints a text summary and writes a JSON report.

The size model is the SIMD layout in include/expcode/format.hpp:
  sign+mantissa byte, 4-bit exponent code, 15-entry codebook, one raw
  exponent byte per escape (code 15). No position side-index.
"""

from __future__ import annotations

import argparse
import json
import os

import ml_dtypes  # noqa: F401  (registers numpy bfloat16)
import numpy as np
from safetensors import safe_open

CODE_BITS = 4
DIRECT_CODES = (1 << CODE_BITS) - 1  # 15; code 15 is escape
GROUP_ROWS = (1, 4, 16, 64, 256)


def f32_to_bf16(x: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even FP32 -> BF16. NaNs are not expected in weights."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    rounded = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
    return rounded.astype(np.uint16)


def as_bf16_bits(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.float32:
        return f32_to_bf16(arr)
    if arr.dtype == np.uint16:
        return np.ascontiguousarray(arr)
    # ml_dtypes.bfloat16 (and any other 2-byte float storage)
    if arr.dtype.itemsize == 2:
        return np.ascontiguousarray(arr).view(np.uint16)
    if arr.dtype == np.float16:
        return f32_to_bf16(arr.astype(np.float32))
    raise TypeError(f"unsupported dtype {arr.dtype}")


def entropy_from_counts(counts: np.ndarray) -> float:
    total = float(counts.sum())
    if total <= 0:
        return 0.0
    p = counts.astype(np.float64)
    p = p[p > 0] / total
    return float(-(p * np.log2(p)).sum())


def tensor_kind(name: str) -> str:
    if name in ("wte.weight", "wpe.weight", "model.embed_tokens.weight"):
        return "embed"
    return "linear"


def gemv_layout(name: str, arr: np.ndarray, model: str) -> tuple[np.ndarray, bool]:
    """Return BF16 bits in GEMV row-major [out, in].

    HF GPT-2 Conv1D stores [in, out]. Qwen nn.Linear stores [out, in].
    Token/position tables are already [rows, width] and are not transposed.
    """
    bits = as_bf16_bits(arr)
    transposed = False
    if model == "gpt2" and name not in ("wte.weight", "wpe.weight") and bits.ndim == 2:
        bits = np.ascontiguousarray(bits.T)
        transposed = True
    return bits, transposed


def row_count_matrix(exps: np.ndarray, chunk_rows: int = 1024) -> np.ndarray:
    """Return int32 histogram [rows, 256]."""
    rows, cols = exps.shape
    counts = np.zeros((rows, 256), dtype=np.int32)
    for r0 in range(0, rows, chunk_rows):
        r1 = min(rows, r0 + chunk_rows)
        block = exps[r0:r1]
        n = r1 - r0
        # Keep the index expression in int32: n * 256 is far below 2^31.
        row_base = (np.arange(n, dtype=np.int32) * 256)[:, None]
        idx = row_base + block.astype(np.int32)
        flat = np.bincount(idx.ravel(), minlength=n * 256)
        counts[r0:r1] = flat.reshape(n, 256)
    return counts


def group_escape(counts: np.ndarray, group_rows: int) -> tuple[int, int]:
    """Return (escape_weights, n_groups) for a shared top-15 codebook per group."""
    rows = counts.shape[0]
    if rows == 0:
        return 0, 0
    if group_rows <= 0 or group_rows >= rows:
        top = np.sort(counts.sum(axis=0))[-DIRECT_CODES:].sum()
        return int(counts.sum() - top), 1
    escapes = 0
    n_full = rows // group_rows
    if n_full:
        grouped = counts[: n_full * group_rows].reshape(n_full, group_rows, 256).sum(axis=1)
        top = np.partition(grouped, -DIRECT_CODES, axis=1)[:, -DIRECT_CODES :].sum(axis=1)
        escapes += int(grouped.sum() - top.sum())
    tail = rows - n_full * group_rows
    n_groups = n_full + (1 if tail else 0)
    if tail:
        g = counts[n_full * group_rows :].sum(axis=0)
        top = np.sort(g)[-DIRECT_CODES:].sum()
        escapes += int(g.sum() - top)
    return escapes, n_groups


def encoded_bytes(n_groups: int, cols: int, escapes: int, rows: int) -> int:
    """Byte size of one matrix in the v1 layout. Must match EncodedMatrix::encode.

    64-byte header, 16-byte codebook per group, tightly packed nibble codes,
    raw sign/mantissa bytes, one raw exponent byte per escape, and a
    (rows + 1) uint32 sentinel offset table so a row can be entered in O(1).
    Sections are not padded.
    """
    code_bytes = rows * ((cols + 1) // 2)
    sm_bytes = rows * cols
    index_bytes = 4 * (rows + 1)
    return 64 + n_groups * 16 + code_bytes + sm_bytes + int(escapes) + index_bytes


def bits_per_weight(n_weights: int, n_groups: int, cols: int, escapes: int, rows: int) -> float:
    if n_weights <= 0:
        return 0.0
    return 8.0 * encoded_bytes(n_groups, cols, escapes, rows) / n_weights


def summarize_counts(counts: np.ndarray, cols: int) -> dict:
    rows = counts.shape[0]
    n_weights = int(rows * cols)
    nunique = np.count_nonzero(counts, axis=1)
    top = np.partition(counts, -DIRECT_CODES, axis=1)[:, -DIRECT_CODES :].sum(axis=1)
    # Rows with fewer than 15 used exponents still partition in zeros.
    row_escape = cols - top
    out = {
        "rows": rows,
        "cols": cols,
        "weights": n_weights,
        "row_nunique_p50": float(np.percentile(nunique, 50)),
        "row_nunique_p90": float(np.percentile(nunique, 90)),
        "row_nunique_p99": float(np.percentile(nunique, 99)),
        "row_nunique_max": int(nunique.max()) if rows else 0,
        "frac_rows_nunique_le_7": float(np.mean(nunique <= 7)) if rows else 0.0,
        "frac_rows_nunique_le_15": float(np.mean(nunique <= 15)) if rows else 0.0,
        "frac_rows_nunique_le_31": float(np.mean(nunique <= 31)) if rows else 0.0,
        "bpw_by_group": {},
        "escape_rate_by_group": {},
    }
    for g in GROUP_ROWS:
        escapes, n_groups = group_escape(counts, g)
        out["bpw_by_group"][str(g)] = bits_per_weight(n_weights, n_groups, cols, escapes, rows)
        out["escape_rate_by_group"][str(g)] = escapes / n_weights if n_weights else 0.0
    escapes, n_groups = group_escape(counts, rows if rows else 1)
    out["bpw_by_group"]["tensor"] = bits_per_weight(n_weights, n_groups, cols, escapes, rows)
    out["escape_rate_by_group"]["tensor"] = escapes / n_weights if n_weights else 0.0
    out["row_escape_rate"] = float(row_escape.sum() / n_weights) if n_weights else 0.0
    return out


def analyze_tensor(name: str, bits: np.ndarray, transposed: bool, kind: str) -> dict:
    assert bits.ndim == 2
    rows, cols = bits.shape
    exps = ((bits >> 7) & np.uint16(0xFF)).astype(np.uint8)
    signs = (bits >> 15).astype(np.uint8)
    mants = (bits & np.uint16(0x7F)).astype(np.uint8)
    exp_hist = np.bincount(exps.ravel(), minlength=256).astype(np.int64)
    sign_hist = np.bincount(signs.ravel(), minlength=2).astype(np.int64)
    mant_hist = np.bincount(mants.ravel(), minlength=128).astype(np.int64)
    counts = row_count_matrix(exps)
    summary = summarize_counts(counts, cols)
    order = np.argsort(exp_hist)[::-1]
    cum = np.cumsum(exp_hist[order])
    total = int(exp_hist.sum())
    coverage = {}
    for k in (1, 2, 4, 8, 15, 16, 32, 64):
        coverage[str(k)] = float(cum[k - 1] / total) if total else 0.0
    summary.update(
        {
            "name": name,
            "kind": kind,
            "shape": [int(rows), int(cols)],
            "transposed_to_gemv": transposed,
            "exp_entropy": entropy_from_counts(exp_hist),
            "sign_entropy": entropy_from_counts(sign_hist),
            "mant_entropy": entropy_from_counts(mant_hist),
            "n_distinct_exp": int(np.count_nonzero(exp_hist)),
            "top_k_coverage": coverage,
            "exp_hist": exp_hist.tolist(),
        }
    )
    return summary


def weighted_merge(records: list[dict], key_groups=("1", "4", "16", "64", "256", "tensor")) -> dict:
    weights = 0
    exp_hist = np.zeros(256, dtype=np.int64)
    sign_h = 0.0
    mant_h = 0.0
    exp_h = 0.0
    # Recompute entropy from summed hist only; sign/mant reported as weight-averaged
    # per-tensor entropies (a lower bound on the pooled entropy is not required).
    bpw_bytes = {k: 0.0 for k in key_groups}  # store weight*bpw
    esc = {k: 0.0 for k in key_groups}
    for rec in records:
        w = rec["weights"]
        weights += w
        exp_hist += np.asarray(rec["exp_hist"], dtype=np.int64)
        exp_h += rec["exp_entropy"] * w
        sign_h += rec["sign_entropy"] * w
        mant_h += rec["mant_entropy"] * w
        for k in key_groups:
            bpw_bytes[k] += rec["bpw_by_group"][k] * w
            esc[k] += rec["escape_rate_by_group"][k] * w
    if weights == 0:
        return {"weights": 0}
    return {
        "weights": weights,
        "bf16_bytes": weights * 2,
        "exp_entropy_pooled": entropy_from_counts(exp_hist),
        "exp_entropy_weight_avg": exp_h / weights,
        "sign_entropy_weight_avg": sign_h / weights,
        "mant_entropy_weight_avg": mant_h / weights,
        "bpw_by_group": {k: bpw_bytes[k] / weights for k in key_groups},
        "escape_rate_by_group": {k: esc[k] / weights for k in key_groups},
        "info_bound_bpw_raw_sign_mant": 8.0 + entropy_from_counts(exp_hist),
        "exp_hist": exp_hist.tolist(),
    }


def analyze_file(path: str, model: str) -> dict:
    records = []
    with safe_open(path, framework="numpy") as handle:
        names = [n for n in handle.keys() if n.endswith("weight")]
        for name in names:
            arr = handle.get_tensor(name)
            if arr.ndim != 2:
                continue
            bits, transposed = gemv_layout(name, arr, model)
            kind = tensor_kind(name)
            print(f"  {name} gemv_shape={list(bits.shape)} kind={kind} transposed={transposed}", flush=True)
            records.append(analyze_tensor(name, bits, transposed, kind))
            del arr, bits
    linear = [r for r in records if r["kind"] == "linear"]
    embed = [r for r in records if r["kind"] == "embed"]
    # Drop per-tensor histograms from the returned records after aggregation
    # to keep the JSON usable; full hists are folded into the pools.
    pooled = {
        "all_2d": weighted_merge(records),
        "linear": weighted_merge(linear),
        "embed": weighted_merge(embed),
    }
    slim = []
    for rec in records:
        item = dict(rec)
        item.pop("exp_hist", None)
        slim.append(item)
    return {"model": model, "source": path, "tensors": slim, "pooled": pooled}


def fmt_bpw(pooled: dict) -> str:
    if pooled.get("weights", 0) == 0:
        return "(none)"
    parts = [f"H(exp)={pooled['exp_entropy_pooled']:.3f}"]
    parts.append(f"bound={pooled['info_bound_bpw_raw_sign_mant']:.3f}")
    for k, v in pooled["bpw_by_group"].items():
        esc = pooled["escape_rate_by_group"][k]
        parts.append(f"g={k}: {v:.3f} bpw (escape {esc:.4f})")
    return " | ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt2", required=True)
    parser.add_argument("--qwen", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    # Self-check the BF16 round-trip of a few exact values.
    check = f32_to_bf16(np.array([1.0, -1.0, 2.0, 0.0], dtype=np.float32))
    expect = np.array([0x3F80, 0xBF80, 0x4000, 0x0000], dtype=np.uint16)
    if not np.array_equal(check, expect):
        raise SystemExit(f"bf16 conversion self-check failed: {check.tolist()}")

    report = {
        "code_bits": CODE_BITS,
        "direct_codes": DIRECT_CODES,
        "layout": "v1",
        "byte_formula": "64 + n_groups*16 + rows*((cols+1)//2) + rows*cols + n_escape + 4*(rows+1)",
        "models": {},
    }
    for model, path in (("gpt2", args.gpt2), ("qwen2.5-0.5b", args.qwen)):
        print(f"analyzing {model}", flush=True)
        report["models"][model] = analyze_file(path, "gpt2" if model == "gpt2" else "qwen")
        pooled = report["models"][model]["pooled"]
        print(f"LINEAR {fmt_bpw(pooled['linear'])}", flush=True)
        print(f"EMBED  {fmt_bpw(pooled['embed'])}", flush=True)
        print(f"ALL    {fmt_bpw(pooled['all_2d'])}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
