#!/usr/bin/env python3
"""Export GEMV-layout BF16 weights to a BPK1 pack. Weights are not committed.

GPT-2 FP32 Conv1D weights are round-to-nearest-even converted to BF16 and
transposed to [out, in]. Qwen BF16 Linear weights are stored as they are.
"""

from __future__ import annotations

import argparse
import struct

import ml_dtypes  # noqa: F401
import numpy as np
from safetensors import safe_open


def f32_to_bf16(x: np.ndarray) -> np.ndarray:
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    rounded = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
    return rounded.astype(np.uint16)


def as_bf16_bits(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.float32:
        return f32_to_bf16(arr)
    if arr.dtype == np.uint16:
        return np.ascontiguousarray(arr)
    if arr.dtype.itemsize == 2:
        return np.ascontiguousarray(arr).view(np.uint16)
    if arr.dtype == np.float16:
        return f32_to_bf16(arr.astype(np.float32))
    raise TypeError(str(arr.dtype))


def want(model: str, name: str, ndim: int) -> bool:
    if ndim > 2 or not (name.endswith(".weight") or name.endswith(".bias")):
        return False
    if model == "gpt2" and name.endswith(".attn.bias") and not name.endswith(".c_attn.bias"):
        return False
    return True


def to_gemv(model: str, name: str, bits: np.ndarray) -> np.ndarray:
    if model == "gpt2" and name.startswith("h.") and name.endswith(".weight") and bits.ndim == 2:
        return np.ascontiguousarray(bits.T)
    return np.ascontiguousarray(bits)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("gpt2", "qwen2"), required=True)
    parser.add_argument("--safetensors", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    check = f32_to_bf16(np.array([1.0, -1.0, 2.0], np.float32))
    if check.tolist() != [0x3F80, 0xBF80, 0x4000]:
        raise SystemExit("bf16 conversion self-check failed")

    tensors = []
    with safe_open(args.safetensors, framework="numpy") as handle:
        for name in handle.keys():
            slice_ = handle.get_slice(name)
            if not want(args.model, name, len(slice_.get_shape())):
                continue
            bits = to_gemv(args.model, name, as_bf16_bits(handle.get_tensor(name)))
            tensors.append((name, bits))
            print(f"  {name} {list(bits.shape)}", flush=True)

    with open(args.out, "wb") as out:
        out.write(struct.pack("<III", 0x314B5042, 1, len(tensors)))
        for name, bits in tensors:
            raw = np.ascontiguousarray(bits, dtype=np.uint16).tobytes()
            name_b = name.encode("utf-8")
            dims = bits.shape
            out.write(struct.pack("<I", len(name_b)))
            out.write(name_b)
            out.write(struct.pack("<I", len(dims)))
            for d in dims:
                out.write(struct.pack("<I", int(d)))
            out.write(struct.pack("<Q", len(raw)))
            out.write(raw)
            pad = (8 - (out.tell() & 7)) & 7
            if pad:
                out.write(b"\0" * pad)
    print(f"wrote {args.out} tensors={len(tensors)}", flush=True)


if __name__ == "__main__":
    main()
