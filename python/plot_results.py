#!/usr/bin/env python3
"""Charts for results/histograms.json and results/bench.json.

Bytes in the benchmark JSON are decimal GB/s (1e9 bytes per second).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"wrote {path}")


def plot_histograms(hist: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    for key, label in (("gpt2", "GPT-2 linear"), ("qwen2.5-0.5b", "Qwen2.5-0.5B linear")):
        counts = np.asarray(hist["models"][key]["pooled"]["linear"]["exp_hist"], dtype=np.float64)
        xs = np.arange(256)
        mask = counts > 0
        ax.semilogy(xs[mask], counts[mask], ".", ms=3, label=label)
    ax.set_xlabel("BF16 exponent")
    ax.set_ylabel("weight count")
    ax.set_xlim(90, 160)
    ax.legend()
    ax.set_title("Pooled linear-weight exponent histogram")
    _save(fig, out / "exp_hist.png")

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    groups = ["1", "4", "16", "64", "256", "tensor"]
    x = np.arange(len(groups))
    width = 0.36
    for i, (key, label) in enumerate((("gpt2", "GPT-2 linear"), ("qwen2.5-0.5b", "Qwen2.5-0.5B linear"))):
        bpw = hist["models"][key]["pooled"]["linear"]["bpw_by_group"]
        ax.bar(x + (i - 0.5) * width, [bpw[g] for g in groups], width, label=label)
    ax.axhline(16.0, color="black", lw=0.8, label="BF16 (16 bpw)")
    ax.set_xticks(x, groups)
    ax.set_ylabel("bits per weight, layout v1")
    ax.set_xlabel("codebook group, in rows (tensor = one codebook)")
    ax.set_ylim(10, 17)
    ax.legend()
    ax.set_title("Encoded size versus codebook group")
    _save(fig, out / "bpw_vs_group.png")


def plot_stream(bench: dict, out: Path) -> None:
    rows = bench["stream"]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0), sharey=True)
    for ax, kernel, title in (
        (axes[0], "read", "read, bytes = 4N"),
        (axes[1], "triad", "triad, bytes = 12N"),
    ):
        picked = [r for r in rows if r["kernel"] == kernel and r.get("store", "normal") == "normal"]
        if kernel == "triad":
            picked = [r for r in rows if r["kernel"] == "triad" and r.get("store") == "normal"]
        sizes = sorted({r["bytes"] for r in picked})
        for nbytes in sizes:
            series = sorted((r for r in picked if r["bytes"] == nbytes), key=lambda r: r["threads"])
            mib = nbytes / (1024 * 1024)
            ax.plot(
                [r["threads"] for r in series],
                [r["GBps"] for r in series],
                marker="o",
                label=f"{mib:.0f} MiB",
            )
        ax.set_xlabel("threads")
        ax.set_title(title)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("GB/s (1e9 bytes/s)")
    fig.suptitle("STREAM-style bandwidth, normal stores")
    _save(fig, out / "stream.png")


def plot_gemv(bench: dict, out: Path) -> None:
    # One panel per real (not tiled) matrix that has an AVX-512 sweep.
    wanted = []
    seen = set()
    for row in bench["gemv"]:
        if row["isa"] != "avx512" or row["name"].endswith("#tiled"):
            continue
        if row["group_rows"] != row["rows"]:
            continue
        key = (row["model"], row["name"])
        if key not in seen:
            seen.add(key)
            wanted.append(key)
    # Keep the figure readable: largest inner-product shapes plus the two heads.
    preferred = [
        ("gpt2", "h.0.mlp.c_proj.weight"),
        ("gpt2", "wte.weight"),
        ("qwen2.5-0.5b", "model.layers.0.mlp.down_proj.weight"),
        ("qwen2.5-0.5b", "model.embed_tokens.weight"),
        ("qwen2.5-0.5b", "model.layers.0.self_attn.k_proj.weight"),
        ("gpt2", "h.0.attn.c_attn.weight"),
    ]
    chosen = [k for k in preferred if k in seen]
    if not chosen:
        chosen = wanted[:6]
    n = len(chosen)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(11.2, 3.3 * rows), squeeze=False)
    for ax, key in zip(axes.ravel(), chosen):
        model, name = key
        for storage, style in (("fp32", ":"), ("bf16", "--"), ("expcode", "-")):
            series = sorted(
                (
                    r
                    for r in bench["gemv"]
                    if r["model"] == model
                    and r["name"] == name
                    and r["isa"] == "avx512"
                    and r["storage"] == storage
                    and r["group_rows"] == r["rows"]
                ),
                key=lambda r: r["threads"],
            )
            if not series:
                continue
            ax.plot(
                [r["threads"] for r in series],
                [r["effective_bf16_GBps"] for r in series],
                marker="o",
                linestyle=style,
                label=storage,
            )
        short = name.split(".")[-2] + "." + name.split(".")[-1] if name.count(".") else name
        shape = next(r for r in bench["gemv"] if r["model"] == model and r["name"] == name)
        ax.set_title(f"{model}\n{short} {shape['rows']}x{shape['cols']}", fontsize=9)
        ax.set_xlabel("threads")
        ax.set_ylabel("effective BF16 GB/s")
        ax.legend(fontsize=8)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("AVX-512 GEMV, effective GB/s = rows*cols*2 / seconds / 1e9")
    _save(fig, out / "gemv_scaling.png")

    tiled = [
        r
        for r in bench["gemv"]
        if r["name"].endswith("#tiled") and r["isa"] == "avx512" and r["group_rows"] == r["rows"]
    ]
    if tiled:
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        names = sorted({r["name"] for r in tiled})
        for name in names:
            for storage, style in (("bf16", "--"), ("expcode", "-")):
                series = sorted(
                    (r for r in tiled if r["name"] == name and r["storage"] == storage),
                    key=lambda r: r["threads"],
                )
                if not series:
                    continue
                ax.plot(
                    [r["threads"] for r in series],
                    [r["effective_bf16_GBps"] for r in series],
                    marker="o",
                    linestyle=style,
                    label=f"{storage} {name}",
                )
        ax.set_xlabel("threads")
        ax.set_ylabel("effective BF16 GB/s")
        ax.legend(fontsize=8)
        ax.set_title("Tiled GEMV, repeated rows, about 512 MiB of BF16")
        _save(fig, out / "gemv_tiled.png")


def plot_decode(bench: dict, out: Path) -> None:
    rows = bench.get("decode") or []
    if not rows:
        return
    models = []
    for row in rows:
        if row["model"] not in models:
            models.append(row["model"])
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    for model in models:
        for storage, style in (("bf16", "--"), ("expcode", "-")):
            series = sorted(
                (r for r in rows if r["model"] == model and r["storage"] == storage),
                key=lambda r: r["threads"],
            )
            ax.plot(
                [r["threads"] for r in series],
                [r["tok_per_s"] for r in series],
                marker="o",
                linestyle=style,
                label=f"{model} {storage}",
            )
    ax.set_xlabel("threads")
    ax.set_ylabel("tokens/s, median step after prefill")
    ax.legend(fontsize=8)
    ax.set_title("Single-token decode")
    _save(fig, out / "decode_toks.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hist", type=Path, default=Path("results/histograms.json"))
    parser.add_argument("--bench", type=Path, default=Path("results/bench.json"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    hist = json.loads(args.hist.read_text())
    plot_histograms(hist, args.out)
    if args.bench.exists():
        bench = json.loads(args.bench.read_text())
        plot_stream(bench, args.out)
        plot_gemv(bench, args.out)
        plot_decode(bench, args.out)


if __name__ == "__main__":
    main()
