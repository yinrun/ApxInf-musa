#!/usr/bin/env python3
"""Tensor-tree comparison for qwen38 (and general) numerical verification.

Phase-0 item 0.4: recreates the missing `scripts/compare.py` that
`crates/apxinf-cuda-new/tests/qwen38_layer_compare.rs` expects: two trees of
float tensors, per-tensor cosine / relative-L2 / abs-max / rel-inf metrics,
threshold pass/fail, and first-divergence localization.

Each SIDE may be either a directory (walked for *.npy, keyed by relative
path) or a single .npz file (keyed by member name). Only keys present on
both sides are compared; missing keys are reported.

Usage:
  python3 scripts/compare.py GOLD ACTUAL [--cosine 0.999] [--rel-l2 0.05]
      [--first-divergence] [--json OUT.json]

Acceptance (self-test): compare a tree against itself -> all cosines 1.0, exit 0.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import numpy as np


def load_side(path: pathlib.Path) -> dict[str, np.ndarray]:
    """-> {key: ndarray} from a .npy directory tree or a .npz file."""
    tensors: dict[str, np.ndarray] = {}
    if path.is_file() and path.suffix == ".npz":
        with np.load(path) as z:
            for key in z.files:
                tensors[key] = z[key]
        return tensors
    if not path.is_dir():
        raise SystemExit(f"ERROR: {path} is neither a .npz file nor a directory")
    for npy in sorted(path.rglob("*.npy")):
        tensors[str(npy.relative_to(path))] = np.load(npy)
    if not tensors:
        raise SystemExit(f"ERROR: no tensors found under {path}")
    return tensors


def metrics(gold: np.ndarray, actual: np.ndarray) -> dict:
    g = gold.reshape(-1).astype(np.float64)
    a = actual.reshape(-1).astype(np.float64)
    if g.shape != a.shape:
        return {"shape_gold": list(gold.shape), "shape_actual": list(actual.shape),
                "error": "shape mismatch"}
    dot = float(np.dot(g, a))
    ng = math.sqrt(float(np.dot(g, g)))
    na = math.sqrt(float(np.dot(a, a)))
    cosine = dot / (ng * na) if ng > 0 and na > 0 else (1.0 if ng == na == 0 else 0.0)
    diff = g - a
    denom = ng if ng > 0 else 1.0
    return {
        "numel": int(g.size),
        "cosine": cosine,
        "rel_l2": math.sqrt(float(np.dot(diff, diff))) / denom,
        "abs_max": float(np.max(np.abs(diff))) if g.size else 0.0,
        "rel_inf": float(np.max(np.abs(diff)) / max(float(np.max(np.abs(g))), 1e-12)) if g.size else 0.0,
    }


def first_divergence(gold: np.ndarray, actual: np.ndarray, rtol=1e-2, atol=1e-3):
    """First index where the tensors disagree beyond (atol, rtol); None if none."""
    g = gold.reshape(-1).astype(np.float64)
    a = actual.reshape(-1).astype(np.float64)
    bad = np.abs(g - a) > (atol + rtol * np.abs(g))
    idx = int(np.argmax(bad)) if bad.any() else None
    if idx is None:
        return None
    return {"index": idx, "gold": float(g[idx]), "actual": float(a[idx])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("gold", type=pathlib.Path)
    ap.add_argument("actual", type=pathlib.Path)
    ap.add_argument("--cosine", type=float, default=0.999,
                    help="per-tensor cosine threshold (layer default 0.999; logits 0.9995)")
    ap.add_argument("--rel-l2", type=float, default=0.05)
    ap.add_argument("--first-divergence", action="store_true")
    ap.add_argument("--json", type=pathlib.Path, default=None)
    args = ap.parse_args()

    gold = load_side(args.gold)
    actual = load_side(args.actual)

    common = sorted(set(gold) & set(actual))
    only_gold = sorted(set(gold) - set(actual))
    only_actual = sorted(set(actual) - set(gold))

    rows, failures = [], []
    for key in common:
        m = metrics(gold[key], actual[key])
        row = {"key": key, **m}
        if args.first_divergence and "error" not in m:
            fd = first_divergence(gold[key], actual[key])
            if fd is not None:
                row["first_divergence"] = fd
        bad = ("error" in m
               or m.get("cosine", 0.0) < args.cosine
               or m.get("rel_l2", math.inf) > args.rel_l2)
        row["pass"] = not bad
        rows.append(row)
        if bad:
            failures.append(key)

    worst_cos = min((r["cosine"] for r in rows if "cosine" in r), default=float("nan"))
    print(f"gold   : {args.gold}  ({len(gold)} tensors)")
    print(f"actual : {args.actual}  ({len(actual)} tensors)")
    print(f"compared: {len(rows)}   cosine>={args.cosine}  rel_l2<={args.rel_l2}")
    if only_gold:
        print(f"only in gold  ({len(only_gold)}): {only_gold[:8]}")
    if only_actual:
        print(f"only in actual ({len(only_actual)}): {only_actual[:8]}")
    print(f"worst cosine: {worst_cos:.8f}")
    if rows:
        print(f"{'key':48s} {'cosine':>10s} {'rel_l2':>10s} {'abs_max':>12s}")
        for r in sorted(rows, key=lambda r: r.get("cosine", -2)):
            mark = "" if r["pass"] else "  FAIL"
            if "error" in r:
                print(f"{r['key']:48s} {r['error']}{mark}")
            else:
                print(f"{r['key']:48s} {r['cosine']:10.6f} {r['rel_l2']:10.6f} "
                      f"{r['abs_max']:12.4e}{mark}")
                if "first_divergence" in r:
                    fd = r["first_divergence"]
                    print(f"    first divergence @ {fd['index']}: gold={fd['gold']:.6g} "
                          f"actual={fd['actual']:.6g}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(
            {"thresholds": {"cosine": args.cosine, "rel_l2": args.rel_l2},
             "summary": {"compared": len(rows), "failures": failures,
                         "worst_cosine": worst_cos,
                         "only_gold": only_gold, "only_actual": only_actual},
             "tensors": rows}, indent=1))
        print(f"json    : {args.json}")
    print("RESULT  :", "PASS" if not failures and not only_actual else "FAIL")
    return 0 if not failures and not only_actual else 1


if __name__ == "__main__":
    sys.exit(main())
