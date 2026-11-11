#!/usr/bin/env python3
"""Audit Qwen/Qwen3.8-27B (HF BF16) tensor names against the planned musa_qwen38 mapping.

Phase-0 item 0.2 of the qwen38 MUSA S5000 port. Two modes:

  name-level : needs only model.safetensors.index.json (runs before shards arrive)
  shape-level: additionally parses each shard's safetensors header (runs after download)

Checks:
  1. Every non-visual, non-mtp tensor name is consumed by the planned map.
  2. Layer counts: 16 full-attention layers ((l+1)%4==0) with self_attn.*,
     48 GDN layers with linear_attn.*, matching crates/apxinf-model/src/qwen38/config.rs.
  3. (shape-level) Shapes/dtypes against qwen38/config.rs constants.

Output: summary to stdout; canonical name map to --out (default
devlocal/qwen38-musa/phase0/name-map.json) for reuse by the musa_qwen38 loader.

Usage:
  python3 scripts/audit_qwen38_hf_names.py --index <dir-or-index.json> [--require-shapes] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import struct
import sys
from collections import Counter

# ---- constants from crates/apxinf-model/src/qwen38/config.rs (checkpoint properties) ----
HIDDEN = 5120
INTERMEDIATE = 17408
VOCAB = 248320
LAYERS = 64
HEADS = 24          # full-attention query heads
KV_HEADS = 4        # full-attention kv heads
HEAD_DIM = 256      # full-attention head dim (q_proj carries q+gate: 2*HEADS*HEAD_DIM)
GDN_K_HEADS = 16
GDN_V_HEADS = 48
GDN_HEAD_DIM = 128
CONV_WIDTH = 4
QKV_WIDTH = 10240   # 16*128 q + 16*128 k + 48*128 v
Z_WIDTH = 6144      # 48*128 gate
A_B_WIDTH = GDN_K_HEADS * GDN_HEAD_DIM  # 2048, in_proj_a / in_proj_b rows


def is_full_attention(layer: int) -> bool:
    return (layer + 1) % 4 == 0


# HF name suffix -> (canonical target, expected shape or None=record-only)
# Shapes are [ *shape ] in torch/row-major order as stored in safetensors.
ATTN_SUFFIXES = {
    "self_attn.q_proj.weight": ("attn.q", [2 * HEADS * HEAD_DIM, HIDDEN]),
    "self_attn.k_proj.weight": ("attn.k", [KV_HEADS * HEAD_DIM, HIDDEN]),
    "self_attn.v_proj.weight": ("attn.v", [KV_HEADS * HEAD_DIM, HIDDEN]),
    "self_attn.o_proj.weight": ("attn.o", [HIDDEN, HEADS * HEAD_DIM]),
    "self_attn.q_norm.weight": ("attn.q_norm", [HEAD_DIM]),  # per-head norm, broadcast across heads
    "self_attn.k_norm.weight": ("attn.k_norm", [HEAD_DIM]),
}
GDN_SUFFIXES = {
    "linear_attn.in_proj_qkv.weight": ("gdn.qkv", [QKV_WIDTH, HIDDEN]),
    "linear_attn.in_proj_z.weight": ("gdn.z", [Z_WIDTH, HIDDEN]),
    # Qwen3-Next gated deltanet: a/b are per-v-head SCALARS (decay/beta), [num_v_heads, hidden]
    "linear_attn.in_proj_a.weight": ("gdn.a", [GDN_V_HEADS, HIDDEN]),
    "linear_attn.in_proj_b.weight": ("gdn.b", [GDN_V_HEADS, HIDDEN]),
    "linear_attn.A_log": ("gdn.a_log", [GDN_V_HEADS]),
    "linear_attn.dt_bias": ("gdn.dt_bias", [GDN_V_HEADS]),
    "linear_attn.conv1d.weight": ("gdn.conv", [QKV_WIDTH, 1, CONV_WIDTH]),
    "linear_attn.norm.weight": ("gdn.norm", [GDN_HEAD_DIM]),  # per-head RMSNorm over head_v_dim
    "linear_attn.out_proj.weight": ("gdn.out", [HIDDEN, Z_WIDTH]),
}
COMMON_SUFFIXES = {
    "input_layernorm.weight": ("norm.input", [HIDDEN]),
    "post_attention_layernorm.weight": ("norm.post", [HIDDEN]),
    "mlp.gate_proj.weight": ("mlp.gate", [INTERMEDIATE, HIDDEN]),
    "mlp.up_proj.weight": ("mlp.up", [INTERMEDIATE, HIDDEN]),
    "mlp.down_proj.weight": ("mlp.down", [HIDDEN, INTERMEDIATE]),
}
ROOT_NAMES = {
    "model.language_model.embed_tokens.weight": ("embed", [VOCAB, HIDDEN]),
    "model.language_model.norm.weight": ("final_norm", [HIDDEN]),
    "lm_head.weight": ("lm_head", [VOCAB, HIDDEN]),
}
MTP_PREFIX = "mtp."
VISUAL_PREFIX = "model.visual."

LAYER_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.(.+)$")


def classify(name: str):
    """-> (kind, canonical, expected_shape, layer) ; kind in
    {text, mtp, visual}; canonical/layer None for skipped kinds."""
    if name.startswith(VISUAL_PREFIX):
        return "visual", None, None, None
    if name.startswith(MTP_PREFIX):
        return "mtp", None, None, None
    if name in ROOT_NAMES:
        canonical, shape = ROOT_NAMES[name]
        return "text", canonical, shape, None
    m = LAYER_RE.match(name)
    if not m:
        return "unknown", None, None, None
    layer, suffix = int(m.group(1)), m.group(2)
    full = is_full_attention(layer)
    table = ATTN_SUFFIXES if full else GDN_SUFFIXES
    if suffix not in table and suffix not in COMMON_SUFFIXES:
        return "unknown", None, None, layer
    if full and suffix in GDN_SUFFIXES:
        return "wrong_kind", suffix, None, layer
    if not full and suffix in ATTN_SUFFIXES:
        return "wrong_kind", suffix, None, layer
    canonical, shape = (table if suffix in table else COMMON_SUFFIXES)[suffix]
    return "text", canonical, shape, layer


def read_shard_headers(ckpt_dir: pathlib.Path, shards: set[str]) -> dict:
    """Parse safetensors headers -> {name: (shape, dtype_str)}."""
    meta = {}
    for shard in sorted(shards):
        path = ckpt_dir / shard
        with open(path, "rb") as fh:
            (header_len,) = struct.unpack("<Q", fh.read(8))
            header = json.loads(fh.read(header_len))
        for name, info in header.items():
            if name == "__metadata__":
                continue
            meta[name] = (info.get("dtype"), info.get("shape"))
    return meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True,
                    help="checkpoint dir (preferred) or path to model.safetensors.index.json")
    ap.add_argument("--require-shapes", action="store_true",
                    help="fail if shard files are missing (shape-level audit)")
    ap.add_argument("--out", default="devlocal/qwen38-musa/phase0/name-map.json")
    args = ap.parse_args()

    idx_path = pathlib.Path(args.index)
    if idx_path.is_dir():
        cand = idx_path / "model.safetensors.index.json"
        if not cand.is_file():
            print(f"ERROR: {cand} not found", file=sys.stderr)
            return 2
        idx_path, ckpt_dir = cand, idx_path
    else:
        ckpt_dir = idx_path.parent

    index = json.loads(idx_path.read_text())
    weight_map: dict[str, str] = index["weight_map"]

    shards = set(weight_map.values())
    shard_files = {s for s in shards if (ckpt_dir / s).is_file()}
    have_shapes = len(shard_files) == len(shards)
    if args.require_shapes and not have_shapes:
        print(f"ERROR: only {len(shard_files)}/{len(shards)} shards present in {ckpt_dir}",
              file=sys.stderr)
        return 2
    header_meta = read_shard_headers(ckpt_dir, shard_files) if have_shapes else {}

    counts = Counter()
    unmapped, wrong_kind, shape_bad, dtype_bad = [], [], [], []
    name_map = {}
    record_only: dict[str, list] = {}

    for name in sorted(weight_map):
        kind, canonical, expected, layer = classify(name)
        counts[kind] += 1
        if kind in ("unknown", "wrong_kind"):
            (unmapped if kind == "unknown" else wrong_kind).append(name)
            continue
        if kind in ("mtp", "visual"):
            name_map[name] = {"kind": kind, "skip": True}
            continue
        entry = {"kind": "text", "target": canonical, "shard": weight_map[name]}
        if layer is not None:
            entry["layer"] = layer
        if header_meta:
            dtype, shape = header_meta.get(name, (None, None))
            entry["dtype"], entry["shape"] = dtype, shape
            if dtype not in ("BF16", "F32"):  # F32 allowed for A_log/dt_bias-style params
                dtype_bad.append((name, dtype))
            if expected is not None and shape != expected:
                shape_bad.append((name, shape, expected))
            if expected is None and shape is not None:
                record_only.setdefault(canonical, []).append(shape)
        name_map[name] = entry

    # layer-count invariants
    text_layers = {e["layer"] for e in name_map.values()
                   if e.get("kind") == "text" and "layer" in e}
    attn_layers = sorted(l for l in text_layers if is_full_attention(l))
    gdn_layers = sorted(l for l in text_layers if not is_full_attention(l))
    layer_ok = (len(attn_layers) == 16 and attn_layers == [l for l in range(LAYERS)
                                                           if is_full_attention(l)]
                and len(gdn_layers) == 48)

    print(f"checkpoint : {ckpt_dir}")
    print(f"tensors    : {len(weight_map)}  ({counts['text']} text, "
          f"{counts['mtp']} mtp skipped, {counts['visual']} visual skipped, "
          f"{counts['unknown']} unmapped, {counts['wrong_kind']} wrong-kind)")
    print(f"layers     : attn={len(attn_layers)} gdn={len(gdn_layers)} expected=16/48 "
          f"{'OK' if layer_ok else 'MISMATCH'}")
    print(f"mode       : {'name+shape+dtype' if have_shapes else 'name-only (shards pending)'}")
    if record_only:
        print("record-only shapes (assert in loader after inspection):")
        for canonical, shapes in sorted(record_only.items()):
            print(f"  {canonical:14s} {shapes[:2]}")
    for label, items in (("UNMAPPED", unmapped), ("WRONG-KIND", wrong_kind),
                         ("SHAPE-MISMATCH", shape_bad), ("DTYPE-UNEXPECTED", dtype_bad)):
        if items:
            print(f"{label}:")
            for it in items[:20]:
                print("  ", it)
    verdict = (not unmapped and not wrong_kind and not shape_bad and not dtype_bad
               and layer_ok)
    print("RESULT     :", "PASS — 0 unmapped text tensors" if verdict else "FAIL")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "summary": {"total": len(weight_map), "text": counts["text"],
                    "mtp": counts["mtp"], "visual": counts["visual"],
                    "attn_layers": len(attn_layers), "gdn_layers": len(gdn_layers),
                    "mode": "shape" if have_shapes else "name-only"},
        "record_only_shapes": record_only,
        "tensors": name_map,
    }, indent=1))
    print(f"name map   : {out}")
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
