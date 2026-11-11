#!/usr/bin/env python3
"""Gold reference dump for Qwen3.8-27B (HF BF16) — Phase 0 item 0.3.

Runs the HF checkpoint text-only on a reference GPU (A100), hooks all 64
decoder layers, and writes a per-prompt NPZ bundle + greedy token gold that
the MUSA port's layer-compare / e2e verification replays against.

Prompts (deterministic, same LCG family as the CUDA acceptance recipe in
crates/apxinf-cuda-new/tests/qwen38_end_to_end.rs:4806 and
examples/qwen38_bench.rs:27-37):
    sanity8  — 8 tokens,   LCG seed 0x2545F4914F6CDD1D ^ 0xAAAA
    mid64    — 64 tokens,  LCG seed 0x2545F4914F6CDD1D ^ 0x5555
    lcg2048  — 2048 tokens,LCG seed 0x2545F4914F6CDD1D          (canonical)

Outputs (under --out-dir, default devlocal/qwen38-musa/gold/):
    sanity8.npz / mid64.npz / lcg2048.npz
        prompt_ids                    i64 [T]
        layer_in_{000..063}          f32 [T, 5120]   (hook input hidden)
        layer_out_{000..063}         f32 [T, 5120]   (hook output hidden)
        logits_last                  f32 [248320]    (last position)
    gold_greedy_tokens.txt            one line, comma-separated ids (greedy 128 from lcg2048)
    gold_greedy_tokens.md5            md5 of that line + "\\n"
    manifest.json                     versions, revision, seeds, tensor inventory

Greedy gold is generated twice and must be byte-identical (exit 1 otherwise).

Usage (A100):
    python3 scripts/gold_qwen38_musa_dump.py --model /path/Qwen3.8-27B \
        [--out devlocal/qwen38-musa/gold] [--device cuda:0] [--skip-greedy]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

import numpy as np
import torch

SEED = 0x2545F4914F6CDD1D
LCG_MUL = 6364136223846793005
LCG_ADD = 1442695040888963407
VOCAB = 248320
NUM_LAYERS = 64
HIDDEN = 5120


def lcg_ids(count: int, seed: int) -> list[int]:
    state = seed & 0xFFFFFFFFFFFFFFFF
    ids = []
    for _ in range(count):
        state = (state * LCG_MUL + LCG_ADD) & 0xFFFFFFFFFFFFFFFF
        ids.append((state >> 16) % (VOCAB - 1000))
    return ids


PROMPTS = {
    "sanity8": lcg_ids(8, SEED ^ 0xAAAA),
    "mid64": lcg_ids(64, SEED ^ 0x5555),
    "lcg2048": lcg_ids(2048, SEED),
}


def find_decoder_layers(model) -> list:
    """Locate the 64 text decoder layers, robust to wrapper naming."""
    import re
    cands = [(name, mod) for name, mod in model.named_modules()
             if re.search(r"(language_model\.)?layers\.\d+$", name)
             and re.fullmatch(r"(model\.)?(language_model\.)?model", name.split(".layers")[0] or "model")
             is None]
    # keep only direct children of a module whose name ends with 'layers'
    layers = [(n, m) for n, m in cands
              if n.rsplit(".layers", 1)[0].endswith(("model", "language_model.model", "language_model"))]
    # deduplicate parents: keep modules exactly one below a '*layers' container
    seen = {}
    for name, mod in cands:
        idx = int(name.rsplit(".", 1)[1])
        parent = name.rsplit(".", 1)[0]
        seen.setdefault(parent, {})[idx] = (name, mod)
    for parent, items in seen.items():
        if len(items) == NUM_LAYERS and max(items) == NUM_LAYERS - 1:
            return [items[i][1] for i in range(NUM_LAYERS)]
    raise SystemExit(f"could not locate {NUM_LAYERS} decoder layers; candidates: "
                     f"{[(p, len(v)) for p, v in seen.items()]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="local checkpoint dir (HF BF16)")
    ap.add_argument("--out", default="devlocal/qwen38-musa/gold")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--skip-greedy", action="store_true")
    args = ap.parse_args()

    import transformers
    from transformers import AutoModelForImageTextToText, AutoModelForCausalLM

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    model = None
    used_class = None
    for auto in (AutoModelForImageTextToText, AutoModelForCausalLM):
        try:
            model = auto.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                         low_cpu_mem_usage=True)
            used_class = auto.__name__
            break
        except Exception as exc:  # noqa: BLE001
            print(f"load via {auto.__name__} failed: {exc}", file=sys.stderr)
    if model is None:
        return 2
    model.to(args.device).eval()
    print(f"loaded via {used_class}; transformers {transformers.__version__}, "
          f"torch {torch.__version__}, dtype bfloat16")

    layers = find_decoder_layers(model)
    print(f"hooked {len(layers)} decoder layers")

    # ---- hooked forward passes -------------------------------------------------
    for tag, ids in PROMPTS.items():
        captured: dict[str, np.ndarray] = {}
        hooks = []

        def mk_hook(key):
            def fn(mod, inputs, output):
                t = output[0] if isinstance(output, tuple) else output
                captured[f"{key}"] = t.detach().float().cpu().numpy()
            return fn

        for i, layer in enumerate(layers):
            hooks.append(layer.register_forward_pre_hook(
                mk_hook(f"layer_in_{i:03d}") and (lambda mod, inputs, _k=f"layer_in_{i:03d}":
                                                 captured.__setitem__(
                                                     _k, inputs[0].detach().float().cpu().numpy()))))
            hooks.append(layer.register_forward_hook(mk_hook(f"layer_out_{i:03d}")))

        input_ids = torch.tensor([ids], dtype=torch.long, device=args.device)
        with torch.no_grad():
            logits = model(input_ids=input_ids, use_cache=False).logits
        for h in hooks:
            h.remove()

        captured["prompt_ids"] = np.asarray(ids, dtype=np.int64)
        captured["logits_last"] = logits[0, -1].detach().float().cpu().numpy()
        np.savez(out_dir / f"{tag}.npz", **captured)
        n_in = sum(k.startswith("layer_in_") for k in captured)
        n_out = sum(k.startswith("layer_out_") for k in captured)
        mb = sum(v.nbytes for v in captured.values()) / 1e6
        print(f"{tag}: {len(ids)} tokens, {n_in}+{n_out} layer tensors, "
              f"logits {captured['logits_last'].shape}, {mb:.0f} MB")

    # ---- greedy gold ------------------------------------------------------------
    manifest = {
        "model": args.model,
        "model_class": used_class,
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "dtype": "bfloat16",
        "device": args.device,
        "lcg": {"seed": SEED, "mul": LCG_MUL, "add": LCG_ADD,
                "formula": "(state>>16) % (VOCAB-1000)"},
        "prompts": {k: len(v) for k, v in PROMPTS.items()},
        "greedy": None,
    }

    if not args.skip_greedy:
        input_ids = torch.tensor([PROMPTS["lcg2048"]], dtype=torch.long, device=args.device)
        runs = []
        for attempt in (1, 2):
            torch.manual_seed(0)
            with torch.no_grad():
                out = model.generate(input_ids=input_ids, max_new_tokens=128,
                                     do_sample=False, num_beams=1,
                                     pad_token_id=0)
            new_tokens = out[0, input_ids.shape[1]:].tolist()
            line = ", ".join(str(t) for t in new_tokens)
            runs.append(line)
            print(f"greedy run {attempt}: {len(new_tokens)} tokens, tail={new_tokens[-6:]}")
        if runs[0] != runs[1]:
            print("ERROR: greedy runs differ — reference is not deterministic", file=sys.stderr)
            return 1
        (out_dir / "gold_greedy_tokens.txt").write_text(runs[0] + "\n")
        digest = hashlib.md5((runs[0] + "\n").encode()).hexdigest()
        (out_dir / "gold_greedy_tokens.md5").write_text(digest + "\n")
        manifest["greedy"] = {"tokens": 128, "line_md5": digest,
                              "format": "comma+space separated ids, trailing newline"}
        print(f"greedy gold: md5={digest}")

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"manifest: {out_dir/'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
