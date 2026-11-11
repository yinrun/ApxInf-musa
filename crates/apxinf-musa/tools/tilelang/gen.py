#!/usr/bin/env python3
"""Compile target-supported TileLang operators into relocatable objects.

Full wrappers are obtained directly from JIT adapters on cold and cached builds.
The explicit output directory must be outside the operator source tree.
"""

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
import hashlib

sys.dont_write_bytecode = True
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # crates/apxinf-musa
KERNELS = ROOT / "kernels" / "tilelang"

FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
MUSA_PATH = Path(os.environ.get("MUSA_PATH", "/usr/local/musa-5.1.0"))
MCC = MUSA_PATH / "bin" / "mcc"
# 已知 offload arch → __MUSA_ARCH_LIST__ 数值；新架构两处一起加。
ARCH_NUM = {"mp_22": "220"}
ROLES = ("part", "reduce")
# 契约版本：旧 inventory 无 def_hash 字段，首跑自动全量重建。
GENERATED_BY = "apxinf-tilelang-gen/3"


def fnv1a(path: Path) -> str:
    value = FNV_OFFSET
    with path.open("rb") as handle:
        for byte in handle.read():
            value = ((value ^ byte) * FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return f"{value:016x}"


def fnv1a_bytes(data: bytes) -> str:
    value = FNV_OFFSET
    for byte in data:
        value = ((value ^ byte) * FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return f"{value:016x}"


def op_definition_digest(py: Path, op: dict, entries: dict, target: dict) -> str:
    """算子的输入身份：定义文件字节 + 算子参数 + 入口符号 + 生成器本体 + 契约版本。

    编辑 tile/shape/flags、入口编号位移（插入新算子）、或 gen.py 自身变化，
    都会改变 def_hash 从而强制重编——缓存键必须是输入身份，而非输出指纹。
    """
    payload = b"\x00".join([
        py.read_bytes(),
        json.dumps(op, sort_keys=True, ensure_ascii=False).encode(),
        json.dumps(entries, sort_keys=True).encode(),
        Path(__file__).read_bytes(),
        GENERATED_BY.encode(),
        json.dumps(target, sort_keys=True).encode(),
    ])
    return fnv1a_bytes(payload)


def load_module(py: Path):
    spec = importlib.util.spec_from_file_location(f"tilelang_kernel_{py.stem}", py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def tilelang_root() -> Path:
    import tilelang
    return Path(tilelang.__file__).resolve().parent.parent


def rename_symbols(source: str, entry: str) -> str:
    """把固定入口符号改成 kTl<N> 唯一入口（Task 1 实验证实的四个替换）。"""
    source = re.sub(r"\bcall\b", entry, source)
    source = re.sub(r"\binit\b", f"{entry}_init", source)
    source = re.sub(r"\bget_last_error\b", f"{entry}_last_error", source)
    source = re.sub(r"\bkernel_kernel\b", f"{entry}_kernel", source)
    return source


def compile_object(mu_file: Path, out_o: Path, flags: list,
                    mcc: Path, arch: str, arch_num: str) -> None:
    tl = tilelang_root()
    cmd = [
        str(mcc), str(mu_file), "-std=c++17", "-c", "-fPIC",
        f"--offload-arch={arch}", f"-D__MUSA_ARCH_LIST__={arch_num}",
        "-mllvm", "-mtgpu-alloc-shared-memory-from-zero=1",
        f"-I{tl / 'src'}",
        f"-I{tl / '3rdparty' / 'mutlass' / 'include'}",
        *flags,
        "-o", str(out_o),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"mcc failed for {mu_file.name}:\n{result.stdout}\n{result.stderr}")


def generated_source(obj):
    """Return the full wrapper from its owning JIT adapter, never guess a path."""
    source = obj.adapter.get_kernel_source(kernel_only=False)
    if not isinstance(source, str) or not all(
        re.search(r"\b" + name + r"\s*\(", source)
        for name in ("call", "init", "get_last_error")
    ):
        raise RuntimeError("TileLang adapter did not return a complete wrapper")
    return source


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arch", default="mp_22",
                        help="mcc offload arch; must be in ARCH_NUM unless --arch-num is given")
    parser.add_argument("--arch-num", default=None,
                        help="override the __MUSA_ARCH_LIST__ number for --arch")
    parser.add_argument("--mcc", type=Path, default=None,
                        help="mcc executable (default $MUSA_PATH/bin/mcc)")
    args = parser.parse_args()
    if args.out.resolve().is_relative_to(KERNELS.resolve()):
        parser.error("--out must be outside kernels/tilelang (Cargo watches this tree)")
    args.out.mkdir(parents=True, exist_ok=True)

    mcc = args.mcc if args.mcc is not None else MCC
    ops = []
    for py in sorted(KERNELS.glob("*.py")):
        ops.extend((py, dict(op)) for op in load_module(py).OPS
                   if args.arch in op["supported_arches"])
    ops.sort(key=lambda pair: pair[1]["id"])
    arch_num = args.arch_num or ARCH_NUM.get(args.arch)
    target_identity = {}
    if ops:
        if arch_num is None:
            raise RuntimeError(f"No architecture number configured for {args.arch}")
        mcc = mcc.resolve(strict=True)
        target_identity = {"arch": args.arch, "arch_num": arch_num, "mcc": str(mcc),
                  "mcc_sha256": hashlib.sha256(mcc.read_bytes()).hexdigest()}
    else:
        print(f"No TileLang operators support {args.arch}; using native operators")

    inventory_path = args.out / "inventory.json"
    old = {}
    if inventory_path.exists():
        old = {a["id"]: a
               for a in json.loads(inventory_path.read_text())["artifacts"]}

    artifacts = []
    for index, (py, op) in enumerate(ops):
        entries = {role: f"kTl{index}_{role}" for role in ROLES}

        # 增量：输入身份不变（定义/参数/入口/生成器）且产物未被篡改才复用旧 .o。
        # 输出指纹（fnv64）只回答「磁盘上的 .o 被动过没有」——编辑 .py 不会
        # 改变它，所以命中必须以输入哈希 def_hash 为准（review 修复）。
        cached = {}
        def_hash = op_definition_digest(py, op, entries, target_identity)
        for role in ROLES:
            target = args.out / f"{op['id']}.{role}.o"
            old_op = old.get(op["id"], {})
            if (target.exists()
                    and old_op.get("def_hash") == def_hash
                    and old_op.get("fnv64", {}).get(role) == fnv1a(target)):
                cached[role] = {"file": target.name, "digest": old_op["fnv64"][role]}

        if len(cached) != len(ROLES):
            mod = load_module(py)
            part_obj, reduce_obj = mod.build(op, entries["part"], entries["reduce"])
            sources = {"part": generated_source(part_obj), "reduce": generated_source(reduce_obj)}
            for role in ROLES:
                if role in cached:
                    continue
                mu_file = args.out / f"{op['id']}.{role}.mu"
                mu_file.write_text(rename_symbols(
                    sources[role], entries[role]))
                target = args.out / f"{op['id']}.{role}.o"
                compile_object(mu_file, target, op.get("flags", []),
                               mcc, args.arch, arch_num)
                mu_file.unlink()  # 源文件仅编译期需要，不入产物目录
                cached[role] = {"file": target.name, "digest": fnv1a(target)}

        artifacts.append({
            "id": op["id"], "type": op["type"], "shape": list(op["shape"]),
            "status": op["status"],
            "scratch_bytes": op["tile"]["splits"] * op["shape"][0] * op["shape"][1] * 4,
            "entries": entries,
            "def_hash": def_hash,
            "files": {role: cached[role]["file"] for role in ROLES},
            "fnv64": {role: cached[role]["digest"] for role in ROLES},
        })

    seen = set()
    for a in artifacts:
        key = tuple(a["shape"])
        assert key not in seen, f"duplicate TileLang shape {key}: {a['id']}"
        seen.add(key)

    inventory_path.write_text(json.dumps(
        {"generated_by": GENERATED_BY, "artifacts": artifacts},
        indent=2) + "\n")
    print(f"generated {len(artifacts)} kernels -> {inventory_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
