# TileLang Operators in the MUSA Backend

TileLang is the MUSA backend's fourth operator source, alongside hand-written
assembly (`kernels/asm`), MUSA-C kernels (`kernels/custom`) and the vendor
libraries. An operator whose tile-language description beats every
hand-written candidate is written once in TileLang; the build compiles it into
the same kernel static library as everything else. The first operators are the
two INT8 decode GEMMs the migration experiment validated (the action MLP down
projection and the attention output); the skeleton is operator-generic.

This document is the contract between the definitions, the generator, the
build and the runtime. The author-facing quickstart is
[`kernels/tilelang/README.md`](../crates/apxinf-musa/kernels/tilelang/README.md);
the user-facing summary is in [musa-backend.md](musa-backend.md).

## Definitions: `crates/apxinf-musa/kernels/tilelang/*.py`

One file per operator family, exposing `OPS` (metadata: `id`, `type`, `shape`
= `(m, n, k)`, `tile`, `flags`, `status`) and
`build(op, entry_part, entry_reduce)` compiling the kernels through the
TileLang JIT. `id` ordering decides the unique entry symbols (`kTl<N>_*`),
which gen.py renames on the generated sources — definitions never see them.
Tuning results are edited directly in `tile` / `flags`.

## Generation: `tools/tilelang/gen.py`, driven from `build.rs`

For every op: build via the JIT, locate the generated sources, rename the
entry symbols, compile with `mcc` into a relocatable object (`.mu` source, the
offload arch and `__MUSA_ARCH_LIST__` come from `--arch` / `--arch-num`, and
the toolchain from `--mcc` — `build.rs` passes its own resolved values), then
FNV-1a fingerprints and the `inventory.json` below. Nothing is published
unless every op succeeded: the inventory is written once, after all objects.

Outputs live in `OUT_DIR/tilelang`, explicitly passed with `--out`, outside
Cargo's watched source tree. Legacy `kernels/tilelang/generated` and
`__pycache__` directories are ignored with a warning when TileLang is enabled;
no files are deleted automatically and feature-off builds do not inspect them. Python bytecode writes are disabled.

The cache identity covers definition bytes, op metadata, entry symbols,
generator version and bytes, target architecture/number, and mcc path/content
hash. A different target or compiler cannot reuse another target's objects.
Full wrapper source comes directly from each JIT adapter's
`get_kernel_source(kernel_only=False)`, including in-memory reused kernels.
There is no shared-cache scan, timestamp heuristic, or retry requirement.

Each definition declares `supported_arches`. The initial operators support
only `mp_22`; other targets produce an empty table without importing TileLang
or invoking mcc. This is native fallback, not a claim of new-architecture
kernel support. Supported targets still require the TileLang toolchain.

`inventory.json` records, per artifact: `id`, `type`, `shape`, `status`,
`scratch_bytes` (`splits * m * n * 4`), `entries`, `def_hash`, `files` and the
`fnv64` output fingerprints — plus a global `generated_by`. Shapes must be
unique; the generator asserts it.

## Build semantics: the `tilelang` feature

On for `apxinf-musa` by default; the model, binding and root crates forward it
as explicit opt-ins (`musa-tilelang` — their dependencies keep
the crate's defaults off, mirroring `musa-mttx`). Three states: **on with the
toolchain** generates, links and fills the compile-time table
(`TILELANG_GEMMS`); **on without the toolchain** is a hard build error naming
what to install; **off** generates nothing, leaves the table empty, and every
GEMM keeps the native route. `apxinf_musa::TILELANG_AVAILABLE` says which
build a process got: true only when the generated operator table is nonempty.

The definition `.py` files and the generator are kernel inputs in full: they
are hashed into the kernel build id (`musa-kb3-*`) and watched file by file —
swapping a TileLang operator changes the id, so measurements keyed on it
cannot mistake one kernel set for another.

## Runtime: dispatch, weights, escape hatch

`launch_quantized` consults the table before the assembly tiles; the weights'
KN prepack keys on the same table (any served `(n, k)` is packed, whatever the
group shape), so the two predicates cannot disagree. A served weight that
nonetheless arrives without its KN pack stays on the native route with a
one-time warning instead of failing the call. `APXINF_TILELANG=off` disables
every linked operator for the run without rebuilding; the value is read once
per process. The first dispatch prints one line (`TileLang route:` …), so a
run's route adoption is visible in its log.

## Adding an operator

Write the definition, build (nothing else changes), and bring to the MR: a
single-operator microbenchmark, the end-to-end L1 p50, and the gold-bundle
accuracy comparison. A device test comparing the kernel against a CPU
reference bit for bit lives in `kernels/gemm/tilelang.rs` — extend it when the
operator family changes; integer GEMM accumulates exactly, so the split order
cannot hide a numerical defect.
