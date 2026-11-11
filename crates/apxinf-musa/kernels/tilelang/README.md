# TileLang 算子目录

本目录是 TileLang 算子定义的作者入口——TileLang 是 APxInf 的第四种算子实现来源
（与 `kernels/asm` 手写汇编、`kernels/custom` MUSA-C、厂商库并列）。设计方案见
`doc/tilelang-kernels.md` 与 `tools/tilelang/gen.py` 的 docstring。

## 目录约定

- 每个算子一个 `.py` 文件，暴露两个名字：
  - `OPS: list[dict]` —— 算子元数据（静态描述）；
  - `build(op: dict, entry_part: str, entry_reduce: str)` —— 编译入口，返回
    `(part, reduce)` 两个 tilelang jit 对象。
- 构建产物位于 `OUT_DIR/tilelang/`，build.rs 显式传入 `--out`；禁止输出到源码目录。
- `supported_arches` 声明算子的目标架构；当前两个算子仅支持 `mp_22`，其它目标生成空表并使用 native。

## OPS 字段

| 字段 | 含义 |
|---|---|
| `id` | 全局唯一标识（stable，进 inventory.json） |
| `type` | 算子类型（首期仅 `gemm-splitk-int8`；新类型需扩展 build.rs 的 Rust 侧表结构） |
| `shape` | `(m, n, k)`；调度按 `(m, n, k)` 精确匹配 |
| `tile` | tile 配置（bm/bn/bk/splits），调优结果直接改这里 |
| `flags` | 传给 tilelang jit 的 `compile_flags`（实测 `-Od3 -O2` 最优） |
| `status` | `enabled` / `experimental` / `off`——构建期表与运行时策略都用它 |

## 添加新算子

1. 新建 `.py`，按上述接口写 `OPS` + `build`；
2. 构建（feature `tilelang` 默认开启）会自动发现并编译它——不需要改 Rust；
3. MR 必须附带三件套：单算子微基准（`tools/tilelang/` 的 harness）、端到端
   L1 P50、gold bundle 精度对比。

## 注意

- tilelang 的 jit 依赖 `inspect.getsource`：算子 `.py` 必须以文件方式被 import
  （gen.py 已如此），不要在 REPL/heredoc 里定义 kernel。
- 入口符号唯一性（`kTl<N>_part` 等）由 gen.py 在生成源码上改名保证，算子文件
  无需关心。

## 缓存与源码归属

缓存身份包含定义、参数、入口、生成器、目标架构、mcc 路径与内容哈希。
产物位于 OUT_DIR，随对应 Cargo target 清理。定义加载不写 __pycache__。
旧版本的 generated/ 与 __pycache__/ 不再使用，仅在开启 feature 时告警，不阻止构建。
完整 wrapper 直接来自 JIT adapter.get_kernel_source(kernel_only=False)，
不扫描共享缓存、不猜测源码归属；相同 reduce 的进程内复用无需重跑。
