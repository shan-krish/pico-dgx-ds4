# Dual-variant serving: stock and abliterated from one engine

The same feature as the SGLang fork's `patches/sglang-dual-variant-abliteration.patch`, ported to this
engine. One checkpoint, one process set, one endpoint, both served names live at once:

```
y  <-  y - lambda_L * m * (r . y) * r        layers 10-35, per row, m = that row's session's variant
```

`r` is one 5120-float direction shared by all 26 layers, `lambda_L` is one float per layer, and `m` is 0
for a stock session (a bitwise no-op) and 1 for an abliterated one. The edit is rank 1, so the artifact is
**10 KB**, not the 1.02 GiB the 26 edited matrices would take.

## Where it is applied

`wo_b` writes the attention output into the residual stream. In this engine that output exists twice:

| path | site | what the kernel sees |
|---|---|---|
| decode / verify / draft steps | `model.cu`, after the `gemv_fp8`/`gemm_fp8_p` of wo_b and before `ops::allreduce_hc` | each rank's **partial sums** (`part`), with the row's session slot |
| prefill chunks | `prefill.cu`, after `sum4_bf16_p` and before `boundary_p` | this rank's **summed rows** (`p.sub`), one variant for the chunk |

The projection is linear, so applying it to partials before the all-reduce gives exactly the same result
as applying it to the sum afterwards: no extra collective, no change to any existing kernel's arithmetic.
Stock rows return before touching a value.

The draft module (DSpark) is **not** touched: its weights were never abliterated.

## How a row knows its variant

Variants belong to sessions, and sessions live in slots. `StepIn::slot` already carries every row's slot
(for every path: one-session steps, batch steps, verify steps), so the kernel looks the variant up in a
`kMaxSlots` table the server sends with each command (`Cmd::slot_var`). No per-row bookkeeping exists in
the model.

## Request path

1. `dsfront` reads the request's `model`. The abliterated name maps to variant 1, anything else to 0.
2. Session ids are minted with the variant in the low bit (`new_id` = `next_id << 1 | variant`), and
   `SessionTable::plan` only matches, forks or reuses sessions of the same variant. A stock request can
   therefore never share KV with an abliterated one — the ids differ before the engine sees them.
3. `FLAG_ABLIT` goes in `SubmitHeader::flags`; rank 0's server stores it on the session and rejects
   (with an error) a request that tries to reuse a session id under the other variant.
4. Each command carries `slot_var[]`; every rank applies it before running the step.

## The artifact

`--ablit <dir>` on the rank binary reads `<dir>/ablit.bin`:

```
"ABLT" | u32 version=1 | u32 K=5120 | u32 nlam=26 | u32 lo=10 | u32 hi=35 | f32 r[5120] | f32 lam[26]
```

made from the SGLang side's `ablit_direction.safetensors` (r, lambda) and `.json` (lambda_layer_order).

## Configuration

| where | flag | meaning |
|---|---|---|
| rank binary | `--ablit ~/models/ablit-dual` | arms the hook; every slot defaults to stock |
| dsfront | `--model-name NAME` | the stock served name |
| dsfront | `--ablit-model-name NAME` | the abliterated served name (both appear in `/v1/models`) |

## Verification

Done here, on one GPU, beside a serving engine:

- `engine/kernels/tests/ablit_test.cu` — mixed rows (alternating slots), per-row activation, stock rows
  bitwise unchanged, uniform (prefill) mode, and `all_var = 0` bitwise no-op. `PASS`, max |err| 0.004.
- `dsw_convert` on the official checkpoint: `1,609 tensors, 72.38 GiB/rank`, 23–29 s per rank.
- `engram_pack` + `make_hashmap_stdlib.py`: 50.7 GB/rank, and the map matched the one the engine was
  validated with.

Not yet done, and it needs the ring:

- end-to-end on four ranks (the SGLang ring owns the CX7 ports; this engine cannot start beside it),
- the quality gate (qeval, the 333-prompt abliteration eval with an LLM judge, needles),
- perf against the published 8.7k–9.5k prefill,
- an in-engine check that a mixed batch (stock + abliterated sessions in one step) gives each row the
  answer it gets alone.

## Risks

- The hook adds one kernel launch per edited layer: 26 per step (decode) or per chunk (prefill). Its
  arithmetic is one dot + one axpy over 5120 values per row, but the launches are what to watch on a
  step that is already launch-bound.
- `StepIn::slot` must be valid for every path; it is filled by `next_in`, `launch_batch` and the spec
  fills, and prefill bypasses it (the uniform mode). Any new path must preserve that.
- The engine refuses to start if `--ablit` names a directory without `ablit.bin`, rather than silently
  serving stock under both names.