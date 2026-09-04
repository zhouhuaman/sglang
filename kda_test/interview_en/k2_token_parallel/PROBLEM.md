# Problem K2 · token_parallel

> Files in this directory: `token_parallel_kernel.py` holds the torch reference
> `token_parallel_torch` and the triton kernel to be rewritten/optimized, `token_parallel_triton`.
> You only modify the kernel; `test.py` handles comparison and timing.

## 0. Operator description

The sequence is cut into 64-token chunks, and each chunk into 16-token windows. For each
16-token window, compute the causal score table of its 16 rows × 16 columns and write two
outputs in two flavors: **Aqk** (query×key, keeps the diagonal) and **Akk** ((k·β)×k, drops the
diagonal). Windows are independent of one another.

## 1. Inputs

| Tensor | shape | Meaning |
|---|---|---|
| `q` | [B, T, H, K] | each token's query vector |
| `k` | [B, T, H, K] | each token's key vector |
| `gk` | [B, T, H, K] | each token's per-channel gate (log2 space), controlling decay |
| `beta` | [B, T, H] | one scalar gate per token |
| `scale` | float | constant `1/sqrt(K)` |

Dimensions: `B`=batch; `T`=token index 0..T−1; `H`=head; `K`=feature channel (length of the last
dim of q/k/gk).

## 2. Outputs

| Tensor | shape | Meaning |
|---|---|---|
| `Aqk` | [B, T, H, 64] | one row of 64 columns per token = the chunk's 64 key positions |
| `Akk` | [B, T, H, 16] | one row of 16 columns per token = the window's 16 compact key positions |

## 3. Computation (first partition tiles → all computation inside one tile)

> Notation: `BT=64`, `BC=16`; token `t`'s window starts at `s=(t//16)*16`, the window's column
> start within the chunk is `w=s%64`. A row token only scores tokens of the **same window** that
> are "itself and earlier" (causal), so each table has only the 4 diagonal 16×16 lower-triangular
> blocks nonzero. `⟨·,·⟩`=K-dim dot product; `⊙`=elementwise multiply; `exp2(v)=2**v`.

### Partition tiles: one CTA = a (chunk, HM heads) whole block

Outputs `Aqk[B,T,H,BT]` (row t's 64 columns = the chunk's 64 key positions) and
`Akk[B,T,H,BC]` (row t's 16 columns = the window's 16 **compact** key positions) are mutually
independent per (chunk, head). The converged kernel maps one CTA to the **whole chunk's 64 rows ×
all K channels** and loops over `HM` heads (target HM=16):

```
grid = (cdiv(T,BT), B·(H//HM))     # → (NT, B·H//16); tile = (chunk, HM heads)
load the whole chunk q/k/g/β [BT,K] once, compute two [BT,BT] tables, block-mask then write back
wide / compact
```

### Single-tile formulas (tile = a (b,h) whole chunk)

```
split the exponential exp2(g[i]-g[j]) = exp2(g[i])·exp2(-g[j]) ⇒ pre-multiply row/column factors:
qe = q·exp2(g)·scale    ,  ke = k·exp2(-g)          # each [BT,K]
Aqk_full = qe @ keᵀ      → keep (diagonal 16-block) ∧ (within block j≤i)      # incl. diagonal
Akk_full = ((k⊙β)⊙exp2(g)) @ keᵀ  → keep (diagonal 16-block) ∧ (within block j<i)   # excl. diagonal
```

- Aqk/Akk differ only in the **row vector** (Aqk uses `q`, Akk uses `k·β`) and whether the
  diagonal is kept; both share the gated-key column `ke`.
- All other blocks/elements are set to 0. Aqk is written back wide as `[B,T,H,64]`; Akk's diagonal
  segment of each row is gathered into 16 columns and written back compact as `[B,T,H,16]`.

### Single-tile code (= kernel body; HM head loop; comments are the whole logic)

```
i_cg, i_hg = tl.program_id(0..1)      # (chunk, head group); decode i_b, chunk start
load whole chunk: qc,kc,gc [BT,K] (out-of-bounds rows → 0), betac [BT]
eg = exp2(gc) ;  eneg = exp2(-gc) ;  ke = kc * eneg       # gated-key column (shared)
Aqk_full = tl.dot(qc * eg * scale, tl.trans(ke))          # [BT,K]@[K,BT] → [BT,BT]
Akk_full = tl.dot((kc * betac[:,None]) * eg, tl.trans(ke))
keep   = (block r//BC == block c//BC) & (in-row r%BC ≥ in-col c%BC)   # diag block ∧ incl-diag lower tri
strict = (block r//BC == block c//BC) & (in-row r%BC > in-col c%BC)   # exclude diagonal
Aqk_full = tl.where(keep,   Aqk_full, 0.0)
Akk_full = tl.where(strict, Akk_full, 0.0)
tl.store(Aqk + row t writes width BT, Aqk_full)                    # wide [BT,BT]
Akk_diag = tl.gather(Akk_full, per-row diagonal-segment column idxs, axis=1)    # gather [BT,BC]
tl.store(Akk + row t writes width BC, Akk_diag)                    # compact [B,T,H,16]
```

> The converged kernel fuses a chunk's 4 windows into one whole-chunk `[64,K]@[K,64]` big dot
> (fewer small dots than per-window), then zeroes everything outside "diagonal 16-block ∧
> within-block causal" — numerically identical to per-window. Ascend MTE columns must be
> monotonic: Akk's compact column mapping is gathered in-kernel via `tl.gather` rather than a
> non-monotonic wide write (the output buffer is padded to NT·BT; stores carry no mask).

## 4. Constraints & acceptance

- Scoring case: `B=1, T=16384, H=96, K=128` (T%64==0); inputs/outputs fp32.
- Correctness: elementwise diff vs `token_parallel_torch` < `1e-2` (currently ~1e-7).
- **You may not change the default `chunk_size=64 / sub_chunk_size=16`** (breaking the upstream
  contract = invalid solution).
- When a tail chunk is shorter than 64, the reference pads with 0 then trims; real-token values are
  unaffected.

```bash
source ../env.sh                        # container env (or `source ../env.sh 4` to pin an idle card)
python3 test.py                         # ① correctness: PASS + max_diff
msprof --output=./prof_k2 --application="python3 test.py --perf --repeats 7 --warmup 3" \
    && python3 test.py --report ./prof_k2        # ② baseline (msprof Task Duration per-call mean)
```

Official baseline ≈ **9.5–9.8 ms/call** (±10%); bar `max_diff < 1e-2`.
