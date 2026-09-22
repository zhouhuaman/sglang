# Problem K1 · gate_chunk_cumsum

> Files in this directory: `gate_chunk_cumsum_kernel.py` holds the torch reference
> `gate_chunk_cumsum_torch` and the triton kernel to be rewritten/optimized,
> `gate_chunk_cumsum_triton` (currently a converged baseline). You only modify the
> kernel; `test.py` handles comparison and timing.

## 0. Operator description

Pass each token's per-channel gating value through an activation, then take a
**chunk-local** prefix sum, producing the cumulative decay `gk` in log2 space (consumed by
the downstream `exp2`-based kernels). Decay deepens linearly with token count: within a
64-token chunk, for the same (b, h, channel), the k-th token's decay = the sum of the gates
of the first k tokens in the chunk. Across chunks there is **no carry-over** (each chunk
restarts from 0).

## 1. Inputs

| Tensor | shape | Meaning |
|---|---|---|
| `x` | [B, T, H, K] | each token's per-channel raw gate (not yet activated) |
| `A_log` | [H] | one log-scale scalar per head, controlling the head's decay strength |
| `dt_bias` | [H*K] | per-head per-channel bias, flattened [H*K]; `dt_bias[h*K+k]` belongs to (h,k) |
| `chunk_size` | int | constant 64 (do not change) |
| `scale` | float | constant `RCP_LN2 ≈ 1.4427` (=1/ln2; converts results from ln space to log2 space) |

Dimensions: `B`=batch; `T`=token index; `H`=head; `K`=channel (last dim of q/k/gk, K=128 here).
`NT = T/64`.

## 2. Outputs

| Tensor | shape | Meaning |
|---|---|---|
| `gk` | [B, T, H, K] | activated + chunk-local-prefix-summed cumulative gate (log2 space) |

## 3. Computation (first partition tiles → all computation inside one tile)

> Notation: `BT=chunk_size=64`; `softplus(v)=log(1+e^v)` (≈v for v≥20, avoids overflow); `exp` is
> the natural exponential. The prefix sum runs along the **time axis**, independently per channel —
> this operator is an elementwise transform of "activation + a prefix-sum taken in 64-step segments
> + log2 scaling", with no matrix multiply.

### Partition tiles: cut into B·H·NT tiles

Output `gk[B,T,H,K]` is fully parallel over (batch, head, chunk, channel). The converged kernel
maps one CTA to a **(batch,head)'s chunk × the whole K range**: K is not tiled (`BS=128 ≥ K` ⇒
`cdiv(K,BS)=1`), tile shape `[BT,K]`.

```
grid = (cdiv(K,BS), NT, B*H)      # target K=128, BS=128 → (1, NT, B*H), total B·H·NT tiles
program_id: (0)=channel block (always 0), (1)=chunk, (2)=packed (batch,head)
one tile: x[tc..tc+63, 0..K-1]; tc = first token of chunk = i_t*BT
```

### Single-tile formulas (tile = a (b,h) chunk block `[BT,K]`)

Load the chunk's raw gates `x_c` and bias; first activate elementwise, then prefix-sum along the
time axis (rows):

```
gate[BT,K] = -exp(A_log[h]) · softplus( x_c[BT,K] + dt_bias[h,:] )   # ① activation
gk[BT,K]   = RCP_LN2 · cumsum_rows( gate[BT,K] )                      # ② prefix sum × log2
```

- `dt_bias[h,:]` broadcasts per row: all tokens of one (h, channel) share one bias; `softplus`
  keeps the parenthesized term positive; `A_log[h]` is one scalar per head controlling the head's
  decay strength; the **minus sign** ⇒ `gate` is always negative ⇒ `gk` monotonically decreases
  (decay deepens with token count).
- `RCP_LN2≈1.4427` converts ln space to log2 space. It is deliberately left as a trailing multiply
  in this kernel — downstream K4/K5/K6 all use `exp2(gk)` (the log2-space exponent applied directly
  as the base), saving one base conversion; the hardware `exp2` is also cheaper than `exp`.

### Single-tile code (= kernel body; comments are the whole logic)

```
i_s, i_t, i_bh = tl.program_id(0..2)      # (channel block, chunk, packed (b,h)); b=i_bh//H, h=i_bh%H
tc = i_t*BT ;  s0 = i_s*BS
rows = tc + tl.arange(0, BT) ;  cols = s0 + tl.arange(0, BS)
mask = (rows[:,None] < T) & (cols[None,:] < K)          # out-of-bounds load → 0
ptr_x = x + b*T*H*K + h*K + rows[:,None]*(H*K) + cols[None,:]
b_x = tl.load(ptr_x, mask=mask, other=0.0).to(tl.float32)   # [BT,BS] raw gates
b_b = tl.load(dt_bias + h*K + cols, mask=cols<K, other=0.0).to(tl.float32)
b_x = b_x + b_b[None,:]                                   # ① add bias, broadcast per column
b_gate = -tl.exp(tl.load(A_log + h)) * _softplus(b_x)     # ② activation (A_log one scalar per head)
b_gate = tl.where(mask, b_gate, 0.0)      # zero invalid rows (before cumsum; safe for tail chunk)
b_gk = tl.cumsum(b_gate, axis=0) * scale  # ③ chunk-local prefix sum (axis=0 = time) × log2
ptr_o = o + b*T*H*K + h*K + rows[:,None]*(H*K) + cols[None,:]
tl.store(ptr_o, b_gk)                     # write back gk [B,T,H,K]
```

> Why the tail chunk is safe: `tl.cumsum` accumulates backward along axis=0; after invalid rows are
> zeroed, the trailing zeros only appear at the segment end and never disturb the prefix sums of the
> valid leading rows — equivalent to the real kernel's `boundary_check` returning 0.

## 4. Constraints & acceptance

- Scoring case: `B=1, T=16384, H=96, K=128` (NT=256, T%64==0); inputs/outputs fp32.
- Correctness: elementwise diff vs `gate_chunk_cumsum_torch` < `1e-2` (currently ~6e-5).
- **You may not change the default `chunk_size=64`** (breaking the upstream contract = invalid
  solution). `scale`, `A_log`, `dt_bias` are given constants/inputs; you may not bypass them
  (changing semantics / hard-coding a case's values = invalid solution).

```bash
source ../env.sh                        # container env (or `source ../env.sh 4` to pin an idle card)
python3 test.py                         # ① correctness: PASS + max_diff
msprof --output=./prof_k1 --application="python3 test.py --perf --repeats 7 --warmup 3" \
    && python3 test.py --report ./prof_k1        # ② baseline (msprof Task Duration per-call mean)
```

Official baseline ≈ **1.96 ms/call** (±10%); bar `max_diff < 1e-2`.
