# Problem K6 · gla_output

> Files in this directory: `gla_output_kernel.py` holds the torch reference `gla_output_torch` and
> the triton kernel to be rewritten/optimized, `gla_output_triton` (currently a converged
> baseline). You only modify the kernel; `test.py` handles comparison and timing.

## 0. Operator description

The last kernel of KDA: it adds two kinds of information to produce the final output —
**cross-chunk history** (the current token's query against the state compressed from "all chunks
before this one") and **intra-chunk exact attention** (causal token-to-token attention inside the
current chunk). Per (batch, chunk, head): the history part multiplies the state snapshot `h` by
`q·exp2(g)`; the intra-chunk part multiplies a causally masked `Aqk` by `v_new`. Both are matrix
multiplies, independent across chunks/heads.

## 1. Inputs

| Tensor | shape | Meaning |
|---|---|---|
| `q` | [B, T, H, K] | each token's query vector |
| `v_new` | [B, T, H, V] | corrected value (K5's residual v_new) |
| `g` | [B, T, H, K] | K1's cumulative gate `gk` (log2 space) |
| `Aqk` | [B, T, H, 64] | intra-chunk causal attention weights (K2/K3 output); row t's 64 columns = the chunk's 64 key positions |
| `h` | [B, NT, H, V, K] | K5 output: the compressed-state snapshot `state` at each chunk start |
| `scale` | float | constant `1/sqrt(K)` |

Dimensions: `B`=batch; `T`=token; `H`=head; `K`=query channel; `V`=value channel (target case
K=V=128). `NT = T/64`; token `t` belongs to chunk `c = t//64`, intra-chunk position `i = t mod 64`.

## 2. Outputs

| Tensor | shape | Meaning |
|---|---|---|
| `o` | [B, T, H, V] | final output, one V-dim vector per token |

## 3. Computation (first partition tiles → all computation inside one tile)

> Notation: `BT=64`; token `t` belongs to chunk `c=t//64`, in-block row `i=t%64`, `tc=64·c`.
> `h_c = h[b,c,h]` ([V,K]) is the compressed-state snapshot at the **start** of chunk c (K5 output;
> it holds only info from before chunk c → naturally causal); `A_c = Aqk[b,tc:tc+64,h]` ([BT,BT],
> row i, column j = in-chunk position j) is the intra-chunk causal score (K2/K3 output, already
> scaled). `tril(A)` zeroes the upper triangle (j>i), keeping the diagonal. `@`=matrix multiply;
> `⊙`=elementwise multiply. Both paths are matrix multiplies, independent per (chunk, head).

### Partition tiles: one CTA = a (chunk, V-slab) × HM heads

Output `o[B,T,H,V]` is independent per (batch, chunk, head). The converged kernel maps one CTA to a
chunk's `BT` rows × one V slab `BV`, looping over `HM` heads (target BV=128 ⇒ the V-slab is the
whole V, HM=16):

```
grid = (cdiv(V,BV), NT, B·(H//HM))    # → (1, NT, B·H//16); tile = (chunk, V-slab)
program_id(0)=V-slab, (1)=chunk, (2)=packed head group; one [BT,BV] accumulator per tile
```

### Single-tile formulas (tile = two paths summed into the final output)

A tile computes the 64 rows' V-slab output of a chunk; both paths accumulate into the **same fp32
accumulator**:

```
o_cross = ( q ⊙ exp2(g) · scale ) @ h_cᵀ     # ① cross-chunk: query reads compressed history  [BT,BV]
o_intra = tril(A_c) @ v_new_c                 # ② intra-chunk: exact attention to ≤i inside chunk [BT,BV]
o_c     = o_cross + o_intra                   # ③ sum = final output
```

- ① the query is first decayed by its own token's per-channel gate and multiplied by `scale`
  (≈ its "present-moment" strength), then dotted with row v of the state — the memory of "all
  earlier chunks" held in the compressed history; `h_c` only holds info before chunk c ⇒ causality
  is built in, no mask needed.
- ② only adds `j≤i` (history/self within this chunk); `tril(A_c)` zeroes A_c's j>i positions (a
  token does not look at later tokens in the same chunk). Row i's causal mask uses
  `tl.where(r>=c, A, 0.0)` (values set to 0 rather than a store-mask; Ascend MTE columns must be
  monotonic).
- ③ neither weight may be dropped; there is no intermediate `qg`/`o_cross` tensor — both merge in a
  register accumulator and are written back once as a whole block.

### Single-tile code (= kernel body; comments are the whole logic)

```
i_v, i_tg, i_hg = tl.program_id(0..2)     # (V-slab, chunk, head group); decode i_b/hg0
r = tl.arange(0, BT); c = tl.arange(0, BT)
for hh in tl.range(HM, num_stages=NS):    # head-merge loop (HM=16); head-invariant terms hoisted out
    b_o = tl.zeros([BT, BV], fp32)        # shared accumulator for both paths
    b_q = load q  [tc:tc+BT, :]      ; b_q = b_q * scale            # [BT,K]
    b_g = load g  [tc:tc+BT, :]                                      # [BT,K]
    b_qg = b_q * exp2(b_g)                                           # [BT,K]
    b_h  = load h  [c, i_v*BV:(i_v+1)*BV, :]                         # [BV,K]
    b_o += tl.dot(b_qg, tl.trans(b_h))     # ① [BT,K]@[K,BV] cross-chunk
    b_A = load Aqk [tc:tc+BT, :BT]                                    # [BT,BT]
    b_A = tl.where(r[:,None] >= c[None,:], b_A, 0.0)   # ② lower-triangular causal (incl. diagonal)
    b_v = load v_new [tc:tc+BT, i_v*BV:(i_v+1)*BV]                   # [BT,BV]
    b_o += tl.dot(b_A, b_v)               # ② [BT,BT]@[BT,BV] intra-chunk
    store o [tc:tc+BT, i_v*BV:(i_v+1)*BV] = b_o   # ③ both paths already summed in the accumulator
```

> The converged kernel hoists the head-invariant causal mask / boundary mask / offset vectors out of
> the head loop (scalar-addressing reduction; target case 6.94→4.75 ms). The K dim is not split into
> small dots (driver uses `BK=min(K,128)`; target K=128 → a single [BT,K]@[K,BV] dot). Tail chunk
> shorter than BT / tail V-slab shorter than BV: row/column masks set to 0 and the causal mask also
> zeroes out-of-bounds rows, never polluting the accumulator (the reference likewise pads with 0
> then trims).

## 4. Constraints & acceptance

- Scoring case: `B=1, T=16384, H=96, K=V=128` (NT=256, T%64==0); inputs/outputs fp32.
- Correctness: output `o` elementwise diff vs `gla_output_torch` < `1e-2` (currently ~1e-7).
- **You may not change the default `chunk_size=64`** (breaking the upstream contract = invalid
  solution). `Aqk`/`h`/`v_new`/`g` are given inputs (upstream-kernel outputs); you may not skip
  either path ①/② or change the two paths' weights.

```bash
source ../env.sh                        # container env (or `source ../env.sh 4` to pin an idle card)
python3 test.py                         # ① correctness: PASS + max_diff
msprof --output=./prof_k6 --application="python3 test.py --perf --repeats 7 --warmup 3" \
    && python3 test.py --report ./prof_k6        # ② baseline (msprof Task Duration per-call mean)
```

Official baseline ≈ **4.53 ms/call** (±10%); bar `max_diff < 1e-2`.
