# Triton Kernel Programming Test: Chunkwise Linear Attention

## 1. Background

### 1.1 The Problem: Attention Bottleneck in Long-Context Inference

In the Transformer architecture, standard Softmax Attention requires every token to compute similarity against all preceding tokens:

```
For a token at position t:
  output[t] = softmax(q[t] · k[0], q[t] · k[1], ..., q[t] · k[T-1]) @ v
```

When the sequence length T = 128K, the attention matrix is `T × T` (~16 billion elements), with O(T²) computational complexity. This cannot fit on a single chip and is the core bottleneck for long-context inference.

### 1.2 The Solution: Chunkwise Linear Attention

KDA (Kimi Delta Attention) combines Linear Attention with chunking to reduce O(T²) computation to O(T):

**Core Idea — Chunked Processing:**

Split T tokens into NT = ceil(T/64) chunks of `chunk_size=64`. Within each chunk, matrix multiplications run in parallel at O(T × 64). Across chunks, a compact "state matrix" carries historical information serially. Overall complexity drops from O(T²) to O(T × 64). At T = 128K, this is ~8 million operations versus 16 billion.

```
T tokens divided into NT chunks:
  |---- Chunk 0 (64 tokens) ----|---- Chunk 1 (64 tokens) ----|---- Chunk 2 (64 tokens) ----| ...

Each chunk is further divided into 4 sub-chunks (16 tokens each):
  | SC0 (0..15) | SC1 (16..31) | SC2 (32..47) | SC3 (48..63) |
```

**Three Key Mechanisms:**

**(a) Gate (Gated Decay):** Each token has an independent "forgetting rate" per key channel. Gate values are negative — after exponentiation they become decay factors in (0, 1], controlling how much old information is retained in the state matrix.

```
gate[t, k] = -exp(A_log[h]) * softplus(raw_gate[t, k] + dt_bias[h, k])
```

Since gate < 0, exp(gate) < 1. The more negative the gate (larger absolute value), the faster old information decays. Each channel learns a different decay rate, allowing the model to flexibly choose how long to retain history.

**(b) Delta Rule (Incremental Update):** Avoid storing redundant information in the state matrix. Predict the value using the current key against the state, then store only the unpredictable residual (delta).

```
predict = k[t] @ state           ← query historical state with current key, predict value
delta   = v[t] - predict         ← keep only the unpredictable new information
state   = state * decay + k[t]^T @ delta   ← decay old info + write new info
```

If the current token's information is fully predictable from historical state, delta = 0 and nothing is written. This "incremental encoding" keeps the state matrix compact.

**(c) State Matrix h:** A `[K, V]` matrix (K rows, V columns) that compresses all tokens before chunk c. Readout: `q[K] @ h[K, V] = [V]`, i.e., a gated query reads cross-chunk historical information from the state.

```
h[c]  = compressed information of all tokens in chunks 0, 1, ..., c-1
o_cross = (q * exp2(g)) @ h[c] * scale    ← read all history from h
```

### 1.3 Full Computation Pipeline

KDA's 6 computation steps form a strict data-dependency chain (each step's output feeds the next):

```
Step 1: Gate Cumsum
  raw_gate → activate to gate → chunk-local prefix sum → log2 conversion → g_cumsum
                                          │
                                          ▼
Step 2: Token Parallel (Diagonal Aqk/Akk)
  q, k, beta, g_cumsum → Aqk, Akk (diagonal blocks)
                                          │
                                          ▼
Step 3: Inter Solve (Block-Tridiagonal Matrix Inversion)
  Aqk, Akk → Akk_inv (decoupling matrix)
                                          │
          ┌───────────────────────────────┘
          ▼
Step 4: Recompute W/U
  Akk_inv, k, v, beta, g_cumsum → w, u, kg
                                          │
                                          ▼
Step 5: Delta Rule H (State Update)
  w, u, kg → h (state matrix snapshots), v_new
                                          │
                                          ▼
Step 6: GLA Output (Final Output Fusion)
  q, v_new, g_cumsum, Aqk, h → o (final attention output)
```

**This test selects Steps 1, 2, and 6 as three programming problems of progressive difficulty:**

| Problem | Corresponding Step | Core Concept | Difficulty |
|---------|-------------------|--------------|------------|
| Problem 1 | Step 1: Gate Cumsum | Gate activation + chunk-local prefix sum | L1 Beginner |
| Problem 2 | Step 2: Token Parallel | Per-token parallelism + gated dot product + causal masking | L2 Intermediate |
| Problem 3 | Step 6: GLA Output | Dual-path fusion + matrix multiplication + K-dimension loop | L3 Advanced |

The three problems form a data-dependency chain: Problem 1's `g_cumsum` → Problem 2's `Aqk` → Problem 3's `o`.

---

## 2. Programming Problems

### 2.1 Conventions

**Target platform:** Triton-Ascend on Ascend 910B2/910B3.

The following conventions apply to all problems:
- `B = 1` (single batch, simplified)
- `BT = 64` (chunk size, each chunk has exactly 64 tokens)
- `H`, `K`, `V` are compile-time constants given per problem
- All Triton kernels use `num_warps=1`. On Ascend, `num_warps` maps a program group to a single AI Core — there is no CUDA-style SIMT warp, but the Triton abstraction handles the mapping transparently.
- Input tensors are bf16 or fp32; all computation and accumulation uses fp32
- Fixed-length sequences only (no VARLEN)

**Ascend-specific constraints (mandatory):**

| # | Constraint | Rule |
|---|-----------|------|
| C1 | **UB capacity** | All live tile data per kernel iteration must fit in 192 KB (Unified Buffer per AI Core). |
| C2 | **Grid size** | Grid size ≤ total AI Core count (20 on 910B2, 24 on 910B3). Use the `TRITON_ALL_BLOCKS_PARALLEL=1` environment variable to avoid multi-round scheduling when grid exceeds core count. |
| C3 | **1D grid preferred** | 2D/3D grids must match physical core topology. Prefer 1D grids and compute multi-dimensional indices manually: `pid_m = pid // num_pid_n; pid_n = pid % num_pid_n`. |
| C4 | **int64/int32 → scalar fallback** | `tl.arange(0, N)` returns `int64` by default. int64 ADD/CMP and int32 LT/GT/LE/GE comparisons degrade to scalar execution (32-128x slower). Cast to fp32 before any comparison: `cols = tl.arange(0, N).to(tl.float32)`. |
| C5 | **No break / continue / return** | The Ascend backend does not support early loop exit (`break`), skip (`continue`), or mid-function `return`. Use mask-based iteration instead. |
| C6 | **Prefer `tl.make_block_ptr`** | Avoid manual pointer arithmetic in `tl.load`/`tl.store` offsets — the Ascend compiler may compute the offset before evaluating the mask, causing DDR out-of-bounds. Use `tl.make_block_ptr` with `boundary_check` and `padding_option="zero"`. |
| C7 | **Matmul alignment** | For `tl.dot`, the N-dimension tile size × dtype bytes must be a multiple of 512 B. Example: for fp16/bf16 (2 B), BLOCK_N must be a multiple of 256. |


---

### Problem 1 (L1): Gate Chunk Cumsum — Gate Activation and Chunk-Local Prefix Sum

#### Position in the Pipeline

This problem corresponds to **Step 1: Gate Cumsum**. The raw gate values `raw_gate` come from upstream network output without any processing. This step activates them into valid gate values, computes a prefix sum within each chunk, and provides `g_cumsum` for all downstream steps.

#### Problem Description

Implement the `gate_chunk_cumsum` kernel, which performs three operations independently per chunk:

**Operation 1: Gate Activation**

Activate raw gate values:

```
gate[t, h, k] = -exp(A_log[h]) * softplus(raw_gate[t, h, k] + dt_bias[h, k])
```

where `softplus(x) = log(1 + exp(x))`. The factor `-exp(A_log[h])` is a negative head-level scale that ensures gate < 0, so `exp(gate) < 1` acts as a decay factor for downstream steps. `dt_bias` is a learnable per-head per-channel bias.

**Operation 2: Chunk-Local Cumsum**

Compute prefix sum along the time axis within each chunk. **Critical constraint: cumsum resets at chunk boundaries** — each chunk starts fresh from 0.

```
For channel k in chunk 0 (64 tokens):
  Input gate (activated):  [g0,    g1,    g2,    ..., g63]
  Output cumsum:           [g0,  g0+g1, g0+g1+g2, ..., g0+g1+...+g63]

For the same channel k in chunk 1 (restarting from 0):
  Input gate:              [g64,      g65,      ..., g127]
  Output cumsum:           [g64,    g64+g65,   ..., g64+g65+...+g127]
```

This design ensures that downstream chunk-wise attention computes intra-chunk decay independently, without needing a global cumulative value across chunks.

**Operation 3: Log2 Space Conversion**

```
output = cumsum_result * RCP_LN2
```

where `RCP_LN2 = 1 / ln(2) ≈ 1.442695`. Since `ln(x) * (1/ln(2)) = log2(x)`, this converts values from natural-log space to log2 space. Downstream kernels use the hardware-efficient `exp2()` rather than `exp()` to recover the decay factor.

#### Input/Output Specification

| Parameter | Shape | Dtype | Description |
|-----------|-------|-------|-------------|
| `raw_gate` | `[B, T, H, K]` | fp32 | Raw gate values (not yet activated) |
| `A_log` | `[H]` | fp32 | Per-head log-scale parameter |
| `dt_bias` | `[H, K]` | fp32 | Per-head per-channel bias |
| `scale` | scalar | fp32 | Output scale factor, passed as `RCP_LN2 = 1.4426950216293335` |
| `output` (return) | `[B, T, H, K]` | fp32 | Activated gate cumsum (in log2 space) |

Compile-time constants: `BT=64` (chunk size), `BS=32` (K-dimension tile size).

#### Computation Diagram

```
Each program processes a [BT, BS] = [64, 32] tile:

     K dim (BS=32 channels)
  ┌──────────────────────────────┐
  │ g[0,0..31]   g[0,32..63]  ...│  ← program handles t0..t63, k0..k31
  │ g[1,0..31]                   │
  │ ...                          │
  │ g[63,0..31]                  │
  └──────────────────────────────┘
  ↑ cumsum along time axis (axis=0)

Each channel is independent; cumsum resets at every chunk boundary.
```

#### Reference Implementation (PyTorch, for verification)

```python
import torch
import math

RCP_LN2 = 1.4426950216293335

def reference_gate_chunk_cumsum(raw_gate, A_log, dt_bias, chunk_size=64):
    """
    raw_gate: [B, T, H, K]  fp32
    A_log:    [H]           fp32
    dt_bias:  [H, K]        fp32
    Returns:  [B, T, H, K]  fp32
    """
    B, T, H, K = raw_gate.shape
    # Step 1: Gate activation
    gate = -torch.exp(A_log)[None, None, :, None] * torch.nn.functional.softplus(
        raw_gate + dt_bias[None, None, :, :]
    )

    # Step 2: Chunk-local cumsum
    output = torch.zeros_like(gate)
    NT = (T + chunk_size - 1) // chunk_size
    for c in range(NT):
        start = c * chunk_size
        end = min(start + chunk_size, T)
        output[:, start:end] = torch.cumsum(gate[:, start:end], dim=1)

    # Step 3: log2 space conversion
    output = output * RCP_LN2
    return output
```

#### Completion Criteria

The Triton kernel output must achieve **RMSE < 1e-5** against the reference implementation, and satisfy:
- Each chunk's cumsum starts from 0; no accumulation across chunk boundaries
- softplus uses linear approximation for x >= 20 to avoid numerical overflow
- Output dtype is fp32

---

### Problem 2 (L2): Token Parallel — Diagonal Aqk/Akk Computation

#### Position in the Pipeline

This problem corresponds to **Step 2: Token Parallel**. The `g_cumsum` output from Problem 1 (gated cumulative values in log2 space) is used to compute gated dot products within each sub-chunk, producing Aqk (diagonal blocks of the attention weight matrix) and Akk (diagonal blocks of the key-key dependency matrix). These two matrices are the inputs for Step 3's inversion and decoupling.

#### Problem Description

Implement the `token_parallel` kernel to compute the diagonal Aqk and Akk blocks within each sub-chunk.

**Token-Parallel Strategy:** Each token gets its own program. The program iterates only over historical tokens within its own sub-chunk (j <= i), avoiding wasted computation.

**Why only diagonal blocks?** A 64×64 chunk matrix is partitioned into a 4×4 grid of 16×16 sub-chunk blocks. Off-diagonal blocks are handled by other kernels. This problem only computes the 4 diagonal blocks (D00, D11, D22, D33), each 16×16:

```
Chunk Matrix (BT × BT = 64 × 64):
        SC0     SC1     SC2     SC3
      +--------+--------+--------+--------+
SC0   |  D00   | (off)  | (off)  | (off)  |
      +--------+--------+--------+--------+
SC1   |  K10   |  D11   | (off)  | (off)  |    Dnn = diagonal blocks ← this problem
      +--------+--------+--------+--------+    Knm = off-diagonal blocks
SC2   |  K20   |  K21   |  D22   | (off)  |
      +--------+--------+--------+--------+
SC3   |  K30   |  K31   |  K32   |  D33   |
      +--------+--------+--------+--------+
```

**Mathematical Definition:**

For token i and historical token j within the same sub-chunk (j <= i):

```
gated_k = k[j] * exp2(g[i] - g[j])

Aqk[i, j] = scale * (q[i] · gated_k)       ← computed when j <= i, otherwise 0
Akk[i, j] = k[i] * beta[i] · gated_k       ← computed when j < i,  otherwise 0
```

Here `exp2(g[i] - g[j])` is the gated decay factor from token j to i. Since g is in log2 space, `exp2` directly recovers the factor. Because g is a cumulative sum of negative values (monotonically decreasing), `g[i] >= g[j]`, so `g[i] - g[j] <= 0` and `exp2(diff) ∈ (0, 1]`.

**Difference between Aqk and Akk:**
- **Aqk:** Dot product of query with gated key, scaled by `scale` to serve as attention weights. At the diagonal (j == i), `exp2(g[i]-g[i]) = 1`, reducing to `q[i] · k[i] * scale`.
- **Akk:** Dot product of key (times beta) with gated key, used later to construct the block-tridiagonal matrix and decouple the linear system. At the diagonal (j == i), the value must be 0 (strictly upper-triangular).

#### Input/Output Specification

| Parameter | Shape | Dtype | Description |
|-----------|-------|-------|-------------|
| `q` | `[B, T, H, K]` | bf16 | Query tensor |
| `k` | `[B, T, H, K]` | bf16 | Key tensor |
| `g` | `[B, T, H, K]` | fp32 | g_cumsum from Problem 1 (log2 space) |
| `beta` | `[B, T, H]` | bf16 | Per-token per-head weight coefficient |
| `scale` | scalar | fp32 | Attention scale factor = `1/sqrt(K)` |
| `Aqk` (output) | `[B, T, H, BT]` | bf16 | Diagonal Aqk blocks, each row filled at sub-chunk columns |
| `Akk` (output) | `[B, T, H, BC]` | fp32 | Diagonal Akk blocks (fp32 for downstream inversion precision) |

Compile-time constants: `BT=64` (chunk size), `BC=16` (sub-chunk size), `BK=next_power_of_2(K)`.

#### Index Calculation

```
Given global token index i_tg:
  i_b = i_tg // T              ← batch index
  i_t = i_tg %  T              ← local token index within batch

  i_c  = i_t // BT             ← chunk index
  i_s  = (i_t % BT) // BC      ← sub-chunk index (0..3)
  i_ts = i_c * BT + i_s * BC   ← sub-chunk start token index

  j iteration range: i_ts .. min(i_t, i_ts + BC - 1)
```

#### Computation Diagram

Using Chunk 0, Sub-chunk 1 (tokens 16..31) as an example:

```
Inside D11 block (16×16):
     j=16 j=17 j=18 ... j=31
i=16 [ C0    X    X  ...  X  ]  ← program_16: j=16  (1 pair)
i=17 [ C1   C2    X  ...  X  ]  ← program_17: j=16,17 (2 pairs)
i=18 [ C3   C4   C5  ...  X  ]  ← program_18: j=16,17,18 (3 pairs)
 ...                            ...
i=31 [ ...  ...  ... ...  Cn ]  ← program_31: j=16..31 (16 pairs)

Each program's workload grows linearly with the token's position in the sub-chunk (1..BC pairs).
```

**Storage Layout:**

```
Aqk [B, T, H, BT]:  write column = j % BT    (absolute column position within the chunk)
Akk [B, T, H, BC]:  write column = j - i_ts  (relative offset within sub-chunk, 0..BC-1)
```

#### Reference Implementation (PyTorch, for verification)

```python
def reference_token_parallel(q, k, g, beta, scale, BT=64, BC=16):
    """
    q:    [B, T, H, K]  bf16
    k:    [B, T, H, K]  bf16
    g:    [B, T, H, K]  fp32  (output of Problem 1)
    beta: [B, T, H]     bf16
    scale: scalar
    Returns: Aqk [B, T, H, BT], Akk [B, T, H, BC]
    """
    B, T, H, K = q.shape
    q = q.float()
    k = k.float()
    beta = beta.float()

    Aqk = torch.zeros(B, T, H, BT, dtype=torch.float32)
    Akk = torch.zeros(B, T, H, BC, dtype=torch.float32)

    for b in range(B):
        for h in range(H):
            for i in range(T):
                i_c = i // BT
                i_s = (i % BT) // BC
                i_ts = i_c * BT + i_s * BC

                for j in range(i_ts, min(i + 1, min(T, i_ts + BC))):
                    gated_k = k[b, j, h] * torch.exp2(g[b, i, h] - g[b, j, h])
                    aqk = scale * torch.dot(q[b, i, h], gated_k)
                    Aqk[b, i, h, j % BT] = aqk

                    if j < i:
                        akk = torch.dot(k[b, i, h] * beta[b, i, h], gated_k)
                        Akk[b, i, h, j - i_ts] = akk

    return Aqk, Akk
```

#### Completion Criteria

The Triton kernel output must satisfy the following error bounds against the reference:
- Aqk: **RMSE < 1e-3**
- Akk: **RMSE < 1e-3**

And satisfy:
- Aqk is valid at diagonal positions (j == i); Akk must be exactly 0 at diagonal positions (j == i)
- K-dimension dot product is accumulated via a tiled loop (cannot load the entire K at once)
- q[i] and k[i] are loaded once and reused across the entire inner loop

---

### Problem 3 (L3): GLA Output — Dual-Path Final Output Fusion

#### Position in the Pipeline

This problem corresponds to **Step 6: GLA Output**, the final step of the KDA pipeline. The preceding steps have produced:
- Step 1: `g_cumsum` (gated cumulative values)
- Steps 2-3: `Aqk` (intra-chunk attention weights) and `Akk_inv` (decoupling matrix)
- Steps 4-5: `h` (cross-chunk state matrix) and `v_new` (Delta-Rule-corrected values)

This problem fuses all intermediate results to produce the final attention output `o`.

#### Problem Description

Implement the `gla_output` kernel, which fuses two computation paths — cross-chunk and intra-chunk — to produce the final output.

**Why dual-path?** Chunkwise Linear Attention separates information into two sources:
- **Cross-chunk information:** The compressed representation of all tokens before chunk c, stored in state matrix h. Reading from h takes only one matrix multiplication O(K×V), without revisiting each historical token.
- **Intra-chunk information:** Exact (uncompressed) interactions within the current chunk, using the Aqk attention matrix and v_new.

The two paths are complementary — the cross-chunk path provides global context (coarse but broad), while the intra-chunk path provides precise local interactions (fine but narrow).

**Mathematical Definition:**

```
For token t in chunk c:
  o[t] = o_cross[t] + o_intra[t]

  Cross-chunk path: o_cross[t] = (q[t] * exp2(g[t])) @ h[c] * scale
                     ↑ q[t] multiplied by the gating factor exp2(g[t]),
                     ↑ amplifying the effective query of the current token
                     ↑ then matrix-multiplied with state matrix h[c] (shape [V, K])
                     ↑ actual computation: [K] @ [V, K]^T = [K] @ [K, V] = [V]
                     ↑ i.e., read historical information from cross-chunk memory
                     ↑ using the gated query

  Intra-chunk path: o_intra[t] = Σ_{j=start..t} Aqk[t, j] * v_new[j]
                     ↑ within the current chunk, weighted sum of v_new
                     ↑ using causal attention weights
                     ↑ Aqk[t, j] is nonzero only for j <= t (naturally causal)
                     ↑ equivalent to: Aqk_chunk @ v_new_chunk, with causal mask
```

#### Input/Output Specification

| Parameter | Shape | Dtype | Description |
|-----------|-------|-------|-------------|
| `q` | `[B, T, H, K]` | bf16 | Query tensor |
| `v_new` | `[B, T, H, V]` | bf16 | Delta-Rule-corrected values |
| `g` | `[B, T, H, K]` | fp32 | g_cumsum from Problem 1 (log2 space) |
| `Aqk` | `[B, T, H, BT]` | bf16 | Aqk attention matrix from Problem 2 |
| `h` | `[B, NT, H, V, K]` | bf16 | State matrix snapshot per chunk (V rows, K cols) |
| `scale` | scalar | fp32 | Attention scale factor = `1/sqrt(K)` |
| `o` (output) | `[B, T, H, V]` | bf16 | Final attention output |

Compile-time constants: `BT=64`, `BK`, `BV` (K and V dimension tile sizes, typically 32).

#### Computation Diagram

```
                  ┌─────────────────────┐
                  │   Input: Chunk c     │
                  │   BT=64, BK=BV=32   │
                  └─────────┬───────────┘
                            │
            ┌───────────────┴───────────────┐
            │                               │
            ▼                               ▼
   ┌──────────────────┐          ┌──────────────────┐
   │ Cross-path       │          │ Intra-path       │
   │ (K-dim loop)     │          │                  │
   │                  │          │  b_A = load      │
   │  for each BK:   │          │    Aqk[BT, BT]   │
   │    b_qg = load   │          │  m_s = causal    │
   │    q[BT, BK] *   │          │    mask (lower   │
   │    exp2(g[BT,BK])│          │    triangular)    │
   │    * scale       │          │  b_A *= m_s      │
   │                  │          │                  │
   │    b_h = load    │          │  b_v = load      │
   │    h[V, K] →     │          │    v_new[BT, BV] │
   │    transpose     │          │                  │
   │    → [BK, BV]    │          │  b_o2 = b_A @    │
   │                  │          │    b_v           │
   │    b_o1 +=       │          │  → [BT, BV]      │
   │    b_qg @ b_h^T  │          └────────┬─────────┘
   │  → [BT, BV]      │                   │
   └────────┬─────────┘                   │
            │                             │
            └──────────┬──────────────────┘
                       │
                       ▼
               b_o = b_o1 + b_o2
                   [BT, BV]
                       │
                       ▼
             store o[chunk_start:chunk_end, V_tile]
```

**Cross-Path Details:**

The cross-path is characterized by a K-dimension loop — h has shape `[V, K]`; the V direction can be loaded in tiles of BV=32, but the K direction requires tiled iteration. Each iteration:
1. Load `q[BT, BK]` and `g[BT, BK]` (64 tokens × 32 channels for the current K-tile)
2. Compute gated query: `b_qg = q * exp2(g) * scale`
3. Load a tile of `h[BV, BK]` and transpose to `[BK, BV]`
4. Matrix multiply and accumulate: `b_o1 += b_qg @ b_h^T` (`[BT, BK] @ [BK, BV] = [BT, BV]`)

**Intra-Path Details:**

1. Load `Aqk[BT, BT]` (attention weights for the current chunk; may be a partial chunk requiring boundary check)
2. Apply causal mask (lower triangular) to zero out the upper triangle
3. Load `v_new[BT, BV]` (corrected values for the current chunk)
4. Matrix multiply: `b_o2 = b_A @ b_v` (`[BT, BT] @ [BT, BV] = [BT, BV]`)

#### Reference Implementation (PyTorch, for verification)

```python
def reference_gla_output(q, v_new, g, Aqk, h, scale, BT=64):
    """
    q:     [B, T, H, K]  bf16
    v_new: [B, T, H, V]  bf16
    g:     [B, T, H, K]  fp32
    Aqk:   [B, T, H, BT] bf16
    h:     [B, NT, H, V, K] bf16  (V rows, K cols)
    scale: scalar
    Returns: o [B, T, H, V]
    """
    B, T, H, K = q.shape
    V = v_new.shape[-1]
    NT = h.shape[1]
    q = q.float()
    v_new = v_new.float()
    Aqk = Aqk.float()
    h = h.float()

    o = torch.zeros(B, T, H, V, dtype=torch.float32)
    causal_mask = torch.tril(torch.ones(BT, BT))

    for b in range(B):
        for head in range(H):
            for c in range(NT):
                start = c * BT
                end = min(start + BT, T)
                n_tokens = end - start

                # Cross-chunk path
                h_c = h[b, c, head]  # [V, K]
                for t in range(n_tokens):
                    idx = start + t
                    q_gated = q[b, idx, head] * torch.exp2(g[b, idx, head]) * scale
                    o[b, idx, head] += q_gated @ h_c.T  # [K] @ [K, V] = [V]

                # Intra-chunk path
                A_chunk = Aqk[b, start:end, head, :n_tokens]  # [n, BT] -> [n, n]
                v_chunk = v_new[b, start:end, head]           # [n, V]
                mask = causal_mask[:n_tokens, :n_tokens]
                o[b, start:end, head] += (A_chunk * mask) @ v_chunk

    return o
```

#### Completion Criteria

The Triton kernel output must achieve **RMSE < 1e-3** against the reference implementation, and satisfy:
- Cross-chunk path correctly implements the K-dimension loop (cannot assume K fits in one load)
- Intra-chunk path correctly applies causal masking (lower triangle valid, upper triangle zero)
- The two paths compute independently and accumulate into the same output (not overwriting)
- Matrix multiplications use `tl.dot`

---

## 3. Appendix

### A. Test Data Generation Script

The following script generates a complete test dataset covering all three problems:

```python
import torch
import math

def generate_test_data(T=128, H=2, K=64, V=64, seed=42):
    """Generate a complete test dataset for all three problems."""
    torch.manual_seed(seed)
    B = 1
    BT = 64

    # Model parameters
    A_log = torch.randn(H) * 0.5
    dt_bias = torch.randn(H, K) * 0.1

    # Problem 1 input
    raw_gate = torch.randn(B, T, H, K)

    # Problem 2 input
    q = torch.randn(B, T, H, K) * 0.1
    k = torch.randn(B, T, H, K) * 0.1
    beta = torch.sigmoid(torch.randn(B, T, H))

    # Problem 1 output = Problem 2's g input
    RCP_LN2 = 1.4426950216293335
    gate = -torch.exp(A_log)[None, None, :, None] * torch.nn.functional.softplus(
        raw_gate + dt_bias[None, None, :, :]
    )
    g_cumsum = torch.zeros_like(gate)
    NT = (T + BT - 1) // BT
    for c in range(NT):
        start = c * BT
        end = min(start + BT, T)
        g_cumsum[:, start:end] = torch.cumsum(gate[:, start:end], dim=1)
    g_cumsum *= RCP_LN2

    # Problem 3 input — v_new and h simulated with random data
    v_new = torch.randn(B, T, H, V) * 0.1
    h = torch.randn(B, NT, H, V, K) * 0.01

    scale = K ** -0.5

    return {
        'raw_gate': raw_gate,
        'A_log': A_log,
        'dt_bias': dt_bias,
        'q': q,
        'k': k,
        'beta': beta,
        'g_cumsum': g_cumsum,  # Problem 1 output / Problem 2 input
        'v_new': v_new,
        'h': h,
        'scale': scale,
        'BT': BT,
    }

# Usage example
data = generate_test_data()

# Problem 1
from reference import reference_gate_chunk_cumsum
g_out_ref = reference_gate_chunk_cumsum(data['raw_gate'], data['A_log'], data['dt_bias'])
g_out_triton = gate_chunk_cumsum(data['raw_gate'], data['A_log'], data['dt_bias'])
assert torch.allclose(g_out_ref, g_out_triton, rtol=1e-5), "Problem 1 failed"

# Problem 2
Aqk_ref, Akk_ref = reference_token_parallel(data['q'], data['k'], data['g_cumsum'],
                                              data['beta'], data['scale'])
Aqk_tri, Akk_tri = token_parallel(data['q'], data['k'], data['g_cumsum'],
                                    data['beta'], data['scale'])
assert torch.allclose(Aqk_ref, Aqk_tri.float(), atol=1e-3), "Problem 2 Aqk failed"
assert torch.allclose(Akk_ref, Akk_tri.float(), atol=1e-3), "Problem 2 Akk failed"

# Problem 3
o_ref = reference_gla_output(data['q'], data['v_new'], data['g_cumsum'],
                               Aqk_tri, data['h'], data['scale'])
o_tri = gla_output(data['q'], data['v_new'], data['g_cumsum'],
                    Aqk_tri, data['h'], data['scale'])
assert torch.allclose(o_ref, o_tri.float(), atol=1e-3), "Problem 3 failed"

print("All tests passed!")
```

### B. Triton API Quick Reference (with Ascend Notes)

```python
# Program indexing
pid = tl.program_id(axis)         # 0/1/2
# Ascend: prefer 1D grid; compute 2D/3D indices manually:
#   pid_m = pid // num_pid_n
#   pid_n = pid % num_pid_n

# Block pointer
p = tl.make_block_ptr(base, shape, strides, offsets, block_shape, order)
data = tl.load(p, boundary_check=(axis0, axis1), padding_option="zero")
tl.store(p, data, boundary_check=(axis0, axis1))
# Ascend: prefer tl.make_block_ptr over raw pointer arithmetic in offsets.
#   The compiler may compute offset before evaluating mask for raw pointers,
#   causing DDR out-of-bounds errors.

# Math operations
tl.exp(x), tl.exp2(x)             # exponential / base-2 exponential
tl.cumsum(x, axis=0)              # prefix sum
tl.dot(a, b)                      # matrix multiply [M,K] @ [K,N] -> [M,N]
tl.trans(x)                       # transpose
tl.sum(x, axis=1)                 # sum along axis
tl.where(cond, a, b)              # conditional selection

# Array construction
tl.arange(0, N)                   # [0, 1, ..., N-1] -- WARNING: returns int64
# Ascend: int64/int32 CMP -> scalar fallback. Always cast for mask generation:
#   idx = tl.arange(0, N).to(tl.float32)
#   mask = idx < limit             # now uses Vector CMP unit (fp32)

tl.zeros([M, N], dtype=tl.float32)
tl.full([M, N], value, dtype=tl.int32)

# Constant types
tl.constexpr                      # compile-time constant marker
```

