# Problem 1 — Gate Cumsum Optimization

## 1. Baseline

The initial baseline runtime was:

**1.999 ms/call**

The first thing I noticed in the implementation was the launch configuration. The baseline used a 3D grid:

```python
grid = (_cdiv(K, BS), num_chunks, B * H)

i_s = tl.program_id(0)   # tile index along S
i_t = tl.program_id(1)   # chunk index
i_bh = tl.program_id(2)  # combined (batch, head) index
```

This produces more blocks than the number of physical vector cores. Since the computation is executed on the vector side and the target device has 48 vector cores, I tested whether explicitly matching the launch grid to the physical core count would reduce scheduling overhead and give more predictable work distribution.

I flattened the original 3D work space into a single linear work index and launched exactly 48 programs:

```python
grid = (48,)

pid = tl.program_id(0)
num_programs = tl.num_programs(0)

total_work: tl.constexpr = B * H * NT * NS

for i in range(pid, total_work, num_programs):
    i_bh = i // (NT * NS)
    i_st = i - i_bh * NT * NS
    i_t = i_st // NS
    i_s = i_st - i_t * NS
```

Each program therefore processes approximately:

```text
ceil(total_work / 48)
```

tiles.

---

## 2. Fixed 48-core Grid and Multibuffering Issue

The first version failed during compilation with:

```text
error: ub overflow, requires 1581312 bits while 1572864 bits available!
(possible reason: tiling basic block is too large or block number is more than
what user expect due to multi-buffer feature is enabled and some ops need extra
local buffer.)

error: Failed to run BiShengHIR pipeline
```

This was unexpected because the tile size itself had not increased relative to the baseline.

My hypothesis was that Triton Ascend was applying multibuffering to the loop

```python
for i in range(pid, total_work, num_programs):
```

and therefore keeping buffers for multiple loop iterations alive simultaneously, which increased UB usage.

To verify this, I explicitly disabled multibuffering:

```python
multibuffer=False
```

The kernel then compiled successfully and passed the correctness test.

### Result

```text
Baseline:       1.999 ms/call
48-core grid:   1.622 ms/call
```

This corresponds to:

- **1.23× speedup**
- **~23% faster than the baseline**

---

## 3. Removing Explicit Mask Computation

The baseline generated masks such as:

```python
masks = (tile_t[:, None] < T) & (tile_s[None, :] < K)
```

These require integer comparisons. On this backend, some integer operations can fall back to scalar execution, so I wanted to remove this work entirely instead of paying for explicit mask generation.

I replaced masked loads/stores with block pointers using boundary checks and zero padding.

For the provided test case, `T % BT == 0`, so the time-dimension boundary check is not strictly required. However, I kept boundary checking in the generalized implementation so that the kernel also works when the dimensions are not exact multiples of the tile sizes.

The baseline also contained:

```python
b_gate = tl.where(masks, b_gate, 0.0)
```

This is unnecessary when the block-pointer load already uses zero padding for out-of-bounds elements and the store uses boundary checks. Invalid output elements are never written, while invalid input elements are loaded as zero.

### Result

```text
Before block pointers:   1.622 ms/call
After block pointers:    1.579 ms/call
```

Relative to the original baseline:

- **1.27× speedup**
- **~27% faster than the baseline**

At this point, the arithmetic itself was already difficult to reduce further, so I looked at memory-access behavior.

---

## 4. Consecutive Work Assignment

The flattened implementation initially distributed work in a strided pattern:

```python
for i in range(pid, total_work, num_programs):
    ...
```

This means neighboring cores operate on interleaved work items.

I changed the assignment so that every core processes one consecutive range:

```python
per_core: tl.constexpr = (
    total_work + num_programs - 1
) // num_programs

for i in range(
    pid * per_core,
    min(total_work, (pid + 1) * per_core),
):
    ...
```

The motivation was to improve locality and make accesses from each core more sequential instead of interleaving work across all 48 cores.

### Result

```text
Strided assignment:      1.579 ms/call
Consecutive assignment:  1.459 ms/call
```

Relative to the original baseline:

- **1.37× speedup**
- **~37% faster than the baseline**

This became the best vector-only implementation.

---

# 5. Alternative Approach: Computing Cumsum on the Cube Unit

I also investigated whether the prefix sum could be moved from the scalar/vector path to the Cube unit.

## 5.1 Behavior of `tl.cumsum`

The gate kernel performs cumsum along the time dimension, with:

```text
BT = 64
```

To understand how `tl.cumsum` is lowered on Ascend, I profiled the following minimal kernel:

```python
@triton.jit
def cumsum(
    x,
    N: tl.constexpr,
):
    ind = tl.arange(0, N)
    v = tl.load(x + ind)
    v = tl.cumsum(v, axis=0)
    tl.store(x + ind, v)
```

For:

```text
N = 10000
```

the `msprof` instruction-level simulator showed a sequential scalar loop with 9,999 iterations.

The dominant instructions were:

```text
Instruction       Pipe      Calls    Cycles      Time
---------------------------------------------------------
ST_XD_XN_IMM      SCALAR    9,999    294,958    119.43 us
LD_XD_XN_IMM      SCALAR    9,999    119,988     66.66 us
ADD (FP32)        SCALAR    9,999     49,995     27.78 us
ADD               SCALAR    9,999      9,999      5.56 us
ADD               SCALAR    9,999      9,999      5.56 us
ADD_IMM           SCALAR    9,999      9,999      5.56 us
```

This confirms that, for this case, `tl.cumsum` is lowered as a sequential dependency chain on the scalar execution pipeline rather than as a dedicated parallel scan instruction.

---

## 5.2 Matrix-Multiplication Formulation

A prefix sum can also be expressed as multiplication by a lower-triangular matrix:

```python
rows = tl.arange(0, BT).to(tl.float32)

mat = (
    rows[:, None] >= rows[None, :]
).to(tl.float32)

b_o = tl.dot(mat, b_gate)
```

This transforms the cumsum into a matrix multiplication that can execute on the Cube unit.

However, the measured kernel runtime was:

**4.603 ms/call**

which is significantly slower than the vector implementation.

One major reason is the data movement required when switching execution domains. In this implementation, the intermediate data effectively follows a path similar to:

```text
GM -> Vector -> GM -> Cube -> GM
```

so the benefit of using the Cube unit is outweighed by additional transfers and synchronization/materialization overhead.

---

## 5.3 Cube Compute Time vs. Existing Cumsum

The profiler showed approximately:

```text
309.6 us
```

of Cube execution per Cube core for the matrix-multiplication cumsum.

For comparison, I estimated the cost of cumsum in the optimized vector kernel by comparing the kernel runtime with and without the cumsum operation:

```text
Vector cumsum contribution ≈ 125 us
```

This is only a rough isolation of the cumsum cost, but it is useful for comparing the two approaches.

The device has:

```text
48 vector cores
24 cube cores
```

so each Cube core must process approximately twice as many chunks as each vector core.

Even if the vector cumsum cost is scaled by 2 to compensate for this difference:

```text
125 us × 2 ≈ 250 us
```

it is still below the measured Cube time:

```text
250 us < 309.6 us
```

Therefore, even before accounting for the additional Vector↔GM↔Cube data movement, the Cube implementation did not provide a compute-side advantage.

For this reason, I abandoned the Cube-based cumsum approach.

---

## 5.4 Vectorized Cumsum Experiment

I also investigated whether the `tl.cumsum` operation itself could be optimized.

Instruction-level profiling of a standalone `tl.cumsum` over 64 FP32 elements showed that it was lowered to a sequential scalar prefix-sum loop. For a 64-element input, the generated implementation executed 63 scalar loads, 63 scalar FP32 additions, and 63 scalar stores.

Since the scan length in the target kernel is `BT = 64`, I experimented with replacing `tl.cumsum` with a parallel Hillis-Steele-style prefix scan. A length-64 prefix sum requires only `log2(64) = 6` dependent stages, with offsets `1, 2, 4, 8, 16, 32`.

The following Triton helper was tested:

```python
@triton.jit
def vector_cumsum(x, BT: tl.constexpr, BS: tl.constexpr):
    t = tl.arange(0, BT)[:, None]

    # Broadcast indices to [BT, BS].
    t = t + tl.zeros((1, BS), tl.int32)

    idx = tl.maximum(t - 1, 0)
    shifted = tl.gather(x, idx, axis=0)
    x = x + tl.where(t >= 1, shifted, 0.0)

    idx = tl.maximum(t - 2, 0)
    shifted = tl.gather(x, idx, axis=0)
    x = x + tl.where(t >= 2, shifted, 0.0)

    idx = tl.maximum(t - 4, 0)
    shifted = tl.gather(x, idx, axis=0)
    x = x + tl.where(t >= 4, shifted, 0.0)

    idx = tl.maximum(t - 8, 0)
    shifted = tl.gather(x, idx, axis=0)
    x = x + tl.where(t >= 8, shifted, 0.0)

    idx = tl.maximum(t - 16, 0)
    shifted = tl.gather(x, idx, axis=0)
    x = x + tl.where(t >= 16, shifted, 0.0)

    idx = tl.maximum(t - 32, 0)
    shifted = tl.gather(x, idx, axis=0)
    x = x + tl.where(t >= 32, shifted, 0.0)

    return x
```

On a standalone 1D experiment, this approach successfully replaced the sequential scalar scan with six vector `VGATHER` operations and six vector FP32 additions. However, the generated code also introduced additional index-generation, masking, and synchronization instructions.

More importantly, applying the same approach to the actual `[BT, BS]` tensor substantially increased temporary UB usage. Each scan stage requires additional index and intermediate tensors over the 2D tile, and the resulting kernel exceeded the available UB capacity during compilation.

I therefore abandoned this optimization direction rather than reducing the tile size or restructuring the kernel around the cumsum.

This was also supported by an earlier experiment where the cumsum computation was removed entirely: the resulting latency improvement was relatively small compared with the total kernel latency. Therefore, despite the inefficient scalar lowering of `tl.cumsum`, the prefix sum is not the primary performance bottleneck.

The kernel is predominantly memory-bound, so increasing implementation complexity and UB pressure to optimize the cumsum is unlikely to provide a meaningful end-to-end performance improvement.

---

# 6. Final Result

The optimization progression was:

| Version | Runtime | Speedup vs. Baseline |
|---|---:|---:|
| Original baseline | 1.999 ms | 1.00× |
| Fixed 48-core grid, multibuffer disabled | 1.622 ms | 1.23× |
| Block pointers / removed explicit masks | 1.579 ms | 1.27× |
| Consecutive per-core work assignment | **1.459 ms** | **1.37×** |

Final result:

```text
1.999 ms/call -> 1.459 ms/call
```

or approximately:

**1.37× speedup (~37% faster than the baseline)**

The main improvements came from:

1. Matching the launch configuration to the 48 physical vector cores.
2. Disabling loop multibuffering after it caused UB overflow.
3. Replacing explicit mask computation with block-pointer boundary checks.
4. Assigning consecutive work ranges to each core to improve memory-access locality.
5. Evaluating and rejecting a Cube-based cumsum implementation after profiling showed that both its compute cost and its required data movement were worse than the optimized vector implementation.

# Problem 2 — Token Parallel Optimization

The baseline runtime for Problem 2 was:

```text
9.466 ms/call
```

For this kernel, I applied the same general optimization strategy described in detail for Problem 1.

The main changes were:

1. **Matching the launch configuration to the physical core count.**  
   The work space was flattened and distributed across the available physical cores rather than relying on a larger multidimensional grid.

2. **Distributing work evenly across cores.**  
   Each core was assigned a comparable amount of work to avoid unnecessary scheduling imbalance and reduce launch/scheduling overhead.

3. **Removing explicit mask computation where possible.**  
   Masked pointer arithmetic was replaced with block pointers using boundary checks and padding when applicable. This avoids generating additional integer comparison and address-calculation operations.

4. **Improving memory-access locality.**  
   Work assignment was arranged so that each program processes more spatially consecutive regions of memory where possible, rather than interleaving accesses across cores.

## Result

```text
Baseline:    9.466 ms/call
Optimized:   8.742 ms/call
```

This corresponds to approximately:

- **1.08× speedup**
- **7.6% latency reduction**

The improvement is smaller than for Problem 1 because this kernel is predominantly **memory-bound**.

Most of the arithmetic operations performed by the kernel are necessary for the algorithm itself, and there is no obvious alternative mathematical formulation that removes enough computation to materially change the end-to-end runtime.

Therefore, the remaining optimization space is primarily related to memory movement: reducing unnecessary transfers, improving locality, increasing effective cache reuse, and avoiding additional intermediate materialization.

---

# Problem 3 — Output Kernel Optimization

The baseline runtime for Problem 3 was:

```text
4.522 ms/call
```

The same optimization principles were applied here:

- flattening the launch grid and matching it more closely to the physical core count;
- distributing work evenly between cores;
- using block pointers and boundary checks instead of explicit mask/address-vector computation where possible;
- assigning work in a way that improves memory-access locality.

## Result

```text
Baseline:    4.522 ms/call
Optimized:   4.355 ms/call
```

This corresponds to approximately:

- **1.04× speedup**
- **3.7% latency reduction**

Similarly to Problem 2, this kernel is predominantly **memory-bound**.

The arithmetic performed by the kernel is directly required by the output computation, so there is relatively little useful computation that can simply be removed or replaced by a substantially cheaper equivalent.

As a result, further optimization mainly depends on improving memory behavior rather than reducing arithmetic instruction count.

---

# Further Experiments for Problems 2 and 3

After applying the general launch-configuration, work-distribution, masking, and block-pointer optimizations, I experimented with several additional approaches.

## Batched Chunk Processing

One direction I investigated was processing multiple consecutive chunks within the same program invocation.

Instead of:

```text
load one chunk
compute
store one chunk
```

the idea was closer to:

```text
load multiple consecutive chunks
compute them together
store multiple chunks
```

The motivation was to expose more locality between neighboring chunks and potentially reuse data while it was still resident in local memory or cache.

However, for the provided chunk and tile sizes, processing several chunks simultaneously significantly increases the amount of live tile data and temporary storage required by a program.

This quickly increases UB usage and makes the approach impractical with the existing tile configuration.

Reducing the tile dimensions can make the batched approach fit, but that introduces another tradeoff: smaller tiles require more loop iterations and more repeated addressing, load/store, and control overhead.

In the configurations I tested, this additional iteration overhead outweighed the potential benefit from processing several chunks together.

Therefore, batching multiple chunks did not improve the final performance.

---

# Final Results

The final measured runtimes for all three problems were:

| Problem | Baseline | Optimized | Speedup | Latency Reduction |
|---|---:|---:|---:|---:|
| Problem 1 — Gate Cumsum | 1.999 ms | **1.459 ms** | **1.37×** | **~27.0%** |
| Problem 2 — Token Parallel | 9.466 ms | **8.742 ms** | **1.08×** | **~7.6%** |
| Problem 3 — Output Kernel | 4.522 ms | **4.355 ms** | **1.04×** | **~3.7%** |

For Problems 2 and 3, the main useful improvements came from the same general techniques established during Problem 1:

1. Matching the launch configuration more closely to the available physical cores.
2. Flattening and evenly distributing work across cores.
3. Avoiding unnecessary explicit mask and address-vector computation.
4. Using block pointers with boundary checks where applicable.
5. Improving the locality of memory accesses through work assignment.

Both Problems 2 and 3 are largely memory-bound, which limits the benefit available from arithmetic-level optimization.

After the basic execution and addressing overheads were reduced, the remaining optimization opportunities were mostly related to memory movement and locality. I explored alternative work decompositions and multi-chunk processing, but the additional UB pressure or increased iteration count outweighed their potential advantages.

After iterative profiling and experimentation, the runtimes above were the best configurations I obtained for the provided test cases.
