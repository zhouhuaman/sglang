# KDA Kernel Optimization · Interview Problem Set (English)

> This is an **English-language, 3-problem package** (K1 gate cumsum / K2 window scoring /
> K6 output fusion — the KDA 6-kernel split's Steps 1, 2 and 6, one problem per step). For each
> problem you are handed a **Triton kernel** (on Ascend 910B2); you reproduce the baseline,
> profile, and propose and land an optimization, or produce a credible "we are already at the
> current constraints' limit" argument. Each problem's full statement (operator overview / math /
> known bottleneck / official baseline / one msprof command / contract) lives in its directory's
> **`PROBLEM.md`**; `Triton_Kernel_Programming_Test_EN.md` is the conceptual background and problem
> brief; this file only covers the **environment** and the **deliverables**.

```
interview_en/
├── README.md                              ← this file (environment + deliverables)
├── Triton_Kernel_Programming_Test_EN.md   ← background & problem brief (Steps 1/2/6 of KDA)
├── env.sh                                 ← container environment setup (source once)
├── k1_gate_chunk_cumsum/                  ← Problem 1 · Step 1 Gate Cumsum (L1)
│   ├── PROBLEM.md                         ← problem statement (K1)
│   ├── gate_chunk_cumsum_kernel.py        ← the kernel you read/modify (the only change point)
│   └── test.py                            ← correctness bar + performance reproduction
├── k2_token_parallel/                     ← Problem 2 · Step 2 Token Parallel (L2)
│   ├── PROBLEM.md
│   ├── token_parallel_kernel.py
│   └── test.py
└── k6_gla_output/                         ← Problem 3 · Step 6 GLA Output (L3)
    ├── PROBLEM.md
    ├── gla_output_kernel.py
    └── test.py
```

The three problems form a data-dependency chain (Problem 1's `g_cumsum` → Problem 2's `Aqk` →
Problem 3's `o`), mirroring KDA's pipeline.

**Common target case** (shared by all three problems; this is the graded configuration):
`D_KV128_H96_T16384` → B=1, T=16384, H=96, K=V=128. All performance is measured on this case;
concrete baselines and measurement commands are in each `PROBLEM.md`.

---

## 1. Test environment

### 1.1 Connecting (VS Code Remote-SSH, passwordless)

| Item | Value |
|---|---|
| Server | `116.204.40.238` (port 22) |
| User | `root` |
| Key | `~/.ssh/devcontainer_root_key` (issued by the admin; `chmod 600`, do not share or commit) |
| Dev container | `triton-ascend-env-zhm` (full Triton/Ascend environment once you are inside) |

First time, append the following to your local `~/.ssh/config`:

```
Host triton-env
    HostName 116.204.40.238
    User root
    Port 22
    IdentityFile ~/.ssh/devcontainer_root_key
    ConnectTimeout 30
```

Install the **Remote - SSH** extension in VS Code → `F1` → `Remote-SSH: Connect to Host...` → pick
`triton-env` (first time: choose Linux and confirm the host key). Once the lower-left corner shows
`SSH: triton-env`, enter the dev container from the terminal:

```bash
docker exec -it triton-ascend-env-zhm bash
```

The container provides: Python 3.11.15, CANN 9.0.0, torch 2.7.1 + torch_npu + **triton 3.2.1
(Ascend)**, 8× Ascend 910B2. `/docker`, `/data` are shared between host and container — edit code in
VS Code (open `/docker/zhm/0505_skill_test/sonnet/sglang/kda_test/interview_en`), run commands in
the container terminal.

### 1.2 NPU etiquette

Inside the container, check for an idle card first; **do not share a card with others** (concurrent
use raises `ERR00100`/`Resource_Busy`):

```bash
npu-smi info                     # inspect each card's AICore utilization; pick an idle one
```

### 1.3 Preparing the environment (once per new terminal)

```bash
cd /docker/zhm/0505_skill_test/sonnet/sglang/kda_test/interview_en
source env.sh                    # or `source env.sh 4` — pin to card 4 only
```

`env.sh` does three things: sources CANN's `set_env.sh`, adds the torch/torch_npu dynamic
libraries to `LD_LIBRARY_PATH`, and sets `TORCH_DEVICE_BACKEND_AUTOLOAD=0` (missing any of these,
torch_npu fails to load). It prints the environment and card status — seeing those is enough. Sanity
check:

```bash
python3 -c "import torch, torch_npu, triton; print(torch.__version__, triton.__version__)"
```

> Iteration tip: each problem only requires editing `<problem-dir>/<op>_kernel.py` (the one change
> point; the filename matches the second half of the directory name: `gate_chunk_cumsum_kernel.py` /
> `token_parallel_kernel.py` / `gla_output_kernel.py`); do not touch `test.py`/`env.sh`. After
> editing, re-run the problem's commands.

---

## 2. Deliverables (markdown)

1. **Reproduced baseline**: the msprof per-call mean (ms) for the target case, together with the
   exact msprof command you ran and the `python3 test.py --report <dir>` output (the number must
   fall within the official range stated in the problem).
2. **Bottleneck claim**: what evidence (msprof op_summary / metrics / ablation experiments) backs
   your judgment.
3. **Changes**: where you changed the kernel and why (one hypothesis per sentence).
4. **Gains**: before vs after ms/call + accuracy numbers under the same protocol (same card, same
   repeats/warmup, same session). **No gain, or you believe it has converged**: provide the evidence
   chain supporting "already at the current constraints' limit" — an honest negative result + a
   convergence argument is not inferior to a fabricated small gain.
5. **Directions abandoned**: optimizations you tried and dropped, with a one-line reason (shows
   breadth of thinking).

**Measurement discipline** (what makes comparisons trustworthy):
- Performance is always the msprof `Task Duration(us)` **per-call mean**; `test.py --perf`'s own
  wall-clock is a snapshot only and is not graded.
- Your gains are measured against the baseline **you reproduced yourself**; do not force-compare
  across machines or time windows; before every run confirm an idle card with `npu-smi info`.
- Correctness follows each problem's `PROBLEM.md` PASS bar (default: `python3 test.py` prints
  PASS).

**Non-negotiable contracts** (detailed per problem in `PROBLEM.md`; violating them = invalid
solution): the public default `chunk_size=64` (K2 additionally `sub_chunk_size=16`) may not change —
these operators are upstream/downstream of one another in the KDA chain and share the same chunk
boundaries of 64; fp32 is the primary contract; accuracy stays at the <1e-2 level.
