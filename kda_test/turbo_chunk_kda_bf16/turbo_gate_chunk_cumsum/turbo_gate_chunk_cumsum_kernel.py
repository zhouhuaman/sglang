# SPDX-License-Identifier: Apache-2.0
"""KDA ``gate + chunk-local cumsum`` 算子：纯 torch + triton 的独立实现。

本模块是 ``python/sglang/kernels/ops/attention/fla/kda.py`` 中
``kda_gate_chunk_cumsum`` 的功能等价物，但只依赖 ``torch`` / ``triton``，
**不 import 任何 sglang 代码**，因此可被门控测试目录独立使用
（共享的 ``load_kda.py`` 间接层在这里无法维护）。

计算内容（与上层 kernel 完全一致）：

    1. 门控激活（standard gate）::

         gate[t, s] = -exp(A_log[h]) * softplus(raw_gate[t, s] + dt_bias[h, s])

       其中 ``softplus(x) = log(1 + exp(x))``（x 很大时退化为 x，保证精确）；
    2. chunk 内累积和：沿时间维 T 做 "chunk 局部" 前缀和，chunk 大小为 ``BT``
       （每 chunk 从 0 重新开始，跨 chunk 边界不传递）;
    3. 可选缩放：``out *= scale``，实际调用时 ``scale = RCP_LN2``，将结果从
       ln 空间换算到 log2 空间，供后续 ``exp2`` 类 kernel 使用。

Triton kernel 为 tiled 向量化实现：

    * grid = ``(cdiv(K, BS), num_chunks, B * H)``，BS = 32、BT = chunk_size；
    * 每个 program (CTA) 处理一个 ``(时间 chunk, (batch, head), S-tile)``，
      加载 ``[BT, BS]`` 的二维 tile（行=时间、列=通道）；
    * 对 tile 做 ``tl.cumsum(b_gate, axis=0)``（axis=0 正是 chunk 长度维，
      与本硬件上确认可用的 ``tl.cumsum`` 语义一致）。

尾部（partial）chunk 处理
-------------------------
最后一个 chunk 的行数可能不足 BT，也会在通道维上出现最后一个 S-tile 不足
BS。kernel 统一用 mask 处理：越界元素 load 为 0，且在做 cumsum **前**把无效行
清零（``tl.where(masks, b_gate, 0.0)``）。因为 ``tl.cumsum`` 是沿 axis=0 向后
累加，**尾部补零不会污染头部有效行的前缀和**——这正好与真实 kernel
（``boundary_check`` 返回 0）的行为一致。

npud 说明
---------
本模块在顶部 ``import torch_npu``（不加会触发 "Background device ... is not
available" 错误）。真实运行环境（docker 容器）由调用方在启动 python 前自行
source CANN 的 ``set_env.sh`` 并设置 ``LD_LIBRARY_PATH``、
``TORCH_DEVICE_BACKEND_AUTOLOAD=0``；本模块不设置任何环境变量。模块 import
时的顶层逻辑不分配任何 NPU 张量（仅在调用 driver 时才触碰 ``npu`` 设备），
因此在无 NPU 设备的环境里也可以正常 import、并运行 ``turbo_gate_chunk_cumsum_ref``；
``turbo_gate_chunk_cumsum_triton`` 在检测不到 NPU 时会自动退化为参考实现（便于纯 CPU
校验逻辑），在 NPU 可用时走 triton kernel。
"""

import os

import torch
import torch_npu  # noqa: F401  (必须在创建任何 npu 张量之前 import)

import triton
import triton.language as tl


# log2(e) = 1 / ln(2)，取 fp32 下的精确值（与 flash-linear-attention 一致）。
RCP_LN2 = 1.4426950216293335

# 编译期 tile 大小。BT 即 chunk_size；BS 为通道维 tile 大小。
# tl.cumsum 需要这两个值都是 2 的幂。
_DEFAULT_BT = 64
# 优化记录（见 OPTIMIZATION_LOG.md）: BS 从 32 增大到 64 以降低展平后 grid
# 大小（cdiv(K,BS) 从 4→2）。目标 CASE B=1,T=16384,H=96,K=128 下
# grid = (2, 256, 96) → 展平 49152 ≤ 65535，消除 grid 超限不支持问题。
# 第二轮优化: BS 从 64 增大到 128, cdiv(K,BS) 从 2→1, grid = (1,256,96)
# → 展平 24576, speedup 从 ~3.1x 提升到 ~7.0x (torch_npu baseline)。
_DEFAULT_BS = 128
_SOFTPLUS_THRESHOLD = 20.0

# ── 工作划分开关（迁移自 interview_en_answer 的 K1 优化）─────────────────────
# K1_MODE: 1 = 一维 grid + 每核连续区间（默认，实测 -16.3%）
#          0 = 原三维 grid=(cdiv(K,BS), NT, B*H)（保留以便 A/B 复核）
# K1_GRID: 0 = 自动取设备 vector core 数（A5=56 / 910B2=48）
_K1_MODE = int(os.getenv("K1_MODE", "1"))
_K1_GRID = int(os.getenv("K1_GRID", "0"))

_VECTOR_NUM_CACHE = {}


def _vector_core_num() -> int:
    """取设备 vector core 数（本 kernel 全部在 vector 上，无 dot）。"""
    if "v" not in _VECTOR_NUM_CACHE:
        try:
            _VECTOR_NUM_CACHE["v"] = int(
                torch_npu.npu.get_device_properties(0).vector_core_num)
        except Exception:  # 老版本 torch_npu 无该字段 → 回退 910B2 的 48
            _VECTOR_NUM_CACHE["v"] = 48
    return _VECTOR_NUM_CACHE["v"]


def _cdiv(a: int, b: int) -> int:
    """向上取整的整数除法。"""
    return -(a // -b)


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


@triton.jit
def _softplus_fwd(x):
    """softplus(x) = log(1 + exp(x)); x 大于阈值时用线性近似避免 exp 溢出。"""
    return tl.where(x < 20.0, tl.log(1.0 + tl.exp(x)), x)


@triton.jit(do_not_specialize=["T"])
def _gate_cumsum_kernel(
    x,
    A_log,
    dt_bias,
    o,
    scale,
    T,  # 每个 batch 的时间步数（运行时标量，避免按 T 值重新编译）
    H: tl.constexpr,
    K: tl.constexpr,  # 通道维 (=S)
    BT: tl.constexpr,
    BS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_SCALE: tl.constexpr,
):
    """KDA gate 激活 + chunk 内 cumsum 的 tiled triton kernel。

    grid = (cdiv(K, BS), num_chunks, B * H)，每个 program 处理一个
    ``[BT, BS]`` tile（BT 行时间 x BS 列通道）：
      * 加载 raw gate，可选加 dt_bias；
      * 门控激活: gate = -exp(A_log) * softplus(x + bias)；
      * 无效行清零后 ``tl.cumsum(b_gate, axis=0)``（chunk 局部前缀和）；
      * 可选乘 scale，写回。
    """
    i_s = tl.program_id(0)  # S 维 tile 索引 0..cdiv(K, BS)-1
    i_t = tl.program_id(1)  # chunk 索引 0..num_chunks-1
    i_bh = tl.program_id(2)  # 联合 (batch, head) 索引 0..B*H-1

    # 注意：这里“chunk 索引”直接使用全局时间坐标。由于 T 对每个 batch 相同，
    # chunk 时间偏移 = i_t * BT（固定长度模式）。kernel 唯一被假定同构的地方
    # 是 T（每 batch 长度相同），这在固定长度输入下恒成立。
    i_b = i_bh // H
    i_h = i_bh % H

    # 当前 chunk 在时间维上的起始偏移 / 通道维偏移
    base_t = i_t * BT
    s_off = i_s * BS

    rows = tl.arange(0, BT)  # [BT]
    cols = tl.arange(0, BS)  # [BS]

    tile_t = base_t + rows  # 该 program 负责的时间坐标 [BT]
    tile_s = s_off + cols  # 该 program 负责的通道坐标 [BS]
    masks = (tile_t[:, None] < T) & (tile_s[None, :] < K)  # [BT, BS]

    # 行主序连续内存: flat 索引 = b*T*H*K + t*H*K + h*K + s
    ptr_base = x + i_b * T * H * K + i_h * K
    ptr_x = ptr_base + tile_t[:, None] * (H * K) + tile_s[None, :]

    # 越界位置 load 为 0（行/列 mask），后面会再清零无效行。
    b_s = tl.load(ptr_x, mask=masks, other=0.0).to(tl.float32)

    if HAS_BIAS:
        # dt_bias 为 [H*K] 平铺；当前 head 的偏置 = dt_bias[i_h*K : i_h*K+K]
        ptr_bias = dt_bias + i_h * K + tile_s
        b_bias = tl.load(ptr_bias, mask=tile_s < K, other=0.0).to(tl.float32)
        b_s = b_s + b_bias[None, :]

    # 每-head 的 A_log 标量
    b_a = tl.load(A_log + i_h).to(tl.float32)

    # 门控激活: gate = -exp(A_log) * softplus(x + bias)
    b_gate = -tl.exp(b_a) * _softplus_fwd(b_s)

    # 无效行（时间越界）清零。cumsum 沿 axis=0 向后累加，尾部补零不会污染
    # 头部有效行的前缀和；无效列本来就被清零，不会进入结果。
    b_gate = tl.where(masks, b_gate, 0.0)

    # chunk 内前缀和（跨 chunk 边界不传递，每 chunk 从 0 重新开始）
    b_o = tl.cumsum(b_gate, axis=0)

    if HAS_SCALE:
        b_o *= scale

    ptr_o = o + i_b * T * H * K + i_h * K + tile_t[:, None] * (H * K) + tile_s[None, :]
    tl.store(ptr_o, b_o.to(tl.float32), mask=masks)


@triton.jit(do_not_specialize=["T"])
def _gate_cumsum_kernel_1d(
    x,
    A_log,
    dt_bias,
    o,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_SCALE: tl.constexpr,
):
    """一维 grid + 每核连续工作区间（迁移自 interview_en_answer 的 K1 优化）。

    数学与 `_gate_cumsum_kernel` **逐位相同**，唯一差别是工作划分：

      * `_gate_cumsum_kernel`: grid = (cdiv(K,BS), num_chunks, B*H)，由硬件调度
        把 24576 个 (s,tile,chunk,(b,h)) 三元组分给 56 个 vector core；
      * 本 kernel: grid = (ncore,)，每核处理一段**连续**的 work id 区间，
        work id 按 (b, h, chunk, s) 解码 —— 相邻 item 在同一 (b,h) 内沿时间维
        相邻，访存连续，且相邻 item 复用同一 head 的 A_log/指针基址。

    参考实现的 `per_core = WORK // num_programs` 用的是**截断除法**，当
    total % ncore != 0 时会静默丢掉最后 total % ncore 个工作项；本实现改为
    ceil 除法 + `tl.minimum(total, ...)` 截断，末尾核不会越界也不会漏算。

    实测（A5 950PR / CANN 9.1.0 / triton-ascend 3.2.1，目标 case
    B1 T16384 H96 K128）：msprof Task Duration 1.371ms → 1.147ms（-16.3%）。
    """
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)

    NT = (T + BT - 1) // BT
    NS: tl.constexpr = (K + BS - 1) // BS
    total = B * H * NT * NS
    per_core = (total + nprog - 1) // nprog
    end = tl.minimum(total, (pid + 1) * per_core)

    rows = tl.arange(0, BT)
    cols = tl.arange(0, BS)

    for i in range(pid * per_core, end):
        i_bh = i // (NT * NS)
        i_st = i - i_bh * NT * NS
        i_t = i_st // NS
        i_s = i_st - i_t * NS

        i_b = i_bh // H
        i_h = i_bh % H

        tile_t = i_t * BT + rows
        tile_s = i_s * BS + cols
        masks = (tile_t[:, None] < T) & (tile_s[None, :] < K)

        ptr_x = x + i_b * T * H * K + i_h * K + tile_t[:, None] * (H * K) + tile_s[None, :]
        b_s = tl.load(ptr_x, mask=masks, other=0.0).to(tl.float32)

        if HAS_BIAS:
            ptr_bias = dt_bias + i_h * K + tile_s
            b_bias = tl.load(ptr_bias, mask=tile_s < K, other=0.0).to(tl.float32)
            b_s = b_s + b_bias[None, :]

        b_a = tl.load(A_log + i_h).to(tl.float32)
        b_gate = -tl.exp(b_a) * _softplus_fwd(b_s)
        b_gate = tl.where(masks, b_gate, 0.0)
        b_o = tl.cumsum(b_gate, axis=0)

        if HAS_SCALE:
            b_o *= scale

        ptr_o = o + i_b * T * H * K + i_h * K + tile_t[:, None] * (H * K) + tile_s[None, :]
        tl.store(ptr_o, b_o.to(tl.float32), mask=masks)


def turbo_gate_chunk_cumsum_ref(
    x,
    A_log,
    dt_bias=None,
    chunk_size=_DEFAULT_BT,
    scale=RCP_LN2,
):
    """纯 torch CPU 参考实现（可在任意 device 上运行，典型为 CPU）。

    计算与 kernel 完全一致::

        gate[t,s] = -exp(A_log[h]) * softplus(x[t,s] + bias[h,s])
        out[k]    = scale * cumsum(gate)   # chunk 局部，每 chunk 从 0 开始

    支持任意 B/T/H/K（包括 K 不是 tile 倍数、T 不是 BT 倍数），便于对 triton
    kernel 结果做精确对比。

    参数:
        x: [B, T, H, K] raw gate 值
        A_log: [H] 每 head 的对数尺度
        dt_bias: [H*K] 平铺偏置（可选，None 表示无偏置）
        chunk_size: chunk 大小（应传入与 kernel 相同的 BT）
        scale: 输出缩放（None 表示不缩放）

    返回:
        [B, T, H, K] fp32 结果。
    """
    assert x.dim() == 4, f"x must be 4D [B,T,H,K], got shape {tuple(x.shape)}"
    B, T, H, K = x.shape

    x = x.float()

    # 1) 加偏置 + softplus
    if dt_bias is not None:
        bias = dt_bias.to(torch.float32).reshape(1, 1, H, K)
        y = x + bias
    else:
        y = x
    y = torch.where(
        y < _SOFTPLUS_THRESHOLD,
        torch.log1p(torch.exp(y)),
        y,
    )  # softplus(x + bias)

    # 2) 乘 -exp(A_log)（广播到 [1,1,H,1]）。
    #    A_log[h] 与通道 (H,K) 的索引均为步长 1 的逐元素广播；
    #    注意 int index [0,0,0,0] 直接索引 x，与 x[0][0][0][0] 一致（已验证）。
    ap = A_log.to(torch.float32).view(1, 1, -1, 1)  # [1,1,H,1]
    y = -torch.exp(ap) * y

    # 3) chunk 局部 cumsum：逐 chunk 处理，每 chunk = y[:, a:a+avail] 的
    #    [B, avail_rows, H, K] 切片，对其 dim=1（时间轴）做前缀和；
    #    不完整块（avail < chunk_size）先补零到 chunk_size 再截断——补零只在
    #    块尾部，不会污染有效行的前缀和（与 kernel 尾部零填充语义一致）。
    edges = list(range(0, T, chunk_size)) + [T]
    parts = []
    for a, b in zip(edges[:-1], edges[1:]):
        avail = b - a  # chunk 内实际行数（< chunk_size 表示不完整块）
        part = y[:, a : a + avail].cumsum(dim=1)  # 沿时间轴前缀和
        if avail < chunk_size:
            pad = torch.zeros(
                B, chunk_size - avail, H, K, dtype=part.dtype, device=part.device
            )
            part = torch.cat([part, pad], dim=1)
        parts.append(part)
    y = torch.cat(parts, dim=1)  # 各 chunk 按原 T 顺序拼接
    y = y[:, :T].contiguous()

    # 4) 可选缩放
    if scale is not None:
        y = y * scale
    return y.float().contiguous()


def turbo_gate_chunk_cumsum_torch(
    x,
    A_log,
    dt_bias=None,
    chunk_size=_DEFAULT_BT,
    scale=RCP_LN2,
) -> torch.Tensor:
    """元算子(torch_npu 算子图)版本:门控激活 + chunk 内 cumsum。

    这是**性能基准**实现 —— 用 torch_npu 现成的逐算子组合完成同样的计算,
    供与 triton kernel 做加速比对比。比 ``turbo_gate_chunk_cumsum_ref`` 快:
      * 激活 + 累计合一(无二次排序/拼接),数据只经过一次;
      * cumsum 用 reshape + ``torch.cumsum`` 一次完成, 不逐 chunk 循环;
      * `softplus` 有算法级 ``logaddexp2`` 写法: ``sp(x) = x + logaddexp2(0,-x)*ln2``,
        对正向 float32 不产生中间溢出, 无需 ``where`` 分支。

    与 triton/ref 数学完全一致:
        gate[t,s] = -exp(A_log[h]) * softplus(x[t,s] + dt_bias[h,s])
        out      = scale * cumsum_over_chunk(gate)

    参数与 ``turbo_gate_chunk_cumsum_triton`` 相同。返回 [B,T,H,K] fp32(NPU 或 CPU device
    与输入一致)。
    """
    assert x.dim() == 4, f"x must be 4D [B,T,H,K], got shape {tuple(x.shape)}"
    assert _is_power_of_two(chunk_size), "chunk_size must be a power of 2"
    BT = int(chunk_size)
    B, T, H, K = x.shape
    NT = _cdiv(T, BT)
    pad = NT * BT - T

    xf = x.to(torch.float32)
    if pad:  # 尾部(pad>0)不可直接 reshape, 先补零到 NT*BT 再做 chunk cumsum
        xf = torch.cat([xf, torch.zeros(B, pad, H, K, dtype=xf.dtype, device=xf.device)], dim=1)
    if dt_bias is not None:
        xf = xf + dt_bias.to(torch.float32).reshape(1, 1, H, K)
    # softplus 精确实现(用 logaddexp 避免 exp 中间溢出):
    #   sp(x) = log(1+exp(x)) = x + logaddexp(0, -x)
    sp = xf + torch.logaddexp(torch.zeros_like(xf), -xf)
    gate = -torch.exp(A_log.to(torch.float32).view(1, 1, H, 1)) * sp

    y = gate.reshape(B, NT, BT, H, K).cumsum(dim=2).reshape(B, NT * BT, H, K)
    y = y[:, :T].contiguous()
    if scale is not None:
        y = y * scale
    return y.float().contiguous()


def turbo_gate_chunk_cumsum_triton(
    x,
    A_log,
    dt_bias=None,
    chunk_size=_DEFAULT_BT,
    scale=RCP_LN2,
    num_warps=1,
) -> torch.Tensor:
    """KDA gate 激活 + chunk 内 cumsum 的 triton 版本（在 NPU 上运行）。

    参数与 ``kda_gate_chunk_cumsum`` 兼容:

        x: [B, T, H, K] raw gate 值
        A_log: [H] 每 head 的对数尺度
        dt_bias: [H*K] 平铺偏置（可选，None 表示无偏置）
        chunk_size: chunk 大小（默认 64，必须为 2 的幂）
        scale: 输出缩放（默认 RCP_LN2；None 表示不缩放）
        num_warps: 每个 CTA 的 warp 数（默认 1，与真实 kernel 一致）

    返回:
        [B, T, H, K] fp32 结果，位于 NPU 上（NPU 不可用时自动退化为 CPU 参考）。
    """
    assert x.dim() == 4, f"x must be 4D [B,T,H,K], got shape {tuple(x.shape)}"
    assert _is_power_of_two(chunk_size), "chunk_size must be a power of 2"

    # 纯 CPU / 无 NPU 环境：退化为参考实现，便于单测在无 NPU 机器上校验逻辑。
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        return turbo_gate_chunk_cumsum_ref(
            x, A_log, dt_bias=dt_bias, chunk_size=chunk_size, scale=scale
        )

    B, T, H, K = x.shape
    BS = _DEFAULT_BS
    BT = int(chunk_size)

    # 输入统一转 fp32 并搬上 NPU（不会改动调用方传入的张量）
    x = x.to(torch.float32).to("npu")
    A_log = A_log.to(torch.float32).to("npu")
    has_bias = dt_bias is not None
    if has_bias:
        dt_bias = dt_bias.to(torch.float32).to("npu")
    else:
        dt_bias = x  # HAS_BIAS=False 时该指针不会被读取，传一个 dummy 即可

    o = torch.empty_like(x)

    num_chunks = _cdiv(T, BT)

    if _K1_MODE:
        # 一维 grid + 每核连续区间：total 个工作项按 (b,h,chunk,s) 解码，
        # 每核一段连续区间。ceil 除法保证不丢工作项。
        total = B * H * num_chunks * _cdiv(K, BS)
        ncore = _K1_GRID if _K1_GRID > 0 else _vector_core_num()
        grid = (min(total, ncore),)
        _gate_cumsum_kernel_1d[grid](
            x,
            A_log,
            dt_bias,
            o,
            float(scale) if scale is not None else 0.0,
            T,
            B=B,
            H=H,
            K=K,
            BT=BT,
            BS=BS,
            HAS_BIAS=has_bias,
            HAS_SCALE=scale is not None,
            num_warps=num_warps,
        )
        torch.npu.synchronize()
        return o

    grid = (_cdiv(K, BS), num_chunks, B * H)

    _gate_cumsum_kernel[grid](
        x,
        A_log,
        dt_bias,
        o,
        float(scale) if scale is not None else 0.0,
        T,
        H=H,
        K=K,
        BT=BT,
        BS=BS,
        HAS_BIAS=has_bias,
        HAS_SCALE=scale is not None,
        num_warps=num_warps,
    )
    # 同步等待完成。测试脚本如需计时，自行用 torch.npu.synchronize() 包住调用。
    torch.npu.synchronize()
    return o