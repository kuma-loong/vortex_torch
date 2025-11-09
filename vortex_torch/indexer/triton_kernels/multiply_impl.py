import torch
import triton
import triton.language as tl
from ..context import Context

@triton.jit
def multiply_bpr_kernel(
x,  # bf16, [B, x_D0, D1] ; D1 is the fastest-changing dim
y,  # bf16, [S, y_D0, D1] ; D1 is the fastest-changing dim
o,  # bf16, [*, o_D0, D1] ; flat buffer (each row writes a [o_D0, D1] tile)
indices,             # int32
winfo_x_indices,     # int32
winfo_y_offsets,     # int32
winfo_y_lens,        # int32
winfo_num_workloads, # int32*
max_chunk_size: tl.constexpr,
x_D0: tl.constexpr,
y_D0: tl.constexpr,
o_D0: tl.constexpr, # max(x_D0, y_D0), upstream will guarantee x_D0, y_D0 are broadcastable
D1: tl.constexpr
):  

    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)

    n_workloads = tl.load(winfo_num_workloads)

    # Static even partitioning of [0, n_workloads)
    per = n_workloads // num_progs
    r   = n_workloads %  num_progs
    start = pid * per + tl.minimum(pid, r)
    end   = start + per + (pid < r)

    # Index vectors
    d_ptr   = tl.arange(0, D1)
    x_d0_ptr   = tl.arange(0, x_D0)
    y_d0_ptr   = tl.arange(0, y_D0)
    o_d0_ptr   = tl.arange(0, o_D0)
    
    idx_ptr = tl.arange(0, max_chunk_size)

    # Stride across B for x
    x_stride = x_D0 * D1

    # Persistent cache: the current x[x_idx] as a whole [x_D0, D1] tile
    current_x_idx = tl.full((), -1, dtype=tl.int32)
    x_i = tl.zeros((x_D0, D1), dtype=tl.bfloat16)

    
    for i in range(start, end):
        # Select x for this workload
        x_idx_i32 = tl.load(winfo_x_indices + i).to(tl.int32)
        if x_idx_i32 != current_x_idx:
            x_base = (x_idx_i32 * x_stride).to(tl.int32)
            # Load x_i: [x_D0, D1] (bf16)
            x_offs = x_base + (x_d0_ptr[:, None] * D1 + d_ptr[None, :]).to(tl.int32)
            x_i = tl.load(x + x_offs)  # bf16
            current_x_idx = x_idx_i32

        # Range of y rows for this workload
        y_len = tl.load(winfo_y_lens + i)
        y_off = tl.load(winfo_y_offsets + i)
        valid = idx_ptr < y_len

        # Row indices for this chunk of y
        y_idx_i32 = tl.load(indices + y_off + idx_ptr, mask=valid, other=0).to(tl.int32)

        # Load y_tile: [rows, y_D0, D1] (bf16)
        # Linear offset: row*y_D0*D1 + y_d0*D1 + d1
        offs_y = (
            (y_idx_i32[:, None, None] * (y_D0 * D1)) +
            (y_d0_ptr[None, :, None]     * D1) +
            d_ptr[None, None, :]
        ).to(tl.int32)
        y_tile_bf16 = tl.load(y + offs_y, mask=valid[:, None, None], other=0.0)  # [rows, y_D0, D1], bf16

        
        # Elementwise multiply in bf16:
        # [rows, y_D0, D1] * [1, x_D0, D1] -> [rows, o_D0, D1] (bf16)
        o_i = y_tile_bf16 * x_i[None, :, :]   # bf16 mult
        

        # Linear output offset: row*o_D0*D1 + o_d0*D1 + d1, where row starts at y_off
        offs_o = (
            ((y_off + idx_ptr[:, None, None]) * (o_D0 * D1)) +
            (o_d0_ptr[None, :, None] * D1) +
            d_ptr[None, None, :]
        ).to(tl.int32)

        tl.store(o + offs_o, o_i, mask=valid[:, None, None])



def multiply_bpr(
x: torch.Tensor,
y: torch.Tensor,
o: torch.Tensor,
ctx: Context
):  
    
    multiply_bpr_kernel[(4 * ctx.num_sms,)](
        x, y, o, 
        ctx.dense_kv_indices,
        ctx.winfo_q_indices,
        ctx.winfo_kv_offsets,
        ctx.winfo_kv_lens,
        ctx.winfo_num_workloads,
        ctx.max_chunk_size,
        x.shape[-2], 
        y.shape[-2], 
        o.shape[-1],
        x.shape[-1], num_warps=8, num_stages=1
    )
