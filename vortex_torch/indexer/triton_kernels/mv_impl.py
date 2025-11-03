import torch
import triton
import triton.language as tl


@triton.jit
def persistent_mv(
x,
y,
o,
indices,
winfo_x_indices,
winfo_y_offsets,
winfo_y_lens,
winfo_num_workloads,
max_chunk_size: tl.constexpr,
D: tl.constexpr,
):
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)

    n_workloads = tl.load(winfo_num_workloads)

    per = n_workloads // num_progs
    r = n_workloads % num_progs
    start = pid * per + tl.minimum(pid, r)
    end = start + per + (pid < r)

    
    d_ptr = tl.arange(0, D)
    idx_ptr = tl.arange(0, max_chunk_size)

    x_stride = D

    current_x_idx = tl.full((), -1, dtype=tl.int32)
    x_i = tl.zeros((D,), dtype=tl.bfloat16)

    for i in range(start, end):
        
        x_idx_i32 = tl.load(winfo_x_indices + i).to(tl.int32)
        if x_idx_i32 != current_x_idx:
            x_base_offs = x_idx_i32 * x_stride
            x_i = tl.load(
                x + x_base_offs + d_ptr
            )
            current_x_idx = x_idx_i32

        y_len = tl.load(winfo_y_lens + i)
        y_off = tl.load(winfo_y_offsets + i)
        valid = idx_ptr < y_len

        
        y_idx_i32 = tl.load(indices + y_off + idx_ptr, mask=valid, other=0).to(tl.int32)

        row_offs = y_idx_i32 * D     # [max_chunk_size]
        offs = row_offs[:, None] + d_ptr[None, :]             

        y_tile = tl.load(y + offs, mask=valid[:, None], other=0.0)
        
        o_i = tl.sum((y_tile * x_i[None, :]).to(tl.float32), 1)
        tl.store(o + y_off + idx_ptr, o_i, mask=valid)


def triton_mv(
x: torch.Tensor,
y: torch.Tensor,
o: torch.Tensor,
indices: torch.Tensor,
winfo_x_indices: torch.Tensor,
winfo_y_offsets: torch.Tensor,
winfo_y_lens: torch.Tensor,
winfo_num_workloads: torch.Tensor,
max_chunk_size: int,
num_sms: int
):  
    
    persistent_mv[(4 * num_sms,)](
        x, y, o, indices, winfo_x_indices, winfo_y_offsets,
        winfo_y_lens, winfo_num_workloads, max_chunk_size,
        x.shape[-1], num_warps=8, num_stages=2
    )
