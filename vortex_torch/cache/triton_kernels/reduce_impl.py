import torch
import triton
import triton.language as tl
from ..context import Context

@triton.jit
def mean_kernel(
    x,                     # *flat* pointer to input pages, row-major per page
    output,                # *flat* pointer to output buffer
    loc,                   # pointer to per-token positions (int32/int64)
    x_D0: tl.constexpr,    # rows   of one page
    x_D1: tl.constexpr,    # cols   of one page
    NUM_KV_HEAD: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    DIM: tl.constexpr      # 0 -> mean over rows (per-column mean); 1 -> mean over cols (per-row mean)
):
    """
    This kernel computes a mean vector for a full page [x_D0, x_D1] exactly once,
    triggered by the *last* token of that page.

    Grid:
      pid0 = token_id  (typically launched as num_tokens)
      pid1 = head_id   (0..NUM_KV_HEAD-1)

    Behavior:
      - Read token_position = loc[token_id]
      - Only proceed if (token_position + 1) % PAGE_SIZE == 0 (page-end)
      - Compute page_id = (token_position // PAGE_SIZE) * NUM_KV_HEAD + head_id
      - Treat x as packed pages; each page is a contiguous [x_D0, x_D1] block in row-major order
      - If DIM == 1: mean over rows -> length x_D1; write at output[page_id, :x_D1]
        If DIM == 2: mean over cols -> length x_D0; write at output[page_id, :x_D0]

    Notes:
      - No upcast to fp32 as requested; reductions happen in the input dtype.
      - Assumes each page is full (no partial page at end).
    """

    token_id = tl.program_id(0)
    head_id = tl.program_id(1)

    # Load the absolute token position for this token_id
    token_position = tl.load(loc + token_id)

    # Only compute at page end to avoid duplicate work
    if (token_position + 1) % PAGE_SIZE != 0:
        return

    # Map (page, head) to a linear page_id
    page_id = (token_position // PAGE_SIZE) * NUM_KV_HEAD + head_id

    # Byte/element offset to the start of this page in `x`
    # Each page is a contiguous block of x_D0 * x_D1 elements
    x_offset = page_id * x_D0 * x_D1

    # Build a 2D index for the page load: row-major layout
    rows = tl.arange(0, x_D0)[:, None]              # shape [x_D0, 1]
    cols = tl.arange(0, x_D1)[None, :]              # shape [1, x_D1]
    # Linearized 2D addressing: i * x_D1 + j
    src_ptr = x + x_offset + rows * x_D1 + cols

    # Load the full page block: shape [x_D0, x_D1], in the underlying dtype of `x`
    page_block = tl.load(src_ptr)

    if DIM == 1:
        # Mean over rows (per-column mean) -> vector length x_D1
        mean_vec = (tl.sum(page_block, axis=0) / x_D0).to(tl.bfloat16)
        dst_ptr = output + page_id * x_D1 + tl.arange(0, x_D1)
        tl.store(dst_ptr, mean_vec)
    else:
        # DIM == 2: Mean over cols (per-row mean) -> vector length x_D0
        mean_vec = (tl.sum(page_block, axis=1) / x_D1).to(tl.bfloat16)
        dst_ptr = output + page_id * x_D0 + tl.arange(0, x_D0)
        tl.store(dst_ptr, mean_vec)


def mean_launcher(
x: torch.Tensor,
output: torch.Tensor,
loc: torch.LongTensor,
ctx: Context,
dim: int
):
    
    NNZ = loc.shape[0]
    NUM_KV_HEAD = ctx.head_num
    
    mean_kernel[(NNZ, NUM_KV_HEAD)](
        x=x,
        output=output,
        loc=loc,
        x_D0=x.shape[1],
        x_D1=x.shape[2],
        NUM_KV_HEAD=NUM_KV_HEAD,
        PAGE_SIZE=ctx.page_size,
        DIM=dim
    )
  
  
  
@triton.jit
def max_kernel(
    x,                     # *flat* pointer to input pages, row-major per page
    output,                # *flat* pointer to output buffer
    loc,                   # pointer to per-token positions (int32/int64)
    x_D0: tl.constexpr,    # rows   of one page
    x_D1: tl.constexpr,    # cols   of one page
    NUM_KV_HEAD: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    DIM: tl.constexpr      # 1 -> max over rows (per-column max); 2 -> max over cols (per-row max)
):
    """
    This kernel computes a max vector for a full page [x_D0, x_D1] exactly once,
    triggered by the *last* token of that page.

    Grid:
      pid0 = token_id  (typically launched as num_tokens)
      pid1 = head_id   (0..NUM_KV_HEAD-1)

    Behavior:
      - Read token_position = loc[token_id]
      - Only proceed if (token_position + 1) % PAGE_SIZE == 0 (page-end)
      - Compute page_id = (token_position // PAGE_SIZE) * NUM_KV_HEAD + head_id
      - Treat x as packed pages; each page is a contiguous [x_D0, x_D1] block in row-major order
      - If DIM == 1: max over rows -> length x_D1; write at output[page_id, :x_D1]
        If DIM == 2: max over cols -> length x_D0; write at output[page_id, :x_D0]

    Notes:
      - No upcast to fp32 as requested; reductions happen in the input dtype.
      - Assumes each page is full (no partial page at end).
    """

    token_id = tl.program_id(0)
    head_id = tl.program_id(1)

    # Load the absolute token position for this token_id
    token_position = tl.load(loc + token_id)

    # Only compute at page end to avoid duplicate work
    if (token_position + 1) % PAGE_SIZE != 0:
        return

    # Map (page, head) to a linear page_id
    page_id = (token_position // PAGE_SIZE) * NUM_KV_HEAD + head_id

    # Byte/element offset to the start of this page in `x`
    # Each page is a contiguous block of x_D0 * x_D1 elements
    x_offset = page_id * x_D0 * x_D1

    # Build a 2D index for the page load: row-major layout
    rows = tl.arange(0, x_D0)[:, None]              # shape [x_D0, 1]
    cols = tl.arange(0, x_D1)[None, :]              # shape [1, x_D1]
    # Linearized 2D addressing: i * x_D1 + j
    src_ptr = x + x_offset + rows * x_D1 + cols

    # Load the full page block: shape [x_D0, x_D1], in the underlying dtype of `x`
    page_block = tl.load(src_ptr)

    if DIM == 1:
        # Max over rows (per-column max) -> vector length x_D1
        max_vec = tl.max(page_block, axis=0).to(tl.bfloat16)
        dst_ptr = output + page_id * x_D1 + tl.arange(0, x_D1)
        tl.store(dst_ptr, max_vec)
    else:
        # DIM == 2: Max over cols (per-row max) -> vector length x_D0
        max_vec = tl.max(page_block, axis=1).to(tl.bfloat16)
        dst_ptr = output + page_id * x_D0 + tl.arange(0, x_D0)
        tl.store(dst_ptr, max_vec)

def max_launcher(
x: torch.Tensor,
output: torch.Tensor,
loc: torch.LongTensor,
ctx: Context,
dim: int
):
    
    NNZ = loc.shape[0]
    NUM_KV_HEAD = ctx.head_num
    
    max_kernel[(NNZ, NUM_KV_HEAD)](
        x=x,
        output=output,
        loc=loc,
        x_D0=x.shape[1],
        x_D1=x.shape[2],
        NUM_KV_HEAD=NUM_KV_HEAD,
        PAGE_SIZE=ctx.page_size,
        DIM=dim
    )
  


@triton.jit
def min_kernel(
    x,                     # *flat* pointer to input pages, row-major per page
    output,                # *flat* pointer to output buffer
    loc,                   # pointer to per-token positions (int32/int64)
    x_D0: tl.constexpr,    # rows   of one page
    x_D1: tl.constexpr,    # cols   of one page
    NUM_KV_HEAD: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    DIM: tl.constexpr      # 1 -> min over rows (per-column min); 2 -> min over cols (per-row min)
):
    """
    This kernel computes a min vector for a full page [x_D0, x_D1] exactly once,
    triggered by the *last* token of that page.

    Grid:
      pid0 = token_id  (typically launched as num_tokens)
      pid1 = head_id   (0..NUM_KV_HEAD-1)

    Behavior:
      - Read token_position = loc[token_id]
      - Only proceed if (token_position + 1) % PAGE_SIZE == 0 (page-end)
      - Compute page_id = (token_position // PAGE_SIZE) * NUM_KV_HEAD + head_id
      - Treat x as packed pages; each page is a contiguous [x_D0, x_D1] block in row-major order
      - If DIM == 1: min over rows -> length x_D1; write at output[page_id, :x_D1]
        If DIM == 2: min over cols -> length x_D0; write at output[page_id, :x_D0]

    Notes:
      - No upcast to fp32 as requested; reductions happen in the input dtype.
      - Assumes each page is full (no partial page at end).
    """

    token_id = tl.program_id(0)
    head_id = tl.program_id(1)

    # Load the absolute token position for this token_id
    token_position = tl.load(loc + token_id)

    # Only compute at page end to avoid duplicate work
    if (token_position + 1) % PAGE_SIZE != 0:
        return

    # Map (page, head) to a linear page_id
    page_id = (token_position // PAGE_SIZE) * NUM_KV_HEAD + head_id

    # Byte/element offset to the start of this page in `x`
    # Each page is a contiguous block of x_D0 * x_D1 elements
    x_offset = page_id * x_D0 * x_D1

    # Build a 2D index for the page load: row-major layout
    rows = tl.arange(0, x_D0)[:, None]              # shape [x_D0, 1]
    cols = tl.arange(0, x_D1)[None, :]              # shape [1, x_D1]
    # Linearized 2D addressing: i * x_D1 + j
    src_ptr = x + x_offset + rows * x_D1 + cols

    # Load the full page block: shape [x_D0, x_D1], in the underlying dtype of `x`
    page_block = tl.load(src_ptr)

    if DIM == 1:
        # Min over rows (per-column min) -> vector length x_D1
        min_vec = tl.min(page_block, axis=0).to(tl.bfloat16)
        dst_ptr = output + page_id * x_D1 + tl.arange(0, x_D1)
        tl.store(dst_ptr, min_vec)
    else:
        # DIM == 2: Min over cols (per-row min) -> vector length x_D0
        min_vec = tl.min(page_block, axis=1).to(tl.bfloat16)
        dst_ptr = output + page_id * x_D0 + tl.arange(0, x_D0)
        tl.store(dst_ptr, min_vec)

def min_launcher(
x: torch.Tensor,
output: torch.Tensor,
loc: torch.LongTensor,
ctx: Context,
dim: int
):
    
    NNZ = loc.shape[0]
    NUM_KV_HEAD = ctx.head_num
    
    min_kernel[(NNZ, NUM_KV_HEAD)](
        x=x,
        output=output,
        loc=loc,
        x_D0=x.shape[1],
        x_D1=x.shape[2],
        NUM_KV_HEAD=NUM_KV_HEAD,
        PAGE_SIZE=ctx.page_size,
        DIM=dim
    )