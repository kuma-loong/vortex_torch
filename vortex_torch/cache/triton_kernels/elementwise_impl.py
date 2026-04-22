import torch
import triton
import triton.language as tl
from ..context import Context
from ...utils import ElementwiseOpType, QuantizationType
from .utils_impl import _quant_view

@triton.jit
def elementwise_pp_kernel(
x, output, loc,
x_D0: tl.constexpr,
x_D1: tl.constexpr,
o_D0: tl.constexpr,
o_D1: tl.constexpr,
NUM_KV_HEAD: tl.constexpr,
PAGE_SIZE: tl.constexpr,
OP_TYPE: tl.constexpr,
alpha: tl.constexpr,
beta: tl.constexpr,
QUANT_TYPE: tl.constexpr,  # 0:bf16, 1:fp8_e5m2, 2:fp8_e4m3
):

    token_id = tl.program_id(0)
    head_id  = tl.program_id(1)

    token_position = tl.load(loc + token_id)

    if (token_position + 1) % PAGE_SIZE != 0:
        return

    page_id  = (token_position // PAGE_SIZE) * NUM_KV_HEAD + head_id
    x_offset = page_id * x_D0 * x_D1
    o_offset = page_id * o_D0 * o_D1

    x_raw = tl.load(x + x_offset + tl.arange(0, x_D0)[:, None] * x_D1 + tl.arange(0, x_D1)[None, :])

    # Cast to fp32 (with bitcast for FP8) so the op runs in fp32.
    if QUANT_TYPE == 0:
        x_f = x_raw.to(tl.float32)
    elif QUANT_TYPE == 1:
        x_f = x_raw.to(tl.float8e5, bitcast=True).to(tl.float32)
    elif QUANT_TYPE == 2:
        x_f = x_raw.to(tl.float8e4nv, bitcast=True).to(tl.float32)

    alpha_f = tl.full((), alpha, dtype=tl.float32)
    beta_f  = tl.full((), beta,  dtype=tl.float32)

    # ----- Elementwise ops (fp32) -----
    if OP_TYPE == 0:
        # piecewise: x >= alpha ? x : beta
        o_f = tl.where(x_f >= alpha_f, x_f, beta_f)

    elif OP_TYPE == 1:
        # σ(alpha, beta; x) = 1 / (1 + exp(beta * x + alpha))
        z = beta_f * x_f + alpha_f
        o_f = 1.0 / (1.0 + tl.exp(z))

    elif OP_TYPE == 2:
        # SiLU(alpha, beta; x) = x / (1 + exp(beta * x + alpha))
        z = beta_f * x_f + alpha_f
        o_f = x_f / (1.0 + tl.exp(z))

    elif OP_TYPE == 3:
        # |beta * x + alpha|
        z = beta_f * x_f + alpha_f
        o_f = tl.abs(z)

    elif OP_TYPE == 4:
        # beta * x + alpha
        o_f = beta_f * x_f + alpha_f

    o_i = o_f.to(tl.bfloat16)

    tl.store(output + o_offset + tl.arange(0, o_D0)[:, None] * o_D1 + tl.arange(0, o_D1)[None, :], o_i)


def elementwise_pp(
x: torch.Tensor,
output: torch.Tensor,
loc: torch.LongTensor,
ctx: Context,
op_type: ElementwiseOpType,
alpha: float,
beta: float,
quantization_type: QuantizationType = QuantizationType.BF16,
):

    NNZ = loc.shape[0]
    NUM_KV_HEAD = ctx.head_num
    x = _quant_view(x, quantization_type)

    elementwise_pp_kernel[(NNZ, NUM_KV_HEAD)](
        x=x,
        output=output,
        loc=loc,
        x_D0=x.shape[1],
        x_D1=x.shape[2],
        o_D0=output.shape[1],
        o_D1=output.shape[2],
        NUM_KV_HEAD=NUM_KV_HEAD,
        PAGE_SIZE=ctx.page_size,
        OP_TYPE=op_type.value,
        alpha=alpha,
        beta=beta,
        QUANT_TYPE=quantization_type.value,
    )


def _elementwise_pp(
x: torch.Tensor,
output: torch.Tensor,
loc: torch.LongTensor,
num_kv_heads: int,
page_size: int,
op_type: ElementwiseOpType,
alpha: float,
beta: float,
quantization_type: QuantizationType = QuantizationType.BF16,
):

    NNZ = loc.shape[0]
    NUM_KV_HEAD = num_kv_heads
    x = _quant_view(x, quantization_type)

    elementwise_pp_kernel[(NNZ, NUM_KV_HEAD)](
        x=x,
        output=output,
        loc=loc,
        x_D0=x.shape[1],
        x_D1=x.shape[2],
        o_D0=output.shape[1],
        o_D1=output.shape[2],
        NUM_KV_HEAD=NUM_KV_HEAD,
        PAGE_SIZE=page_size,
        OP_TYPE=op_type.value,
        alpha=alpha,
        beta=beta,
        QUANT_TYPE=quantization_type.value,
    )
        

@triton.jit
def elementwise_rp_kernel(
    x, output, loc,
    x_D0: tl.constexpr,   # rows of x tile
    x_D1: tl.constexpr,   # cols of x tile
    o_D0: tl.constexpr,   # rows of output tile
    o_D1: tl.constexpr,   # cols of output tile
    NUM_KV_HEAD: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    OP_TYPE: tl.constexpr,  # 0: piecewise(x>=alpha?x:beta), 1: sigmoid(affine), 2: silu-like, 3: abs(affine), 4: affine
    alpha: tl.constexpr,
    beta: tl.constexpr,
    QUANT_TYPE: tl.constexpr,  # 0:bf16, 1:fp8_e5m2, 2:fp8_e4m3
):
    # -----------------------------
    # Program indices
    # -----------------------------
    token_id = tl.program_id(0)  # token-major index
    head_id  = tl.program_id(1)  # head index

    # -----------------------------
    # Trigger only on end-of-page tokens
    # -----------------------------
    token_position = tl.load(loc + token_id)
    if (token_position + 1) % PAGE_SIZE != 0:
        return

    # -----------------------------
    # Input offset: token-major (RAGGED)
    # -----------------------------
    x_off = (token_id * NUM_KV_HEAD + head_id) * x_D0 * x_D1

    # -----------------------------
    # Output offset: page-major (PAGED)
    # -----------------------------
    page_idx = token_position // PAGE_SIZE
    page_id  = page_idx * NUM_KV_HEAD + head_id
    o_off    = page_id * o_D0 * o_D1

    # -----------------------------
    # Build 2D row-major indices and load tile
    # -----------------------------
    x_rows = tl.arange(0, x_D0)[:, None]
    x_cols = tl.arange(0, x_D1)[None, :]
    x_raw = tl.load(x + x_off + x_rows * x_D1 + x_cols)  # assumes full tile; no mask

    # Cast to fp32 (with bitcast for FP8) so the op runs in fp32.
    if QUANT_TYPE == 0:
        x_f = x_raw.to(tl.float32)
    elif QUANT_TYPE == 1:
        x_f = x_raw.to(tl.float8e5, bitcast=True).to(tl.float32)
    elif QUANT_TYPE == 2:
        x_f = x_raw.to(tl.float8e4nv, bitcast=True).to(tl.float32)

    alpha_f = tl.full((), alpha, dtype=tl.float32)
    beta_f  = tl.full((), beta,  dtype=tl.float32)

    # -----------------------------
    # Elementwise ops (fp32)
    # -----------------------------
    if OP_TYPE == 0:
        # piecewise: x >= alpha ? x : beta
        o_f = tl.where(x_f >= alpha_f, x_f, beta_f)

    elif OP_TYPE == 1:
        # sigma(alpha, beta; x) = 1 / (1 + exp(beta * x + alpha))
        z = beta_f * x_f + alpha_f
        o_f = 1.0 / (1.0 + tl.exp(z))

    elif OP_TYPE == 2:
        # silu-like(alpha, beta; x) = x / (1 + exp(beta * x + alpha))
        z = beta_f * x_f + alpha_f
        o_f = x_f / (1.0 + tl.exp(z))

    elif OP_TYPE == 3:
        # abs(beta * x + alpha)
        z = beta_f * x_f + alpha_f
        o_f = tl.abs(z)

    else:  # OP_TYPE == 4
        # affine: beta * x + alpha
        o_f = beta_f * x_f + alpha_f

    o_i = o_f.to(tl.bfloat16)

    # -----------------------------
    # Store to output page (PAGED)
    # -----------------------------
    o_rows = tl.arange(0, o_D0)[:, None]
    o_cols = tl.arange(0, o_D1)[None, :]
    tl.store(output + o_off + o_rows * o_D1 + o_cols, o_i)


def elementwise_rp(
x: torch.Tensor,
output: torch.Tensor,
loc: torch.LongTensor,
ctx: Context,
op_type: ElementwiseOpType,
alpha: float,
beta: float,
quantization_type: QuantizationType = QuantizationType.BF16,
):

    NNZ = loc.shape[0]
    NUM_KV_HEAD = ctx.head_num
    x = _quant_view(x, quantization_type)

    elementwise_rp_kernel[(NNZ, NUM_KV_HEAD)](
        x=x,
        output=output,
        loc=loc,
        x_D0=x.shape[1],
        x_D1=x.shape[2],
        o_D0=output.shape[1],
        o_D1=output.shape[2],
        NUM_KV_HEAD=NUM_KV_HEAD,
        PAGE_SIZE=ctx.page_size,
        OP_TYPE=op_type.value,
        alpha=alpha,
        beta=beta,
        QUANT_TYPE=quantization_type.value,
    )


def _elementwise_rp(
x: torch.Tensor,
output: torch.Tensor,
loc: torch.LongTensor,
num_kv_heads: int,
page_size: int,
op_type: ElementwiseOpType,
alpha: float,
beta: float,
quantization_type: QuantizationType = QuantizationType.BF16,
):

    NNZ = loc.shape[0]
    NUM_KV_HEAD = num_kv_heads
    x = _quant_view(x, quantization_type)

    elementwise_rp_kernel[(NNZ, NUM_KV_HEAD)](
        x=x,
        output=output,
        loc=loc,
        x_D0=x.shape[1],
        x_D1=x.shape[2],
        o_D0=output.shape[1],
        o_D1=output.shape[2],
        NUM_KV_HEAD=NUM_KV_HEAD,
        PAGE_SIZE=page_size,
        OP_TYPE=op_type.value,
        alpha=alpha,
        beta=beta,
        QUANT_TYPE=quantization_type.value,
    )
    
@triton.jit
def elementwise_pr_kernel(
    x, output, loc,
    x_D0: tl.constexpr,  # rows of x page
    x_D1: tl.constexpr,  # cols of x page
    o_D0: tl.constexpr,  # rows of output tile
    o_D1: tl.constexpr,  # cols of output tile
    NUM_KV_HEAD: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    OP_TYPE: tl.constexpr,  # 0: piecewise(x>=alpha?x:beta), 1: sigmoid(affine), 2: silu-like, 3: abs(affine), 4: affine
    alpha: tl.constexpr,
    beta: tl.constexpr,
    QUANT_TYPE: tl.constexpr,  # 0:bf16, 1:fp8_e5m2, 2:fp8_e4m3
):
    # -----------------------------
    # Program indices
    # -----------------------------
    token_id = tl.program_id(0)  # token-major index
    head_id  = tl.program_id(1)  # head index

    # -----------------------------
    # Trigger only on end-of-page tokens
    # -----------------------------
    token_position = tl.load(loc + token_id)
    if (token_position + 1) % PAGE_SIZE != 0:
        return

    # -----------------------------
    # Page-major indexing for x (PAGED)
    # -----------------------------
    page_idx = token_position // PAGE_SIZE
    page_id  = page_idx * NUM_KV_HEAD + head_id
    x_off    = page_id * x_D0 * x_D1

    # -----------------------------
    # Token-major indexing for output (RAGGED)
    # -----------------------------
    out_token_lin = (token_id * NUM_KV_HEAD + head_id)
    o_off         = out_token_lin * o_D0 * o_D1

    # -----------------------------
    # Build 2D row-major indices and load x page
    # -----------------------------
    x_rows = tl.arange(0, x_D0)[:, None]
    x_cols = tl.arange(0, x_D1)[None, :]
    x_raw = tl.load(x + x_off + x_rows * x_D1 + x_cols)  # assumes full page; no mask

    # Cast to fp32 (with bitcast for FP8) so the op runs in fp32.
    if QUANT_TYPE == 0:
        x_f = x_raw.to(tl.float32)
    elif QUANT_TYPE == 1:
        x_f = x_raw.to(tl.float8e5, bitcast=True).to(tl.float32)
    elif QUANT_TYPE == 2:
        x_f = x_raw.to(tl.float8e4nv, bitcast=True).to(tl.float32)

    alpha_f = tl.full((), alpha, dtype=tl.float32)
    beta_f  = tl.full((), beta,  dtype=tl.float32)

    # -----------------------------
    # Elementwise ops (fp32)
    # -----------------------------
    if OP_TYPE == 0:
        # piecewise: x >= alpha ? x : beta
        o_f = tl.where(x_f >= alpha_f, x_f, beta_f)

    elif OP_TYPE == 1:
        # sigma(alpha, beta; x) = 1 / (1 + exp(beta * x + alpha))
        z = beta_f * x_f + alpha_f
        o_f = 1.0 / (1.0 + tl.exp(z))

    elif OP_TYPE == 2:
        # silu-like(alpha, beta; x) = x / (1.0 + exp(beta * x + alpha))
        z = beta_f * x_f + alpha_f
        o_f = x_f / (1.0 + tl.exp(z))

    elif OP_TYPE == 3:
        # abs(beta * x + alpha)
        z = beta_f * x_f + alpha_f
        o_f = tl.abs(z)

    else:  # OP_TYPE == 4
        # affine: beta * x + alpha
        o_f = beta_f * x_f + alpha_f

    o_i = o_f.to(tl.bfloat16)

    # -----------------------------
    # Store into token-major output (RAGGED)
    # -----------------------------
    o_rows = tl.arange(0, o_D0)[:, None]
    o_cols = tl.arange(0, o_D1)[None, :]
    tl.store(output + o_off + o_rows * o_D1 + o_cols, o_i)


def elementwise_pr(
x: torch.Tensor,
output: torch.Tensor,
loc: torch.LongTensor,
ctx: Context,
op_type: ElementwiseOpType,
alpha: float,
beta: float,
quantization_type: QuantizationType = QuantizationType.BF16,
):

    NNZ = loc.shape[0]
    NUM_KV_HEAD = ctx.head_num
    x = _quant_view(x, quantization_type)

    elementwise_pr_kernel[(NNZ, NUM_KV_HEAD)](
        x=x,
        output=output,
        loc=loc,
        x_D0=x.shape[1],
        x_D1=x.shape[2],
        o_D0=output.shape[1],
        o_D1=output.shape[2],
        NUM_KV_HEAD=NUM_KV_HEAD,
        PAGE_SIZE=ctx.page_size,
        OP_TYPE=op_type.value,
        alpha=alpha,
        beta=beta,
        QUANT_TYPE=quantization_type.value,
    )


def _elementwise_pr(
x: torch.Tensor,
output: torch.Tensor,
loc: torch.LongTensor,
num_kv_heads: int,
page_size: int,
op_type: ElementwiseOpType,
alpha: float,
beta: float,
quantization_type: QuantizationType = QuantizationType.BF16,
):

    NNZ = loc.shape[0]
    NUM_KV_HEAD = num_kv_heads
    x = _quant_view(x, quantization_type)

    elementwise_pr_kernel[(NNZ, NUM_KV_HEAD)](
        x=x,
        output=output,
        loc=loc,
        x_D0=x.shape[1],
        x_D1=x.shape[2],
        o_D0=output.shape[1],
        o_D1=output.shape[2],
        NUM_KV_HEAD=NUM_KV_HEAD,
        PAGE_SIZE=page_size,
        OP_TYPE=op_type.value,
        alpha=alpha,
        beta=beta,
        QUANT_TYPE=quantization_type.value,
    )
    
@triton.jit
def elementwise_rr_kernel(
    x, output, loc,
    x_D0: tl.constexpr,  # rows of x tile
    x_D1: tl.constexpr,  # cols of x tile
    o_D0: tl.constexpr,  # rows of output tile
    o_D1: tl.constexpr,  # cols of output tile
    NUM_KV_HEAD: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    OP_TYPE: tl.constexpr,  # 0: piecewise(x>=alpha?x:beta), 1: sigmoid(affine), 2: silu-like, 3: abs(affine), 4: affine
    alpha: tl.constexpr,
    beta: tl.constexpr,
    QUANT_TYPE: tl.constexpr,  # 0:bf16, 1:fp8_e5m2, 2:fp8_e4m3
):
    # -----------------------------
    # Program indices
    # -----------------------------
    token_id = tl.program_id(0)  # token-major linear index
    head_id  = tl.program_id(1)  # head index

    # -----------------------------
    # End-of-page trigger
    # -----------------------------
    token_position = tl.load(loc + token_id)
    if (token_position + 1) % PAGE_SIZE != 0:
        return

    # -----------------------------
    # Token-major offsets (RAGGED) for x and output
    # -----------------------------
    x_off = (token_id * NUM_KV_HEAD + head_id) * x_D0 * x_D1
    o_off = (token_id * NUM_KV_HEAD + head_id) * o_D0 * o_D1

    # -----------------------------
    # Build 2D row-major indices and load x tile
    # -----------------------------
    x_rows = tl.arange(0, x_D0)[:, None]
    x_cols = tl.arange(0, x_D1)[None, :]
    x_raw = tl.load(x + x_off + x_rows * x_D1 + x_cols)  # assumes full tile; no mask

    # Cast to fp32 (with bitcast for FP8) so the op runs in fp32.
    if QUANT_TYPE == 0:
        x_f = x_raw.to(tl.float32)
    elif QUANT_TYPE == 1:
        x_f = x_raw.to(tl.float8e5, bitcast=True).to(tl.float32)
    elif QUANT_TYPE == 2:
        x_f = x_raw.to(tl.float8e4nv, bitcast=True).to(tl.float32)

    alpha_f = tl.full((), alpha, dtype=tl.float32)
    beta_f  = tl.full((), beta,  dtype=tl.float32)

    # -----------------------------
    # Elementwise unary ops (fp32)
    # -----------------------------
    if OP_TYPE == 0:
        # piecewise: x >= alpha ? x : beta
        o_f = tl.where(x_f >= alpha_f, x_f, beta_f)

    elif OP_TYPE == 1:
        # sigma(alpha, beta; x) = 1 / (1 + exp(beta * x + alpha))
        z = beta_f * x_f + alpha_f
        o_f = 1.0 / (1.0 + tl.exp(z))

    elif OP_TYPE == 2:
        # silu-like(alpha, beta; x) = x / (1.0 + exp(beta * x + alpha))
        z = beta_f * x_f + alpha_f
        o_f = x_f / (1.0 + tl.exp(z))

    elif OP_TYPE == 3:
        # abs(beta * x + alpha)
        z = beta_f * x_f + alpha_f
        o_f = tl.abs(z)

    else:  # OP_TYPE == 4
        # affine: beta * x + alpha
        o_f = beta_f * x_f + alpha_f

    o_i = o_f.to(tl.bfloat16)

    # -----------------------------
    # Store to token-major output (RAGGED)
    # -----------------------------
    o_rows = tl.arange(0, o_D0)[:, None]
    o_cols = tl.arange(0, o_D1)[None, :]
    tl.store(output + o_off + o_rows * o_D1 + o_cols, o_i)


def elementwise_rr(
x: torch.Tensor,
output: torch.Tensor,
loc: torch.LongTensor,
ctx: Context,
op_type: ElementwiseOpType,
alpha: float,
beta: float,
quantization_type: QuantizationType = QuantizationType.BF16,
):

    NNZ = loc.shape[0]
    NUM_KV_HEAD = ctx.head_num
    x = _quant_view(x, quantization_type)

    elementwise_rr_kernel[(NNZ, NUM_KV_HEAD)](
        x=x,
        output=output,
        loc=loc,
        x_D0=x.shape[1],
        x_D1=x.shape[2],
        o_D0=output.shape[1],
        o_D1=output.shape[2],
        NUM_KV_HEAD=NUM_KV_HEAD,
        PAGE_SIZE=ctx.page_size,
        OP_TYPE=op_type.value,
        alpha=alpha,
        beta=beta,
        QUANT_TYPE=quantization_type.value,
    )


def _elementwise_rr(
x: torch.Tensor,
output: torch.Tensor,
loc: torch.LongTensor,
num_kv_heads: int,
page_size: int,
op_type: ElementwiseOpType,
alpha: float,
beta: float,
quantization_type: QuantizationType = QuantizationType.BF16,
):

    NNZ = loc.shape[0]
    NUM_KV_HEAD = num_kv_heads
    x = _quant_view(x, quantization_type)

    elementwise_rr_kernel[(NNZ, NUM_KV_HEAD)](
        x=x,
        output=output,
        loc=loc,
        x_D0=x.shape[1],
        x_D1=x.shape[2],
        o_D0=output.shape[1],
        o_D1=output.shape[2],
        NUM_KV_HEAD=NUM_KV_HEAD,
        PAGE_SIZE=page_size,
        OP_TYPE=op_type.value,
        alpha=alpha,
        beta=beta,
        QUANT_TYPE=quantization_type.value,
    )