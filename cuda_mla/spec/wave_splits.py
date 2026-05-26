# Wave-aware split-KV selection (host-side, cuda-graph-safe: bs + SM count only).
# Calibrated on B200: optimal split-KV puts ~one wave of 2 CTAs/SM => bs*splits<=~296,
# so splits = floor(2*SM / bs), clamped (low-bs cap avoids tiny-chunk/stage2 overhead).
def wave_splits(bs, sm_count=148, cap=32):
    return max(1, min(cap, (2 * sm_count) // bs))
