please help me check, why `disable_overlap_schedule=True` work well in vortex_torch while `disable_overlap_schedule=False` can work well. 

In run_ruler.py,  produces either 0% accuracy or illegal memory access, Other models produce > 97% accuracy, which is normal.

You must use vortex_v1 as conda environment. Do not use GPU id=0, which has problem. 

You might need to add torch.cuda.synchronize() or print(.....) to inspect what is happening in GPUs.

We are using third_party/sglang/v0.5.9/sglang as our serving backend.

Help me fix this problem in both flashinfer and trtllm backends in vortex_torch. Verify this with run_ruler.py but with `disable_overlap_schedule=False`.