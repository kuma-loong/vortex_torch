please help me check, why Qwen/Qwen3-32B Qwen/Qwen3-4B models work well in vortex_torch while Qwen/Qwen3-30B-A3B cannot.

In run_ruler.py, Qwen/Qwen3-30B-A3B produces either 0% accuracy or illegal memory access, Other models produce > 97% accuracy, which is normal.

You must use vortex_v1 as conda environment. Do not use GPU id=0, which has problem. 

You might need to add torch.cuda.synchronize() or print(.....) to inspect what is happening in GPUs.

We are using third_party/sglang/v0.5.9/sglang as our serving backend.


