import json
import os
import sys
import sglang as sgl
import vortex_torch
from transformers import AutoTokenizer
def main():
    # $1: HF model id (positional, optional). Default: Qwen/Qwen3-4B.
    model_name = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-4B"

    # Baseline toggle: ENABLE_VORTEX_SPARSITY=0 runs dense sglang (no vortex
    # sparse path) to confirm the reference accuracy; default (1) is sparse.
    enable_vortex_sparsity = os.environ.get("ENABLE_VORTEX_SPARSITY", "1") == "1"
    print(f"[run_ruler] enable_vortex_sparsity={enable_vortex_sparsity}", flush=True)

    default_policy = r"""
const int static_kv_budget = topk_val + block_reserved_bos + block_reserved_eos;
const int dynamic_kv_budget = int(cached_block_len * topk_ratio);
return max(static_kv_budget, dynamic_kv_budget);
"""

    llm = sgl.Engine(model_path=model_name, 
                    disable_cuda_graph=False,
                    page_size=16,
                    vortex_block_size=16,
                    vortex_topk_val=29,
                    disable_overlap_schedule=False,
                    kv_cache_dtype="auto",
                    vortex_dtype="bfloat16",
                    attention_backend="flashinfer",
                    vortex_schedule_policy=default_policy,
                    enable_vortex_sparsity=enable_vortex_sparsity,
                    vortex_block_reserved_bos=1,
                    vortex_block_reserved_eos=2,
                    vortex_layers_skip=list(range(1)),
                    vortex_module_name="gqa_block_sparse_attention",
                    vortex_attention_backend="trtllm",
                    trust_remote_code=True,
                    #vortex_module_path="submissions/example_block_sparse_attention.py",
                    vortex_max_seq_lens=8192,
                    mem_fraction_static=0.9,
                    vortex_workload_chunk_size=32,
                    vortex_compilation_cache_dir="~/.vortex_compilation_cache",
                    tp_size=1,
                    )
    
    with open("examples/validation.jsonl", "r", encoding="utf-8") as f:
        ruler_data = [json.loads(line)["input"] for line in f]

    with open("examples/validation.jsonl", "r", encoding="utf-8") as f:
        ruler_outputs = [json.loads(line)["outputs"][0] for line in f]
    
    texts = [
        [{"role":"user","content": x}] for x in ruler_data
    ]
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prompts = [
        tokenizer.apply_chat_template(
        text,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    ) for text in texts
    ]
    # MiniMax-M2 is a reasoning model: it emits a chain-of-thought preamble
    # before stating the answer, so a 64-token cap truncates the answer
    # (RULER scores by substring match). Give it room to finish.
    sampling_params = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "max_new_tokens": 1024}
    accuracy = 0
    with open("examples/ruler_output.jsonl", "w", encoding="utf-8") as f:
            o = llm.generate(prompts, sampling_params)
            for res, answer in zip(o, ruler_outputs):
                    json.dump(res, f, ensure_ascii=False)
                    f.write("\n")
                    if answer in res["text"]:
                        accuracy += 1.0
    print(f"Ruler Accuracy: {accuracy / len(ruler_outputs) * 100:.2f}%")

if __name__ == "__main__":
    main()
