import sglang as sgl
from transformers import AutoTokenizer
from lighteval.metrics.dynamic_metrics import (
    ExprExtractionConfig,
    LatexExtractionConfig,
    MultilingualExtractiveMatchMetric
)
from lighteval.tasks.requests import Doc
from lighteval.utils.language import Language
import jsonlines
from lighteval.models.model_output import ModelResponse
from datetime import datetime
from datasets import load_dataset, Dataset, concatenate_datasets

MATH_QUERY_TEMPLATE = """
Solve the following math problem efficiently and clearly.  The last line of your response should be of the following format: 'Therefore, the final answer is: $\\boxed{{ANSWER}}$. I hope it is correct' (without quotes) where ANSWER is just the final number or expression that solves the problem. Think step by step before answering.

{Question}
""".strip()

def generate_requests(dataset: Dataset, field_name: str, data_format: str, trial: int = 1, rank: int = 0, world_size: int = 1):
    requests = []

    # Step 1: Expand dataset trial times
    if trial > 1:
        dataset = Dataset.from_dict(dataset.to_dict().copy())  # ensure copy
        datasets = [dataset] * trial
        dataset = concatenate_datasets(datasets)
    
    total = len(dataset)
    
    # Step 2: Partition across ranks
    per_proc = total // world_size
    remainder = total % world_size
    start = rank * per_proc + min(rank, remainder)
    end = start + per_proc + (1 if rank < remainder else 0)
    subset = dataset.select(list(range(start, end)))

    # Step 3: Format requests
    for data in subset:
        conversations = [
            {"role": "user", "content": data_format.format(Question=data[field_name])}
        ]
        data["conversations"] = conversations
        requests.append(data)

    return requests

def verify_algos(
trials: int = 2,
topk_val: int = 30,
vortex_module_name: str = "gqa_block_sparse_attention",
model_name: str = "Qwen/Qwen3-1.7B"
):  
    
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file = f"{current_time}.jsonl"
    llm = sgl.Engine(model_path=model_name, 
                    disable_cuda_graph=False,
                    page_size=16,
                    vortex_topk_val=topk_val,   
                    disable_overlap_schedule=True,
                    attention_backend="flashinfer",
                    enable_vortex_sparsity=True,
                    vortex_page_reserved_bos=1,
                    vortex_page_reserved_eos=1,
                    vortex_layers_skip=list(range(1)),
                    vortex_module_name=vortex_module_name,
                    vortex_max_seq_lens=12288,
                    mem_fraction_static=0.8
                    )
    
    dataset = load_dataset("math-ai/amc23", split="test")
    requests = generate_requests(dataset, "question", MATH_QUERY_TEMPLATE)
    requests = requests * trials
    
    texts = [
        x["conversations"] for x in requests
    ]
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prompts = [
        tokenizer.apply_chat_template(
        text,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    ) for text in texts
    ]

    sampling_params = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "max_new_tokens": 8192}
    
    o = llm.generate(prompts, sampling_params)
    gold_metric =  MultilingualExtractiveMatchMetric(
            language=Language.ENGLISH,
            fallback_mode="first_match",
            precision=5,
            gold_extraction_target=(ExprExtractionConfig(),),
            pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig(boxed_match_priority=0)),
            aggregation_function=max,
        )
    
    results = []
    for data, item in zip(requests, o):
        golds = [data["answer"]]
        target = Doc(query=data["question"],choices=golds, gold_index=0)
        predictions = item["text"]
        try:
            result = gold_metric.compute(model_response=ModelResponse(text=[predictions]), doc=target)
        except:
            result = 0.0
        
        results.append(
            {
                "score": float(result),
                "prediction": [predictions],
                "choices": golds,
                "query": data["question"],
                "e2e_latency": item["meta_info"]["e2e_latency"],
                "num_tokens": item["meta_info"]["completion_tokens"]
            }
        )
    

    total_accuracy = 0.0
    total_tokens = 0
    e2e_time = 0
    count = 0
    unique_result = {}

    for item in results:
        total_accuracy += item['score']
        count += 1
        total_tokens += item["num_tokens"]
        e2e_time = max(e2e_time, item["e2e_latency"])
        if item['query'] not in unique_result:
            unique_result[item['query']] = item["score"]
        else:
            unique_result[item['query']] = max(item["score"], unique_result[item['query']])

    global_summary = {
        f'mean@{trials}': total_accuracy / count if count > 0 else 0,
        f'pass@{trials}': sum(unique_result.values()) / len(unique_result),
        'total_example': count,
        "e2e_time": e2e_time,
        "total_tokens": total_tokens, 
        "throughput": total_tokens / e2e_time
    }
    
    results.insert(0, global_summary)

    with jsonlines.open(output_file, "w") as f:
        f.write_all(results)

    return global_summary

if __name__ == "__main__":
   summary = verify_algos()
   print(summary)

