# GLM-4.7-Flash · MLA sparse-attention experiments (AIME26, mean@16)

**Driver:** `examples/algo2.sh` → `examples/verify_algo.py`
&nbsp;·&nbsp; **Data:** `examples/aime26_glm.jsonl`
&nbsp;·&nbsp; **Model:** `zai-org/GLM-4.7-Flash`
&nbsp;·&nbsp; **trials:** 16 &nbsp;·&nbsp; **block=page=16** &nbsp;·&nbsp; **gen ≤ 32768 tok** &nbsp;·&nbsp; **tp=1**

## Plan

| # | module | backend | sparsity | runs |
|---|--------|---------|----------|------|
| 1 | `full_attention` | `flashinfer` (dense MLA) | — (dense baseline) | **1** (topk ignored) |
| 2 | `rope_aware_block_sparse_mla` | `cuda_mla` decode + Triton **tensor-core** indexer, **no layer skip** | topk ∈ {61,93,125,157,253} | 5 |
| 3 | `lserve_centroid_mla` | `cuda_mla` decode + Triton **tensor-core** indexer, **no layer skip** | topk ∈ {61,93,125,157,253} | 5 |

*Why these settings.* `full_attention` is dense, so it cannot use a vortex sparse
backend (those require `enable_vortex_sparsity=True`) — it runs on the dense
flashinfer MLA path as the accuracy ceiling / throughput-floor reference, once.
The two sparse modules run on the geometry-agnostic **`cuda_mla`** hand-CUDA decode;
**`--vortex-use-tensor-core`** turns on the bf16-MMA Triton indexer (hence
`--vortex-impl-backend triton`); **`--vortex-layers-skip`** with no values keeps
*every* layer sparse. The topk sweep traces the accuracy/throughput Pareto.


## Results

*Generated 2026-05-27 23:42:46 from `summary-glm4.7-flash/` (21 run(s)).*

| module | backend | topk | blk | TC | skip | mean@16 | pass@16 | pass@8 | pass@4 | tok/s | out_tok | e2e_s |
|---|---|--:|--:|:--:|:--:|--:|--:|--:|--:|--:|--:|--:|
| `full_attention` | triton | 253 | 32 | · | [0] | 0.7042 | 0.9333 | 0.8929 | 0.8474 | 980.5 | 8585368 | 8755.8 |
| `lserve_centroid_mla` | cuda_mla | 29 | 64 | ✓ | none | 0.7021 | 0.9000 | 0.8812 | 0.8487 | 2204.3 | 8457942 | 3837.1 |
| `lserve_centroid_mla` | cuda_mla | 61 | 32 | ✓ | none | 0.7125 | 0.9000 | 0.8782 | 0.8416 | 4225.9 | 8362906 | 1978.9 |
| `lserve_centroid_mla` | cuda_mla | 61 | 64 | ✓ | none | 0.7417 | 0.9333 | 0.8821 | 0.8485 | 2096.3 | 8211048 | 3917.0 |
| `lserve_centroid_mla` | cuda_mla | 61 | 16 | ✓ | none | 0.6771 | 0.8667 | 0.8582 | 0.8192 | 4454.0 | 8734383 | 1961.0 |
| `lserve_centroid_mla` | cuda_mla | 93 | 16 | ✓ | none | 0.7042 | 0.9000 | 0.8661 | 0.8363 | 4183.3 | 8522305 | 2037.2 |
| `lserve_centroid_mla` | cuda_mla | 125 | 32 | ✓ | none | 0.7354 | 0.9000 | 0.8795 | 0.8524 | 3627.9 | 8581143 | 2365.3 |
| `lserve_centroid_mla` | cuda_mla | 157 | 16 | ✓ | none | 0.7125 | 0.9000 | 0.8839 | 0.8478 | 4236.4 | 8569544 | 2022.9 |
| `lserve_centroid_mla` | cuda_mla | 253 | 16 | ✓ | none | 0.7479 | 0.9000 | 0.8889 | 0.8637 | 3924.7 | 8466111 | 2157.1 |
| `rope_aware_block_sparse_mla` | cuda_mla | 29 | 64 | ✓ | none | 0.7083 | 0.9000 | 0.8752 | 0.8444 | 4616.0 | 8552721 | 1852.9 |
| `rope_aware_block_sparse_mla` | triton | 61 | 16 | ✓ | none | 0.6479 | 0.9000 | 0.8754 | 0.8170 | 3121.7 | 8712928 | 2791.1 |
| `rope_aware_block_sparse_mla` | cuda_mla | 61 | 16 | ✓ | none | 0.6646 | 0.9000 | 0.8639 | 0.8106 | 3012.0 | 8608008 | 2857.9 |
| `rope_aware_block_sparse_mla` | cuda_mla | 61 | 64 | ✓ | none | 0.7271 | 0.8667 | 0.8487 | 0.8273 | 4113.7 | 8479595 | 2061.3 |
| `rope_aware_block_sparse_mla` | cuda_mla | 61 | 32 | ✓ | none | 0.7063 | 0.9333 | 0.8884 | 0.8399 | 4383.3 | 8624164 | 1967.5 |
| `rope_aware_block_sparse_mla` | triton | 93 | 16 | ✓ | none | 0.2667 | 0.6667 | 0.5876 | 0.4918 | 2979.4 | 3430788 | 1151.5 |
| `rope_aware_block_sparse_mla` | cuda_mla | 93 | 16 | ✓ | none | 0.7146 | 0.9000 | 0.8876 | 0.8538 | 2994.1 | 8441414 | 2819.4 |
| `rope_aware_block_sparse_mla` | cuda_mla | 125 | 32 | ✓ | none | 0.7542 | 0.9333 | 0.8983 | 0.8668 | 4104.1 | 8224331 | 2003.9 |
| `rope_aware_block_sparse_mla` | triton | 125 | 16 | ✓ | none | 0.2750 | 0.6333 | 0.5685 | 0.4817 | 2169.5 | 2598418 | 1197.7 |
| `rope_aware_block_sparse_mla` | cuda_mla | 125 | 16 | ✓ | none | 0.7250 | 0.9333 | 0.9204 | 0.8797 | 2928.4 | 8420886 | 2875.6 |
| `rope_aware_block_sparse_mla` | cuda_mla | 157 | 16 | ✓ | none | 0.7333 | 0.9333 | 0.9119 | 0.8784 | 2842.0 | 8437797 | 2968.9 |
| `rope_aware_block_sparse_mla` | cuda_mla | 253 | 16 | ✓ | none | 0.7438 | 0.9000 | 0.8912 | 0.8643 | 3884.0 | 8324371 | 2143.2 |

## Original summaries (raw)

<details><summary><code>zai-org_GLM-4.7-Flash_full_attention_16trials_tp1_2026-05-27_21-50-58.json</code></summary>

```json
{
  "mean@16": 0.7041666666666667,
  "pass@16": 0.9333333333333333,
  "pass@4": 0.8473992673992673,
  "pass@8": 0.8928800828800829,
  "total_example": 480,
  "e2e_time": 8755.758232831955,
  "total_tokens": 8585368,
  "throughput": 980.5396370821395,
  "args": {
    "trials": 16,
    "topk_val": 253,
    "topk_ratio": 0.0,
    "block_size": 32,
    "page_size": 32,
    "workload_chunk_size": 64,
    "generation_max_new_tokens": 32768,
    "max_input_length": 4096,
    "vortex_module_name": "full_attention",
    "model_name": "zai-org/GLM-4.7-Flash",
    "sparse_attention": false,
    "mem": 0.9,
    "data_path": "examples/aime26_glm.jsonl",
    "tp_size": 1,
    "kv_cache_dtype": "auto",
    "attention_backend": "triton",
    "vortex_attention_backend": "trtllm",
    "vortex_impl_backend": "cuda",
    "vortex_use_tensor_core": false,
    "vortex_layers_skip": [
      0
    ]
  }
}
```
</details>

<details><summary><code>zai-org_GLM-4.7-Flash_lserve_centroid_mla_16trials_tp1_2026-05-27_23-42-43.json</code></summary>

```json
{
  "mean@16": 0.7020833333333333,
  "pass@16": 0.9,
  "pass@4": 0.8486996336996336,
  "pass@8": 0.8811706811706812,
  "total_example": 480,
  "e2e_time": 3837.0725333690643,
  "total_tokens": 8457942,
  "throughput": 2204.269511833719,
  "args": {
    "trials": 16,
    "topk_val": 29,
    "topk_ratio": 0.0,
    "block_size": 64,
    "page_size": 64,
    "workload_chunk_size": 64,
    "generation_max_new_tokens": 32768,
    "max_input_length": 4096,
    "vortex_module_name": "lserve_centroid_mla",
    "model_name": "zai-org/GLM-4.7-Flash",
    "sparse_attention": true,
    "mem": 0.9,
    "data_path": "examples/aime26_glm.jsonl",
    "tp_size": 1,
    "kv_cache_dtype": "auto",
    "attention_backend": "cuda_mla",
    "vortex_attention_backend": "trtllm",
    "vortex_impl_backend": "triton",
    "vortex_use_tensor_core": true,
    "vortex_layers_skip": []
  }
}
```
</details>

<details><summary><code>zai-org_GLM-4.7-Flash_lserve_centroid_mla_16trials_tp1_2026-05-27_23-08-48.json</code></summary>

```json
{
  "mean@16": 0.7125,
  "pass@16": 0.9,
  "pass@4": 0.8416300366300367,
  "pass@8": 0.8781714581714581,
  "total_example": 480,
  "e2e_time": 1978.94144821167,
  "total_tokens": 8362906,
  "throughput": 4225.949184882348,
  "args": {
    "trials": 16,
    "topk_val": 61,
    "topk_ratio": 0.0,
    "block_size": 32,
    "page_size": 32,
    "workload_chunk_size": 64,
    "generation_max_new_tokens": 32768,
    "max_input_length": 4096,
    "vortex_module_name": "lserve_centroid_mla",
    "model_name": "zai-org/GLM-4.7-Flash",
    "sparse_attention": true,
    "mem": 0.9,
    "data_path": "examples/aime26_glm.jsonl",
    "tp_size": 1,
    "kv_cache_dtype": "auto",
    "attention_backend": "cuda_mla",
    "vortex_attention_backend": "trtllm",
    "vortex_impl_backend": "triton",
    "vortex_use_tensor_core": true,
    "vortex_layers_skip": []
  }
}
```
</details>

<details><summary><code>zai-org_GLM-4.7-Flash_lserve_centroid_mla_16trials_tp1_2026-05-27_23-42-27.json</code></summary>

```json
{
  "mean@16": 0.7416666666666667,
  "pass@16": 0.9333333333333333,
  "pass@4": 0.8484798534798534,
  "pass@8": 0.882051282051282,
  "total_example": 480,
  "e2e_time": 3916.970139980316,
  "total_tokens": 8211048,
  "throughput": 2096.2753624773004,
  "args": {
    "trials": 16,
    "topk_val": 61,
    "topk_ratio": 0.0,
    "block_size": 64,
    "page_size": 64,
    "workload_chunk_size": 64,
    "generation_max_new_tokens": 32768,
    "max_input_length": 4096,
    "vortex_module_name": "lserve_centroid_mla",
    "model_name": "zai-org/GLM-4.7-Flash",
    "sparse_attention": true,
    "mem": 0.9,
    "data_path": "examples/aime26_glm.jsonl",
    "tp_size": 1,
    "kv_cache_dtype": "auto",
    "attention_backend": "cuda_mla",
    "vortex_attention_backend": "trtllm",
    "vortex_impl_backend": "triton",
    "vortex_use_tensor_core": true,
    "vortex_layers_skip": []
  }
}
```
</details>

<details><summary><code>zai-org_GLM-4.7-Flash_lserve_centroid_mla_16trials_tp1_2026-05-27_21-07-15.json</code></summary>

```json
{
  "mean@16": 0.6770833333333334,
  "pass@16": 0.8666666666666667,
  "pass@4": 0.8191941391941392,
  "pass@8": 0.8582413882413883,
  "total_example": 480,
  "e2e_time": 1961.034833908081,
  "total_tokens": 8734383,
  "throughput": 4453.966267694255,
  "args": {
    "trials": 16,
    "topk_val": 61,
    "topk_ratio": 0.0,
    "block_size": 16,
    "page_size": 16,
    "workload_chunk_size": 64,
    "generation_max_new_tokens": 32768,
    "max_input_length": 4096,
    "vortex_module_name": "lserve_centroid_mla",
    "model_name": "zai-org/GLM-4.7-Flash",
    "sparse_attention": true,
    "mem": 0.9,
    "data_path": "examples/aime26_glm.jsonl",
    "tp_size": 1,
    "kv_cache_dtype": "auto",
    "attention_backend": "cuda_mla",
    "vortex_attention_backend": "trtllm",
    "vortex_impl_backend": "triton",
    "vortex_use_tensor_core": true,
    "vortex_layers_skip": []
  }
}
```
</details>

<details><summary><code>zai-org_GLM-4.7-Flash_lserve_centroid_mla_16trials_tp1_2026-05-27_21-08-29.json</code></summary>

```json
{
  "mean@16": 0.7041666666666667,
  "pass@16": 0.9,
  "pass@4": 0.8363003663003663,
  "pass@8": 0.8660968660968661,
  "total_example": 480,
  "e2e_time": 2037.1976444721222,
  "total_tokens": 8522305,
  "throughput": 4183.347169640134,
  "args": {
    "trials": 16,
    "topk_val": 93,
    "topk_ratio": 0.0,
    "block_size": 16,
    "page_size": 16,
    "workload_chunk_size": 64,
    "generation_max_new_tokens": 32768,
    "max_input_length": 4096,
    "vortex_module_name": "lserve_centroid_mla",
    "model_name": "zai-org/GLM-4.7-Flash",
    "sparse_attention": true,
    "mem": 0.9,
    "data_path": "examples/aime26_glm.jsonl",
    "tp_size": 1,
    "kv_cache_dtype": "auto",
    "attention_backend": "cuda_mla",
    "vortex_attention_backend": "trtllm",
    "vortex_impl_backend": "triton",
    "vortex_use_tensor_core": true,
    "vortex_layers_skip": []
  }
}
```
</details>

<details><summary><code>zai-org_GLM-4.7-Flash_lserve_centroid_mla_16trials_tp1_2026-05-27_23-15-15.json</code></summary>

```json
{
  "mean@16": 0.7354166666666667,
  "pass@16": 0.9,
  "pass@4": 0.8523626373626373,
  "pass@8": 0.8795493395493394,
  "total_example": 480,
  "e2e_time": 2365.326204776764,
  "total_tokens": 8581143,
  "throughput": 3627.8898794891065,
  "args": {
    "trials": 16,
    "topk_val": 125,
    "topk_ratio": 0.0,
    "block_size": 32,
    "page_size": 32,
    "workload_chunk_size": 64,
    "generation_max_new_tokens": 32768,
    "max_input_length": 4096,
    "vortex_module_name": "lserve_centroid_mla",
    "model_name": "zai-org/GLM-4.7-Flash",
    "sparse_attention": true,
    "mem": 0.9,
    "data_path": "examples/aime26_glm.jsonl",
    "tp_size": 1,
    "kv_cache_dtype": "auto",
    "attention_backend": "cuda_mla",
    "vortex_attention_backend": "trtllm",
    "vortex_impl_backend": "triton",
    "vortex_use_tensor_core": true,
    "vortex_layers_skip": []
  }
}
```
</details>

<details><summary><code>zai-org_GLM-4.7-Flash_lserve_centroid_mla_16trials_tp1_2026-05-27_21-45-11.json</code></summary>

```json
{
  "mean@16": 0.7125,
  "pass@16": 0.9,
  "pass@4": 0.8478205128205129,
  "pass@8": 0.8838539238539238,
  "total_example": 480,
  "e2e_time": 2022.85959649086,
  "total_tokens": 8569544,
  "throughput": 4236.351358673607,
  "args": {
    "trials": 16,
    "topk_val": 157,
    "topk_ratio": 0.0,
    "block_size": 16,
    "page_size": 16,
    "workload_chunk_size": 64,
    "generation_max_new_tokens": 32768,
    "max_input_length": 4096,
    "vortex_module_name": "lserve_centroid_mla",
    "model_name": "zai-org/GLM-4.7-Flash",
    "sparse_attention": true,
    "mem": 0.9,
    "data_path": "examples/aime26_glm.jsonl",
    "tp_size": 1,
    "kv_cache_dtype": "auto",
    "attention_backend": "cuda_mla",
    "vortex_attention_backend": "trtllm",
    "vortex_impl_backend": "triton",
    "vortex_use_tensor_core": true,
    "vortex_layers_skip": []
  }
}
```
</details>

<details><summary><code>zai-org_GLM-4.7-Flash_lserve_centroid_mla_16trials_tp1_2026-05-27_21-47-25.json</code></summary>

```json
{
  "mean@16": 0.7479166666666667,
  "pass@16": 0.9,
  "pass@4": 0.8636813186813187,
  "pass@8": 0.8888603988603988,
  "total_example": 480,
  "e2e_time": 2157.129903793335,
  "total_tokens": 8466111,
  "throughput": 3924.7107858976215,
  "args": {
    "trials": 16,
    "topk_val": 253,
    "topk_ratio": 0.0,
    "block_size": 16,
    "page_size": 16,
    "workload_chunk_size": 64,
    "generation_max_new_tokens": 32768,
    "max_input_length": 4096,
    "vortex_module_name": "lserve_centroid_mla",
    "model_name": "zai-org/GLM-4.7-Flash",
    "sparse_attention": true,
    "mem": 0.9,
    "data_path": "examples/aime26_glm.jsonl",
    "tp_size": 1,
    "kv_cache_dtype": "auto",
    "attention_backend": "cuda_mla",
    "vortex_attention_backend": "trtllm",
    "vortex_impl_backend": "triton",
    "vortex_use_tensor_core": true,
    "vortex_layers_skip": []
  }
}
```
</details>

<details><summary><code>zai-org_GLM-4.7-Flash_rope_aware_block_sparse_mla_16trials_tp1_2026-05-27_22-30-29.json</code></summary>

```json
{
  "mean@16": 0.7083333333333334,
  "pass@16": 0.9,
  "pass@4": 0.8443956043956043,
  "pass@8": 0.8752059052059052,
  "total_example": 480,
  "e2e_time": 1852.856770515442,
  "total_tokens": 8552721,
  "throughput": 4615.964458829021,
  "args": {
    "trials": 16,
    "topk_val": 29,
    "topk_ratio": 0.0,
    "block_size": 64,
    "page_size": 64,
    "workload_chunk_size": 64,
    "generation_max_new_tokens": 32768,
    "max_input_length": 4096,
    "vortex_module_name": "rope_aware_block_sparse_mla",
    "model_name": "zai-org/GLM-4.7-Flash",
    "sparse_attention": true,
    "mem": 0.9,
    "data_path": "examples/aime26_glm.jsonl",
    "tp_size": 1,
    "kv_cache_dtype": "auto",
    "attention_backend": "cuda_mla",
    "vortex_attention_backend": "trtllm",
    "vortex_impl_backend": "triton",
    "vortex_use_tensor_core": true,
    "vortex_layers_skip": []
  }
}
```
</details>

<details><summary><code>zai-org_GLM-4.7-Flash_rope_aware_block_sparse_mla_16trials_tp1_2026-05-27_20-12-09.json</code></summary>

```json
{
  "mean@16": 0.6479166666666667,
  "pass@16": 0.9,
  "pass@4": 0.817032967032967,
  "pass@8": 0.8753768453768453,
  "total_example": 480,
  "e2e_time": 2791.0943007469177,
  "total_tokens": 8712928,
  "throughput": 3121.6888650692867,
  "args": {
    "trials": 16,
    "topk_val": 61,
    "topk_ratio": 0.0,
    "block_size": 16,
    "page_size": 16,
    "workload_chunk_size": 64,
    "generation_max_new_tokens": 32768,
    "max_input_length": 4096,
    "vortex_module_name": "rope_aware_block_sparse_mla",
    "model_name": "zai-org/GLM-4.7-Flash",
    "sparse_attention": true,
    "mem": 0.9,
    "data_path": "examples/aime26_glm.jsonl",
    "tp_size": 1,
    "kv_cache_dtype": "auto",
    "attention_backend": "triton",
    "vortex_attention_backend": "trtllm",
    "vortex_impl_backend": "triton",
    "vortex_use_tensor_core": true,
    "vortex_layers_skip": []
  }
}
```
</details>

<details><summary><code>zai-org_GLM-4.7-Flash_rope_aware_block_sparse_mla_16trials_tp1_2026-05-27_20-31-31.json</code></summary>

```json
{
  "mean@16": 0.6645833333333333,
  "pass@16": 0.9,
  "pass@4": 0.8106227106227106,
  "pass@8": 0.8638539238539239,
  "total_example": 480,
  "e2e_time": 2857.8958954811096,
  "total_tokens": 8608008,
  "throughput": 3012.008944626338,
  "args": {
    "trials": 16,
    "topk_val": 61,
    "topk_ratio": 0.0,
    "block_size": 16,
    "page_size": 16,
    "workload_chunk_size": 64,
    "generation_max_new_tokens": 32768,
    "max_input_length": 4096,
    "vortex_module_name": "rope_aware_block_sparse_mla",
    "model_name": "zai-org/GLM-4.7-Flash",
    "sparse_attention": true,
    "mem": 0.9,
    "data_path": "examples/aime26_glm.jsonl",
    "tp_size": 1,
    "kv_cache_dtype": "auto",
    "attention_backend": "cuda_mla",
    "vortex_attention_backend": "trtllm",
    "vortex_impl_backend": "triton",
    "vortex_use_tensor_core": true,
    "vortex_layers_skip": []
  }
}
```
</details>

<details><summary><code>zai-org_GLM-4.7-Flash_rope_aware_block_sparse_mla_16trials_tp1_2026-05-27_22-33-58.json</code></summary>

```json
{
  "mean@16": 0.7270833333333333,
  "pass@16": 0.8666666666666667,
  "pass@4": 0.8273260073260074,
  "pass@8": 0.8486713286713287,
  "total_example": 480,
  "e2e_time": 2061.3013858795166,
  "total_tokens": 8479595,
  "throughput": 4113.709454661781,
  "args": {
    "trials": 16,
    "topk_val": 61,
    "topk_ratio": 0.0,
    "block_size": 64,
    "page_size": 64,
    "workload_chunk_size": 64,
    "generation_max_new_tokens": 32768,
    "max_input_length": 4096,
    "vortex_module_name": "rope_aware_block_sparse_mla",
    "model_name": "zai-org/GLM-4.7-Flash",
    "sparse_attention": true,
    "mem": 0.9,
    "data_path": "examples/aime26_glm.jsonl",
    "tp_size": 1,
    "kv_cache_dtype": "auto",
    "attention_backend": "cuda_mla",
    "vortex_attention_backend": "trtllm",
    "vortex_impl_backend": "triton",
    "vortex_use_tensor_core": true,
    "vortex_layers_skip": []
  }
}
```
</details>

<details><summary><code>zai-org_GLM-4.7-Flash_rope_aware_block_sparse_mla_16trials_tp1_2026-05-27_22-32-23.json</code></summary>

```json
{
  "mean@16": 0.70625,
  "pass@16": 0.9333333333333333,
  "pass@4": 0.8398717948717949,
  "pass@8": 0.8884123284123284,
  "total_example": 480,
  "e2e_time": 1967.5020496845245,
  "total_tokens": 8624164,
  "throughput": 4383.306234106758,
  "args": {
    "trials": 16,
    "topk_val": 61,
    "topk_ratio": 0.0,
    "block_size": 32,
    "page_size": 32,
    "workload_chunk_size": 64,
    "generation_max_new_tokens": 32768,
    "max_input_length": 4096,
    "vortex_module_name": "rope_aware_block_sparse_mla",
    "model_name": "zai-org/GLM-4.7-Flash",
    "sparse_attention": true,
    "mem": 0.9,
    "data_path": "examples/aime26_glm.jsonl",
    "tp_size": 1,
    "kv_cache_dtype": "auto",
    "attention_backend": "cuda_mla",
    "vortex_attention_backend": "trtllm",
    "vortex_impl_backend": "triton",
    "vortex_use_tensor_core": true,
    "vortex_layers_skip": []
  }
}
```
</details>

<details><summary><code>zai-org_GLM-4.7-Flash_rope_aware_block_sparse_mla_16trials_tp1_2026-05-27_18-45-09.json</code></summary>

```json
{
  "mean@16": 0.26666666666666666,
  "pass@16": 0.6666666666666666,
  "pass@4": 0.49179487179487175,
  "pass@8": 0.5876275576275576,
  "total_example": 480,
  "e2e_time": 1151.4940140247345,
  "total_tokens": 3430788,
  "throughput": 2979.4232173284277,
  "args": {
    "trials": 16,
    "topk_val": 93,
    "topk_ratio": 0.0,
    "block_size": 16,
    "page_size": 16,
    "workload_chunk_size": 64,
    "generation_max_new_tokens": 32768,
    "max_input_length": 4096,
    "vortex_module_name": "rope_aware_block_sparse_mla",
    "model_name": "zai-org/GLM-4.7-Flash",
    "sparse_attention": true,
    "mem": 0.9,
    "data_path": "examples/aime26_glm.jsonl",
    "tp_size": 1,
    "kv_cache_dtype": "auto",
    "attention_backend": "triton",
    "vortex_attention_backend": "trtllm",
    "vortex_impl_backend": "triton",
    "vortex_use_tensor_core": true,
    "vortex_layers_skip": []
  }
}
```
</details>

<details><summary><code>zai-org_GLM-4.7-Flash_rope_aware_block_sparse_mla_16trials_tp1_2026-05-27_20-30-51.json</code></summary>

```json
{
  "mean@16": 0.7145833333333333,
  "pass@16": 0.9,
  "pass@4": 0.8538461538461539,
  "pass@8": 0.8875602175602175,
  "total_example": 480,
  "e2e_time": 2819.3768174648285,
  "total_tokens": 8441414,
  "throughput": 2994.0708697429395,
  "args": {
    "trials": 16,
    "topk_val": 93,
    "topk_ratio": 0.0,
    "block_size": 16,
    "page_size": 16,
    "workload_chunk_size": 64,
    "generation_max_new_tokens": 32768,
    "max_input_length": 4096,
    "vortex_module_name": "rope_aware_block_sparse_mla",
    "model_name": "zai-org/GLM-4.7-Flash",
    "sparse_attention": true,
    "mem": 0.9,
    "data_path": "examples/aime26_glm.jsonl",
    "tp_size": 1,
    "kv_cache_dtype": "auto",
    "attention_backend": "cuda_mla",
    "vortex_attention_backend": "trtllm",
    "vortex_impl_backend": "triton",
    "vortex_use_tensor_core": true,
    "vortex_layers_skip": []
  }
}
```
</details>

<details><summary><code>zai-org_GLM-4.7-Flash_rope_aware_block_sparse_mla_16trials_tp1_2026-05-27_22-32-59.json</code></summary>

```json
{
  "mean@16": 0.7541666666666667,
  "pass@16": 0.9333333333333333,
  "pass@4": 0.8668315018315018,
  "pass@8": 0.8982905982905983,
  "total_example": 480,
  "e2e_time": 2003.9386732578278,
  "total_tokens": 8224331,
  "throughput": 4104.083178668139,
  "args": {
    "trials": 16,
    "topk_val": 125,
    "topk_ratio": 0.0,
    "block_size": 32,
    "page_size": 32,
    "workload_chunk_size": 64,
    "generation_max_new_tokens": 32768,
    "max_input_length": 4096,
    "vortex_module_name": "rope_aware_block_sparse_mla",
    "model_name": "zai-org/GLM-4.7-Flash",
    "sparse_attention": true,
    "mem": 0.9,
    "data_path": "examples/aime26_glm.jsonl",
    "tp_size": 1,
    "kv_cache_dtype": "auto",
    "attention_backend": "cuda_mla",
    "vortex_attention_backend": "trtllm",
    "vortex_impl_backend": "triton",
    "vortex_use_tensor_core": true,
    "vortex_layers_skip": []
  }
}
```
</details>

<details><summary><code>zai-org_GLM-4.7-Flash_rope_aware_block_sparse_mla_16trials_tp1_2026-05-27_18-45-50.json</code></summary>

```json
{
  "mean@16": 0.275,
  "pass@16": 0.6333333333333333,
  "pass@4": 0.48166666666666663,
  "pass@8": 0.5684615384615384,
  "total_example": 480,
  "e2e_time": 1197.7052600383759,
  "total_tokens": 2598418,
  "throughput": 2169.4970262690035,
  "args": {
    "trials": 16,
    "topk_val": 125,
    "topk_ratio": 0.0,
    "block_size": 16,
    "page_size": 16,
    "workload_chunk_size": 64,
    "generation_max_new_tokens": 32768,
    "max_input_length": 4096,
    "vortex_module_name": "rope_aware_block_sparse_mla",
    "model_name": "zai-org/GLM-4.7-Flash",
    "sparse_attention": true,
    "mem": 0.9,
    "data_path": "examples/aime26_glm.jsonl",
    "tp_size": 1,
    "kv_cache_dtype": "auto",
    "attention_backend": "triton",
    "vortex_attention_backend": "trtllm",
    "vortex_impl_backend": "triton",
    "vortex_use_tensor_core": true,
    "vortex_layers_skip": []
  }
}
```
</details>

<details><summary><code>zai-org_GLM-4.7-Flash_rope_aware_block_sparse_mla_16trials_tp1_2026-05-27_20-31-48.json</code></summary>

```json
{
  "mean@16": 0.725,
  "pass@16": 0.9333333333333333,
  "pass@4": 0.8797435897435898,
  "pass@8": 0.9204428904428904,
  "total_example": 480,
  "e2e_time": 2875.6124556064606,
  "total_tokens": 8420886,
  "throughput": 2928.379999044083,
  "args": {
    "trials": 16,
    "topk_val": 125,
    "topk_ratio": 0.0,
    "block_size": 16,
    "page_size": 16,
    "workload_chunk_size": 64,
    "generation_max_new_tokens": 32768,
    "max_input_length": 4096,
    "vortex_module_name": "rope_aware_block_sparse_mla",
    "model_name": "zai-org/GLM-4.7-Flash",
    "sparse_attention": true,
    "mem": 0.9,
    "data_path": "examples/aime26_glm.jsonl",
    "tp_size": 1,
    "kv_cache_dtype": "auto",
    "attention_backend": "cuda_mla",
    "vortex_attention_backend": "trtllm",
    "vortex_impl_backend": "triton",
    "vortex_use_tensor_core": true,
    "vortex_layers_skip": []
  }
}
```
</details>

<details><summary><code>zai-org_GLM-4.7-Flash_rope_aware_block_sparse_mla_16trials_tp1_2026-05-27_20-33-20.json</code></summary>

```json
{
  "mean@16": 0.7333333333333333,
  "pass@16": 0.9333333333333333,
  "pass@4": 0.8783699633699633,
  "pass@8": 0.9119114219114218,
  "total_example": 480,
  "e2e_time": 2968.9209332466125,
  "total_tokens": 8437797,
  "throughput": 2842.041667567412,
  "args": {
    "trials": 16,
    "topk_val": 157,
    "topk_ratio": 0.0,
    "block_size": 16,
    "page_size": 16,
    "workload_chunk_size": 64,
    "generation_max_new_tokens": 32768,
    "max_input_length": 4096,
    "vortex_module_name": "rope_aware_block_sparse_mla",
    "model_name": "zai-org/GLM-4.7-Flash",
    "sparse_attention": true,
    "mem": 0.9,
    "data_path": "examples/aime26_glm.jsonl",
    "tp_size": 1,
    "kv_cache_dtype": "auto",
    "attention_backend": "cuda_mla",
    "vortex_attention_backend": "trtllm",
    "vortex_impl_backend": "triton",
    "vortex_use_tensor_core": true,
    "vortex_layers_skip": []
  }
}
```
</details>

<details><summary><code>zai-org_GLM-4.7-Flash_rope_aware_block_sparse_mla_16trials_tp1_2026-05-27_21-10-14.json</code></summary>

```json
{
  "mean@16": 0.74375,
  "pass@16": 0.9,
  "pass@4": 0.8643223443223443,
  "pass@8": 0.8912483812483812,
  "total_example": 480,
  "e2e_time": 2143.240613937378,
  "total_tokens": 8324371,
  "throughput": 3884.0114105094244,
  "args": {
    "trials": 16,
    "topk_val": 253,
    "topk_ratio": 0.0,
    "block_size": 16,
    "page_size": 16,
    "workload_chunk_size": 64,
    "generation_max_new_tokens": 32768,
    "max_input_length": 4096,
    "vortex_module_name": "rope_aware_block_sparse_mla",
    "model_name": "zai-org/GLM-4.7-Flash",
    "sparse_attention": true,
    "mem": 0.9,
    "data_path": "examples/aime26_glm.jsonl",
    "tp_size": 1,
    "kv_cache_dtype": "auto",
    "attention_backend": "cuda_mla",
    "vortex_attention_backend": "trtllm",
    "vortex_impl_backend": "triton",
    "vortex_use_tensor_core": true,
    "vortex_layers_skip": []
  }
}
```
</details>

