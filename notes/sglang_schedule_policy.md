# What `srt/managers/schedule_policy.py` does

This file contains the **two helper objects the scheduler uses to decide
*which* waiting requests to run, and *how many* of them fit**. The
scheduler itself (`scheduler.py`) only orchestrates; the policy file
owns the actual ordering and the admission predicate.

It exports three things.

---

## 1. The policy enums + `SchedulePolicy` class (the **ordering** layer)

Two families of policies:

- `CacheAwarePolicy` (lines 61–65): `LPM` (longest-prefix-match) and
  `DFS_WEIGHT` — both consult the radix prefix cache.
- `CacheAgnosticPolicy` (lines 68–73): `FCFS`, `LOF`
  (longest-output-first), `RANDOM`.

`SchedulePolicy.calc_priority(waiting_queue)` (line 97) reorders the
waiting queue **in place**:

- FCFS: short-circuit, no sort (line 98–100).
- `_determine_active_policy` (line 130): if LPM is on but the queue has
  > 128 reqs, falls back to FCFS — prefix matching at scale is too
  expensive.
- For cache-aware policies it calls `_compute_prefix_matches`
  (line 154), which:
  - Calls `tree_cache.match_prefix(rid, prefix_ids)` for each request to
    populate `r.prefix_indices`, `r.last_node`, `r.last_host_node`,
    `r.host_hit_length` (this is what `PrefillAdder` later locks).
  - Runs **in-batch prefix-cache deduplication**: keeps a temporary
    `waiting_queue_radix_tree` (line 90) and, if multiple waiting
    requests share a prefix not already cached, demotes the duplicates
    (`temporary_deprioritized`, line 161) so only one runs first and
    warms the cache.
- Then sorts by:
  - `_sort_by_longest_prefix` (line 197): biggest prefix-cache hit first.
  - `_sort_by_dfs_weight` (line 210): groups requests by shared
    `last_node` and traverses the radix tree depth-first, weighting
    subtrees by the number of waiting requests they contain — pulls
    together requests that will share KV.
  - `_sort_by_longest_output` / `_sort_randomly` for the cache-agnostic
    variants.

`_validate_and_adjust_policy` (line 136) silently downgrades any
cache-aware policy to FCFS if the tree cache is disabled.

---

## 2. `AddReqResult` enum (lines 264–267)

Three-state result the scheduler reads after each `add_one_req` call:

- `CONTINUE` — admission succeeded, keep draining the queue.
- `NO_TOKEN` — KV-pool budget exhausted, stop.
- `OTHER`    — prefill / chunk budget exhausted, stop.

---

## 3. `PrefillAdder` class (the **admission** layer)

Built fresh by the scheduler at the top of every
`get_new_batch_prefill()`. Holds the running budgets for one
prefill-batch assembly:

- `rem_total_tokens` (line 314):
  `allocator.available_size() + tree_cache.evictable_size() − reservation` —
  note that **evictable cache counts as free**, which is how admission
  implicitly triggers eviction.
- `rem_input_tokens`, `rem_chunk_tokens`: per-pass and per-chunk prefill
  caps.
- `rem_total_token_offset` (line 302): pre-charges expected decode
  growth `max_new_tokens * new_token_ratio` for every already-running
  request, so admission doesn't oversubscribe future decode steps.

### Key methods

- **`add_one_req(req, has_chunked_req)`** (line 462) — the main
  admission predicate. Checks `total_tokens >= rem_total_tokens`
  (NO_TOKEN), `real_input_tokens >= rem_input_tokens` (OTHER), then
  under `_lock_node` (line 377) re-checks budget, optionally
  `init_load_back`s host-tier prefix bytes, and either appends the
  request whole or as a chunked prefill (`new_chunked_req`, line 522).
  On success it calls `tree_cache.inc_lock_ref(req.last_node)` to pin
  the prefix.
- **`add_one_req_ignore_eos`** (line 385) — special path for
  `ignore_eos=True` requests; uses `req_states` to simulate forward
  decode and refuses admission if the projected occupancy would starve
  the batch
  (`min_free_tokens <= IGNORE_EOS_RESERVE_TOKENS * bs`, line 432).
- **`add_chunked_req`** (line 359) — continues a prefill chunk left
  over from the previous iteration (the `new_chunked_req` baton).
- **`_lock_node`** (line 377) — context manager that increments the
  tree-cache lock-ref while the admission decision is being made, so a
  concurrent eviction cannot pull the matched prefix out from under the
  request.
- **`budget_state`** (line 333) — collapses the three live counters
  into the `AddReqResult` returned to the scheduler loop.

---

## Configurable thresholds (env-tunable)

- `CLIP_MAX_NEW_TOKENS_ESTIMATION` (line 38, default 4096) — clips the
  worst-case decode estimate so a request with `max_new_tokens=1M`
  doesn't single-handedly close admission.
- `IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD` (line 46, default 32) —
  only run the in-batch dedup check for requests whose existing prefix
  hit is below this.
- `IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD` (line 53,
  default 32) — when the in-batch tree shows a shared prefix at least
  this long, demote the duplicate.
- `IGNORE_EOS_RESERVE_TOKENS` (line 58, hard-coded 1) — per-request
  slack reserved in the ignore-EOS admission projection.

---

## One-line summary

`schedule_policy.py` answers two questions for the scheduler:
**(a) in what order should I look at the waiting queue?**
(`SchedulePolicy`) and **(b) does this specific request fit, and at
what cost?** (`PrefillAdder` returning `AddReqResult`). The scheduler
in `scheduler.py` does the loop; this file does the decisions.
