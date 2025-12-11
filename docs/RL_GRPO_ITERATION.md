# GRPO Sampling Cadence and Dataserver Iterations

This note explains how TRL’s GRPO trainer samples prompts, why kolibrify’s RL dataserver iterations can lag when `num_generations > 1`, and how kolibrify maps indices to `/sample` iterations to keep curriculum stages aligned with optimizer steps.

## Actors

- **TRL GRPOTrainer**: trains using multiple generations per prompt.
- **RepeatSampler (TRL)**: sampler GRPO uses to repeat each unique prompt `num_generations` times.
- **RemoteRLDataset (kolibrify)**: map‑style dataset that calls the RL dataserver’s `/sample` endpoint.
- **Reward function (kolibrify)**: posts completions to `/grade` once per trainer step.

## What GRPO actually consumes per step

GRPO defines a *generation batch* per trainer (optimizer) step:

- `generation_batch_size = per_device_train_batch_size * steps_per_generation`
- GRPO requires `generation_batch_size % num_generations == 0`
- **Unique prompts per step**:
  - `prompts_per_step = generation_batch_size / num_generations`

`RepeatSampler` emits indices so that each unique prompt index is repeated `num_generations` times before moving on.

Example (`per_device_train_batch_size=64`, `steps_per_generation=1`, `num_generations=16`):

- `generation_batch_size = 64`
- `prompts_per_step = 64 / 16 = 4`
- Sampler indices per step:
  - Step 0: `0×16, 1×16, 2×16, 3×16`
  - Step 1: `4×16, 5×16, 6×16, 7×16`
  - …

So GRPO performs **one optimizer step per 4 unique prompts**, each with 16 generations.

## Why `/sample` iterations lagged before

Older kolibrify logic treated dataset indices as if they advanced linearly through a full per‑device batch:

- It computed:
  - `iteration = (idx // per_device_batch_size) // grad_accum`
- It cached `/sample` per `(iteration, accum_step)` and requested `batch_size = per_device_batch_size`.

With RepeatSampler, indices stay in the first `per_device_batch_size` range for many trainer steps because only `prompts_per_step` unique indices appear each step. Therefore `iteration` advanced about `num_generations` times slower than optimizer steps, keeping the dataserver stuck in early curriculum stages.

## Current kolibrify mapping (Fix A)

Kolibrify now defines “iteration” in terms of GRPO generation steps. The key
detail is that RepeatSampler advances indices in units of *unique prompts*:
the same index is repeated `num_generations` times, but the index value only
increments once per unique prompt.

1. Compute unique prompts per GRPO step:
   - If `num_generations == 1`: `prompts_per_step = per_device_batch_size`
   - Else:
     - If divisible: `prompts_per_step = per_device_batch_size // num_generations`
     - If not divisible: warn and fall back to `prompts_per_step = 1`
2. Interpret dataset indices in units of unique prompts:
   - `micro_batch = base_idx // prompts_per_step`
   - `generation_step = micro_batch // grad_accum`
   - `accum_step = micro_batch % grad_accum`
3. Select which unique prompt within the step:
   - `prompt_slot = base_idx % prompts_per_step`
4. Cache and sample per generation step:
   - `POST /sample {iteration: generation_step, batch_size: prompts_per_step}`

Effect: each GRPO optimizer step causes **exactly one** `/sample` call, and curriculum stages keyed to iteration boundaries align with training steps.

## Notes and tradeoffs

- This logic is per‑rank. Dataset length is still scaled by `WORLD_SIZE` to avoid DistributedSampler truncation, but indices are wrapped back into the base range so each rank progresses through iterations identically.
- Smaller `prompts_per_step` means more frequent, smaller `/sample` calls. If you want to keep full‑batch sampling per step, use an alternative implementation (Fix B in issue discussion).

## Where the code lives

- Dataset mapping and caching: `kolibrify/kolibrify/rl/data.py`
- Wiring of `num_generations`: `kolibrify/kolibrify/rl_train.py`
- Cadence tests: `kolibrify/tests/test_rl_dataset_grpo_iteration.py`
