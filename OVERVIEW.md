# Kolibrify Repository Overview (for AI agents)

This file is a high‑signal map of the repo: what lives where, how the main training flows work, and which classes/functions matter. It’s meant to let a fresh agent become productive quickly.

## Top‑level layout

At repo root (`/workdir/kolibrify`):

- `kolibrify/` – the Python package.
- `docs/` – user docs and deeper design notes.
- `examples/` – runnable configs + minimal datasets.
- `tests/` – pytest suite (unit/integration‑style).
- `*_config_template.yaml` – config templates for SFT/DPO/RL.
- `rl_config_template.yaml` – RL trainer config template.
- `README.md` – user‑facing overview and CLI usage.

## Package layout (`kolibrify/`)

### CLI entrypoints (scripts)

These are the main “bins” referenced in `pyproject.toml`:

- **SFT**
  - `kolibrify/sft_run.py` – CLI `kolibrify-sft`; builds model/dataset, creates `trl.SFTTrainer`, runs training.
  - `kolibrify/sft/` – SFT dataset loading + collator.
- **DPO**
  - `kolibrify/dpo_run.py` – CLI `kolibrify-dpo`; uses `trl.DPOTrainer` patched by Unsloth.
  - `kolibrify/dpo/` – DPO dataset loading/config.
- **RL (GRPO/GSPO)**
  - `kolibrify/rl_train.py` – CLI `kolibrify-rl-train`; launches TRL `GRPOTrainer`, wires remote dataset + reward fn.
  - `kolibrify/rl/` – RL trainer‑side pieces (dataset, reward fn, RL config).
  - `kolibrify/rl_dataserver/` – RL dataserver process (FastAPI) for sampling + grading prompts.

Other utilities:

- `kolibrify/merge_lora.py` – CLI `kolibrify-merge`, merges LoRA into base weights.
- `kolibrify/predict.py` – CLI `kolibrify-predict`, batched inference.
- `kolibrify/push_to_hub.py` – CLI `kolibrify-push`.
- `kolibrify/chat.py` – CLI `kolibrify-chat`.
- `kolibrify/eval/` + `kolibrify/inference/` – evaluation/inference helpers.

### Shared training utilities

- `kolibrify/common_training.py`
  - `setup_seeds()` – deterministic seeds.
  - `ensure_c_compiler()` – makes Triton happy in conda images.
  - `common_training_setup(...)` – used by SFT/DPO: loads model via `get_model`, loads curriculum datasets, applies LoRA, optional CPU offload.
  - `apply_lora_adapter(...)` – thin wrapper around `unsloth.FastLanguageModel.get_peft_model`.
  - `run_training(...)` – `trainer.train(...)` + save adapter/tokenizer.
- `kolibrify/runtime_env.py`
  - `prepare_unsloth_environment()` – environment tweaks before importing Unsloth/TRL.

### Core model/data abstractions

Re‑exported in `kolibrify/core/__init__.py`:

- `kolibrify/core/model_utils.py`
  - `get_model(...)` – central model loader; handles Unsloth fast loading, token additions, max seq length, quantization, etc.
  - `free_mem()` – CUDA memory cleanup.
  - `cpu_offload_embeddings(...)` – optional embedding offload.
- `kolibrify/core/data_utils.py`
  - `load_jsonl(...)` – local JSONL loader.
  - `SimpleDataGen` / `CurriculumDataGen` – curriculum iterator building per‑stage mixes.
  - `ChatMLFormatter` – converts dataset records to ChatML prompts (shared across SFT/DPO/Predict).
- `kolibrify/core/config.py`
  - `BaseConfig` + `load_base_config(...)` – SFT/DPO config parsing with stage curriculum.
  - `save_config(...)` – persists sanitized config to output dir.

## Curriculum / stages concept

Curriculum is expressed as ordered **stages** with `until_step` boundaries:

- In SFT/DPO trainer configs: stages are used to build a mixed dataset iterator in `core/data_utils.py` and `sft/load_dataset` or `dpo/load_dataset`.
- In RL dataserver config: stages are used to choose which dataset IDs are sampled for a given `iteration` (see `rl_dataserver/server.py:_find_stage`).

Key invariant: *“iteration” is the clock that moves stages forward.* Ensuring trainer‑side iteration matches optimizer steps is crucial.

## RL system in detail

### Two processes

1. **Dataserver** (`kolibrify-rl-dataserver`)
   - Code: `kolibrify/rl_dataserver/server.py`, `kolibrify/rl_dataserver/cli.py`.
   - FastAPI endpoints:
     - `GET /meta` – total iterations + stage list.
     - `POST /sample {iteration, batch_size}` – returns prompt batch for that iteration.
     - `POST /grade {iteration, items}` – grades completions with configured graders.
   - Verbose logs:
     - `/sample` request params printed in `create_app(...).sample` when `--verbose`.
     - `/grade` reward summary printed similarly.

2. **Trainer** (`kolibrify-rl-train`)
   - Code: `kolibrify/rl_train.py`.
   - Builds:
     - `RemoteRLDataset` from `kolibrify/rl/data.py` (map‑style because GRPOTrainer disallows IterableDataset).
     - reward function from `kolibrify/rl/rewards.py` that posts to `/grade`.
   - Uses TRL `GRPOTrainer` + `GRPOConfig` (or GSPO via `importance_sampling_level`).

### RemoteRLDataset iteration mapping (important)

File: `kolibrify/rl/data.py`

- The dataset must translate TRL sampler indices into dataserver iterations.
- GRPO uses TRL `RepeatSampler`: each **unique prompt index** is repeated `num_generations` times per step.
- To keep curriculum aligned with optimizer steps, kolibrify defines:
  - `prompts_per_step = per_device_batch_size // num_generations` (fallback to 1 if not divisible).
  - `generation_step = (base_idx // prompts_per_step) // grad_accum`
  - `prompt_slot = base_idx % prompts_per_step`
  - `/sample(iteration=generation_step, batch_size=prompts_per_step)` is cached per `(generation_step, accum_step)`.

See `docs/RL_GRPO_ITERATION.md` for the full rationale and examples.

### Reward function iteration

File: `kolibrify/rl/rewards.py`

- `build_remote_reward_fn(...)` returns a callable for TRL.
- It flattens nested generations, aligns `sample_id`s, and posts to `/grade`.
- It maintains an internal `iteration_counter` that increments **once per reward_fn call** (i.e., per trainer step).

### Graders

Docs: `docs/GRADERS.md`

- Built‑ins in `kolibrify/rl_dataserver/graders.py` and `builtin_graders.py`.
- User‑configurable via dataserver YAML:
  - `python_graders` for local classes.
  - `external_graders` for HTTP graders.
- `DatasetReward` combines grader outputs with weights and built‑in reasoning format checking.

## Testing

- All tests are pytest in `tests/`.
- RL‑specific:
  - `tests/test_rl_stage_sampling.py` – stage boundary behavior.
  - `tests/test_rl_dataset_grpo_iteration.py` – verifies GRPO cadence mapping.
- Most tests monkeypatch network calls rather than spinning servers.

Run locally:

- full suite: `pytest -q`
- RL cadence only: `pytest -q tests/test_rl_dataset_grpo_iteration.py`

## Common modification hotspots

- **Curriculum/stage logic**
  - SFT/DPO mixing: `kolibrify/core/data_utils.py` + `sft/` or `dpo/` loaders.
  - RL stage selection: `kolibrify/rl_dataserver/server.py:_find_stage`.
- **Model loading / tokens**
  - `kolibrify/core/model_utils.py:get_model`.
- **RL sampling/iteration**
  - `kolibrify/rl/data.py:RemoteRLDataset`.
- **Reward shaping / parsing**
  - `kolibrify/rl/rewards.py` and dataserver graders.

## Constraints / non‑goals

- Datasets are local JSONL, not HF hub datasets.
- Unsloth and TRL are treated as external deps; kolibrify should adapt around their APIs rather than patching them (except for Unsloth’s provided trainer patches).

If you’re an agent starting work, begin by reading `README.md`, then the relevant doc in `docs/`, then the specific module listed above for your task.

