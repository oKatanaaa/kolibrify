# Qwen2.5 3B GRPO (GSM8K, Russian reasoning) Example

This variant shows how to plug in a **local Python grader** via the dataserver config. It fine-tunes `unsloth/Qwen2.5-3B-Instruct` on GSM8K with GRPO, keeps the built-in `<think>...</think>` format reward and math graders, and adds a custom grader that rewards Russian-language reasoning.

## Files
- `prepare_data.py` – downloads GSM8K train split and sets a Russian system prompt.
- `rl_data_config.yaml` – dataserver config wiring the custom `RussianReasoningGrader` (loaded from `custom_graders.py` via `python_graders`).
- `custom_graders.py` – local grader implementation.
- `rl_training_config.yaml` – GRPO hyperparameters/output paths.

## Steps
1) Prep data (writes `data/gsm8k_train.jsonl`):
```bash
python prepare_data.py
```

2) Start the RL dataserver (new terminal/tab):
```bash
kolibrify-rl-dataserver rl_data_config.yaml --host 127.0.0.1 --port 9000
```

3) Run GRPO training:
```bash
kolibrify-rl-train rl_training_config.yaml
```

## Notes
- The dataserver always enforces the `<think>\n...\n</think>\nfinal response` format reward; `RussianReasoningGrader` boosts samples whose reasoning section is mostly Cyrillic.
- Grader weights in `rl_data_config.yaml` combine math correctness, number-only output, and the new Russian reasoning bonus (all normalized alongside the built-in format grader).
- Outputs save under `experiments_rl/<config-name>/` (per Kolibrify convention). If you change ports/hosts, update `data.server_url` in `rl_training_config.yaml`.
