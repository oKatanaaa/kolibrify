# Qwen2.5 3B GRPO (GSM8K) with Length Cap

This example mirrors the upstream `Qwen2.5_(3B)-GRPO` notebook using Kolibrify’s RL pipeline. It fine-tunes `unsloth/Qwen2.5-3B-Instruct` on GSM8K with GRPO, uses the built-in Qwen-style `<think>...</think>` reasoning-format reward, a math-exact grader for correctness, and a multiplicative `completion_length_cap` guard that zeroes rewards if completions exceed 200 tokens.

## Files
- `prepare_data.py` – downloads and prepares GSM8K train split into JSONL.
- `rl_data_config.yaml` – dataserver config (adds math + number graders and the multiplicative length cap; reasoning-format reward is auto-added by the server).
- `rl_training_config.yaml` – GRPO training config approximating the notebook hyperparams.

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
- The dataserver always enforces the `<think>\n...\n</think>\nfinal response` format reward; the math-exact + number graders score answers, then the `completion_length_cap` multiplicatively gates the reward if the completion is longer than 200 tokens.
- Hyperparameters match the notebook where supported (max_steps=250, num_generations=8, lr=5e-6, cosine schedule, LoRA r=64, seq len 1024, prompt len 256, completion len 200).
- Outputs save under `experiments_rl/<config-name>/` (per Kolibrify convention). If you change ports/hosts, update `data.server_url` in `rl_training_config.yaml`.
