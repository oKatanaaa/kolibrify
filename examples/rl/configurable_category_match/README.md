# Configurable CategoryMatchGrader Example

This example shows how to reuse the built-in `category_match` grader multiple times with different parameters using `init_kwargs` and the `builtin` alias. The datasets are **placeholders** (no data files are provided); the configs are meant for reference.

## Files
- `rl_data_config.yaml` – RL dataserver config wiring two differently-parameterized category graders.
- `rl_training_config.yaml` – Minimal training config that points to the dataserver.

## Usage (demo)
- Copy these configs and replace the dataset JSONL paths with your data.
- Start the dataserver:
  ```bash
  kolibrify-rl-dataserver rl_data_config.yaml --host 127.0.0.1 --port 9000
  ```
- Launch training (after you have real data at the referenced paths):
  ```bash
  kolibrify-rl-train rl_training_config.yaml
  ```

Notes:
- The dataserver always adds the built-in reasoning-format grader automatically; the configs below show only the additional graders you define.
- `init_kwargs` are forwarded to the grader constructor, letting you reuse `category_match` with different `allowed_categories`.
