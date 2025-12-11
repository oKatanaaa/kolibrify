# Reinforcement Learning in Kolibrify

Kolibrify’s RL pipeline is intentionally split into two cooperating pieces: a lightweight RL dataserver that owns data sampling and reward grading, and the training process that runs GRPO/GSPO with Unsloth models.

## Architecture
- **RL dataserver** (`kolibrify-rl-dataserver`): FastAPI service that samples training prompts and grades completions. It loads datasets from local JSONL files, applies graders, and returns scalar rewards. Curriculum is expressed as stages in the dataserver config; each stage mixes datasets with weights and has an `until_step` boundary. Run with `--verbose` to log `/sample` request params and per-iteration `/grade` summaries.
- **RL trainer** (`kolibrify-rl-train`): Runs GRPO/GSPO using TRL. It streams prompts from the dataserver (per-device batches), generates completions, and posts them back for grading. All model/hyperparameter choices live in the training YAML.

## Built-in Response Format
- Kolibrify enforces a Qwen3-style reasoning format for *all* RL runs via a built-in grader on the dataserver:
  ```
  <think>
  reasoning
  </think>
  final response
  ```
  Perfect match rewards 1.0; partial credit is given for tags/order/final response presence. No config flag is needed—the grader is automatically attached to every dataset.
- Completions are parsed before grading; `GraderInput` carries `completion`, parsed `reasoning`, and `answer` (final response). Parse failures set `reasoning`/`answer` to `None` but grading still proceeds.

## Dataserver Config (rl_data_config.yaml)
- `paths.data_root`: base directory for datasets.
- `datasets`: map of dataset id -> `{path, graders, grader_weights}`. Paths are relative to `data_root`. Graders listed here are **added on top of** the built-in format grader.
- `stages`: ordered list defining curriculum. Each stage has `until_step` and a weighted mix of dataset ids; sampling switches stages as iterations pass the boundaries.
- `external_graders`: optional remote HTTP graders (see `ExternalHttpGrader`).
- `python_graders`: optional local Python graders loaded from modules or file paths (`module:Attr` or `path/to/grader.py:Attr`).

## Graders
- Built-in: `reasoning_format` (auto), `math_exact`, `json_valid`, `json_schema`, `xml_schema`, `category_match` (instantiate via `python_graders` with `builtin: category_match`). See `docs/GRADERS.md` for full grader behaviors and config examples.
- `json_schema`: expects `record["metadata"]["schema"]` to be a JSON-serialized object schema with `required` keys and optional `allow_additional_properties`. Valid JSON gets at least 0.5; correct schema is 1.0; extra keys 0.9; missing required 0.8; both 0.7. If JSON only parses after stripping markdown fences or via `ast.literal_eval`, the final score is multiplied by 0.9.
- `xml_schema`: expects `record["metadata"]["schema"]` to be a JSON string containing `{"root_tag": "answer"}`. Well-formed XML earns 0.5, wrong root tag 0.8, correct root 1.0; parsing via fenced-block stripping applies a 0.9 multiplier.
- Python/local: point to importable classes or module paths via `python_graders` in the dataserver config; the referenced attribute must implement `grade_batch` and typically subclasses `Grader`. You can pass `init_kwargs` to parameterize a class constructor, and you can reuse built-ins via `builtin: <alias>` (e.g., `category_match`).
- External: register remote HTTP graders in the dataserver config; payload includes prompt/system_prompt/answer/metadata plus the parsed `reasoning` and `final_response`.
- Grader weights are normalized (including the built-in one).

## Training Config (rl_training_config.yaml)
- Standard model knobs (base model, LoRA/4bit, sequence length, gradient checkpointing).
- GRPO/GSPO knobs (max_steps, batch, num_generations, learning rate/schedule, prompt/completion lengths). If `rl.rl_algorithm` is `gspo`, importance sampling is enabled automatically.
- `data.server_url` points to the dataserver (must be running).
- Outputs are saved under `paths.output_dir/<config-name>/`.

## Run Flow
1) Start dataserver with an RL data config.
2) Launch `kolibrify-rl-train` with an RL training config.
3) (Optional) Merge/save adapters after training using existing Kolibrify utilities.
