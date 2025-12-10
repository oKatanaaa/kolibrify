# Kolibrify Graders

Kolibrify ships several built-in graders and lets you add your own (Python modules or HTTP services). Graders are combined per-dataset with weights; the reasoning-format grader is always added automatically. You can also add multiplicative graders that gate the final reward (see the section at the end).

## Built-in Graders (quick lookup)
- `math_exact` – accept if the final response contains the gold number (exact string match).
- `number_only` – reward short numeric outputs; penalize extra characters around the first number.
- `json_valid` – reward parseable JSON (optionally require keys).
- `json_schema` – validate object shape against a minimal schema.
- `xml_schema` – check XML well-formedness and root tag.
- `category_match` – graded credit for matching categories (parameterized with allowed categories).
- `completion_length_cap` – gates reward to zero when completion tokens exceed a cap (parameterized; best used as a multiplicative guard).
- `reasoning_format` – auto-attached; encourages `<think>...</think>` + final answer.

## Basic Config: Using Built-ins
Built-ins (except `category_match`, which needs parameters) are ready to use by name inside `datasets.<id>.graders`. No `python_graders` entry is needed for these simple cases.

```yaml
paths:
  data_root: "data"

datasets:
  gsm8k:
    path: "gsm8k_train.jsonl"
    graders: [math_exact, number_only]   # built-ins by name
    grader_weights: [2.0, 1.0]

stages:
  - name: main
    until_step: 100
    datasets:
      - id: gsm8k
        weight: 1.0

external_graders: {}
python_graders: {}
```

The server will also attach the reasoning-format grader automatically and normalize weights.

## Detailed Built-in Behavior

### Reasoning Format (auto)
- Added automatically; not listed in config.
- Rewards Qwen-style `<think>...</think>` reasoning followed by a final response.
- Scoring (0–1): bonuses for seeing `<think>`/`</think>` and correct ordering; penalty for reversed tags. Expects non-empty text after `</think>` (+0.5, else -0.25). Penalizes glued divider/space (-0.1). Penalizes extra reasoning blocks (-0.15 each). Clamped to [0, 1].
- Use when you want consistent chain-of-thought formatting.

### Math Exact (`math_exact`)
- Purpose: correctness for numeric QA (e.g., GSM8K).
- Looks at the final response; extracts numbers and compares to the first number in `record["answer"]`. If any number matches exactly, reward 1.0, else 0.0. Empty final responses → 0.0.

### Number Only (`number_only`)
- Purpose: force terse numeric outputs.
- Finds the first number in the final response. Counts surrounding characters. Reward tiers: 1.0 (no extras), 0.5 (<10 extra chars), 0.4 (<20), 0.3 (<30), 0.2 (<40), 0.1 (<50), else 0.0. Missing/unnumbered responses → 0.0.

### Completion Length Cap (`completion_length_cap`) *(parameterized)*
- Purpose: hard cap on completion length; typically used multiplicatively to zero out overlong generations.
- Parameters: `max_completion_tokens` (required, >0), `treat_missing_as_fail` (default true).
- Behavior: uses `completion_tokens` supplied by the training loop. Reward 1.0 if token count ≤ cap; 0.0 if over. Missing token counts reward 0.0 unless `treat_missing_as_fail=False`.
- Config: declare under `python_graders` with `builtin: completion_length_cap`; see the parametrized graders section for a config example.

### JSON Valid (`json_valid`)
- Purpose: ensure responses are valid JSON; optionally require keys.
- Tries `json.loads` on the completion. If parse fails → 0.0. If parse succeeds → 1.0, unless `record["metadata"]["expected_json_schema"]["required"]` is present; if so, all those keys must exist or reward becomes 0.0.

### JSON Schema (`json_schema`)
- Purpose: validate object shape from `record["metadata"]["schema"]` (JSON string).
- Parsing: `json.loads`; if only via fenced block stripping or `ast.literal_eval`, multiply final score by 0.9. Unparsable → 0.0.
- Type: any valid JSON → at least 0.5; non-object returns 0.5 directly.
- Fields: `required` list; `allow_additional_properties` (default true).
- Scoring: missing required + extra keys → 0.7; missing required → 0.8; extra keys when disallowed → 0.9; perfect match → 1.0 (then apply any 0.9 penalty).

### XML Schema (`xml_schema`)
- Purpose: simple XML wrapper check using `record["metadata"]["schema"]` with `{"root_tag": "answer"}`.
- Parsing: XML parse; if only after stripping a fenced block, multiply by 0.9. Unparsable → 0.0.
- Well-formed XML → 0.5. Wrong root tag → 0.8. Correct root → 1.0 (then apply any 0.9 penalty).

## Parametrized Graders (Category Match + Length Cap)
Two built-ins require `init_kwargs` and therefore live under `python_graders`: `category_match` and `completion_length_cap`.

### Category Match (`category_match`)
- Config: use `builtin: category_match` with an `allowed_categories` list.
- Use when you want graded credit for categorical labels across multiple datasets with different label sets.

Example: two differently-parameterized category graders reused across datasets.
```yaml
python_graders:
  task_type_match:
    builtin: category_match
    init_kwargs:
      allowed_categories: ["Truthfulness", "Summarization", "Math"]

  response_format_match:
    builtin: category_match
    init_kwargs:
      allowed_categories:
        - Choice from provided options
        - Free-form response
        - Structured response
        - Numeric
        - Hybrid
        - Extractive answers
        - Code/program/logical form generation
        - Multiple formats

datasets:
  task_types:
    path: "task_types.jsonl"
    graders: [task_type_match]
    grader_weights: [1.0]
  response_formats:
    path: "response_formats.jsonl"
    graders: [response_format_match]
    grader_weights: [1.0]
```

Category Match reward tiers (expects `record["answer"]` or `record["expected_category"]`):
- 1.0 exact match (case-sensitive)
- 0.8 case-insensitive exact match
- 0.5 expected category contained in response (case-insensitive) with extra text
- 0.3 wrong category but is one of `allowed_categories`
- 0.0 otherwise or missing expected/response

### Completion Length Cap (`completion_length_cap`)
- Config: use `builtin: completion_length_cap` and set `max_completion_tokens` (required, >0). Optional `treat_missing_as_fail` (default true) controls what happens if a token count isn’t provided.
- Use case: hard-stop completions that exceed a token budget. Works best as a multiplicative grader (see the section at the end).

Example: gate GSM8K rewards if completions exceed 200 tokens.
```yaml
python_graders:
  length_cap_guard:
    builtin: completion_length_cap
    init_kwargs:
      max_completion_tokens: 200

datasets:
  gsm8k:
    path: "gsm8k_train.jsonl"
    graders: [math_exact, number_only]
    grader_weights: [2.0, 1.0]
    multiplicative_graders: [length_cap_guard]   # multiply the main reward by this guard
```

Completion Length Cap reward:
- 1.0 if `completion_tokens` ≤ `max_completion_tokens`
- 0.0 if above the cap
- Missing `completion_tokens` → 0.0 by default, or 1.0 if `treat_missing_as_fail=False`
The training loop sends `completion_tokens` to the dataserver; align the cap with your generation limits (e.g., `training.max_completion_length`).

## Custom Graders (Python)
- Declare under `python_graders` with either `import: "module:Attr"` or `path: "path/to/file.py:Attr"`. The target must be or return a `Grader`.
- Pass constructor params via `init_kwargs`.
- Refer to the grader by the key you choose under `python_graders` inside each dataset’s `graders` list.

Example:
```yaml
python_graders:
  russian_reasoning:
    path: "./custom_graders.py:RussianReasoningGrader"
    init_kwargs:
      min_letters: 30

datasets:
  gsm8k_ru:
    path: "gsm8k_ru.jsonl"
    graders: [math_exact, russian_reasoning]
    grader_weights: [1.0, 1.0]
```

## External HTTP Graders
- Declare under `external_graders` with `type: remote_http`, `url`, and `timeout_s`.
- Reference them by name in `datasets.<id>.graders`.
- Payload sent: prompt/system_prompt/answer/metadata, plus the completion, parsed reasoning, final response, and optional `completion_index`.
- The service must return `results: [{sample_id, reward, completion_index?}]`; rewards are used as-is.

## Multiplicative Graders (end-to-end reward gates)
- In each dataset block you can add `multiplicative_graders: []`. The dataserver first computes the weighted sum of `graders` (plus the auto-added `reasoning_format` grader), then multiplies that score by the product of all multiplicative grader rewards.
- You can point to any grader type here (built-in, python, or HTTP), but multiplicative is best for hard gates (e.g., 0/1). The recommended built-in is `completion_length_cap`, which zeros rewards if completions exceed a token cap.
- Example:
```yaml
datasets:
  gsm8k:
    path: "gsm8k_train.jsonl"
    graders: [math_exact, number_only]
    grader_weights: [2.0, 1.0]
    multiplicative_graders: [length_cap_guard]

python_graders:
  length_cap_guard:
    builtin: completion_length_cap
    init_kwargs:
      max_completion_tokens: 200
```
- The training loop now sends `completion_tokens` with each completion so `completion_length_cap` can enforce the cap. Keep the cap aligned with your `training.max_completion_length` or stricter if you want to encourage shorter reasoning.
