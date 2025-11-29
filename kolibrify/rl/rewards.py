from collections.abc import Mapping
import time

import requests


def _completion_to_text(completion) -> str:
    """Best-effort extraction of text content from a completion object."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, Mapping) and "content" in completion:
        try:
            return str(completion.get("content", ""))
        except Exception:
            return ""
    try:
        return str(completion)
    except Exception:
        return ""


def build_remote_reward_fn(
    server_url: str,
    *,
    request_timeout_seconds: float = 60.0,
    max_retries: int = 3,
    retry_backoff_seconds: float = 1.0,
):
    session = requests.Session()
    iteration_counter = {"value": 0}

    def _with_retries(fn, op_name: str):
        nonlocal session
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                return fn()
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    delay = retry_backoff_seconds * (2**attempt)
                    print(f"{op_name} failed (attempt {attempt + 1}/{max_retries + 1}): {e}. Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                    session = requests.Session()
                else:
                    print(f"{op_name} failed after {max_retries + 1} attempts: {e}")
        if last_error:
            raise last_error

    def reward_fn(completions, **kwargs):
        # TRL passes dataset columns as keyword lists (e.g., sample_id=[...]).
        # Fall back to the legacy "batch" kwarg for compatibility.
        sample_ids = kwargs.get("sample_id") or []
        if not sample_ids:
            batch = kwargs.get("batch") or []
            sample_ids = [item.get("sample_id") for item in batch]

        # TRL/GRPO can pass multiple generations per prompt. Flatten any nested
        # completions and repeat sample_ids so every generated completion is
        # graded individually.
        completions_flat = []
        completion_sample_ids = []

        # Determine how many generations we expect so we can repeat sample_ids.
        num_generations = kwargs.get("num_generations")
        if num_generations is None and sample_ids and len(sample_ids) > 0:
            num_generations = len(completions) // len(sample_ids) if len(sample_ids) else 1
        num_generations = num_generations or 1

        # Flatten completions and align sample_ids.
        if any(isinstance(c, list) for c in completions):
            for idx, completion in enumerate(completions):
                sid = sample_ids[idx] if idx < len(sample_ids) else None
                if isinstance(completion, list):
                    for gen_completion in completion:
                        completions_flat.append(gen_completion)
                        completion_sample_ids.append(sid)
                else:
                    completions_flat.append(completion)
                    completion_sample_ids.append(sid)
        else:
            completions_flat = list(completions)
            if sample_ids and len(sample_ids) > 0:
                # Repeat each sample_id for its generations so grade() can map rewards.
                repeat = len(completions_flat) // len(sample_ids) or 1
                completion_sample_ids = [
                    sample_ids[idx // repeat] if idx // repeat < len(sample_ids) else None
                    for idx in range(len(completions_flat))
                ]
            else:
                completion_sample_ids = [None for _ in completions_flat]

        payload = {
            "iteration": iteration_counter["value"],
            "items": [
                {
                    "sample_id": sid,
                    "completion": _completion_to_text(completion),
                    "completion_index": idx,
                }
                for idx, (sid, completion) in enumerate(
                    zip(completion_sample_ids, completions_flat)
                )
            ],
        }

        rewards = [0.0 for _ in completions_flat]

        try:
            def _post_grade():
                response = session.post(
                    f"{server_url.rstrip('/')}/grade",
                    json=payload,
                    timeout=request_timeout_seconds,
                )
                response.raise_for_status()
                return response.json()

            data = _with_retries(_post_grade, f"Grading iteration {iteration_counter['value']}")
            reward_map = {}
            for item in data.get("results", []):
                if "completion_index" in item:
                    reward_map[item.get("completion_index")] = item.get("reward", 0.0)
                elif "sample_id" in item:
                    reward_map.setdefault(item.get("sample_id"), item.get("reward", 0.0))

            rewards = [
                float(reward_map.get(idx, reward_map.get(sid, 0.0)))
                for idx, sid in enumerate(completion_sample_ids)
            ]
        except Exception as e:
            print(f"Failed to grade iteration {iteration_counter['value']}: {e}")

        iteration_counter["value"] += 1
        return rewards

    return reward_fn
