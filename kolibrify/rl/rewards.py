import requests


def build_remote_reward_fn(server_url: str):
    session = requests.Session()
    iteration_counter = {"value": 0}

    def reward_fn(completions, **kwargs):
        batch = kwargs.get("batch") or []
        sample_ids = [item.get("sample_id") for item in batch]

        # TRL/GRPO can pass multiple generations per prompt. Flatten any nested
        # completions and repeat sample_ids so every generated completion is
        # graded individually.
        completions_flat = []
        completion_sample_ids = []

        # Determine how many generations we expect so we can repeat sample_ids
        # when completions are already flattened.
        num_generations = kwargs.get("num_generations")
        if num_generations is None and sample_ids:
            if len(completions) % len(sample_ids) == 0:
                num_generations = len(completions) // len(sample_ids)
            else:
                num_generations = 1
        num_generations = num_generations or 1

        for idx, completion in enumerate(completions):
            if isinstance(completion, list):
                sid = sample_ids[idx] if idx < len(sample_ids) else None
                for gen_completion in completion:
                    completions_flat.append(gen_completion)
                    completion_sample_ids.append(sid)
            else:
                sid_idx = idx // num_generations if sample_ids else None
                sid = sample_ids[sid_idx] if sid_idx is not None and sid_idx < len(sample_ids) else None
                completions_flat.append(completion)
                completion_sample_ids.append(sid)

        payload = {
            "iteration": iteration_counter["value"],
            "items": [
                {
                    "sample_id": sid,
                    "completion": completion,
                    "completion_index": idx,
                }
                for idx, (sid, completion) in enumerate(
                    zip(completion_sample_ids, completions_flat)
                )
            ],
        }

        rewards = [0.0 for _ in completions_flat]

        try:
            response = session.post(
                f"{server_url.rstrip('/')}/grade",
                json=payload,
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()
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
