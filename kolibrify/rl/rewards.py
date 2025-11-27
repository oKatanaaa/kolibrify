import requests


def build_remote_reward_fn(server_url: str):
    session = requests.Session()
    iteration_counter = {"value": 0}

    def reward_fn(completions, **kwargs):
        batch = kwargs.get("batch") or []
        sample_ids = [item.get("sample_id") for item in batch]

        payload = {
            "iteration": iteration_counter["value"],
            "items": [
                {"sample_id": sid, "completion": completion}
                for sid, completion in zip(sample_ids, completions)
            ],
        }

        rewards = [0.0 for _ in completions]

        try:
            response = session.post(
                f"{server_url.rstrip('/')}/grade",
                json=payload,
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()
            reward_map = {
                item.get("sample_id"): item.get("reward", 0.0)
                for item in data.get("results", [])
            }
            rewards = [float(reward_map.get(sid, 0.0)) for sid in sample_ids]
        except Exception as e:
            print(f"Failed to grade iteration {iteration_counter['value']}: {e}")

        iteration_counter["value"] += 1
        return rewards

    return reward_fn
