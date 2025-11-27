import itertools
import os
from typing import Dict, Iterable

import datasets
import requests

datasets.disable_caching()


def _fetch_meta(session: requests.Session, server_url: str) -> Dict:
    try:
        response = session.get(f"{server_url.rstrip('/')}/meta", timeout=30)
        if response.ok:
            return response.json()
    except Exception as e:
        print(f"Failed to fetch /meta from RL dataserver: {e}")
    return {}


def _sample_batch(session: requests.Session, server_url: str, iteration: int, batch_size: int) -> list:
    payload = {"iteration": iteration, "batch_size": batch_size}
    try:
        response = session.post(
            f"{server_url.rstrip('/')}/sample",
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("samples", [])
    except Exception as e:
        print(f"Failed to fetch samples at iteration {iteration}: {e}")
    return []


def _build_prompt(system_prompt: str | None, user_prompt: str) -> list:
    prompt = []
    if system_prompt and system_prompt.strip():
        prompt.append({"role": "system", "content": system_prompt})

    prompt.append({"role": "user", "content": user_prompt})
    return prompt


def _rl_generator(server_url: str, batch_size: int) -> Iterable[Dict]:
    session = requests.Session()
    meta = _fetch_meta(session, server_url)
    total_iterations = meta.get("total_iterations")

    for iteration in itertools.count():
        if total_iterations is not None and iteration >= total_iterations:
            break

        samples = _sample_batch(session, server_url, iteration, batch_size)
        if not samples:
            continue

        for sample in samples:
            prompt = _build_prompt(sample.get("system_prompt"), sample.get("prompt", ""))
            row = {
                "prompt": prompt,
                "sample_id": sample.get("sample_id"),
            }
            if "metadata" in sample:
                row["metadata"] = sample.get("metadata")
            yield row


def create_rl_dataset(server_url: str, per_device_batch_size: int) -> datasets.IterableDataset:
    effective_world_size = int(os.environ.get("WORLD_SIZE", 1))
    batch_size = per_device_batch_size * effective_world_size
    print(f"Creating RL dataset with server at {server_url} using batch size {batch_size}")
    return datasets.IterableDataset.from_generator(
        lambda: _rl_generator(server_url, batch_size)
    )
