import os
from typing import Dict, List

import requests
import torch


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


class RemoteRLDataset(torch.utils.data.Dataset):
    """Map-style wrapper around the RL dataserver (GRPOTrainer disallows IterableDataset)."""
    def __init__(self, server_url: str, per_device_batch_size: int):
        # Each rank/process creates its own dataset, so we should sample only the
        # per-device batch. Multiplying by WORLD_SIZE would oversubscribe the dataserver
        # (global batch becomes per_device_batch_size * WORLD_SIZE^2) and delay stage
        # boundaries that rely on iteration counts.
        self.server_url = server_url.rstrip("/")
        self.batch_size = per_device_batch_size
        self.session = requests.Session()

        meta = _fetch_meta(self.session, self.server_url)
        total_iterations = meta.get("total_iterations")
        if total_iterations is None:
            print("WARNING: RL dataserver did not return total_iterations. Defaulting to 1,000 iterations.")
            total_iterations = 1000

        self.total_iterations = int(total_iterations)
        self._iteration_cache: Dict[int, List[Dict]] = {}

        world_size = os.environ.get("WORLD_SIZE")
        world_size_note = f" (WORLD_SIZE={world_size})" if world_size else ""
        print(
            "Creating RL dataset with server at "
            f"{self.server_url} for {self.total_iterations} iterations "
            f"using per-device batch size {self.batch_size}{world_size_note}"
        )

    def __len__(self) -> int:
        return self.total_iterations * self.batch_size

    def _rows_for_iteration(self, iteration: int) -> List[Dict]:
        if iteration not in self._iteration_cache:
            samples = _sample_batch(self.session, self.server_url, iteration, self.batch_size)
            if not samples:
                raise RuntimeError(f"RL dataserver returned no samples at iteration {iteration}.")

            rows: List[Dict] = []
            for sample in samples:
                prompt = _build_prompt(sample.get("system_prompt"), sample.get("prompt", ""))
                row = {
                    "prompt": prompt,
                    "sample_id": sample.get("sample_id"),
                }
                if "metadata" in sample:
                    row["metadata"] = sample.get("metadata")
                rows.append(row)

            self._iteration_cache[iteration] = rows

        return self._iteration_cache[iteration]

    def __getitem__(self, idx: int) -> Dict:
        if idx < 0 or idx >= len(self):
            raise IndexError

        iteration = idx // self.batch_size
        offset = idx % self.batch_size
        rows = self._rows_for_iteration(iteration)

        if not rows:
            raise IndexError(f"No RL samples available for iteration {iteration}.")

        # If the dataserver returned fewer than the requested batch, recycle rows as needed.
        return rows[offset % len(rows)]


def create_rl_dataset(server_url: str, per_device_batch_size: int) -> torch.utils.data.Dataset:
    return RemoteRLDataset(server_url, per_device_batch_size)
