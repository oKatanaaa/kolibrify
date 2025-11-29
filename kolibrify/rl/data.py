import os
import time
from typing import Dict, List

import requests
import torch


def _fetch_meta(
    session: requests.Session,
    server_url: str,
    timeout_seconds: float,
    max_retries: int,
    retry_backoff_seconds: float,
    reset_session=None,
) -> Dict:
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            response = session.get(f"{server_url.rstrip('/')}/meta", timeout=timeout_seconds)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                delay = retry_backoff_seconds * (2**attempt)
                print(f"Failed to fetch /meta from RL dataserver (attempt {attempt + 1}/{max_retries + 1}): {e}. Retrying in {delay:.1f}s...")
                time.sleep(delay)
                if reset_session:
                    reset_session()
            else:
                print(f"Failed to fetch /meta from RL dataserver after {max_retries + 1} attempts: {e}")
    return {}


def _sample_batch(
    session: requests.Session,
    server_url: str,
    iteration: int,
    batch_size: int,
    timeout_seconds: float,
    max_retries: int,
    retry_backoff_seconds: float,
    reset_session=None,
) -> list:
    payload = {"iteration": iteration, "batch_size": batch_size}
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            response = session.post(
                f"{server_url.rstrip('/')}/sample",
                json=payload,
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            data = response.json()
            samples = data.get("samples", [])
            if not samples:
                raise RuntimeError("Dataserver returned no samples.")
            return samples
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                delay = retry_backoff_seconds * (2**attempt)
                print(
                    f"Failed to fetch samples at iteration {iteration} (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                    f"Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)
                if reset_session:
                    reset_session()
            else:
                print(f"Failed to fetch samples at iteration {iteration} after {max_retries + 1} attempts: {e}")
    return []


def _build_prompt(system_prompt: str | None, user_prompt: str) -> list:
    prompt = []
    if system_prompt and system_prompt.strip():
        prompt.append({"role": "system", "content": system_prompt})

    prompt.append({"role": "user", "content": user_prompt})
    return prompt


class RemoteRLDataset(torch.utils.data.Dataset):
    """Map-style wrapper around the RL dataserver (GRPOTrainer disallows IterableDataset)."""
    def __init__(
        self,
        server_url: str,
        per_device_batch_size: int,
        gradient_accumulation_steps: int,
        *,
        max_retries: int = 3,
        retry_backoff_seconds: float = 1.0,
        request_timeout_seconds: float = 60.0,
        meta_timeout_seconds: float = 30.0,
    ):
        # Each rank/process creates its own dataset, so we should sample only the
        # per-device batch. Multiplying by WORLD_SIZE would oversubscribe the dataserver
        # (global batch becomes per_device_batch_size * WORLD_SIZE^2) and delay stage
        # boundaries that rely on iteration counts.
        self.server_url = server_url.rstrip("/")
        self.batch_size = per_device_batch_size
        self.grad_accum = max(int(gradient_accumulation_steps), 1)
        self.max_retries = max(int(max_retries), 0)
        self.retry_backoff_seconds = max(float(retry_backoff_seconds), 0.0)
        self.request_timeout_seconds = float(request_timeout_seconds)
        self.meta_timeout_seconds = float(meta_timeout_seconds)
        self.session = requests.Session()

        meta = _fetch_meta(
            self.session,
            self.server_url,
            timeout_seconds=self.meta_timeout_seconds,
            max_retries=self.max_retries,
            retry_backoff_seconds=self.retry_backoff_seconds,
            reset_session=self._reset_session,
        )
        total_iterations = meta.get("total_iterations")
        if total_iterations is None:
            print("WARNING: RL dataserver did not return total_iterations. Defaulting to 1,000 iterations.")
            total_iterations = 1000

        self.total_iterations = int(total_iterations)
        # Cache responses per (iteration, grad_accum_step) so each micro-batch
        # makes its own /sample request instead of sharing across accumulation.
        self._iteration_cache: Dict[tuple[int, int], List[Dict]] = {}

        world_size = os.environ.get("WORLD_SIZE")
        world_size_note = f" (WORLD_SIZE={world_size})" if world_size else ""
        print(
            "Creating RL dataset with server at "
            f"{self.server_url} for {self.total_iterations} iterations "
            f"using per-device batch size {self.batch_size}, "
            f"grad_accum={self.grad_accum}{world_size_note}"
        )

    def _reset_session(self):
        try:
            self.session.close()
        except Exception:
            pass
        self.session = requests.Session()

    def __len__(self) -> int:
        # Each dataserver iteration corresponds to one optimizer step; the trainer
        # issues grad_accum micro-batches per step, so scale the dataset length to
        # avoid wrapping and resampling early stages.
        return self.total_iterations * self.grad_accum * self.batch_size

    def _rows_for_iteration(self, iteration: int, accum_step: int) -> List[Dict]:
        cache_key = (iteration, accum_step)
        if cache_key not in self._iteration_cache:
            samples = _sample_batch(
                self.session,
                self.server_url,
                iteration,
                self.batch_size,
                timeout_seconds=self.request_timeout_seconds,
                max_retries=self.max_retries,
                retry_backoff_seconds=self.retry_backoff_seconds,
                reset_session=self._reset_session,
            )
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

            self._iteration_cache[cache_key] = rows

        return self._iteration_cache[cache_key]

    def __getitem__(self, idx: int) -> Dict:
        if idx < 0 or idx >= len(self):
            raise IndexError

        micro_batch = idx // self.batch_size
        iteration = micro_batch // self.grad_accum
        accum_step = micro_batch % self.grad_accum
        offset = idx % self.batch_size
        rows = self._rows_for_iteration(iteration, accum_step)

        if not rows:
            raise IndexError(f"No RL samples available for iteration {iteration}.")

        # If the dataserver returned fewer than the requested batch, recycle rows as needed.
        return rows[offset % len(rows)]


def create_rl_dataset(
    server_url: str,
    per_device_batch_size: int,
    gradient_accumulation_steps: int,
    *,
    max_retries: int = 3,
    retry_backoff_seconds: float = 1.0,
    request_timeout_seconds: float = 60.0,
    meta_timeout_seconds: float = 30.0,
) -> torch.utils.data.Dataset:
    return RemoteRLDataset(
        server_url,
        per_device_batch_size,
        gradient_accumulation_steps,
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
        request_timeout_seconds=request_timeout_seconds,
        meta_timeout_seconds=meta_timeout_seconds,
    )
