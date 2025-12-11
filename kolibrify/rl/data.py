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
        num_generations: int = 1,
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
        self.num_generations = max(int(num_generations), 1)

        # GRPO (via TRL RepeatSampler) repeats each unique prompt num_generations times.
        # To keep curriculum aligned with optimizer steps, we advance dataserver
        # iterations per GRPO "generation step" rather than per raw sampler index.
        # See docs/RL_GRPO_ITERATION.md for a full explanation.
        if self.num_generations == 1:
            self.prompts_per_step = self.batch_size
        else:
            if self.batch_size % self.num_generations != 0:
                print(
                    f"WARNING: per_device_batch_size={self.batch_size} not divisible by "
                    f"num_generations={self.num_generations}. Falling back to 1 prompt/step."
                )
                self.prompts_per_step = 1
            else:
                self.prompts_per_step = self.batch_size // self.num_generations

        # In TRL GRPO, sampler indices advance in units of unique prompts
        # (each index is then repeated num_generations times). So a GRPO step
        # corresponds to prompts_per_step *unique* indices.
        self.max_retries = max(int(max_retries), 0)
        self.retry_backoff_seconds = max(float(retry_backoff_seconds), 0.0)
        self.request_timeout_seconds = float(request_timeout_seconds)
        self.meta_timeout_seconds = float(meta_timeout_seconds)
        self._base_length = None
        self._dataset_length = None
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

        # HuggingFace trainers wrap map-style datasets with DistributedSampler for DDP,
        # which divides the dataset length by WORLD_SIZE. If we return only the per-device
        # length here, large WORLD_SIZE values prevent later curriculum stages from ever
        # being reached (e.g., with 8 GPUs, iterations may stay below 100 forever).
        # To keep per-rank sampling aligned with the intended iteration schedule, scale the
        # reported length by WORLD_SIZE but wrap indices back into the base range inside
        # __getitem__. This keeps the number of optimizer steps per rank at total_iterations
        # while avoiding dataserver oversubscription.
        world_size = 1
        try:
            world_size = max(int(os.environ.get("WORLD_SIZE", "1")), 1)
        except Exception:
            world_size = 1

        # Base length counts unique prompt indices. Each GRPO step consumes
        # prompts_per_step unique indices per rank.
        self._base_length = self.total_iterations * self.grad_accum * self.prompts_per_step
        self._dataset_length = self._base_length * world_size

        # Cache responses per (generation_step, grad_accum_step) so each GRPO step
        # makes its own /sample request instead of sharing across accumulation.
        self._iteration_cache: Dict[tuple[int, int], List[Dict]] = {}

        world_size = os.environ.get("WORLD_SIZE")
        world_size_note = f" (WORLD_SIZE={world_size})" if world_size else ""
        print(
            "Creating RL dataset with server at "
            f"{self.server_url} for {self.total_iterations} iterations "
            f"using per-device batch size {self.batch_size} "
            f"(prompts_per_step={self.prompts_per_step}, num_generations={self.num_generations}), "
            f"grad_accum={self.grad_accum}, "
            f"dataset_length={self._dataset_length}{world_size_note}"
        )

    def _reset_session(self):
        try:
            self.session.close()
        except Exception:
            pass
        self.session = requests.Session()

    def __len__(self) -> int:
        # The length is scaled by WORLD_SIZE so DistributedSampler in DDP
        # does not truncate the effective iteration range per rank.
        return self._dataset_length

    def _rows_for_iteration(self, generation_step: int, accum_step: int) -> List[Dict]:
        cache_key = (generation_step, accum_step)
        if cache_key not in self._iteration_cache:
            samples = _sample_batch(
                self.session,
                self.server_url,
                generation_step,
                self.prompts_per_step,
                timeout_seconds=self.request_timeout_seconds,
                max_retries=self.max_retries,
                retry_backoff_seconds=self.retry_backoff_seconds,
                reset_session=self._reset_session,
            )
            if not samples:
                raise RuntimeError(f"RL dataserver returned no samples at iteration {generation_step}.")

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

        # Map any DistributedSampler-strided index back into the base range
        # so curriculum stages advance correctly on every rank.
        base_idx = idx % self._base_length

        micro_batch = base_idx // self.prompts_per_step
        generation_step = micro_batch // self.grad_accum
        accum_step = micro_batch % self.grad_accum
        prompt_slot = base_idx % self.prompts_per_step
        rows = self._rows_for_iteration(generation_step, accum_step)

        if not rows:
            raise IndexError(f"No RL samples available for iteration {generation_step}.")

        # If the dataserver returned fewer than the requested batch, recycle rows as needed.
        return rows[prompt_slot % len(rows)]


def create_rl_dataset(
    server_url: str,
    per_device_batch_size: int,
    gradient_accumulation_steps: int,
    *,
    num_generations: int = 1,
    max_retries: int = 3,
    retry_backoff_seconds: float = 1.0,
    request_timeout_seconds: float = 60.0,
    meta_timeout_seconds: float = 30.0,
) -> torch.utils.data.Dataset:
    return RemoteRLDataset(
        server_url,
        per_device_batch_size,
        gradient_accumulation_steps,
        num_generations=num_generations,
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
        request_timeout_seconds=request_timeout_seconds,
        meta_timeout_seconds=meta_timeout_seconds,
    )
