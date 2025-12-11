from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pytest


def _repeat_sampler_indices(steps: int, prompts_per_step: int, num_generations: int) -> List[int]:
    """Emulate TRL RepeatSampler: each unique idx repeated num_generations times."""
    out: List[int] = []
    for step in range(steps):
        for p in range(prompts_per_step):
            base = step * prompts_per_step + p
            out.extend([base] * num_generations)
    return out


def test_grpo_num_generations_advances_sample_every_step(monkeypatch):
    """With num_generations>1, dataset should call /sample once per GRPO step."""
    import kolibrify.rl.data as data_mod

    calls: List[Tuple[int, int]] = []

    def fake_fetch_meta(*args, **kwargs) -> Dict[str, Any]:
        return {"total_iterations": 100}

    def fake_sample_batch(
        session, server_url, iteration: int, batch_size: int, *args, **kwargs
    ) -> List[Dict[str, Any]]:
        calls.append((iteration, batch_size))
        return [
            {"prompt": f"p{iteration}-{i}", "system_prompt": None, "sample_id": f"{iteration}-{i}"}
            for i in range(batch_size)
        ]

    monkeypatch.setattr(data_mod, "_fetch_meta", fake_fetch_meta)
    monkeypatch.setattr(data_mod, "_sample_batch", fake_sample_batch)

    per_device_batch_size = 64
    num_generations = 16
    prompts_per_step = per_device_batch_size // num_generations

    dataset = data_mod.RemoteRLDataset(
        server_url="http://fake",
        per_device_batch_size=per_device_batch_size,
        gradient_accumulation_steps=1,
        num_generations=num_generations,
    )

    indices = _repeat_sampler_indices(steps=3, prompts_per_step=prompts_per_step, num_generations=num_generations)
    for idx in indices:
        _ = dataset[idx]

    # One /sample per step, iterations 0,1,2.
    assert calls == [(0, prompts_per_step), (1, prompts_per_step), (2, prompts_per_step)]


def test_num_generations_one_keeps_old_cadence(monkeypatch):
    """num_generations=1 should behave like the previous per_device_batch_size cadence."""
    import kolibrify.rl.data as data_mod

    calls: List[Tuple[int, int]] = []

    monkeypatch.setattr(data_mod, "_fetch_meta", lambda *a, **k: {"total_iterations": 100})

    def fake_sample_batch(session, server_url, iteration: int, batch_size: int, *args, **kwargs):
        calls.append((iteration, batch_size))
        return [
            {"prompt": f"p{iteration}-{i}", "system_prompt": None, "sample_id": f"{iteration}-{i}"}
            for i in range(batch_size)
        ]

    monkeypatch.setattr(data_mod, "_sample_batch", fake_sample_batch)

    dataset = data_mod.RemoteRLDataset(
        server_url="http://fake",
        per_device_batch_size=8,
        gradient_accumulation_steps=1,
        num_generations=1,
    )

    # Touch indices across two per-device batches.
    for idx in range(0, 16):
        _ = dataset[idx]

    assert calls == [(0, 8), (1, 8)]

