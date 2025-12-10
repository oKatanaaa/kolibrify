import json
import random
from pathlib import Path

import yaml

from kolibrify.rl_dataserver.config import load_config
from kolibrify.rl_dataserver.server import RLDataServer


def _write_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write(json.dumps(record) + "\n")


def _build_config(tmp_path: Path, stages) -> Path:
    """Create a minimal RL dataserver config and datasets under tmp_path."""
    data_root = tmp_path / "data"
    datasets_cfg = {}
    for stage in stages:
        for ds in stage["datasets"]:
            ds_id = ds["id"]
            if ds_id in datasets_cfg:
                continue
            _write_jsonl(data_root / f"{ds_id}.jsonl", {"prompt": f"{ds_id} prompt", "answer": "ok"})
            datasets_cfg[ds_id] = {
                "path": f"{ds_id}.jsonl",
                "graders": [],
                "grader_weights": [],
                "multiplicative_graders": [],
            }

    cfg = {
        "paths": {"data_root": str(data_root)},
        "python_graders": {},
        "external_graders": {},
        "datasets": datasets_cfg,
        "stages": stages,
    }
    cfg_path = tmp_path / "rl_stage_config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return cfg_path


def test_stage_switch_sampling(tmp_path, monkeypatch):
    """Datasets from later stages should only appear after the until_step boundary."""
    stages = [
        {"name": "warm", "until_step": 2, "datasets": [{"id": "warm-ds", "weight": 1.0}]},
        {"name": "main", "until_step": 4, "datasets": [{"id": "main-ds", "weight": 1.0}]},
    ]
    cfg_path = _build_config(tmp_path, stages)
    random.seed(0)
    server = RLDataServer(str(cfg_path), verbose=False)

    warm_samples = server.sample_batch(iteration=0, batch_size=8)
    main_samples = server.sample_batch(iteration=2, batch_size=8)  # iteration >= until_step switches stage

    assert {s.dataset_id for s in warm_samples} == {"warm-ds"}
    assert {s.dataset_id for s in main_samples} == {"main-ds"}


def test_stage_weights_are_normalized(tmp_path):
    """Stage weights should be normalized and honored when sampling."""
    stages = [
        {"name": "warm", "until_step": 1, "datasets": [{"id": "a", "weight": 1.0}]},
        {
            "name": "main",
            "until_step": 3,
            "datasets": [
                {"id": "b", "weight": 1.0},
                {"id": "c", "weight": 3.0},
            ],
        },
    ]
    cfg_path = _build_config(tmp_path, stages)

    # Config loader normalizes weights.
    cfg = load_config(str(cfg_path))
    main_stage = cfg.stages[1]
    weights = [d.weight for d in main_stage.datasets]
    assert abs(sum(weights) - 1.0) < 1e-6
    assert weights[0] == 0.25 and weights[1] == 0.75

    # Sampling should reflect the normalized proportions.
    random.seed(123)
    server = RLDataServer(str(cfg_path), verbose=False)
    samples = server.sample_batch(iteration=1, batch_size=400)
    counts = {"b": 0, "c": 0}
    for s in samples:
        counts[s.dataset_id] += 1

    ratio_b = counts["b"] / (counts["b"] + counts["c"])
    # With weights 1:3 we expect ~0.25 for "b".
    assert 0.15 <= ratio_b <= 0.35
