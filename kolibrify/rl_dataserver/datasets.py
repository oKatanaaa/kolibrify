from __future__ import annotations

import json
import pathlib
from typing import Dict, List

from .config import DatasetConfig


def load_jsonl(path: pathlib.Path) -> List[dict]:
    records: List[dict] = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def load_datasets(configs: Dict[str, DatasetConfig]) -> Dict[str, List[dict]]:
    datasets: Dict[str, List[dict]] = {}
    for dataset_id, cfg in configs.items():
        datasets[dataset_id] = load_jsonl(cfg.path)
    return datasets


__all__ = ["load_datasets", "load_jsonl"]
