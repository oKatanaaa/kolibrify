from __future__ import annotations

import dataclasses
import pathlib
from typing import Dict, List
import yaml


@dataclasses.dataclass
class PathsConfig:
    data_root: pathlib.Path


@dataclasses.dataclass
class ExternalGraderConfig:
    type: str
    url: str
    timeout_s: float


@dataclasses.dataclass
class DatasetConfig:
    path: pathlib.Path
    graders: List[str]
    grader_weights: List[float]


@dataclasses.dataclass
class StageDatasetConfig:
    id: str
    weight: float


@dataclasses.dataclass
class StageConfig:
    name: str
    until_step: int
    datasets: List[StageDatasetConfig]


@dataclasses.dataclass
class RLDataConfig:
    paths: PathsConfig
    external_graders: Dict[str, ExternalGraderConfig]
    datasets: Dict[str, DatasetConfig]
    stages: List[StageConfig]


class ConfigError(Exception):
    pass


def _validate_dataset_config(dataset_id: str, cfg: DatasetConfig) -> None:
    if cfg.graders or cfg.grader_weights:
        if len(cfg.graders) != len(cfg.grader_weights):
            raise ConfigError(
                f"Dataset '{dataset_id}' must have the same number of graders and grader_weights"
            )


def _normalize_weights(items: List[StageDatasetConfig]) -> List[StageDatasetConfig]:
    total = sum(d.weight for d in items)
    if total <= 0:
        raise ConfigError("Stage dataset weights must sum to a positive value")
    normalized: List[StageDatasetConfig] = []
    for d in items:
        normalized.append(StageDatasetConfig(id=d.id, weight=d.weight / total))
    return normalized


def load_config(path: str) -> RLDataConfig:
    config_path = pathlib.Path(path)
    with config_path.open("r") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ConfigError("Config file must define a mapping at the top level")

    if "paths" not in raw or "data_root" not in raw["paths"]:
        raise ConfigError("'paths.data_root' is required in the config")

    paths = PathsConfig(data_root=pathlib.Path(raw["paths"]["data_root"]))

    external_graders: Dict[str, ExternalGraderConfig] = {}
    for name, cfg in (raw.get("external_graders") or {}).items():
        if cfg.get("type") != "remote_http":
            raise ConfigError(f"Unsupported external grader type for '{name}'")
        external_graders[name] = ExternalGraderConfig(
            type=cfg["type"], url=cfg["url"], timeout_s=float(cfg.get("timeout_s", 60))
        )

    datasets: Dict[str, DatasetConfig] = {}
    for dataset_id, cfg in (raw.get("datasets") or {}).items():
        dataset_cfg = DatasetConfig(
            path=paths.data_root / cfg["path"],
            graders=list(cfg.get("graders") or []),
            grader_weights=[float(w) for w in (cfg.get("grader_weights") or [])],
        )
        _validate_dataset_config(dataset_id, dataset_cfg)
        datasets[dataset_id] = dataset_cfg

    stages: List[StageConfig] = []
    for stage_raw in raw.get("stages", []):
        datasets_raw = stage_raw.get("datasets") or []
        for dataset_raw in datasets_raw:
            dataset_id = dataset_raw["id"]
            if dataset_id not in datasets:
                raise ConfigError(
                    f"Stage '{stage_raw['name']}' references unknown dataset '{dataset_id}'"
                )
        stage_datasets = [
            StageDatasetConfig(id=d["id"], weight=float(d["weight"])) for d in datasets_raw
        ]
        normalized = _normalize_weights(stage_datasets)
        stages.append(
            StageConfig(
                name=stage_raw["name"],
                until_step=int(stage_raw["until_step"]),
                datasets=normalized,
            )
        )

    if not stages:
        raise ConfigError("At least one stage must be defined in the config")

    return RLDataConfig(
        paths=paths,
        external_graders=external_graders,
        datasets=datasets,
        stages=stages,
    )


__all__ = [
    "ConfigError",
    "DatasetConfig",
    "ExternalGraderConfig",
    "PathsConfig",
    "RLDataConfig",
    "StageConfig",
    "StageDatasetConfig",
    "load_config",
]
