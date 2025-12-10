from __future__ import annotations

import dataclasses
import pathlib
from typing import Dict, List, Optional
import yaml
from .builtin_graders import BUILTIN_PYTHON_GRADERS


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
class PythonGraderConfig:
    module: Optional[str]
    target: str
    path: Optional[pathlib.Path] = None
    builtin: Optional[str] = None
    init_kwargs: Dict[str, object] = dataclasses.field(default_factory=dict)


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
    python_graders: Dict[str, PythonGraderConfig]
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
    config_path = pathlib.Path(path).expanduser().resolve()
    config_dir = config_path.parent
    with config_path.open("r") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ConfigError("Config file must define a mapping at the top level")

    if "paths" not in raw or "data_root" not in raw["paths"]:
        raise ConfigError("'paths.data_root' is required in the config")

    raw_data_root = pathlib.Path(raw["paths"]["data_root"])
    if raw_data_root.is_absolute():
        data_root = raw_data_root
    else:
        data_root = config_dir / raw_data_root
    paths = PathsConfig(data_root=data_root)

    external_graders: Dict[str, ExternalGraderConfig] = {}
    for name, cfg in (raw.get("external_graders") or {}).items():
        if cfg.get("type") != "remote_http":
            raise ConfigError(f"Unsupported external grader type for '{name}'")
        external_graders[name] = ExternalGraderConfig(
            type=cfg["type"], url=cfg["url"], timeout_s=float(cfg.get("timeout_s", 60))
        )

    python_graders: Dict[str, PythonGraderConfig] = {}
    for name, cfg in (raw.get("python_graders") or {}).items():
        if not isinstance(cfg, dict):
            raise ConfigError(f"python_graders.{name} must be a mapping")

        init_kwargs = cfg.get("init_kwargs") or {}
        if not isinstance(init_kwargs, dict):
            raise ConfigError(f"python_graders.{name}.init_kwargs must be a mapping if provided")

        import_spec = cfg.get("import")
        path_spec = cfg.get("path")
        builtin_spec = cfg.get("builtin")
        specs = [bool(import_spec), bool(path_spec), bool(builtin_spec)]
        if sum(specs) != 1:
            raise ConfigError(
                f"python_graders.{name} must define exactly one of 'import', 'path', or 'builtin'"
            )

        if import_spec is not None:
            if not isinstance(import_spec, str):
                raise ConfigError(f"python_graders.{name}.import must be a string")
            if ":" not in import_spec:
                raise ConfigError(
                    f"python_graders.{name}.import must be in the form 'module:Attr'"
                )
            module_name, target = import_spec.rsplit(":", 1)
            if not module_name or not target:
                raise ConfigError(
                    f"python_graders.{name}.import must include both module and attribute"
                )
            python_graders[name] = PythonGraderConfig(
                module=module_name, target=target, init_kwargs=init_kwargs
            )
        elif path_spec is not None:
            if not isinstance(path_spec, str):
                raise ConfigError(f"python_graders.{name}.path must be a string")
            if ":" not in path_spec:
                raise ConfigError(
                    f"python_graders.{name}.path must be in the form 'path/to/file.py:Attr'"
                )
            path_str, target = path_spec.rsplit(":", 1)
            if not path_str or not target:
                raise ConfigError(
                    f"python_graders.{name}.path must include both file path and attribute"
                )
            grader_path = pathlib.Path(path_str)
            if not grader_path.is_absolute():
                grader_path = (config_dir / grader_path).resolve()
            if not grader_path.exists():
                raise ConfigError(f"python_graders.{name}.path does not exist: {grader_path}")
            python_graders[name] = PythonGraderConfig(
                module=None,
                target=target,
                path=grader_path,
                init_kwargs=init_kwargs,
            )
        else:
            if not isinstance(builtin_spec, str):
                raise ConfigError(f"python_graders.{name}.builtin must be a string")
            if builtin_spec not in BUILTIN_PYTHON_GRADERS:
                raise ConfigError(
                    f"python_graders.{name}.builtin references unknown grader '{builtin_spec}'"
                )
            module_name, target = BUILTIN_PYTHON_GRADERS[builtin_spec]
            python_graders[name] = PythonGraderConfig(
                module=module_name,
                target=target,
                builtin=builtin_spec,
                init_kwargs=init_kwargs,
            )

    datasets: Dict[str, DatasetConfig] = {}
    for dataset_id, cfg in (raw.get("datasets") or {}).items():
        dataset_path = pathlib.Path(cfg["path"])
        if not dataset_path.is_absolute():
            dataset_path = paths.data_root / dataset_path
        dataset_cfg = DatasetConfig(
            path=dataset_path,
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
        python_graders=python_graders,
        datasets=datasets,
        stages=stages,
    )


__all__ = [
    "ConfigError",
    "DatasetConfig",
    "ExternalGraderConfig",
    "PythonGraderConfig",
    "PathsConfig",
    "RLDataConfig",
    "StageConfig",
    "StageDatasetConfig",
    "load_config",
]
