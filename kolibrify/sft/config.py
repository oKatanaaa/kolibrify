import os
from dataclasses import dataclass, field, _MISSING_TYPE
import yaml
from typing import List, Union, Optional

from kolibrify.core import BaseConfig


@dataclass
class DatasetConfig:
    path: str
    weight: float = 1.0


@dataclass
class StageConfig:
    name: str
    until_step: int
    datasets: List[DatasetConfig]


@dataclass
class TrainingConfig(BaseConfig):
    stages: List[StageConfig] = field(default_factory=lambda: [])


def load_stage_configs(stage_dicts: list) -> List[StageConfig]:
    stages = []
    for stage_dict in stage_dicts:
        stage_name = stage_dict['name']
        until_step = stage_dict['until_step']
        dataset_dicts = stage_dict['datasets']
        dataset_configs = []
        for dataset_dict in dataset_dicts:
            # Accept either a simple path string or a dictionary with details.
            if isinstance(dataset_dict, str):
                dataset_configs.append(DatasetConfig(path=dataset_dict))
                continue

            # Support short-form {some_label: "path"} while transitioning configs.
            if 'path' not in dataset_dict and len(dataset_dict) == 1:
                path_val = list(dataset_dict.values())[0]
                dataset_dict = {'path': path_val}

            path = dataset_dict['path']
            weight = dataset_dict.get('weight', 1.0)
            dataset_configs.append(DatasetConfig(path=path, weight=weight))
        
        stage = StageConfig(stage_name, until_step, dataset_configs)
        stages.append(stage)
    return stages


def load_training_config(config_path) -> tuple[dict, TrainingConfig]:
    config_abspath = os.path.abspath(config_path)
    config_dir = os.path.dirname(config_abspath)

    with open(config_abspath) as f:
        config = yaml.safe_load(f)
    
    _config = dict([(k, v) for k, v in config.items() if v is not None])
    
    # Extract the config filename (without extension) from the path
    config_filename = os.path.splitext(os.path.basename(config_abspath))[0]
    
    # Update the output_dir to include the config filename
    assert 'output_dir' in _config
    output_dir = _config['output_dir']
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(config_dir, output_dir)
    _config['output_dir'] = os.path.join(output_dir, config_filename)
    # Also update the original config dict
    config['output_dir'] = _config['output_dir']
    
    # Check data integrity
    fields = TrainingConfig.__dataclass_fields__
    missing_keys = list(set(fields.keys()) - set(_config.keys()))
    if len(missing_keys) > 0:
        for k in missing_keys:
            default_val = fields[k].default
            if isinstance(default_val, _MISSING_TYPE):
                default_val = fields[k].default_factory()
            print(f'WARNING! Missing key: {k}. Setting to default value: {default_val}')

    stages = load_stage_configs(_config['stages'])
    _config['stages'] = stages
    
    training_config = TrainingConfig(**_config)
    
    if training_config.add_imstart_token:
        assert 'embed_tokens' in training_config.modules_to_save and 'lm_head' in training_config.modules_to_save, \
            "add_imstart_token=True, but you don't train embed_tokens and lm_head. Set modules_to_save to [\"embed_tokens\", \"lm_head\"]"
            
    return config, training_config


if __name__ == "__main__":
    config = load_training_config("training_config.yaml")
    print(config)
