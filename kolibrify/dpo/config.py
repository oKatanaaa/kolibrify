from dataclasses import dataclass, field, _MISSING_TYPE
import yaml
import os
from typing import List, Union, Optional

from kolibrify.core import BaseConfig

@dataclass
class DatasetConfig:
    accepted: str
    rejected: str
    n_samples: int = -1


@dataclass
class StageConfig:
    name: str
    epochs: float
    datasets: List[DatasetConfig]


@dataclass
class TrainingConfig(BaseConfig):
    stages: List[StageConfig] = field(default_factory=lambda: [])


def load_stage_configs(stage_dicts: list) -> List[StageConfig]:
    stages = []
    for stage_dict in stage_dicts:
        stage_name = list(stage_dict.keys())[0]
        stage_dict = stage_dict[stage_name]
        epochs = stage_dict['epochs']
        dataset_dicts = stage_dict['datasets']
        dataset_configs = []
        for dataset_dict in dataset_dicts:
            v = list(dataset_dict.values())[0]
            accepted = v['accepted']
            rejected = v['rejected']
            n_samples = v.get('n_samples', -1)
            dataset_configs.append(
                DatasetConfig(accepted=accepted, rejected=rejected, n_samples=n_samples))
        
        stage = StageConfig(name=stage_name, epochs=epochs, datasets=dataset_configs)
        stages.append(stage)
    return stages


def load_training_config(config_path) -> tuple[dict, TrainingConfig]:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    _config = dict([(k, v) for k, v in config.items() if v is not None])
    
    # Extract the config filename (without extension) from the path
    config_filename = os.path.splitext(os.path.basename(config_path))[0]
    
    # Update the output_dir to include the config filename
    assert 'output_dir' in _config
    _config['output_dir'] = os.path.join(_config['output_dir'], config_filename)
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