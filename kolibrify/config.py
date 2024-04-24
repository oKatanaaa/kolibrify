from dataclasses import dataclass, field, _MISSING_TYPE
import yaml
from typing import List, Union, Optional


@dataclass
class DatasetConfig:
    path: str
    n_samples: int = -1


@dataclass
class StageConfig:
    name: str
    epochs: float
    datasets: List[DatasetConfig]


@dataclass
class TrainingConfig:
    model: str
    stages: List[StageConfig]
    continued_training: bool = False
    checkpoint: Optional[str] = None
    access_token: Union[str, None] = None
    output_dir: str = "experiments"
    val_dataset_file: Union[str, None] = None
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: int = 0
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    modules_to_save: List[str] = field(default_factory=lambda: ["embed_tokens", "lm_head"])
    use_rslora: bool = False
    micro_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    learning_rate: float = 1e-5
    lr_scheduler_type: str = "linear"
    lr_scheduler_kwargs: Optional[dict] = None
    warmup_steps: int = 0
    max_ctx_len: int = 2048
    logging_steps: int = 5
    eval_steps: int = 60
    save_steps: int = 60
    save_total_limit: int = 3
    add_imstart_token: bool = True
    load_in_4bit: bool = True


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
            if isinstance(v, str):
                # Only path is provided
                dataset_configs.append(DatasetConfig(v))
                continue
            
            path = v['path']
            n_samples = v.get('n_samples', -1)
            dataset_configs.append(DatasetConfig(path, n_samples))
        
        stage = StageConfig(stage_name, epochs, dataset_configs)
        stages.append(stage)
    return stages


def load_training_config(config_path) -> tuple[dict, TrainingConfig]:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    _config = dict([(k, v) for k, v in config.items() if v is not None])
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
            
    return config, TrainingConfig(**_config)


if __name__ == "__main__":
    config = load_training_config("training_config.yaml")
    print(config)