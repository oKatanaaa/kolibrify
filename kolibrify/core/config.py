from dataclasses import dataclass, field, _MISSING_TYPE
from typing import List, Union, Optional
import yaml
import os


@dataclass
class BaseConfig:
    model: str
    continued_training: bool = False
    checkpoint: Optional[str] = None
    access_token: Union[str, None] = None
    output_dir: str = "experiments"
    val_dataset_file: Union[str, None] = None
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: int = 0
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    modules_to_save: Optional[List[str]] = None
    use_rslora: bool = False
    micro_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    group_by_seq_len: bool = False
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
    map_eos_to_imend: bool = True
    custom_tokens: Optional[List[str]] = None
    load_in_4bit: bool = True
    cpu_offload_embeddings: bool = False


def load_base_config(config_path) -> tuple[dict, BaseConfig]:
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    
    _config = dict([(k, v) for k, v in config_dict.items() if v is not None])
    # Check data integrity
    fields = BaseConfig.__dataclass_fields__
    missing_keys = list(set(fields.keys()) - set(_config.keys()))
    if len(missing_keys) > 0:
        for k in missing_keys:
            default_val = fields[k].default
            if isinstance(default_val, _MISSING_TYPE):
                default_val = fields[k].default_factory()
            print(f'WARNING! Missing key: {k}. Setting to default value: {default_val}')
    _config.pop('stages')
    
    training_config = BaseConfig(**_config)
    
    if training_config.add_imstart_token:
        assert 'embed_tokens' in training_config.modules_to_save and 'lm_head' in training_config.modules_to_save, \
            "add_imstart_token=True, but you don't train embed_tokens and lm_head. Set modules_to_save to [\"embed_tokens\", \"lm_head\"]"
            
    return config_dict, training_config


def save_config(config_dict: dict):
    output_dir = config_dict['output_dir']
    # Remove token info
    config_dict['access_token'] = None
    with open(os.path.join(output_dir, 'kolibrify-config.yaml'), 'w') as f:
        yaml.safe_dump(config_dict, f, sort_keys=False)
        print('Saved config in the output directory.')
