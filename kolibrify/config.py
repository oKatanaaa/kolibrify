from dataclasses import dataclass, field, _MISSING_TYPE
import yaml
from typing import List, Union, Optional

@dataclass
class TrainingConfig:
    model: str
    train_datasets: List[dict]
    continued_training: bool = False
    checkpoint: Optional[str] = None
    access_token: Union[str, None] = None
    output_dir: str = "experiments"
    val_dataset_file: Union[str, None] = None
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: int = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    modules_to_save: List[str] = field(default_factory=lambda: ["embed_tokens", "lm_head"])
    micro_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-5
    train_steps: int = 180
    warmup_steps:int = 100
    max_ctx_len: int = 1512
    logging_steps: int = 5
    eval_steps: int = 60
    save_steps: int = 60
    save_total_limit: int = 3
    update_tokenizer: bool = False
    load_in_4bit: bool = True


def load_training_config(config_path) -> TrainingConfig:
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
    
    return TrainingConfig(**_config)


if __name__ == "__main__":
    config = load_training_config("training_config.yaml")
    print(config)