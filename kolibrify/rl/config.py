import os
from dataclasses import dataclass, field, _MISSING_TYPE
from typing import List, Optional

import yaml


@dataclass
class RLModelConfig:
    base_model: str
    continued_training: bool = False
    checkpoint: Optional[str] = None
    access_token: Optional[str] = None
    load_in_4bit: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: int = 0
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    modules_to_save: Optional[List[str]] = field(default_factory=lambda: ["embed_tokens", "lm_head"])
    use_rslora: bool = False
    gradient_checkpointing: str | bool = "unsloth"
    max_seq_length: int = 2048
    add_imstart_token: bool = True
    map_eos_to_imend: bool = True
    custom_tokens: Optional[List[str]] = None
    cpu_offload_embeddings: bool = False
    merge: bool = False

    @property
    def max_ctx_len(self):
        return self.max_seq_length


@dataclass
class RLTrainingConfig:
    max_steps: int = 200_000
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    num_generations: int = 2
    temperature: float = 1.0
    learning_rate: float = 5e-5
    weight_decay: float = 0.001
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "linear"
    optim: str = "adamw_8bit"
    logging_steps: int = 10
    save_steps: int = 1000
    report_to: str = "none"
    max_prompt_length: int = 1024
    max_completion_length: int = 1024


@dataclass
class RLAlgorithmConfig:
    rl_algorithm: str = "grpo"


@dataclass
class RLDataConfig:
    server_url: str


@dataclass
class RLPathsConfig:
    output_dir: str = "experiments_rl"


@dataclass
class RLConfig:
    model: RLModelConfig
    training: RLTrainingConfig
    rl: RLAlgorithmConfig
    data: RLDataConfig
    paths: RLPathsConfig


def _warn_missing(prefix: str, raw_cfg: dict, dataclass_type):
    fields = dataclass_type.__dataclass_fields__
    missing_keys = list(set(fields.keys()) - set(raw_cfg.keys()))
    if len(missing_keys) > 0:
        for k in missing_keys:
            default_val = fields[k].default
            if isinstance(default_val, _MISSING_TYPE):
                default_val = fields[k].default_factory()
            print(f"WARNING! Missing key in {prefix}: {k}. Setting to default value: {default_val}")


def load_rl_config(config_path) -> tuple[dict, RLConfig]:
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    if "paths" not in config_dict:
        raise ValueError("Config must contain 'paths' section with output_dir")

    # Extract the config filename (without extension) from the path
    config_filename = os.path.splitext(os.path.basename(config_path))[0]

    # Update the output_dir to include the config filename
    base_output_dir = config_dict["paths"].get("output_dir", RLPathsConfig.output_dir)
    config_dict["paths"]["output_dir"] = os.path.join(base_output_dir, config_filename)

    model_cfg = config_dict.get("model", {})
    training_cfg = config_dict.get("training", {})
    rl_cfg = config_dict.get("rl", {})
    data_cfg = config_dict.get("data", {})

    _warn_missing("model", model_cfg, RLModelConfig)
    _warn_missing("training", training_cfg, RLTrainingConfig)
    _warn_missing("rl", rl_cfg, RLAlgorithmConfig)
    _warn_missing("data", data_cfg, RLDataConfig)

    model = RLModelConfig(**model_cfg)
    training = RLTrainingConfig(**training_cfg)
    rl = RLAlgorithmConfig(**rl_cfg)
    data = RLDataConfig(**data_cfg)
    paths = RLPathsConfig(**config_dict["paths"])

    if model.add_imstart_token:
        assert model.modules_to_save is not None and "embed_tokens" in model.modules_to_save and "lm_head" in model.modules_to_save, (
            "add_imstart_token=True, but you don't train embed_tokens and lm_head. "
            "Set modules_to_save to [\"embed_tokens\", \"lm_head\"]"
        )

    rl_config = RLConfig(
        model=model,
        training=training,
        rl=rl,
        data=data,
        paths=paths,
    )

    return config_dict, rl_config
