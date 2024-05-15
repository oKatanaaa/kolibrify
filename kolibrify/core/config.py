from dataclasses import dataclass, field
from typing import List, Union, Optional


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
    cpu_offload_embeddings: bool = False
