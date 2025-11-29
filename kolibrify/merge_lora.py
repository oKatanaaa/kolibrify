import argparse
import os
import pathlib
import yaml

from .core import get_model, load_base_config
from .rl.config import load_rl_config


def merge(config_path, checkpoint=None, base_model=None):
    """
    Merge a LoRA checkpoint (SFT or RL) into full weights.

    Supports:
    - SFT configs that use BaseConfig (top-level output_dir)
    - RL configs that use RLConfig (paths.output_dir)
    """
    resolved_config_path = pathlib.Path(config_path).expanduser().resolve()
    config_dir = resolved_config_path.parent

    with resolved_config_path.open() as f:
        raw_cfg = yaml.safe_load(f)

    is_rl = isinstance(raw_cfg, dict) and "paths" in raw_cfg
    if is_rl:
        _, config = load_rl_config(str(resolved_config_path))
        adapter_path = config.paths.output_dir
        max_seq_len = config.model.max_seq_length
        add_imstart_token = config.model.add_imstart_token
        map_eos = config.model.map_eos_to_imend
        new_tokens = config.model.custom_tokens
    else:
        _, config = load_base_config(str(resolved_config_path))
        adapter_path = config.output_dir
        max_seq_len = config.max_ctx_len
        add_imstart_token = config.add_imstart_token
        map_eos = config.map_eos_to_imend
        new_tokens = config.custom_tokens

    adapter_path = pathlib.Path(adapter_path)
    if checkpoint is not None:
        adapter_path = adapter_path / checkpoint

    # Resolve relative paths relative to the config file location
    if not adapter_path.is_absolute():
        adapter_path = config_dir / adapter_path
    adapter_path = adapter_path.resolve()

    if not adapter_path.is_dir():
        raise FileNotFoundError(f"Adapter path does not exist: {adapter_path}")

    # Do not load on gpu to avoid OOM
    model, tokenizer = get_model(
        str(adapter_path),
        load_in_4bit=False,
        device_map=None,
        max_seq_length=max_seq_len,
        loading_lora=True,
        add_imstart_token=add_imstart_token,
        map_eos=map_eos,
        new_tokens=new_tokens,
    )
    print("Loaded model.")

    model.save_pretrained_merged(
        os.path.join(adapter_path, "merged"),
        tokenizer,
        save_method="merged_16bit",
    )
    print("Merged and saved.")


def run():
    parser = argparse.ArgumentParser(description="Merge LoRA adapters into the base model")
    parser.add_argument("config_path", help="Path to the configuration YAML file")
    parser.add_argument("--checkpoint", help="Path to a specific checkpoint to merge", default=None)
    parser.add_argument("--base_model", help="Path to a specific base model", default=None)
    
    args = parser.parse_args()
    merge(args.config_path, args.checkpoint, args.base_model)
