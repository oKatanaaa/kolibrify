import os
import argparse

from .core import get_model, load_base_config


def merge(config_path, checkpoint=None, base_model=None):
    _, config = load_base_config(config_path)
    adapter_path = config.output_dir
    if checkpoint is not None:
        adapter_path = os.path.join(adapter_path, checkpoint)
    
    # Do not load on gpu to avoid OOM
    model, tokenizer = get_model(
        adapter_path, load_in_4bit=False, device_map=None,
        max_seq_length=config.max_ctx_len,
        loading_lora=True, 
        add_imstart_token=config.add_imstart_token,
        map_eos=config.map_eos_to_imend,
        new_tokens=config.custom_tokens)
    print('Loaded model.')
    
    model.save_pretrained_merged(os.path.join(adapter_path, "merged"), tokenizer, save_method = "merged_16bit",)
    print('Merged and saved.')


def run():
    parser = argparse.ArgumentParser(description="Merge LoRA adapters into the base model")
    parser.add_argument("config_path", help="Path to the configuration YAML file")
    parser.add_argument("--checkpoint", help="Path to a specific checkpoint to merge", default=None)
    parser.add_argument("--base_model", help="Path to a specific base model", default=None)
    
    args = parser.parse_args()
    merge(args.config_path, args.checkpoint, args.base_model)
