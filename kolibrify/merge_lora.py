import os
import typer
from typing_extensions import Annotated

from .core import get_model, load_base_config


def merge(
    config_path: Annotated[str, typer.Argument()] = "training_config.yaml",
    checkpoint: Annotated[str, typer.Option()] = None,
    base_model: Annotated[str, typer.Option()] = None
):
    _, config = load_base_config(config_path)
    adapter_path = config.output_dir
    if checkpoint is not None:
        adapter_path = os.path.join(adapter_path, checkpoint)
    
    # Do not load in 8-bit to be able to merge
    # Do not load on gpu to avoid OOM
    model, tokenizer = get_model(
        adapter_path, load_in_4bit=False, device_map=None,
        max_seq_length=config.max_ctx_len,
        loading_lora=True, 
        add_imstart_token=config.add_imstart_token,
        map_eos=config.map_eos_to_imend)
    print('Loaded model.')
    
    model.save_pretrained_merged(os.path.join(adapter_path, "merged"), tokenizer, save_method = "merged_16bit",)
    print('Merged and saved.')


def run():
    typer.run(merge)