import typer
from typing_extensions import Annotated
from peft import PeftModel
import os

from .model_utils import get_model
from .config import load_training_config


def merge(
    config_path: Annotated[str, typer.Argument()] = "training_config.yaml",
    adapter_path: Annotated[str, typer.Option()] = None,
    base_model: Annotated[str, typer.Option()] = None
):
    config = load_training_config(config_path)
    
    if adapter_path is None:
        adapter_path = config.output_dir
    
    if base_model is None:
        base_model = config.model
    
    # Do not load in 8-bit to be able to merge
    # Do not load on gpu to avoid OOM
    model, tokenizer = get_model(base_model, load_in_4bit=False, device_map=None,
                                 max_seq_length=config.max_ctx_len, 
            update_tokenizer=config.update_tokenizer)
    print('Loaded model.')
    
    model.save_pretrained_merged(os.path.join(adapter_path, "merged"), tokenizer, save_method = "merged_16bit",)
    print('Merged and saved.')


def run():
    typer.run(merge)