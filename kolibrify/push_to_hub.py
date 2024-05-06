import typer
from typing_extensions import Annotated
from peft import PeftModel
import os

from .model_utils import get_model
from .config import load_training_config


def share(
    config_path: Annotated[str, typer.Argument()],
    hf_repo: Annotated[str, typer.Argument(help="Name of your HF repository. DO NOT CREATE IT PRIOR TO PUBLISHING.")],
    quantize: Annotated[str, typer.Option(help="Type of quantization. See llama.ccp for supported quants.")] = None,
    hf_token: Annotated[str, typer.Option(help="Your huggingface token.")] = None
):
    _, config = load_training_config(config_path)
    token = config.access_token
    if hf_token is not None:
        token = hf_token
    
    model_path = os.path.join(config.output_dir, 'merged')
    
    # Do not load in 8-bit to be able to merge
    # Do not load on gpu to avoid OOM
    model, tokenizer = get_model(
        model_path, load_in_4bit=False, device_map=None,
        max_seq_length=config.max_ctx_len,
        loading_lora=False, add_imstart_token=False)
    print('Loaded model.')
    
    if quantize is not None:
        model.push_to_hub_gguf(hf_repo, tokenizer, token=token, quantization_method=quantize)
    else:
        model.push_to_hub_merged(hf_repo, tokenizer, token=token)
    print('Pushed to hub.')


def run():
    typer.run(share)