import argparse
import os

from .core import get_model, load_base_config


def share(
    config_path, hf_repo, quantize=None, hf_token=None
):
    _, config = load_base_config(config_path)
    token = config.access_token
    if hf_token is not None:
        token = hf_token
    
    model_path = os.path.join(config.output_dir, 'merged')
    assert os.path.exists(model_path), f'Model {model_path} does not exist.'
    
    # Do not load in 8-bit to be able to merge
    # Do not load on gpu to avoid OOM
    model, tokenizer = get_model(
        model_path, load_in_4bit=False, device_map=None,
        max_seq_length=config.max_ctx_len,
        loading_lora=False, add_imstart_token=False, map_eos=False)
    print('Loaded model.')
    
    if quantize is not None:
        model.push_to_hub_gguf(hf_repo, tokenizer, token=token, quantization_method=quantize)
    else:
        model.push_to_hub_merged(hf_repo, tokenizer, token=token)
    print('Pushed to hub.')


def run():
    parser = argparse.ArgumentParser(description="Push a model to the Hugging Face Hub")
    parser.add_argument("config_path", help="Path to the configuration YAML file")
    parser.add_argument("hf_repo", help="Name of the Hugging Face repository to push to")
    parser.add_argument("--quantize", help="Type of quantization to use", default=None)
    parser.add_argument("--hf_token", help="Hugging Face token (overrides config token)", default=None)
    
    args = parser.parse_args()
    share(args.config_path, args.hf_repo, args.quantize, args.hf_token)