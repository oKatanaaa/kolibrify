import argparse
import os
import torch

from .core.config import load_base_config
from huggingface_hub import HfApi

def share(
    config_path, hf_repo, quantize=None, hf_token=None
):
    _, config = load_base_config(config_path)
    token = config.access_token
    if hf_token is not None:
        token = hf_token
    
    model_path = os.path.join(config.output_dir, 'merged')
    assert os.path.exists(model_path), f'The model has not been merged. Please merge the model first before pushing to hub.'
    
    hf_api = HfApi(token=token)
    
    if quantize is not None:
        raise NotImplementedError('At the moment quantization is not supported.'    \
            'Please use external utilities to quantize the model')
        model.push_to_hub_gguf(hf_repo, tokenizer, token=token, quantization_method=quantize)
    else:
        hf_api.upload_folder(repo_id=hf_repo, folder_path=model_path, path_in_repo='.', repo_type='model')

    print('Pushed to hub.')


def run():
    parser = argparse.ArgumentParser(description="Push a model to the Hugging Face Hub")
    parser.add_argument("config_path", help="Path to the configuration YAML file")
    parser.add_argument("hf_repo", help="Name of the Hugging Face repository to push to")
    parser.add_argument("--quantize", help="Type of quantization to use", default=None)
    parser.add_argument("--hf_token", help="Hugging Face token (overrides config token)", default=None)
    
    args = parser.parse_args()
    share(args.config_path, args.hf_repo, args.quantize, args.hf_token)