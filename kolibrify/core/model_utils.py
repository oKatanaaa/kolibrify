import gc
import os
import torch
from peft import PeftConfig
from transformers import AutoTokenizer
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

from .config import BaseConfig


def free_mem():
    gc.collect()
    torch.cuda.empty_cache()


# Simplify the shit belows
def _load_tokenizer(model_name_or_path, hf_token=None):
    """
    Prefer a local tokenizer next to the provided path (e.g., adapter dir),
    otherwise fall back to loading from the model name / base path.
    """
    if os.path.isdir(model_name_or_path):
        # If the path is a directory, try to load tokenizer from it directly.
        try:
            return AutoTokenizer.from_pretrained(model_name_or_path, token=hf_token)
        except Exception:
            pass
    return AutoTokenizer.from_pretrained(model_name_or_path, token=hf_token)


def get_source_vocab_size(model_name, loading_lora, hf_token):
    if loading_lora:
        # When loading a LoRA adapter, prefer tokenizer inside the adapter path if present.
        try:
            tok = _load_tokenizer(model_name, hf_token)
            return len(tok.get_vocab())
        except Exception:
            peft_config = PeftConfig.from_pretrained(model_name, token=hf_token)
            _model_name = peft_config.base_model_name_or_path
            tok = _load_tokenizer(_model_name, hf_token)
            return len(tok.get_vocab())

    tok = _load_tokenizer(model_name, hf_token)
    return len(tok.get_vocab())


def vocab_has_imend(model_name, loading_lora, hf_token):
    if loading_lora:
        try:
            tok = _load_tokenizer(model_name, hf_token)
        except Exception:
            peft_config = PeftConfig.from_pretrained(model_name, token=hf_token)
            _model_name = peft_config.base_model_name_or_path
            tok = _load_tokenizer(_model_name, hf_token)
        return '<|im_end|>' in tok.get_vocab()

    tok = _load_tokenizer(model_name, hf_token)
    return '<|im_end|>' in tok.get_vocab()


def determine_new_vocab_size(model_name: str, hf_token: str, loading_lora: bool,
                             add_imstart_token: bool, map_eos: bool, new_tokens: list):
    # Always anchor on the tokenizer that lives next to the provided path (adapter or base).
    tok = _load_tokenizer(model_name, hf_token)
    existing_vocab = tok.get_vocab()

    # When loading a LoRA adapter, ensure the base model is resized exactly to this tokenizer size
    # (to avoid size mismatches when merging). Do not attempt incremental adds; the adapter already
    # knows its vocabulary.
    if loading_lora:
        source_vocab_size = len(existing_vocab)
        print(f'Source vocab size: {source_vocab_size}')
        return source_vocab_size

    # For base loads, only add truly missing tokens.
    missing = 0
    if add_imstart_token and '<|im_start|>' not in existing_vocab:
        missing += 1

    if new_tokens is not None:
        for t in new_tokens:
            if t not in existing_vocab:
                missing += 1

    if not map_eos and '<|im_end|>' not in existing_vocab:
        missing += 1

    source_vocab_size = len(existing_vocab)
    print(f'Source vocab size: {source_vocab_size}')

    if missing == 0:
        return None
    return source_vocab_size + missing


def get_model(
    model_name, load_in_4bit=True, 
    max_seq_length=4096, device_map='auto', 
    add_imstart_token=False, map_eos=True,
    hf_token=None, loading_lora=False, new_tokens=None
):
    resize_model_vocab = determine_new_vocab_size(
        model_name, hf_token, loading_lora, add_imstart_token, map_eos=map_eos, new_tokens=new_tokens
    )
    if resize_model_vocab is None:
        print("Vocab won't be resized.")
    else:
        print("Model vocab will be resized to", resize_model_vocab)
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        load_in_4bit=load_in_4bit,
        max_seq_length=max_seq_length,
        device_map=device_map,
        token=hf_token,
        resize_model_vocab=resize_model_vocab
    )
    
    tokenizer = update_tokenizer(tokenizer, add_imstart_token, map_eos, new_tokens)
    _update_model_eos_id(model, tokenizer)

    return model, tokenizer


def update_tokenizer(tokenizer, add_imstart_token, map_eos, new_tokens):
    print('map_eos', map_eos)
    additional_special_tokens = []
    if add_imstart_token:
        additional_special_tokens.append('<|im_start|>')
    
    if new_tokens is not None:
        additional_special_tokens.extend(new_tokens)
    
    if len(additional_special_tokens) > 0:
        tokenizer.add_special_tokens({'additional_special_tokens': additional_special_tokens})
    
    if not map_eos:
        print('Not mapping eos token, adding a new one.')
        if tokenizer.get_vocab().get('<|im_end|>') is None:
            tokenizer.add_special_tokens({'eos_token': '<|im_end|>'})
        else:
            print('Eos token already exists.')

    # Make sure pad token is not the same as eos token
    if tokenizer.pad_token_id == tokenizer.eos_token_id:
        print('Pad token is the same as eos token.')
        print('Updating pad token to unk token.')
        tokenizer.pad_token_id = tokenizer.unk_token_id
    elif tokenizer.pad_token_id is None:
        print('Pad token is not set. Falling back to eos token for padding.')
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if tokenizer.padding_side == 'left':
        print('Padding side is left.')
        print('Updating padding side to right.')
        tokenizer.padding_side = 'right'

    tokenizer = get_chat_template(
        tokenizer,
        chat_template="chatml",
        map_eos_token=map_eos
    )

    print(f'Updated tokenizer. Vocab len: {len(tokenizer)}')
    return tokenizer


def _update_model_eos_id(model, tokenizer):
    model.config.eos_token_id = tokenizer.get_vocab()['<|im_end|>']
    model.generation_config.eos_token_id = tokenizer.get_vocab()['<|im_end|>']


def to_cuda_wrapper(method):
    def fn(indices):
        tensor = method(indices.cpu())
        return tensor.cuda()
    return fn


def cpu_offload_embeddings(lora_model, config: BaseConfig):
    # [NOTE] Trainable embeddings and lm_head won't be offloaded since it is not 
    # compatible with 8bit optimizers
    
    # Offload the new embedding layer that is used during training
    # lora_model.base_model.model.model.embed_tokens.modules_to_save.default = \
    #    lora_model.base_model.model.model.embed_tokens.modules_to_save.default.cpu()
    # lora_model.base_model.model.model.embed_tokens.modules_to_save.default.forward = \
    #    to_cuda_wrapper(lora_model.base_model.model.model.embed_tokens.modules_to_save.default.forward)
    
    # Offload the original embedding layer
    if 'embed_tokens' in config.modules_to_save:
        print('Offloaded embed_tokens to CPU.')
        lora_model.base_model.model.model.embed_tokens.original_module = \
            lora_model.base_model.model.model.embed_tokens.original_module.cpu()
    if 'lm_head' in config.modules_to_save:
        print('Offloaded lm_head to CPU.')
        lora_model.base_model.model.lm_head.original_module = \
            lora_model.base_model.model.lm_head.original_module.cpu()
    
    if config.modules_to_save == [] or config.modules_to_save is None:
        print('Nothing to offload.')
    
    
