from tokenizers import Tokenizer
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from peft import PeftConfig
import gc
import torch

from .core.config import BaseConfig


def free_mem():
    gc.collect()
    torch.cuda.empty_cache()


def get_source_vocab_size(model_name, loading_lora, token):
    if loading_lora:
        peft_config = PeftConfig.from_pretrained(model_name, token = token)
        _model_name = peft_config.base_model_name_or_path
        _tokenizer = Tokenizer.from_pretrained(_model_name)
        return _tokenizer.get_vocab_size()

    _tokenizer = Tokenizer.from_pretrained(model_name)
    return _tokenizer.get_vocab_size()


def determine_new_vocab_size(model_name, token, loading_lora, add_imstart_token, map_eos):
    basic_modifier = int(add_imstart_token) + int(not map_eos)
    source_vocab_size = get_source_vocab_size(model_name, loading_lora, token)
    print(f'Source vocab size: {source_vocab_size}')
    resize_model_vocab = None
    
    if basic_modifier > 0:
        resize_model_vocab = source_vocab_size + basic_modifier
    
    return resize_model_vocab


def get_model(
    model_name, load_in_4bit=True, 
    max_seq_length=4096, device_map='auto', 
    add_imstart_token=False, map_eos=True,
    token=None, loading_lora=False
):
    resize_model_vocab = determine_new_vocab_size(
        model_name, token, loading_lora, add_imstart_token, map_eos=map_eos
    )
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        load_in_4bit=load_in_4bit,
        max_seq_length=max_seq_length,
        device_map=device_map,
        token=token,
        resize_model_vocab=resize_model_vocab
    )
    
    tokenizer = update_tokenizer(tokenizer, add_imstart_token, map_eos)
    return model, tokenizer


def update_tokenizer(tokenizer, add_imstart_token, map_eos):
    if add_imstart_token:
        tokenizer.add_special_tokens({'additional_special_tokens': ['<|im_start|>']})
    
    if not map_eos:
        print('Not mapping eos token, adding a new one.')
        tokenizer.add_special_tokens({'eos_token': '<|im_end|>'})

    # Make sure pad token is not the same as eos token
    if tokenizer.pad_token_id == tokenizer.eos_token_id:
        print('Pad token is the same as eos token.')
        print('Updating pad token to unk token.')
        tokenizer.pad_token_id = tokenizer.unk_token_id

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
    
    