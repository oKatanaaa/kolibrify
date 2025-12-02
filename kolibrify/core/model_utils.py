import gc
import torch
from unsloth import FastLanguageModel, add_new_tokens
from unsloth.chat_templates import get_chat_template
from unsloth.tokenizer_utils import mean_of_trained_tokens

from .config import BaseConfig


def free_mem():
    gc.collect()
    torch.cuda.empty_cache()


# Simplify the shit belows
def get_model(
    model_name, load_in_4bit=True, 
    max_seq_length=4096, device_map='auto', 
    add_imstart_token=False, map_eos=True,
    hf_token=None, loading_lora=False, new_tokens=None
):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        load_in_4bit=load_in_4bit,
        max_seq_length=max_seq_length,
        device_map=device_map,
        token=hf_token,
    )

    tokenizer = update_tokenizer_and_model(
        model,
        tokenizer,
        add_imstart_token=add_imstart_token,
        map_eos=map_eos,
        new_tokens=new_tokens,
    )
    _update_model_eos_id(model, tokenizer)

    return model, tokenizer


def update_tokenizer_and_model(model, tokenizer, add_imstart_token, map_eos, new_tokens):
    vocab = tokenizer.get_vocab()
    tokens_to_add = []
    if add_imstart_token and '<|im_start|>' not in vocab:
        tokens_to_add.append('<|im_start|>')
    if new_tokens:
        tokens_to_add.extend([token for token in new_tokens if token not in vocab])

    if tokens_to_add:
        _add_tokens(model, tokenizer, tokens_to_add)

    if not map_eos and '<|im_end|>' in tokenizer.get_vocab():
        tokenizer.eos_token = '<|im_end|>'

    if tokenizer.pad_token_id == tokenizer.eos_token_id:
        tokenizer.pad_token_id = tokenizer.unk_token_id
    elif tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if tokenizer.padding_side == 'left':
        tokenizer.padding_side = 'right'

    tokenizer = get_chat_template(
        tokenizer,
        chat_template="chatml",
        map_eos_token=map_eos
    )

    return tokenizer


def _update_model_eos_id(model, tokenizer):
    model.config.eos_token_id = tokenizer.get_vocab()['<|im_end|>']
    model.generation_config.eos_token_id = tokenizer.get_vocab()['<|im_end|>']


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
    

def _add_tokens(model, tokenizer, tokens):
    if not tokens:
        return

    if model is None:
        tokenizer.add_tokens(tokens, special_tokens=False)
        return

    embed_rows = model.get_input_embeddings().weight.shape[0]
    vocab_len = len(tokenizer)
    # Qwen3 and similar models ship padded embedding matrices (rows > vocab),
    # so prefer reusing those slots before triggering a resize.
    padding_slots = max(embed_rows - vocab_len, 0)
    tokens_without_resize = tokens[:padding_slots]
    tokens_requiring_resize = tokens[padding_slots:]

    if tokens_without_resize:
        _use_padded_embeddings(model, tokenizer, tokens_without_resize)

    if tokens_requiring_resize:
        _call_add_new_tokens(model, tokenizer, tokens_requiring_resize)


def _call_add_new_tokens(model, tokenizer, tokens):
    if not tokens:
        return

    try:
        add_new_tokens(model, tokenizer, new_tokens=tokens, special_tokens=False)
    except TypeError:
        add_new_tokens(model, tokenizer, new_tokens=tokens)


def _use_padded_embeddings(model, tokenizer, tokens):
    start_idx = len(tokenizer)
    tokenizer.add_tokens(tokens, special_tokens=False)
    end_idx = len(tokenizer)

    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()

    mean_embedding, mean_lm_head = mean_of_trained_tokens(model)
    mean_embedding = mean_embedding.to(torch.float32)
    mean_lm_head = mean_lm_head.to(torch.float32)

    indicator_untrained = torch.amax(input_embeddings.weight, axis=1) <= 1e-16
    trained_mask = ~indicator_untrained

    std_embedding = input_embeddings.weight[trained_mask].to(torch.float32).std(
        dim=0, unbiased=False
    ).clamp_min(1e-6)
    std_lm_head = (
        output_embeddings.weight[trained_mask].to(torch.float32).std(
            dim=0, unbiased=False
        ).clamp_min(1e-6)
        if output_embeddings is not None
        else None
    )

    num_new = end_idx - start_idx

    with torch.no_grad():
        input_embeddings.weight[start_idx:end_idx] = torch.normal(
            mean_embedding.expand(num_new, -1),
            std_embedding.expand(num_new, -1),
        )
        if output_embeddings is not None:
            output_embeddings.weight[start_idx:end_idx] = torch.normal(
                mean_lm_head.expand(num_new, -1),
                std_lm_head.expand(num_new, -1),
            )

    _mark_embeddings_trainable(model)
    _update_vocab_size(model, len(tokenizer))

    if model.config.tie_word_embeddings:
        model.tie_weights()


def _mark_embeddings_trainable(model):
    internal_model = model
    while hasattr(internal_model, "model"):
        internal_model._need_to_train_embeddings = True
        internal_model = internal_model.model
    internal_model._need_to_train_embeddings = True


def _update_vocab_size(model, new_size):
    current_model = model
    while hasattr(current_model, "model") and hasattr(current_model, "config"):
        if hasattr(current_model.config, "vocab_size"):
            current_vocab = getattr(current_model.config, "vocab_size", new_size)
            current_model.config.update({"vocab_size": max(current_vocab, new_size)})
        current_model = current_model.model
    if hasattr(current_model, "config") and hasattr(current_model.config, "vocab_size"):
        current_vocab = getattr(current_model.config, "vocab_size", new_size)
        current_model.config.update({"vocab_size": max(current_vocab, new_size)})
    
