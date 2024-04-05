from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from transformers import AutoModelForCausalLM, AutoTokenizer
from tokenizers import Tokenizer
from peft import PeftConfig


def get_model(
    model_name, load_in_4bit=True, 
    max_seq_length=4096, device_map='auto', 
    do_update_tokenizer=False, token=None, resize_model_vocab=None
):
    #if resize_model_vocab is not None:
    # peft_config = PeftConfig.from_pretrained(model_name, token = token)
    # _model_name = peft_config.base_model_name_or_path
    if resize_model_vocab is None:
        _tokenizer = Tokenizer.from_pretrained(model_name)
        resize_model_vocab = _tokenizer.get_vocab_size() + 1
        
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        load_in_4bit=load_in_4bit,
        max_seq_length=max_seq_length,
        device_map=device_map,
        token=token,
        resize_model_vocab=resize_model_vocab
    )
    tokenizer = update_tokenizer(tokenizer)
    # if resize_model_vocab is None:
    #     model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer


def update_tokenizer(tokenizer):
    tokenizer.add_special_tokens({'additional_special_tokens': ['<|im_start|>']})
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="chatml",
        map_eos_token=True
    )
    print(f'Updated tokenizer. New vocab len: {len(tokenizer)}')
    return tokenizer