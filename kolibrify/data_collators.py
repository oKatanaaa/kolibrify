from trl import DataCollatorForCompletionOnlyLM
from tokenizers import Tokenizer


def get_data_collator(tokenizer, mask_assistant_responses=True):
    if mask_assistant_responses:
        print('Loss for user responses will be masked during training.')
        return masking_collator(tokenizer)
    
    print("Loss for user responses won't be masked during training.")
    return None


def masking_collator(tokenizer: Tokenizer):
    assert '<|im_start|>' in tokenizer.get_vocab(), \
        'The tokenizer must contain <|im_start|> as a separate token to mask user responses correctly.'
        
    instruction_template = "<|im_start|>user"
    response_template = "<|im_start|>assistant"
    collator = DataCollatorForCompletionOnlyLM(
        instruction_template=instruction_template, 
        response_template=response_template, 
        tokenizer=tokenizer, 
        mlm=False
    )
    return collator
