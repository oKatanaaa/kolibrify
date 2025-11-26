from trl.trainer.sft_trainer import DataCollatorForLanguageModeling


def get_data_collator(tokenizer, mask_assistant_responses=True):
    if mask_assistant_responses:
        print('Loss for user responses will be masked during training.')
    else:
        print("Loss for user responses won't be masked during training.")

    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        pad_token_id=tokenizer.pad_token_id,
        completion_only_loss=mask_assistant_responses,
    )
