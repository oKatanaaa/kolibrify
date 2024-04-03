import typer
from typing_extensions import Annotated
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

import torch
torch.manual_seed(0)
import random
random.seed(0)

import transformers
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from unsloth import FastLanguageModel
from .data_utils import load_dataset
from .model_utils import get_model
from .config import load_training_config


def main(
    config_path: Annotated[str, typer.Argument()] = "training_config.yaml"
):
    config = load_training_config(config_path)
    print(config)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        transient=True,
    ) as progress:
        # --- Load datasets and model
        task1 = progress.add_task(description="Loading dataset...", total=None)
        train_data, val_data, data_iterations = load_dataset(
            stages=config.stages, 
            val_dataset_path=config.val_dataset_file
        )
        progress.print('Total data iterations:', data_iterations)
        progress.advance(task1)
        
        task2 = progress.add_task(description="Loading model...", total=None)
        model, tokenizer = get_model(
            model_name=config.model, 
            max_seq_length=config.max_ctx_len, 
            do_update_tokenizer=config.update_tokenizer, 
            token=config.access_token,
            load_in_4bit=config.load_in_4bit,
            resize_model_vocab=32001 if config.continued_training else None
        )
        import gc
        import torch

        def free_mem():
            gc.collect()
            torch.cuda.empty_cache()
        free_mem()
        progress.advance(task2)
    
        # --- Setup all training stuff
        task3 = progress.add_task(description="Setting up training...", total=None)
        if not config.continued_training:
            model = FastLanguageModel.get_peft_model(
                model, 
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                target_modules=config.lora_target_modules,
                modules_to_save=config.modules_to_save,
                lora_dropout=config.lora_dropout,
                bias="none",
                use_gradient_checkpointing=True,
                max_seq_length=config.max_ctx_len,
                random_state=322
            )
        model.print_trainable_parameters()
        
        total_batch_size = config.micro_batch_size * config.gradient_accumulation_steps
        training_steps = data_iterations // total_batch_size
        progress.print('Total training steps:', training_steps)
        
        training_arguments = transformers.TrainingArguments(
            per_device_train_batch_size=config.micro_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            warmup_steps=config.warmup_steps,
            max_steps=training_steps,
            max_grad_norm=1.0,
            learning_rate=config.learning_rate,
            lr_scheduler_type=config.lr_scheduler_type,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps=config.logging_steps,
            optim="adamw_8bit",
            adam_beta1=0.9,
            adam_beta2=0.95,
            adam_epsilon=0.00001,
            evaluation_strategy="steps" if val_data is not None else "no",
            save_strategy="steps",
            eval_steps=config.eval_steps,
            save_steps=config.save_steps,
            output_dir=config.output_dir,
            save_total_limit=config.save_total_limit,
            report_to="tensorboard"
        )
        instruction_template = "<|im_start|>user"
        response_template = "<|im_start|>assistant"
        collator = DataCollatorForCompletionOnlyLM(
            instruction_template=instruction_template, 
            response_template=response_template, 
            tokenizer=tokenizer, 
            mlm=False
        )
        progress.advance(task3)
    
    # --- Start training
    print("Start training")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_data,
        eval_dataset=val_data,
        dataset_text_field='prompt',
        packing=False,
        max_seq_length=config.max_ctx_len,
        args=training_arguments,
        data_collator=collator
    )
    if config.checkpoint is not None:
        print(f'Starting from checkpoint: {config.checkpoint}')
        
    trainer.train(resume_from_checkpoint=config.checkpoint)
    print('Finished training')
    # --- Save
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

def run():
    typer.run(main)

    