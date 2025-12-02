import argparse
import os
import torch

from .runtime_env import prepare_unsloth_environment

prepare_unsloth_environment()

import unsloth
from trl import SFTTrainer, SFTConfig

from .common_training import (
    setup_seeds, common_training_setup, run_training
)
from .core import save_config
from .sft import load_dataset, load_training_config, get_data_collator


def main(config_path):
    print('start training')
    # Set random seeds
    setup_seeds()
    
    # Load configuration
    print("Loading configuration...")
    config_dict, config = load_training_config(config_path)
    print(config)
    
    # Create output directory and save config
    print(f"Creating output directory: {config.output_dir}")
    os.makedirs(config.output_dir, exist_ok=True)
    save_config(config_dict)
    
    # Setup common training components
    model, tokenizer, train_data, val_data, data_iterations = common_training_setup(
        config_dict=config_dict,
        config=config,
        load_dataset_fn=load_dataset
    )
    
    # Get data collator (SFT-specific)
    print("Setting up data collator...")
    collator = get_data_collator(tokenizer, mask_assistant_responses=config.add_imstart_token)
    
    # Calculate training steps
    total_batch_size = config.micro_batch_size * config.gradient_accumulation_steps
    training_steps = config.stages[-1].until_step
    expected_iterations = training_steps * total_batch_size
    if data_iterations != expected_iterations:
        print(f"WARNING: Data iterations ({data_iterations}) do not match expected iterations ({expected_iterations}).")
    print(f'Total training steps: {training_steps}')
    
    # Create SFT-specific training arguments
    print("Creating SFT training config...")
    training_args = SFTConfig(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.micro_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_steps=config.warmup_steps,
        max_steps=training_steps,
        max_grad_norm=config.max_grad_norm,
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler_type,
        lr_scheduler_kwargs=config.lr_scheduler_kwargs,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=config.logging_steps,
        optim="adamw_8bit",
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-5,
        gradient_checkpointing=True,
        eval_strategy="steps" if val_data is not None else "no",
        save_strategy="steps",
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        report_to="none",
        # SFT specific parameters
        dataset_text_field="prompt",
        packing=False,
        max_seq_length=config.max_ctx_len,
        dataset_kwargs={"skip_prepare_dataset": True}
    )
    print('col names', train_data.column_names)
    
    # --- Start training with SFT-specific parameters
    print("Initializing SFT Trainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=training_args,
        data_collator=collator,
    )
    
    # Run the training process
    run_training(trainer, config, tokenizer=tokenizer)

    if config.merge:
        model.save_pretrained_merged(os.path.join(config.output_dir, "merged"), tokenizer, save_method = "merged_16bit")


def run():
    parser = argparse.ArgumentParser(description="Run supervised fine-tuning")
    parser.add_argument("config_path", help="Path to the configuration YAML file")
    args = parser.parse_args()
    
    main(args.config_path)
