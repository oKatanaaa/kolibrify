import argparse
import os
import torch

from .runtime_env import prepare_unsloth_environment

prepare_unsloth_environment()

from trl import DPOTrainer, DPOConfig
from unsloth import PatchDPOTrainer
# Apply the patch for DPOTrainer
PatchDPOTrainer()

from .common_training import (
    setup_seeds, common_training_setup, run_training
)
from .core import save_config
from .dpo import load_dataset, load_training_config


def main(config_path):
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
    
    # Calculate training steps
    total_batch_size = config.micro_batch_size * config.gradient_accumulation_steps
    training_steps = data_iterations // total_batch_size
    print(f'Total training steps: {training_steps}')
    
    # Create DPO-specific training arguments
    print("Creating DPO training config...")
    training_args = DPOConfig(
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
        # DPO specific parameters
        max_length=config.max_ctx_len,
        max_prompt_length=config.max_ctx_len,
        max_completion_length=config.max_ctx_len,
        beta=0.1,  # Default DPO beta value
    )
    
    # --- Start training with DPO-specific parameters
    print("Initializing DPO Trainer...")
    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # DPO-specific parameter
        tokenizer=tokenizer,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=training_args,
    )
    
    # Run the training process
    run_training(trainer, config, tokenizer=tokenizer)


def run():
    parser = argparse.ArgumentParser(description="Run direct preference optimization")
    parser.add_argument("config_path", help="Path to the configuration YAML file")
    args = parser.parse_args()
    
    main(args.config_path)
