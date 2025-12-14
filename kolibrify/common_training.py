# kolibrify/common_training.py
import os
import torch
import random
import transformers
import logging
import shutil

from .core import get_model, free_mem, cpu_offload_embeddings
from .core import save_config

def setup_seeds():
    """Set random seeds for reproducibility."""
    torch.manual_seed(0)
    random.seed(0)

def ensure_c_compiler():
    """
    Triton needs a working C compiler. Prefer the conda toolchain if GCC/CC
    is not already available in PATH.
    """
    if os.environ.get("CC") or shutil.which("gcc") or shutil.which("cc"):
        return

    conda_prefix = os.environ.get("CONDA_PREFIX") or os.environ.get("MAMBA_ROOT_PREFIX") or "/opt/conda"
    if conda_prefix is None:
        return

    cc_candidate = os.path.join(conda_prefix, "bin", "x86_64-conda-linux-gnu-gcc")
    cxx_candidate = os.path.join(conda_prefix, "bin", "x86_64-conda-linux-gnu-g++")

    if os.path.exists(cc_candidate):
        os.environ["CC"] = cc_candidate
    if os.path.exists(cxx_candidate):
        os.environ.setdefault("CXX", cxx_candidate)

def common_training_setup(
    config_dict, 
    config, 
    load_dataset_fn
):
    """
    Common training setup used by both SFT and DPO training processes.
    
    Args:
        config_dict: Raw config dictionary
        config: Processed config object
        load_dataset_fn: Function to load the dataset
        
    Returns:
        model, tokenizer, train_data, val_data, data_iterations
    """
    ensure_c_compiler()

    # --- Load model
    print("Loading model...")
    model, tokenizer = get_model(
        model_name=config.model, 
        max_seq_length=config.max_ctx_len,
        hf_token=config.access_token,
        load_in_4bit=config.load_in_4bit,
        add_imstart_token=config.add_imstart_token,
        map_eos=config.map_eos_to_imend,
        new_tokens=config.custom_tokens
    )
    free_mem()
    print("Model loaded.")
    
    # --- Load dataset
    print("Loading dataset...")
    train_data, val_data, data_iterations = load_dataset_fn(
        stages=config.stages, 
        val_dataset_path=config.val_dataset_file,
        tokenizer=tokenizer,
        config=config
    )
    print(f'Total data iterations: {data_iterations}')

    # --- Setup LoRA adapter if not continuing training
    if not config.continued_training:
        model = apply_lora_adapter(model, config)
        
        if config.cpu_offload_embeddings:
            cpu_offload_embeddings(model, config)
            free_mem()
            
    model.print_trainable_parameters()
    
    return model, tokenizer, train_data, val_data, data_iterations

def apply_lora_adapter(model, config, gradient_checkpointing: str | bool = "unsloth"):
    """Apply LoRA adapter to model with given configuration."""
    from unsloth import FastLanguageModel

    print("Applying LoRA adapter...")
    tied_embeddings = bool(getattr(getattr(model, "config", None), "tie_word_embeddings", False))
    return FastLanguageModel.get_peft_model(
        model,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        modules_to_save=config.modules_to_save,
        lora_dropout=config.lora_dropout,
        bias="none",
        use_gradient_checkpointing=gradient_checkpointing,
        max_seq_length=config.max_ctx_len,
        random_state=322,
        use_rslora=config.use_rslora,
        ensure_weight_tying=tied_embeddings,
    )

def run_training(trainer, config, tokenizer=None):
    """Run the training process with the configured trainer."""
    if config.checkpoint is not None:
        print(f'Starting from checkpoint: {config.checkpoint}')
        
    print("Beginning training...")
    trainer.train(resume_from_checkpoint=config.checkpoint)
    print('Finished training')
    
    print("Saving model...")
    # --- Save
    trainer.model.save_pretrained(config.output_dir)
    if tokenizer is not None:
        tokenizer.save_pretrained(config.output_dir)
    print("Model saved successfully.")
