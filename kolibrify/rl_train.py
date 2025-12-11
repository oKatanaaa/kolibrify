import argparse
import copy
import os
import pathlib
import torch
import yaml

from .runtime_env import prepare_unsloth_environment

prepare_unsloth_environment()

import unsloth  # noqa: F401
from trl import GRPOConfig, GRPOTrainer

from .common_training import apply_lora_adapter, ensure_c_compiler, setup_seeds
from .core import cpu_offload_embeddings, free_mem, get_model
from .rl import build_remote_reward_fn, create_rl_dataset, load_rl_config


def save_config(config_dict: dict, output_dir: str):
    config_to_save = copy.deepcopy(config_dict)
    if isinstance(config_to_save.get("model"), dict):
        config_to_save["model"]["access_token"] = None
    with open(os.path.join(output_dir, "kolibrify-config.yaml"), "w") as f:
        yaml.safe_dump(config_to_save, f, sort_keys=False)
        print("Saved config in the output directory.")


def main(config_path):
    setup_seeds()
    ensure_c_compiler()

    resolved_config_path = pathlib.Path(config_path).expanduser().resolve()
    config_dir = resolved_config_path.parent

    print("Loading configuration...")
    config_dict, config = load_rl_config(str(resolved_config_path))

    output_dir = pathlib.Path(config.paths.output_dir)
    if not output_dir.is_absolute():
        output_dir = (config_dir / output_dir).resolve()
    config.paths.output_dir = str(output_dir)
    config_dict.setdefault("paths", {})["output_dir"] = str(output_dir)

    print(config)

    print(f"Creating output directory: {config.paths.output_dir}")
    os.makedirs(config.paths.output_dir, exist_ok=True)
    save_config(config_dict, config.paths.output_dir)

    print("Loading model...")
    device_map = {"": torch.cuda.current_device()} if torch.cuda.is_available() else "cpu"

    model, tokenizer = get_model(
        model_name=config.model.base_model,
        max_seq_length=config.model.max_seq_length,
        hf_token=config.model.access_token,
        load_in_4bit=config.model.load_in_4bit,
        add_imstart_token=config.model.add_imstart_token,
        map_eos=config.model.map_eos_to_imend,
        new_tokens=config.model.custom_tokens,
        device_map=device_map,
    )
    free_mem()
    print("Model loaded.")

    model = apply_lora_adapter(
        model,
        config.model,
        gradient_checkpointing=config.model.gradient_checkpointing,
    )

    target_device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(target_device)

    if config.model.cpu_offload_embeddings:
        cpu_offload_embeddings(model, config.model)
        free_mem()

    model.print_trainable_parameters()

    print("Building RL dataset...")
    rl_dataset = create_rl_dataset(
        server_url=config.data.server_url,
        per_device_batch_size=config.training.per_device_train_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        num_generations=config.training.num_generations,
        max_retries=config.data.max_retries,
        retry_backoff_seconds=config.data.retry_backoff_seconds,
        request_timeout_seconds=config.data.request_timeout_seconds,
        meta_timeout_seconds=config.data.meta_timeout_seconds,
    )

    print("Preparing reward function...")
    reward_fn = build_remote_reward_fn(
        config.data.server_url,
        request_timeout_seconds=config.data.request_timeout_seconds,
        max_retries=config.data.max_retries,
        retry_backoff_seconds=config.data.retry_backoff_seconds,
        tokenizer=tokenizer,
    )

    importance_sampling_level = "token"
    if config.rl.rl_algorithm.lower() == "gspo":
        importance_sampling_level = "sequence"

    print("Creating GRPO config...")
    shuffle_dataset = False  # Iteration mapping assumes deterministic ordering of samples.
    assert shuffle_dataset is False, "RL dataserver iteration mapping relies on shuffle_dataset=False"
    training_args = GRPOConfig(
        output_dir=config.paths.output_dir,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        num_generations=config.training.num_generations,
        temperature=config.training.temperature,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        warmup_ratio=config.training.warmup_ratio,
        lr_scheduler_type=config.training.lr_scheduler_type,
        optim=config.training.optim,
        logging_steps=config.training.logging_steps,
        save_steps=config.training.save_steps,
        report_to=config.training.report_to,
        max_prompt_length=config.training.max_prompt_length,
        max_completion_length=config.training.max_completion_length,
        max_steps=config.training.max_steps,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        gradient_checkpointing=bool(config.model.gradient_checkpointing),
        importance_sampling_level=importance_sampling_level,
        shuffle_dataset=shuffle_dataset,
        num_iterations=config.rl.num_iterations,
        epsilon=config.rl.epsilon,
        scale_rewards=config.rl.scale_rewards,
        loss_type=config.rl.loss_type,
    )

    print("Initializing GRPO Trainer...")
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_fn],
        args=training_args,
        train_dataset=rl_dataset,
    )

    if config.model.checkpoint is not None:
        print(f"Starting from checkpoint: {config.model.checkpoint}")

    print("Beginning RL training...")
    trainer.train(resume_from_checkpoint=config.model.checkpoint)
    print("Finished training")

    print("Saving model...")
    trainer.model.save_pretrained(config.paths.output_dir)
    if config.model.merge:
        trainer.model.save_pretrained_merged(
            os.path.join(config.paths.output_dir, "merged"),
            tokenizer,
            save_method="merged_16bit",
        )
    print("Model saved successfully.")


def run():
    parser = argparse.ArgumentParser(description="Run RL training (GRPO/GSPO)")
    parser.add_argument("config_path", help="Path to the RL configuration YAML file")
    args = parser.parse_args()

    main(args.config_path)
