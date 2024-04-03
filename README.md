# Kolibrify README

## Overview

Kolibrify is a lightweight library designed for curriculum and general fine-tuning of large language models (LLMs). Building on the premise of ChatML format, popularized by OpenAI, it allows for easy compatibility with existing serving frameworks. Key features include support for LoRA (Low-Rank Adaptation) fine-tuning, datasets mixing and multi-stage continual training. Kolibrify aims to streamline the process of adapting LLMs to specific tasks or datasets employing curriculum learning for best results.

Kolibrify leverages the power of Unsloth for accelerated model training and reduced memory usage, making it a practical choice for developers looking to fine-tune LLMs cheaply.

## Usage

Kolibrify is equipped with two primary scripts for training and merging fine-tuned models. Ensure you have a YAML configuration file based on the `training_config_template.yaml` provided, tailored to your project's needs.

### Training

To start training, ensure your dataset is in the JSONL format specified, with records like `{"messages": [{"role": "role", "content": "content"}]}`. Adjust `training_config.yaml` as necessary, specifying model and dataset paths among other parameters.

```bash
kolibrify-train --config_path training_config.yaml
```

#### Arguments Explanation

- `--config_path`: Path to your training configuration file, detailing model specifications, dataset locations, and training parameters.

### Merging LoRA Parameters

After training, use the merge script to incorporate the fine-tuned LoRA parameters back into the base model. This step is crucial for deploying your fine-tuned model effectively.

```bash
kolibrify-merge --config_path training_config.yaml --adapter_path path_to_adapter --base_model your_base_model
```

#### Arguments Explanation

- `--config_path`: Path to your training configuration file used during the fine-tuning process.
- `--adapter_path`: (Optional) Directory where the fine-tuned adapter parameters are saved. Defaults to the `output_dir` specified in your configuration file.
- `--base_model`: (Optional) The identifier of your base model. Defaults to the `model` parameter in your configuration file if not explicitly provided.

## Configuration

See `training_config_template.yaml` for a comprehensive list of adjustable parameters tailored to your training and fine-tuning needs. This includes model identifiers, dataset paths, LoRA parameters, training iterations, learning rate, and more, providing a flexible foundation for model adaptation.