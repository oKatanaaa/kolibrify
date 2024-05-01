# Kolibrify

## Overview

Kolibrify is a lightweight 'library' (just a couple of simple scripts) designed for curriculum and general fine-tuning of large language models (LLMs) for instruction following. Building on the premise of ChatML format, popularized by OpenAI, it allows for easy compatibility with existing serving frameworks. Key features include support for LoRA/QLoRA fine-tuning, datasets mixing and multi-stage continual training. Kolibrify aims to streamline the process of adapting LLMs to specific tasks or datasets employing curriculum learning for best results.

Kolibrify leverages the power of [Unsloth](https://github.com/unslothai/unsloth) for accelerated model training and reduced memory usage, making it a practical choice for developers looking to fine-tune LLMs with limited compute.

> [!NOTE]
> This project serves as a foundation for my research on curriculum learning. It has a unique setup in that it doesn't directly load datasets from Hugging Face, instead relying on local files. As a result, you'll need to download the datasets and save them on your machine to use Kolibrify. I designed it this way to facilitate rapid prototyping, as frequent data processing and pushing datasets to the hub slows down progress. By default, it assumes the use of the ChatML template, although I plan to add support for other templates in the near future.

## Installation

1. `git clone https://github.com/oKatanaaa/kolibrify`
2. `cd kolibrify`
3. `pip install -e .`
4. Done!

## Usage

Kolibrify is equipped with two primary scripts for training and merging fine-tuned models: `kolibrify-train` and `kolibrify-merge`. To run those you have to make a YAML configuration file based on the `training_config_template.yaml`, tailored to your project's needs.

### Training

To start training, ensure your dataset is in the JSONL format specified, with records like `{"messages": [{"role": "role", "content": "content"}]}`. Adjust `training_config.yaml` as necessary, specifying model and dataset paths among other parameters.

```bash
kolibrify-train training_config.yaml
```

- `training_config.yaml`: Path to your training configuration file, detailing model specifications, dataset locations, and training parameters.

### Merging LoRA Parameters

After training, use the merge script to incorporate the fine-tuned LoRA parameters back into the base model. This step is necessary for deploying your fine-tuned model with vLLM (and other serving frameworks).

```bash
kolibrify-merge training_config.yaml
```

- `training_config.yaml`: Path to your training configuration file used during the fine-tuning process.

The model will be saved in the `merged` folder where your adapter was saved.

> [!NOTE]
> If possible, kindly provide the kolibrify config you used in your model's repo so that others can reproduce your results.

> [!NOTE]
> See `examples` folder for a full example of finetuning Mistral model with Kolibrify on Dolphin dataset.


## Configuration

See `training_config_template.yaml` for a comprehensive list of adjustable parameters tailored to your training and fine-tuning needs. This includes model identifiers, dataset paths, LoRA parameters, training iterations, learning rate, and more, providing a flexible foundation for model adaptation.

> [!NOTE]
> This project is in early development stages and will be updated frequently. If you encounter bugs or would like it to support some specific features, kindly make a corresponding issue. Contributions are welcome.

## Acknowledgements

Huge thanks to the [Unsloth](https://github.com/unslothai/unsloth) team for their amazing project that enabled me to do this research with limited resources at larger scales!
