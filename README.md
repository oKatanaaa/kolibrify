# Kolibrify

## Overview

Kolibrify is a lightweight 'library' (just a couple of simple scripts) designed for curriculum and general fine-tuning of large language models (LLMs) for instruction following. Building on the premise of ChatML format, popularized by OpenAI, it allows for easy compatibility with existing serving frameworks. Key features include support for LoRA/QLoRA fine-tuning, Direct Preference Optimization, datasets mixing and multi-stage continual training. Kolibrify aims to streamline the process of adapting LLMs to specific tasks or datasets employing curriculum learning for best results.

Kolibrify leverages the power of [Unsloth](https://github.com/unslothai/unsloth) for accelerated model training and reduced memory usage, making it a practical choice for developers looking to fine-tune LLMs with limited compute.

> [!NOTE]
> This project serves as a foundation for my research on curriculum learning. It has a unique setup in that it doesn't directly load datasets from Hugging Face, instead relying on local files. As a result, you'll need to download the datasets and save them on your machine to use Kolibrify. I designed it this way to facilitate rapid prototyping, as frequent data processing and pushing datasets to the hub slows down progress. By default, it assumes the use of the ChatML template, although I plan to add support for other templates in the near future.

## Installation

1. `git clone https://github.com/oKatanaaa/kolibrify`
2. `cd kolibrify`
3. `pip install -e .`
4. Done!

## Usage

Kolibrify is equipped with four primary scripts for training, merging and testing fine-tuned models: 
- `kolibrify-sft` - supervised finetuning.
- `kolibrify-dpo` - direct preference optimization.
- `kolibrify-merge` - merging lora adapters.
- `kolibrify-predict` - generating predictions using a finetuned model.
- `kolibrify-eval-ifeval` - evaluating a finetuned model using instruction-following eval.
- `kolibrify-push` - pushing a model to a Huggingface repo.
To run those you have to make a YAML configuration file based on the `training_config_template.yaml`, tailored to your project's needs.

Below is a brief rundown regarding each command. See the `examples` folder for detailed explanations.

### Training

#### Supervised fine-tuning

To start training, ensure your dataset is in the JSONL format specified, with records like `{"messages": [{"role": "role", "content": "content"}]}`. Adjust `training_config.yaml` as necessary, specifying model and dataset paths among other parameters.

```bash
kolibrify-sft training_config.yaml
```

- `training_config.yaml`: Path to your training configuration file, detailing model specifications, dataset locations, and training parameters.

> [!NOTE]
> See `examples` folder for a full example of finetuning Mistral model with Kolibrify on Dolphin dataset.

#### Direct preference optimization

To start training, ensure your dataset is in the JSONL format specified, with records like `{"messages": [{"role": "role", "content": "content"}]}`. Adjust `training_config.yaml` as necessary, specifying model and dataset paths among other parameters.

```bash
kolibrify-dpo training_config.yaml
```

- `training_config.yaml`: Path to your training configuration file, detailing model specifications, dataset locations, and training parameters.

> [!NOTE]
> See `examples` folder for a full example of finetuning Mistral model with Kolibrify on Dolphin dataset.

### Merging LoRA Parameters

After training, use the merge script to incorporate the fine-tuned LoRA parameters back into the base model. This step is necessary for deploying your fine-tuned model with vLLM (and other serving frameworks).

```bash
kolibrify-merge training_config.yaml
```

- `training_config.yaml`: Path to your training configuration file used during the fine-tuning process.

The model will be saved in the `merged` folder where your adapter was saved.

### Prediction

For generating predictions with a fine-tuned model, use `kolibrify-predict`. Specify the path to your configuration, input dataset, and output location for the results:

```bash
kolibrify-predict config.yaml /path/to/dataset /path/to/output --gpus 1,2,3
```

- `config.yaml`: Configuration file used during model training.
- `/path/to/dataset`: Can be a path to a JSONL file or a directory containing multiple JSONL files.
- `/path/to/output`: Output path for saving the model's predictions. If `/path/to/dataset` is a JSONL file, `/path/to/output` must be a JSONL as well. The same goes for a path to a folder with datasets.
- `--gpus`: Comma-separated GPU indices to use for the inference.

### Evaluation

For instruction following evaluation, use `kolibrify-eval-ifeval`. It allows evaluating the model's performance in following instructions in either English or Russian:

```bash
kolibrify-eval-ifeval config.yaml --eval-lang en --gpus 1,2,3
```

- `config.yaml`: Configuration file specifying paths and model details.
- `--eval-lang`: Language for the evaluation (`en` for English, `ru` for Russian).
- `--gpus`: Comma-separated GPU indices to use for the evaluation.

This script will evaluate the model using the Instruction-Following Eval benchmark and save results and logs to the specified output directories.

### Pushing

To a model to Huggingface, use `kolibrify-push`. At the moment you can push either a merged model or its quantized version:
- pushing merged model
```bash
kolibrify-push config.yaml repo-name
```
- pushing quantized model
```bash
kolibrify-push config.yaml repo-name --quantize quant
```

- `config.yaml`: Configuration file specifying paths and model details.
- `repo-name`: Name of the Huggingface repo.
- `--quantize`: Name of the quant to push (e.g. q8_0).
- `--hf_token`: You Huggingface token. By default the token is taken from your config file.

> [!NOTE]
> Do not create the repo manually, it will be created automatically.
> But if you created the repo beforehand, make sure it is completely empty. Otherwise the push will fail.

## Configuration

See `training_config_template.yaml` for a comprehensive list of adjustable parameters tailored to your training and fine-tuning needs. This includes model identifiers, dataset paths, LoRA parameters, training iterations, learning rate, and more, providing a flexible foundation for model adaptation.

> [!NOTE]
> This project is in early development stages and will be updated frequently. If you encounter bugs or would like it to support some specific features, kindly make a corresponding issue. Contributions are welcome.

> [!NOTE]
> If possible, kindly provide the kolibrify config you used in your model's repo so that others can reproduce your results.

## Workflow example

To illustrate how I personally use kolibrify, here is an example workflow:
```bash
# Download necessary datasets and preprocess them
# Make a configuration file config.yaml
kolibrify-sft config.yaml
kolibrify-merge config.yaml
# Do IFEval for both languages
# Use 4 GPUs for 4x evaluation speed up 
kolibrify-eval-ifeval config.yaml --eval-lang en --gpus 0,1,2,3
kolibrify-eval-ifeval config.yaml --eval-lang ru --gpus 0,1,2,3
kolibrify-push config.yaml kaleinaNyan/model
kolibrify-push config.yaml kaleinaNyan/model.gguf --quantize q8_0
```


## Acknowledgements

Huge thanks to the [Unsloth](https://github.com/unslothai/unsloth) team for their amazing project that enabled me to do this research with limited resources at larger scales!
