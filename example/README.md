## Overview

This example demonstrates how to fine-tune a model using kolibrify, incorporating a curriculum-based approach. It uses two smaller subsets of the Dolphin dataset, kindly provided by the cognitivecomputations team. Each subset contains 1,000 samples: one with GPT-3 responses and the other with GPT-4 responses.

### How to Run This Example

Follow these steps to get started:

1. Navigate to the example directory with `cd`.
2. Execute `kolibrify-train --config_path dolphin-mistral-test.yaml`.
   - If your system is equipped with multiple GPUs, use `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 kolibrify-train dolphin-mistral-test.yaml`. Substitute `CUDA_VISIBLE_DEVICES` with the correct GPU ID if 0 is in use.
   - To run the training in the background, use `nohup CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 kolibrify-train dolphin-mistral-test.yaml > dolphin-mistral-test.log 2>&1 &`. This command logs all outputs to `dolphin-mistral-test.log` in the current directory.
3. After training, load the model with unsloth as usual. Currently, kolibrify does not support uploading adapters directly to Hugging Face; you'll need to handle this step on your own.
4. To integrate LoRA adapters with the base model, execute `kolibrify-merge --config_path dolphin-mistral-test.yaml`. Include the CUDA environment variables from above if your system has multiple GPUs.
5. You're all set! The model is now ready for use.

> [!NOTE]
> During the merging process in step 4, kolibrify integrates adapters into a 16fp model rather than a 4bit unquantized one, as this has shown to enhance performance downstream. This is the default merging method for any model.

### Explanation of Parameters

Here's a quick guide to some key parameters in the `dolphin-mistral-test.yaml` config file. Parameters like `lora_r` and `learning_rate` are self-explanatory.

#### `stages`

This parameter defines the 'training stages'. These stages represent different dataset mixtures that are processed sequentially by the model.

For the current configuration, the model first trains on "dolphin_gpt3_small.jsonl" for two epochs without mixing with "dolphin_gpt4_small.jsonl". Then, it trains on "dolphin_gpt4_small.jsonl", similar to the training methodology of the Orca model as described in [this paper](https://arxiv.org/pdf/2306.02707). However, unlike the paper's approach, kolibrify maintains gradient information across training stages, treating the data as a continuous dataset.

> [!NOTE]
> You can add multiple datasets within a single stage, they will be mixed together.

> [!NOTE]
> You can add as many stages as you need.

#### `add_imstart_token`

By default, kolibrify trains models using the ChatML template. Setting `add_imstart_token` to True adds the `<|im_start|>` token to the tokenizer's vocabulary and resizes the model's embedding matrices accordingly. This inclusion activates sequence masking in user prompts, a common training technique for instruction-following large language models. For more details on this training method, refer to the [Hugging Face documentation on SFT trainer](https://huggingface.co/docs/trl/en/sft_trainer#train-on-completions-only).