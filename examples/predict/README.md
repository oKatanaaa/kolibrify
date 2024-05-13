## Overview

This example demonstrates how to use `kolibrify-predict` to generate predictions using the Kolibrify model. It supports predictions for multiple datasets stored in a folder and features multi-GPU data-parallel inference to accelerate processing.

### How to Run This Example

Follow these steps to get started:

1. Ensure you have the necessary files ready:
   - `config.yaml`: This is the configuration file specifying the model settings. It is assummed that the model has already been trained and merged.
   - `test.jsonl`: This is the input dataset for which you want to generate predictions.

2. Execute the following command in your terminal:
   ```
   kolibrify-predict config.yaml test.jsonl output.jsonl
   ```

### Explanation of Parameters

Here's an explanation of the parameters used in the `kolibrify-predict` command:

- `config_path`: Specifies the path to the configuration file (`config.yaml`). This file contains settings for the model.
- `dataset_path`: Specifies the path to the input dataset (`test.jsonl` or a directory containing multiple `.jsonl` files) containing samples for which predictions will be generated.
- `dataset_save_path` (optional): Specifies the path to save the output predictions. If not provided, the default path is `output.jsonl`. Ensure it's a directory if the input is also a directory.
- `backend` (optional): Specifies the backend to use for prediction. Default is `vllm`.
- `type` (optional): Currently supports `last` type, though not all types are supported at the moment.
- `temp` (optional): Specifies the temperature for sampling. Default is `0`.
- `top_p` (optional): Specifies the top cumulative probability to sample from. Default is `0.95`.
- `max_output_tokens` (optional): Specifies the maximum number of tokens in the output. Default is `4096`.
- `gpus` (optional): Comma separated GPU ids to use for data-parallel inference. Default is `0`.

> [!NOTE]
> vLLM will use ctx_len from the config as maximum context lenght.

> [!NOTE]
> Inference is always in 16bit.

Adjust these parameters according to your specific use case and requirements.

### Dataset Format

Ensure that your dataset is structured in the OpenAI JSONL format, with each file containing a list of dicts in the following format:
```
{"messages": [{"role": "user", "content": "your query here"}, ...]}
```
If the dataset contains an assistant reply as the last message, it will be skipped, and a new reply will be generated.