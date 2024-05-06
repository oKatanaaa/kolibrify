## Overview

This example demonstrates how to use `kolibrify-predict` to generate predictions using the Kolibrify model. It utilizes a configuration file and a dataset for input and outputs the predictions to a specified file.

### How to Run This Example

Follow these steps to get started:

1. Ensure you have the necessary files ready:
   - `config.yaml`: This is the configuration file specifying the model settings. It is assummed that the model has already been trained and merged.
   - `test.jsonl`: This is the input dataset for which you want to generate predictions.

2. Execute the following command in your terminal:
   ```
   kolibrify-predict config.yaml test.jsonl output.jsonl
   ```
    - To run prediction on a particular device run `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 kolibrify-train dolphin-mistral-test.yaml > dolphin-mistral-test.log 2>&1 &`.

### Explanation of Parameters

Here's an explanation of the parameters used in the `kolibrify-predict` command:

- `config_path`: Specifies the path to the configuration file (`config.yaml`). This file contains settings for the model.
- `dataset_path`: Specifies the path to the input dataset (`test.jsonl`) containing samples for which predictions will be generated.
- `dataset_save_path` (optional): Specifies the path to save the output predictions. If not provided, the default path is `output.jsonl`.
- `backend` (optional): Specifies the backend to use for prediction. Default is `vllm`.
- `type` (optional): Specifies the type of prediction to generate. Default is `last`.
- `temp` (optional): Specifies the temperature for sampling. Default is `0`.
- `top_p` (optional): Specifies the top cumulative probability to sample from. Default is `0.95`.
- `max_output_tokens` (optional): Specifies the maximum number of tokens in the output. Default is `4096`.

Adjust these parameters according to your specific use case and requirements.
