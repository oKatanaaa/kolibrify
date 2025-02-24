## README for Model Evaluation

### Overview

This README provides instructions on how to evaluate the Kolibrify model using Instruction-Following Eval. This evaluation script can handle both English and Russian datasets, and it provides an interface to specify various parameters for customization.

### How to Run Evaluation

To evaluate the model, follow these steps:

1. **Prepare Configuration File:**
   - Ensure you have the `config.yaml` file ready. This configuration file should specify the settings used during model training. The model must be already merged.

2. **Execute the Evaluation Command:**
   - Run the following command in your terminal to start the model evaluation:
     ```
     kolibrify-eval-ifeval config.yaml --eval-lang en
     ```
   - Replace `en` with `ru` if you wish to evaluate the model on the Russian IFEval.

### Explanation of Command Parameters

- `config_path`: Path to the configuration file (`config.yaml`), which contains model settings.
- `eval_lang`: Language of the evaluation dataset (`en` for English, `ru` for Russian). Default is `en`.
- `backend` (optional): Backend used for model inference. Default is `vllm`.
- `type` (optional): Type of prediction to generate. Currently ignored.
- `temp` (optional): Temperature for sampling. Default is `0`.
- `top_p` (optional): Top cumulative probability to sample from. Default is `0.95`.
- `max_output_tokens` (optional): Maximum number of tokens in the output. Default is `4096`.
- `gpus` (optional): Specifies which GPUs to use for data-parallel inference. Default is `0`.

### Output

The evaluation script will save the evaluation results into the model's folder (specified as `output_dir` in the config) by creating  a separate subfolder containing eval results:
- `en_ifeval_results` if English was selected.
- `ru_ifeval_results` if Russian was selected.

### Additional Notes

- Ensure that the model is properly merged as specified in `config.yaml` before running the evaluation.
- The script supports multi-GPU configuration for faster processing. Ensure that the appropriate GPU indices are specified if multiple GPUs are available.
- Evaluation takes several minutes on a single RTX 3090 GPU.

## Acknowledgements

Big thanks to [IFEval](https://github.com/google-research/google-research/tree/master/instruction_following_eval) and [ruIFEval](https://github.com/NLP-Core-Team/ruIFEval) for their work.