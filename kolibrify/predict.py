import typer
from typing_extensions import Annotated
from peft import PeftModel
import os
import vllm
import json
from copy import deepcopy

from kolibrify.config import TrainingConfig, load_training_config
from kolibrify.model_utils import get_model
from kolibrify.data_utils import format_chatml


class VllmModel:
    def __init__(self, merged_model_path: str, config: TrainingConfig, temp=0.0, top_p=0.95, max_tokens=4096):
        self.vllm_model = vllm.LLM(merged_model_path, max_model_len=max_tokens)
        self.sampling_params = vllm.SamplingParams(temperature=temp, top_p=top_p, max_tokens=max_tokens)

    def predict(self, convs):
        prompts = format_prompts_vllm(convs)
        responses = self.vllm_model.generate(prompts=prompts, sampling_params=self.sampling_params)
        # Extract responses
        openai_responses = []
        for r in responses:
            openai_response = {
                'role': 'assistant',
                'content': r.outputs[0].text
            }
            openai_responses.append(openai_response)
        return openai_responses
        
        

def load_dataset(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    lines = list(map(json.loads, lines))

    # Remove assistant (or anything that's not user) responses
    for line in lines:
        msgs = line['messages']
        if msgs[-1]['role'] != 'user':
            line['messages'] = msgs[:-1]
    
    return lines


def save_dataset(conversations, path):
    with open(path, 'w', encoding='utf-8') as f:
        for conv in conversations:
            f.write(json.dumps(conv) + '\n')


def format_prompts_vllm(messages):
    prompts = []
    for conv in messages:
        prompt = format_chatml(conv)['prompt']
        # Add prefix for assistant response
        prompt += '\n<|im_start|>assistant\n'
        prompts.append(prompt)
    return prompts


def predict(
    config: TrainingConfig,
    conversations: list,
    backend: str = 'vllm',
    type: str = 'last',
    temp: float = 0,
    top_p: float = 0.95,
    max_output_tokens: int = 4096
):
    conversations = deepcopy(conversations)
    
    model_path = os.path.join(config.output_dir, 'merged')
    if backend == 'vllm':
        model = VllmModel(model_path, config, temp=temp, top_p=top_p, max_tokens=max_output_tokens)
    else:
        raise ValueError('At the moment only vllm backend is supported.')
    print('Loaded model.')

    responses = model.predict(conversations)
    print('Finished generating responses.')
    
    for conv, response in zip(conversations, responses):
        conv['messages'].append(response)
        
    return conversations


def main(
    config_path: Annotated[str, typer.Argument()],
    dataset_path: Annotated[str, typer.Argument()],
    dataset_save_path: Annotated[str, typer.Argument()] = 'output.jsonl',
    backend: Annotated[str, typer.Option()] = 'vllm',
    type: Annotated[str, typer.Option()] = 'last',
    temp: Annotated[float, typer.Option()] = 0,
    top_p: Annotated[float, typer.Option()] = 0.95,
    max_output_tokens: Annotated[int, typer.Option()] = 4096
):
    _, config = load_training_config(config_path)
    conversations = load_dataset(dataset_path)
    
    conversations = predict(
        config=config,
        conversations=conversations,
        backend=backend,
        type=type,
        temp=temp,
        top_p=top_p,
        max_output_tokens=max_output_tokens
    )
    
    save_dataset(conversations, dataset_save_path)
    print('Saved responses to', dataset_save_path)

def run():
    typer.run(main)