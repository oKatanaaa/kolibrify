import json
import typer
import os
from typing_extensions import Annotated
from typing import List
import unsloth

from kolibrify.inference import load_model, predict
from kolibrify.core import load_base_config

from ifeval.cli import run_evaluation
from ifeval.utils.config import Config
from ifeval.utils.huggingface import get_default_dataset
from ifeval.core.evaluation import InputExample



def convert_to_oai(data: List[InputExample]):
    conversations = []
    for example in data:
        conversations.append(
            {'messages': [
                {'role': 'user', 'content': example.prompt}
            ]}
        )
    return conversations


def save_responses(conversations, path):
    responses = []
    for conv in conversations:
        msgs = conv['messages']
        responses.append(
            {
                'prompt': msgs[0]['content'],
                'response': msgs[1]['content']
            }
        )
    
    with open(path, 'w', encoding='utf-8') as f:
        for resp in responses:
            f.write(json.dumps(resp, ensure_ascii=False) + '\n')

def main(
    config_path: Annotated[str, typer.Argument()],
    checkpoint: Annotated[str, typer.Option()] = None,
    eval_lang: Annotated[str, typer.Option()] = 'en',
    backend: Annotated[str, typer.Option()] = 'vllm',
    type: Annotated[str, typer.Option()] = 'last',
    temp: Annotated[float, typer.Option()] = 0,
    top_p: Annotated[float, typer.Option()] = 0.95,
    min_p: Annotated[float, typer.Option()] = 0.05,
    max_output_tokens: Annotated[int, typer.Option()] = 2048,
    gpus: Annotated[str, typer.Option()] = '0'
):
    _, config = load_base_config(config_path)
    if checkpoint is not None:
        config.output_dir = os.path.join(config.output_dir, checkpoint)
    model = load_model(config, backend, temp, top_p, min_p, max_output_tokens, gpus)
    assert os.path.exists(os.path.join(config.output_dir, "merged")), "The model must be merged but is not."
    
    eval_data = get_default_dataset(eval_lang)
    conversations = convert_to_oai(eval_data)
    conversations = predict(
        model=model,
        conversations=conversations,
        type=type
    )
    model.finalize()
    
    output_save_path = os.path.join(config.output_dir, eval_lang + '_ifeval_results')
    os.makedirs(output_save_path, exist_ok=True)
    responses_path = os.path.join(output_save_path, 'responses.jsonl')
    save_responses(conversations, responses_path)
    print('Saved responses to', responses_path)

    eval_config = Config(
        # Do not provide input data path, we use default data
        language=eval_lang,
        input_response_path=responses_path,
        output_dir=output_save_path
    )
    
    run_evaluation(eval_config)

def run():
    typer.run(main)