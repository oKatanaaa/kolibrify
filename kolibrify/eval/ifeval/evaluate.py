import subprocess
import json
import typer
import os
from typing_extensions import Annotated

from kolibrify.inference import load_model, predict
from kolibrify.config import load_training_config


package_path = os.path.dirname(__file__)
evaldata_path = os.path.join(
    package_path, 
    'data'
)
RU_EVAL_PATH = os.path.join(evaldata_path, 'ru.jsonl')
EN_EVAL_PATH = os.path.join(evaldata_path, 'en.jsonl')


def load_eval_data(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    
    lines = list(map(json.loads, lines))
    
    conversations = []
    for line in lines:
        conversations.append(
            {'messages': [
                {'role': 'user', 'content': line['prompt']}
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
    eval_lang: Annotated[str, typer.Option()] = 'en',
    backend: Annotated[str, typer.Option()] = 'vllm',
    type: Annotated[str, typer.Option()] = 'last',
    temp: Annotated[float, typer.Option()] = 0,
    top_p: Annotated[float, typer.Option()] = 0.95,
    max_output_tokens: Annotated[int, typer.Option()] = 4096,
    gpus: Annotated[str, typer.Option()] = '0'
):
    _, config = load_training_config(config_path)
    model = load_model(config, backend, temp, top_p, max_output_tokens, gpus)
    assert os.path.exists(os.path.join(config.output_dir, "merged")), "The model must be merged but is not."
    assert eval_lang in ['en', 'ru'], f"Only 'ru' and 'en' are supported, but got {eval_lang}."
    if eval_lang == 'en':
        eval_file_path = EN_EVAL_PATH
    else:
        eval_file_path = RU_EVAL_PATH
    
    conversations = load_eval_data(eval_file_path)
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
    
    # Launch evaluation
    logs_path = os.path.join(output_save_path, 'eval.logs')
    if eval_lang == 'en':
        arg_prefix = ["python",
            "-m",
            "kolibrify.eval.ifeval.en.evaluation_main"]
    else:
        arg_prefix = ["python",
            "-m",
            "kolibrify.eval.ifeval.ru.evaluation_main"]
    with open(logs_path, 'w') as f:
        subprocess.run(
            arg_prefix + [
            f"--input_data={eval_file_path}",
            f"--input_response_data={responses_path}",
            f"--output_dir={output_save_path}"],
            stdout=f
        )

def run():
    typer.run(main)