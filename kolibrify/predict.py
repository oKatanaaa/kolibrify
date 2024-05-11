import typer
from typing_extensions import Annotated
from peft import PeftModel
import os


from copy import deepcopy
from glob import glob

from kolibrify.config import load_training_config
from kolibrify.inference import (
    load_model, load_dataset, save_dataset, predict
)


def main(
    config_path: Annotated[str, typer.Argument()],
    dataset_path: Annotated[str, typer.Argument()],
    dataset_save_path: Annotated[str, typer.Argument()] = 'output.jsonl',
    backend: Annotated[str, typer.Option()] = 'vllm',
    type: Annotated[str, typer.Option()] = 'last', # not supported atm
    temp: Annotated[float, typer.Option()] = 0,
    top_p: Annotated[float, typer.Option()] = 0.95,
    max_output_tokens: Annotated[int, typer.Option()] = 4096,
    gpus: Annotated[str, typer.Option()] = '0'
):
    _, config = load_training_config(config_path)
    model = load_model(config, backend, temp, top_p, max_output_tokens, gpus)
    print('Loaded model.')
    
    if os.path.isdir(dataset_path):
        # Folder with multiple datasets is provided
        assert not dataset_save_path.endswith('.jsonl'), 'Dataset save path must be a directory when load path is a directory.'
        os.makedirs(dataset_save_path, exist_ok=True)
        print('Received a directory.')

        # Load data
        dataset_paths = glob(os.path.join(dataset_path, '*.jsonl'))
        assert len(dataset_paths) > 0, 'No .jsonl files found in the dataset directory.'
        print('Found', len(dataset_paths), 'datasets.')

        conv_list = []
        for dataset_path in dataset_paths:
            conv_list.append(load_dataset(dataset_path))
        
        # Generate and save predictions
        for dataset_path, conversations in zip(dataset_paths, conv_list):
            print('Generating responses for', dataset_path)
            conversations = predict(
                model=model,
                conversations=conversations,
                type=type
            )
            save_path = os.path.join(dataset_save_path, os.path.basename(dataset_path))
            save_dataset(conversations, save_path)
            print(f'Saved responses for {dataset_path} to {save_path}')
    else:
        # Path to a single dataset file is provided
        conversations = load_dataset(dataset_path)
        
        conversations = predict(
            model=model,
            conversations=conversations,
            type=type
        )
        
        save_dataset(conversations, dataset_save_path)
        print('Saved responses to', dataset_save_path)
    
    model.finalize()

def run():
    typer.run(main)