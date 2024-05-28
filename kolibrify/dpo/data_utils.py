import datasets
import random
import copy
import json
from typing import List

from kolibrify.core import CurriculumDataGen, SimpleDataGen, load_jsonl

from .config import StageConfig, DatasetConfig


def load_jsonl_data(datasets: List[DatasetConfig], epochs: float):
    """
    Loads a dataset which samples stored as jsons per line.
    
    Example:
    {'messages': [...]}
    {'messages': [...]}
    {'messages': [...]}
    ...
    {'messages': [...]}
    """
    total_samples = []
    for dataset in datasets:
        accepted = load_jsonl(dataset.accepted)
        rejected = load_jsonl(dataset.rejected)
        assert len(accepted) == len(rejected), \
            f'Number of accepted and rejected samples must be equal, but got {len(accepted)} and {len(rejected)} instead.'
        
        accepted_rejected_pairs = [{'accepted': a, 'rejected': r} for a, r in zip(accepted, rejected)]

        random.shuffle(accepted_rejected_pairs)
        n_lines = dataset.n_samples if dataset.n_samples != -1 else len(accepted_rejected_pairs)
        accepted_rejected_pairs = accepted_rejected_pairs[:n_lines]
        
        samples = []
        for pair in accepted_rejected_pairs:
            samples.append(pair)
  
        print(f'Read {len(samples)} samples from {dataset.accepted} and {dataset.rejected}.')
        total_samples.extend(samples)
    print(f'Total pairs accumulated: {len(total_samples)}')
    return SimpleDataGen(total_samples, epochs)


# TODO: update curriculum sampling
def load_dataset(stages: List[StageConfig], val_dataset_path=None, format_fn=None):
    if format_fn is None:
        format_fn = format_chatml

    training_datagens = {}
    total_data_iterations = 0
    for stage in stages:
        epochs = stage.epochs
        dataset_configs = stage.datasets
        datagen = load_jsonl_data(dataset_configs, epochs)
        total_data_iterations += datagen.iterations
        training_datagens[stage.name] = datagen
    
    curriculum_datagen = CurriculumDataGen(training_datagens)

    train_dataset = datasets.Dataset.from_generator(curriculum_datagen) \
        .map(format_dpo_pairs, load_from_cache_file=False)
    
    val_dataset = None
    if val_dataset_path is not None:
        print('Val dataset is not supported for DPO at the moment. Provided path will be ignored.')

    return train_dataset, val_dataset, total_data_iterations


def format_dpo_pairs(pair):
    accepted, rejected = pair['accepted'], pair['rejected']
    accepted_conv, rejected_conv = accepted['messages'], rejected['messages']

    prompt = format_chatml(accepted_conv[:-1])['prompt'] + '<|im_start|>assistant\n'
    chosen = accepted_conv[-1]['content']
    rejected = rejected_conv[0]['content']
    
    return {'prompt': prompt, 'chosen': chosen, 'rejected': rejected}


def format_chatml(chat) -> str:
    """
    Uses https://github.com/openai/openai-python/blob/main/chatml.md as chat format.
    """
    raw_chat_text = ""
    for item in chat:
        if len(raw_chat_text) > 0:
            raw_chat_text += '\n'
        role = item['role']
        content = item['content']
        raw_chat_text += f"<|im_start|>{role}\n{content}<|im_end|>"
    return {'prompt': raw_chat_text}
