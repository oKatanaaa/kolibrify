import datasets
import random
import copy
import json
from typing import List

from .sft.config import StageConfig, DatasetConfig

class CurriculumDataGen:
    def __init__(self, simple_data_gens):
        self.datagens = simple_data_gens
        
    def __iter__(self):
        for stage_name, datagen in self.datagens.items():
            print(f'Stage {stage_name} data sampling.')
            for sample in datagen:
                yield sample

    def __call__(self):
        return self.__iter__()


class SimpleDataGen:
    def __init__(self, samples, epochs: float):
        self.samples = copy.deepcopy(samples)
        random.shuffle(self.samples)
        self.iterations = int(len(self.samples) * epochs)
        self.current_iter = -1
        
    def __iter__(self):
        return self
            
    def __next__(self):
        self.current_iter += 1
        if self.current_iter == self.iterations:
            raise StopIteration()
        
        if self.current_iter % len(self.samples) == 0:
            # An epoch has passed, reshuffle dataset
            random.shuffle(self.samples)
        
        idx = self.current_iter % len(self.samples)
        return self.samples[idx]

    def __call__(self):
        return self.__iter__()


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
        samples = []
        with open(dataset.path, 'r') as f:
            lines = f.readlines()
        
        random.shuffle(lines)
        n_lines = dataset.n_samples if dataset.n_samples != -1 else len(lines)
        lines = lines[:n_lines]
        
        for line in lines:
            samples.append(json.loads(line))
  
        print(f'Read {len(samples)} samples from {dataset.path}.')
        total_samples.extend(samples)
    print(f'Total samples accumulated: {len(total_samples)}')
    return SimpleDataGen(total_samples, epochs)


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
        .map(format_chatml, load_from_cache_file=False)
    
    val_dataset = None
    if val_dataset_path:
        validation_gen = load_jsonl_data(val_dataset_path)
        val_dataset = datasets.Dataset.from_generator(validation_gen) \
            .map(format_chatml, load_from_cache_file=False)
    return train_dataset, val_dataset, total_data_iterations


def format_chatml(chat: list[dict[str, str]] = dict()) -> str:
    """
    Uses https://github.com/openai/openai-python/blob/main/chatml.md as chat format.
    """
    chat = chat['messages']
    raw_chat_text = ""
    for item in chat:
        if len(raw_chat_text) > 0:
            raw_chat_text += '\n'
        role = item['role']
        content = item['content']
        raw_chat_text += f"<|im_start|>{role}\n{content}<|im_end|>"
    return {'prompt': raw_chat_text}
