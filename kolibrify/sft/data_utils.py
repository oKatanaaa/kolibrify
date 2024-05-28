import datasets
datasets.disable_caching()
import random
import json
from typing import List

from .config import StageConfig, DatasetConfig
from kolibrify.core.config import BaseConfig
from kolibrify.core.data_utils import SimpleDataGen, CurriculumDataGen, ChatMLFormatter


def load_jsonl_data(datasets: List[DatasetConfig], epochs: float, format_fn_batched=None):
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

    if format_fn_batched is not None:
        print('format_fn is not None, applying format_fn')
        total_samples = format_fn_batched(total_samples)
        # _total_samples = []
        # for sample in tqdm(total_samples):
        #     _total_samples.append(format_fn(sample))
        # total_samples = _total_samples
        print('format_fn applied')

    return SimpleDataGen(total_samples, epochs)

def get_dataset_features():
    return datasets.Features({
        'messages': datasets.features.Sequence(datasets.features.Value('null', None), id=None),
        'prompt': datasets.features.Value('string', None),
        'labels': datasets.features.Sequence(datasets.Value('uint64', None), id=None),
        'input_ids': datasets.features.Sequence(datasets.Value('uint64', None), id=None),
        'attention_mask': datasets.features.Sequence(datasets.Value('uint64', None), id=None)
    })

def load_dataset(stages: List[StageConfig], tokenizer, config: BaseConfig, val_dataset_path=None):
    format_fn  = ChatMLFormatter(tokenizer, config.max_ctx_len)

    training_datagens = {}
    total_data_iterations = 0
    for stage in stages:
        epochs = stage.epochs
        dataset_configs = stage.datasets
        datagen = load_jsonl_data(dataset_configs, epochs, format_fn_batched=None)
        total_data_iterations += datagen.iterations
        training_datagens[stage.name] = datagen
    
    curriculum_datagen = CurriculumDataGen(training_datagens)

    # Have to provide features, otherwise TRL trainer will fail
    features = get_dataset_features()
    train_dataset = datasets.IterableDataset.from_generator(curriculum_datagen) \
        .map(
            format_fn.format_batched, 
            batch_size=config.micro_batch_size, 
            batched=True, 
            features=features
        )
    
    val_dataset = None
    if val_dataset_path:
        validation_gen = load_jsonl_data(val_dataset_path)
        val_dataset = datasets.Dataset.from_generator(validation_gen) \
            .map(format_fn, load_from_cache_file=False)
    return train_dataset, val_dataset, total_data_iterations
