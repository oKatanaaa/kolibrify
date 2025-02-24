import datasets
datasets.disable_caching()
import random
import json
import copy
from typing import List
from tokenizers import Tokenizer

from .config import StageConfig, DatasetConfig
from kolibrify.core.config import BaseConfig
from kolibrify.core.data_utils import SimpleDataGen, CurriculumDataGen, ChatMLFormatter


def add_seq_len_to_samples(samples: List[dict], tokenizer: Tokenizer) -> List[dict]:
    """
    For each sample in `samples`, concatenate all the 'content' strings 
    from the sample['messages'] (joined by a space) and compute the token count 
    using tokenizer.encode_batch. The computed length is stored in sample['seq_len'].
    """
    for sample in samples:
        # Extract the 'content' strings from each message.
        content_strs = [msg.get('content', '') for msg in sample.get('messages', [])]
        concatenated = " ".join(content_strs)
        tokenized = tokenizer.tokenize(concatenated)
        sample['seq_len'] = len(tokenized)
    return samples


class GroupedSimpleDataGen:
    """
    A modified version of SimpleDataGen that groups samples by their 'seq_len'. 
    At the beginning of each epoch, all samples are sorted by 'seq_len', then split 
    into groups (with each group having group_size samples), the groups are shuffled, 
    and finally samples are yielded sequentially from the shuffled groups.
    """
    def __init__(self, samples: List[dict], epochs: float, group_size: int):
        self.samples = copy.deepcopy(samples)
        self.group_size = group_size
        self.iterations = int(len(self.samples) * epochs)
        self.current_iter = -1
        self.num_samples = len(self.samples)
        self.ordered_samples = []  # will hold the new ordering for the current epoch

    def __iter__(self):
        self.current_iter = -1
        return self

    def __next__(self):
        self.current_iter += 1
        if self.current_iter >= self.iterations:
            raise StopIteration()

        # At the start of each epoch, re-sort and re-group the samples.
        if self.current_iter % self.num_samples == 0:
            sorted_samples = sorted(self.samples, key=lambda s: s['seq_len'])
            groups = [sorted_samples[i:i + self.group_size] for i in range(0, len(sorted_samples), self.group_size)]
            random.shuffle(groups)  # shuffle groups to reintroduce randomness
            self.ordered_samples = [s for group in groups for s in group]

        idx = self.current_iter % self.num_samples
        return self.ordered_samples[idx]

    def __call__(self):
        return self.__iter__()


def load_jsonl_data(datasets_list: List[DatasetConfig],
                    epochs: float,
                    format_fn_batched=None) -> List[dict]:
    """
    Loads a dataset stored as JSON lines.
    Each line in the file is expected to be a JSON object representing a sample with a 'messages' key.
    This function only loads and shuffles the raw samples.
    
    If a format_fn_batched is provided, it is applied to the final list of samples.
    """
    total_samples = []
    for dataset in datasets_list:
        samples = []
        with open(dataset.path, 'r') as f:
            lines = f.readlines()

        random.shuffle(lines)
        n_lines = dataset.n_samples if dataset.n_samples != -1 else len(lines)
        lines = lines[:n_lines]

        for line in lines:
            sample = json.loads(line)
            samples.append(sample)

        print(f"Read {len(samples)} samples from {dataset.path}.")
        total_samples.extend(samples)
    print(f"Total samples accumulated: {len(total_samples)}")

    if format_fn_batched is not None:
        print("format_fn is not None, applying format_fn")
        total_samples = format_fn_batched(total_samples)
        print("format_fn applied")

    return total_samples


def get_dataset_features():
    return datasets.Features({
        'messages': datasets.features.Sequence(datasets.features.Value('null', None), id=None),
        'prompt': datasets.features.Value('string', None),
        'labels': datasets.features.Sequence(datasets.Value('uint64', None), id=None),
        'input_ids': datasets.features.Sequence(datasets.Value('uint64', None), id=None),
        'attention_mask': datasets.features.Sequence(datasets.Value('uint64', None), id=None)
    })


def load_dataset(stages: List[StageConfig],
                 tokenizer,
                 config: BaseConfig,
                 val_dataset_path=None):
    """
    Loads the dataset for training and validation.
    
    For each stage:
      - load_jsonl_data is called to obtain the list of raw samples.
      - If group_by_seq_len is True, the samples are modified via add_seq_len_to_samples
        (which computes a naive token count as the sequence length) and then wrapped in
        GroupedSimpleDataGen.
      - Otherwise, the original SimpleDataGen is used.
      
    In the final pipeline, the samples (raw dictionaries) are converted to tokens by the 
    ChatMLFormatter.format_batched function.
    """
    format_fn = ChatMLFormatter(tokenizer, config.max_ctx_len)

    training_datagens = {}
    total_data_iterations = 0

    for stage in stages:
        epochs = stage.epochs
        dataset_configs = stage.datasets

        samples = load_jsonl_data(dataset_configs, epochs, format_fn_batched=None)

        if config.group_by_seq_len:
            samples = add_seq_len_to_samples(samples, tokenizer)
            datagen = GroupedSimpleDataGen(samples, epochs, config.micro_batch_size)
        else:
            datagen = SimpleDataGen(samples, epochs)

        total_data_iterations += datagen.iterations
        training_datagens[stage.name] = datagen

    curriculum_datagen = CurriculumDataGen(training_datagens)

    features = get_dataset_features()
    train_dataset = datasets.IterableDataset.from_generator(curriculum_datagen)\
        .map(
            format_fn.format_batched,
            batch_size=config.micro_batch_size,
            batched=True,
            features=features
        )

    val_dataset = None
    if val_dataset_path:
        # For validation, we simply load the raw samples without grouping.
        val_dataset_config = [DatasetConfig(path=val_dataset_path, n_samples=-1)]
        validation_samples = load_jsonl_data(val_dataset_config, 1, format_fn_batched=None)
        val_datagen = SimpleDataGen(validation_samples, 1)
        val_dataset = datasets.Dataset.from_generator(val_datagen)\
            .map(format_fn, load_from_cache_file=False)
    return train_dataset, val_dataset, total_data_iterations