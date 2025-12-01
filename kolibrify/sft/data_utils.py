import datasets
datasets.disable_caching()
import random
import json
import copy
from tqdm import tqdm
from typing import List

from .config import StageConfig, DatasetConfig
from kolibrify.core.config import BaseConfig
from kolibrify.core.data_utils import SimpleDataGen, CurriculumDataGen, ChatMLFormatter

# Seems to be a good trade-off of memory for speed
TOKENIZATION_BATCH_SIZE = 1024


def add_seq_len_to_samples(samples: List[dict], tokenizer, batch_size=64) -> List[dict]:
    """
    Optimized implementation using the tokenizer's native batching capabilities.
    This leverages the tokenizer's ability to process multiple inputs at once.
    
    Args:
        samples: List of samples to process
        tokenizer: The tokenizer object
        batch_size: Number of samples to process in each batch
        
    Returns:
        The samples with 'seq_len' added to each
    """
    # Process in batches to avoid potential memory issues with very large datasets
    total_batches = (len(samples) + batch_size - 1) // batch_size
    
    # Create a progress bar using tqdm
    with tqdm(total=total_batches, desc="Estimating length") as pbar:
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i+batch_size]
            
            # Extract all content for the batch
            batch_texts = []
            for sample in batch:
                content_strs = [msg.get('content', '') for msg in sample.get('messages', [])]
                batch_texts.append(" ".join(content_strs))
            
            # Use the tokenizer's batch processing capabilities
            encoded = tokenizer(
                batch_texts,
                truncation=False,     # Don't truncate to get actual lengths
                padding=False,        # No padding to get accurate lengths
                return_tensors=None,  # Don't convert to tensors
                add_special_tokens=True  # Include special tokens in the count
            )
            
            # Extract the lengths from the input_ids
            input_ids = encoded['input_ids']
            for j, sample in enumerate(batch):
                sample['seq_len'] = len(input_ids[j])
            
            # Update the progress bar
            pbar.update(1)
    
    return samples


def load_jsonl_dataset(dataset: DatasetConfig) -> List[dict]:
    """
    Load a single JSONL dataset.
    """
    samples = []
    with open(dataset.path, 'r') as f:
        lines = f.readlines()

    random.shuffle(lines)

    for line in lines:
        sample = json.loads(line)
        samples.append(sample)

    print(f"Read {len(samples)} samples from {dataset.path}.")
    return samples


def _shuffle_by_seq_len(samples: List[dict], group_size: int) -> List[dict]:
    """Helper to sort by seq_len, group, shuffle groups, and flatten."""
    sorted_samples = sorted(samples, key=lambda s: s['seq_len'])
    groups = [sorted_samples[i:i + group_size] for i in range(0, len(sorted_samples), group_size)]
    random.shuffle(groups)
    return [s for group in groups for s in group]


class WeightedDatasetGen:
    """
    Samples from multiple datasets using provided weights for a fixed number of iterations.
    Supports optional grouping by sequence length to reduce padding.
    """
    def __init__(self, dataset_samples: List[List[dict]], weights: List[float], iterations: int,
                 group_by_seq_len: bool, group_size: int):
        assert len(dataset_samples) == len(weights), "Each dataset needs a matching weight"
        assert iterations > 0, "Iterations must be positive"

        self.iterations = iterations
        self.group_by_seq_len = group_by_seq_len
        self.group_size = group_size
        self.current_iter = 0

        # Normalize weights to avoid zero-total issues
        total_weight = sum(weights)
        if total_weight <= 0:
            raise ValueError("Sum of dataset weights must be positive")
        self.weights = [w / total_weight for w in weights]

        self.datasets = []
        for samples in dataset_samples:
            samples_copy = copy.deepcopy(samples)
            if len(samples_copy) == 0:
                raise ValueError("Dataset is empty")

            dataset_entry = {
                "samples": samples_copy,
                "ordered": None,
                "idx": -1,
            }

            if self.group_by_seq_len:
                dataset_entry["ordered"] = _shuffle_by_seq_len(samples_copy, self.group_size)
            else:
                random.shuffle(dataset_entry["samples"])

            self.datasets.append(dataset_entry)

    def __iter__(self):
        self.current_iter = 0
        return self

    def _reshuffle_if_needed(self, dataset_idx: int):
        ds = self.datasets[dataset_idx]
        if self.group_by_seq_len:
            if ds["idx"] % len(ds["ordered"]) == 0:
                ds["ordered"] = _shuffle_by_seq_len(ds["samples"], self.group_size)
        else:
            if ds["idx"] % len(ds["samples"]) == 0:
                random.shuffle(ds["samples"])

    def __next__(self):
        if self.current_iter >= self.iterations:
            raise StopIteration()

        dataset_idx = random.choices(range(len(self.datasets)), weights=self.weights, k=1)[0]
        ds = self.datasets[dataset_idx]
        ds["idx"] += 1
        self._reshuffle_if_needed(dataset_idx)

        if self.group_by_seq_len:
            sample_idx = ds["idx"] % len(ds["ordered"])
            sample = ds["ordered"][sample_idx]
        else:
            sample_idx = ds["idx"] % len(ds["samples"])
            sample = ds["samples"][sample_idx]

        self.current_iter += 1
        return sample

    def __call__(self):
        return self.__iter__()


def get_dataset_features(include_completion_mask: bool = False):
    features = {
        'messages': datasets.features.Sequence(datasets.features.Value('null', None), id=None),
        'prompt': datasets.features.Value('string', None),
        'labels': datasets.features.Sequence(datasets.Value('uint64', None), id=None),
        'input_ids': datasets.features.Sequence(datasets.Value('uint64', None), id=None),
        'attention_mask': datasets.features.Sequence(datasets.Value('uint64', None), id=None)
    }
    if include_completion_mask:
        features['completion_mask'] = datasets.features.Sequence(datasets.Value('uint64', None), id=None)
    return datasets.Features(features)


def load_dataset(stages: List[StageConfig],
                 tokenizer,
                 config: BaseConfig,
                 val_dataset_path=None,
                 return_plain_dataset=False):
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
    mask_responses = getattr(config, "add_imstart_token", False)
    format_fn = ChatMLFormatter(
        tokenizer,
        config.max_ctx_len,
        mask_assistant_responses=mask_responses
    )

    training_datagens = {}
    total_data_iterations = 0
    total_batch_size = config.micro_batch_size * config.gradient_accumulation_steps
    prev_until_step = 0

    for stage in stages:
        stage_steps = stage.until_step - prev_until_step
        assert stage_steps > 0, f"Stage {stage.name} until_step must be greater than previous stage"

        stage_iterations = stage_steps * total_batch_size

        dataset_samples = []
        dataset_weights = []
        for dataset_config in stage.datasets:
            samples = load_jsonl_dataset(dataset_config)
            if config.group_by_seq_len:
                samples = add_seq_len_to_samples(samples, tokenizer, batch_size=TOKENIZATION_BATCH_SIZE)
            dataset_samples.append(samples)
            dataset_weights.append(dataset_config.weight)

        datagen = WeightedDatasetGen(
            dataset_samples=dataset_samples,
            weights=dataset_weights,
            iterations=stage_iterations,
            group_by_seq_len=config.group_by_seq_len,
            group_size=config.micro_batch_size
        )

        total_data_iterations += datagen.iterations
        training_datagens[stage.name] = datagen
        prev_until_step = stage.until_step

    curriculum_datagen = CurriculumDataGen(training_datagens)

    features = get_dataset_features(include_completion_mask=mask_responses)
    
    if not return_plain_dataset:
        train_dataset = datasets.IterableDataset.from_generator(curriculum_datagen)\
            .map(
                format_fn.format_batched,
                batch_size=config.micro_batch_size,
                batched=True,
                features=features
            )
    else:
        train_dataset = datasets.Dataset.from_generator(curriculum_datagen)

    val_dataset = None
    if val_dataset_path:
        # For validation, we simply load the raw samples without grouping.
        val_dataset_config = DatasetConfig(path=val_dataset_path)
        validation_samples = load_jsonl_dataset(val_dataset_config)
        val_datagen = SimpleDataGen(validation_samples, 1)
        val_dataset = datasets.Dataset.from_generator(val_datagen)\
            .map(format_fn, load_from_cache_file=False)
    return train_dataset, val_dataset, total_data_iterations
