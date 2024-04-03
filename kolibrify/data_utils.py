import datasets
import random
import copy
import json


class SimpleDataGen:
    def __init__(self, samples):
        self.samples = samples
        
    def __iter__(self):
        samples = copy.deepcopy(self.samples)
        random.shuffle(samples)
        for sample in samples:
            yield sample
            
    def __call__(self):
        return self.__iter__()


def load_jsonl_data(filepaths):
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
    for filepath in filepaths:
        samples = []
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            samples.append(
                json.loads(line)
            )
        print(f'Read {len(samples)} samples from {filepath}.')
        total_samples.extend(samples)
    print(f'Total samples accumulated: {len(total_samples)}')
    return SimpleDataGen(total_samples)


def load_dataset(train_datasets, val_dataset_path=None, format_fn=None):
    if format_fn is None:
        format_fn = format_chatml
        
    training_gen = load_jsonl_data(map(lambda x: x['path'], train_datasets))
    train_dataset = datasets.Dataset.from_generator(training_gen) \
        .map(format_chatml, load_from_cache_file=False)
    
    val_dataset = None
    if val_dataset_path:
        validation_gen = load_jsonl_data(val_dataset_path)
        val_dataset = datasets.Dataset.from_generator(validation_gen) \
            .map(format_chatml, load_from_cache_file=False)
    return train_dataset, val_dataset


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
