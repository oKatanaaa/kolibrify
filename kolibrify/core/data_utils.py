import random
import copy
import json
from typing import List


def load_jsonl(path):
    with open(path, 'r') as f:
        return [json.loads(line) for line in f]


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


class ChatMLFormatter:
    @staticmethod
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
        return raw_chat_text
    
    def __init__(self, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def tokenize(self, prompt: str) -> List[str]:
        result = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_len,
            padding=True
        )
        result["labels"] = result["input_ids"].copy()
        return result
    
    def __call__(self, sample):
        prompt = ChatMLFormatter.format_chatml(sample)
        out_dict = self.tokenize(prompt)
        out_dict['prompt'] = prompt
        return out_dict
    
    def format_batched(self, samples):
        _samples = []
        for sample in samples['messages']:
            _samples.append({'messages': sample})
        prompts = [ChatMLFormatter.format_chatml(sample) for sample in _samples]
        batched_dict = self.tokenize(prompts)
        # out_dicts = []
        # for i in range(len(prompts)):
        #     input_ids = batched_dict['input_ids'][i]
        #     labels = batched_dict['labels'][i]
        #     out_dicts.append(
        #         {'input_ids': input_ids, 'labels': labels, 'prompt': prompts[i]}
        #     )
        return batched_dict