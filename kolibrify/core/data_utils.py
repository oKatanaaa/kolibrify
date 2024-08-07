import random
import copy
import json
from typing import List
from jsonschema import validate

from .data_consts import *


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
    def format_chatml(chat: list[dict[str, str]]) -> str:
        """
        Uses https://github.com/openai/openai-python/blob/main/chatml.md as chat format.
        """
        raw_chat_text, msgs = ChatMLFormatter.generate_system_message(chat)
        for msg in msgs:
            if len(raw_chat_text) > 0:
                raw_chat_text += '\n'
            raw_chat_text += ChatMLFormatter.format_msg(msg, chat)
        return raw_chat_text
    
    @staticmethod
    def generate_system_message(chat: list[dict[str, str]]):
        messages = chat['messages']
        tools = chat.get('tools')

        if messages[0]['role'] != ROLE_SYSTEM and tools is None:
            return "", messages

        system_message = IM_START + 'system\n'
        if messages[0]['role'] == ROLE_SYSTEM:
            content = messages[0]['content']
            system_message += content
            # Remove system message from messages so that it does not go
            # into future formatting methods
            messages = messages[1:]

        if tools is not None:
            validate(instance=tools, schema=TOOLS_SCHEMA)
            if messages[0]['role'] == ROLE_SYSTEM:
                system_message += '\n'
            system_message += TOOLS_PROMPT_EN.format(tools=json.dumps(tools))
        
        system_message += IM_END
        return system_message, messages

    @staticmethod
    def format_msg(msg, chat):
        # At the moment chat is not used
        # But it is planned to use its context to validate tool calls
        role = msg['role']
        if role == ROLE_USER:
            return ChatMLFormatter.format_user_msg(msg)
        elif role == ROLE_ASSISTANT:
            return ChatMLFormatter.format_assistant_msg(msg)
        elif role == 'tool':
            return ChatMLFormatter.format_tool_response(msg)
        elif role == ROLE_SYSTEM:
            raise ValueError("System message should not be in messages")
    
    @staticmethod
    def format_user_msg(msg):
        return MSG_TEMPLATE.format(role=ROLE_USER, content=msg['content'])
    
    @staticmethod
    def format_assistant_msg(msg):
        msg_content = ""
        content = msg.get('content')
        tool_call = msg.get('tool_call')

        if content is not None:
            msg_content += content

        if msg.get('tool_call') is not None:
            if msg_content:
                msg_content += '\n'
            msg_content += ChatMLFormatter.format_tool_call(tool_call)

        assert msg_content != "", "Empty assistant message"

        return MSG_TEMPLATE.format(role=ROLE_ASSISTANT, content=msg_content)
    
    @staticmethod
    def format_tool_call(tool_call):
        tool_name = tool_call['tool_name']
        tool_args = json.dumps(tool_call['arguments'])
        tool_msg = TOOL_CALL_TEMPLATE.format(name=tool_name, arguments=tool_args)
        return tool_msg

    @staticmethod
    def format_tool_response(tool_msg):
        tool_name = tool_msg['name']
        tool_response = tool_msg['content']
        tool_response_xml = TOOL_RESPONSE_TEMPLATE.format(name=tool_name, response=tool_response)
        return MSG_TEMPLATE.format(role=ROLE_USER, content=tool_response_xml)

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
        return batched_dict