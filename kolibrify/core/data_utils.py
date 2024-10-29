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
    def __init__(self, tokenizer, max_len, debug=False, return_tensors=None):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.debug = debug
        self.return_tensors = return_tensors

    def __call__(self, sample, postfix=None):
        prompt = self.format_chatml(sample)

        if postfix is not None:
            # Useful to add `\n<|im_start|>assistant\n` during inference
            prompt += postfix

        out_dict = self.tokenize(prompt)
        out_dict['prompt'] = prompt
        return out_dict
    
    def tokenize(self, prompt: str) -> List[str]:
        result = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_len,
            padding=True,
            return_tensors=self.return_tensors
        )
        result["labels"] = copy.deepcopy(result["input_ids"])
        return result

    def format_chatml(self, chat: list[dict[str, str]]) -> str:
        """
        Uses https://github.com/openai/openai-python/blob/main/chatml.md as chat format.
        """
        tool_ctx = self.extract_tool_ctx(chat)
        raw_chat_text, msgs = self.generate_system_message(chat)
        for msg in msgs:
            if len(raw_chat_text) > 0:
                raw_chat_text += '\n'
            raw_chat_text += self.format_msg(msg, tool_ctx)
        return raw_chat_text
    
    def extract_tool_ctx(self, chat: list[dict[str, str]]):
        tools = chat.get('tools')

        if tools is None or not self.debug:
            return None
        
        validate(instance=tools, schema=TOOLS_SCHEMA)
        
        tool_ctx = dict()
        for tool in tools:
            tool = tool['function']
            tool_ctx[tool['name']] = tool['parameters']
        return tool_ctx
    
    def generate_system_message(self, chat: list[dict[str, str]]):
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
            # Tools schema has already been validated
            if messages[0]['role'] == ROLE_SYSTEM:
                system_message += '\n'
            system_message += TOOLS_PROMPT_EN.format(tools=json.dumps(tools))
        
        system_message += IM_END
        return system_message, messages

    def format_msg(self, msg, tool_ctx):
        # At the moment chat is not used
        # But it is planned to use its context to validate tool calls
        role = msg['role']
        if role == ROLE_USER:
            return self.format_user_msg(msg)
        elif role == ROLE_ASSISTANT:
            return self.format_assistant_msg(msg, tool_ctx)
        elif role == 'tool':
            return self.format_tool_response(msg, tool_ctx)
        elif role == ROLE_SYSTEM:
            raise ValueError("System message should not be in messages")
    
    def format_user_msg(self, msg):
        return MSG_TEMPLATE.format(role=ROLE_USER, content=msg['content'])
    
    def format_assistant_msg(self, msg, tool_ctx):
        msg_content = ""
        content = msg.get('content')
        tool_call = msg.get('tool_call')

        if content is not None:
            msg_content += content

        if msg.get('tool_call') is not None:
            if msg_content:
                msg_content += '\n'
            msg_content += self.format_tool_call(tool_call, tool_ctx)

        assert msg_content != "", "Empty assistant message"

        return MSG_TEMPLATE.format(role=ROLE_ASSISTANT, content=msg_content)
    
    def format_tool_call(self, tool_call, tool_ctx):
        if self.debug:
            assert tool_ctx is not None, 'There are no tools, but a tool call exists'
            validate(instance=tool_call, schema=TOOL_CALL_SCHEMA)

        tool_name = tool_call['name']
        tool_args = tool_call['arguments']

        if self.debug:
            assert tool_name in tool_ctx, f"Tool {tool_name} not found in tool context"
            tool_args_shema = tool_ctx[tool_name]
            validate(instance=tool_args, schema=tool_args_shema)
        
        tool_args = json.dumps(tool_args)
        tool_msg = TOOL_CALL_TEMPLATE.format(name=tool_name, arguments=tool_args)
        return tool_msg

    def format_tool_response(self, tool_msg, tool_ctx):
        if self.debug:
            validate(instance=tool_msg, schema=TOOL_RESPONSE_SCHEMA)
            assert tool_ctx is not None, 'There are no tools, but a tool response exists'
            assert tool_msg['name'] in tool_ctx, f"Tool {tool_msg['name']} not found in tool context"
            
        tool_name = tool_msg['name']
        tool_response = tool_msg['content']
        tool_response_xml = TOOL_RESPONSE_TEMPLATE.format(name=tool_name, response=tool_response)
        return MSG_TEMPLATE.format(role=ROLE_USER, content=tool_response_xml)

    def format_batched(self, samples):
        _samples = []
        for sample in samples['messages']:
            _samples.append({'messages': sample})
        prompts = [self.format_chatml(sample) for sample in _samples]
        batched_dict = self.tokenize(prompts)
        return batched_dict
