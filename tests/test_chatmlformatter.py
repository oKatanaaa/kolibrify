import unittest
import json

from kolibrify.core.data_utils import ChatMLFormatter
from kolibrify.core.data_consts import *


class TestChatMLFormatter(unittest.TestCase):
    def setUp(self):
        # Setup any required variables or objects
        self.chatml_formatter = ChatMLFormatter(None, 512)  # tokenizer and max_len set to 512 for example
        self.maxDiff = None

    def test_format_chatml_regular_messages(self):
        chat = {
            'messages': [
                {'role': 'user', 'content': 'Hello!'},
                {'role': 'assistant', 'content': 'Hi there! How can I assist you today?'}
            ]
        }
        formatted_chat = self.chatml_formatter.format_chatml(chat)
        expected_output = (
            IM_START + ROLE_USER + "\nHello!" + IM_END + '\n' + 
            IM_START + ROLE_ASSISTANT + "\nHi there! How can I assist you today?" + IM_END
        )
        self.assertEqual(formatted_chat, expected_output)

    def test_format_chatml_with_system_message(self):
        chat = {
            'messages': [
                {'role': 'system', 'content': 'You are an AI assistant.'},
                {'role': 'user', 'content': 'Hello!'},
                {'role': 'assistant', 'content': 'Hi there! How can I assist you today?'}
            ]
        }
        formatted_chat = self.chatml_formatter.format_chatml(chat)
        expected_output = (
            IM_START + ROLE_SYSTEM + "\nYou are an AI assistant." + IM_END + '\n' + 
            IM_START + ROLE_USER + "\nHello!" + IM_END + '\n' + 
            IM_START + ROLE_ASSISTANT + "\nHi there! How can I assist you today?" + IM_END
        )
        self.assertEqual(formatted_chat, expected_output)

    def test_format_chatml_with_tool_call(self):
        chat = {
            'messages': [
                {'role': 'user', 'content': 'What is the weather like today in Boston?'},
                {'role': 'assistant', 'content': 'Let me check the weather for you.', 'tool_call': {
                    "tool_name": "get_current_weather",
                    "arguments": {
                        "location": "Boston",
                        "format": "celsius"
                    }}},
                {'role': 'tool', 'name': 'get_current_weather', 'content': 'The weather today is sunny with a high of 25°C.'}
            ],
            'tools': [
                {
                    "type": "function",
                    "function": {
                        "name": "get_current_weather",
                        "description": "Get the current weather",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "The city and state, e.g. San Francisco, CA",
                                },
                                "format": {
                                    "type": "string",
                                    "enum": ["celsius", "fahrenheit"],
                                    "description": "The temperature unit to use. Infer this from the users location.",
                                },
                            },
                            "required": ["location", "format"],
                        },
                    }
                }
            ]
        }
        formatted_chat = self.chatml_formatter.format_chatml(chat)
        expected_output = (
            IM_START + ROLE_SYSTEM + '\n' + TOOLS_PROMPT_EN.format(tools=json.dumps(chat['tools'])) + IM_END + '\n' +
            IM_START + ROLE_USER + "\nWhat is the weather like today in Boston?" + IM_END + '\n' + 
            IM_START + ROLE_ASSISTANT + "\nLet me check the weather for you.\n" +
            TOOL_CALL_TEMPLATE.format(
                name="get_current_weather",
                arguments=json.dumps({"location": "Boston", "format": "celsius"})
            ) + IM_END + '\n' +
            IM_START + ROLE_USER + "\n" +
            TOOL_RESPONSE_TEMPLATE.format(
                name="get_current_weather",
                response="The weather today is sunny with a high of 25°C."
            ) + IM_END
        )
        self.assertEqual(formatted_chat, expected_output)


if __name__ == '__main__':
    unittest.main()
