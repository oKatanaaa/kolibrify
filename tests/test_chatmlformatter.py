import unittest
import json
from jsonschema import ValidationError

from kolibrify.core.data_utils import ChatMLFormatter
from kolibrify.core.data_consts import *


class TestChatMLFormatter(unittest.TestCase):
    def setUp(self):
        # Setup any required variables or objects
        self.chatml_formatter = ChatMLFormatter(None, 512, True)  # tokenizer and max_len set to 512 for example
        self.maxDiff = None

    def test_regular_messages(self):
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

    def test_with_system_message(self):
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

    def test_tool_call(self):
        chat = {
            'messages': [
                {'role': 'user', 'content': 'What is the weather like today in Boston?'},
                {'role': 'assistant', 'content': 'Let me check the weather for you.', 'tool_call': {
                    "name": "get_current_weather",
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

    def test_incorrect_tool_specification(self):
        chat = {
            'messages': [
                {'role': 'user', 'content': 'What is the weather like today in Boston?'},
                {'role': 'assistant', 'content': 'Let me check the weather for you.', 'tool_call': {
                    "name": "get_current_weather",
                    "arguments": {
                        "location": "Boston",
                        "format": "celsius"
                    }}},
                {'role': 'tool', 'name': 'get_current_weather', 'content': 'The weather today is sunny with a high of 25°C.'}
            ],
            'tools': [
                {
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
                    }
                }
            ]
        }
        
        with self.assertRaises(ValidationError) as context:
            self.chatml_formatter.format_chatml(chat)

    def test_incorrect_tool_call_schema(self):
        chat = {
            'messages': [
                {'role': 'user', 'content': 'What is the weather like today in Boston?'},
                {'role': 'assistant', 'content': 'Let me check the weather for you.', 'tool_call': {
                    "toolname": "get_current_weather",
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
        
        with self.assertRaises(ValidationError) as context:
            self.chatml_formatter.format_chatml(chat)
    
    def test_incorrect_tool_call_args(self):
        chat = {
            'messages': [
                {'role': 'user', 'content': 'What is the weather like today in Boston?'},
                {'role': 'assistant', 'content': 'Let me check the weather for you.', 'tool_call': {
                    "name": "get_current_weather",
                    "arguments": {
                        "city": "Boston",
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
        
        with self.assertRaises(ValidationError) as context:
            self.chatml_formatter.format_chatml(chat)

    def test_tool_without_tools(self):
        chat = {
            'messages': [
                {'role': 'user', 'content': 'What is the weather like today in Boston?'},
                {'role': 'assistant', 'content': 'Let me check the weather for you.', 'tool_call': {
                    "name": "get_current_weather",
                    "arguments": {
                        "city": "Boston",
                        "format": "celsius"
                    }}},
                {'role': 'tool', 'name': 'get_current_weather', 'content': 'The weather today is sunny with a high of 25°C.'}
            ]
        }
        
        with self.assertRaises(AssertionError) as context:
            self.chatml_formatter.format_chatml(chat)

        self.assertIn("There are no tools, but a tool call exists", str(context.exception))

    def test_wrong_tool_name(self):
        chat = {
            'messages': [
                {'role': 'user', 'content': 'What is the weather like today in Boston?'},
                {'role': 'assistant', 'content': 'Let me check the weather for you.', 'tool_call': {
                    "name": "get_weather",
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
        
        with self.assertRaises(AssertionError) as context:
            self.chatml_formatter.format_chatml(chat)

        self.assertIn("Tool get_weather not found in tool context", str(context.exception))

    def test_wrong_tool_response_schema(self):
        chat = {
            'messages': [
                {'role': 'user', 'content': 'What is the weather like today in Boston?'},
                {'role': 'assistant', 'content': 'Let me check the weather for you.', 'tool_call': {
                    "name": "get_current_weather",
                    "arguments": {
                        "location": "Boston",
                        "format": "celsius"
                    }}},
                {'role': 'tool', 'toolname': 'get_current_weather', 'content': 'The weather today is sunny with a high of 25°C.'}
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
        
        with self.assertRaises(ValidationError) as context:
            self.chatml_formatter.format_chatml(chat)

    def test_wrong_tool_response_name(self):
        chat = {
            'messages': [
                {'role': 'user', 'content': 'What is the weather like today in Boston?'},
                {'role': 'assistant', 'content': 'Let me check the weather for you.', 'tool_call': {
                    "name": "get_current_weather",
                    "arguments": {
                        "location": "Boston",
                        "format": "celsius"
                    }}},
                {'role': 'tool', 'name': 'get_weather', 'content': 'The weather today is sunny with a high of 25°C.'}
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
        
        with self.assertRaises(AssertionError) as context:
            self.chatml_formatter.format_chatml(chat)

        self.assertIn("Tool get_weather not found in tool context", str(context.exception))

    def test_tool_response_without_tools(self):
        chat = {
            'messages': [
                {'role': 'user', 'content': 'What is the weather like today in Boston?'},
                {'role': 'assistant', 'content': 'Let me check the weather for you.'},
                {'role': 'tool', 'name': 'get_current_weather', 'content': 'The weather today is sunny with a high of 25°C.'}
            ]
        }
        
        with self.assertRaises(AssertionError) as context:
            self.chatml_formatter.format_chatml(chat)

        self.assertIn("There are no tools, but a tool response exists", str(context.exception))

if __name__ == '__main__':
    unittest.main()
