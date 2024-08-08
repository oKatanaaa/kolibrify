IM_START = '<|im_start|>'
IM_END = '<|im_end|>'
ROLE_SYSTEM = 'system'
ROLE_USER = 'user'
ROLE_ASSISTANT = 'assistant'
MSG_TEMPLATE = IM_START + "{role}\n{content}" + IM_END

TOOLS_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "type": {
                "type": "string",
                "enum": ["function"]
            },
            "function": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string"
                    },
                    "description": {
                        "type": "string"
                    },
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": ["object"]
                            },
                            "properties": {
                                "type": "object",
                                "patternProperties": {
                                    "^.*$": {
                                        "type": "object",
                                        "properties": {
                                            "type": {
                                                "type": "string",
                                                "enum": ["string", "integer", "boolean", "object", "array", "number", "null", "float"]
                                            },
                                            "description": {
                                                "type": "string"
                                            },
                                            "enum": {
                                                "type": "array",
                                                "items": {
                                                    "type": ["string", "integer", "boolean", "number", "null"]
                                                }
                                            }
                                        },
                                        "required": ["type", "description"]
                                    }
                                },
                                "additionalProperties": False
                            },
                            "required": {
                                "type": "array",
                                "items": {
                                    "type": "string"
                                }
                            }
                        },
                        "required": ["type", "properties", "required"]
                    }
                },
                "required": ["name", "description"]
            }
        },
        "required": ["type", "function"]
    }
}

TOOL_CALL_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string"
        },
        "arguments": {
            "type": "object",
            "patternProperties": {
                "^[a-zA-Z_][a-zA-Z0-9_]*$": {
                    "type": ["object", "string", "number", "boolean", "array", "null"]
                }
            },
            "additionalProperties": False
        }
    },
    "required": ["name", "arguments"],
    "additionalProperties": False
}

TOOL_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "role": {
            "type": "string",
            "enum": ["tool"]
        },
        "name": {
            "type": "string"
        },
        "content": {
            "type": "string"
        }
    },
    "required": ["name", "content"],
    "additionalProperties": False
}


TOOLS_PROMPT_EN = """### TOOLS
You have access to the following tools:

{tools}

To use a tool, please provide the tool name and arguments as a valid JSON object. Your response and the JSON should have the following structure:
<tool_call>
{{
    "tool_name": "tool name",
    "arguments": {{
        "argument_name": "argument value"
    }}
}}
</tool_call>

For example:
<tool_call>
{{
    "tool_name": "search",
    "arguments": {{
        "query": "recent movies"
    }}
}}
</tool_call>

You may have some commentaries for the user, for example:
Sure, I'll look for recent movies for you.
<tool_call>
{{
    "tool_name": "search",
    "arguments": {{
        "query": "recent movies"
    }}
}}
</tool_call>

The user will provide you with the tool response in the following format:
<tool_response>
{{"tool name": "response"}}
</tool_response>

For example:
<tool_response>
{{"search": "Here are the recent movies..."}}
</tool_response>

You will then use the tool response to complete the user's request.
""".strip('\n')



TOOLS_PROMPT_RU = """### ИНСТРУМЕНТЫ
У тебя есть доступ к следующим инструментам:

{tools}

Чтобы использовать инструмент, предоставь название инструмента и аргументы в виде корректного JSON объекта. Твой ответ и JSON должны иметь следующую структуру:
<tool_call>
{{
    "tool_name": "название инструмента",
    "arguments": {{
        "название аргумента": "значение аргумента"
    }}
}}
</tool_call>

Например:
<tool_call>
{{
    "tool_name": "search",
    "arguments": {{
        "query": "недавние фильмы"
    }}
}}
</tool_call>

Ты можешь добавить комментарии для пользователя, например:
Конечно, я поищу для вас недавние фильмы.
<tool_call>
{{
    "tool_name": "search",
    "arguments": {{
        "query": "недавние фильмы"
    }}
}}
</tool_call>

Пользователь предоставит тебе ответ инструмента в следующем формате:
<tool_response>
{{"название инструмента": "ответ инструмента"}}
</tool_response>

Например:
<tool_response>
{{"search": "Вот недавние фильмы..."}}
</tool_response>

Затем используй ответ инструмента, чтобы выполнить запрос пользователя.
""".strip('\n')

TOOL_CALL_TEMPLATE = "<tool_call>\n{{\"tool_name\": {name}, \"arguments\": {arguments}}}\n</tool_call>"
TOOL_RESPONSE_TEMPLATE = "<tool_response>\n{{\"tool_name\": {name}, \"response\": {response}}}\n</tool_response>"