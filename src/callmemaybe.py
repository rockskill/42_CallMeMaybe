from pydantic import BaseModel
from src.llm import LLM
from src.encoder import Encoder
from src.function import Function
import json
import re


# Mapping for encoding of function arguments that are regular expressions.
REGEX_MAPPING: list[tuple] = [
    (['vowel', 'vowels'], r'[aeiouAEIOU]'),
    (
        ['consonant', 'consonants'],
        r'[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]',
    ),
    (['digit', 'digits', 'number', 'numbers'], r'\\d+'),
    (['uppercase', 'upper', 'capital'], r'[A-Z]+'),
    (['lowercase', 'lower'], r'[a-z]+'),
    (['letter', 'letters', 'alphabetic'], r'[a-zA-Z]+'),
    (['space', 'spaces', 'whitespaces'], r'\\s+'),
    (['punctuation', 'special'], r'[^\w\s]'),
    (['alphanumeric'], r'\\w+'),
    (['newline', 'newlines'], r'\\n+'),
    (['tab', 'tabs'], r'\\t+'),
]


def replace_backslashes_quotes(prompt: str) -> str:
    """Replaces backslashes and doupble quotes in prompt."""
    return prompt.replace('\\', '\\\\').replace('"', '\\"')


class CallMeMaybe(BaseModel):
    """Class that represents the function processor."""
    llm: LLM
    encoder: Encoder
    functions: dict[str, Function]
    # Tokenized attributes for LLM instructions.
    token_definitions: list[int]
    token_instruction_prefix: list[int]
    token_instruction_suffix: list[int]

    def __init__(self, llm: LLM, functions_definition: str) -> None:
        # The encoder used is given by LLM-object passed on creation.
        encoder: Encoder = llm.encoder

        # Load functions from input JSON file and create a Function-object
        # for each loaded function. Each Function-object has embedded encoder
        # taken from given LLM-object.
        functions: dict = {}
        with open(functions_definition, 'r') as input_file:
            for function in json.load(input_file):
                functions[function['name']] = Function(function, encoder)

        token_definitions = [token
                             for function in functions.values()
                             for token in function.token_definition]
        token_instruction_prefix = encoder.encode(
            '<|im_start|>system\n'
            'You are provided with function signatures '
            'within <tools></tools> XML tags:\n'
            '<tools>\n'
        )
        token_instruction_suffix = encoder.encode(
            '</tools>\n'
            'For each function call, return a json '
            'object within <tool_call></tool_call> tags:\n'
            '<tool_call>\n'
            '{"name": <function-name>, "arguments": <args-json-object>}\n'
            '</tool_call>\n'
            '<|im_end|>\n'
        )

        super().__init__(
            encoder=encoder,
            llm=llm,
            functions=functions,
            token_definitions=token_definitions,
            token_instruction_prefix=token_instruction_prefix,
            token_instruction_suffix=token_instruction_suffix
        )

    def regex_pattern(self, text: str) -> list[int]:
        """Resolves the regex pattern from prompt keywords."""
        words = {word.strip('\'\".,!?').lower() for word in text.split()}
        for key_words, pattern in REGEX_MAPPING:
            if words & set(key_words):
                return self.encoder.encode(pattern)

        match = re.search(r"['\"](\w+)['\"]", text)
        if match:
            return self.encoder.encode(match.group(1))

        return self.encoder.encode(r'\w+')

    def set_tools(self, function: Function | None = None) -> None:
        """Updates the LLM context with function definitions."""
        if function is not None:
            definitions = function.token_definition
        else:
            definitions = self.token_definitions
        updated_definitions = self.token_instruction_prefix + definitions\
            + self.token_instruction_prefix
        self.llm.set_llm_instruction(updated_definitions)

    def add_arguments(self,
                      func: Function,
                      tokens: list[int], text: str) -> list[int]:
        """Prepares all arguments for a function call."""
        for i, arg_name in enumerate(func.params):
            arg_type = func.params[arg_name]
            if i > 0:
                tokens += self.encoder.encode(', ')
            tokens += self.encoder.encode(f'"{arg_name}": ')

            if arg_name == 'regex':
                tokens += self.encoder.encode('"')
                tokens += self.regex_pattern(text)
                tokens += self.encoder.encode('"')
                continue
            if arg_type == 'boolean':
                options = [
                    self.encoder.encode('True'),
                    self.encoder.encode('False')
                ]
            else:
                options = self.encoder.tokens(text)
            if arg_type == 'string':
                tokens += self.encoder.encode('"')
            next_option = self.llm.next_option(tokens, options)
            if arg_type == 'number' or arg_type == 'float':
                parameter = self.encoder.decode(next_option)
                if parameter.isdigit():
                    next_option += self.encoder.encode('.0')
            tokens += next_option
            if arg_type == 'string':
                tokens += self.encoder.encode('"')
        tokens += self.encoder.encode('}\n')

        return tokens

    def process_function(self, prompt: str) -> str:
        prompt = replace_backslashes_quotes(prompt)
        text: str = (
            '<|im_start|>user\n' +
            prompt +
            '\n<|im_end|>\n'
            '<|im_start|>assistant\n'
            '<tool_call>\n'
            '{"name": "'
        )
        tokens = self.encoder.encode(text)
        self.set_tools()

        function_names = [func.token_name for func in self.functions.values()]
        function_name = self.llm.next_option(tokens, function_names)
        decoded_function = self.functions[self.encoder.decode(function_name)]
        tokens += decoded_function.token_name
        tokens += self.encoder.encode('", "arguments": {')
        self.set_tools(decoded_function)
        tokens = self.add_arguments(decoded_function, tokens, prompt)
        tokens += self.encoder.encode('}')

        raw = self.encoder.decode(tokens)
        tool_json = raw[raw.find('{"name":'):]
        data = json.loads(tool_json)

        return (
            '\t{\n'
            f'\t\t"prompt": "{prompt}",\n'
            f'\t\t"name": "{data["name"]}",\n'
            f'\t\t"parameters": "{json.dumps(data["arguments"])}"\n'
            '\t}'
        )
