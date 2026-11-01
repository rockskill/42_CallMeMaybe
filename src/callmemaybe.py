from pydantic import BaseModel
from src.llm import LLM
from src.encoder import Encoder
from src.function import Function
import json


class CallMeMaybe(BaseModel):
    """Class that represents the function processor."""
    llm: LLM
    encoder: Encoder
    functions: dict[str, Function]
    token_definitions: list[int]
    token_instruction_prefix: list[int]
    token_instruction_suffix: list[int]

    def __init__(self, llm: LLM, functions_definition: str) -> None:
        encoder: Encoder = llm.encoder

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

    def process_function(self, prompt: str) -> str:
        text: str = (
            '<|im_start|>user\n' + 
            prompt + 
            '\n<|im_end|>\n'
            '<|im_start|>assistant\n'
            '<tool_call>\n'
            '{"name": "'
        )
        tokens = self.encoder.encode(text)

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
