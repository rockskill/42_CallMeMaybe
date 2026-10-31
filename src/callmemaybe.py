from pydantic import BaseModel
from src.llm import LLM
from src.encoder import Encoder
import json


class CallMeMaybe(BaseModel):
    """Class that represents the function processor."""
    def __init__(self, llm: LLM, functions_definition: str) -> None:
        encoder: Encoder = llm.encoder

        functions: dict = {}
        with open(functions_definition, 'r') as input_file:
            for function in json.load(input_file):
                functions[function['name']] = Function(function, encoder)

        token_definitions = None
        token_instruction_prefix = None
        token_instruction_suffix = None

        super().__init__(
            encoder=encoder,
            llm=llm,
            functions=functions,
            token_definitions=token_definitions,
            token_instruction_prefix=token_instruction_prefix,
            token_instruction_suffix=token_instruction_suffix
        )
