from pydantic import BaseModel, PrivateAttr
from typing import Any
from src.encoder import Encoder
import json


class Function(BaseModel):
    """
    A class that represents function that can be called.
    It has name, description and parameters attributes,
    and also attributes holding its tokenized versions.
    The very last attribute holds the tokenized version
    for the definition of the whole function. It is built
    up from tokenized function name, tokenized function
    description and tokenized function parameters.
    """
    _name: str = PrivateAttr()
    _token_name: list[int] = PrivateAttr()
    _description: str = PrivateAttr()
    _token_description: list[int] = PrivateAttr()
    _params: dict[str, str] = PrivateAttr()
    _token_params: dict[str, list[int]] = PrivateAttr()
    _token_definition: list[int] = PrivateAttr()

    def __init__(self,
                 function: dict[str, Any],
                 encoder: Encoder) -> None:
        """
        Function constructor method for Function objects creation.
        It uses an encoder passed as a parameter to create tokenized
        forms of function name, function description and function parameters.
        """
        super().__init__()
        self._name: str = function['name']
        self._token_name: list[int] = encoder.encode(self._name)
        self._description: str = function.get('description', '')
        self._token_description: list[int] = encoder.encode(self._description)
        self._params = {
            key: value['type']
            for key, value in function['parameters'].items()
        }
        self._token_params = {
            key: encoder.encode(value['type'])
            for key, value in function['parameters'].items()
        }
        self._token_definition = encoder.encode(self._convert_to_tool_schema())

    def _convert_to_tool_schema(self) -> str:
        return json.dumps({
            "name": self._name,
            "description": self._description,
            "parameters": {
                "type": "object",
                "properties": {
                    key: {"type": value}
                    for key, value in self._params.items()
                },
                "required": list(self._params.keys())
            }
        })

    @property
    def name(self) -> str:
        return self._name

    @property
    def token_name(self) -> list[int]:
        return self._token_name

    @property
    def description(self) -> str:
        return self._description

    @property
    def token_description(self) -> list[int]:
        return self._token_description

    @property
    def params(self) -> dict[str, str]:
        return self._params

    @property
    def token_params(self) -> dict[str, list[int]]:
        return self._token_params

    @property
    def param_names(self) -> list[str]:
        return list(self._params.keys())

    @property
    def token_definition(self) -> list[int]:
        return self._token_definition
