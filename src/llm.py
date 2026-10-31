from pydantic import BaseModel
from llm_sdk import Small_LLM_Model
from src.encoder import Encoder


class LLM(BaseModel):
    """A class representing LLM"""
    _llm: Small_LLM_Model
    _encoder: Encoder
    _token_instruction: list[int] | None

    def __init__(self, llm: Small_LLM_Model, encoder: Encoder) -> None:
        super().__init__()
        self._llm = llm
        self._encoder = encoder
        self._token_instruction = None

    @property
    def encoder(self) -> Encoder:
        return self._encoder
