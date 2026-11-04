from pydantic import BaseModel, PrivateAttr
from llm_sdk import Small_LLM_Model
from src.encoder import Encoder
import numpy as np


class LLM(BaseModel):
    """A class representing LLM."""
    _llm: Small_LLM_Model = PrivateAttr()
    _encoder: Encoder = PrivateAttr()
    _token_instruction: list[int] | None = PrivateAttr()

    def __init__(self, llm: Small_LLM_Model, encoder: Encoder) -> None:
        super().__init__()
        self._llm = llm
        self._encoder = encoder
        self._token_instruction = None

    @property
    def encoder(self) -> Encoder:
        return self._encoder

    def set_llm_instruction(self, instruction: list[int] | str) -> None:
        """Sets the instruction for LLM."""
        if isinstance(instruction, str):
            instruction = self._encoder.encode(instruction)
        self._token_instruction = instruction

    def _apply_mask(self,
                    mask: set[int] | list[int],
                    logits: list[float]) -> list[float]:
        """
        Sets all forbidden token scores to -infinity and
        returns updated list of logits.
        """
        masked = np.full_like(logits, -float('inf'))
        for id in mask:
            masked[id] = logits[id]
        return list(masked)

    def get_logits(self,
                   tokens: list[int],
                   mask: set[int] | None = None) -> list[float]:
        """
        Returns the list of logits for provided tokens.
        """
        instruction: list[int] = self._token_instruction \
            if self._token_instruction is not None \
            else []
        logits: list[float] = self._llm.get_logits_from_input_ids(
            instruction + tokens
        )
        if mask is not None:
            logits = self._apply_mask(mask, logits)
        return logits

    def next_token(self,
                   tokens: list[int],
                   mask: set[int] | None = None) -> int:
        """Returns the next token for the provided tokens."""
        logits: list[float] = self.get_logits(tokens, mask)
        # Computes currently best token token as argmax of all logits.
        best_token: int = int(np.argmax(logits))
        return best_token

    def next_option(self,
                    tokens: list[int],
                    options: list[list[int]]) -> list[int]:
        """Returns the best option from allowed options."""
        best_option: list[int] = []
        context: list[int] = tokens + best_option
        while options:
            allowed_options: set[int] = {option[0] for option in options}
            next_token: int = self.next_token(context, allowed_options)
            best_option.append(next_token)
            context.append(next_token)
            options = [
                option[1:] for option in options
                if option[0] == next_token and len(option) > 1
            ]
        return best_option
