from pydantic import BaseModel, PrivateAttr
from typing import Any
import re


WORD_PATTERN = re.compile(r'''
    "(?:\\.|[^"])*" |
    '(?:\\.|[^"])*' |
    \S+
''', re.VERBOSE)


class Encoder(BaseModel):
    """Class that represents a custom encoder for LLM."""
    _vocab: list[str | None] = PrivateAttr()
    _trie: dict[str, Any] = PrivateAttr()

    def __init__(self, tokens: dict[str, int]) -> None:
        vocab: list[str | None] = [None] * len(tokens)
        trie: dict[str, Any] = {}
        for word, token in tokens.items():
            vocab[token] = word
            node: dict[str, Any] = trie
            for c in word:
                node = node.setdefault(c, {})
            node['token'] = token
        super().__init__()
        self._vocab = vocab
        self._trie = trie

    def encode(self, text: str) -> list[int]:
        """Translates human text to a list of tokens for LLM."""
        indeces: list[int] = []
        node: dict[str, Any] = {}
        i: int = 0
        j: int = 0
        match_id: int | None
        match_length: int
        while i < len(text):
            node = self._trie
            match_id = None
            match_length = -1
            j = i
            while j < len(text) and text[j] in node:
                node = node[text[j]]
                j += 1
                if 'token' in node:
                    match_id = node['token']
                    match_length = j - i
            if match_id is not None:
                indeces.append(match_id)
                i += match_length
            else:
                i += 1
        return indeces

    def encode_words(self, text: str) -> set[int]:
        """Returns all possible tokens from the string."""
        indeces: set[int] = set()
        words: list[str] = WORD_PATTERN.findall(text)
        for word in words:
            word = word.strip('.,!?')
            word = word.strip('"\'')
            if not word:
                continue
            for token_id in self.encode(word):
                indeces.add(token_id)
            indeces.add(self.encode(' ' + word)[0])
        return indeces
