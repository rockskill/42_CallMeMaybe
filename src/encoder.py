from pydantic import BaseModel, PrivateAttr
from typing import Any
import re


WORD_PATTERN = re.compile(r'''
    "(?:\\.|[^"])*" |
    '(?:\\.|[^"])*' |
    \S+
''', re.VERBOSE)


class Encoder(BaseModel):
    """
    Class that represents a custom encoder for LLM.
    
    """
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
        match_id: int | None = None
        match_length: int = -1
        text = convert_for_machine(text)
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

    def tokenize_fragments(self, text: str) -> list[list[int]]:
        """Returns tokenized prompt fragments."""
        indeces: list[list[int]] = []
        # comment
        colon_match = re.search(r':\s*(.+)$', text)
        if colon_match:
            indeces.append(self.encode(colon_match.group(1).strip()))
        # comment
        unescaped = text.replace('\\"', '"')
        fragments = WORD_PATTERN.findall(unescaped)
        for fragment in fragments:
            fragment = fragment.strip('".,!?:;\\')
            fragment = fragment.strip("'")
            if not fragment:
                continue
            indeces.append(self.encode(fragment))
        return indeces

    def decode(self, tokens: list[int] | int) -> str:
        """Translates LLM tokens to human-readable text."""
        if isinstance(tokens, int):
            return self._vocab[tokens] or ''
        return convert_for_human(
            ''.join(self._vocab[t] or '' for t in tokens)
        )


def convert_for_machine(text: str) -> str:
    """
    Replaces spaces, tabs and new line escape sequences
    with dedicated characters for machine processing.
    """
    return text.replace(' ', 'Ġ').replace('\n', 'Ċ').replace('\t', 'ĉ')


def convert_for_human(text: str) -> str:
    """
    Replaces dedicated characters for machine processing
    with spaces, tabs and new line escape sequences.
    """
    return text.replace('Ġ', ' ').replace('Ċ', '\n').replace('ĉ', '\t')
