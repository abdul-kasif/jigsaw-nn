from operator import index
import re
from collections import Counter
from collections.abc import Iterable

# Special token used by my neural network
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN]


# Recognizes:
# 1. URLs
# 2. Words, including contractions.
# 3. Individual punctuation characters
TOKEN_PATTERN = re.compile(
    r"https?://\S+|www\.\S+|[\w]+(?:['’][\w]+)*|[^\w\s]",
    flags=re.UNICODE,
)


def tokenize(text: str) -> list[str]:
    """
    Convert a comment into a list of tokens.

    Preserves capitalization, punctuation and contractions.
    """

    if not isinstance(text, str):
        raise TypeError("Input text must be a string.")

    return TOKEN_PATTERN.findall(text)


class Vocabulary:
    """
    Maps tokens to Integer IDs and Integer IDs back to tokens.
    """

    def __init__(self, tokens: list[str]):
        self.itos = list(tokens)
        self.stoi = {token: index for index, token in enumerate(self.itos)}

        self.pad_id = self.stoi[PAD_TOKEN]
        self.unk_id = self.stoi[UNK_TOKEN]

    def __len__(self) -> int:
        return len(self.stoi)

    def __contain__(self, token: str) -> bool:
        return token in self.stoi

    def encode(self, tokens: list[str]) -> list[int]:
        """
        Convert tokens into integer IDs.

        Unknown tokens are mapped to UNK_TOKEN.
        """

        return [self.stoi.get(token, self.unk_id) for token in tokens]

    def decode(self, token_ids: list[int]) -> list[str]:
        """
        Convert Integer IDs back into tokens.
        """

        return [
            self.itos[token_id] if 0 <= token_id < len(self.itos) else UNK_TOKEN
            for token_id in token_ids
        ]


def build_vocabulary(
    texts: Iterable[str],
    min_freq: int = 2,
    max_vocab_size: int | None = 30_000,
) -> Vocabulary:
    """
    Build a vocabulary from training text.

    Args:
        texts: Iterable of training comments.
        min_freq: Minimum frequency required for a token.
        max_vocab_size: Maximum total vocabulary size,
                         including special tokens.
                         None means no maximum.

    Returns:
        A Vocabulary instance.
    """

    if min_freq < 1:
        raise ValueError("min_freq must be at least 1.")

    if max_vocab_size is not None and max_vocab_size < len(SPECIAL_TOKENS):
        raise ValueError(f"max_vocab_size must be at least {len(SPECIAL_TOKENS)}.")

    token_counts = Counter()

    for text in texts:
        token_counts.update(tokenize(text))

    # Filter out infrequent tokens.
    filtered_tokens = [
        (token, frequency)
        for token, frequency in token_counts.items()
        if frequency >= min_freq
    ]

    # Sort by descending frequency
    # Alphabetical order breaks ties deterministically.
    filtered_tokens.sort(key=lambda item: [-item[1], item[0]])

    # Reserve space for special tokens.
    if max_vocab_size is not None:
        available_slots = max_vocab_size - len(SPECIAL_TOKENS)
        filtered_tokens = filtered_tokens[:available_slots]

    vocabulary_tokens = SPECIAL_TOKENS + [token for token, _ in filtered_tokens]

    return Vocabulary(vocabulary_tokens)
