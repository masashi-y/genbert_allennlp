import dataclasses
import logging
from typing import Callable, List, Optional, Tuple, Union

from allennlp.data.tokenizers import (PretrainedTransformerTokenizer,
                                      SpacyTokenizer, Token, Tokenizer)

from genbert.datasets.utils import convert_word_to_number
from genbert.util import split_tokens_by_hyphen

# from allennlp.common.util import sanitize_wordpiece


logger = logging.getLogger(__name__)


@dataclasses.dataclass
class TokenIndexMapper:
    start: int
    end: int


def ensure_token_indices(
    tokens: List[Token], start_end_indices: Optional[Tuple[int, int]] = None,
) -> List[Token]:

    new_tokens = []
    offset = 0
    for token in tokens:
        length = len(token.text)
        if token.text.startswith("##"):
            length -= 2

        if start_end_indices is not None:
            start_index, end_index = start_end_indices
        else:
            start_index = offset
            end_index = offset + length

        new_tokens.append(
            dataclasses.replace(token, idx=start_index, idx_end=end_index)
        )
        offset += length
    return new_tokens


def split_digits_into_chars(
    tokens: List[Token],
    convert_token_to_ids: Optional[Callable[[str], int]] = None,
    unk_id: Optional[int] = None,
) -> List[Token]:

    digits = set("#0123456789")
    new_tokens = []
    for token in tokens:
        chars = set(token.text)
        if chars.issubset(digits) and chars != {"#"}:
            start_index = token.idx or 0
            end_index = token.idx_end or len(token.text)
            for index, digit in enumerate(token.text.replace("#", "")):
                if token.text.startswith("##") or index > 0:
                    digit = "##" + digit
                token = Token(digit, idx=start_index, idx_end=end_index)
                if convert_token_to_ids is not None:
                    token.text_id = convert_token_to_ids(digit)
                    assert unk_id is None or token.text_id != unk_id
                new_tokens.append(token)
        else:
            new_tokens.append(token)
    return new_tokens


@Tokenizer.register("digits_aware")
class DigitsAwareTransformerTokenizer(PretrainedTransformerTokenizer):
    def __init__(
        self, transformer_model_name, include_more_numbers: bool = False
    ) -> None:
        super().__init__(
            transformer_model_name,
            add_special_tokens=False,
            tokenizer_kwargs={"do_lower_case": True},
        )

        self._word_tokenizer = SpacyTokenizer()

        self.include_more_numbers = include_more_numbers
        self._unk_id = self.convert_tokens_to_ids("[UNK]")

    def convert_tokens_to_ids(
        self, tokens: Union[Token, List[Token]]
    ) -> Union[Token, List[Token]]:
        result = self.tokenizer.convert_tokens_to_ids(tokens)
        return result

    def normalize_digits(self, tokens: List[Token],) -> List[Token]:
        new_tokens = []
        for token in tokens:
            number = convert_word_to_number(
                token.text, self.include_more_numbers)
            if number is not None:
                new_tokens.append(dataclasses.replace(token, text=str(number)))
            else:
                new_tokens.append(token)
        return new_tokens

    def str_to_tokens(self, text) -> List[Token]:
        tokens = self._word_tokenizer.tokenize(text)
        tokens = split_tokens_by_hyphen(tokens)
        tokens = self.normalize_digits(tokens)
        return tokens

    def str_to_wordpieces(self, text_or_token) -> List[Token]:
        if isinstance(text_or_token, Token):
            text = text_or_token.text
        else:
            text = text_or_token

        wordpieces = super().tokenize(text)

        if isinstance(text_or_token, Token):

            if text_or_token.idx is not None and text_or_token.idx_end is not None:
                start_end_indices = (text_or_token.idx, text_or_token.idx_end)
            else:
                start_end_indices = None

            wordpieces = ensure_token_indices(
                wordpieces, start_end_indices=start_end_indices
            )

        wordpieces = split_digits_into_chars(
            wordpieces, self.convert_tokens_to_ids, self._unk_id
        )
        return wordpieces

    def tokens_to_wordpieces(
        self,
        tokens: List[Token],
        with_mapping: bool = False,
        skip_tokens: List[Token] = None,
    ) -> Union[List[Token], Tuple[List[Token], str]]:

        mapping = []
        new_tokens = []
        for token in tokens:
            start = len(new_tokens)
            if skip_tokens is not None and token in skip_tokens:
                new_tokens.append(token)
            else:
                new_tokens.extend(self.str_to_wordpieces(token))
            mapping.append(TokenIndexMapper(start, len(new_tokens) - 1))

        if with_mapping:
            return new_tokens, mapping
        return new_tokens

    def tokenize(self, text: str) -> Union[List[Token], Tuple[List[Token], str]]:
        tokens = self.str_to_tokens(text)
        wordpieces = self.tokens_to_wordpieces(tokens)
        return wordpieces
