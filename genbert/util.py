from typing import List, Optional, Union

import numpy
import torch
from allennlp.data import TextFieldTensors
from allennlp.data.tokenizers import Token
from allennlp.nn.util import (get_text_field_mask,
                              get_token_ids_from_text_field_tensors)


# I follow the original implementation of using these symbols
# representing START/END_SYMBOL, since bert's vocab doesn't contain `@start@`
START_SYMBOL, END_SYMBOL, SEP, DUMMY = "@", "\\", ";", "!"

ANSWER_GENERATION, ANSWER_SPAN = range(2)


def detokenize(tokens: List[str]) -> List[str]:
    text = " ".join(tokens)

    # De-tokenize WordPieces that have been split off.
    text = text.replace(" ##", "").replace("##", "")

    # Clean whitespace
    return " ".join(text.strip().split())


def post_process_decoded_output(text: str) -> str:
    # remove space around decimal point
    processed = ".".join(x.strip() for x in text.split("."))
    try:
        # '.' is a decimal only if final str is a number
        float(processed)
    except ValueError:
        processed = text

    # remove space around "-"
    result = "-".join(x.strip() for x in processed.split("-"))
    return result


def split_token_by_delimiter(token: Token, delimiter: str) -> List[Token]:
    """
    This is modified by the original AllenNLP implementation so this
    explicitly specifies `idx_end` field in Token class.
    """
    split_tokens = []
    char_offset = token.idx
    for sub_str in token.text.split(delimiter):
        if sub_str:
            split_tokens.append(
                Token(text=sub_str, idx=char_offset,
                      idx_end=char_offset + len(sub_str))
            )
            char_offset += len(sub_str)
        split_tokens.append(
            Token(text=delimiter, idx=char_offset,
                  idx_end=char_offset + len(delimiter))
        )
        char_offset += len(delimiter)
    if split_tokens:
        split_tokens.pop(-1)
        char_offset -= len(delimiter)
        return split_tokens
    return [token]


def split_tokens_by_hyphen(tokens: List[Token]) -> List[Token]:
    hyphens = ["-", "–", "~"]
    new_tokens: List[Token] = []

    for token in tokens:
        if any(hyphen in token.text for hyphen in hyphens):
            unsplit_tokens = [token]
            split_tokens: List[Token] = []
            for hyphen in hyphens:
                for unsplit_token in unsplit_tokens:
                    if hyphen in token.text:
                        split_tokens += split_token_by_delimiter(
                            unsplit_token, hyphen)
                    else:
                        split_tokens.append(unsplit_token)
                unsplit_tokens, split_tokens = split_tokens, []
            new_tokens += unsplit_tokens
        else:
            new_tokens.append(token)

    return new_tokens


def to_numpy(x: Union[numpy.ndarray, torch.Tensor]) -> numpy.ndarray:
    if isinstance(x, numpy.ndarray):
        return x
    return x.detach().cpu().numpy()


def get_token_type_ids_from_text_field_tensors(
    text_field_tensors: TextFieldTensors,
    sep_token_index: Optional[int] = None,
    mask: Optional[torch.BoolTensor] = None,
) -> torch.LongTensor:

    for _, indexer_tensors in text_field_tensors.items():
        for argument_name, tensor in indexer_tensors.items():
            if argument_name == "type_ids":
                return tensor

    token_ids = get_token_ids_from_text_field_tensors(text_field_tensors)
    if mask is None:
        mask = get_text_field_mask(text_field_tensors)

    assert (
        len(token_ids.size()) == 2
        and len(mask.size()) == 2
        and sep_token_index is not None
        and mask is not None
    ), (
        "`type_ids` entry not found in the TextFieldTensors. "
        "`sep_token_index` and `mask` may need to be explicitly passed"
    )

    # find where the first [SEP] token is, and assign ones at indices after that.
    onehot_sep_tokens = (token_ids == sep_token_index).long()
    type_ids = (onehot_sep_tokens.cumsum(
        dim=-1) - onehot_sep_tokens > 0).long()
    type_ids = type_ids * mask.long()
    return type_ids
