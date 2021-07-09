import logging
from typing import Dict, List, Optional, Tuple

import torch
from allennlp.common.util import pad_sequence_to_length
from allennlp.data.token_indexers.token_indexer import (IndexedTokenList,
                                                        TokenIndexer)
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary
from overrides import overrides

logger = logging.getLogger(__name__)


@TokenIndexer.register("my_pretrained_transformer")
class PretrainedTransformerIndexer(TokenIndexer):
    def __init__(
        self, model_name: str, namespace: str = "tags", max_length: int = None, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self._namespace = namespace
        self._model_name = model_name
        allennlp_tokenizer = PretrainedTransformerTokenizer(model_name)
        self._added_to_vocabulary = False

        self._num_added_start_tokens = len(
            allennlp_tokenizer.single_sequence_start_tokens
        )
        self._num_added_end_tokens = len(
            allennlp_tokenizer.single_sequence_end_tokens)
        self.pad_token_id = allennlp_tokenizer.tokenizer.pad_token_id
        self._max_length = max_length
        if self._max_length is not None:
            num_added_tokens = len(allennlp_tokenizer.tokenize("a")) - 1
            self._effective_max_length = (  # we need to take into account special tokens
                self._max_length - num_added_tokens
            )
            if self._effective_max_length <= 0:
                raise ValueError(
                    "max_length needs to be greater than the number of special tokens inserted."
                )

    def _add_encoding_to_vocabulary_if_needed(self, vocab: Vocabulary) -> None:
        """
        Copies tokens from ```transformers``` model's vocab to the specified namespace.
        """
        if self._added_to_vocabulary:
            return

        allennlp_tokenizer = PretrainedTransformerTokenizer(self._model_name)
        tokenizer = allennlp_tokenizer.tokenizer

        try:
            vocab_items = tokenizer.get_vocab().items()
        except NotImplementedError:
            vocab_items = (
                (tokenizer.convert_ids_to_tokens(idx), idx)
                for idx in range(tokenizer.vocab_size)
            )
        for word, idx in vocab_items:
            vocab._token_to_index[self._namespace][word] = idx
            vocab._index_to_token[self._namespace][idx] = word

        self._added_to_vocabulary = True

    @overrides
    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]):
        # If we only use pretrained models, we don't need to do anything here.
        pass

    @overrides
    def tokens_to_indices(
        self, tokens: List[Token], vocabulary: Vocabulary
    ) -> IndexedTokenList:

        self._add_encoding_to_vocabulary_if_needed(vocabulary)

        indices, type_ids = self._extract_token_and_type_ids(tokens)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        output: IndexedTokenList = {
            "token_ids": indices,
            "mask": [True] * len(indices),
            "type_ids": type_ids,
        }

        return self._postprocess_output(output)

    @overrides
    def indices_to_tokens(
        self, indexed_tokens: IndexedTokenList, vocabulary: Vocabulary
    ) -> List[Token]:
        token_ids = indexed_tokens["token_ids"]
        type_ids = indexed_tokens.get("type_ids")

        return [
            Token(
                text=vocabulary.get_token_from_index(
                    token_ids[i], self._namespace),
                text_id=token_ids[i],
                type_id=type_ids[i] if type_ids is not None else None,
            )
            for i in range(len(token_ids))
        ]

    def _extract_token_and_type_ids(
        self, tokens: List[Token]
    ) -> Tuple[List[int], Optional[List[int]]]:
        """
        Roughly equivalent to `zip(*[(token.text_id, token.type_id) for token in tokens])`,
        with some checks.
        """
        indices: List[int] = []
        type_ids: List[int] = []
        for token in tokens:
            if getattr(token, "text_id", None) is not None:
                # `text_id` being set on the token means that we aren't using the vocab, we just use
                # this id instead. Id comes from the pretrained vocab.
                # It is computed in PretrainedTransformerTokenizer.
                indices.append(token.text_id)
            else:
                raise KeyError(
                    "Using PretrainedTransformerIndexer but field text_id is not set"
                    f" for the following token: {token.text}"
                )

            if type_ids is not None and getattr(token, "type_id", None) is not None:
                type_ids.append(token.type_id)
            else:
                type_ids.append(0)

        return indices, type_ids

    def _postprocess_output(self, output: IndexedTokenList) -> IndexedTokenList:
        """
        Takes an IndexedTokenList about to be returned by `tokens_to_indices()` and adds any
        necessary postprocessing, e.g. long sequence splitting.

        The input should have a `"token_ids"` key corresponding to the token indices. They should
        have special tokens already inserted.
        """
        if self._max_length is not None:
            raise NotImplementedError(
                "this version of PretrainedTransformerIndexer does not support max_length"
            )

        return output

    @overrides
    def get_empty_token_list(self) -> IndexedTokenList:
        output: IndexedTokenList = {
            "token_ids": [], "mask": [], "type_ids": []}
        if self._max_length is not None:
            output["segment_concat_mask"] = []
        return output

    @overrides
    def as_padded_tensor_dict(
        self, tokens: IndexedTokenList, padding_lengths: Dict[str, int]
    ) -> Dict[str, torch.Tensor]:
        tensor_dict = {}
        for key, val in tokens.items():
            if key == "type_ids":
                padding_value = 0
                mktensor = torch.LongTensor
            elif key == "mask" or key == "wordpiece_mask":
                padding_value = False
                mktensor = torch.BoolTensor
            elif len(val) > 0 and isinstance(val[0], bool):
                padding_value = False
                mktensor = torch.BoolTensor
            else:
                padding_value = self.pad_token_id
                if padding_value is None:
                    # Some tokenizers don't have padding tokens and rely on the mask only.
                    padding_value = 0
                mktensor = torch.LongTensor

            tensor = mktensor(
                pad_sequence_to_length(
                    val, padding_lengths[key], default_value=lambda: padding_value
                )
            )

            tensor_dict[key] = tensor
        return tensor_dict

    def __eq__(self, other):
        if isinstance(other, PretrainedTransformerIndexer):
            for key in self.__dict__:
                if self.__dict__[key] != other.__dict__[key]:
                    return False
            return True
        return NotImplemented
