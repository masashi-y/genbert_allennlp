import html
import json
import logging
import random
from typing import Any, Dict, List, Optional, Tuple

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import (
    Field, LabelField, ListField, MetadataField, SpanField, TextField)
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token
from overrides import overrides

from genbert.datasets.utils import (
    extract_answer_info_from_annotation, find_valid_spans,
)

from genbert.digits_aware_tokenizer import DigitsAwareTransformerTokenizer
from genbert.pretrained_transformer_indexer import PretrainedTransformerIndexer
from genbert.util import (ANSWER_GENERATION, ANSWER_SPAN, END_SYMBOL, SEP,
                          START_SYMBOL)

logger = logging.getLogger(__name__)


@DatasetReader.register("my_drop")
class DropReader(DatasetReader):

    def __init__(
        self,
        transformer_model_name: str = "bert-base-cased",
        max_number_of_spans: int = 6,
        max_input_sequence_length: int = 512,
        max_decode_sequence_length: int = 20,
        include_more_numbers: bool = False,
        max_instances: int = None,
        shuffle: bool = False,
        ** kwargs,
    ) -> None:
        super().__init__(**kwargs)

        # this dataset reader works with this tokenizer only
        self._tokenizer = DigitsAwareTransformerTokenizer(
            transformer_model_name, include_more_numbers=include_more_numbers
        )
        self._token_indexers = {
            "tokens": PretrainedTransformerIndexer(
                transformer_model_name, namespace="tokens"
            )
        }
        self.max_input_sequence_length = max_input_sequence_length
        self.max_decode_sequence_length = max_decode_sequence_length
        self.include_more_numbers = include_more_numbers

        self.max_number_of_spans = max_number_of_spans
        self.max_instances = max_instances
        self.shuffle = shuffle
        self._sep_tokens = self._tokenizer.sequence_pair_mid_tokens
        self._num_start_tokens = len(
            self._tokenizer.sequence_pair_start_tokens)
        self._num_mid_tokens = len(self._tokenizer.sequence_pair_mid_tokens)
        self._num_end_tokens = len(self._tokenizer.sequence_pair_end_tokens)

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        logger.info("Reading file at %s", file_path)
        with open(file_path) as dataset_file:
            dataset = json.load(dataset_file)
        logger.info("Reading the dataset")
        kept_count, skip_count = 0, 0
        examples = dataset.items()

        if self.shuffle:
            logger.info("Shuffling the dataset")
            examples = list(examples)
            random.shuffle(examples)

        def process(passage_id, passage_text, question_answer):

            passage_text = html.unescape(passage_text)
            question_id = question_answer["query_id"]
            question_text = html.unescape(
                question_answer["question"].strip())
            answer_annotations = []
            if "answer" in question_answer:
                answer_annotations.append(question_answer["answer"])
            if "validated_answers" in question_answer:
                answer_annotations += question_answer["validated_answers"]

            if self.max_instances is not None and kept_count >= self.max_instances:
                return None

            preprocessed_dict = self.preprocess(
                question_text,
                passage_text,
                question_id,
                passage_id,
                answer_annotations,
            )

            if preprocessed_dict is not None:
                instance = self.text_to_instance(**preprocessed_dict)
                if instance is not None:
                    return instance

            return None

        iterator = (
            (passage_id, passage_info["passage"], question_answer)
            for passage_id, passage_info in examples
            for question_answer in passage_info["qa_pairs"]
        )

        instances = (process(*arg) for arg in iterator)

        for instance in instances:
            if instance is not None:
                kept_count += 1
                yield instance
            else:
                skip_count += 1

        logger.info(
            "Skipped %d questions, kept %d questions.", skip_count, kept_count
        )

    def preprocess(
        self,  # type: ignore
        question_text: str,
        passage_text: str,
        question_id: str = None,
        passage_id: str = None,
        answer_annotations: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:

        passage_tokens = self._tokenizer.str_to_tokens(passage_text)
        question_tokens = self._tokenizer.str_to_tokens(question_text)

        answer_type: str = None
        answer_texts: List[str] = []
        if answer_annotations:
            # Currently we only use the first annotated answer here, but actually
            # this doesn't affect the training, because we only have one annotation
            # for the train set.
            answer_type, answer_texts = extract_answer_info_from_annotation(
                answer_annotations[0]
            )

        if answer_type is None:
            return None

        # Tokenize the answer text in order to find the matched span based on token
        tokenized_answer_texts = [
            " ".join(
                token.text for token in self._tokenizer.str_to_tokens(answer_text))
            for answer_text in answer_texts
        ]

        unique_answer_texts = sorted(
            set(tokenized_answer_texts), key=tokenized_answer_texts.index
        )
        passage_text_as_key = " ".join(token.text for token in passage_tokens)
        normalized_answer_texts = sorted(
            unique_answer_texts, key=passage_text_as_key.find
        )

        answer_as_generation = self._make_answer_as_generation(
            normalized_answer_texts)

        if tokenized_answer_texts:
            valid_question_spans = find_valid_spans(
                question_tokens, tokenized_answer_texts
            )
            valid_passage_spans = find_valid_spans(
                passage_tokens, tokenized_answer_texts
            )
        else:
            valid_question_spans, valid_passage_spans = [], []

        question_tokens, question_mapping = self._tokenizer.tokens_to_wordpieces(
            question_tokens, with_mapping=True, skip_tokens=self._sep_tokens
        )

        question_offset = self._num_start_tokens
        valid_question_spans = [
            (
                question_mapping[start].start + question_offset,
                question_mapping[end].end + question_offset,
            )
            for start, end in valid_question_spans
        ]

        passage_tokens, passage_mapping = self._tokenizer.tokens_to_wordpieces(
            passage_tokens, with_mapping=True, skip_tokens=self._sep_tokens
        )

        passage_offset = (
            self._num_start_tokens +
            len(question_tokens) + self._num_mid_tokens
        )
        valid_passage_spans = [
            (
                passage_mapping[start].start + passage_offset,
                passage_mapping[end].end + passage_offset,
            )
            for start, end in valid_passage_spans
        ]

        assert (
            self.max_input_sequence_length is None
            or self.max_input_sequence_length > len(question_tokens)
        ), "`max_input_sequence_length` value is too small"

        if self.max_input_sequence_length is not None:
            passage_length_limit = (
                self.max_input_sequence_length
                - self._num_start_tokens
                - self._num_mid_tokens
                - self._num_end_tokens
                - len(question_tokens)
            )

            passage_tokens = passage_tokens[:passage_length_limit]
            valid_passage_spans = [
                (start, end)
                for start, end in valid_passage_spans
                if start < self.max_input_sequence_length - self._num_end_tokens
                and end < self.max_input_sequence_length - self._num_end_tokens
            ]

        # these are used in the evaluation, to extract the spans from `passage_text`
        # and not `passage_tokens`. This ensures that the extracted spans are not
        # pre-processed (e.g. lowercasing) and helps perform the evaluation correctly
        passage_question_text = question_text + " " + passage_text
        passage_offset = len(question_text) + 1
        token_offsets = (
            [(-1, -1)] * self._num_start_tokens
            + [(token.idx, token.idx_end) for token in question_tokens]
            + [(-1, -1)] * self._num_mid_tokens
            + [
                (token.idx + passage_offset, token.idx_end + passage_offset)
                if token not in self._sep_tokens
                else (-1, -1)
                for token in passage_tokens
            ]
            + [(-1, -1)] * self._num_end_tokens
        )

        metadata = {
            "question_id": question_id,
            "passage_id": passage_id,
            "original_passage": passage_text,
            "original_question": question_text,
            "original_passage_question": passage_question_text,
            "token_offsets": token_offsets,
            "question_tokens": [token.text for token in question_tokens],
            "passage_tokens": [token.text for token in passage_tokens],
            "tokenized_answer_texts": tokenized_answer_texts,
            "answer_annotations": answer_annotations,
            "answer_info": {
                "answer_texts": answer_texts,
                "answer_passage_spans": valid_passage_spans,
                "answer_question_spans": valid_question_spans,
                "answer_generation": answer_as_generation,
            },
        }

        output_dict = {
            "question_tokens": question_tokens,
            "passage_tokens": passage_tokens,
            "valid_question_spans": valid_question_spans,
            "valid_passage_spans": valid_passage_spans,
            "answer_as_generation": answer_as_generation,
            "answer_annotations": answer_annotations,
            "metadata": metadata,
        }

        return output_dict

    def _make_answer_as_generation(self, answer_texts: List[str]) -> List[Token]:

        results = [
            Token(
                START_SYMBOL,
                text_id=self._tokenizer.convert_tokens_to_ids(START_SYMBOL),
            )
        ]
        for i, answer in enumerate(answer_texts):
            if 0 < i < len(answer_texts) - 1:
                results.append(
                    Token(text=SEP, text_id=self._tokenizer.convert_tokens_to_ids(SEP))
                )
            results.extend(self._tokenizer.str_to_wordpieces(answer))

        results.append(
            Token(END_SYMBOL, text_id=self._tokenizer.convert_tokens_to_ids(END_SYMBOL))
        )

        return results[: self.max_decode_sequence_length]

    @ overrides
    def text_to_instance(
        self,  # type: ignore
        answer_annotations: List[Dict] = None,
        question_tokens: List[Token] = None,
        passage_tokens: List[Token] = None,
        valid_question_spans: List[Tuple[int, int]] = None,
        valid_passage_spans: List[Tuple[int, int]] = None,
        answer_as_generation: List[Token] = None,
        metadata: Dict[str, Any] = None,
    ) -> Instance:

        fields: Dict[str, Field] = {}

        question_passage_field = TextField(
            self._tokenizer.add_special_tokens(
                question_tokens, passage_tokens),
            self._token_indexers,
        )

        fields["question_with_context"] = question_passage_field

        if answer_annotations is not None:
            answer_as_span = (
                len(valid_question_spans) > 0 or len(valid_passage_spans) > 0
            )
            fields["answer_head_types"] = LabelField(
                ANSWER_SPAN if answer_as_span else ANSWER_GENERATION, skip_indexing=True
            )

            passage_span_fields = [
                SpanField(span[0], span[1], question_passage_field)
                for span in valid_passage_spans[: self.max_number_of_spans]
            ]
            if not passage_span_fields:
                passage_span_fields.append(
                    SpanField(-1, -1, question_passage_field))
            fields["answer_as_passage_spans"] = ListField(passage_span_fields)

            question_span_fields = [
                SpanField(span[0], span[1], question_passage_field)
                for span in valid_question_spans[: self.max_number_of_spans]
            ]
            if not question_span_fields:
                question_span_fields.append(
                    SpanField(-1, -1, question_passage_field))
            fields["answer_as_question_spans"] = ListField(
                question_span_fields)

            fields["answer_as_generation"] = TextField(
                answer_as_generation, self._token_indexers
            )

        fields["metadata"] = MetadataField(metadata)

        return Instance(fields)


if __name__ == "__main__":
    import argparse
    import pickle
    from pathlib import Path
    from allennlp.common.params import Params

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "param_path",
        help=(
            "path to parameter file describing the model to be "
            "trained (only `dataset_reader` key is used)"
        ),
    )
    parser.add_argument(
        "--input-file",
        help="input file to read. if not specified, use all paths specified in param_path",
    )
    parser.add_argument(
        "--output-file",
        help="name of pickle output file. used only when --input-file is specified.",
    )

    parser.add_argument(
        "--overrides",
        help=(
            "a json(net) structure used to override the experiment configuration, "
            "e.g., '{\"iterator.batch_size\": 16}'. Nested parameters can be "
            "specified either with nested dictionaries or with dot syntax."
        ),
    )

    args = parser.parse_args()

    if args.overrides is not None:
        params_overrides = json.loads(args.overrides)
    else:
        params_overrides = {}

    params = Params.from_file(
        args.param_path, params_overrides=params_overrides)

    logging.basicConfig(level=logging.INFO)

    if args.input_file:
        logger.info('using "dataset_reader" params')
        reader = DatasetReader.from_params(params.pop("dataset_reader"))

        logging.info(
            "%s is specified. reading instances from the file", args.input_file
        )
        instances = reader.read(args.input_file)

        out_path = args.output_file or Path(
            args.input_file).with_suffix(".pickle")
        logging.info("writing result instances to %s", out_path)
        with open(out_path, "wb") as f:
            pickle.dump(instances, f)

    else:
        logging.info(
            "--input-file is not specified. searching files in param file")

        logger.info('using "dataset_reader" params')
        default_reader = DatasetReader.from_params(
            params.pop("dataset_reader"))

        if "validation_dataset_reader" in params:
            logger.info('using "validation_dataset_reader" params')
            validation_reader = DatasetReader.from_params(
                params.pop("validation_dataset_reader")
            )
        else:
            validation_reader = default_reader

        for reader, data_path in (
            (default_reader, "train_data_path"),
            (validation_reader, "validation_data_path"),
            (validation_reader, "test_data_path"),
        ):
            if data_path in params:
                logging.info('found "%s": "%s"', data_path, params[data_path])

                instances = reader.read(params[data_path])

                out_path = Path(params[data_path]).with_suffix(".pickle")
                logging.info("writing result instances to %s", out_path)
                with out_path.open("wb") as f:
                    pickle.dump(list(instances), f)
