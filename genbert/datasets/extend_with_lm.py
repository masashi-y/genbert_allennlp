import itertools
import json
import logging
import random
from typing import Optional

import numpy
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import ArrayField
from allennlp.data.tokenizers import Tokenizer
from overrides import overrides

logger = logging.getLogger(__name__)


@DatasetReader.register("extend_with_mlm")
class ExtendWithMaskedLanguageModeling(DatasetReader):
    def __init__(
        self,
        dataset_reader: DatasetReader,
        tokenizer: Tokenizer,
        mlm_train_data: str,
        shuffle: bool = True,
        max_seq_len: Optional[int] = 512,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.dataset_reader = dataset_reader
        self.shuffle = shuffle
        self.mlm_train_data = mlm_train_data
        self.max_seq_len = max_seq_len

    @overrides
    def _read(self, file_path: str):
        instances = self.dataset_reader.read(file_path)

        with open(self.mlm_train_data, "r") as f:
            mlm_instances = []

            for line in f:
                example = json.loads(line)
                mlm_instances.append(
                    self.text_to_instance(
                        example["tokens"],
                        example["masked_lm_positions"],
                        example["masked_lm_labels"],
                    )
                )

            logger.info("loaded %d mlm training instances", len(mlm_instances))

        if self.shuffle:
            logger.info("shuffling mlm training instances")
            random.shuffle(mlm_instances)

        if len(instances) > len(mlm_instances):
            mlm_instances = itertools.cycle(mlm_instances)

        for instance, (input_ids, input_mask, label_ids) in zip(
            instances, mlm_instances
        ):
            instance.add_field("masked_lm_input_ids", input_ids)
            instance.add_field("masked_lm_input_mask", input_mask)
            instance.add_field("masked_lm_label_ids", label_ids)

            yield instance

    @overrides
    def text_to_instance(self, tokens, masked_lm_positions, masked_lm_labels):
        PADDING_INDEX = 0
        seq_length = len(tokens)

        assert self.max_seq_len is None or seq_length <= self.max_seq_len

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        masked_label_ids = self.tokenizer.convert_tokens_to_ids(masked_lm_labels)

        input_field = ArrayField(
            numpy.array(input_ids, dtype=numpy.int), dtype=numpy.int
        )
        mask_field = ArrayField(
            numpy.ones(seq_length, dtype=numpy.bool), dtype=numpy.bool
        )

        lm_label_array = numpy.full(
            seq_length, dtype=numpy.int, fill_value=PADDING_INDEX
        )
        lm_label_array[masked_lm_positions] = masked_label_ids
        lm_label_field = ArrayField(lm_label_array, dtype=numpy.int)

        return input_field, mask_field, lm_label_field
