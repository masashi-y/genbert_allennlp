import logging
import os
import pickle
from typing import List, Optional, Union

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from overrides import overrides

logger = logging.getLogger(__name__)


@DatasetReader.register("pickled")
class PickleReader(DatasetReader):
    def __init__(self, use_only_first_examples: Optional[Union[int, List[int]]] = None):
        super().__init__()
        self.use_only_first_examples = use_only_first_examples or -1

    @overrides
    def _read(self, file_paths: str):

        file_paths = file_paths.split(",")

        if isinstance(self.use_only_first_examples, int):
            use_only_first_examples = [self.use_only_first_examples]
        else:
            use_only_first_examples = self.use_only_first_examples

        assert len(file_paths) == len(
            use_only_first_examples
        ), "number of elements in `file_path` and `use_only_first_examples` must be the same"

        instances = []
        for file_path, max_instance in zip(file_paths, use_only_first_examples):

            logger.info("Reading file at %s", file_path)
            with open(os.path.join(file_path), "rb") as dataset_file:
                this_instances = pickle.load(dataset_file)
                if max_instance >= 0:
                    logger.info("Use only first %d examples", max_instance)
                    this_instances = this_instances[:max_instance]
                instances.extend(this_instances)

        return instances

    @overrides
    def text_to_instance(self) -> Instance:
        pass
