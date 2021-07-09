import math
import string
from typing import Any, Dict, List, Optional, Tuple

from allennlp.common.util import sanitize_wordpiece
from allennlp.data.tokenizers import Token
from allennlp_models.rc.dataset_readers.drop import WORD_NUMBER_MAP, DropReader
from word2number.w2n import word_to_num


def extract_answer_info_from_annotation(
    answer_annotation: Dict[str, Any]
) -> Tuple[str, List[str]]:

    answer_type = None
    if answer_annotation["spans"]:
        answer_type = "spans"
    elif answer_annotation["number"]:
        answer_type = "number"
    elif any(answer_annotation["date"].values()):
        answer_type = "date"

    answer_content = answer_annotation[answer_type] if answer_type is not None else None

    answer_texts: List[str] = []
    if answer_type is None:  # No answer
        pass
    elif answer_type == "spans":
        # answer_content is a list of string in this case
        answer_texts = answer_content
    elif answer_type == "date":
        # answer_content is a dict with "month", "day", "year" as the keys
        date_tokens = [
            answer_content[key]
            for key in ["month", "day", "year"]
            if key in answer_content and answer_content[key]
        ]
        answer_texts = [" ".join(date_tokens)]
    elif answer_type == "number":
        # answer_content is a string of number
        answer_texts = [answer_content]
    return answer_type, answer_texts


# def convert_word_to_number(word: str, try_to_include_more_numbers=False):
#     return DropReader.convert_word_to_number(word, try_to_include_more_numbers)
def convert_word_to_number(
    word: str, try_to_include_more_numbers: bool = False, sanitize: bool = False,
) -> Optional[int]:
    """
    Currently we only support limited types of conversion.
    """

    if sanitize:
        word = sanitize_wordpiece(word)

    if try_to_include_more_numbers:
        # strip all punctuations from the sides of the word, except for the negative sign
        punctruations = string.punctuation.replace("-", "")
        word = word.strip(punctruations)
        # some words may contain the comma as deliminator
        word = word.replace(",", "")
        # word2num will convert hundred, thousand ... to number, but we skip it.
        if word in ["hundred", "thousand", "million", "billion", "trillion"]:
            return None
        try:
            number = word_to_num(word)
        except ValueError:
            try:
                number = int(word)
                if number == 0:
                    number = None
            except ValueError:
                try:
                    number = float(word)
                    if not math.isfinite(number) or math.isclose(number, 0.0):
                        number = None
                except ValueError:
                    number = None
        return number

    no_comma_word = word.replace(",", "")
    if no_comma_word in WORD_NUMBER_MAP:
        number = WORD_NUMBER_MAP[no_comma_word]
    else:
        try:
            number = int(no_comma_word)
        except ValueError:
            number = None
    return number


def find_valid_spans(
    passage_tokens: List[Token], answer_texts: List[str]
) -> List[Tuple[int, int]]:
    return DropReader.find_valid_spans(passage_tokens, answer_texts)
