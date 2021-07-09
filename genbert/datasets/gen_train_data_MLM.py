import collections
from argparse import ArgumentParser
from pathlib import Path
from random import randint, random, shuffle

import jsonlines
import numpy as np
import json
from tqdm import tqdm, trange

from genbert.datasets.utils import convert_word_to_number
from genbert.digits_aware_tokenizer import DigitsAwareTransformerTokenizer

MaskedLmInstance = collections.namedtuple(
    "MaskedLmInstance", ["index", "label"])
Token = collections.namedtuple(
    "Token", ["text", "is_number"], defaults=[False])


def create_masked_lm_predictions(
    tokens, masked_lm_prob, max_predictions_per_seq, mask_type,
):
    """Creates the predictions for the masked LM objective. This is mostly copied from the Google BERT repo, but
    with several refactors to clean it up and remove a lot of unnecessary variables."""
    cand_indices = []
    for (i, token) in enumerate(tokens):
        if token.text == "[CLS]" or token.text == "[SEP]":
            continue
        # Whole Word Masking means that if we mask all of the wordpieces
        # corresponding to an original word. When a word has been split into
        # WordPieces, the first token does not have any marker and any subsequence
        # tokens are prefixed with ##. So whenever we see the ## token, we
        # append it to the previous set of word indexes.
        if (
            mask_type == "whole_word_mask"
            and len(cand_indices) >= 1
            and token.text.startswith("##")
        ):
            cand_indices[-1].append(i)
        elif mask_type == "number_mask":
            if (
                token.is_number
                and len(cand_indices) >= 1
                and token.text.startswith("##")
            ):
                cand_indices[-1].append(i)
            elif token.is_number:
                cand_indices.append([i])
        else:
            cand_indices.append([i])

    num_to_mask = min(
        max_predictions_per_seq, max(
            1, int(round(len(tokens) * masked_lm_prob)))
    )
    shuffle(cand_indices)
    masked_lms = []
    covered_indexes = set()
    for index_set in cand_indices:
        if len(masked_lms) >= num_to_mask:
            break
        #         # If adding a whole-word mask would exceed the maximum number of
        #         # predictions, then just skip this candidate.
        #         if len(masked_lms) + len(index_set) > num_to_mask:
        #             continue
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue

        op = np.random.choice(["mask", "orig"], p=[0.9, 0.1])
        for index in index_set:
            covered_indexes.add(index)

            # 80% of the time, replace with [MASK]
            if op == "mask":
                masked_token = Token("[MASK]")
            elif op == "orig":
                # 10% of the time, keep original
                masked_token = tokens[index]
            masked_lms.append(MaskedLmInstance(
                index=index, label=tokens[index].text))
            tokens[index] = masked_token

    #     assert len(masked_lms) <= num_to_mask
    masked_lms = sorted(masked_lms, key=lambda x: x.index)
    mask_indices = [p.index for p in masked_lms]
    masked_token_labels = [p.label for p in masked_lms]

    return tokens, mask_indices, masked_token_labels


def create_instances_from_document(
    document,
    max_seq_length,
    short_seq_prob,
    masked_lm_prob,
    max_predictions_per_seq,
    mask_type,
):
    # document is a list of toknzd sents
    # Account for [CLS], [SEP]
    max_num_tokens = max_seq_length - 2

    # We *usually* want to fill up the entire sequence since we are padding
    # to `max_seq_length` anyways, so short sequences are generally wasted
    # computation. However, we *sometimes*
    # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
    # sequences to minimize the mismatch between pre-training and fine-tuning.
    # The `target_seq_length` is just a rough target however, whereas
    # `max_seq_length` is a hard limit.
    target_seq_length = max_num_tokens
    if random() < short_seq_prob:
        target_seq_length = randint(2, max_num_tokens)

    # We DON'T just concatenate all of the tokens from a document into a long
    # sequence and choose an arbitrary split point because this would make the
    # next sentence prediction task too easy. Instead, we split the input into
    # segments "A" and "B" based on the actual "sentences" provided by the user
    # input.
    instances = []
    current_chunk = []
    current_length = 0
    i = 0
    while i < len(document):
        segment = document[i]
        current_chunk.append(
            segment
        )  # append to sents to current_chunk until target_seq_length
        current_length += len(segment)

        if i == len(document) - 1 or current_length >= target_seq_length:
            tokens = sum(current_chunk, [])
            truncate_seq(tokens, max_num_tokens)
            any_number = any(token.is_number for token in tokens)
            if tokens and any_number:
                tokens = [Token("[CLS]")] + tokens + [Token("[SEP]")]
                # The segment IDs are 0 for the [CLS] token, the A tokens and the first [SEP]
                # They are 1 for the B tokens and the final [SEP]
                segment_ids = [0 for _ in range(len(tokens))]

                (
                    tokens,
                    masked_lm_positions,
                    masked_lm_labels,
                ) = create_masked_lm_predictions(
                    tokens, masked_lm_prob, max_predictions_per_seq, mask_type,
                )

                instance = {
                    "tokens": [token.text for token in tokens],
                    "segment_ids": segment_ids,
                    "masked_lm_positions": masked_lm_positions,
                    "masked_lm_labels": masked_lm_labels,
                }
                instances.append(instance)
            # reset and start new chunk
            current_chunk = []
            current_length = 0
        i += 1

    return instances


def truncate_seq(tokens, max_num_tokens):
    """Truncates a list to a maximum sequence length"""
    while len(tokens) > max_num_tokens:
        assert len(tokens) >= 1
        # We want to sometimes truncate from the front and sometimes from the
        # back to add more randomness and avoid biases.
        if random() < 0.5:
            del tokens[0]
        else:
            tokens.pop()


def main():
    parser = ArgumentParser(
        description="""Creates whole-word-masked instances for MLM task. MLM_paras.jsonl is a list of dicts each with a key 'sents' and val a list of sentences of some document.\n
    Usage: python gen_train_data_MLM.py --train_corpus MLM_paras.jsonl --bert_model bert-base-uncased --output_dir data/MLM_train/ --do_lower_case --max_predictions_per_seq 65 --do_whole_word_mask --digitize """
    )

    parser.add_argument("--train_corpus", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--bert_model",
        type=str,
        required=True,
        choices=[
            "bert-base-uncased",
            "bert-large-uncased",
            "bert-base-cased",
            "bert-base-multilingual",
            "bert-base-chinese",
        ],
    )
    parser.add_argument(
        "--mask_type",
        default="random",
        choices=["random", "whole_word_mask", "number_mask"],
        help="Whether to use whole word / number masking rather than per-WordPiece masking.",
    )
    parser.add_argument(
        "--epochs_to_generate",
        type=int,
        default=1,
        help="Number of epochs of data to pregenerate",
    )
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument(
        "--short_seq_prob",
        type=float,
        default=0.1,
        help="Probability of making a short sentence as a training example",
    )
    parser.add_argument(
        "--masked_lm_prob",
        type=float,
        default=0.15,
        help="Probability of masking each token for the LM task",
    )
    parser.add_argument(
        "--max_predictions_per_seq",
        type=int,
        default=65,
        help="Maximum number of tokens to mask in each sequence",
    )

    include_more_numbers = True

    args = parser.parse_args()

    tokenizer = DigitsAwareTransformerTokenizer(
        args.bert_model, include_more_numbers=include_more_numbers
    )
    with jsonlines.open(args.train_corpus, "r") as reader:
        data = [d for d in tqdm(reader.iter())]
    docs = []
    for d in tqdm(data):
        doc = []
        for sent in d["sents"]:
            tokens = tokenizer.str_to_tokens(sent)

            token_is_number = [
                convert_word_to_number(
                    token.text, include_more_numbers) is not None
                for token in tokens
            ]
            wordpieces = [
                Token(wordpiece.text, is_number)
                for token, is_number in zip(tokens, token_is_number)
                for wordpiece in tokenizer.str_to_wordpieces(token)
            ]
            doc.append(wordpieces)
        if doc:
            docs.append(doc)

    # docs is a list of docs - each doc is a list of sents - each sent is list of tokens
    args.output_dir.mkdir(exist_ok=True)
    for epoch in trange(args.epochs_to_generate, desc="Epoch"):
        epoch_filename = args.output_dir / f"epoch_{epoch}.jsonl"
        num_instances = 0
        with epoch_filename.open("w") as epoch_file:
            for doc_idx in trange(len(docs), desc="Document"):
                doc_instances = create_instances_from_document(
                    docs[doc_idx],
                    max_seq_length=args.max_seq_len,
                    short_seq_prob=args.short_seq_prob,
                    masked_lm_prob=args.masked_lm_prob,
                    max_predictions_per_seq=args.max_predictions_per_seq,
                    mask_type=args.mask_type,
                )
                for instance in doc_instances:
                    epoch_file.write(json.dumps(instance) + "\n")
                    num_instances += 1
        metrics_file = args.output_dir / f"epoch_{epoch}_metrics.jsonl"
        with metrics_file.open("w") as metrics_file:
            metrics = {
                "num_training_examples": num_instances,
                "max_seq_len": args.max_seq_len,
            }
            metrics_file.write(json.dumps(metrics))


if __name__ == "__main__":
    main()

"""python -m genbert.datasets.gen_train_data_MLM --train_corpus MLM_paras.jsonl --bert_model bert-base-uncased --output_dir out --max_predictions_per_seq 65 --mask_type number_mask"""
