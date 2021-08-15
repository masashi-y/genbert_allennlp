import logging
from typing import Any, Dict, List, Optional, Union

import numpy
import torch
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator
from allennlp.nn.util import (get_text_field_mask,
                              get_token_ids_from_text_field_tensors)
from overrides import overrides

from genbert.metrics import Metrics
from genbert.transformers.modeling import BertConfig, BertTransformer
from genbert.util import (ANSWER_GENERATION, ANSWER_SPAN, END_SYMBOL,
                          detokenize,
                          get_token_type_ids_from_text_field_tensors,
                          post_process_decoded_output, to_numpy)

logger = logging.getLogger(__name__)


@Model.register("genbert")
class GenBERT(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        bert_model_name_or_config: Union[str,
                                         Dict[str, Any]] = "bert-base-uncased",
        target_namespace: str = "tokens",
        max_decoding_steps: int = 20,
        masked_lm_loss_coef: float = 1.0,
        prediction_type: str = 'both',
        do_random_shift: bool = False,
        initializer: InitializerApplicator = InitializerApplicator(),
        **kwargs,
    ) -> None:
        super().__init__(vocab, **kwargs)

        if isinstance(bert_model_name_or_config, str):
            self._genbert = BertTransformer.from_pretrained(
                bert_model_name_or_config, prediction_type=prediction_type,
            )
            bert_config = self._genbert.config
        else:
            bert_config = BertConfig(30522, **bert_model_name_or_config)
            self._genbert = BertTransformer(
                bert_config, prediction_type=prediction_type,
            )

        self.hidden_size = bert_config.hidden_size
        self.vocab_size = bert_config.vocab_size
        self.max_position_embeddings = bert_config.max_position_embeddings
        self._masked_lm_loss_coef = masked_lm_loss_coef
        self.max_decoding_steps = max_decoding_steps
        self._target_namespace = target_namespace

        assert prediction_type in ('span', 'generation', 'both')
        self._do_span_prediction = prediction_type in ('span', 'both')
        self._do_generation = prediction_type in ('generation', 'both')

        self._do_random_shift = do_random_shift

        self.metrics = Metrics()

        initializer(self)

    @overrides
    def forward(  # type: ignore
        self,
        question_with_context: TextFieldTensors,
        answer_head_types: Optional[torch.LongTensor] = None,
        answer_as_question_spans: Optional[torch.LongTensor] = None,
        answer_as_passage_spans: Optional[torch.LongTensor] = None,
        answer_as_generation: Optional[TextFieldTensors] = None,
        masked_lm_input_ids: Optional[torch.LongTensor] = None,
        masked_lm_input_mask: Optional[torch.LongTensor] = None,
        masked_lm_label_ids: Optional[torch.LongTensor] = None,
        metadata: List[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """main forward function of the calculator-extended version of GenBERT 

        Args:
            question_with_context (TextFieldTensors): TextFieldTensors object representing
                a standard "[CLS], question_tokens, [SEP], passage_tokens, [SEP]" sequence.
            answer_head_types (Optional[torch.LongTensor], optional): (batch_size,)
                representing labels whether the answer should be predicted using either
                of span selection (1) or generation (0)
            answer_as_question_spans (Optional[torch.LongTensor], optional):
                (batch size, max_number_of_spans, 2) labels for span selection in question texts.
            answer_as_passage_spans (Optional[torch.LongTensor], optional):
                (batch size, max_number_of_spans, 2) labels for span selection in passage texts.
            answer_as_generation (Optional[TextFieldTensors], optional):
                TextFieldTensors object representing generation targets.
            metadata (List[Dict[str, Any]], optional):
                metadata for each batch item. Defaults to None.

        Returns:
            Dict[str, torch.Tensor]: output dictionary
        """

        output_dict = {}

        if self.training and masked_lm_input_ids is not None:
            assert masked_lm_label_ids is not None and masked_lm_input_mask is not None
            mlm_loss = self._genbert(
                input_ids=masked_lm_input_ids,
                input_mask=masked_lm_input_mask,
                target_ids=masked_lm_label_ids,
                task="mlm",
                random_shift=self._do_random_shift,
            )
            output_dict["mlm_loss"] = mlm_loss
            self.metrics.loss_metrics("mlm", mlm_loss)

        # shape (question_with_context_ids): (batch_size, seq_len)
        question_with_context_ids = get_token_ids_from_text_field_tensors(
            question_with_context
        )
        # shape (question_with_context_mask): (batch_size, seq_len)
        question_with_context_mask = get_text_field_mask(question_with_context)

        # shape (question_with_context_token_type_ids): (batch_size, seq_len)
        question_with_context_token_type_ids = get_token_type_ids_from_text_field_tensors(
            question_with_context,
            sep_token_index=self.vocab.get_token_index(
                "[SEP]", self._target_namespace),
            mask=question_with_context_mask,
        )

        # compute the loss
        if answer_head_types is not None:

            assert (
                answer_as_question_spans is not None
                and answer_as_passage_spans is not None
                and answer_as_generation is not None
            )

            answer_as_generation_tensor = get_token_ids_from_text_field_tensors(
                answer_as_generation
            )
            answer_as_generation_mask = get_text_field_mask(
                answer_as_generation)

            _, loss, gen_loss, span_loss, type_loss = self._genbert(
                input_ids=question_with_context_ids,
                token_type_ids=question_with_context_token_type_ids,
                input_mask=question_with_context_mask,
                target_ids=answer_as_generation_tensor,
                target_mask=answer_as_generation_mask,
                answer_as_question_spans=answer_as_question_spans,
                answer_as_passage_spans=answer_as_passage_spans,
                head_type=answer_head_types,
                random_shift=self.training and self._do_random_shift,
            )

            self.metrics.loss_metrics("span", span_loss)
            self.metrics.loss_metrics("type", type_loss)
            self.metrics.loss_metrics("gen", gen_loss)

            output_dict["loss"] = loss
            if self._masked_lm_loss_coef > 0.0 and "mlm_loss" in output_dict:
                output_dict["loss"] += (
                    self._masked_lm_loss_coef * output_dict["mlm_loss"]
                )

            output_dict["gen_loss"] = gen_loss
            output_dict["type_loss"] = type_loss
            output_dict["span_loss"] = span_loss

        # make prediction
        if not self.training:

            dec_preds, type_preds, start_preds, end_preds = self._genbert(
                input_ids=question_with_context_ids,
                token_type_ids=question_with_context_token_type_ids,
                input_mask=question_with_context_mask,
                task="inference",
                max_decoding_steps=self.max_decoding_steps,
            )
            # here segment_ids are only used to get the best span prediction
            # dec_preds: [bsz, max_deocoding_steps], has start_tok
            generated_tokens = self.indices_to_tokens(dec_preds[:, 1:])

            batch_size = dec_preds.size(0)

            for key in (
                "question_id",
                "question_tokens",
                "passage_tokens",
                "token_offsets",
                "original_question",
                "original_passage",
                "original_passage_question",
            ):
                output_dict[key] = [meta[key] for meta in metadata]

            output_dict["answer"] = []
            output_dict["answer_annotations"] = []

            for i in range(batch_size):

                answer_dict: Dict[str, Any] = {}

                passage_question_str = metadata[i]["original_passage_question"]
                offsets = metadata[i]["token_offsets"]
                start_index = to_numpy(start_preds[i])
                end_index = to_numpy(end_preds[i])

                answer_dict["span"] = passage_question_str[
                    offsets[start_index][0]:offsets[end_index][1]
                ]

                answer_dict["generation"] = post_process_decoded_output(
                    generated_tokens[i]
                )
                if self._do_span_prediction and type_preds[i] == ANSWER_SPAN:
                    answer_dict["answer_type"] = "span"
                    answer_dict["answer"] = answer_dict["span"]
                elif self._do_generation and type_preds[i] == ANSWER_GENERATION:
                    answer_dict["answer_type"] = "generation"
                    answer_dict["answer"] = answer_dict["generation"]
                else:
                    raise ValueError("Unsupported answer ability")

                answer_annotations = metadata[i].get("answer_annotations", [])
                output_dict["answer_annotations"].append(answer_annotations)
                output_dict["answer"].append(answer_dict)
                if answer_annotations:
                    self.metrics.drop(
                        answer_dict["answer"], answer_annotations)
                self.metrics.type_accuracy(type_preds, answer_head_types)

        return output_dict

    def indices_to_tokens(self, batch_indices: numpy.ndarray) -> List[List[str]]:

        end_index = self.vocab.get_token_index(
            END_SYMBOL, self._target_namespace)

        batch_indices = to_numpy(batch_indices)

        results = []
        for indices in batch_indices:
            # Beam search gives us the top k results for each source sentence in the batch
            # but we just want the single best.
            if len(indices.shape) > 1:
                indices = indices[0]
            indices = list(indices)
            # Collect indices till the first end_symbol
            if end_index in indices:
                indices = indices[: indices.index(end_index)]
            tokens = [
                self.vocab.get_token_from_index(
                    x, namespace=self._target_namespace)
                for x in indices
            ]
            # remove wordpiece prefixes
            tokens = detokenize(tokens)
            results.append(tokens)

        return results

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return self.metrics.get_metrics(reset)
