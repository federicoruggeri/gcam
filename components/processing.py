from __future__ import annotations

import torch as th
from torch.nn.utils.rnn import pad_sequence


class DiscreteCollator:
    """
    Torch data collator for SAM Binary model (check modeling/discrete.py)
    """

    def __init__(
            self,
            padding_value
    ):
        self.padding_value = padding_value

    def __call__(
            self,
            batch
    ):
        labels, kb_labels, input_ids, attention_mask = zip(*[list(item.values()) for item in batch])
        input_ids = pad_sequence([th.tensor(item) for item in input_ids], batch_first=True, padding_value=self.padding_value)
        attention_mask = pad_sequence([th.tensor(item) for item in attention_mask], batch_first=True, padding_value=0)

        labels = th.tensor(labels, dtype=th.long)

        return input_ids, attention_mask, labels


class MultilabelCollator:
    """
    Torch data collator for GCAM Fine-grained model (check modeling/discrete.py)
    """

    def __init__(
            self,
            padding_value
    ):
        self.padding_value = padding_value

    def __call__(
            self,
            batch
    ):
        labels, kb_labels, input_ids, attention_mask = zip(*[list(item.values()) for item in batch])
        input_ids = pad_sequence([th.tensor(item) for item in input_ids], batch_first=True, padding_value=self.padding_value)
        attention_mask = pad_sequence([th.tensor(item) for item in attention_mask], batch_first=True, padding_value=0)

        labels = th.tensor(labels, dtype=th.long)
        kb_labels = th.tensor(kb_labels, dtype=th.long)

        return input_ids, attention_mask, labels, kb_labels


class GuidelineCollator:
    """
    Torch data collator for GCAM Text-based models (check modeling/guidelines.py)
    """

    def __init__(
            self,
            kb_input_ids,
            kb_attention_mask,
            padding_value
    ):
        self.padding_value = padding_value
        self.kb_input_ids = kb_input_ids
        self.kb_attention_mask = kb_attention_mask

    def __call__(
            self,
            batch
    ):
        labels, kb_labels, input_ids, attention_mask = zip(*[list(item.values()) for item in batch])
        input_ids = pad_sequence([th.tensor(item) for item in input_ids], batch_first=True, padding_value=self.padding_value)
        attention_mask = pad_sequence([th.tensor(item) for item in attention_mask], batch_first=True, padding_value=0)

        labels = th.tensor(labels, dtype=th.long)
        kb_labels = th.tensor(kb_labels, dtype=th.long)

        return input_ids, attention_mask, labels, kb_labels, self.kb_input_ids, self.kb_attention_mask
