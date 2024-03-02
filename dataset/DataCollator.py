from typing import ClassVar,Any
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorForLanguageModeling
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn.utils.rnn as rnn_utils


@dataclass
class DataCollator(DataCollatorForLanguageModeling):
    CHAIN_TO_LABEL: ClassVar[dict] = {0: 0, 1: 1}
    NUM_CHAINS: ClassVar[int] = 2

    tokenizer: PreTrainedTokenizerBase
    pad_to_multiple_of: Optional[int] = None
    missing_res_token: str = "[UNK]"

    def __call__(self, examples):
        if isinstance(examples[0], (dict, BatchEncoding)):
            input_ids, batch = self.process_examples(examples)
            batched_input_ids = self.tokenizer.pad(
                input_ids,
                return_tensors="pt",
                pad_to_multiple_of=self.pad_to_multiple_of)
        else:
            assert NotImplementedError

        # Update batch with padded input_ids and attention_mask
        batch.update(batched_input_ids)

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask)
        else:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        return batch

    def process_examples(self, examples):
        examples_ = []
        for i, example in enumerate(examples):
            cls_label = self.CHAIN_TO_LABEL[example["cls_label"]]

            if "feature" in example:
                has_feature = True
                examples_.append(
                    dict(input_ids=example['input_ids'],
                         cls_label=cls_label,
                         feature=example['feature']))
            else:
                has_feature = False
                examples_.append(
                    dict(input_ids=example['input_ids'],
                         cls_label=cls_label
                         ))

        examples = examples_
        batch = {
            key: [example[key] for example in examples]
            for key in examples[0].keys()
        }

        batch["cls_label"] = torch.tensor(batch["cls_label"])
        if has_feature:
            batch["feature"] = torch.stack(batch["feature"])

        input_ids = {"input_ids": batch["input_ids"]}

        return input_ids, batch

    def mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        import torch

        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


def lstm_collate_fn(data):
    x = [i[0] for i in data]
    y = [i[1] for i in data]
    feature = [i[2] for i in data]
    length = [i[3] for i in data]

    y = torch.stack(y)
    feature = torch.stack(feature)

    x = rnn_utils.pad_sequence(x, batch_first=True, padding_value=0)
    x = x.float()

    x = rnn_utils.pad_sequence(x, batch_first=True, padding_value=0)
    x_packed = rnn_utils.pack_padded_sequence(x, batch_first=True, lengths=length, enforce_sorted=False)

    return x_packed, feature, y


def cnn_collate_fn(data):
    x = [i[0] for i in data]
    y = [i[1] for i in data]
    feature = [i[2] for i in data]
    length = [i[3] for i in data]

    y = torch.stack(y)
    feature = torch.stack(feature)

    x = rnn_utils.pad_sequence(x, batch_first=True, padding_value=0)
    x = x.float()

    return x, feature, y