from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from torch import nn
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertLMPredictionHead, ModelOutput


def exists(x):
    return x is not None


class Seq_Heads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)
        self.cls = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        cls_score = self.cls(pooled_output)
        return prediction_scores, cls_score


class SeqAndProperty_Heads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)
        self.cls = nn.Sequential(
            nn.Linear(config.hidden_size + 9, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, sequence_output, pooled_output, feature):
        prediction_scores = self.predictions(sequence_output)
        pooled_output = self.dropout(pooled_output)
        cls_input = torch.concat([pooled_output, feature.float()], dim=1)
        cls_score = self.cls(cls_input)
        return prediction_scores, cls_score


@dataclass
class ModelOutput(ModelOutput):
    masked_lm_loss : Optional[torch.FloatTensor] = None
    cls_loss : Optional[torch.FloatTensor] = None
    prediction_logits: torch.FloatTensor = None
    cls_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class MyBERT(BertPreTrainedModel):
    def __init__(self, config, use_property=False):
        super().__init__(config)

        self.bert = BertModel(config)
        if use_property:
            self.cls = SeqAndProperty_Heads(config)
        else:
            self.cls = Seq_Heads(config)
        self.init_weights()
        self.num_label = 2

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def forward(
        self,
        input_ids,
        feature=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        cls_label=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if exists(
            return_dict) else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output, pooled_output = outputs[:2]
        if exists(feature):
            prediction_scores, cls_score = self.cls(sequence_output, pooled_output, feature)
        else:
            prediction_scores, cls_score = self.cls(sequence_output, pooled_output)

        b = input_ids.shape[0]
        total_loss, masked_lm_loss, cls_loss = None, None, None
        if exists(labels):
            mlm_loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = mlm_loss_fct(
                prediction_scores.view(-1, self.config.vocab_size),
                labels.view(-1))

        if exists(cls_label):
            species_freqs = torch.bincount(cls_label,
                                           minlength=self.num_label)
            species_weights = b / (species_freqs * self.num_label)
            species_loss_fct = nn.CrossEntropyLoss(weight=species_weights)
            cls_loss = species_loss_fct(
                cls_score.view(-1, self.num_label),
                cls_label.view(-1))

        masked_lm_loss = masked_lm_loss if exists(masked_lm_loss) else 0
        cls_loss = cls_loss if exists(cls_loss) else 0

        if not return_dict:
            output = (prediction_scores, cls_loss,) + outputs[2:]
            return ((total_loss, ) + output) if exists(total_loss) else output

        return ModelOutput(
            masked_lm_loss=masked_lm_loss,
            cls_loss=cls_loss,
            prediction_logits=prediction_scores,
            cls_logits=cls_score,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class LstmNet(nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super(LstmNet, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size,
                            batch_first=True,
                            num_layers=2,
                            bidirectional=True,
                            dropout=dropout,
                            )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size * 2 + 9, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, feature=None):
        lstm_out, (hidden, cell) = self.lstm(x)
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        hidden = self.dropout(hidden)
        hidden = torch.concat([hidden, feature.float()], dim=1)
        out = self.mlp(hidden)
        return out


class CnnNet(nn.Module):
    def __init__(self, input_channel, output_channel, dropout):
        super(CnnNet, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(input_channel, output_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(output_channel),
            nn.ReLU()
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(output_channel, output_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(output_channel),
            nn.ReLU()
        )
        self.mlp = nn.Sequential(
            nn.Linear(output_channel + 9, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, feature=None):
        x = x.transpose(1, 2)
        x = self.conv_block1(x)
        x = x + self.conv_block2(x)
        x = self.pool(x)
        x = x.transpose(1, 2)
        x = x.reshape(x.size(0), -1)
        x = self.dropout(x)
        x = torch.concat([x, feature.float()], dim=1)
        out = self.mlp(x)
        return out