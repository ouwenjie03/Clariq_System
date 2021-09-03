#!/usr/bin/env python
# encoding: utf-8

"""
@author: ouwenjie
@license: NeteaseGame Licence 
@contact: ouwenjie@corp.netease.com


"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import ElectraPreTrainedModel, ElectraModel


class ElectraForSequenceClassificationMultiTask(ElectraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.num_regs = config.num_regs

        self.electra = ElectraModel(config)

        self.classifier = nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.num_labels),
        )
        self.regressor = nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.num_regs),
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        values=None,
    ):
        outputs = self.electra(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        last_hidden_state = outputs[0]
        # classify
        is_next_logits = self.classifier(last_hidden_state[:, 0])
        regs = self.regressor(last_hidden_state[:, 1])

        outputs = (is_next_logits, regs, )

        if labels is not None:
            is_next_loss = F.cross_entropy(is_next_logits.view(-1, self.num_labels), labels.view(-1))
            reg_loss = F.mse_loss(regs.view(-1), values.view(-1))

            loss = is_next_loss + reg_loss

            outputs = (loss, (is_next_loss, reg_loss),) + outputs

        return outputs
