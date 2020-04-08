import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertModel
from torch.autograd import Variable
import logging

logger = logging.getLogger(__name__)


def compute_loss(outputs, labels, loss_method='binary'):
    loss = 0.
    if loss_method == 'binary':
        labels = labels.unsqueeze(1)
        loss = F.binary_cross_entropy(torch.sigmoid(outputs), labels)
    elif loss_method == 'cross_entropy':
        loss = F.cross_entropy(outputs, labels)
    else:
        raise Exception("loss_method {binary or cross_entropy} error. ")
    return loss


class Bert(nn.Module):

    def __init__(self, config, num=0):
        super(Bert, self).__init__()
        model_config = BertConfig.from_pretrained(
            config.config_file[num],
            num_labels=config.num_labels,
            finetuning_task=config.task,
        )
        # 计算loss的方法
        self.loss_method = config.loss_method
        self.multi_drop = config.multi_drop

        self.bert = BertModel.from_pretrained(
            config.model_name_or_path[num],
            config=model_config,
        )
        if config.requires_grad:
            for param in self.bert.parameters():
                param.requires_grad = True
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.hidden_size = config.hidden_size[num]
        if self.loss_method in ['binary', 'focal_loss', 'ghmc']:
            self.classifier = nn.Linear(self.hidden_size, 1)
        else:
            self.classifier = nn.Linear(self.hidden_size, self.num_labels)
        
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        pooled_output = outputs[1]
        
        out = None
        loss = 0
        for i in range(self.multi_drop):
            pooled_output = self.dropout(pooled_output)
            out = self.classifier(pooled_output)
            if labels is not None:
                if i == 0:
                    loss = compute_loss(out, labels, loss_method=self.loss_method) / self.multi_drop
                else:
                    loss += loss / self.multi_drop
        
        if self.loss_method in ['binary']:
            out = torch.sigmoid(out).flatten()

        return out, loss
