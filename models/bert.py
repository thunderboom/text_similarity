import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertModel
from torch.autograd import Variable


def compute_loss(outputs, labels, loss_method='binary'):
    loss = 0.
    if loss_method == 'binary':
        labels = labels.unsqueeze(1)
        loss = F.binary_cross_entropy(torch.sigmoid(outputs), labels)
    elif loss_method == 'cross_entropy':
        loss = F.cross_entropy(outputs, labels)

    return loss


class Bert(nn.Module):

    def __init__(self, config):
        super(Bert, self).__init__()
        model_config = BertConfig.from_pretrained(
            config.config_file,
            num_labels=config.num_labels,
            finetuning_task=config.task,
        )
        # 计算loss的方法
        self.loss_method = config.loss_method

        self.bert = BertModel.from_pretrained(
            config.model_name_or_path,
            config=model_config,
        )
        if config.requires_grad:
            for param in self.bert.parameters():
                param.requires_grad = True
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if self.loss_method == 'binary':
            self.classifier = nn.Linear(config.hidden_size, 1)
        else:
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # add the weighted layer
        self.hidden_weight = config.weighted_layer_tag         # must modify the config.json
        self.pooling_tag = config.pooling_tag

        if self.hidden_weight:
            self.weight_layer = config.weighted_layer_num
            #self.weight = torch.zeros(self.weight_layer).to(config.device)
            self.weight = torch.nn.Parameter(torch.FloatTensor(self.weight_layer), requires_grad=True)
            self.softmax = nn.Softmax()
            self.pooler = nn.Sequential(nn.Linear(768, 768), nn.Tanh())

        elif self.pooling_tag:
            self.maxPooling = nn.MaxPool1d(64)
            self.avgPooling = nn.AvgPool1d(64)
            self.pooler = nn.Sequential(nn.Linear(768*3, 768), nn.Tanh())

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            labels = None,
            n = 1
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        if self.hidden_weight:     #weighted sum [CLS] hidden layer
            all_hidden_state = outputs[2]
            self.weight.requires_grad = True
            weight = self.softmax(self.weight)
            for i in range(self.weight_layer):
                if i == 0:
                    weighted_sum_state = all_hidden_state[-(i+1)][:, 0] * weight[i]
                else:
                    weighted_sum_state += all_hidden_state[-(i+1)][:, 0] * weight[i]
            pooled_output = self.pooler(weighted_sum_state)

        elif self.pooling_tag:     #Avg Max Formal concat
            last_hidden = outputs[0]
            tag_output = last_hidden[:, 0, :]
            last_hidden = last_hidden.permute((0, 2, 1))  #32, 768, 64
            avg_pooling = self.avgPooling(last_hidden).squeeze()
            max_pooling = self.maxPooling(last_hidden).squeeze()
            pooling_output = torch.cat((tag_output, avg_pooling, max_pooling), dim=1)
            pooled_output = self.pooler(pooling_output)
        else:
            pooled_output = outputs[1]

        out = None
        loss = 0
        for i in range(n):
            pooled_output = self.dropout(pooled_output)
            out = self.classifier(pooled_output)
            if labels is not None:
                if i == 0:
                    loss = compute_loss(out, labels, loss_method=self.loss_method) / n
                else:
                    loss += loss / n

        if self.loss_method == 'binary':
            out = torch.sigmoid(out).flatten()

        return out, loss


class BertSentence(nn.Module):

    def __init__(self, config):
        super(BertSentence, self).__init__()
        # 计算loss的方法
        self.loss_method = config.loss_method

        model_config = BertConfig.from_pretrained(
            config.config_file,
            num_labels=config.num_labels,
            finetuning_task=config.task,
        )
        self.bert = BertModel.from_pretrained(
            config.model_name_or_path,
            config=model_config,
        )
        if config.requires_grad:
            for param in self.bert.parameters():
                param.requires_grad = True
        self.pooler = nn.Sequential(nn.Linear(config.hidden_size*2, config.hidden_size), nn.Tanh())
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if self.loss_method == 'binary':
            self.classifier = nn.Linear(config.hidden_size, 1)
        else:
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(
            self,
            q1_input_ids=None,
            q1_attention_mask=None,
            q1_token_type_ids=None,
            q2_input_ids=None,
            q2_attention_mask=None,
            q2_token_type_ids=None,
            labels=None,
            n = 1
    ):
        outputs1 = self.bert(
            q1_input_ids,
            attention_mask=q1_attention_mask,
            token_type_ids=q1_token_type_ids,
        )
        outputs2 = self.bert(
            q2_input_ids,
            attention_mask=q2_attention_mask,
            token_type_ids=q2_token_type_ids,
        )
        q1_hidden = outputs1[0][:, 0, :]
        q2_hidden = outputs2[0][:, 0, :]
        # easy concat
        query_hidden = torch.cat((q1_hidden, q2_hidden), dim=1)
        # classfier
        pooled_output = self.pooler(query_hidden)
        out = None
        loss = 0
        for i in range(n):
            pooled_output = self.dropout(pooled_output)
            out = self.classifier(pooled_output)
            if labels is not None:
                if i == 0:
                    loss = compute_loss(out, labels, loss_method=self.loss_method) / n
                else:
                    loss += loss / n

        if self.loss_method == 'binary':
            out = torch.sigmoid(out).flatten()

        return out, loss
