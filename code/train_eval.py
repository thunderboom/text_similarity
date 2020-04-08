# coding: UTF-8
import os
import copy
import logging
import numpy as np
import torch
from sklearn import metrics
import time

from transformers import AdamW, get_linear_schedule_with_warmup

logger = logging.getLogger(__name__)


def model_train(config, model, train_iter, dev_iter=None):
    start_time = time.time()

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    diff_part = ["bert.embeddings", "bert.encoder"]
    if config.diff_learning_rate is False:
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": config.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0
             },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate)
    else:
        logger.info("use the diff learning rate")
        # the formal is basic_bert part, not include the pooler
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if
                           not any(nd in n for nd in no_decay) and any(nd in n for nd in diff_part)],
                "weight_decay": config.weight_decay,
                "lr": config.learning_rate
            },
            {
                "params": [p for n, p in model.named_parameters() if
                        any(nd in n for nd in no_decay) and any(nd in n for nd in diff_part)],
                "weight_decay": 0.0,
                "lr": config.learning_rate
             },
            {
                "params": [p for n, p in model.named_parameters() if
                           not any(nd in n for nd in no_decay) and not any(nd in n for nd in diff_part)],
                "weight_decay": config.weight_decay,
                "lr": config.head_learning_rate
            },
            {
                "params": [p for n, p in model.named_parameters() if
                        any(nd in n for nd in no_decay) and not any(nd in n for nd in diff_part)],
                "weight_decay": 0.0,
                "lr": config.head_learning_rate
             },
        ]
        optimizer = AdamW(optimizer_grouped_parameters)

    t_total = len(train_iter) * config.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=t_total * config.warmup_proportion, num_training_steps=t_total
    )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Train Num examples = %d", config.train_num_examples)
    logger.info("  Dev Num examples = %d", config.dev_num_examples)
    logger.info("  Num Epochs = %d", config.num_train_epochs)
    logger.info("  Instantaneous batch size GPU/CPU = %d", config.batch_size)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Train device:%s", config.device)

    global_batch = 0  # 记录进行到多少batch
    dev_best_acc = 0
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升

    predict_all = []
    labels_all = []
    best_model = copy.deepcopy(model)

    for epoch in range(config.num_train_epochs):
        logger.info('Epoch [{}/{}]'.format(epoch + 1, config.num_train_epochs))
        # scheduler.step() # 学习率衰减
        for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(train_iter):
            global_batch += 1
            model.train()

            input_ids = torch.tensor(input_ids).type(torch.LongTensor).to(config.device)
            attention_mask = torch.tensor(attention_mask).type(torch.LongTensor).to(config.device)
            token_type_ids = torch.tensor(token_type_ids).type(torch.LongTensor).to(config.device)
            if config.loss_method in ['binary']:
                labels_tensor = torch.tensor(labels).type(torch.FloatTensor).to(config.device)
            else:
                labels_tensor = torch.tensor(labels).type(torch.LongTensor).to(config.device)

            outputs, loss = model(input_ids, attention_mask, token_type_ids, labels_tensor)

            model.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            outputs = outputs.cpu().detach().numpy()
            predic = list(np.array(outputs >= config.prob_threshold, dtype='int'))
            labels_all.extend(labels)
            predict_all.extend(predic)

            if global_batch % 100 == 0:

                train_acc = metrics.accuracy_score(labels_all, predict_all)
                predict_all = []
                labels_all = []

                # dev 数据
                dev_acc, dev_loss = train_acc, loss
                improve = ''
                if dev_iter is not None:
                    dev_acc, dev_loss, _ = model_evaluate(config, model, dev_iter)

                    if dev_acc > dev_best_acc:
                        dev_best_acc = dev_acc
                        improve = '*'
                        last_improve = global_batch
                        best_model = copy.deepcopy(model)
                    else:
                        improve = ''

                time_dif = time.time() - start_time
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.6f},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.6f},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                logger.info(msg.format(global_batch, loss.cpu().data.item(), train_acc, dev_loss.cpu().data.item(), dev_acc, time_dif, improve))

            if config.early_stop and global_batch - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                logger.info("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    return best_model


def model_evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = []
    predict_prob = []
    labels_all = []
    total_inputs_error = []
    with torch.no_grad():
        for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(data_iter):

            input_ids = torch.tensor(input_ids).type(torch.LongTensor).to(config.device)
            attention_mask = torch.tensor(attention_mask).type(torch.LongTensor).to(config.device)
            token_type_ids = torch.tensor(token_type_ids).type(torch.LongTensor).to(config.device)

            if config.loss_method in ['binary']:
                labels = torch.tensor(labels).type(torch.FloatTensor).to(config.device) if not test else None
            else:
                labels = torch.tensor(labels).type(torch.LongTensor).to(config.device) if not test else None

            outputs, loss = model(input_ids, attention_mask, token_type_ids, labels)

            outputs = outputs.cpu().detach().numpy()
            predic = list(np.array(outputs >= config.prob_threshold, dtype='int'))

            predict_all.extend(predic)
            predict_prob.extend(list(outputs))

            if not test:
                labels = labels.data.cpu().numpy()
                labels_all.extend(list(labels))
                loss_total += loss

                input_ids = input_ids.data.cpu().detach().numpy()
                classify_error = get_classify_error(input_ids, predic, labels, outputs)
                total_inputs_error.extend(classify_error)

    if test:
        if config.out_prob:
            return predict_prob
        return predict_all
    acc = metrics.accuracy_score(labels_all, predict_all)
    return acc, loss_total / len(data_iter), total_inputs_error


def get_classify_error(input_ids, predict, labels, proba, input_ids_pair=None):
    error_list = []
    error_idx = predict != labels
    error_sentences = input_ids[error_idx == True]
    total_sentences = []
    if input_ids_pair is not None:
        error_sentences_pair = input_ids_pair[error_idx == True]
        for sentence1, sentence2 in zip(error_sentences, error_sentences_pair):
            total_sentences.append(np.array(sentence1.tolist()+[117]+sentence2.tolist(), dtype=int))
    else:
        total_sentences = error_sentences

    true_label = labels[error_idx == True]
    pred_proba = proba[error_idx == True]
    for sentences, label, prob in zip(total_sentences, true_label, pred_proba):
        error_dict = {}
        error_dict['sentence_ids'] = sentences
        error_dict['true_label'] = label
        error_dict['proba'] = prob
        error_list.append(error_dict)

    return error_list


def model_save(config, model, num=0, name=None):
    if not os.path.exists(config.save_path[num]):
        os.makedirs(config.save_path[num])
    if name is not None:
        file_name = os.path.join(config.save_path[num], name + '.pkl')
    else:
        file_name = os.path.join(config.save_path[num], config.save_file[num]+'.pkl')
    torch.save(model.state_dict(), file_name)
    logger.info("model saved, path: %s", file_name)


def model_load(config, model, num=0, device='cpu'):
    file_name = os.path.join(config.save_path[num], config.save_file[num]+'.pkl')
    logger.info('loading model: %s', file_name)
    model.load_state_dict(torch.load(file_name,
                                     map_location=device if device == 'cpu' else "{}:{}".format(device, 0)))

