import logging
import os
import torch
import torch.utils.data as Data
from torch.utils.data import DataLoader
from processors.TryDataProcessor import TryDataProcessor
from utils import convert_examples_to_features, BuildDataSet, config_to_json_string, random_seed
from transformers import BertTokenizer
from models.bert import Bert
from train_eval import model_train, model_test, model_save, model_load
import time
import json


class NewsConfig:

    def __init__(self):
        absdir = os.path.dirname(os.path.abspath(__file__))

        _pretrain_path = '/pretrain_models/chinese_wwm_ext_pytorch'
        _config_file = 'bert_config.json'
        _model_file = 'pytorch_model.bin'
        _tokenizer_file = 'vocab.txt'
        _data_path = '/try_data'

        self.models_name = 'base_wwm_ext_pytorch'
        self.task = 'base_try_data'
        self.config_file = os.path.join(absdir + _pretrain_path, _config_file)
        self.model_name_or_path = os.path.join(absdir + _pretrain_path, _model_file)
        self.tokenizer_file = os.path.join(absdir + _pretrain_path, _tokenizer_file)
        self.data_dir = absdir + _data_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')              # 设备
        self.device_id = 2
        self.do_lower_case = True
        self.label_on_test_set = True
        self.requires_grad = True
        self.class_list = []
        self.num_labels = 2
        self.train_num_examples = 0
        self.dev_num_examples = 0
        self.test_num_examples = 0
        self.hidden_dropout_prob = 0.1
        self.hidden_size = 768
        self.require_improvement = 1000                                                         # 若超过1000batch效果还没提升，则提前结束训练
        self.num_train_epochs = 8                                                               # epoch数
        self.batch_size = 32                                                                     # mini-batch大小
        self.pad_size = 64                                                                      # 每句话处理成的长度
        self.learning_rate = 2e-5                                                               # 学习率
        self.weight_decay = 0.01                                                                # 权重衰减因子
        self.warmup_proportion = 0.1                                                            # Proportion of training to perform linear learning rate warmup for.
        # logging
        self.is_logging2file = True
        self.logging_dir = absdir + '/logging' + '/' + self.task + '/' + self.models_name
        # save
        self.load_save_model = False
        self.save_path = absdir + '/model_saved' + '/' + self.task
        self.dev_split = 0.1
        self.test_split = 0.1
        self.seed = 369


def thucNews_task(config):

    if config.device.type == 'cuda':
        torch.cuda.set_device(config.device_id)

    random_seed(config.seed)

    tokenizer = BertTokenizer.from_pretrained(config.tokenizer_file,
                                              do_lower_case=config.do_lower_case)
    processor = TryDataProcessor()
    config.class_list = processor.get_labels()
    config.num_labels = len(config.class_list)

    total_examples = processor.get_train_examples(config.data_dir)
    total_features = convert_examples_to_features(
        total_examples,
        tokenizer,
        config.class_list,
        config.pad_size
    )
    config.dev_num_examples = int(len(total_features) * config.dev_split)
    config.test_num_examples = int(len(total_features) * config.test_split)
    config.train_num_examples = len(total_features) - config.dev_num_examples - config.test_num_examples
    train_data, dev_data, test_data = Data.random_split(
        total_features, [config.train_num_examples, config.dev_num_examples, config.test_num_examples])

    train_features = [train_data.dataset[idx] for idx in train_data.indices]
    dev_features = [dev_data.dataset[idx] for idx in dev_data.indices]
    test_features = [test_data.dataset[idx] for idx in test_data.indices]

    train_dataset = BuildDataSet(train_features)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    dev_dataset = BuildDataSet(dev_features)
    dev_loader = DataLoader(dev_dataset, batch_size=config.batch_size, shuffle=True)
    test_dataset = BuildDataSet(test_features)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)
    #
    logging.info("self config %s", config_to_json_string(config))
    bert_model = Bert(config).to(config.device)
    if config.load_save_model:
        model_load(config, bert_model)
    model_train(config, bert_model, train_loader, dev_loader)
    model_test(config, bert_model, test_loader)
    model_save(config, bert_model)


if __name__ == '__main__':

    config = NewsConfig()
    logging_filename = None
    if config.is_logging2file is True:
        file = time.strftime('%Y-%m-%d_%H-%M-%S') + '.log'
        logging_filename = os.path.join(config.logging_dir, file)
        if not os.path.exists(config.logging_dir):
            os.makedirs(config.logging_dir)

    logging.basicConfig(filename=logging_filename, format='%(levelname)s: %(message)s', level=logging.INFO)

    thucNews_task(config)


