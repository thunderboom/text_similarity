import os
import logging
import torch
import time
from transformers import BertTokenizer

from bert import Bert
from DataProcessor import DataProcessor
from cross_validation import cross_validation
from utils import random_seed, config_to_json_string
from train_eval import model_save, model_load

MODEL_CLASSES = {
   'bert':  Bert
}


class RobertaPairConfig:

    def __init__(self):
        # 预训练模型路径
        self.pretrain_path = '../data/External/pretrain_models/roberta_large_pair'
        _config_file = 'config.json'
        _model_file = 'pytorch_model.bin'
        _tokenizer_file = 'vocab.txt'

        # 数据路径
        self.data_dir = '../data/Dataset'
        self.other_data_dir = '../data/External/other_data'

        # 使用的模型
        self.use_model = 'bert'

        self.models_name = 'roberta_large_pair'
        self.task = 'TianChi'
        self.config_file = [os.path.join(self.pretrain_path, _config_file)]
        self.model_name_or_path = [os.path.join(self.pretrain_path, _model_file)]
        self.tokenizer_file = os.path.join(self.pretrain_path, _tokenizer_file)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')              # 设备
        self.do_lower_case = True
        self.requires_grad = True
        self.class_list = []
        self.num_labels = 2
        self.train_num_examples = 0
        self.dev_num_examples = 0
        self.test_num_examples = 0
        self.hidden_dropout_prob = 0.1
        self.hidden_size = [1024]
        self.early_stop = False
        self.require_improvement = 800
        self.num_train_epochs = 5                                                               # epoch数
        self.batch_size = 16                                                                     # mini-batch大小
        self.pad_size = 64                                                                      # 每句话处理成的长度
        self.learning_rate = 8e-6                                                              # 学习率
        self.head_learning_rate = 1e-3                                                          # 后面的分类层的学习率
        self.weight_decay = 0.01                                                                # 权重衰减因子
        self.warmup_proportion = 0.1                                                            # Proportion of training to perform linear learning rate warmup for.
        self.k_fold = 5
        self.prob_threshold = 0.40
        self.multi_drop = 5
        # logging
        self.is_logging2file = True
        self.logging_dir = '../user_data/logging' + '/' + self.models_name
        # save
        self.load_save_model = False
        self.save_path = ['../user_data/model_data']
        self.save_file = [self.models_name]
        self.seed = 369
        # 增强数据
        self.data_augment = True
        # [train_augment, train_dev_augment, chip2019, new_category]
        # only_train下使用train_augment, full_train下使用train_dev_augment
        self.data_augment_method = ['train_dev_augment', 'new_category', 'chip2019']
        # 计算loss的方法
        self.loss_method = 'binary'  # [ binary, cross_entropy]
        # 差分学习率
        self.diff_learning_rate = False
        # train pattern
        self.pattern = 'full_train'  # [only_train, full_train, predict]
        # preprocessing
        self.stop_word_valid = False
        # prob
        self.out_prob = True


def roberta_pair_task(config):

    tokenizer = BertTokenizer.from_pretrained(config.tokenizer_file,
                                              do_lower_case=config.do_lower_case)
    processor = DataProcessor(config)
    config.class_list = processor.get_labels()
    config.num_labels = len(config.class_list)

    train_examples = processor.get_train_examples()
    dev_examples = processor.get_dev_examples()
    augment_examples = processor.read_data_augment(config.data_augment_method)

    cur_model = MODEL_CLASSES[config.use_model]
    model = cur_model(config)

    logging.info("self config %s", config_to_json_string(config))

    model_example, dev_evaluate, predict_label = cross_validation(
        config=config,
        model=model,
        tokenizer=tokenizer,
        train_examples=train_examples,
        dev_examples=dev_examples,
        pattern=config.pattern,
        train_enhancement=augment_examples if config.data_augment else None,
        test_examples=None)
    logging.info("dev_evaluate: {}".format(dev_evaluate))

    if config.pattern == 'full_train':
        model_save(config, model_example)

    return dev_evaluate


if __name__ == '__main__':

    config = RobertaPairConfig()
    random_seed(config.seed)
    logging_filename = None
    if config.is_logging2file is True:
        file = time.strftime('%Y-%m-%d_%H-%M-%S') + '.log'
        logging_filename = os.path.join(config.logging_dir, file)
        if not os.path.exists(config.logging_dir):
            os.makedirs(config.logging_dir)

    logging.basicConfig(filename=logging_filename, format='%(levelname)s: %(message)s', level=logging.INFO)

    roberta_pair_task(config)


