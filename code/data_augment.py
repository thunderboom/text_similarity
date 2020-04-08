import os, sys
import copy
import torch
import logging
from transformers import BertTokenizer

from bert import Bert
from DataProcessor import DataProcessor
from cross_validation import cross_validation
from utils import random_seed, config_to_json_string, combined_result, sentence_reverse
from train_eval import model_save, model_load
from augment_utils import sentence_set_pair, augment_data_save, \
    new_category_generate, examples_extract

MODEL_CLASSES = {
   'bert':  Bert
}


class Config:

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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        self.num_train_epochs = 5       # epoch数
        self.batch_size = 16            # mini-batch大小
        self.pad_size = 64              # 每句话处理成的长度
        self.learning_rate = 8e-6       # 学习率
        self.head_learning_rate = 1e-3  # 后面的分类层的学习率
        self.weight_decay = 0.01        # 权重衰减因子
        self.warmup_proportion = 0.1    # Proportion of training to perform linear learning rate warmup for.
        self.k_fold = 5
        self.prob_threshold = 0.40
        self.multi_drop = 5
        #
        self.load_save_model = False
        self.save_path = ['../user_data/model_data']
        self.save_file = ['roberta_large_pair_for_augment']
        self.seed = 369
        # 增强数据
        self.data_augment = False
        self.data_augment_method = []
        # 计算loss的方法
        self.loss_method = 'binary'
        # 差分学习率
        self.diff_learning_rate = False
        # train pattern
        self.pattern = 'full_train'
        # preprocessing
        self.stop_word_valid = True
        # prob
        self.out_prob = True
        # 增强数据生成配置
        self.transmit_augment = True
        self.train_augment_save_file = 'train_augment.csv'
        self.train_dev_augment_save_file = 'train_dev_augment.csv'
        self.category_augment = True
        self.category_augment_save_file = 'new_category.csv'
        self.chip2019_augment = True
        self.retrain_model = True
        self.reverse_tag = True
        self.prob_range = (0.20, 0.80)
        self.chip2019_augment_save_file = 'chip2019.csv'


def chip2019_extract(config):

    config.stop_word_valid = False
    processor = DataProcessor(config)
    config.class_list = processor.get_labels()
    config.num_labels = len(config.class_list)

    original_chip2019_examples = processor.get_original_chip2019_examples()
    if config.reverse_tag:  # 交换
        reverse_test_examples = sentence_reverse(original_chip2019_examples)
        all_test_examples = [original_chip2019_examples, reverse_test_examples]
    else:
        all_test_examples = [original_chip2019_examples]

    cur_model = MODEL_CLASSES[config.use_model]

    tokenizer = BertTokenizer.from_pretrained(config.tokenizer_file,
                                              do_lower_case=config.do_lower_case)
    model = cur_model(config)
    modle_file = os.path.join(config.save_path[0], config.save_file[0] + '.pkl')

    if not os.path.isfile(modle_file) or config.retrain_model:
        print("{} not exit.".format(modle_file))
        # 不存在模型文件
        # 读取训练数据
        config.batch_size = 16
        train_examples = processor.get_train_examples()
        dev_examples = processor.get_dev_examples()
        if config.data_augment:
            augment_examples = processor.read_data_augment(config.data_augment_method)
        else:
            augment_examples = None

        model_example, dev_evaluate, predict_label = cross_validation(
            config=config,
            model=model,
            tokenizer=tokenizer,
            train_examples=train_examples,
            dev_examples=dev_examples,
            pattern=config.pattern,
            train_enhancement=augment_examples,
            test_examples=None)
        model_save(config, model_example)

    model_load(config, model, device='cpu')
    model.to(config.device)
    config.batch_size = 512
    single_model_predict = []
    for test_examples in all_test_examples:
        _, _, predict_label = cross_validation(
            config=config,
            model=model,
            tokenizer=tokenizer,
            train_examples=None,
            dev_examples=None,
            pattern='predict',
            train_enhancement=None,
            test_examples=test_examples)
        single_model_predict.append(predict_label)
    predict_prob = combined_result(single_model_predict, pattern='average')
    save_file = os.path.join(config.other_data_dir, config.chip2019_augment_save_file)
    print('save_file{}'.format(save_file))
    examples_extract(original_chip2019_examples, predict_prob,
                     save_file, sel_prob=config.prob_range, random_state=config.seed)


def augment_task(config):

    processor = DataProcessor(config)
    train_examples = processor.get_train_examples()
    dev_examples = processor.get_dev_examples()
    print("train_examples: {}".format(len(train_examples)))
    print("dev_examples: {}".format(len(dev_examples)))

    if config.transmit_augment:
        print('starting transmit data augment.')
        train_augment = sentence_set_pair(train_examples, random_state=config.seed)
        augment_data_save(train_augment, os.path.join(config.other_data_dir, config.train_augment_save_file))

        new_train_augment = copy.deepcopy(train_examples)
        new_train_augment.extend(dev_examples)
        print(len(new_train_augment))
        train_dev_augment = sentence_set_pair(new_train_augment, random_state=config.seed)
        augment_data_save(train_dev_augment, os.path.join(config.other_data_dir, config.train_dev_augment_save_file))
    if config.category_augment:
        print('starting new category data augment.')
        medicine_examples = processor.get_medicine_examples()
        save_path = os.path.join(config.other_data_dir, config.category_augment_save_file)
        new_category_generate(train_examples, dev_examples, medicine_examples, save_path)
    if config.chip2019_augment:
        print('starting extract chip2019 data augment.')
        chip2019_extract(config)


if __name__ == '__main__':

    config = Config()
    random_seed(config.seed)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    augment_task(config)


