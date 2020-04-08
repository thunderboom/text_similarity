import os
import torch
import time
import numpy as np
import pandas as pd
import argparse
from transformers import BertTokenizer

from bert import Bert
from DataProcessor import DataProcessor
from cross_validation import cross_validation
from utils import random_seed, config_to_json_string, combined_result, sentence_reverse
from train_eval import model_save, model_load


MODEL_CLASSES = {
   'bert':  Bert,
}


class TestConfig:

    def __init__(self, arg):
        # 预训练模型路径
        self.pretrain_path = ['../data/External/pretrain_models/ERNIE',
                              '../data/External/pretrain_models/roberta_large_pair',
                              '../data/External/pretrain_models/chinese_roberta_wwm_large_ext_pytorch']
        _config_file = ['bert_config.json', 'config.json', 'bert_config.json']
        _model_file = ['pytorch_model.bin', 'pytorch_model.bin', 'pytorch_model.bin']
        _tokenizer_file = ['vocab.txt', 'vocab.txt', 'vocab.txt']

        # 数据路径
        self.data_dir = '../data/Dataset'
        self.other_data_dir = '../data/External/other_data'

        # 使用的模型
        self.use_model = 'bert'

        self.models_name = ['ernie', 'roberta_large_pair', 'roberta_wwm_large']
        self.task = 'TianChi'
        self.config_file = [os.path.join(pretrain_path_try, config_file_try)
                            for pretrain_path_try, config_file_try in zip(self.pretrain_path, _config_file)]

        self.model_name_or_path = [os.path.join(pretrain_path_try, model_file_try)
                                   for pretrain_path_try, model_file_try in zip(self.pretrain_path, _model_file)]

        self.tokenizer_file = [os.path.join(pretrain_path_try, tokenizer_file_try)
                               for pretrain_path_try, tokenizer_file_try in zip(self.pretrain_path, _tokenizer_file)]

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.do_lower_case = True
        self.requires_grad = True
        self.class_list = []
        self.num_labels = 2
        self.train_num_examples = 0
        self.dev_num_examples = 0
        self.test_num_examples = 0
        self.hidden_dropout_prob = 0.1
        self.hidden_size = [768, 1024, 1024]
        self.early_stop = False
        self.require_improvement = 500
        self.num_train_epochs = 5
        self.batch_size = arg.bs
        self.pad_size = 64
        self.learning_rate = 2e-5
        self.weight_decay = 0.01
        self.warmup_proportion = 0.1
        self.k_fold = 5
        self.prob_threshold = 0.40
        self.multi_drop = 5
        # multi_model
        self.model_num = 3
        # save
        self.load_save_model = True
        self.save_path = ['../user_data/model_data'] * self.model_num
        self.save_file = ['ernie', 'roberta_large_pair', 'roberta_wwm_large']
        self.seed = 369
        self.loss_method = 'binary'
        # 增强数据
        self.data_augment = False
        self.data_augment_method = []
        # 差分学习率
        self.diff_learning_rate = False
        # test data path
        self.test_data_dir = arg.in_path   # 测试路径(不包括文件名 默认为test.csv)
        self.save_data_path = arg.out_path
        # preprocessing
        self.stop_word_valid = False
        # multi model
        self.reverse_tag = True
        self.out_prob = True


def test_task(config):

    print('cude: {}'.format(torch.cuda.is_available()))
    print('cur device {}'.format(config.device.type))
    start_time = time.time()

    processor = DataProcessor(config)
    config.class_list = processor.get_labels()
    config.num_labels = len(config.class_list)

    test_examples = processor.get_test_examples(config.test_data_dir)
    # 交换
    if config.reverse_tag:
        reverse_test_examples = sentence_reverse(test_examples)
        all_examples = [test_examples, reverse_test_examples]
    else:
        all_examples = [test_examples]

    cur_model = MODEL_CLASSES[config.use_model]
    print('loading data time: {:.6f}s'.format(time.time()-start_time))

    all_predict = []
    for i in range(config.model_num):
        model_time_s = time.time()
        print('the model of {} starting...'.format(config.models_name[i]))
        tokenizer = BertTokenizer.from_pretrained(config.tokenizer_file[i],
                                                  do_lower_case=config.do_lower_case)
        model = cur_model(config, num=i)
        model_load(config, model, num=i, device='cpu')
        model.to(config.device)
        print("\tloading pre-train model, cost time {:.6f}s".format(time.time() - model_time_s))

        single_model_predict = []
        for e_idx, t_examples in enumerate(all_examples):
            example_time = time.time()
            _, _, predict_label = cross_validation(
                config=config,
                model=model,
                tokenizer=tokenizer,
                train_examples=None,
                dev_examples=None,
                pattern='predict',
                train_enhancement=None,
                test_examples=t_examples)
            single_model_predict.append(predict_label)
            print("\ttest dataset:{}, cost time {:.6f}s, total time {:.6f}s".format(e_idx+1, time.time()-example_time, time.time()-start_time))

        print("# time {:.6f}s, total time {:.6f}s".format(time.time() - model_time_s, time.time()-start_time))
        predict_prob = combined_result(single_model_predict, pattern='average')
        all_predict.append(predict_prob)

    final_predict_label = combined_result(all_predict, pattern='average')
    final_predict_label = np.asarray(final_predict_label >= config.prob_threshold, dtype=np.int)

    index = list(pd.read_csv(os.path.join(config.test_data_dir, 'test.csv'), encoding='utf-8')['id'])
    df_upload = pd.DataFrame({'id': index, 'label': final_predict_label})
    df_upload.to_csv(config.save_data_path, index=False)
    print('\ntotal time {:.6f}s'.format(time.time()-start_time))
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-in_path", help="input test path", type=str)
    parser.add_argument('-out_path', help="output save path", type=str)
    parser.add_argument("-bs", help="batch_size", type=int, default=512)
    args = parser.parse_args()
    config = TestConfig(args)
    random_seed(config.seed)
    test_task(config)


