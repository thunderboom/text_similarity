import logging
from processors.TryDataProcessor import TryDataProcessor
from transformers import BertTokenizer
from models.bert import Bert, BertSentence
from utils.train_eval import *
from utils.utils import *
from torch.utils.data import DataLoader
import argparse
import pandas as pd
import numpy as np
import time

MODEL_CLASSES = {
   'bert':  Bert,
   'bert_sentence': BertSentence,
}


MODEL_CLASSES_ = {
   'bert':  (convert_examples_to_features, BuildDataSet,
             model_train, model_evaluate),
   'bert_sentence': (convert_examples_to_features_sentence,
                     BuildDataSetSentence, model_train_sentence, model_evaluate_sentence),
}



class NewsConfig:

    def __init__(self, arg):
        absdir = os.path.dirname(os.path.abspath(__file__))
        _pretrain_path = ['/pretrain_models/ERNIE', '/pretrain_models/roberta_large_pair',
                          '/pretrain_models/chinese_roberta_wwm_large_ext_pytorch']
        _config_file = ['bert_config.json', 'config.json', 'bert_config.json']
        _model_file = ['pytorch_model.bin', 'pytorch_model.bin', 'pytorch_model.bin']
        _tokenizer_file = ['vocab.txt', 'vocab.txt', 'vocab.txt']
        _data_path = '/real_data'

        # 使用的模型
        self.use_model = 'bert'

        self.models_name = ['ernie', 'roberta_large_pair', 'roberta_wwm_large']
        self.task = 'base_real_data'
        self.config_file = [os.path.join(absdir + pretrain_path_try, config_file_try)
                            for pretrain_path_try, config_file_try in zip(_pretrain_path, _config_file)]

        self.model_name_or_path = [os.path.join(absdir + pretrain_path_try, model_file_try)
                                   for pretrain_path_try, model_file_try in zip(_pretrain_path, _model_file)]

        self.tokenizer_file = [os.path.join(absdir + pretrain_path_try, tokenizer_file_try)
                               for pretrain_path_try, tokenizer_file_try in zip(_pretrain_path, _tokenizer_file)]

        self.data_dir = absdir + _data_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')              # 设备
        self.device_id = 0
        self.do_lower_case = True
        self.label_on_test_set = True
        self.requires_grad = True
        self.class_list = []
        self.num_labels = 2
        self.train_num_examples = 0
        self.dev_num_examples = 0
        self.test_num_examples = 0
        self.hidden_dropout_prob = 0.1
        self.hidden_size = [768, 1024, 1024]
        self.require_improvement = 700 if self.use_model == 'bert' else 1000                    # 若超过1000batch效果还没提升，则提前结束训练
        self.num_train_epochs = 8                                                               # epoch数
        self.batch_size = arg.bs                                                                # mini-batch大小
        self.pad_size = 64                                                                      # 每句话处理成的长度
        self.learning_rate = 2e-5                                                               # 学习率
        self.weight_decay = 0.01                                                                # 权重衰减因子
        self.warmup_proportion = 0.1                                                            # Proportion of training to perform linear learning rate warmup for.
        self.k_fold = 5
        # logging
        self.is_logging2file = False
        self.logging_dir = absdir + '/logging' + '/' + self.task + '/' 
        # save
        self.load_save_model = True   #load the saved model
        self.save_path = [absdir + '/model_saved' + '/' + self.task +'/' + model_name_try + '/' for model_name_try in self.models_name]
        self.save_file = ['ernie', 'roberta_large_pair', 'roberta_wwm_large']
        self.dev_split = 0.1
        self.test_split = 0.1
        self.seed = 369
        #模型定义
        # Bert的后几层加权输出
        self.weighted_layer_tag = False
        self.weighted_layer_num = 3
        # 拼接max_pooling和avg_pooling
        self.pooling_tag = False
        # 计算loss的方法
        self.loss_method = 'binary'  # [ binary, cross_entropy]
        # 说明
        self.z_test = 'multi-sample-drop:1'
        # 增强数据
        self.data_augment = False
        self.data_augment_args = 'transmit'
        # Bert的后几层加权输出
        self.weighted_layer_tag = False
        self.weighted_layer_num = 12
        # 拼接max_pooling和avg_pooling
        self.pooling_tag = False
        # 差分学习率
        self.diff_learning_rate = False
        # multi-task
        self.multi_loss_tag = False
        self.multi_loss_weight = 0.5
        self.multi_class_list = []   # 第二任务标签
        self.multi_num_labels = 0    # 第二任务标签 数量
        
        #test data path
        self.test_data_dir = arg.in_path
        self.save_data_path = arg.out_path
        #model vote
        self.model_vote_tag = False
        # preprocessing
        self.stop_word_valid = False
        self.medicine_valid = False
        self.symptom_valid = False
        self.medicine_replace_word = ''
        self.symptom_replace_word = ''
        #multi model
        self.reverse_tag = True
        #multi_model
        self.multi_model = True
        self.model_num = 3
        self.out_prob = True


def sentence_reverse(test_examples):
    reverse_test_examples = []
    for example in test_examples:
        try_example = [example[1], example[0], example[2], example[3]]
        reverse_test_examples.append(try_example)
    return reverse_test_examples


def combined_result(all_result, weight=None, type='average'):
    def average_result(all_result):  #shape:[num_model, axis]
        all_result = np.asarray(all_result, dtype=np.float)
        return np.mean(all_result, axis=0)

    def weighted_result(all_result, weight):
        all_result = np.asarray(all_result, dtype=np.float)
        return np.average(all_result, axis=0, weights=weight)

    if type == 'weighted':
        return weighted_result(all_result, weight)
    elif type == 'average':
        return average_result(all_result)
    else:
        raise ValueError("the combined type is incorrect")


def thucNews_task(config):

    print('cude: {}'.format(torch.cuda.is_available()))
    print('cur device {}'.format(config.device.type))
    start_time = time.time()

    random_seed(config.seed)

    processor = TryDataProcessor(config)
    config.class_list = processor.get_labels()
    config.num_labels = len(config.class_list)

    test_examples = processor.get_test_examples(config.test_data_dir)      #test_data_dir是全路径
    if config.reverse_tag:  #交换
        reverse_test_examples = sentence_reverse(test_examples)
        all_examples = [test_examples, reverse_test_examples]
    else:
        all_examples = [test_examples]
    cur_model = MODEL_CLASSES[config.use_model]
    print('loading data time: {:.6f}s'.format(time.time()-start_time))
    if config.multi_model:         #多模型
        all_predict = []
        for i in range(config.model_num):
            model_time_s = time.time()
            print('the model of {} starting...'.format(config.models_name[i]))
            tokenizer = BertTokenizer.from_pretrained(config.tokenizer_file[i],
                                                      do_lower_case=config.do_lower_case)
            model = cur_model(config, num=i)       #使用索引拿到相关参数
            model_load(config, model, num=i, device='cpu')
            model.to(config.device)
            convert_to_features, build_data_set, _, evaluate_module = MODEL_CLASSES_[config.use_model]
            single_model_predict = []
            print("\tloading pre-train model, cost time {:.6f}s".format(time.time()-model_time_s))
            for e_idx, examples in enumerate(all_examples):
                example_time = time.time()
                test_features = convert_to_features(
                examples=examples,
                tokenizer=tokenizer,
                label_list=config.class_list,
                second_label_list=config.multi_class_list if config.multi_loss_tag else None,
                max_length=config.pad_size,
                data_type='test')
                test_dataset = build_data_set(test_features)
                test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
                predict_label = evaluate_module(config, model, test_loader, test=True)
                single_model_predict.append(predict_label)
                print("\ttest dataset:{}, cost time {:.6f}s, total time {:.6f}s".format(e_idx+1, time.time()-example_time, time.time()-start_time))
            print("# time {:.6f}s, total time {:.6f}s".format(time.time() - model_time_s, time.time()-start_time))
            single_model_predict = combined_result(single_model_predict, type='average')  #计算reverse平均
            # print('single_model_predict:', single_model_predict)
            all_predict.append(single_model_predict)

        # print("use the {} model".format(len(all_predict)))
        if config.model_vote_tag:  #使用投票
            all_predict = np.asarray(all_predict > 0.4, dtype=np.int)
            predict_label = k_fold_volt_predict(all_predict)
        else:  #使用加权平均
            predict_label = combined_result(all_predict, type='average')
            predict_label = np.asarray(predict_label > 0.4, dtype=np.int)

    else:  #单模
        tokenizer = BertTokenizer.from_pretrained(config.tokenizer_file[0],
                                                  do_lower_case=config.do_lower_case)
        model = cur_model(config)
        model_load(config, model, device='cpu')
        model.to(config.device)
        single_model_predict = []
        for examples in all_examples:
            convert_to_features, build_data_set, _, evaluate_module = MODEL_CLASSES_[config.use_model]
            test_features = convert_to_features(
                examples=examples,
                tokenizer=tokenizer,
                label_list=config.class_list,
                second_label_list=config.multi_class_list if config.multi_loss_tag else None,
                max_length=config.pad_size,
                data_type='test'
            )
            test_dataset = build_data_set(test_features)
            test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
            predict_label = evaluate_module(config, model, test_loader, test=True)
            single_model_predict.append(predict_label)
        predict_label = combined_result(single_model_predict, type='average')
        predict_label = np.asarray(predict_label > 0.5, dtype=np.int)

    index = list(pd.read_csv(config.test_data_dir, encoding='utf-8')['id'])
    df_upload = pd.DataFrame({'id':index, 'label':predict_label})
    df_upload.to_csv(config.save_data_path, index=False)
    print('\ntotal time {:.6f}s'.format(time.time()-start_time))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-in_path", help="input test path", type=str)
    parser.add_argument('-out_path', help="output save path", type=str)
    parser.add_argument("-bs", help="batch_size", type=int, default=32)
    args = parser.parse_args()

    config = NewsConfig(args)
    thucNews_task(config)


