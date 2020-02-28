import logging
from processors.TryDataProcessor import TryDataProcessor
from transformers import BertTokenizer
from models.bert import Bert, BertSentence
import os
from utils.k_fold import cross_validation
from utils.augment import DataAugment
from utils.train_eval import *
from utils.utils import *
from torch.utils.data import DataLoader
import argparse
import json
import pandas as pd

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
        _pretrain_path = '/pretrain_models/bert-base-chinese'
        _config_file = 'config.json'
        _model_file = 'pytorch_model.bin'
        _tokenizer_file = 'vocab.txt'
        _data_path = '/real_data'

        # 使用的模型
        self.use_model = 'bert'

        self.models_name = 'base_bert'
        self.task = 'base_real_data'
        self.config_file = os.path.join(absdir + _pretrain_path, _config_file)
        self.model_name_or_path = os.path.join(absdir + _pretrain_path, _model_file)
        self.tokenizer_file = os.path.join(absdir + _pretrain_path, _tokenizer_file)
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
        self.hidden_size = 768
        self.require_improvement = 700 if self.use_model == 'bert' else 1000                    # 若超过1000batch效果还没提升，则提前结束训练
        self.num_train_epochs = 8                                                               # epoch数
        self.batch_size = arg.bs                                                                # mini-batch大小
        self.pad_size = 64 if self.use_model == 'bert' else 32                                  # 每句话处理成的长度
        self.learning_rate = 2e-5                                                               # 学习率
        self.weight_decay = 0.01                                                                # 权重衰减因子
        self.warmup_proportion = 0.1                                                            # Proportion of training to perform linear learning rate warmup for.
        self.k_fold = 8
        # logging
        self.is_logging2file = True
        self.logging_dir = absdir + '/logging' + '/' + self.task + '/' + self.models_name
        # save
        self.load_save_model = True   #load the saved model
        self.save_path = absdir + '/model_saved' + '/' + self.task
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
        #test data path
        self.test_data_dir = arg.in_path
        self.save_data_path = arg.out_path


def thucNews_task(config):

    def save_json(labels, path):
        with open(path, 'w', encoding='utf-8') as fw:
            fw.write('{' + '\n')
            for idx, label in enumerate(labels):
                slice = str("[{\"id\":") + str(idx) + str(",\"label\":") + str(label) + str("}],") + '\n'
                fw.write(slice)
            fw.write('}' + '\n')
        return None

    random_seed(config.seed)

    tokenizer = BertTokenizer.from_pretrained(config.tokenizer_file,
                                               do_lower_case=config.do_lower_case)
    processor = TryDataProcessor()
    config.class_list = processor.get_labels()
    config.num_labels = len(config.class_list)

    test_examples = processor.get_test_examples(config.test_data_dir)      #test_data_dir是全路径
    cur_model = MODEL_CLASSES[config.use_model]
    model = cur_model(config)
    if config.load_save_model:  #加载保存模型
        model_load(config, model, device='cpu')
    model.to(config.device)
    convert_to_features, build_data_set, _, evaluate_module = MODEL_CLASSES_[config.use_model]
    test_features = convert_to_features(
        test_examples,
        tokenizer,
        config.class_list,
        config.pad_size,
        data_type='test'
    )
    test_dataset = build_data_set(test_features)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    predict_label = evaluate_module(config, model, test_loader, test=True)
    index = list(pd.read_csv(config.test_data_dir, encoding='utf-8')['id'])
    df_upload = pd.DataFrame({'id':index, 'label':predict_label})
    df_upload.to_csv(config.save_data_path, index=False)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-in_path", help="input test path", type=str)
    parser.add_argument('-out_path', help="output save path", type=str)
    parser.add_argument("-bs", help="batch_size", type=int, default=32)
    args = parser.parse_args()

    config = NewsConfig(args)
    thucNews_task(config)


