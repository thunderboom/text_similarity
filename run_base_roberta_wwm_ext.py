from processors.TryDataProcessor import TryDataProcessor
from transformers import BertTokenizer
from models.bert import Bert, BertSentence

from utils.k_fold import cross_validation
from utils.augment import DataAugment
from utils.utils import *
from utils.train_eval import *

MODEL_CLASSES = {
   'bert':  Bert,
   'bert_sentence': BertSentence,
}


class NewsConfig:

    def __init__(self):
        absdir = os.path.dirname(os.path.abspath(__file__))
        _pretrain_path = '/pretrain_models/chinese_roberta_wwm_ext_pytorch'
        _config_file = 'bert_config.json'
        _model_file = 'pytorch_model.bin'
        _tokenizer_file = 'vocab.txt'
        _data_path = '/real_data'

        # 使用的模型
        self.use_model = 'bert'

        self.models_name = 'base_roberta_wwm_ext'
        self.task = 'base_real_data'
        self.config_file = os.path.join(absdir + _pretrain_path, _config_file)
        self.model_name_or_path = os.path.join(absdir + _pretrain_path, _model_file)
        self.tokenizer_file = os.path.join(absdir + _pretrain_path, _tokenizer_file)
        self.data_dir = absdir + _data_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')              # 设备
        self.device_id = 3
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
        self.early_stop = True
        self.require_improvement = 700 if self.use_model == 'bert' else 1000                                                           # 若超过1000batch效果还没提升，则提前结束训练
        self.num_train_epochs = 8                                                               # epoch数
        self.batch_size = 32                                                                     # mini-batch大小
        self.pad_size = 64 if self.use_model == 'bert' else 32                                                                       # 每句话处理成的长度
        self.learning_rate = 2e-5                                                               # 学习率
        self.weight_decay = 0.01                                                                # 权重衰减因子
        self.warmup_proportion = 0.1                                                            # Proportion of training to perform linear learning rate warmup for.
        self.k_fold = 8
        # logging
        self.is_logging2file = True
        self.logging_dir = absdir + '/logging' + '/' + self.task + '/' + self.models_name
        # save
        self.load_save_model = False
        self.save_path = absdir + '/model_saved' + '/' + self.task
        self.dev_split = 0.1
        self.test_split = 0.1
        self.seed = 369
        # 增强数据
        self.data_augment = False
        self.data_augment_args = 'sameword'
        # Bert的后几层加权输出
        self.weighted_layer_tag = False
        self.weighted_layer_num = 12
        # 拼接max_pooling和avg_pooling
        self.pooling_tag = False
        # 计算loss的方法
        self.loss_method = 'binary'  # [ binary, cross_entropy]
        self.z_test = "multi-sample-drop:1"


def thucNews_task(config):

    if config.device.type == 'cuda':
        torch.cuda.set_device(config.device_id)

    random_seed(config.seed)

    tokenizer = BertTokenizer.from_pretrained(config.tokenizer_file,
                                              do_lower_case=config.do_lower_case)
    processor = TryDataProcessor()
    config.class_list = processor.get_labels()
    config.num_labels = len(config.class_list)

    train_examples = processor.get_train_examples(config.data_dir)
    dev_examples = processor.get_dev_examples(config.data_dir)
    # test_examples = processor.get_test_examples(config.data_dir)
    test_examples = None

    cur_model = MODEL_CLASSES[config.use_model]
    model = cur_model(config)

    logging.info("self config %s", config_to_json_string(config))

    if config.load_save_model:
        model_load(config, model, device='cpu')

    dev_evaluate, predict_label = cross_validation(
        config, train_examples, dev_examples,
        model, tokenizer, pattern='k-fold',
        train_enhancement=DataAugment().dataAugment if config.data_augment else None,
        enhancement_arg=config.data_augment_args,
        test_examples=test_examples)
    logging.info(dev_evaluate)


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
