import numpy as np
import torch.utils.data as Data
from torch.utils.data import DataLoader

from utils.utils import *
from train_eval import *

logger = logging.getLogger(__name__)


class KFoldDataLoader(object):

    def __init__(self, examples, nums=5):
        self.max_nums = nums
        self.cur_nums = 0
        self.examples = examples
        np.random.shuffle(self.examples)
        self.step = len(self.examples) // nums

    def __iter__(self):
        return self

    def __next__(self):
        if self.cur_nums < self.max_nums:
            if self.cur_nums == 0:
                dev_data = self.examples[:self.step]
                train_data = self.examples[self.step:]
            elif self.cur_nums == self.max_nums - 1:
                train_data = self.examples[:self.cur_nums * self.step]
                dev_data = self.examples[self.cur_nums * self.step:]
            else:
                train_data = self.examples[:self.cur_nums * self.step] + self.examples[(self.cur_nums + 1) * self.step:]
                dev_data = self.examples[self.cur_nums * self.step:(self.cur_nums + 1) * self.step]
            self.cur_nums += 1
            return train_data, dev_data
        else:
            raise StopIteration


def k_fold_cross_validation(
        config,
        train_examples,
        model,
        tokenizer,
        train_enhancement=None,
        enhancement_arg=None,
        test_examples=None,
):
    """
    :param config:
    :param train_examples:
    :param model:
    :param tokenizer:
    :param train_enhancement: 数据增强的接口，仅作用在train上，返回的数据需和train_examples形式一样
    :param test_examples:
    :return: dev_evaluate : list [dev_acc,...]
             k_fold_predict_label : list. if not test_examples, k-fold predict on test.
    """
    dev_evaluate = []
    test_evaluate = []
    k_fold_predict_label = []
    k_fold_loader = KFoldDataLoader(train_examples, nums=config.k_fold)
    idx = 0
    for train_data, dev_data in k_fold_loader:
        idx += 1
        logger.info('k-fold CrossValidation: # %d', idx)

        # 加载模型
        model_example = copy.deepcopy(model).to(config.device)

        # 数据增强
        if train_enhancement:
            ext_data = train_enhancement(train_data, enhancement_arg)
            logger.info('通过数据增强后，新增数据: %d', len(ext_data))
            train_data.extend(ext_data)

        config.train_num_examples = len(train_data)
        config.dev_num_examples = len(dev_data)

        # 特征转化
        train_features = convert_examples_to_features(
            train_data,
            tokenizer,
            config.class_list,
            config.pad_size
        )
        dev_features = convert_examples_to_features(
            dev_data,
            tokenizer,
            config.class_list,
            config.pad_size
        )

        train_dataset = BuildDataSet(train_features)
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        dev_dataset = BuildDataSet(dev_features)
        dev_loader = DataLoader(dev_dataset, batch_size=config.batch_size, shuffle=True)

        model_train(config, model_example, train_loader, dev_loader)
        dev_acc, dev_loss = model_evaluate(config, model_example, dev_loader)
        logger.info('evaluate: acc: {0:>6.2%}, loss: {1:>.6f}'.format(dev_acc, dev_loss))
        dev_evaluate.append(dev_acc)

        if test_examples:

            test_features = convert_examples_to_features(
                test_examples,
                tokenizer,
                config.class_list,
                config.pad_size
            )
            test_dataset = BuildDataSet(test_features)
            test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)
            test_acc, test_loss, _, _, predict_label = model_evaluate(config, model_example, test_loader, test=True)
            logger.info('test: acc: {0:>6.2%}, loss: {1:>.6f}'.format(test_acc, test_loss))
            k_fold_predict_label.append(list(predict_label))
            test_evaluate.append(test_acc)

        if config.device.type == 'gpu':
            torch.cuda.empty_cache()
        else:
            del model_example
    logger.info('K models dev acc mean: {}'.format(np.array(dev_evaluate).mean()))
    if test_examples:
        logger.info('K models test acc mean: {}'.format(np.array(test_evaluate).mean()))

    return dev_evaluate, k_fold_predict_label