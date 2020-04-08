import logging
import os
import csv
import re

logger = logging.getLogger(__name__)


class DataProcessor:

    def __init__(self, config):

        self.stop_word_list = None
        self.data_dir = config.data_dir
        self.other_data_dir = config.other_data_dir

        if config.stop_word_valid:  # 读取停用词
            file_path = os.path.join(self.other_data_dir, "stop_word.txt")
            self.stop_word_list = self._read_dictionary(file_path)

    def get_train_examples(self):
        return self._read_csv(os.path.join(self.data_dir, "train.csv"))

    def get_dev_examples(self):
        return self._read_csv(os.path.join(self.data_dir, "dev.csv"))

    def get_test_examples(self, test_path=None):
        if test_path is None:
            return self._read_csv(os.path.join(self.data_dir, "test.csv"))
        else:
            return self._read_csv(os.path.join(test_path, "test.csv"))

    def get_medicine_examples(self):
        return self._read_dictionary(os.path.join(self.other_data_dir, "medicine.txt"))

    def get_original_chip2019_examples(self):
        return self._read_chip2019_csv(os.path.join(self.other_data_dir, "original_chip2019.csv"))

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def get_dev_labels(self):
        label_list = []
        input_file = os.path.join(self.data_dir, "dev.csv")
        with open(input_file, "r", encoding="utf-8") as f:
            tsv_list = list(csv.reader(f, delimiter=','))
            for line in tsv_list[1:]:
                label_list.append(int(line[4]))
        return label_list

    def read_data_augment(self, augment_list):

        data_augment = []
        for augment in augment_list:
            input_file = os.path.join(self.other_data_dir, augment+".csv")
            logger.info('reading {}'.format(input_file))
            try:
                data = self._read_csv(input_file)
                data_augment.extend(data)
            except Exception as e:
                logger.info(str(e))
                logger.info('read err {}'.format(input_file))
        return data_augment

    def _read_csv(self, input_file):
        """
        :param input_file:
        :return: list [ sentences1,sentences2,label, category]
                if not sentences2, set None. the others same.
        """
        data_list = []
        is_test = True if 'test' in input_file else False
        with open(input_file, "r", encoding="utf-8") as f:
            tsv_list = list(csv.reader(f, delimiter=','))
            for line in tsv_list[1:]:
                text_a = line[2]
                text_b = line[3]
                label = line[4]
                category = line[1]
                if is_test:
                    label = '0'
                text_a, text_b = self.text_preprocessing(text_a, text_b, self.stop_word_list)
                data_list.append([text_a, text_b, label, category])
        return data_list

    def _read_chip2019_csv(self, input_file):
        """
        :param input_file:
        :return: list [ sentences1,sentences2,label, category]
                if not sentences2, set None. the others same.
        """
        data_list = []
        with open(input_file, "r", encoding="utf-8") as f:
            tsv_list = list(csv.reader(f, delimiter=','))
            for line in tsv_list[1:]:
                text_a = line[0]
                text_b = line[1]
                label = line[2]
                category = line[3]
                text_a, text_b = self.text_preprocessing(text_a, text_b, self.stop_word_list)
                data_list.append([text_a, text_b, label, category])
        return data_list

    @classmethod
    def _read_dictionary(cls, input_file):
        dict_list = []
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f.readlines():
                line = line.strip()
                if len(line) > 0:
                    dict_list.append(line)
        return dict_list

    @classmethod
    def text_preprocessing(cls, text_a, text_b, dictionary=None, replace=''):
        flag_a, flag_b = False, False
        if dictionary is not None:
            for word in dictionary:
                if not flag_a and word in text_a:
                    flag_a = True
                    text_a = re.sub(word, replace, text_a)
                if not flag_b and word in text_b:
                    flag_b = True
                    text_b = re.sub(word, replace, text_b)
                if flag_a and flag_b:
                    break
        return text_a, text_b
