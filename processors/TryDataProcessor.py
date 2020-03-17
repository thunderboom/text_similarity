import logging
import os
import csv
import re

logger = logging.getLogger(__name__)


class TryDataProcessor:

    def __init__(self, config):

        self.stop_word_list = None
        self.medicine_list = None
        self.symptom_list = None
        self.data_dir = config.data_dir

        self.medicine_replace_word = config.medicine_replace_word
        self.symptom_replace_word = config.symptom_replace_word

        if config.stop_word_valid:  # 读取停用词
            file_path = os.path.join(config.data_dir, "stop_word.txt")
            self.stop_word_list = self._read_dictionary(file_path)
        if config.medicine_valid:  # 读取医药用词
            file_path = os.path.join(config.data_dir, "medicine.txt")
            self.medicine_list = self._read_dictionary(file_path)
        if config.symptom_valid:  # 读取症状用词
            file_path = os.path.join(config.data_dir, "symptom.txt")
            self.symptom_list = self._read_dictionary(file_path)

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train_20200228.csv")))
        return self._read_csv(os.path.join(data_dir, "train_20200228.csv"))

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._read_csv(os.path.join(data_dir, "dev_20200228.csv"))

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._read_csv(data_dir)

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def get_second_task_labels(self):
        return ["肺炎", "支原体肺炎", "上呼吸道感染", "哮喘",
                "胸膜炎", "肺气肿", "感冒", "咳血"]

    def get_dev_labels(self, data_dir):
        label_list = []
        input_file = os.path.join(data_dir, "dev_20200228.csv")
        with open(input_file, "r", encoding="utf-8") as f:
            tsv_list = list(csv.reader(f, delimiter=','))
            for line in tsv_list[1:]:
                label_list.append(int(line[4]))
        return label_list

    def read_data_augment(self, augment_list):

        data_augment = []
        for augment in augment_list:
            input_file = os.path.join(self.data_dir, augment+".csv")
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
                text_a, text_b = self.text_preprocessing(text_a, text_b, self.medicine_list, self.medicine_replace_word)
                text_a, text_b = self.text_preprocessing(text_a, text_b, self.symptom_list, self.symptom_replace_word)
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


# if __name__ == "__main__":
#
#     class Config:
#         def __init__(self):
#             self.stop_word_valid = True
#             self.medicine_valid = True
#             self.symptom_valid = True
#             self.data_dir = '../real_data'
#             self.medicine_replace_word = '药物'
#             self.symptom_replace_word = '疾病'
#
#     config = Config()
#     processor = TryDataProcessor(config)
#     text_a, text_b = processor.text_preprocessing('治哮喘到北京德胜门中医院怎么样', '肺气肿引发心脏问题能打感冒预防针吗',
#                                                   processor.symptom_list, config.symptom_replace_word)
#     print(text_a, text_b)
#     text_a, text_b = processor.text_preprocessing('小孩子常用阿奇霉素有什么副作用', '多潘立酮混悬液和什么药物吃有反应',
#                                                   processor.medicine_list, config.medicine_replace_word)
#     print(text_a, text_b)
#     text_a, text_b = processor.text_preprocessing('请问肺气肿复发怎么办', '你好孩子4岁咳嗽可以吃罗红霉素吗',
#                                                   processor.stop_word_list)
#     print(text_a, text_b)
