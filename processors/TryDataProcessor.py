import logging
import os
import csv

logger = logging.getLogger(__name__)


class TryDataProcessor:

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

    def get_all_examples(self, data_dir):
        """See base class."""
        train = self.get_train_examples(data_dir)
        dev = self.get_dev_examples(data_dir)
        test = self.get_test_examples(data_dir)
        return train, dev, test

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

    @classmethod
    def _read_csv(cls, input_file):
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
                if is_test:
                    data_list.append([line[2], line[3], '0', line[1]])
                else:
                    data_list.append([line[2], line[3], line[4], line[1]])
                #data_list.append([line[0], line[1], line[2], line[3]])
        return data_list


# if __name__ == "__main__":
#     print(TryDataProcessor._read_csv(input_file='../try_data/train.csv'))