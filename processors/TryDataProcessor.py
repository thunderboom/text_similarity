from utils import InputExample
import logging
import os
import csv

logger = logging.getLogger(__name__)


class TryDataProcessor:

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.csv")))
        return self._create_examples(self._read_csv(os.path.join(data_dir, "train.csv"),), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return []

    def get_test_examples(self, data_dir):
        """See base class."""
        return []

    def get_augment_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "augment.csv")))
        return self._create_examples(self._read_csv(os.path.join(data_dir, "augment.csv"),), "augment")

    def get_all_examples(self, data_dir):
        """See base class."""
        train = self.get_train_examples(data_dir)
        dev = self.get_dev_examples(data_dir)
        test = self.get_test_examples(data_dir)
        return train, dev, test

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i+1)
            text_a = line[0]
            text_b = line[1]
            label = line[2]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    @classmethod
    def _read_csv(cls, input_file):
        data_list = []
        with open(input_file, "r", encoding="utf-8-sig") as f:
            tsv_list = list(csv.reader(f))
            for line in tsv_list[1:]:
                data_list.append([line[0], line[1], line[2], line[3]])
        return data_list

# if __name__ == "__main__":
#     print(TryDataProcessor._read_csv(input_file=''))