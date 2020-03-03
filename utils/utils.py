import copy
import json
import logging
import numpy as np
import torch.utils.data as Data
import torch


logger = logging.getLogger(__name__)


class InputExample(object):
    """
    A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
        text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
        label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """

    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """
    A single set of features of data.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        label: Label corresponding to the input
    """

    def __init__(self, input_ids, attention_mask=None, token_type_ids=None, label=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


# 单句特征存储单元
class InputFeaturesSentence(object):

    def __init__(self, input_ids_1=None, attention_mask_1=None, token_type_ids_1=None, input_ids_2=None,
                 attention_mask_2=None, token_type_ids_2=None, label=None):
        self.input_ids_1 = input_ids_1
        self.attention_mask_1 = attention_mask_1
        self.token_type_ids_1 = token_type_ids_1
        self.input_ids_2 = input_ids_2
        self.attention_mask_2 = attention_mask_2
        self.token_type_ids_2 = token_type_ids_2
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


# 多任务
class InputFeaturesMultiTask(object):

    def __init__(self, input_ids, attention_mask=None, token_type_ids=None, label=None, second_label=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label
        self.second_label = second_label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def convert_examples_to_features(
    examples,
    tokenizer,
    label_list,
    second_label_list=None,
    max_length=512,
    pad_token=0,
    pad_token_segment_id=0,
    data_type=None,
):
    """
    :param examples: List [ sentences1,sentences2,label, category]
    :param tokenizer: Instance of a tokenizer that will tokenize the examples
    :param label_list: List of labels.
    :param second_label_list:
    :param max_length: Maximum example length
    :param pad_token: 0
    :param pad_token_segment_id: 0
    :return: [(example.guid, input_ids, attention_mask, token_type_ids, label), ......]
    """
    label_map = {label: i for i, label in enumerate(label_list)}

    if second_label_list:
        second_label_map = {label: i for i, label in enumerate(second_label_list)}

    features = []
    for (index, example) in enumerate(examples):
        inputs = tokenizer.encode_plus(example[0], example[1], add_special_tokens=True, max_length=max_length)
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
        attention_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)

        input_ids = input_ids + ([pad_token] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        if example[2] is not None:
            label = label_map[example[2]]
        else:
            label = 0

        if second_label_list:
                second_label = second_label_map[example[3]]
        else:
            second_label = 0

        # if index < 3:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s" % (index))
        #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
        #     logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
        #
        #     if example[2] is not None:
        #         logger.info("label: %s (id = %d)" % (example[2], label))

        if second_label_list:
            features.append(
                InputFeaturesMultiTask(input_ids, attention_mask, token_type_ids, label, second_label)
            )
        else:
            features.append(
                InputFeatures(input_ids, attention_mask, token_type_ids, label)
            )

    return features


class BuildDataSet(Data.Dataset):
    """
    将经过convert_examples_to_features的数据 包装成 Dataset
    """
    def __init__(self, features):
        self.features = features

    def __getitem__(self, index):
        feature = self.features[index]
        input_ids = np.array(feature.input_ids)
        attention_mask = np.array(feature.attention_mask)
        token_type_ids = np.array(feature.token_type_ids)
        label = np.array(feature.label)

        return input_ids, attention_mask, token_type_ids, label

    def __len__(self):
        return len(self.features)


# 单句特征
def convert_examples_to_features_sentence(
    examples,
    tokenizer,
    label_list,
    second_label_list=None,
    max_length=512,
    pad_token=0,
    pad_token_segment_id=0,
    data_type=None,
):
    """
    :param examples: List [ sentences1,sentences2,label, category]
    :param tokenizer: Instance of a tokenizer that will tokenize the examples
    :param label_list: List of labels.
    :param second_label_list:
    :param max_length: Maximum example length
    :param pad_token: 0
    :param pad_token_segment_id: 0
    :return: [(example.guid, input_ids, attention_mask, token_type_ids, label), ......]
    """
    label_map = {label: i for i, label in enumerate(label_list)}

    if second_label_list:
        second_label_map = {label: i for i, label in enumerate(second_label_list)}

    features = []
    for (index, example) in enumerate(examples):

        if data_type == 'train' and np.random.random() < 0.5:
            example[0], example[1] = example[1], example[0]

        inputs_1 = tokenizer.encode_plus(example[0], add_special_tokens=True, max_length=max_length)
        input_ids_1, token_type_ids_1 = inputs_1["input_ids"], inputs_1["token_type_ids"]
        attention_mask_1 = [1] * len(input_ids_1)

        inputs_2 = tokenizer.encode_plus(example[1], add_special_tokens=True, max_length=max_length)
        input_ids_2, token_type_ids_2 = inputs_2["input_ids"], inputs_2["token_type_ids"]
        attention_mask_2 = [1] * len(input_ids_2)

        # Zero-pad up to the sequence length.
        padding_length_1 = max_length - len(input_ids_1)
        input_ids_1 = input_ids_1 + ([pad_token] * padding_length_1)
        attention_mask_1 = attention_mask_1 + ([0] * padding_length_1)
        token_type_ids_1 = token_type_ids_1 + ([pad_token_segment_id] * padding_length_1)

        padding_length_2 = max_length - len(input_ids_2)
        input_ids_2 = input_ids_2 + ([pad_token] * padding_length_2)
        attention_mask_2 = attention_mask_2 + ([0] * padding_length_2)
        token_type_ids_2 = token_type_ids_2 + ([pad_token_segment_id] * padding_length_2)

        if example[2] is not None:
            label = label_map[example[2]]
        else:
            label = 0

        if second_label_list:
                second_label = second_label_map[example[3]]
        else:
            second_label = 0

        # if index < 3:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s" % (index))
        #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
        #     logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
        #
        #     if example[2] is not None:
        #         logger.info("label: %s (id = %d)" % (example[2], label))

        features.append(
            InputFeaturesSentence(input_ids_1, attention_mask_1, token_type_ids_1,
                                  input_ids_2,  attention_mask_2, token_type_ids_2, label)
        )

    return features


class BuildDataSetSentence(Data.Dataset):
    """
    convert_examples_to_features_sentence 包装成 Dataset
    """
    def __init__(self, features):
        self.features = features

    def __getitem__(self, index):
        feature = self.features[index]

        input_ids_1 = np.array(feature.input_ids_1)
        attention_mask_1 = np.array(feature.attention_mask_1)
        token_type_ids_1 = np.array(feature.token_type_ids_1)

        input_ids_2 = np.array(feature.input_ids_2)
        attention_mask_2 = np.array(feature.attention_mask_2)
        token_type_ids_2 = np.array(feature.token_type_ids_2)

        label = np.array(feature.label)

        return input_ids_1, attention_mask_1, token_type_ids_1, \
               input_ids_2, attention_mask_2, token_type_ids_2, label

    def __len__(self):
        return len(self.features)


class BuildDataSetMultiTask(Data.Dataset):
    """
    将经过convert_examples_to_features的数据 包装成 Dataset
    """
    def __init__(self, features):
        self.features = features

    def __getitem__(self, index):
        feature = self.features[index]
        input_ids = np.array(feature.input_ids)
        attention_mask = np.array(feature.attention_mask)
        token_type_ids = np.array(feature.token_type_ids)
        label = np.array(feature.label)
        second_label = np.array(feature.second_label)

        return input_ids, attention_mask, token_type_ids, label, second_label

    def __len__(self):
        return len(self.features)


def config_to_dict(config):

    output = copy.deepcopy(config.__dict__)
    if hasattr(config.__class__, "model_type"):
        output["model_type"] = config.__class__.model_type
    output['device'] = config.device.type
    return output


def config_to_json_string(config):
    """Serializes this instance to a JSON string."""
    return json.dumps(config_to_dict(config), indent=2, sort_keys=True) + '\n'


def random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_test_split(config, examples):

    config.test_num_examples = int(len(examples) * config.test_split)
    config.train_num_examples = len(examples) - config.test_num_examples
    train_data, test_data = Data.random_split(examples, [config.train_num_examples, config.test_num_examples])

    train_examples = [train_data.dataset[idx] for idx in train_data.indices]
    test_examples = [test_data.dataset[idx] for idx in test_data.indices]

    return train_examples, test_examples


def k_fold_volt_predict(predict_labels):
    predict_sets = np.array(predict_labels)
    last_predict = []
    for idx in range(predict_sets.shape[1]):
        pred = predict_sets[:, idx].mean()
        last_predict.append(int(pred >= 0.50))

    return last_predict
