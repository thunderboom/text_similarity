import numpy as np
import pandas as pd


def medicine_symptom_dict(train_path, dev_path, save_path):

    def common_str(query_list):  # 最长公共字串
        short_str = sorted(query_list, key=lambda i:len(i), reverse=False)[0]
        longest_str = ''
        for prior in range(len(short_str)-1):
            for latter in range(prior+1, len(short_str)):
                try_str = short_str[prior: latter]
                if min([try_str in str_ for str_ in query_list]) and len(try_str) > len(longest_str):
                    longest_str = try_str
                else:
                    continue
        return longest_str

    df_train = pd.read_csv(train_path)
    df_dev = pd.read_csv(dev_path)
    df_all = pd.concat([df_train, df_dev], axis=0).reset_index()
    common_str_list = []
    querys_list = np.unique(df_all['query1'])
    for query in querys_list:
        query_try_list = df_all[df_all['query1']==query]['query2'].tolist()
        query_try_list.append(query)
        longest_str = common_str(query_try_list)
        if len(longest_str) > 1:
            common_str_list.append(longest_str)
    common_str_list = list(np.unique(common_str_list))
    with open(save_path, 'w+', encoding='utf-8') as fw:
        for str_ in common_str_list:
            fw.write(str_+'\n')
    print("finish generate")
    return None


if __name__ == "__main__":
    train_path = '../data/Dataset/train.csv'
    dev_path = '../data/Dataset/dev.csv'
    save_path = '../user_data/tmp_data/try_medicine_sypmtom.txt'
    medicine_symptom_dict(train_path, dev_path, save_path)
