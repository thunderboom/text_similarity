import pandas as pd
import re
import numpy as np
import copy


def list_dataframe(datalist):
    """
    list to DataFrame
    """
    question1 = []
    question2 = []
    label = []
    category = []
    for row in datalist:
        question1.append(row[0])
        question2.append(row[1])
        label.append(int(row[2]))
        category.append(row[3])
    return pd.DataFrame({'query1': question1, 'query2': question2, 'label': label, 'category': category})


def dataframe_list(dataframe):
    """
    DataFrame to list
    """
    data_list = []
    for idx, row in dataframe.iterrows():
        row_list = [row['query1'], row['query2'], str(row['label']), row['category']]
        data_list.append(row_list)
    return data_list


def augment_data_save(data_examples, file_name):
    """
    保存增强后的数据
    :param data_examples:
    :param file_name:
    :return:
    """
    question1 = []
    question2 = []
    label = []
    category = []
    for row in data_examples:
        question1.append(row[0])
        question2.append(row[1])
        label.append(int(row[2]))
        category.append(row[3])
    idx = [i for i in range(len(question1))]
    save_df = pd.DataFrame({'id': idx, 'category': category, 'query1': question1, 'query2': question2, 'label': label})
    save_df.to_csv(file_name, index=False)
    print("file {} saved, lens {}".format(file_name, len(save_df)))


def sentence_set_pair(train_examples, random_state=20):
    questions1 = []
    questions2 = []
    labels = []
    categories = []
    df_train = list_dataframe(train_examples)
    column1, column2, column3, column4 = 'query1', 'query2', 'label', 'category'
    query_1_list = list(np.unique(df_train['query1']))
    for query_tag in query_1_list:
        df_query = df_train[df_train['query1'] == query_tag]
        query_same_set = df_query[df_query['label'] == 1]['query2'].tolist()
        query_diff_set = df_query[df_query['label'] == 0]['query2'].tolist()
        category = list(df_query['category'])[0]
        if len(query_same_set) >= 1:  # 如果有与query1相似的问题
            if len(query_diff_set) >= 1:  # 类别间
                for query_1 in query_same_set:
                    for query_2 in query_diff_set:
                        questions1.append(query_1)
                        questions2.append(query_2)
                        labels.append('0')
                        categories.append(category)
            if len(query_same_set) >= 2:  # 类别内
                for i in range(len(query_same_set) - 1):
                    for j in range(i + 1, len(query_same_set)):
                        questions1.append(query_same_set[i])
                        questions2.append(query_same_set[j])
                        labels.append('1')
                        categories.append(category)
    new_df = pd.DataFrame(
        {column1: questions1, column2: questions2, column3: labels, column4: categories})
    positive_num = len(new_df[new_df[column3] == '1'])
    negative_num = len(new_df[new_df[column3] == '0'])

    extract_num = int(min(positive_num, negative_num))
    df_postive = new_df[new_df[column3] == '1']
    df_negative = new_df[new_df[column3] == '0']
    df_postive = df_postive.sample(n=extract_num, replace=False, random_state=random_state)
    df_negative = df_negative.sample(n=extract_num, replace=False, random_state=random_state)
    new_df = pd.concat([df_postive, df_negative], ignore_index=True)
    new_df = new_df.sample(frac=1.0, replace=False, random_state=random_state)
    print("choose the balanced augment data")
    return dataframe_list(new_df)


def new_category_generate(train_examples, dev_examples, medicine_examples, new_category_path):
    """
    新的category生成函数
    :param train_path:
    :param dev_path:
    :param medicine_path:
    :param new_category_path:
    :return:
    """
    def create_category_data(df, query_index, add_category, num, medicine_dict):
        category_list, query1_list, query2_list, label_list = [], [], [], []
        for category in add_category:
            medicine_new = medicine_dict[category]
            sample_idx = 0
            k = -1
            while sample_idx < num:
                for idx, query_ in enumerate(query_index):
                    wrong_words = ['关系']
                    if not max([word in query_ for word in wrong_words]):
                        df_try = df[df['query1'] == query_]
                        for index_, (index, row) in enumerate(df_try.iterrows()):  # 病名
                            query1_try = re.sub(r'呼吸道反复感染|呼吸道感染|上呼吸道感染|上呼道感染|感染上呼吸道|变异性哮喘|过敏性哮喘|哮喘|支原体肺炎|肺炎支原体|支原体感染',
                                                category, row['query1'])
                            query2_try = re.sub(r'呼吸道反复感染|呼吸道感染|上呼吸道感染|上呼道感染|感染上呼吸道|变异性哮喘|过敏性哮喘|哮喘|支原体肺炎|肺炎支原体|支原体感染',
                                                category, row['query2'])
                            for medicine in medicine_list:
                                if medicine in query1_try:
                                    if index_ == 0:
                                        if k < len(medicine_new) - 1:
                                            k += 1
                                        else:
                                            k = 0
                                    medicine_try = medicine_new[k]
                                    query1_try = re.sub(medicine, medicine_try, query1_try)
                                    query2_try = re.sub(medicine, medicine_try, query2_try)
                            query1_list.append(query1_try)
                            query2_list.append(query2_try)
                            category_list.append(category)
                            label_list.append(row['label'])
                            sample_idx += 1
                        query_index.remove(query_)
                        if sample_idx >= num:
                            break

                    else:
                        continue

        new_df = pd.DataFrame(
            {'id': range(len(category_list)), 'category': category_list, 'query1': query1_list, 'query2': query2_list,
             'label': label_list})
        return new_df

    df_train = list_dataframe(train_examples)
    df_dev = list_dataframe(dev_examples)
    df_all = pd.concat([df_train, df_dev], axis=0).reset_index()

    medicine_list = medicine_examples
    # processing
    category = ['上呼吸道感染', '哮喘', '支原体肺炎']
    new_category = ['支气管炎', '肺结核']
    basic_query = []
    query_list = df_train['query1'].tolist() + df_dev['query1'].tolist()
    for query in query_list:
        for cater in category:
            if cater in query:
                basic_query.append(query)
                break
            else:
                continue
    basic_query = list(np.unique(basic_query))

    medicine_dict = {
        '支气管炎': ['桂龙咳喘宁胶囊', '联邦克立停', '氨茶碱', '美川清', '博利康尼', '复方氯喘片', '异丙托溴铵气雾剂', '博利康尼'],
        '肺结核': ['异烟肼', '酶联素', '链霉素', '利福平', '乙胺丁醇', '吡嗪酰胺', '对氨水杨酸']
    }
    new_df = create_category_data(df=df_all, query_index=basic_query, add_category=new_category, num=1000,
                                  medicine_dict=medicine_dict)
    new_df.to_csv(new_category_path, encoding='utf-8', index=False)
    print('amount of data extracted from chip2019:{}'.format(len(new_df)))
    print("generate finished.")


def examples_extract(data, predict_label_prob, saved_file, sel_prob=(0.3, 0.8), random_state=None):

    predict_label_prob = np.array(predict_label_prob)
    data = np.array(data)

    # 选择预测区间的 原数据
    sel_prob_start = predict_label_prob > sel_prob[0]
    sel_prob_end = predict_label_prob < sel_prob[1]
    sel_data = data[sel_prob_start & sel_prob_end]
    print('sel_data: {}'.format(len(sel_data)))

    # 选择预测区间的 预测概率
    sel_data_prob = predict_label_prob[sel_prob_start & sel_prob_end]
    sel_data_id = [i for i in range(len(sel_data))]

    df_sel = pd.DataFrame(
        {'id': sel_data_id, 'category': list(sel_data[:, 3]), 'query1': list(sel_data[:, 0]),
         'query2': list(sel_data[:, 1]), 'label': list(sel_data[:, 2]), 'prob': list(sel_data_prob)})

    # 1:1抽取
    positive_num = len(df_sel[df_sel['label'] == '1'])
    negative_num = len(df_sel[df_sel['label'] == '0'])
    print('positive_num:{} negative_num:{}'.format(positive_num, negative_num))
    extract_num = int(min(positive_num, negative_num))

    df_postive = df_sel[df_sel['label'] == '1']
    df_negative = df_sel[df_sel['label'] == '0']
    df_postive = df_postive.sample(n=extract_num, replace=False, random_state=random_state)
    df_negative = df_negative.sample(n=extract_num, replace=False, random_state=random_state)
    new_df = pd.concat([df_postive, df_negative], ignore_index=True)
    new_df = new_df.sample(frac=1.0, replace=False, random_state=random_state)
    print('amount of data extracted from chip2019:{}'.format(len(new_df)))
    new_df.to_csv(saved_file, index=False)







