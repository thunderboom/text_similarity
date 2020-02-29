import Levenshtein
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import jieba
import os
import pandas as pd



class FeaureExtraction():
    def __init__(self):
        self.column1 = 'query1'
        self.column2 = 'query2'
        self.file_path = '../real_data/symptom.txt'

    def graph_feature(self, df, graph_type='undirected'):  # udirected graph
        def q1_q2_in_intersect(row, column1, column2, q_in_dict):
            return (len(set(q_in_dict[row[column1]]).intersection(set(q_in_dict[row[column2]]))))
        def q1_q2_out_intersect(row, column1, column2, q_out_dict):
            return (len(set(q_out_dict[row[column1]]).intersection(set(q_out_dict[row[column2]]))))

        column1, column2 = self.column1, self.column2
        columns = df.columns
        ques = df[[column1, column2]]
        q_in_dict = defaultdict(set)
        q_out_dict = defaultdict(set)
        for i in tqdm(range(ques.shape[0])):
            q_out_dict[ques[column1][i]].add(ques[column2][i])
            q_in_dict[ques[column2][i]].add(ques[column1][i])
            if graph_type == 'undirected':
                q_out_dict[ques[column2][i]].add(ques[column1][i])
                q_in_dict[ques[column1][i]].add(ques[column2][i])
        if graph_type == 'undirected':
            df[column1 + '_' + graph_type] = df[column1].apply(lambda x: len(q_in_dict[x]))
            df[column2 + '_' + graph_type] = df[column2].apply(lambda x: len(q_in_dict[x]))
            df[column1 + '_' + column2 + '_' + 'intersect'] = df[[column1, column2]].apply(
                    lambda x: q1_q2_in_intersect(x, column1, column2, q_in_dict), axis=1)
        else:
            df[column1 + '_in_' + graph_type] = df[column1].apply(lambda x: len(q_in_dict[x]))
            df[column2 + '_in_' + graph_type] = df[column2].apply(lambda x: len(q_in_dict[x]))
            df[column1 + '_out_' + graph_type] = df[column1].apply(lambda x: len(q_out_dict[x]))
            df[column2 + '_out_' + graph_type] = df[column2].apply(lambda x: len(q_out_dict[x]))
            df[column1 + '_' + column2 + '_in_' + 'intersect'] = df[[column1, column2]].apply(
                    lambda x: q1_q2_in_intersect(x, column1, column2, q_in_dict), axis=1)
            df[column1 + '_' + column2 + '_out_' + 'intersect'] = df[[column1, column2]].apply(
                    lambda x: q1_q2_out_intersect(x, column1, column2, q_in_dict), axis=1)
        columns = list(set(df.columns) - set(columns))
        return df[columns]

    def distance_feature(self, df):
        column1, column2 = self.column1, self.column2
        columns = df.columns
        df['distance'] = df[[column1, column2]].apply(
            lambda x: Levenshtein.distance(x[column1], x[column2]), axis=1)
        df['ratio'] = df[[column1, column2]].apply(
            lambda x: Levenshtein.ratio(x[column1], x[column2]), axis=1)
        df['jaro'] = df[[column1, column2]].apply(
            lambda x: Levenshtein.jaro(x[column1], x[column2]), axis=1)
        df['jaro_winkler'] = df[[column1, column2]].apply(
            lambda x: Levenshtein.jaro_winkler(x[column1], x[column2]), axis=1)
        new_columns = list(set(df.columns) - set(columns))
        return df[new_columns]

    def extrem_feautre(self, df):
        def extreme_test(query):
            extreme_words = ['最']
            for word in extreme_words:
                if word in query:
                    return True
        df['extreme_tag_1'] = df['query1'].apply(lambda x: extreme_test(x))
        df['extreme_tag_2'] = df['query2'].apply(lambda x: extreme_test(x))
        df['extreme_tag'] = df['extreme_tag_1'] == df['extreme_tag_2']
        return df[['extreme_tag']]

    def dictionary_feature(self, df):
        def read_file(file_path):
            words_list = []
            with open(file_path, 'r', encoding='utf-8') as fr:
                for line in fr.readlines():
                    if '的' not in line:
                        words_list.append(line.strip('\n'))
            return words_list
        # def save_file(word_list):
        #     with open('./used_word.txt', 'w+', encoding='utf-8') as fw:
        #         for word in word_list:
        #             fw.write(word + '\n')
        def query_compare(query1, query2, words_list):
            query1_set = []
            query2_set = []
            for word in words_list:
                if word in query1:
                    query1_set.append(word)
                    #used_word.append(word)
                if word in query2:
                    query2_set.append(word)
                    #used_word.append(word)
            query_list = list((set(query1_set) | set(query2_set))- (set(query1_set) & set(query1_set)))
            if len(query_list) < 1:
                return 0
            else:
                return 1
        #used_word = []
        column1, column2 = self.column1, self.column2
        #症状
        symptom_words_list = read_file(self.file_path)
        df['symptom_tag'] = df[[column1, column2]].apply(
            lambda x: query_compare(x[column1], x[column2], words_list=symptom_words_list), axis=1
        )
        # used_word = list(set(used_word))
        # save_file(used_word)
        return df[['symptom_tag']]


    def word_bag_feature(self, df):
        def compute_similarity(query1, query2, type='dice'):
            query1_set = set(query1)
            query2_set = set(query2)
            if type == 'dice':
                return len(query1_set & query2_set) / (len(query1_set) + len(query2_set))
            elif type == 'jaccard':
                return len(query1_set & query2_set) / len(query1_set | query2_set)
            elif type == 'sim_set':
                return len(query1_set & query2_set) / min(len(query1_set), len(query2_set))
            else:
                return ValueError("The input type error")

        column1, column2 = self.column1, self.column2
        columns = df.columns
        df['word_bag_dice'] = df[[column1, column2]].apply(
            lambda  x: compute_similarity(x[column1], x[column2], 'dice'), axis=1
        )
        df['word_bag_jaccard'] = df[[column1, column2]].apply(
            lambda x: compute_similarity(x[column1], x[column2], 'jaccard'), axis=1
        )
        df['word_sim'] = df[[column1, column2]].apply(        #from error test
            lambda x: compute_similarity(x[column1], x[column2], 'sim_set'), axis=1
        )
        new_columns = list(set(df.columns) - set(columns))
        return df[new_columns]


    def Ngram_feature(self, df, n=2):
        column1, column2 = self.column1, self.column2
        distance = []
        intersection = []
        columns = df.columns
        for index, row in df.iterrows():
            q1 = row[column1]
            q2 = row[column2]
            q1_list = ' '.join(jieba.cut(q1)).split()
            q2_list = ' '.join(jieba.cut(q2)).split()
            q1_combined = set()
            q2_combined = set()
            for i in range(len(q1_list) - n+1):
                q1_combined.add(''.join(q1_list[i: i+n]))
            for i in range(len(q2_list) - n+1):
                q2_combined.add(''.join(q2_list[i: i+n]))
            difference = len(set.union(q1_combined.difference(q2_combined), q2_combined.difference(q1_combined)))
            similarity = len(q1_combined.intersection(q2_combined))
            distance.append(difference)
            intersection.append(similarity)
        df[str(n)+'_gram_difference'] = distance
        df[str(n)+'_gram_similariy'] = intersection
        new_columns = list(set(df.columns) - set(columns))
        return df[new_columns]



    
    def feature_label_plot(self, df, labels, feature_type):
        if feature_type != 'box' and feature_type != 'point':
            raise ValueError('feature_type must choose from box and point')
        df['label'] = labels
        for column in df.columns:
            if feature_type == 'box':
                plt.figure(figsize=(12, 8))
                sns.boxplot(x='label', y=column, data=df)
                plt.xlabel('label tag', fontsize=20)
                plt.ylabel('feature {} value'.format(column), fontsize=20)
                plt.show()
            else:
                grouped_df = df.groupby(column)['label'].\
                    aggregate(np.mean).reset_index()
                plt.figure(figsize=(12, 8))
                sns.pointplot(grouped_df[column].values, grouped_df['label'].values)
                plt.ylabel('Mean label', fontsize=20)
                plt.xlabel('feature {} value'.format(column), fontsize=20)
                plt.xticks(rotation='vertical')
                plt.show()
        return None



#test
if __name__ == "__main__":
    train_path = os.path.join('../real_data/', 'train.csv')
    dev_path = os.path.join('../real_data/', 'dev.csv')
    train_data = pd.read_csv(train_path)
    dev_data = pd.read_csv(dev_path)
    train_data = pd.concat([train_data, dev_data], axis=0, ignore_index=True)
    print(len(train_data))
    Extraction = FeaureExtraction()
    #总共11个特征
    #df_feature = Extraction.Ngram_feature(train_data, n=1)  #2个特征   #上下文
    #df_feature = Extraction.distance_feature(train_data)     #4个特征  #距离
    #df_feature = Extraction.graph_feature(train_data, graph_type='directed')  #数据集统计特征
    df_feature = Extraction.word_bag_feature(train_data)  #3个特征   #词袋特征
    #df_feature = Extraction.extrem_feautre(train_data)    #1个特征   #文本特征(极端)
    #df_feature = Extraction.dictionary_feature(train_data)  #1个特征  #文本特征(pair)
    Extraction.feature_label_plot(df_feature, train_data['label'], feature_type='box')

