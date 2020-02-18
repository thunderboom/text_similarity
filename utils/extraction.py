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
        self.column1 = 'question1'
        self.column2 = 'question2'

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
        df['distance1'] = df[[column1, column2]].apply(
            lambda x: Levenshtein.distance(x[column1], x[column2]) / max(len(x[column1]), len(x[column2])), axis=1)
        df['distance2'] = df[[column1, column2]].apply(
            lambda x: Levenshtein.distance(x[column1], x[column2]) / max(1, abs(len(x[column1]) - len(x[column2]))),
            axis=1)
        df['ratio'] = df[[column1, column2]].apply(lambda x: Levenshtein.ratio(x[column1], x[column2]),
                                                               axis=1)
        df['jaro'] = df[[column1, column2]].apply(lambda x: Levenshtein.jaro(x[column1], x[column2]),
                                                              axis=1)
        df['jaro_winkler'] = df[[column1, column2]].apply(
            lambda x: Levenshtein.jaro_winkler(x[column1], x[column2]), axis=1)
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


# #test
# if __name__ == "__main__":
#     path = os.path.join('../try_data/', 'train.csv')
#     train_data = pd.read_csv(path)
#     Extraction = FeaureExtraction()
#     #df = Extraction.Ngram_feature(train_data[:1000], n=1)
#     #df = Extraction.graph_feature(train_data[:1000], graph_type='undirected')
#     df = Extraction.distance_feature(train_data[:1000])
#     Extraction.feature_label_plot(df, train_data['label'][:1000], feature_type='box')

