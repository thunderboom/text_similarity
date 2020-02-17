import Levenshtein
import numpy as np
import os
import pandas as pd

def distance_feature(df_train):
    column1, column2 = 'question1', 'question2'
    columns = df_train.columns
    df_train['distance'] = df_train[[column1, column2]].apply(lambda x:Levenshtein.distance(x[column1], x[column2]), axis=1)
    df_train['distance1'] = df_train[[column1, column2]].apply(lambda x:Levenshtein.distance(x[column1], x[column2]) / max(len(x[column1]), len(x[column2])), axis=1)
    df_train['distance2'] = df_train[[column1, column2]].apply(lambda x:Levenshtein.distance(x[column1], x[column2]) / max(1, abs(len(x[column1])-len(x[column2]))), axis=1)
    df_train['ratio'] = df_train[[column1, column2]].apply(lambda x: Levenshtein.ratio(x[column1], x[column2]), axis=1)
    df_train['jaro'] = df_train[[column1, column2]].apply(lambda x: Levenshtein.jaro(x[column1], x[column2]), axis=1)
    df_train['jaro_winkler'] = df_train[[column1, column2]].apply(lambda x: Levenshtein.jaro_winkler(x[column1], x[column2]), axis=1)
    new_columns = df_train.columns - columns
    return np.asarray(df_train[new_columns], dtype=float)



def graph_feature(df):
    def q_index(df_train):
        q_list = list(np.unique(list(df_train['question1']) + list(df_train['question2'])))
        q_id = {}
        for idx, q in enumerate(q_list):
            q_id[q] = idx
        return q_id
    def ng_matrix(df_train, q_id):
        neighbor_matrix = np.zeros((len(q_id), len(q_id)))
        df_graph = df_train[df_train['label'] == 1]
        for index, row in df_graph.iterrows():
            i = row['q1_id']
            j = row['q2_id']
            neighbor_matrix[i, j] += 1
        return neighbor_matrix
    def compute_indot(text, ng_matrix, q_id):
        i = q_id[text]
        in_dot = np.sum(ng_matrix[i, :])
        return in_dot
    def compute_outdot(text, ng_matrix, q_id):
        i = q_id[text]
        out_dot = np.sum(ng_matrix[:, i])
        return out_dot

    column1, column2 = 'question1', 'question2'
    df = df[[column1, column2, 'label']]
    q_id = q_index(df)
    df['q1_id'] = df[column1].apply(lambda x: q_id[x])
    df['q2_id'] = df[column2].apply(lambda x: q_id[x])
    neighbor_matrix = ng_matrix(df, q_id)
    df['q1_indot'] = df[column1].apply(lambda x: compute_indot(x, neighbor_matrix, q_id))
    df['q1_outdot'] = df[column1].apply(lambda x: compute_outdot(x, neighbor_matrix, q_id))
    df['q1_dot'] = df['q1_indot'] + df['q1_outdot']
    df['q2_indot'] = df[column2].apply(lambda x: compute_indot(x, neighbor_matrix, q_id))
    df['q2_outdot'] = df[column2].apply(lambda x: compute_outdot(x, neighbor_matrix, q_id))
    df['q2_dot'] = df['q2_indot'] + df['q2_outdot']
    columns = ['q1_indot', 'q1_outdot', 'q1_dot', 'q2_indot', 'q2_outdot', 'q2_dot']
    return np.asarray(df[columns], dtype=np.int32)




# if __name__ == "__main__":
#     path = os.path.join('./try_data/', 'train.csv')
#     train_data = pd.read_csv(path)
#     print(distance_feature(train_data[:10000]))
#     print(graph_feature(train_data[:10000]))
