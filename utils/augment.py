import pandas as pd
import jieba
import random
import os

class DataAugment():
    def __init__(self):
        self.columns = ['question1', 'question2', 'label', 'category']
        self.save_path = '../try_data/augment.csv'

    def sentences_dropthemword(self, df_train, key_words, prob):
        '''prob=False, drop the keyword all from the sentence'''
        '''prob=True, if two sentence both have keyword, %50 drop it,one sentence have keyword %25 drop it'''
        def drop_themword(word, key_words):
            '''key words is about '''
            if word not in key_words:
                return word
            else:
                return ''
        new_questions1 = []
        new_questions2 = []
        new_labels = []
        new_categories = []
        column1, column2, column3, column4 = self.columns[0], self.columns[1], self.columns[2], self.columns[3]
        for index, row in df_train.iterrows():
            q1, q2, label, category = row[column1], row[column2], row[column3], row[column4]
            q1 = list(jieba.cut(q1))
            q2 = list(jieba.cut(q2))
            new_q1 = ''.join([drop_themword(word, key_words) for word in q1])
            new_q2 = ''.join([drop_themword(word, key_words) for word in q2])
            if len(q1) != len(new_q1) or len(q2) != len(new_q2):
                if prob == False:
                    new_questions1.append(new_q1)
                    new_questions2.append(new_q2)
                    new_labels.append(label)
                    new_categories.append(category)
                else:
                    print('sample')
                    if random.random() > 0.5:
                        if len(q1) != len(new_q1) and len(q2) != len(new_q2):
                            new_questions1.append(new_q1)
                            new_questions2.append(new_q2)
                            new_labels.append(label)
                            new_categories.append(category)
                        else:
                            if random.random() > 0.5:
                                new_questions1.append(new_q1)
                                new_questions2.append(new_q2)
                                new_labels.append(label)
                                new_categories.append(category)
        new_df = pd.DataFrame({column1:new_questions1, column2:new_questions2, column3:new_labels, column4:new_categories})
        return new_df

    def Pseudolabel(self):
        '''伪标签'''
        return None

    def symmetric_sentence(self, df_train):
        """对称性"""
        column1, column2 = 'question1', 'question2'
        question1 = list(df_train[column1])
        df_train[column1] = list(df_train[column2])
        df_train[column2] = question1
        return df_train

    def dataAugment(self, df_train, *args):
        #args:'symmetric'=> symmetric_sentence, 'themword'=>sentences_dropthemword, 'pseudo'=>Pseudolabel
        for arg in args:
            if arg not in ['symmetric', 'themword', 'pseudo']:
                raise ValueError('the input must choose from [''symmetric', 'themword', 'pseudo'']')
        keywords = ['糖尿病', '高血压', '艾滋病', '乳腺癌', '乙肝']
        prob = False
        for idx, arg in enumerate(args):
            if arg == 'symmetric':
                new_data = self.symmetric_sentence(df_train)
            elif arg == 'themword':
                new_data = self.sentences_dropthemword(df_train, keywords, prob)
            elif arg == 'pseudo':
                new_data = self.Pseudolabel()
            else:
                raise ValueError("The input must choose from ['symmetric','themword','pseudo']")
            if idx == 0:
                all_data = new_data
            else:
                all_data = pd.concat((all_data, new_data), ignore_index=True)
        all_data.to_csv(self.save_path, index=False)
        print('genarate data finished')
        return None


#test
if __name__ == "__main__":
    path = os.path.join('../try_data/', 'train.csv')
    train_data = pd.read_csv(path)
    Augment = DataAugment()
    print(Augment.dataAugment(train_data, 'themword'))

