EDA
https://www.kaggle.com/huskylovers/sentence-pair-eda/

数据增强说明：

调用augment.py生成augment.csv文件。

在run.py增加读入功能模块，修改参考run_bert_base.py文件。


utils/augment.py

数据增强功能包括：去掉主题词，修改语句顺序，伪标签（未完成）

utils/extraction.py

特征提取模块包括:距离特征，Ngram特征，图特征。并嵌入了点图和箱型图的可视功能。




## 提交历史说明
- 成绩：0.9477  
> 说明：  
base-bert + full-train + 5 epoch  
mutil-sample-drop:4  
lr:2e-5  

- 成绩：0.9546 
> 说明：  
ERNIE + full-train  
mutil-sample-drop:4  
lr:2e-5  

## 小想法
- 额外数据：
>（1）使用chip2019的数据。筛选方法：使用已经训练好的模型对chip2019的数据进行预测，从中选出识别正确的样本作为增强数据。即在不对模型效果变动的情况下，增强部分噪音数据。


- 训练方法：
> （1）使用某种方式，减少epoch数，使loss和acc尽可能一致。
- 模型融合
> 视角修正
不是单纯的不同预训练模型进行叠加，而是间接使用了预训练模型当中的很多机制。也可以同一个预训练模型，融合不同的机制。
> 原则
在效果上相差不大的基础上，以增加模型的多样性为目的，多样性评价标准（计算验证集上两个模型的prob预测的相似度）。
> 模型融合示例
no focal_loss + focal_loss
no_weighted_sum + weighted_sum
                    


