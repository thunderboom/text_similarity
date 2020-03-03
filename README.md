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

