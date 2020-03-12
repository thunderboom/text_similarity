# Base Images
## 从天池基础镜像构建
# FROM registry.cn-shanghai.aliyuncs.com/tcc-public/python:3

FROM registry.cn-shanghai.aliyuncs.com/tcc-public/pytorch:1.1.0-cuda10.0-py3

RUN mkdir -p /models
COPY models/bert.py /models

RUN mkdir -p /loss
COPY loss/focal_loss.py /loss

RUN mkdir -p /real_data
COPY real_data/* /real_data/

#RUN mkdir -p /pretrain_models/bert-base-chinese
#COPY pretrain_models/bert-base-chinese/* /pretrain_models/bert-base-chinese/

#RUN mkdir -p /pretrain_models/chinese_roberta_wwm_large_ext_pytorch
#COPY pretrain_models/chinese_roberta_wwm_large_ext_pytorch/* /pretrain_models/chinese_roberta_wwm_large_ext_pytorch/

RUN mkdir -p /pretrain_models/ERNIE
COPY pretrain_models/ERNIE/* /pretrain_models/ERNIE/

RUN mkdir -p /pretrain_models/roberta_large_pair
COPY pretrain_models/roberta_large_pair/* /pretrain_models/roberta_large_pair/

RUN mkdir -p /processors
COPY processors/* /processors/
RUN mkdir -p /utils
COPY utils/* /utils/
COPY test.py /

#RUN mkdir -p model_saved/base_real_data/base_bert
#RUN mkdir -p model_saved/base_try_data

#RUN mkdir -p model_saved/base_real_data/roberta_wwm_large_ext_pytorch
#COPY model_saved/base_real_data/roberta_wwm_large_ext_pytorch/roberta_wwm_large_ext_pytorch.pkl /model_saved/base_real_data/roberta_wwm_large_ext_pytorch/

RUN mkdir -p model_saved/base_real_data/ernie
COPY model_saved/base_real_data/ernie/* /model_saved/base_real_data/ernie/

RUN mkdir -p model_saved/base_real_data/roberta_large_pair
COPY model_saved/base_real_data/roberta_large_pair/* /model_saved/base_real_data/roberta_large_pair/

COPY run.sh /
COPY requirements.txt /
RUN mkdir -p /package_need
COPY package_need3/* /package_need/
#RUN pip download -r /requirements.txt -d /package_need -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install -r /requirements.txt --no-index --find-links=/package_need
RUN mkdir -p /tcdata
WORKDIR /

CMD ["sh","run.sh"]