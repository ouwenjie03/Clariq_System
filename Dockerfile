FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime
MAINTAINER ouwenjie@corp.netease.com

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai

RUN apt-get update \
    && apt-get install -y vim locales ntp tzdata\
    && rm -rf /var/lib/apt/lists/*

# 设置 locales and time
RUN echo 'en_US.UTF-8 UTF-8' >>  /etc/locale.gen \
    && /usr/sbin/locale-gen \
    && ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime \
    && service ntp restart
RUN echo 'export LC_ALL=en_US.UTF-8' >> /root/.bashrc
RUN ln -s -f /bin/bash /bin/sh

ADD ./requirements.txt /workspace/.
RUN pip install -r /workspace/requirements.txt

RUN mkdir /workspace/data
ADD ./data/train.tsv /workspace/data/train.tsv
ADD ./data/nltk_data /workspace/data/nltk_data
ADD ./data/question_bank.tsv /workspace/data/question_bank.tsv
ADD ./models /workspace/models

ADD ./BertForMultitask.py /workspace/BertForMultitask.py
ADD ./Interface.py /workspace/Interface.py




