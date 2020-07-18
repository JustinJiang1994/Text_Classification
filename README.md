# 中文文本分类

基于tensorflow2.0中的keras进行中文的文本分类

## 环境

Tensorflow.keras  

Python3  



## 数据集

使用THUCNews进行训练与测试（由于数据集太大，无法上传到Github，可自行下载）  

数据集划分如下：  
训练集cnews.train.txt 50000条  
验证集cnews.val.txt 5000条  
测试集cnews.test.txt 10000条  
共分为10个类别："体育","财经","房产","家居","教育","科技","时尚","时政","游戏","娱乐"。
cnews.vocab.txt为词汇表，字符级，大小为5000。



## 文件说明

将数据集中的四份数据存放至data资料夹中  

src中的文件为代码存放资料夹：

preprocess.py:加载数据与预处理相关函数。

model.py:模型结构主体。

config.py:配置文件，包含路径配置与各模型的相关参数。

train.py:主函数文件