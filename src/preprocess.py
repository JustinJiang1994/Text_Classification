# -*- coding: utf-8 -*-

"""
Created on 2020-07-18 10:50
@Author  : Justin Jiang
@Email   : jw_jiang@pku.edu.com

数据加载
"""

import numpy as np
import tensorflow.keras as keras
from config import Config
import jieba

class preprocesser(object):

    def __init__(self):
        self.config = Config()

    def read_txt(self, txt_path):
        """
        读取文档数据
        :param txt_path:文档路径
        :return: 该文档中的标签数组与文本数组
        """
        with open(txt_path, "r", encoding='utf-8') as f:
            data = f.readlines()
        labels = []
        contents = []
        for line in data:
            label, content = line.strip().split('\t')
            labels.append(label)
            contents.append(content)
        return labels, contents

    def get_vocab_id(self):
        """
        读取分词文档
        :return:分词数组与各分词索引的字典
        """
        vocab_path = self.config.get("data_path", "vocab_path")
        with open(vocab_path, "r", encoding="utf-8") as f:
            infile = f.readlines()
        vocabs = list([word.replace("\n", "") for word in infile])
        vocabs_dict = dict(zip(vocabs, range(len(vocabs))))
        return vocabs, vocabs_dict

    def get_category_id(self):
        """
        返回分类种类的索引
        :return: 返回分类种类的字典
        """
        categories = ["体育", "财经", "房产", "家居", "教育", "科技", "时尚", "时政", "游戏", "娱乐"]
        cates_dict = dict(zip(categories, range(len(categories))))
        return cates_dict

    def word2idx(self, txt_path, max_length):
        """
        将语料中各文本转换成固定max_length后返回各文本的标签与文本tokens
        :param txt_path: 语料路径
        :param max_length: pad后的长度
        :return: 语料pad后表示与标签
        """
        # vocabs:分词词汇表
        # vocabs_dict:各分词的索引
        vocabs, vocabs_dict = self.get_vocab_id()
        # cates_dict:各分类的索引
        cates_dict = self.get_category_id()

        # 读取语料
        labels, contents = self.read_txt(txt_path)
        # labels_idx：用来存放语料中的分类
        labels_idx = []
        # contents_idx:用来存放语料中各样本的索引
        contents_idx = []

        # 遍历语料
        for idx in range(len(contents)):
            # tmp:存放当前语句index
            tmp = []
            # 将该idx(样本)的标签加入至labels_idx中
            labels_idx.append(cates_dict[labels[idx]])
            # contents[idx]:为该语料中的样本遍历项
            # 遍历contents中各词并将其转换为索引后加入contents_idx中
            for word in contents[idx]:
                if word in vocabs:
                    tmp.append(vocabs_dict[word])
                else:
                    # 第5000位设置为未知字符
                    tmp.append(5000)
            # 将该样本index后结果存入contents_idx作为结果等待传回
            contents_idx.append(tmp)


        # 将各样本长度pad至max_length
        x_pad = keras.preprocessing.sequence.pad_sequences(contents_idx, max_length)
        y_pad = keras.utils.to_categorical(labels_idx, num_classes=len(cates_dict))

        return x_pad, y_pad


    def word2idx_for_sample(self, sentence, max_length):
        # vocabs:分词词汇表
        # vocabs_dict:各分词的索引
        vocabs, vocabs_dict = self.get_vocab_id()
        result = []
        # 遍历语料
        for word in sentence:
            # tmp:存放当前语句index
                if word in vocabs:
                    result.append(vocabs_dict[word])
                else:
                    # 第5000位设置为未知字符，实际中为vocabs_dict[5000]，使得vocabs_dict长度变成len(vocabs_dict+1)
                    result.append(5000)

        x_pad = keras.preprocessing.sequence.pad_sequences([result], max_length)
        return x_pad


if __name__ == '__main__':
    test = preprocesser()
    # tmp_path = '../data/training_sample.txt'
    #
    # test.word2idx(tmp_path, 600)
    # x_pad, y_pad = test.word2idx(tmp_path, 600)
    #
    # print(len(x_pad[0]))
    # print(x_pad[0])
    # print(len(x_pad[1]))
    # print(x_pad[1])
    #
    # print(y_pad)
    print(test.word2idx_for_sample("马晓旭意外受伤让国奥警惕 无奈大雨格外青睐殷家军记者傅亚雨沈阳报道 来到沈阳，国奥队依然没有摆脱雨水的困扰。7月31日下午6点，国奥队的日常训练再度受到大雨的干扰，无奈之下队员们只慢跑了25分钟就草草收场。31日上午10点，国奥队在奥体中心外场训练的时候，天就是阴沉沉的，气象预报显示当天下午沈阳就有大雨，但幸好队伍上午的训练并没有受到任何干扰。下午6点，当球队抵达训练场时，大雨已经下了几个小时，而且丝毫没有停下来的意思。抱着试一试的态度，球队开始了当天下午的例行训练，25分钟过去了，天气没有任何转好的迹象，为了保护球员们，国奥队决定中止当天的训练，全队立即返回酒店。在雨中训练对足球队来说并不是什么稀罕事，但在奥运会即将开始之前，全队变得“娇贵”了。在沈阳最后一周的训练，国奥队首先要保证现有的球员不再出现意外的伤病情况以免影响正式比赛，因此这一阶段控制训练受伤、控制感冒等疾病的出现被队伍放在了相当重要的位置。而抵达沈阳之后，中后卫冯萧霆就一直没有训练，冯萧霆是7月27日在长春患上了感冒，因此也没有参加29日跟塞尔维亚的热身赛。队伍介绍说，冯萧霆并没有出现发烧症状，但为了安全起见，这两天还是让他静养休息，等感冒彻底好了之后再恢复训练。由于有了冯萧霆这个例子，因此国奥队对雨中训练就显得特别谨慎，主要是担心球员们受凉而引发感冒，造成非战斗减员。而女足队员马晓旭在热身赛中受伤导致无缘奥运的前科，也让在沈阳的国奥队现在格外警惕，“训练中不断嘱咐队员们要注意动作，我们可不能再出这样的事情了。”一位工作人员表示。从长春到沈阳，雨水一路伴随着国奥队，“也邪了，我们走到哪儿雨就下到哪儿，在长春几次训练都被大雨给搅和了，没想到来沈阳又碰到这种事情。”一位国奥球员也对雨水的“青睐”有些不解。", 600))