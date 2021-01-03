# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
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
        # infile = pd.read_csv("vocab_path")
        vocabs = list([word.replace("\n", "") for word in infile])
        vocabs_dict = dict(zip(vocabs, range(len(vocabs))))
        return vocabs, vocabs_dict

    def get_category_id(self):
        """
        返回分类种类的索引
        :return: 返回分类种类的字典
        """
        categories = ["1", "0"]
        # categories = ["体育", "财经", "房产", "家居", "教育", "科技", "时尚", "时政", "游戏", "娱乐"]
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
    print(test.word2idx_for_sample(
        "#完全娱乐#好久没分享完娱的视频了，这次是完娱篮球课http://t.cn/zRQVhB8 秋秋为了一雪前耻要求开展篮球比赛[给劲]，看得出他真的很拼啦，最后投球的姿势帅呆了[花心]不过中间那个“我愿意”是怎么回事[挖鼻屎]某人会吃醋哦~",
        600))
