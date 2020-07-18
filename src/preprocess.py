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
        vocabs = list(set([word for word in infile]))
        vocabs_dict = dict(zip(vocabs, range(len(vocabs))))
        return vocabs, vocabs_dict

    def get_category_id(self):
        """
        返回分类种类的索引
        :return: 返回分类种类的字典
        """
        categories = ["体育","财经","房产","家居","教育","科技","时尚","时政","游戏","娱乐"]
        cates_dict = dict(zip(categories, range(len(categories))))
        return cates_dict

    def word2idx(self, txt_path, max_length):
        """
        将语料中各文本转换成固定max_length后返回各文本的标签与文本tokens
        :param txt_path: 语料路径
        :param max_length: pad后的长度
        :return: 语料pad后表示与标签
        """
        # _:分词词汇表，这边没用上
        # vocabs_dict:各分词的索引
        _, vocabs_dict = self.get_vocab_id()
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
            # 将该idx(样本)的标签加入至labels_idx中
            labels_idx.append(cates_dict[labels[idx]])
            # contents[idx]:为该语料中的样本遍历项
            # 遍历contents中各词并将其转换为索引后加入contents_idx中
            contents_idx.append([vocabs_dict[word] for word in contents[idx] if word in vocabs_dict])

        # 将各样本长度pad至max_length
        x_pad = keras.preprocessing.sequence.pad_sequences(contents_idx, max_length)
        y_pad = keras.utils.to_categorical(labels_idx, num_classes=len(cates_dict))

        return x_pad, y_pad

    def batch_iter(self, x, y, batch_size):
        num_batch = int((len(x)-1)/batch_size) + 1

        rand_idx = np.random.permutation(np.arange(len(x)))
        x_shuffle = x [rand_idx]
        y_shuffle = y[rand_idx]

        for idx in range(num_batch):
            start_idx = idx * batch_size
            end_idx = min((idx+1) * batch_size, len(x))

            yield x_shuffle[start_idx:end_idx], y_shuffle[start_idx:end_idx]

if __name__ == '__main__':
    test = preprocesser()
    trainingSet_path = test.config.get("data_path", "trainingSet_path")
    print(test.word2idx(trainingSet_path, 150))