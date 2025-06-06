# -*- coding: utf-8 -*-
"""
文本预处理工具函数
"""
import jieba
from pathlib import Path
from typing import List, Tuple, Dict

def read_txt(txt_path: str) -> Tuple[List[str], List[str]]:
    """
    读取文档数据
    :param txt_path: 文档路径
    :return: 标签数组与文本数组
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

def get_vocab(vocab_path: str) -> Tuple[List[str], Dict[str, int]]:
    """
    读取词汇表
    :param vocab_path: 词汇表路径
    :return: 词汇表list和词到索引的dict
    """
    with open(vocab_path, "r", encoding="utf-8") as f:
        infile = f.readlines()
    vocabs = [word.strip() for word in infile]
    vocabs_dict = {word: idx for idx, word in enumerate(vocabs)}
    return vocabs, vocabs_dict

def get_category_id(categories: List[str] = None) -> Dict[str, int]:
    """
    返回分类种类的索引
    :param categories: 类别列表
    :return: 分类到索引的dict
    """
    if categories is None:
        categories = ["体育", "财经", "房产", "家居", "教育", "科技", "时尚", "时政", "游戏", "娱乐"]
    return {cat: idx for idx, cat in enumerate(categories)}

def tokenize(text: str) -> List[str]:
    """
    使用jieba分词
    :param text: 输入文本
    :return: 分词结果
    """
    return list(jieba.cut(text))
