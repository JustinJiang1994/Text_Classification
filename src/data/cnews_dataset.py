import os
import re
import jieba
import numpy as np
from typing import Tuple, List, Dict
from .base import BaseDataset

class CNewsDataset(BaseDataset):
    """中文新闻数据集处理类"""
    
    def __init__(self, data_dir: str, max_length: int = 512, min_freq: int = 2):
        super().__init__(data_dir, max_length)
        self.min_freq = min_freq
        self._build_vocab()
    
    def _build_vocab(self):
        """构建词表"""
        # 加载训练数据
        train_texts, _ = self.load_data('cnews.train.txt')
        
        # 统计词频
        word_freq = {}
        for text in train_texts:
            words = jieba.lcut(text)
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # 过滤低频词
        word_freq = {k: v for k, v in word_freq.items() if v >= self.min_freq}
        
        # 构建词表
        self.vocab = {
            '<PAD>': 0,
            '<UNK>': 1,
            **{word: idx + 2 for idx, word in enumerate(word_freq.keys())}
        }
        self.vocab_size = len(self.vocab)
    
    def load_data(self, filename: str) -> Tuple[List[str], List[int]]:
        """加载数据文件
        
        Args:
            filename: 数据文件名
            
        Returns:
            texts: 文本列表
            labels: 标签列表
        """
        texts, labels = [], []
        filepath = os.path.join(self.data_dir, filename)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                label, text = line.strip().split('\t')
                
                # 更新标签映射
                if label not in self.label_to_id:
                    label_id = len(self.label_to_id)
                    self.label_to_id[label] = label_id
                    self.id_to_label[label_id] = label
                
                texts.append(text)
                labels.append(self.label_to_id[label])
        
        return texts, np.array(labels)
    
    def preprocess_text(self, text: str) -> str:
        """文本预处理
        
        Args:
            text: 原始文本
            
        Returns:
            处理后的文本
        """
        # 去除特殊字符和数字
        text = re.sub(r'[^\u4e00-\u9fa5]', '', text)
        # 分词
        words = jieba.lcut(text)
        # 截断或填充
        if len(words) > self.max_length:
            words = words[:self.max_length]
        else:
            words.extend(['<PAD>'] * (self.max_length - len(words)))
        return ' '.join(words)
    
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """将文本转换为模型输入格式
        
        Args:
            texts: 文本列表
            
        Returns:
            编码后的文本数组
        """
        encoded_texts = []
        for text in texts:
            # 预处理文本
            processed_text = self.preprocess_text(text)
            # 转换为词ID
            word_ids = [
                self.vocab.get(word, self.vocab['<UNK>'])
                for word in processed_text.split()
            ]
            encoded_texts.append(word_ids)
        return np.array(encoded_texts)
    
    def get_class_weights(self) -> Dict[int, float]:
        """计算类别权重，用于处理类别不平衡
        
        Returns:
            类别权重字典
        """
        _, labels = self.load_data('cnews.train.txt')
        class_counts = np.bincount(labels)
        total_samples = len(labels)
        class_weights = {
            i: total_samples / (len(class_counts) * count)
            for i, count in enumerate(class_counts)
        }
        return class_weights 