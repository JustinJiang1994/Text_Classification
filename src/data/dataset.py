import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from .preprocess import read_txt, get_vocab, get_category_id, tokenize
from typing import Tuple, List

class TextDataset:
    def __init__(self, txt_path: str, vocab_path: str, max_length: int, categories: List[str] = None):
        self.txt_path = txt_path
        self.vocab, self.vocab_dict = get_vocab(vocab_path)
        self.max_length = max_length
        self.categories = categories or ["体育", "财经", "房产", "家居", "教育", "科技", "时尚", "时政", "游戏", "娱乐"]
        self.cate_dict = get_category_id(self.categories)
        self.num_classes = len(self.categories)

    def encode_samples(self) -> Tuple[np.ndarray, np.ndarray]:
        labels, contents = read_txt(self.txt_path)
        labels_idx = [self.cate_dict[label] for label in labels]
        contents_idx = []
        for content in contents:
            # 可选：分词（如数据已分好可跳过）
            # tokens = tokenize(content)
            tokens = list(content)
            idxs = [self.vocab_dict.get(word, 5000) for word in tokens]
            contents_idx.append(idxs)
        x_pad = pad_sequences(contents_idx, self.max_length)
        y_pad = to_categorical(labels_idx, num_classes=self.num_classes)
        return x_pad, y_pad

    def encode_single(self, sentence: str) -> np.ndarray:
        # tokens = tokenize(sentence)
        tokens = list(sentence)
        idxs = [self.vocab_dict.get(word, 5000) for word in tokens]
        x_pad = pad_sequences([idxs], self.max_length)
        return x_pad
