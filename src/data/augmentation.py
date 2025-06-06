import random
import jieba
import synonyms
from typing import List, Tuple
import numpy as np
from googletrans import Translator
from tqdm import tqdm

class TextAugmenter:
    def __init__(self, prob: float = 0.3):
        """
        文本增强器
        :param prob: 每个词被替换的概率
        """
        self.prob = prob
        self.translator = Translator()

    def synonym_replacement(self, text: str, n: int = 1) -> str:
        """
        同义词替换
        :param text: 输入文本
        :param n: 替换词数量
        :return: 增强后的文本
        """
        words = list(jieba.cut(text))
        n = min(n, len(words))
        new_words = words.copy()
        random_word_list = list(set([word for word in words if len(word) > 1]))
        random.shuffle(random_word_list)
        num_replaced = 0
        
        for random_word in random_word_list:
            synonyms_list = synonyms.nearby(random_word)[0]
            if len(synonyms_list) > 1:
                synonym = random.choice(synonyms_list[1:])
                for idx, word in enumerate(new_words):
                    if word == random_word and random.random() < self.prob:
                        new_words[idx] = synonym
                        num_replaced += 1
                        break
            if num_replaced >= n:
                break
                
        return ''.join(new_words)

    def back_translation(self, text: str, target_lang: str = 'en') -> str:
        """
        回译增强
        :param text: 输入文本
        :param target_lang: 目标语言
        :return: 增强后的文本
        """
        try:
            # 翻译成目标语言
            translated = self.translator.translate(text, dest=target_lang)
            # 翻译回中文
            back_translated = self.translator.translate(translated.text, dest='zh-cn')
            return back_translated.text
        except Exception as e:
            print(f"回译失败: {e}")
            return text

    def augment_batch(self, texts: List[str], labels: List[int], 
                     methods: List[str] = ['synonym', 'back_translation'],
                     n_augment: int = 1) -> Tuple[List[str], List[int]]:
        """
        批量数据增强
        :param texts: 文本列表
        :param labels: 标签列表
        :param methods: 增强方法列表
        :param n_augment: 每个样本增强次数
        :return: 增强后的文本和标签
        """
        augmented_texts = []
        augmented_labels = []
        
        for text, label in tqdm(zip(texts, labels), total=len(texts), desc="数据增强"):
            augmented_texts.append(text)
            augmented_labels.append(label)
            
            for _ in range(n_augment):
                method = random.choice(methods)
                if method == 'synonym':
                    aug_text = self.synonym_replacement(text)
                elif method == 'back_translation':
                    aug_text = self.back_translation(text)
                else:
                    continue
                    
                augmented_texts.append(aug_text)
                augmented_labels.append(label)
                
        return augmented_texts, augmented_labels

def get_augmented_dataset(texts: List[str], labels: List[int], 
                         augment_ratio: float = 0.3) -> Tuple[List[str], List[int]]:
    """
    获取增强后的数据集
    :param texts: 原始文本列表
    :param labels: 原始标签列表
    :param augment_ratio: 增强比例
    :return: 增强后的文本和标签
    """
    augmenter = TextAugmenter()
    n_augment = int(len(texts) * augment_ratio)
    
    # 随机选择样本进行增强
    indices = np.random.choice(len(texts), n_augment, replace=False)
    texts_to_augment = [texts[i] for i in indices]
    labels_to_augment = [labels[i] for i in indices]
    
    # 进行数据增强
    aug_texts, aug_labels = augmenter.augment_batch(
        texts_to_augment, 
        labels_to_augment,
        methods=['synonym', 'back_translation'],
        n_augment=1
    )
    
    # 合并原始数据和增强数据
    all_texts = texts + aug_texts
    all_labels = labels + aug_labels
    
    return all_texts, all_labels 