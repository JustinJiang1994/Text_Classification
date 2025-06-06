import unittest
import numpy as np
from src.data.dataset import TextDataset
from src.data.augmentation import TextAugmenter

class TestDataProcessing(unittest.TestCase):
    def setUp(self):
        # 创建测试数据
        self.test_texts = [
            "这是一个测试句子",
            "另一个测试句子",
            "第三个测试句子"
        ]
        self.test_labels = [0, 1, 2]
        self.vocab = ["这", "是", "一个", "测试", "句子", "另", "第三"]
        self.vocab_dict = {word: idx for idx, word in enumerate(self.vocab)}
        
        # 创建测试数据集
        self.dataset = TextDataset(
            txt_path="test.txt",
            vocab_path="test_vocab.txt",
            max_length=10,
            categories=["类别1", "类别2", "类别3"]
        )
        self.dataset.vocab = self.vocab
        self.dataset.vocab_dict = self.vocab_dict

    def test_text_encoding(self):
        # 测试文本编码
        encoded = self.dataset.encode_single(self.test_texts[0])
        self.assertEqual(encoded.shape, (1, 10))  # 检查padding后的长度
        
        # 测试批量编码
        x, y = self.dataset.encode_samples()
        self.assertEqual(len(x), len(self.test_texts))
        self.assertEqual(len(y), len(self.test_labels))

    def test_data_augmentation(self):
        augmenter = TextAugmenter(prob=0.3)
        
        # 测试同义词替换
        aug_text = augmenter.synonym_replacement(self.test_texts[0])
        self.assertIsInstance(aug_text, str)
        self.assertTrue(len(aug_text) > 0)
        
        # 测试回译
        aug_text = augmenter.back_translation(self.test_texts[0])
        self.assertIsInstance(aug_text, str)
        self.assertTrue(len(aug_text) > 0)
        
        # 测试批量增强
        aug_texts, aug_labels = augmenter.augment_batch(
            self.test_texts,
            self.test_labels,
            methods=['synonym'],
            n_augment=1
        )
        self.assertEqual(len(aug_texts), len(self.test_texts) * 2)
        self.assertEqual(len(aug_labels), len(self.test_labels) * 2)

if __name__ == '__main__':
    unittest.main() 