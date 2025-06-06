import unittest
import numpy as np
import tensorflow as tf
from src.models.textcnn import build_textcnn
from src.models.lstm import build_lstm
from src.models.bert import build_bert

class TestModels(unittest.TestCase):
    def setUp(self):
        self.vocab_size = 5000
        self.seq_length = 100
        self.num_classes = 10
        self.batch_size = 32

    def test_textcnn(self):
        model = build_textcnn(
            vocab_size=self.vocab_size,
            seq_length=self.seq_length,
            num_classes=self.num_classes
        )
        
        # 测试模型输出形状
        x = np.random.randint(0, self.vocab_size, (self.batch_size, self.seq_length))
        y = model.predict(x)
        self.assertEqual(y.shape, (self.batch_size, self.num_classes))
        
        # 测试模型编译
        self.assertTrue(model.optimizer is not None)
        self.assertTrue(model.loss is not None)

    def test_lstm(self):
        model = build_lstm(
            vocab_size=self.vocab_size,
            seq_length=self.seq_length,
            num_classes=self.num_classes
        )
        
        # 测试模型输出形状
        x = np.random.randint(0, self.vocab_size, (self.batch_size, self.seq_length))
        y = model.predict(x)
        self.assertEqual(y.shape, (self.batch_size, self.num_classes))
        
        # 测试模型编译
        self.assertTrue(model.optimizer is not None)
        self.assertTrue(model.loss is not None)

    def test_bert(self):
        model, tokenizer = build_bert(
            num_classes=self.num_classes,
            max_length=self.seq_length
        )
        
        # 测试模型输入
        text = "这是一个测试句子"
        input_ids, attention_mask, token_type_ids = tokenizer(
            text,
            max_length=self.seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='tf'
        )
        
        # 测试模型输出形状
        y = model.predict([input_ids, attention_mask, token_type_ids])
        self.assertEqual(y.shape, (1, self.num_classes))
        
        # 测试模型编译
        self.assertTrue(model.optimizer is not None)
        self.assertTrue(model.loss is not None)

if __name__ == '__main__':
    unittest.main() 