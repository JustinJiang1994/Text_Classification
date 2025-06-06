import tensorflow as tf
from transformers import TFBertModel, BertTokenizer
from .base import BaseModel

class BERTClassifier(BaseModel):
    """BERT模型实现"""
    
    def __init__(self, vocab_size: int, embedding_dim: int, num_classes: int, max_length: int,
                 model_name: str = 'bert-base-chinese', dropout_rate: float = 0.1):
        super().__init__(vocab_size, embedding_dim, num_classes, max_length)
        self.model_name = model_name
        self.dropout_rate = dropout_rate
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.bert_model = TFBertModel.from_pretrained(model_name)
    
    def build(self) -> tf.keras.Model:
        """构建BERT模型架构"""
        # 输入层
        input_ids = tf.keras.layers.Input(shape=(self.max_length,), dtype=tf.int32, name='input_ids')
        attention_mask = tf.keras.layers.Input(shape=(self.max_length,), dtype=tf.int32, name='attention_mask')
        token_type_ids = tf.keras.layers.Input(shape=(self.max_length,), dtype=tf.int32, name='token_type_ids')
        
        # BERT层
        bert_output = self.bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # 使用[CLS]标记的输出
        pooled_output = bert_output[1]
        
        # Dropout层
        dropout = tf.keras.layers.Dropout(self.dropout_rate)(pooled_output)
        
        # 全连接层
        dense = tf.keras.layers.Dense(256, activation='relu')(dropout)
        outputs = tf.keras.layers.Dense(self.num_classes, activation='softmax')(dense)
        
        return tf.keras.Model(
            inputs=[input_ids, attention_mask, token_type_ids],
            outputs=outputs
        )
    
    def preprocess_text(self, texts):
        """预处理文本为BERT输入格式"""
        return self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='tf'
        )
    
    def predict(self, texts, *args, **kwargs):
        """重写预测方法以处理文本输入"""
        inputs = self.preprocess_text(texts)
        return super().predict(inputs, *args, **kwargs)
    
    def fit(self, texts, labels, *args, **kwargs):
        """重写训练方法以处理文本输入"""
        inputs = self.preprocess_text(texts)
        return super().fit(inputs, labels, *args, **kwargs)
    
    def evaluate(self, texts, labels, *args, **kwargs):
        """重写评估方法以处理文本输入"""
        inputs = self.preprocess_text(texts)
        return super().evaluate(inputs, labels, *args, **kwargs)

def encode_text(text: str, tokenizer, max_length: int = 512):
    """
    使用BERT tokenizer对文本进行编码
    :param text: 输入文本
    :param tokenizer: BERT tokenizer
    :param max_length: 最大序列长度
    :return: 编码后的input_ids, attention_mask, token_type_ids
    """
    encoding = tokenizer(
        text,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='tf'
    )
    return (
        encoding['input_ids'],
        encoding['attention_mask'],
        encoding['token_type_ids']
    )
