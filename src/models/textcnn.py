import tensorflow as tf
from .base import BaseModel

class TextCNN(BaseModel):
    """TextCNN模型实现"""
    
    def __init__(self, vocab_size: int, embedding_dim: int, num_classes: int, max_length: int,
                 num_filters: int = 128, filter_sizes: list = [3, 4, 5], dropout_rate: float = 0.5):
        super().__init__(vocab_size, embedding_dim, num_classes, max_length)
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        self.dropout_rate = dropout_rate
    
    def build(self) -> tf.keras.Model:
        """构建TextCNN模型架构"""
        inputs = tf.keras.layers.Input(shape=(self.max_length,))
        
        # 词嵌入层
        embedding = tf.keras.layers.Embedding(
            self.vocab_size, 
            self.embedding_dim,
            input_length=self.max_length
        )(inputs)
        
        # 卷积层
        conv_outputs = []
        for filter_size in self.filter_sizes:
            conv = tf.keras.layers.Conv1D(
                filters=self.num_filters,
                kernel_size=filter_size,
                activation='relu',
                padding='same'
            )(embedding)
            pool = tf.keras.layers.GlobalMaxPooling1D()(conv)
            conv_outputs.append(pool)
        
        # 合并所有卷积输出
        concat = tf.keras.layers.Concatenate()(conv_outputs)
        
        # Dropout层
        dropout = tf.keras.layers.Dropout(self.dropout_rate)(concat)
        
        # 全连接层
        dense = tf.keras.layers.Dense(128, activation='relu')(dropout)
        outputs = tf.keras.layers.Dense(self.num_classes, activation='softmax')(dense)
        
        return tf.keras.Model(inputs=inputs, outputs=outputs)
