import tensorflow as tf
from .base import BaseModel

class LSTM(BaseModel):
    """LSTM模型实现"""
    
    def __init__(self, vocab_size: int, embedding_dim: int, num_classes: int, max_length: int,
                 lstm_units: int = 128, dropout_rate: float = 0.5, bidirectional: bool = True):
        super().__init__(vocab_size, embedding_dim, num_classes, max_length)
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.bidirectional = bidirectional
    
    def build(self) -> tf.keras.Model:
        """构建LSTM模型架构"""
        inputs = tf.keras.layers.Input(shape=(self.max_length,))
        
        # 词嵌入层
        embedding = tf.keras.layers.Embedding(
            self.vocab_size,
            self.embedding_dim,
            input_length=self.max_length
        )(inputs)
        
        # LSTM层
        if self.bidirectional:
            lstm = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(self.lstm_units, return_sequences=True)
            )(embedding)
            lstm = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(self.lstm_units)
            )(lstm)
        else:
            lstm = tf.keras.layers.LSTM(self.lstm_units, return_sequences=True)(embedding)
            lstm = tf.keras.layers.LSTM(self.lstm_units)(lstm)
        
        # Dropout层
        dropout = tf.keras.layers.Dropout(self.dropout_rate)(lstm)
        
        # 全连接层
        dense = tf.keras.layers.Dense(128, activation='relu')(dropout)
        outputs = tf.keras.layers.Dense(self.num_classes, activation='softmax')(dense)
        
        return tf.keras.Model(inputs=inputs, outputs=outputs)
