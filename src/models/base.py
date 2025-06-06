from abc import ABC, abstractmethod
import tensorflow as tf

class BaseModel(ABC):
    """文本分类模型的基础类"""
    
    def __init__(self, vocab_size: int, embedding_dim: int, num_classes: int, max_length: int):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.max_length = max_length
        self.model = None
        
    @abstractmethod
    def build(self) -> tf.keras.Model:
        """构建模型架构"""
        pass
    
    def compile(self, optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']):
        """编译模型"""
        if self.model is None:
            self.model = self.build()
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    def fit(self, *args, **kwargs):
        """训练模型"""
        if self.model is None:
            raise ValueError("Model must be compiled before training")
        return self.model.fit(*args, **kwargs)
    
    def evaluate(self, *args, **kwargs):
        """评估模型"""
        if self.model is None:
            raise ValueError("Model must be compiled before evaluation")
        return self.model.evaluate(*args, **kwargs)
    
    def predict(self, *args, **kwargs):
        """模型预测"""
        if self.model is None:
            raise ValueError("Model must be compiled before prediction")
        return self.model.predict(*args, **kwargs)
    
    def save(self, filepath: str):
        """保存模型"""
        if self.model is None:
            raise ValueError("No model to save")
        self.model.save(filepath)
    
    @classmethod
    def load(cls, filepath: str):
        """加载模型"""
        model = tf.keras.models.load_model(filepath)
        return model
