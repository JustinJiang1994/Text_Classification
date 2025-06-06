from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Any
import numpy as np

class BaseDataset(ABC):
    """数据集处理的基础类"""
    
    def __init__(self, data_dir: str, max_length: int = 512):
        self.data_dir = data_dir
        self.max_length = max_length
        self.label_to_id: Dict[str, int] = {}
        self.id_to_label: Dict[int, str] = {}
        self.vocab: Dict[str, int] = {}
        self.vocab_size: int = 0
    
    @abstractmethod
    def load_data(self, filename: str) -> Tuple[List[str], List[int]]:
        """加载数据文件
        
        Args:
            filename: 数据文件名
            
        Returns:
            texts: 文本列表
            labels: 标签列表
        """
        pass
    
    @abstractmethod
    def preprocess_text(self, text: str) -> str:
        """文本预处理
        
        Args:
            text: 原始文本
            
        Returns:
            处理后的文本
        """
        pass
    
    @abstractmethod
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """将文本转换为模型输入格式
        
        Args:
            texts: 文本列表
            
        Returns:
            编码后的文本数组
        """
        pass
    
    def get_label_mapping(self) -> Tuple[Dict[str, int], Dict[int, str]]:
        """获取标签映射
        
        Returns:
            label_to_id: 标签到ID的映射
            id_to_label: ID到标签的映射
        """
        return self.label_to_id, self.id_to_label
    
    def get_vocab_info(self) -> Tuple[Dict[str, int], int]:
        """获取词表信息
        
        Returns:
            vocab: 词表字典
            vocab_size: 词表大小
        """
        return self.vocab, self.vocab_size
    
    def save_vocab(self, filepath: str):
        """保存词表
        
        Args:
            filepath: 保存路径
        """
        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
    
    def load_vocab(self, filepath: str):
        """加载词表
        
        Args:
            filepath: 词表文件路径
        """
        import json
        with open(filepath, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)
        self.vocab_size = len(self.vocab)
    
    def save_label_mapping(self, filepath: str):
        """保存标签映射
        
        Args:
            filepath: 保存路径
        """
        import json
        mapping = {
            'label_to_id': self.label_to_id,
            'id_to_label': {str(k): v for k, v in self.id_to_label.items()}
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2)
    
    def load_label_mapping(self, filepath: str):
        """加载标签映射
        
        Args:
            filepath: 映射文件路径
        """
        import json
        with open(filepath, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
        self.label_to_id = mapping['label_to_id']
        self.id_to_label = {int(k): v for k, v in mapping['id_to_label'].items()} 