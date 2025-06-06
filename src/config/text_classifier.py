from typing import Dict, Any, List
from .base import BaseConfig
from ..utils.logger import get_logger

logger = get_logger(__name__)

class TextClassifierConfig(BaseConfig):
    """文本分类配置类"""
    
    def validate_config(self) -> bool:
        """验证配置是否有效
        
        Returns:
            配置是否有效
        """
        required_keys = {
            'data': ['train_file', 'val_file', 'test_file', 'max_length'],
            'model': ['model_type', 'vocab_size', 'embedding_dim', 'num_classes'],
            'training': ['batch_size', 'epochs', 'learning_rate', 'optimizer']
        }
        
        # 检查必需配置项
        for section, keys in required_keys.items():
            if section not in self.config:
                logger.error(f"缺少配置节: {section}")
                return False
            
            for key in keys:
                if key not in self.config[section]:
                    logger.error(f"缺少配置项: {section}.{key}")
                    return False
        
        # 验证模型类型
        valid_model_types = ['textcnn', 'lstm', 'bert']
        if self.config['model']['model_type'] not in valid_model_types:
            logger.error(f"无效的模型类型: {self.config['model']['model_type']}")
            return False
        
        # 验证优化器
        valid_optimizers = ['adam', 'sgd']
        if self.config['training']['optimizer'] not in valid_optimizers:
            logger.error(f"无效的优化器: {self.config['training']['optimizer']}")
            return False
        
        # 验证数值范围
        if self.config['data']['max_length'] <= 0:
            logger.error("max_length必须大于0")
            return False
        
        if self.config['model']['vocab_size'] <= 0:
            logger.error("vocab_size必须大于0")
            return False
        
        if self.config['model']['embedding_dim'] <= 0:
            logger.error("embedding_dim必须大于0")
            return False
        
        if self.config['model']['num_classes'] <= 0:
            logger.error("num_classes必须大于0")
            return False
        
        if self.config['training']['batch_size'] <= 0:
            logger.error("batch_size必须大于0")
            return False
        
        if self.config['training']['epochs'] <= 0:
            logger.error("epochs必须大于0")
            return False
        
        if self.config['training']['learning_rate'] <= 0:
            logger.error("learning_rate必须大于0")
            return False
        
        return True
    
    def get_default_config(self) -> Dict[str, Any]:
        """获取默认配置
        
        Returns:
            默认配置字典
        """
        return {
            'data': {
                'train_file': 'data/raw/cnews.train.txt',
                'val_file': 'data/raw/cnews.val.txt',
                'test_file': 'data/raw/cnews.test.txt',
                'max_length': 512,
                'shuffle_buffer_size': 10000,
                'preprocessing': {
                    'remove_urls': True,
                    'remove_emails': True,
                    'remove_numbers': False,
                    'remove_punctuation': False,
                    'remove_whitespace': True,
                    'lowercase': True
                }
            },
            'model': {
                'model_type': 'textcnn',
                'vocab_size': 50000,
                'embedding_dim': 300,
                'num_classes': 10,
                'textcnn': {
                    'num_filters': 128,
                    'filter_sizes': [2, 3, 4, 5],
                    'dropout_rate': 0.5
                },
                'lstm': {
                    'lstm_units': 128,
                    'bidirectional': True,
                    'dropout_rate': 0.5
                },
                'bert': {
                    'model_name': 'bert-base-chinese',
                    'dropout_rate': 0.1,
                    'fine_tune': True
                }
            },
            'training': {
                'batch_size': 32,
                'epochs': 10,
                'learning_rate': 1e-3,
                'optimizer': 'adam',
                'momentum': 0.9,
                'early_stopping_patience': 5,
                'reduce_lr_patience': 2,
                'reduce_lr_factor': 0.5,
                'min_lr': 1e-6,
                'metrics': ['accuracy', 'precision', 'recall', 'f1'],
                'checkpoint_dir': 'models/checkpoints',
                'log_dir': 'models/logs',
                'save_best_only': True
            },
            'augmentation': {
                'enabled': True,
                'methods': {
                    'synonym_replacement': {
                        'enabled': True,
                        'max_words': 3,
                        'prob': 0.3
                    },
                    'random_deletion': {
                        'enabled': True,
                        'prob': 0.2
                    },
                    'random_swap': {
                        'enabled': True,
                        'max_swaps': 3,
                        'prob': 0.3
                    },
                    'random_insertion': {
                        'enabled': True,
                        'max_insertions': 3,
                        'prob': 0.3
                    }
                }
            },
            'evaluation': {
                'batch_size': 32,
                'output_dir': 'evaluation',
                'save_predictions': True,
                'plot_confusion_matrix': True,
                'plot_roc_curves': True,
                'analyze_errors': True,
                'top_k_errors': 5,
                'analyze_feature_importance': False
            }
        }
    
    def get_model_config(self) -> Dict[str, Any]:
        """获取模型配置
        
        Returns:
            模型配置字典
        """
        model_type = self.config['model']['model_type']
        model_config = {
            'vocab_size': self.config['model']['vocab_size'],
            'embedding_dim': self.config['model']['embedding_dim'],
            'num_classes': self.config['model']['num_classes'],
            'max_length': self.config['data']['max_length']
        }
        
        # 添加模型特定配置
        if model_type == 'textcnn':
            model_config.update(self.config['model']['textcnn'])
        elif model_type == 'lstm':
            model_config.update(self.config['model']['lstm'])
        elif model_type == 'bert':
            model_config.update(self.config['model']['bert'])
        
        return model_config
    
    def get_training_config(self) -> Dict[str, Any]:
        """获取训练配置
        
        Returns:
            训练配置字典
        """
        return self.config['training']
    
    def get_data_config(self) -> Dict[str, Any]:
        """获取数据配置
        
        Returns:
            数据配置字典
        """
        return self.config['data']
    
    def get_augmentation_config(self) -> Dict[str, Any]:
        """获取数据增强配置
        
        Returns:
            数据增强配置字典
        """
        return self.config['augmentation']
    
    def get_evaluation_config(self) -> Dict[str, Any]:
        """获取评估配置
        
        Returns:
            评估配置字典
        """
        return self.config['evaluation'] 