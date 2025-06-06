import tensorflow as tf
from typing import Dict, Any, List
from .base import BaseTrainer
from ..utils.logger import get_logger
import numpy as np

logger = get_logger(__name__)

class TextClassifierTrainer(BaseTrainer):
    """文本分类训练器"""
    
    def _init_optimizer(self):
        """初始化优化器"""
        learning_rate = self.config.get('learning_rate', 1e-3)
        optimizer_name = self.config.get('optimizer', 'adam')
        
        if optimizer_name == 'adam':
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_name == 'sgd':
            momentum = self.config.get('momentum', 0.9)
            self.optimizer = tf.keras.optimizers.SGD(
                learning_rate=learning_rate,
                momentum=momentum
            )
        else:
            raise ValueError(f"不支持的优化器: {optimizer_name}")
        
        logger.info(f"使用优化器: {optimizer_name}, 学习率: {learning_rate}")
    
    def _init_loss(self):
        """初始化损失函数"""
        loss_name = self.config.get('loss', 'categorical_crossentropy')
        
        if loss_name == 'categorical_crossentropy':
            self.loss_fn = tf.keras.losses.CategoricalCrossentropy(
                from_logits=self.config.get('from_logits', False)
            )
        elif loss_name == 'sparse_categorical_crossentropy':
            self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=self.config.get('from_logits', False)
            )
        else:
            raise ValueError(f"不支持的损失函数: {loss_name}")
        
        logger.info(f"使用损失函数: {loss_name}")
    
    def _init_metrics(self):
        """初始化评估指标"""
        metrics: List[str] = self.config.get('metrics', ['accuracy'])
        self.metrics = []
        
        for metric_name in metrics:
            if metric_name == 'accuracy':
                self.metrics.append(tf.keras.metrics.CategoricalAccuracy())
            elif metric_name == 'precision':
                self.metrics.append(tf.keras.metrics.Precision())
            elif metric_name == 'recall':
                self.metrics.append(tf.keras.metrics.Recall())
            elif metric_name == 'f1':
                self.metrics.append(tf.keras.metrics.F1Score())
            else:
                raise ValueError(f"不支持的评估指标: {metric_name}")
        
        logger.info(f"使用评估指标: {metrics}")
    
    def train_with_augmentation(
        self,
        train_data: tuple,
        val_data: tuple = None,
        augmentation_config: Dict[str, Any] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """使用数据增强进行训练
        
        Args:
            train_data: 训练数据 (X_train, y_train)
            val_data: 验证数据 (X_val, y_val)
            augmentation_config: 数据增强配置
            **kwargs: 其他训练参数
            
        Returns:
            训练历史记录
        """
        from ..data.augmentation import TextAugmenter
        
        if augmentation_config is None:
            augmentation_config = {}
        
        # 创建数据增强器
        augmenter = TextAugmenter(**augmentation_config)
        
        # 获取原始训练数据
        X_train, y_train = train_data
        
        # 应用数据增强
        logger.info("开始数据增强...")
        augmented_texts = []
        augmented_labels = []
        
        for text, label in zip(X_train, y_train):
            # 对每个样本进行增强
            augmented = augmenter.augment(text)
            augmented_texts.extend(augmented)
            augmented_labels.extend([label] * len(augmented))
        
        # 将增强后的数据转换为模型输入格式
        X_train_aug = self.dataset.encode_texts(augmented_texts)
        y_train_aug = np.array(augmented_labels)
        
        # 合并原始数据和增强数据
        X_train_combined = np.concatenate([X_train, X_train_aug])
        y_train_combined = np.concatenate([y_train, y_train_aug])
        
        logger.info(f"数据增强完成，原始样本数: {len(X_train)}, 增强后样本数: {len(X_train_combined)}")
        
        # 使用增强后的数据进行训练
        return self.train(
            train_data=(X_train_combined, y_train_combined),
            val_data=val_data,
            **kwargs
        ) 