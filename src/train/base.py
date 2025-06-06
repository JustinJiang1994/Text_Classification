from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import tensorflow as tf
import numpy as np
from ..models.base import BaseModel
from ..data.base import BaseDataset
from ..utils.logger import get_logger

logger = get_logger(__name__)

class BaseTrainer(ABC):
    """训练器基类"""
    
    def __init__(
        self,
        model: BaseModel,
        dataset: BaseDataset,
        config: Dict[str, Any],
        model_dir: str = "models"
    ):
        """初始化训练器
        
        Args:
            model: 模型实例
            dataset: 数据集实例
            config: 训练配置
            model_dir: 模型保存目录
        """
        self.model = model
        self.dataset = dataset
        self.config = config
        self.model_dir = model_dir
        
        # 训练相关属性
        self.optimizer = None
        self.loss_fn = None
        self.metrics = None
        self.callbacks = []
        
        # 初始化训练组件
        self._init_optimizer()
        self._init_loss()
        self._init_metrics()
        self._init_callbacks()
    
    @abstractmethod
    def _init_optimizer(self):
        """初始化优化器"""
        pass
    
    @abstractmethod
    def _init_loss(self):
        """初始化损失函数"""
        pass
    
    @abstractmethod
    def _init_metrics(self):
        """初始化评估指标"""
        pass
    
    def _init_callbacks(self):
        """初始化回调函数"""
        # 模型检查点
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=f"{self.model_dir}/checkpoints/model-{{epoch:02d}}-{{val_loss:.4f}}.h5",
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True,
            mode='min',
            verbose=1
        )
        
        # 早停
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.config.get('early_stopping_patience', 5),
            restore_best_weights=True,
            verbose=1
        )
        
        # 学习率调度器
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        )
        
        # TensorBoard
        tensorboard = tf.keras.callbacks.TensorBoard(
            log_dir=f"{self.model_dir}/logs",
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        )
        
        self.callbacks.extend([
            checkpoint,
            early_stopping,
            lr_scheduler,
            tensorboard
        ])
    
    def train(
        self,
        train_data: Tuple[np.ndarray, np.ndarray],
        val_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """训练模型
        
        Args:
            train_data: 训练数据 (X_train, y_train)
            val_data: 验证数据 (X_val, y_val)
            **kwargs: 其他训练参数
            
        Returns:
            训练历史记录
        """
        logger.info("开始训练模型...")
        
        # 准备训练数据
        X_train, y_train = train_data
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_dataset = train_dataset.shuffle(
            buffer_size=self.config.get('shuffle_buffer_size', 10000)
        ).batch(
            self.config.get('batch_size', 32)
        ).prefetch(
            tf.data.AUTOTUNE
        )
        
        # 准备验证数据
        if val_data is not None:
            X_val, y_val = val_data
            val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
            val_dataset = val_dataset.batch(
                self.config.get('batch_size', 32)
            ).prefetch(
                tf.data.AUTOTUNE
            )
        else:
            val_dataset = None
        
        # 训练模型
        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=self.config.get('epochs', 10),
            callbacks=self.callbacks,
            **kwargs
        )
        
        logger.info("模型训练完成")
        return history.history
    
    def evaluate(
        self,
        test_data: Tuple[np.ndarray, np.ndarray]
    ) -> Dict[str, float]:
        """评估模型
        
        Args:
            test_data: 测试数据 (X_test, y_test)
            
        Returns:
            评估指标
        """
        logger.info("开始评估模型...")
        
        X_test, y_test = test_data
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        test_dataset = test_dataset.batch(
            self.config.get('batch_size', 32)
        ).prefetch(
            tf.data.AUTOTUNE
        )
        
        # 评估模型
        metrics = self.model.evaluate(test_dataset, return_dict=True)
        
        # 记录评估结果
        for metric_name, metric_value in metrics.items():
            logger.info(f"{metric_name}: {metric_value:.4f}")
        
        return metrics
    
    def save_model(self, filepath: str):
        """保存模型
        
        Args:
            filepath: 保存路径
        """
        logger.info(f"保存模型到 {filepath}")
        self.model.save(filepath)
    
    def load_model(self, filepath: str):
        """加载模型
        
        Args:
            filepath: 模型文件路径
        """
        logger.info(f"从 {filepath} 加载模型")
        self.model = tf.keras.models.load_model(filepath) 