from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from ..models.base import BaseModel
from ..data.base import BaseDataset
from ..utils.logger import get_logger

logger = get_logger(__name__)

class BaseEvaluator(ABC):
    """评估器基类"""
    
    def __init__(
        self,
        model: BaseModel,
        dataset: BaseDataset,
        config: Dict[str, Any],
        output_dir: str = "evaluation"
    ):
        """初始化评估器
        
        Args:
            model: 模型实例
            dataset: 数据集实例
            config: 评估配置
            output_dir: 评估结果输出目录
        """
        self.model = model
        self.dataset = dataset
        self.config = config
        self.output_dir = output_dir
        
        # 获取标签映射
        self.label_to_id, self.id_to_label = dataset.get_label_mapping()
        self.num_classes = len(self.label_to_id)
    
    @abstractmethod
    def evaluate(
        self,
        test_data: Tuple[np.ndarray, np.ndarray],
        **kwargs
    ) -> Dict[str, float]:
        """评估模型性能
        
        Args:
            test_data: 测试数据 (X_test, y_test)
            **kwargs: 其他评估参数
            
        Returns:
            评估指标
        """
        pass
    
    def predict(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> Tuple[np.ndarray, np.ndarray]:
        """预测文本类别
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小
            
        Returns:
            predictions: 预测的类别ID
            probabilities: 预测的概率分布
        """
        # 文本预处理和编码
        encoded_texts = self.dataset.encode_texts(texts)
        
        # 批量预测
        dataset = tf.data.Dataset.from_tensor_slices(encoded_texts)
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        # 获取预测结果
        probabilities = self.model.predict(dataset)
        predictions = np.argmax(probabilities, axis=1)
        
        return predictions, probabilities
    
    def get_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        output_file: Optional[str] = None
    ) -> str:
        """生成分类报告
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            output_file: 输出文件路径
            
        Returns:
            分类报告文本
        """
        # 将标签ID转换为标签名称
        y_true_names = [self.id_to_label[y] for y in y_true]
        y_pred_names = [self.id_to_label[y] for y in y_pred]
        
        # 生成分类报告
        report = classification_report(
            y_true_names,
            y_pred_names,
            target_names=list(self.label_to_id.keys()),
            digits=4
        )
        
        # 保存报告
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"分类报告已保存到: {output_file}")
        
        return report
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        output_file: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8)
    ):
        """绘制混淆矩阵
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            output_file: 输出文件路径
            figsize: 图像大小
        """
        # 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        
        # 绘制混淆矩阵
        plt.figure(figsize=figsize)
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=list(self.label_to_id.keys()),
            yticklabels=list(self.label_to_id.keys())
        )
        plt.title('混淆矩阵')
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # 保存图像
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"混淆矩阵已保存到: {output_file}")
        
        plt.close()
    
    def plot_learning_curves(
        self,
        history: Dict[str, List[float]],
        output_file: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 6)
    ):
        """绘制学习曲线
        
        Args:
            history: 训练历史记录
            output_file: 输出文件路径
            figsize: 图像大小
        """
        plt.figure(figsize=figsize)
        
        # 绘制训练和验证损失
        plt.subplot(1, 2, 1)
        plt.plot(history['loss'], label='训练损失')
        plt.plot(history['val_loss'], label='验证损失')
        plt.title('模型损失')
        plt.xlabel('轮次')
        plt.ylabel('损失')
        plt.legend()
        
        # 绘制训练和验证准确率
        plt.subplot(1, 2, 2)
        plt.plot(history['accuracy'], label='训练准确率')
        plt.plot(history['val_accuracy'], label='验证准确率')
        plt.title('模型准确率')
        plt.xlabel('轮次')
        plt.ylabel('准确率')
        plt.legend()
        
        plt.tight_layout()
        
        # 保存图像
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"学习曲线已保存到: {output_file}")
        
        plt.close()
    
    def analyze_errors(
        self,
        texts: List[str],
        y_true: np.ndarray,
        y_pred: np.ndarray,
        top_k: int = 5,
        output_file: Optional[str] = None
    ):
        """分析预测错误的样本
        
        Args:
            texts: 文本列表
            y_true: 真实标签
            y_pred: 预测标签
            top_k: 每个类别展示的错误样本数
            output_file: 输出文件路径
        """
        # 找出预测错误的样本
        errors = []
        for text, true_label, pred_label in zip(texts, y_true, y_pred):
            if true_label != pred_label:
                errors.append({
                    'text': text,
                    'true_label': self.id_to_label[true_label],
                    'pred_label': self.id_to_label[pred_label]
                })
        
        # 按真实标签分组
        error_groups = {}
        for error in errors:
            true_label = error['true_label']
            if true_label not in error_groups:
                error_groups[true_label] = []
            error_groups[true_label].append(error)
        
        # 生成错误分析报告
        report = []
        report.append("预测错误分析报告")
        report.append("=" * 50)
        
        for true_label, errors in error_groups.items():
            report.append(f"\n真实标签: {true_label}")
            report.append("-" * 30)
            
            # 统计预测错误的分布
            pred_dist = {}
            for error in errors:
                pred_label = error['pred_label']
                pred_dist[pred_label] = pred_dist.get(pred_label, 0) + 1
            
            # 输出预测分布
            report.append("预测分布:")
            for pred_label, count in sorted(pred_dist.items(), key=lambda x: x[1], reverse=True):
                report.append(f"  - 预测为 {pred_label}: {count} 个样本")
            
            # 输出错误样本
            report.append("\n错误样本示例:")
            for error in errors[:top_k]:
                report.append(f"  - 文本: {error['text'][:100]}...")
                report.append(f"    预测为: {error['pred_label']}")
                report.append()
        
        # 保存报告
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report))
            logger.info(f"错误分析报告已保存到: {output_file}")
        
        return '\n'.join(report) 