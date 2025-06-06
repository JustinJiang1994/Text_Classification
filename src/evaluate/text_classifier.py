from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from .base import BaseEvaluator
from ..utils.logger import get_logger

logger = get_logger(__name__)

class TextClassifierEvaluator(BaseEvaluator):
    """文本分类评估器"""
    
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
        X_test, y_test = test_data
        
        # 准备测试数据
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        test_dataset = test_dataset.batch(
            self.config.get('batch_size', 32)
        ).prefetch(
            tf.data.AUTOTUNE
        )
        
        # 评估模型
        logger.info("开始评估模型...")
        metrics = self.model.evaluate(test_dataset, return_dict=True)
        
        # 获取预测结果
        y_pred, y_prob = self.predict(X_test)
        
        # 生成评估报告
        report = self.get_classification_report(
            y_test,
            y_pred,
            output_file=f"{self.output_dir}/classification_report.txt"
        )
        logger.info("\n分类报告:\n" + report)
        
        # 绘制混淆矩阵
        self.plot_confusion_matrix(
            y_test,
            y_pred,
            output_file=f"{self.output_dir}/confusion_matrix.png"
        )
        
        # 绘制ROC曲线
        self.plot_roc_curves(
            y_test,
            y_prob,
            output_file=f"{self.output_dir}/roc_curves.png"
        )
        
        # 分析错误样本
        if 'texts' in kwargs:
            self.analyze_errors(
                kwargs['texts'],
                y_test,
                y_pred,
                output_file=f"{self.output_dir}/error_analysis.txt"
            )
        
        return metrics
    
    def plot_roc_curves(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        output_file: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8)
    ):
        """绘制ROC曲线
        
        Args:
            y_true: 真实标签（one-hot编码）
            y_prob: 预测概率
            output_file: 输出文件路径
            figsize: 图像大小
        """
        plt.figure(figsize=figsize)
        
        # 计算每个类别的ROC曲线
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(self.num_classes):
            # 将真实标签转换为二分类形式
            y_true_binary = (y_true == i).astype(int)
            
            # 计算ROC曲线
            fpr[i], tpr[i], _ = roc_curve(y_true_binary, y_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            
            # 绘制ROC曲线
            plt.plot(
                fpr[i],
                tpr[i],
                label=f'{self.id_to_label[i]} (AUC = {roc_auc[i]:.3f})'
            )
        
        # 绘制对角线
        plt.plot([0, 1], [0, 1], 'k--')
        
        # 设置图表属性
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假正例率 (False Positive Rate)')
        plt.ylabel('真正例率 (True Positive Rate)')
        plt.title('各类别的ROC曲线')
        plt.legend(loc="lower right")
        plt.grid(True)
        
        # 保存图像
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"ROC曲线已保存到: {output_file}")
        
        plt.close()
    
    def analyze_feature_importance(
        self,
        texts: List[str],
        top_k: int = 10,
        output_file: Optional[str] = None
    ):
        """分析特征重要性（仅适用于可解释的模型）
        
        Args:
            texts: 文本列表
            top_k: 每个类别展示的重要特征数
            output_file: 输出文件路径
        """
        # 获取模型的特征重要性（需要模型支持）
        if not hasattr(self.model, 'get_feature_importance'):
            logger.warning("当前模型不支持特征重要性分析")
            return
        
        # 获取预测结果
        predictions, _ = self.predict(texts)
        
        # 分析每个类别的特征重要性
        report = []
        report.append("特征重要性分析报告")
        report.append("=" * 50)
        
        for class_id in range(self.num_classes):
            # 获取该类别的样本
            class_texts = [text for text, pred in zip(texts, predictions) if pred == class_id]
            if not class_texts:
                continue
            
            # 获取特征重要性
            importance = self.model.get_feature_importance(class_texts)
            
            # 生成报告
            report.append(f"\n类别: {self.id_to_label[class_id]}")
            report.append("-" * 30)
            report.append("重要特征:")
            
            # 输出top-k重要特征
            for feature, score in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_k]:
                report.append(f"  - {feature}: {score:.4f}")
        
        # 保存报告
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report))
            logger.info(f"特征重要性分析报告已保存到: {output_file}")
        
        return '\n'.join(report) 