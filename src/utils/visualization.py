import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from typing import List, Dict, Any
import pandas as pd

def plot_training_history(history: Dict[str, List[float]], save_path: str = None):
    """
    绘制训练历史
    :param history: 训练历史数据
    :param save_path: 保存路径
    """
    plt.figure(figsize=(12, 4))
    
    # 绘制损失
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='训练损失')
    plt.plot(history['val_loss'], label='验证损失')
    plt.title('模型损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # 绘制准确率
    plt.subplot(1, 2, 2)
    plt.plot(history['accuracy'], label='训练准确率')
    plt.plot(history['val_accuracy'], label='验证准确率')
    plt.title('模型准确率')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                         labels: List[str], save_path: str = None):
    """
    绘制混淆矩阵
    :param y_true: 真实标签
    :param y_pred: 预测标签
    :param labels: 标签名称列表
    :param save_path: 保存路径
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title('混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_classification_metrics(metrics: Dict[str, float], save_path: str = None):
    """
    绘制分类指标
    :param metrics: 分类指标字典
    :param save_path: 保存路径
    """
    plt.figure(figsize=(10, 6))
    metrics_df = pd.DataFrame(metrics.items(), columns=['Metric', 'Value'])
    sns.barplot(x='Metric', y='Value', data=metrics_df)
    plt.title('分类指标')
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_attention_weights(attention_weights: np.ndarray, 
                         tokens: List[str], 
                         save_path: str = None):
    """
    绘制注意力权重
    :param attention_weights: 注意力权重矩阵
    :param tokens: 词元列表
    :param save_path: 保存路径
    """
    plt.figure(figsize=(12, 8))
    sns.heatmap(attention_weights, 
                xticklabels=tokens,
                yticklabels=tokens,
                cmap='YlOrRd')
    plt.title('注意力权重可视化')
    plt.xlabel('词元')
    plt.ylabel('词元')
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_model_comparison(models_metrics: Dict[str, Dict[str, float]], 
                         metric_name: str = 'accuracy',
                         save_path: str = None):
    """
    绘制模型比较图
    :param models_metrics: 各模型指标字典
    :param metric_name: 要比较的指标名称
    :param save_path: 保存路径
    """
    plt.figure(figsize=(10, 6))
    models = list(models_metrics.keys())
    metrics = [metrics[metric_name] for metrics in models_metrics.values()]
    
    plt.bar(models, metrics)
    plt.title(f'模型{metric_name}比较')
    plt.xlabel('模型')
    plt.ylabel(metric_name)
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()
