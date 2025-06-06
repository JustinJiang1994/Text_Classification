import argparse
from config import TEXTCNN_CONFIG, LSTM_CONFIG, DATASET_CONFIG, MODEL_SAVE_DIR
from data.dataset import TextDataset
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import os

MODEL_CONFIG_MAP = {
    'textcnn': TEXTCNN_CONFIG,
    'lstm': LSTM_CONFIG,
}

def main():
    parser = argparse.ArgumentParser(description="中文文本分类评估脚本")
    parser.add_argument('--model', type=str, default='textcnn', choices=['textcnn', 'lstm'], help='选择模型类型')
    args = parser.parse_args()

    model_config = MODEL_CONFIG_MAP[args.model]
    print(f"评估模型: {args.model}")

    # 加载测试集
    test_dataset = TextDataset(
        txt_path=str(DATASET_CONFIG['test_file']),
        vocab_path=str(DATASET_CONFIG['vocab_file']),
        max_length=model_config['max_sequence_length'],
        categories=DATASET_CONFIG['categories']
    )
    x_test, y_test = test_dataset.encode_samples()

    # 加载模型
    model_path = os.path.join(MODEL_SAVE_DIR, f"{args.model}_model.h5")
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        return
    model = load_model(model_path)
    print(f"模型已加载: {model_path}")

    # 预测
    y_pred = model.predict(x_test)
    y_true = np.argmax(y_test, axis=1)
    y_pred_label = np.argmax(y_pred, axis=1)

    # 输出评估指标
    acc = accuracy_score(y_true, y_pred_label)
    print(f"准确率: {acc:.4f}")
    print("分类报告：")
    print(classification_report(y_true, y_pred_label, target_names=DATASET_CONFIG['categories']))

if __name__ == '__main__':
    main() 