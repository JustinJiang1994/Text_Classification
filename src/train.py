import argparse
from config import TEXTCNN_CONFIG, LSTM_CONFIG, DATASET_CONFIG, MODEL_SAVE_DIR
from data.dataset import TextDataset
from models.textcnn import build_textcnn
from models.lstm import build_lstm
import os

MODEL_MAP = {
    'textcnn': (build_textcnn, TEXTCNN_CONFIG),
    'lstm': (build_lstm, LSTM_CONFIG),
}

def main():
    parser = argparse.ArgumentParser(description="中文文本分类训练脚本")
    parser.add_argument('--model', type=str, default='textcnn', choices=['textcnn', 'lstm'], help='选择模型类型')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    args = parser.parse_args()

    # 选择模型和配置
    build_fn, model_config = MODEL_MAP[args.model]
    print(f"使用模型: {args.model}")

    # 加载数据
    train_dataset = TextDataset(
        txt_path=str(DATASET_CONFIG['train_file']),
        vocab_path=str(DATASET_CONFIG['vocab_file']),
        max_length=model_config['max_sequence_length'],
        categories=DATASET_CONFIG['categories']
    )
    val_dataset = TextDataset(
        txt_path=str(DATASET_CONFIG['val_file']),
        vocab_path=str(DATASET_CONFIG['vocab_file']),
        max_length=model_config['max_sequence_length'],
        categories=DATASET_CONFIG['categories']
    )
    x_train, y_train = train_dataset.encode_samples()
    x_val, y_val = val_dataset.encode_samples()

    # 构建模型
    model = build_fn(
        vocab_size=DATASET_CONFIG['vocab_size'],
        seq_length=model_config['max_sequence_length'],
        num_classes=len(DATASET_CONFIG['categories']),
        embedding_dim=model_config['embedding_dim'],
        **{k: v for k, v in model_config.items() if k not in ['embedding_dim', 'max_sequence_length']}
    )
    model.summary()

    # 训练
    model.fit(
        x_train, y_train,
        batch_size=model_config['batch_size'],
        epochs=args.epochs,
        validation_data=(x_val, y_val),
        verbose=1
    )

    # 保存模型
    save_path = os.path.join(MODEL_SAVE_DIR, f"{args.model}_model.h5")
    model.save(save_path)
    print(f"模型已保存到: {save_path}")

if __name__ == '__main__':
    main() 