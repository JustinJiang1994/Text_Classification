# -*- coding: utf-8 -*-

"""
Created on 2020-07-19 00:20
@Author  : Justin Jiang
@Email   : jw_jiang@pku.edu.com

配置模型、路径、与训练相关参数
"""

import os
from pathlib import Path

# 项目根目录
ROOT_DIR = Path(__file__).parent.parent

# 数据目录
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_SAVE_DIR = DATA_DIR / "models"

# 创建必要的目录
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_SAVE_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# 数据集配置
DATASET_CONFIG = {
    "train_file": RAW_DATA_DIR / "cnews.train.txt",
    "val_file": RAW_DATA_DIR / "cnews.val.txt",
    "test_file": RAW_DATA_DIR / "cnews.test.txt",
    "vocab_file": RAW_DATA_DIR / "cnews.vocab.txt",
    "categories": ["体育", "财经", "房产", "家居", "教育", "科技", "时尚", "时政", "游戏", "娱乐"],
    "vocab_size": 5000,
    "max_sequence_length": 600,
}

# 模型通用配置
MODEL_CONFIG = {
    "embedding_dim": 128,
    "num_classes": len(DATASET_CONFIG["categories"]),
    "dropout_rate": 0.5,
    "learning_rate": 0.001,
    "batch_size": 64,
    "epochs": 10,
    "early_stopping_patience": 3,
}

# TextCNN 模型配置
TEXTCNN_CONFIG = {
    **MODEL_CONFIG,
    "num_filters": 128,
    "filter_sizes": [2, 3, 4, 5],
    "model_name": "textcnn",
}

# LSTM 模型配置
LSTM_CONFIG = {
    **MODEL_CONFIG,
    "lstm_units": 128,
    "num_layers": 2,
    "model_name": "lstm",
}

# BERT 模型配置
BERT_CONFIG = {
    **MODEL_CONFIG,
    "bert_model_name": "bert-base-chinese",
    "max_sequence_length": 512,
    "model_name": "bert",
}

# 训练配置
TRAIN_CONFIG = {
    "log_dir": ROOT_DIR / "logs",
    "checkpoint_dir": MODEL_SAVE_DIR / "checkpoints",
    "tensorboard_dir": ROOT_DIR / "logs" / "tensorboard",
    "save_best_only": True,
    "save_weights_only": True,
    "monitor": "val_accuracy",
    "mode": "max",
}

# 评估配置
EVAL_CONFIG = {
    "metrics": ["accuracy", "precision", "recall", "f1"],
    "confusion_matrix": True,
    "classification_report": True,
}

# 日志配置
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "standard",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.FileHandler",
            "level": "INFO",
            "formatter": "standard",
            "filename": ROOT_DIR / "logs" / "app.log",
            "mode": "a",
        },
    },
    "loggers": {
        "": {
            "handlers": ["console", "file"],
            "level": "INFO",
            "propagate": True
        }
    }
}

class Config(object):
    def __init__(self):
        self.config_dict = {
            "data_path": {
                "vocab_path": "../data/cnews.vocab.txt",
                "trainingSet_path": "../data/cnews.train.txt",
                "valSet_path": "../data/cnews.val.txt",
                "testingSet_path": "../data/cnews.test.txt"
            },
            "CNN_training_rule": {
                "embedding_dim": 64,
                "seq_length": 600,
                "num_classes": 10,

                "conv1_num_filters": 128,
                "conv1_kernel_size": 1,

                "conv2_num_filters": 64,
                "conv2_kernel_size": 1,

                "vocab_size": 5000,

                "hidden_dim": 128,

                "dropout_keep_prob": 0.5,
                "learning_rate": 1e-3,

                "batch_size": 64,
                "epochs": 5,

                "print_per_batch": 100,
                "save_per_batch": 1000
            },
            "LSTM": {
                "seq_length": 600,
                "num_classes": 10,
                "vocab_size": 5000,
                "batch_size": 64
            },
            "result": {
                "CNN_model_path": "../result/CNN_model.h5",
                "LSTM_model_path": "../result/LSTM_model.h5"
            }
        }

    def get(self, section, name):
        return self.config_dict[section][name]